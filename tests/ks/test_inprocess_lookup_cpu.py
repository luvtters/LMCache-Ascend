# SPDX-License-Identifier: Apache-2.0
"""Run with system Python, without activating the NPU package."""

from pathlib import Path
import importlib.util
import sys
import unittest


source = Path(__file__).resolve().parents[2] / "lmcache_ascend/integration/vllm/inprocess_lookup.py"
spec = importlib.util.spec_from_file_location("inprocess_lookup_cpu", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class StepLookupTests(unittest.TestCase):
    def setUp(self):
        self.client = module.StepLookupClient(2)

    def query(self):
        self.assertIsNone(self.client.lookup([1, 2, 3], "r", {"tenant": "a"}))
        query, = self.client.take_requests()
        self.assertEqual((query.tokens, query.configs), ([1, 2, 3], {"tenant": "a"}))
        return query.sequence

    def test_common_prefix_requires_both_ranks(self):
        seq = self.query()
        self.client.receive({seq: {1: 2}})
        self.assertIsNone(self.client.lookup([1, 2, 3], "r"))
        self.assertEqual(self.client.take_requests(), [])
        self.client.receive({seq: {1: 2, 0: 3}})
        self.assertEqual(self.client.lookup_cache("r"), 2)

    def test_reissued_request_ignores_late_reply(self):
        old = self.query()
        self.client.clear_lookup_status("r")
        seq = self.query()
        self.client.receive({old: {0: 3, 1: 3}, seq: {0: 3}})
        self.assertIsNone(self.client.lookup_cache("r"))
        self.client.receive({seq: {1: 0}})
        self.assertEqual(self.client.lookup_cache("r"), 0)

    def test_cancel_before_dispatch(self):
        self.client.lookup([1], "r")
        self.client.clear_lookup_status("r")
        self.assertEqual(self.client.lookup_cache("r"), -1)
        self.assertEqual(self.client.take_requests(), [])

    def test_invalid_rank_and_inconsistent_replay(self):
        seq = self.query()
        with self.assertRaises(ValueError):
            self.client.receive({seq: {2: 0}})
        self.client.receive({seq: {1: 2}})
        with self.assertRaises(ValueError):
            self.client.receive({seq: {1: 1}})


if __name__ == "__main__":
    unittest.main()
