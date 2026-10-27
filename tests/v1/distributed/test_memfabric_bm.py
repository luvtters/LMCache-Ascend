# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch checks for the BM wrapper; no MemFabric SDK is initialized."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import unittest


class BMCopyTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_bm.py"
        spec = importlib.util.spec_from_file_location("memfabric_bm_cpu_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.session = module.BMSession.__new__(module.BMSession)
        self.session.thread_id = threading.get_ident()
        self.session.closed = False
        self.session.peer_departed = threading.Event()
        self.session.departed_rank = None
        self.session.bm = SimpleNamespace(BmCopyType=SimpleNamespace(
            H2G="host-to-global", G2H="global-to-host",
            L2G="local-device-to-global", G2L="global-to-local-device"))
        self.session.handle = SimpleNamespace(copy_data=MagicMock(return_value=0))

    def test_compatibility_and_direct_directions_remain_distinct_and_sync(self):
        session = self.session
        calls = [
            (lambda: session.copy(1, 2, 32, store=True), "host-to-global"),
            (lambda: session.copy(1, 2, 32, store=False), "global-to-host"),
            (lambda: session.copy_npu_to_global(1, 2, 32), "local-device-to-global"),
            (lambda: session.copy_global_to_npu(1, 2, 32), "global-to-local-device"),
        ]
        for copy, expected in calls:
            with self.subTest(direction=expected):
                session.handle.copy_data.reset_mock()
                copy()
                session.handle.copy_data.assert_called_once_with(1, 2, 32, expected, 0)

    def test_copy_keeps_owner_and_native_error_checks(self):
        self.session.thread_id = -1
        with self.assertRaisesRegex(RuntimeError, "owner thread"):
            self.session.copy_npu_to_global(1, 2, 32)
        self.session.handle.copy_data.assert_not_called()
        self.session.thread_id = threading.get_ident()
        self.session.handle.copy_data.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "BM copy failed: 7"):
            self.session.copy_global_to_npu(1, 2, 32)


if __name__ == "__main__":
    unittest.main()
