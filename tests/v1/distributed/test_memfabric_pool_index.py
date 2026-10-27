# SPDX-License-Identifier: Apache-2.0
"""Run directly with unittest; no torch, native extensions, or NPU required."""

import importlib.util
from pathlib import Path
import sys
import unittest


def load_index():
    path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_pool_index.py"
    spec = importlib.util.spec_from_file_location("memfabric_pool_index", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PoolIndex


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = load_index()(pool_bytes=128, slot_size=64, storage_rank=0,
                                 world_size=3)
        self.pool.register(rank=1, member_id="P")
        self.pool.register(rank=2, member_id="D")
        self.owner = dict(epoch=self.pool.epoch, rank=1, member_id="P")
        self.reader = dict(epoch=self.pool.epoch, rank=2, member_id="D")

    def prepare(self, key):
        return self.pool.prepare_store(**self.owner, key=key, size=32)

    def store(self, key):
        item = self.prepare(key)
        self.pool.commit_store(**self.owner, write_id=item["write_id"])
        return item["descriptor"]

    def test_write_hidden_until_commit_and_pending_slots_not_reused(self):
        first = self.prepare("a")
        self.prepare("b")
        self.assertEqual(self.pool.lookup(keys=["a"])["descriptors"], [])
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.prepare("c")
        self.pool.commit_store(**self.owner, write_id=first["write_id"])
        self.assertEqual(len(self.pool.lookup(keys=["a"])["descriptors"]), 1)

    def test_read_pin_prevents_eviction_and_old_generation_rejected(self):
        old = self.store("a")
        lease = self.pool.acquire(**self.reader, descriptor=old)
        self.store("b")
        self.store("c")
        self.assertEqual(self.pool.lookup(keys=["a"])["descriptors"], [old])
        self.pool.release(**self.reader, lease_id=lease["lease_id"], dma_complete=True)
        # Make a oldest and reuse its slot.
        self.pool.lookup(keys=["c"])
        self.store("d")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.pool.acquire(**self.reader, descriptor=old)

    def test_abort_without_dma_completion_quarantines_until_explicit_drain(self):
        item = self.prepare("a")
        self.pool.abort_store(**self.owner, write_id=item["write_id"], dma_complete=False)
        self.store("b")
        self.store("c")
        self.assertEqual(self.pool.status()["quarantined_slots"], 1)
        self.assertEqual(self.pool.lookup(keys=["a"])["descriptors"], [])
        with self.assertRaisesRegex(RuntimeError, "abort"):
            self.pool.commit_store(**self.owner, write_id=item["write_id"])
        self.pool.abort_store(**self.owner, write_id=item["write_id"], dma_complete=True)
        self.assertEqual(self.pool.status()["quarantined_slots"], 0)

    def test_incomplete_release_keeps_pin_and_blocks_pool_reuse(self):
        a, b = self.store("a"), self.store("b")
        l1 = self.pool.acquire(**self.reader, descriptor=a)
        self.pool.acquire(**self.reader, descriptor=b)
        self.pool.release(**self.reader, lease_id=l1["lease_id"], dma_complete=False)
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.prepare("c")
        self.pool.release(**self.reader, lease_id=l1["lease_id"], dma_complete=True)
        self.prepare("c")

    def test_lookup_prefix_pin_idempotence_and_release(self):
        self.store("a")
        self.store("c")
        result = self.pool.lookup(keys=["a", "b", "c"], request_id="scheduler:1")
        self.assertEqual(len(result["descriptors"]), 1)
        self.assertEqual(result, self.pool.lookup(keys=["a", "b", "c"],
                                                request_id="scheduler:1"))
        self.assertEqual(self.pool.status()["total_pins"], 1)
        with self.assertRaises(ValueError):
            self.pool.lookup(keys=["c"], request_id="scheduler:1")
        self.pool.release(epoch=self.pool.epoch, request_id="scheduler:1", dma_complete=True)
        self.assertEqual(self.pool.status()["total_pins"], 0)

    def test_fixed_member_replacement_poison_and_epoch_checks(self):
        self.store("a")
        with self.assertRaisesRegex(RuntimeError, "epoch"):
            self.pool.prepare_store(**dict(self.owner, epoch="old"), key="b", size=32)
        with self.assertRaisesRegex(RuntimeError, "member"):
            self.pool.register(rank=1, member_id="replacement")
        with self.assertRaisesRegex(RuntimeError, "fixed"):
            self.pool.lookup(keys=["a"])
        self.assertFalse(self.pool.status()["ready"])

    def test_reject_wrong_shard_and_immutable_layout(self):
        self.store("a")
        duplicate = self.prepare("a")
        self.assertTrue(duplicate["existing"])
        self.assertIsNone(duplicate["write_id"])
        with self.assertRaises(ValueError):
            self.pool.lookup(keys=["a"], shard=1)
        with self.assertRaises(ValueError):
            self.pool.prepare_store(**self.owner, key="a", size=33)

    def test_close_retains_active_protection_and_never_reopens(self):
        item = self.prepare("a")
        self.pool.close()
        self.assertEqual(self.pool.status()["pending_writes"], 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.pool.commit_store(**self.owner, write_id=item["write_id"])

    def test_partial_lookup_release_retains_other_slots(self):
        self.store("a")
        self.store("b")
        self.pool.lookup(keys=["a", "b"], request_id="scheduler:2")
        self.pool.release(epoch=self.pool.epoch, request_id="scheduler:2",
                          keys=["a"], dma_complete=True)
        self.assertEqual(self.pool.status()["total_pins"], 1)
        self.prepare("c")
        self.assertEqual(len(self.pool.lookup(keys=["b"])["descriptors"]), 1)
        self.pool.release(epoch=self.pool.epoch, request_id="scheduler:2", dma_complete=True)
        self.assertEqual(self.pool.status()["total_pins"], 0)

    def test_descriptor_contains_layout_and_returns_isolated_metadata(self):
        layout = dict(shape=[2, 1, 128, 64], dtype="torch.bfloat16", format="2LTD")
        item = self.pool.prepare_store(**self.owner, key="layout", size=32, layout=layout)
        self.assertEqual(item["descriptor"]["layout"], layout)
        layout["shape"][0] = 999
        item["descriptor"]["layout"]["shape"][1] = 999
        self.pool.commit_store(**self.owner, write_id=item["write_id"])
        desc = self.pool.lookup(keys=["layout"])["descriptors"][0]
        self.assertEqual(desc["layout"]["shape"], [2, 1, 128, 64])


if __name__ == "__main__":
    unittest.main()
