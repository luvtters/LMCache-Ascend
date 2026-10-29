"""Actual adapter + core CPU checks with a fake BM session, no NPU initialization."""
from collections import namedtuple
import select
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from lmcache_ascend.v1.distributed import memfabric_l2_adapter as module


Key = namedtuple("Key", "object_group_id")


class Memory:
    def __init__(self):
        self.raw_tensor = torch.zeros(64, dtype=torch.uint8)
        self.refs = 1

    @property
    def data_ptr(self): return self.raw_tensor.data_ptr()
    def get_size(self): return 64
    def get_shapes(self): return [torch.Size([64])]
    def get_dtypes(self): return [torch.uint8]
    def get_memory_format(self): return None
    def ref_count_up(self): self.refs += 1
    def ref_count_down(self): self.refs -= 1


class AdapterFailureChecks(unittest.TestCase):
    def adapter(self):
        session = SimpleNamespace(peer_departed=threading.Event(), departed_rank=1,
            local_pool_bytes=1 << 30, peer_address=lambda rank: 4096,
            wait=lambda: None, close=lambda: None)
        with patch.object(module, "BMSession", return_value=session):
            adapter = module.MemFabricL2Adapter({"world_size": 2, "slot_size_mb": 256})
        self.addCleanup(adapter.close)
        return adapter, session

    def test_peer_left_lookup_signals_and_raises_instead_of_empty_miss(self):
        adapter, session = self.adapter()
        session.peer_departed.set()
        task = adapter.submit_lookup_and_lock_task([Key(0)], {})
        self.assertTrue(select.select([adapter.get_lookup_and_lock_event_fd()], [], [], 2)[0])
        with self.assertRaisesRegex(RuntimeError, "peer rank 1 left"):
            adapter.query_lookup_and_lock_result(task)
        self.assertEqual(adapter._tasks["lookup"], {})
        self.assertFalse(adapter.report_status()["is_healthy"])

    def test_poisoned_load_signals_and_raises(self):
        adapter, _ = self.adapter()
        adapter._poisoned = True
        adapter._last_error = "previous BM copy failed"
        task = adapter.submit_load_task([Key(0)], [Memory()])
        self.assertTrue(select.select([adapter.get_load_event_fd()], [], [], 2)[0])
        with self.assertRaisesRegex(RuntimeError, "previous BM copy failed"):
            adapter.query_load_result(task)
        self.assertEqual(adapter._tasks["load"], {})

    def test_copy_error_propagates_and_quarantines_buffer_until_close(self):
        adapter, session = self.adapter()
        def copy(*args, **kwargs): raise RuntimeError("BM wait failed")
        session.copy = copy
        obj, key = Memory(), Key(0)
        adapter._entries[key] = module._Entry(0, 64, adapter._layout(obj), pins=1)
        task = adapter.submit_load_task([key], [obj])
        self.assertTrue(select.select([adapter.get_load_event_fd()], [], [], 2)[0])
        with self.assertRaisesRegex(RuntimeError, "BM wait failed"):
            adapter.query_load_result(task)
        self.assertEqual(adapter._tasks["load"], {})
        self.assertFalse(adapter.report_status()["is_healthy"])
        self.assertEqual(obj.refs, 2)
        adapter.submit_unlock([key])
        adapter._executor.submit(lambda: None).result(timeout=2)
        self.assertEqual(adapter.report_status()["total_pins"], 0)
        adapter.close()
        self.assertEqual(obj.refs, 1)
        self.assertEqual(adapter._quarantine, [])

    def test_capacity_failure_is_zero_bytes_and_remains_healthy(self):
        adapter, _ = self.adapter()
        obj = Memory()
        adapter._free = []
        for index in range(4):
            adapter._entries[Key(index)] = module._Entry(index, 64, adapter._layout(obj), pins=1)
        task = adapter.submit_store_task([Key(5)], [obj])
        self.assertTrue(select.select([adapter.get_store_event_fd()], [], [], 2)[0])
        result = adapter.pop_completed_store_tasks()[task]
        self.assertFalse(result.is_successful())
        self.assertEqual(result.bytes_transferred(), 0)
        self.assertTrue(adapter.report_status()["is_healthy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
