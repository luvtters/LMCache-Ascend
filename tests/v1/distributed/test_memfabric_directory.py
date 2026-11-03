# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the shared BM slot directory."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
from threading import Barrier
import unittest


# Load this pure metadata module without loading the Ascend plugin and native SDK.
_path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_directory.py"
_spec = importlib.util.spec_from_file_location("directory_under_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = _module.PoolDirectory(2, 64)
        self.layout = [[[4, 8]], ["torch.uint8"], "binary"]

    def op(self, rank, name, **args):
        return self.directory.apply(rank, name, args)

    def store(self, key, rank=0, size=32, salt=""):
        slot = self.op(rank, "reserve", key=key, size=size, layout=self.layout, cache_salt=salt)
        self.op(rank, "commit", key=key, slot=slot)
        return slot

    def test_competing_writers_cannot_publish_or_recycle_an_inflight_slot(self):
        self.directory = _module.PoolDirectory(1, 64)
        key = '{"hash":10}'
        start = Barrier(2)

        def race(rank):
            start.wait(timeout=5)
            try:
                return self.op(rank, "reserve", key=key, size=32, layout=self.layout)
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            slots = list(pool.map(race, range(2)))
        self.assertCountEqual(slots, [0, None])
        writer = slots.index(0)
        reader = 1 - writer
        self.assertIsNone(self.op(reader, "pin", key=key))
        with self.assertRaises(ValueError):
            self.op(reader, "commit", key=key, slot=0)
        with self.assertRaises(ValueError):
            self.op(writer, "commit", key=key, slot=1)
        with self.assertRaises(ValueError):
            self.op(reader, "delete", key=key)
        with self.assertRaises(ValueError):
            self.op(reader, "reserve", key="other", size=32, layout=self.layout)
        self.assertEqual(self.op(reader, "status"), {
            "objects": 0, "pending_writes": 1, "evicted_objects": 0, "free_slots": 0,
            "total_pins": 0, "locked_objects": 0, "total_bytes_used": 0,
            "bytes_by_cache_salt": {},
        })
        self.op(writer, "commit", key=key, slot=0)
        with self.assertRaises(ValueError):
            self.op(writer, "commit", key=key, slot=0)
        self.assertIsNone(self.op(reader, "reserve", key=key, size=32, layout=self.layout))
        with self.assertRaises(ValueError):
            self.op(reader, "reserve", key=key, size=16, layout=self.layout)
        with self.assertRaises(ValueError):
            self.op(reader, "reserve", key=key, size=32, layout=[[], [], "different"])
        self.assertIsNone(self.op(reader, "pin", key=key, hint=[[[1]], ["torch.uint8"]]))
        self.assertEqual(self.op(reader, "pin", key=key, hint=self.layout[:2]),
                         {"slot": 0, "size": 32, "layout": self.layout})

    def test_each_reader_keeps_its_pins_and_slot_until_last_unlock(self):
        self.directory = _module.PoolDirectory(1, 64)
        key = '{"hash":20}'
        slot = self.store(key, salt="tenant")
        for rank in (0, 1, 1):
            self.assertEqual(self.op(rank, "pin", key=key)["slot"], slot)
        self.op(0, "delete", key=key)
        self.assertIsNone(self.op(1, "pin", key=key))
        self.op(0, "unpin", key=key)
        self.op(0, "unpin", key=key)
        self.assertIsNone(self.op(0, "load", key=key))
        self.op(1, "unpin", key=key)
        status = self.op(1, "status")
        self.assertEqual((status["locked_objects"], status["total_pins"]), (1, 1))
        self.assertEqual(status["bytes_by_cache_salt"], {"tenant": 32})
        self.assertEqual(self.op(1, "load", key=key)["slot"], slot)
        for new_key in (key, "other"):
            with self.assertRaises(ValueError):
                self.op(0, "reserve", key=new_key, size=32, layout=self.layout)
        self.op(1, "unpin", key=key)
        self.assertIsNone(self.op(1, "load", key=key))
        status = self.op(0, "status")
        self.assertEqual((status["objects"], status["free_slots"], status["evicted_objects"]),
                         (0, 1, 0))
        self.assertEqual(status["bytes_by_cache_salt"], {})
        self.assertEqual(self.store(key, rank=1), slot)

    def test_unpinned_lru_and_usage_follow_both_writers(self):
        a, b = self.store("a", size=32), self.store("b", rank=1, size=16, salt="B")
        self.assertEqual(self.op(0, "status")["total_bytes_used"], 48)
        self.op(1, "pin", key="a")
        self.op(1, "unpin", key="a")
        self.assertEqual(self.store("c", size=8), b)
        self.assertIsNone(self.op(1, "pin", key="b"))
        self.op(0, "pin", key="a")
        self.op(1, "pin", key="c")
        self.op(1, "unpin", key="c")
        self.assertEqual(self.store("d", rank=1, size=24), b)
        self.assertEqual(self.op(0, "load", key="a")["slot"], a)
        self.op(0, "delete", key="d")
        self.assertEqual(self.op(0, "status"), {
            "objects": 1, "pending_writes": 0, "evicted_objects": 2, "free_slots": 1,
            "total_pins": 1, "locked_objects": 1, "total_bytes_used": 32,
            "bytes_by_cache_salt": {"": 32},
        })
        # Evictions of b and c and the explicit deletion of d go to their
        # respective publishers, regardless of the rank replacing the slot.
        self.assertEqual(self.directory.take_retired(0), [["c", 8]])
        self.assertEqual(self.directory.take_retired(1), [["b", 16], ["d", 24]])
        self.assertEqual(self.directory.take_retired(1), [])

    def test_only_last_unpin_retires_and_republication_keeps_generations(self):
        self.store("a")
        self.op(1, "pin", key="a")
        self.op(1, "delete", key="a")
        self.assertEqual(self.directory.take_retired(0), [])
        self.op(1, "unpin", key="a")
        self.assertEqual(self.directory.take_retired(1), [])
        slot = self.op(0, "reserve", key="a", size=16, layout=self.layout)
        self.assertEqual(self.directory.take_retired(0), [["a", 32]])
        self.op(0, "commit", key="a", slot=slot)
        self.op(1, "delete", key="a")
        self.op(1, "delete", key="a")
        self.assertEqual(self.directory.take_retired(1), [])
        self.assertEqual(self.directory.take_retired(0), [["a", 16]])

    def test_retirements_are_bounded_and_previous_generation_beats_backlog(self):
        keys = [json.dumps({"model": "m" * 3900, "i": i}) for i in range(12)]
        self.directory = _module.PoolDirectory(len(keys), 64)
        for key in keys:
            self.store(key)
        for key in keys:
            self.op(1, "delete", key=key)
        with self.assertRaises(ValueError):
            self.op(0, "reserve", key="x" * 4096, size=32, layout=self.layout)
        slot = self.op(0, "reserve", key=keys[-1], size=16, layout=self.layout)
        first = self.directory.take_retired(0)
        self.assertEqual(first[0], [keys[-1], 32])
        self.assertLessEqual(len(json.dumps(first)), 32 * 1024)
        self.assertLess(len(first), len(keys))
        self.op(0, "commit", key=keys[-1], slot=slot)
        second = self.directory.take_retired(0)
        self.assertCountEqual([key for key, _ in first + second], keys)
        self.assertTrue(all(key != keys[-1] for key, _ in second))
        self.assertEqual(self.directory.take_retired(0), [])

    def test_reject_invalid_requests_before_slot_assignment(self):
        for args in ({"key": "a", "size": 65, "layout": self.layout},
                     {"key": "a", "size": 1},
                     {"key": "a", "size": 1, "layout": [[[4]], [object()], "binary"]},
                     {"key": "a", "size": 1, "layout": self.layout, "pointer": 123}):
            with self.assertRaises(ValueError):
                self.directory.apply(0, "reserve", args)
        with self.assertRaises(ValueError):
            self.op(0, "abort", key="a")
        self.assertEqual(self.op(0, "status")["free_slots"], 2)


if __name__ == "__main__":
    unittest.main()
