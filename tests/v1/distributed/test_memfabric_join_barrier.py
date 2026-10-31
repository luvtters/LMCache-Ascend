# SPDX-License-Identifier: Apache-2.0
"""Real local HTTP metadata barrier, fake native JOIN callbacks; no SDK/NPU."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from test_memfabric_direct_hbm import load_bm
from test_memfabric_l2_rdma import MemoryObj


class JoinBarrierTests(unittest.TestCase):
    def setUp(self):
        self.module = load_bm()
        path = Path(__file__).parents[3] / "examples/memfabric/direct_coordination.py"
        spec = importlib.util.spec_from_file_location("join_barrier_http", path)
        coord = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(coord)
        self.directory = tempfile.TemporaryDirectory()
        self.server = coord.make_server("127.0.0.1", 0, self.directory.name, "fresh")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)
        self.coord = coord

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def session(self, rank, bits=True):
        session = self.module.BMSession.__new__(self.module.BMSession)
        session.thread_id = threading.get_ident()
        session.rank_id, session.world_size = rank, 2
        session._local_join_complete = threading.Event()
        session._join_barrier_complete = False
        session.peer_departed = threading.Event()
        session._join_nonce = "new-session-" + str(rank)
        session._join_contract = dict(world_size=2, store_url="tcp://fresh:28580",
            pool_gb=1, pool_memory_type="host", data_op_type="DEVICE_RDMA", enable_56bits_gva=bits)
        session.bm = SimpleNamespace(BmGroupEvent=SimpleNamespace(JOIN_EVENT=0, LEAVE_EVENT=1))
        return session

    def test_two_ranks_wait_for_each_own_join_callback_then_all_http_events(self):
        sessions = []
        release_second_callback = threading.Event()
        def run(rank):
            session = self.session(rank)
            sessions.append(session)
            if rank:
                release_second_callback.wait(2)
            session._on_group_event(rank, 0)
            session.wait_fixed_group_ready(self.url, "fresh", 2)
            return session._join_barrier_complete
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(run, 0)
            second = pool.submit(run, 1)
            deadline = time.monotonic() + 1
            while not (Path(self.directory.name) / "fresh/bm-joined-0.json").exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            self.assertFalse(any(s._join_barrier_complete for s in sessions))
            release_second_callback.set()
            self.assertTrue(first.result(2))
            self.assertTrue(second.result(2))

    def test_missing_rank_times_out_without_marking_payload_ready(self):
        session = self.session(0)
        session._on_group_event(0, 0)
        with self.assertRaisesRegex(TimeoutError, "JOIN metadata barrier timed out"):
            session.wait_fixed_group_ready(self.url, "fresh", 0.08)
        self.assertFalse(session._join_barrier_complete)

    def test_remote_join_callback_is_not_a_substitute_for_own_completion(self):
        session = self.session(0)
        session._on_group_event(1, 0)
        with self.assertRaisesRegex(TimeoutError, "local after-mmap JOIN callback"):
            session.wait_fixed_group_ready(self.url, "fresh", 0.02)
        self.assertFalse(list((Path(self.directory.name) / "fresh").glob("*.json")))

    def test_two_rank_transport_mismatch_fails_both_instead_of_becoming_ready(self):
        def run(rank):
            session = self.session(rank, bits=not bool(rank))
            session._on_group_event(rank, 0)
            with self.assertRaisesRegex(ValueError, "metadata contract mismatch"):
                session.wait_fixed_group_ready(self.url, "fresh", 2)
            return session._join_barrier_complete
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.submit(run, 0), pool.submit(run, 1)
            self.assertFalse(first.result(2))
            self.assertFalse(second.result(2))

    def test_cpu_io_without_barrier_is_rejected_before_registration(self):
        session = self.session(0)
        session.data_op_type, session.closed, session._completion_unknown = "DEVICE_RDMA", False, False
        with self.assertRaisesRegex(RuntimeError, "requires the fixed-group JOIN"):
            session.copy_cpu_object(MemoryObj(), 1 << 30, 18 << 20, store=True)


if __name__ == "__main__":
    unittest.main()
