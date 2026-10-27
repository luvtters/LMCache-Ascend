# SPDX-License-Identifier: Apache-2.0
"""CPU-only control plane checks, runnable directly with unittest."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch


def load_server():
    directory = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed"
    index_name = "lmcache_ascend.v1.distributed.memfabric_pool_index"
    index_spec = importlib.util.spec_from_file_location(index_name, directory / "memfabric_pool_index.py")
    index = importlib.util.module_from_spec(index_spec)
    engine = types.ModuleType("lmcache.v1.multiprocess.engine_module")
    engine.DiscoverableModule = type("DiscoverableModule", (), {})
    extension = types.ModuleType("lmcache.v1.multiprocess.ext_server_module")
    extension.server_module_handler = lambda name: lambda func: func
    with patch.dict(sys.modules, {index_name: index, engine.__name__: engine,
                                  extension.__name__: extension}):
        index_spec.loader.exec_module(index)
        spec = importlib.util.spec_from_file_location("memfabric_direct_server", directory / "memfabric_direct_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.MemFabricDirectServerModule


class FakeSession:
    def __init__(self):
        self.owner = threading.get_ident()
        self.peer_departed = threading.Event()
        self.departed_rank = 2
        self.closed = False

    def close(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("wrong BM owner thread")
        self.closed = True


class ServerTests(unittest.TestCase):
    config = dict(device_id=8, rank_id=0, world_size=3, pool_gb=1,
                  slot_size_mb=32, store_url="tcp://127.0.0.1:28580")

    def call(self, server, method, **data):
        return json.loads(getattr(server, method)(json.dumps(data).encode()))

    def wait_ready(self, server):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if server.report_status()["transfer_ready"]:
                return
            time.sleep(0.005)
        self.fail("service never ready")

    def test_background_join_does_not_block_registration_and_owner_closes(self):
        started, joined = threading.Event(), threading.Event()
        sessions = []

        def factory(**kwargs):
            started.set()
            joined.wait(2)
            session = FakeSession()
            sessions.append(session)
            return session

        server = load_server()(None, self.config, session_factory=factory)
        self.assertTrue(started.wait(1))
        response = self.call(server, "register", rank=1, member_id="P")
        self.assertTrue(response["ok"])
        self.assertFalse(response["transfer_ready"])
        self.assertFalse(response["registration_ready"])
        self.assertFalse(self.call(server, "lookup", keys=["a"])["ok"])
        joined.set()
        self.wait_ready(server)
        server.close()
        self.assertTrue(sessions[0].closed)

    def test_startup_error_and_peer_departure_are_reported_unavailable(self):
        def fail(**kwargs):
            raise RuntimeError("join failed")

        server = load_server()(None, self.config, session_factory=fail)
        server._owner_thread.join(2)
        self.assertIn("join failed", server.report_status()["last_error"])
        self.assertFalse(self.call(server, "register", rank=1, member_id="P")["ok"])
        server.close()
        session_list = []

        def good(**kwargs):
            session = FakeSession()
            session_list.append(session)
            return session

        server = load_server()(None, self.config, session_factory=good)
        self.wait_ready(server)
        session_list[0].peer_departed.set()
        deadline = time.monotonic() + 2
        while server.report_status()["transfer_ready"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(self.call(server, "lookup", keys=[])["ok"])
        self.assertIn("rank 2", server.report_status()["last_error"])
        server.close()

    def test_json_control_never_accepts_payload_and_close_waits_for_write_drain(self):
        server = load_server()(None, self.config, session_factory=lambda **kw: FakeSession())
        self.wait_ready(server)
        owner = self.call(server, "register", rank=1, member_id="P")
        identity = dict(epoch=owner["epoch"], rank=1, member_id="P")
        bad = self.call(server, "prepare_store", **identity, key="a", size=32, payload=[1, 2])
        self.assertFalse(bad["ok"])
        item = self.call(server, "prepare_store", **identity, key="a", size=32)
        self.assertTrue(item["ok"])
        closing = threading.Thread(target=server.close)
        closing.start()
        deadline = time.monotonic() + 1
        while not server.report_status()["closed"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(closing.is_alive())
        self.call(server, "abort_store", **identity, write_id=item["write_id"], dma_complete=True)
        closing.join(2)
        self.assertFalse(closing.is_alive())
        self.assertFalse(self.call(server, "lookup", keys=[])["ok"])

    def test_config_requires_explicit_physical_device(self):
        config = dict(self.config)
        del config["device_id"]
        with self.assertRaises(ValueError):
            load_server()(None, config, session_factory=lambda **kw: FakeSession())


if __name__ == "__main__":
    unittest.main()
