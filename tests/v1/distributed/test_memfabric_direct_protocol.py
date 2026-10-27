# SPDX-License-Identifier: Apache-2.0
"""CPU checks using real LMCache module discovery, routing, and envelopes.

Run with the matching LMCache checkout on PYTHONPATH. Only Ascend package
activation and the BM session are replaced; this is in-process control-plane
validation, not RequestClient/ZMQ, NPU, or inference validation.
"""

from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import json
import sys
import threading
import time
import unittest

from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    build_server_module_router,
    load_server_module_classes,
)
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest


class CpuSession:
    """Replace only the device-owning BM boundary."""

    def __init__(self, **kwargs):
        self.owner = threading.get_ident()
        self.peer_departed = threading.Event()
        self.departed_rank = -1

    def close(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("BM close escaped its owner thread")


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[3] / "lmcache_ascend"
        packages = {}
        for suffix in ("", ".v1", ".v1.distributed"):
            name = "lmcache_ascend" + suffix
            package = ModuleType(name)
            package.__path__ = [str(directory.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(directory)]
            packages[name] = package
        self.packages = patch.dict(sys.modules, packages)
        self.packages.start()
        self.addCleanup(self.packages.stop)
        config = dict(device_id=0, rank_id=0, world_size=3, pool_gb=1,
                      slot_size_mb=32, store_url="tcp://127.0.0.1:28580")
        loaded = load_server_module_classes([ExtServerModuleSpec(
            "lmcache_ascend.v1.distributed.memfabric_direct_server", config)])
        self.assertEqual(len(loaded), 1)
        server_type, plugin_config = loaded[0]
        self.server = server_type(None, plugin_config, session_factory=CpuSession)
        self.addCleanup(self.server.close)
        deadline = time.monotonic() + 2
        while not self.server.report_status()["transfer_ready"]:
            if time.monotonic() >= deadline:
                self.fail("CPU session did not become ready")
            time.sleep(0.005)
        self.router = build_server_module_router(None, [self.server])
        self.assertIsNotNone(self.router)

    def call(self, method, **payload):
        response = self.router.server_module_call(ServerModuleCallRequest(
            "memfabric_direct." + method, json.dumps(payload).encode("utf-8")))
        self.assertTrue(response.success, response.error)
        return json.loads(response.payload)

    def test_discovered_module_reserve_commit_lookup_and_release(self):
        owner = self.call("register", rank=1, member_id="P")
        identity = dict(epoch=owner["epoch"], rank=1, member_id="P")
        item = self.call("prepare_store", **identity, key="chunk", size=32,
                         layout={"shape": [2, 1, 2, 4], "dtype": "bf16"})
        self.assertTrue(item["ok"])
        self.assertEqual(self.call("lookup", keys=["chunk"])["descriptors"], [])
        self.call("commit_store", **identity, write_id=item["write_id"])
        lookup = self.call("lookup", keys=["chunk"], request_id="query")
        self.assertEqual(len(lookup["descriptors"]), 1)
        lease = self.call("acquire", **identity, descriptor=lookup["descriptors"][0])
        self.assertEqual(self.server.report_status()["total_pins"], 2)
        self.call("release", **identity, lease_id=lease["lease_id"], dma_complete=True)
        self.call("release", epoch=owner["epoch"], request_id="query", dma_complete=True)
        self.assertEqual(self.server.report_status()["total_pins"], 0)

    def test_metadata_only_handlers_and_unknown_method_errors(self):
        result = self.call("prepare_store", payload=[1, 2, 3])
        self.assertFalse(result["ok"])
        response = self.router.server_module_call(ServerModuleCallRequest(
            "memfabric_direct.unknown", b"{}"))
        self.assertFalse(response.success)


if __name__ == "__main__":
    unittest.main()
