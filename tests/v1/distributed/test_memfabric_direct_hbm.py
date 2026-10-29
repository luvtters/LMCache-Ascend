# SPDX-License-Identifier: Apache-2.0
"""CPU media contracts. Fake SDK boundaries do not establish HBM reachability."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from test_memfabric_direct import DIRECT, Client, PoolIndex, Session, Transfer, key
from test_memfabric_direct_server import FakeSession, load_server


def load_bm():
    path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_bm.py"
    spec = importlib.util.spec_from_file_location("bm_media_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HbmContracts(unittest.TestCase):
    def test_role_threads_explicit_device_medium_and_gate_expectation(self):
        path = Path(__file__).parents[3] / "examples/memfabric/cross_host_role.py"
        spec = importlib.util.spec_from_file_location("role_media_cpu", path)
        role = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(role)
        for medium in ("host", "device"):
            args = SimpleNamespace(root="/private", python="/venv/python", topology="T3",
                stage="inference", tp_size=2, pool_gb=2, session_id="cpu-media", skip_blocks=0,
                model="/readonly/model", p_devices=[2, 3], d_devices=[10, 11], pool_device=8,
                pool_memory_type=medium)
            for name in ("service", "p", "d", "gate"):
                args.role = name
                plan = role.build_plan(args)
                self.assertEqual(plan["pool_memory_type"], medium)
                if name == "gate":
                    self.assertEqual(plan["argv"][plan["argv"].index("--expected-pool-memory-type") + 1], medium)
                else:
                    config = (plan["config"]["config"] if name == "service" else
                              plan["config"]["kv_connector_extra_config"]["lmcache.mp.direct"])
                    self.assertEqual(config.get("pool_memory_type", "host"), medium)
                    self.assertEqual(config["world_size"], 5)
                    self.assertEqual(config["rank_id"], {"service": 0, "p": 1, "d": 3}[name])

    def test_native_create_geometry_and_peer_memory_type_for_all_ranks(self):
        for supplied, medium in (({}, "host"), ({"pool_memory_type": "host"}, "host"),
                                 ({"pool_memory_type": "device"}, "device")):
            module = load_bm()
            for rank in range(3):
                with self.subTest(supplied=supplied, rank=rank):
                    handle = MagicMock()
                    handle.join.return_value = handle.wait.return_value = handle.leave.return_value = 0
                    handle.peer_rank_ptr.return_value = 4096
                    handle.gva_to_va.return_value = 8192
                    bm = SimpleNamespace(BmConfig=SimpleNamespace,
                        initialize=MagicMock(return_value=0), create2=MagicMock(return_value=handle),
                        uninitialize=MagicMock(), BmDataOpType=SimpleNamespace(SDMA=1),
                        BmMemType=SimpleNamespace(HOST=10, DEVICE=11, LOCAL_DEVICE=12))
                    mf = SimpleNamespace(bm=bm, initialize=MagicMock(return_value=0), uninitialize=MagicMock())
                    torch = SimpleNamespace(npu=SimpleNamespace(set_device=MagicMock()))
                    with patch.dict("sys.modules", {"torch": torch, "torch_npu": SimpleNamespace(),
                                                    "memfabric_hybrid": mf}), patch.object(
                            module.importlib.metadata, "version", return_value="1.2.1"):
                        session = module.BMSession(device_id=0, rank_id=rank, world_size=3,
                            pool_gb=2, store_url="cpu-only", **supplied)
                        expected = dict(id=0, data_op_type=1,
                            local_dram_size=(2 << 30) if medium == "host" else 1 << 30,
                            max_dram_size=(2 << 30) if medium == "host" else 1 << 30)
                        if medium == "device":
                            expected.update(local_hbm_size=2 << 30, max_hbm_size=2 << 30)
                        bm.create2.assert_called_once_with(**expected)
                        session.peer_address(0)
                        self.assertEqual(handle.peer_rank_ptr.call_args.args, (0, 10 if medium == "host" else 11))
                        session.close()

    def test_invalid_medium_rejected_before_native(self):
        module = load_bm()
        for value in (None, "hbm", "HOST", 0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "pool_memory_type"):
                module.BMSession(device_id=0, rank_id=0, world_size=3,
                                 pool_gb=1, store_url="cpu-only", pool_memory_type=value)

    def test_pool_rejects_mixed_member_media_and_descriptors_report_device(self):
        pool = PoolIndex(pool_bytes=128, slot_size=64, storage_rank=0,
                         world_size=3, pool_memory_type="device")
        with self.assertRaisesRegex(ValueError, "memory types differ"):
            pool.register(rank=1, member_id="P")
        pool.register(rank=1, member_id="P", pool_memory_type="device")
        identity = dict(epoch=pool.epoch, rank=1, member_id="P")
        reservation = pool.prepare_store(**identity, key="chunk", size=32)
        self.assertEqual(reservation["descriptor"]["pool_memory_type"], "device")
        self.assertEqual(pool.status()["pool_memory_type"], "device")
        self.assertEqual(pool.status()["pool_selected_sdk_memory_type"], "DEVICE")
        self.assertEqual(pool.status()["bm_declared_local_dram_bytes"], 1 << 30)
        self.assertEqual(pool.status()["bm_declared_local_hbm_bytes"], 128)
        pool.abort_store(**identity, write_id=reservation["write_id"], dma_complete=True)

    def test_server_passes_device_medium_and_status_without_changing_default(self):
        for supplied, expected in (({}, "host"), ({"pool_memory_type": "device"}, "device")):
            received = []
            ready = threading.Event()

            def factory(**kwargs):
                received.append(kwargs)
                ready.set()
                return FakeSession()

            config = dict(device_id=0, rank_id=0, world_size=3, pool_gb=1,
                          slot_size_mb=32, store_url="cpu-only", **supplied)
            server = load_server()(None, config, session_factory=factory)
            self.assertTrue(ready.wait(2))
            try:
                self.assertEqual(server.report_status()["pool_memory_type"], expected)
                self.assertEqual(received[0].get("pool_memory_type", "host"), expected)
                if not supplied:
                    self.assertNotIn("pool_memory_type", received[0])
            finally:
                server.close()

    def test_worker_passes_device_medium_and_rejects_wrong_descriptor_media(self):
        client = Client()
        client.index = PoolIndex(pool_bytes=1 << 30, slot_size=32 << 20,
                                storage_rank=0, world_size=2, pool_memory_type="device")
        received = []

        def factory(**kwargs):
            received.append(kwargs)
            return Session(**kwargs)

        worker = DIRECT.MemFabricDirectTransferContext(instance_id=1, req_client=client,
            tokens_per_chunk=128, mq_timeout=2, session_factory=factory, transfer_factory=Transfer,
            extra_config={"lmcache.mp.direct": dict(device_id=0, rank_id=1, world_size=2,
                pool_gb=1, store_url="cpu-only", pool_memory_type="device")})
        try:
            worker.register({}, "qwen3", 1, 1, 2)
            self.assertEqual(received[0]["pool_memory_type"], "device")
            self.assertEqual(worker.registration["pool_memory_type"], "device")
            self.assertTrue(worker.submit_store("P", key(), {}, [[0]], None, 1).result())
            descriptor = client.index.lookup(keys=DIRECT.chunk_keys(key(), 128))["descriptors"][0]
            wrong = dict(descriptor, pool_memory_type="host")
            with self.assertRaisesRegex(ValueError, "descriptor"):
                worker._address(wrong, descriptor["key"])
            self.assertTrue(worker.submit_retrieve("D", key(), {}, [[0]], None, 1).result())
            self.assertEqual(client.index.status()["total_pins"], 0)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
