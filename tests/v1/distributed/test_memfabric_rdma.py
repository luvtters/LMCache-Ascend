# SPDX-License-Identifier: Apache-2.0
"""CPU control contracts only; fake native SDK does not establish RDMA reachability."""
from types import SimpleNamespace
import gc
import json
import unittest
from unittest.mock import MagicMock, patch
import weakref

from test_memfabric_direct_hbm import load_bm
from test_memfabric_direct import DIRECT, Client, PoolIndex, Session, Transfer
from test_memfabric_direct_server import FakeSession, load_server


class FakeTensor(SimpleNamespace):
    pass


def tensor(pointer=1024, size=32):
    return FakeTensor(device=SimpleNamespace(type="npu", index=0),
        data_ptr=lambda: pointer, numel=lambda: size, element_size=lambda: 1,
        is_contiguous=lambda: True)


class RdmaContracts(unittest.TestCase):
    def make_session(self):
        module = load_bm()
        handle = MagicMock()
        handle.join.return_value = handle.leave.return_value = handle.wait.return_value = 0
        handle.register.return_value = handle.unregister.return_value = handle.copy_data.return_value = 0
        handle.peer_rank_ptr.return_value = 4096
        bm = SimpleNamespace(BmConfig=lambda: SimpleNamespace(set_nic=MagicMock()),
            initialize=MagicMock(return_value=0), create2=MagicMock(return_value=handle),
            uninitialize=MagicMock(), BmDataOpType=SimpleNamespace(SDMA=1, DEVICE_RDMA=2),
            BmCopyType=SimpleNamespace(L2G=3, G2L=4),
            BmMemType=SimpleNamespace(HOST=10, DEVICE=11, LOCAL_DEVICE=12))
        mf = SimpleNamespace(bm=bm, initialize=MagicMock(return_value=0), uninitialize=MagicMock())
        torch = SimpleNamespace(npu=SimpleNamespace(set_device=MagicMock()))
        with patch.dict("sys.modules", {"torch": torch, "torch_npu": SimpleNamespace(),
                "memfabric_hybrid": mf}), patch.object(
                    module.importlib.metadata, "version", return_value="1.2.1"):
            session = module.BMSession(device_id=0, rank_id=0, world_size=2,
                pool_gb=1, store_url="cpu-only", data_op_type="DEVICE_RDMA",
                device_nic="tcp://127.0.0.1:10005", enable_56bits_gva=True)
        return module, session, handle, bm

    def test_explicit_create_and_register_copy_close_order(self):
        module, session, handle, bm = self.make_session()
        scratch = tensor()
        try:
            bm.create2.assert_called_once_with(id=0, data_op_type=2,
                local_dram_size=1 << 30, max_dram_size=1 << 30,
                enable_56bits_gva=True, flags=0)
            bm.initialize.call_args.args[3].set_nic.assert_called_once_with("tcp://127.0.0.1:10005")
            handle.gva_to_va.assert_not_called()
            with self.assertRaisesRegex(ValueError, "registered tensor"):
                session.copy_npu_to_global(1024, 4096, 32)
            session.register_npu_tensor(scratch)
            session.register_npu_tensor(scratch)
            handle.register.assert_called_once_with(1024, 32)
            self.assertIs(session._registered_tensors[1024][0], scratch)
            session.copy_npu_to_global(1024, 4096, 32)
            session.copy_global_to_npu(4096, 1024, 32)
            self.assertEqual(handle.copy_data.call_args.args, (4096, 1024, 32, 4, 0))
            session.close()
            names = [c[0] for c in handle.mock_calls]
            self.assertLess(names.index("unregister"), names.index("leave"))
            self.assertLess(names.index("leave"), names.index("destroy"))
            self.assertEqual(session._registered_tensors, {})
        finally:
            if not session.closed:
                session.close()

    def test_failed_copy_retains_tensor_and_rejects_noop_wait_cleanup(self):
        module, session, handle, bm = self.make_session()
        scratch = tensor()
        session.register_npu_tensor(scratch)
        handle.copy_data.return_value = 9
        try:
            with self.assertRaisesRegex(RuntimeError, "BM copy failed"):
                session.copy_npu_to_global(1024, 4096, 32)
            with self.assertRaisesRegex(RuntimeError, "completion unknown"):
                session.close()
            self.assertIs(session._registered_tensors[1024][0], scratch)
            self.assertIn(session, module.BMSession._uncertain_sessions)
            handle.unregister.assert_not_called()
            handle.wait.assert_not_called()
            handle.destroy.assert_not_called()
        finally:
            # Fake SDK only: manually reset the injected failure for unit teardown.
            session._completion_unknown = False
            session.close()
            module.BMSession._uncertain_sessions.remove(session)

    def test_invalid_options_fail_before_sdk_import(self):
        module = load_bm()
        for supplied in ({"data_op_type": "rdma"}, {"data_op_type": "DEVICE_RDMA"},
                {"device_nic": "invalid"}, {"device_nic": "tcp://host:70000"},
                {"device_nic": 1}, {"enable_56bits_gva": "true"}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                module.BMSession(device_id=0, rank_id=0, world_size=2,
                    pool_gb=1, store_url="cpu-only", **supplied)

    def test_unregister_failure_retains_session_and_tensor_after_caller_drops_them(self):
        module, session, handle, bm = self.make_session()
        scratch = tensor()
        session.register_npu_tensor(scratch)
        session_ref, scratch_ref = weakref.ref(session), weakref.ref(scratch)
        handle.unregister.return_value = 8
        with self.assertRaisesRegex(RuntimeError, "unregister NPU tensor failed: 8"):
            session.close()
        self.assertFalse(session.closed)
        self.assertTrue(module.BMSession._active)
        handle.leave.assert_not_called()
        handle.destroy.assert_not_called()
        # Remove mock-held callback references as well as caller references.
        handle.reset_mock()
        del session, scratch
        gc.collect()
        self.assertIsNotNone(session_ref())
        self.assertIsNotNone(scratch_ref())
        self.assertIs(module.BMSession._uncertain_sessions[0], session_ref())
        self.assertIs(session_ref()._registered_tensors[1024][0], scratch_ref())

    def test_each_native_teardown_exception_retains_registered_allocations(self):
        for stage in ("wait", "unregister", "leave", "destroy", "bm_uninitialize", "mf_uninitialize"):
            with self.subTest(stage=stage):
                module, session, handle, bm = self.make_session()
                scratch = tensor()
                session.register_npu_tensor(scratch)
                operation = (bm.uninitialize if stage == "bm_uninitialize" else
                             session.mf.uninitialize if stage == "mf_uninitialize" else
                             getattr(handle, stage))
                operation.side_effect = KeyboardInterrupt("injected teardown failure")
                with self.assertRaisesRegex(KeyboardInterrupt, "injected teardown failure"):
                    session.close()
                self.assertIn(session, module.BMSession._uncertain_sessions)
                self.assertIs(session._registered_tensors[1024][0], scratch)
                self.assertFalse(session.closed)
                self.assertTrue(module.BMSession._active)

    def test_unvalidated_cpu_and_invalid_tensor_registration_fail_explicitly(self):
        module, session, handle, bm = self.make_session()
        try:
            with self.assertRaisesRegex(RuntimeError, "CPU MemoryObj"):
                session.copy(1, 2, 32, store=True)
            wrong = tensor()
            wrong.device.type = "cpu"
            with self.assertRaisesRegex(ValueError, "contiguous tensor"):
                session.register_npu_tensor(wrong)
            session.register_npu_tensor(tensor())
            with self.assertRaisesRegex(ValueError, "another retained tensor"):
                session.register_npu_tensor(tensor())
            with self.assertRaisesRegex(ValueError, "registered tensor"):
                session.copy_global_to_npu(4096, 1040, 32)
        finally:
            session.close()

    def test_worker_synchronizes_and_registers_owned_scratch(self):
        sessions = []
        client = Client()
        client.index = PoolIndex(pool_bytes=1 << 30, slot_size=32 << 20, storage_rank=0,
            world_size=2, data_op_type="DEVICE_RDMA", enable_56bits_gva=True)
        class RdmaSession(Session):
            def register_npu_tensor(self, value):
                self.record("register_tensor")
                self.retained = value
        def factory(**kwargs):
            sessions.append(RdmaSession(**kwargs))
            return sessions[-1]
        worker = DIRECT.MemFabricDirectTransferContext(instance_id=1, req_client=client,
            tokens_per_chunk=128, mq_timeout=2, session_factory=factory, transfer_factory=Transfer,
            extra_config={"lmcache.mp.direct": dict(device_id=0, rank_id=1, world_size=2,
                pool_gb=1, store_url="cpu-only", data_op_type="DEVICE_RDMA",
                device_nic="tcp://127.0.0.1:10005", enable_56bits_gva=True)})
        try:
            worker.register({}, "qwen3", 1, 1, 2)
            self.assertIs(sessions[0].retained, worker.transfer.buffer)
            self.assertEqual(worker.transfer.operations[0], "synchronize")
            self.assertEqual(sessions[0].operations[0], "register_tensor")
            self.assertEqual(worker.registration["protocol"], "DEVICE_RDMA")
            self.assertTrue(worker.registration["enable_56bits_gva"])
            sent = next(payload for method, payload in client.calls if method == "register")
            self.assertEqual(sent["transport_contract_version"], 1)
        finally:
            worker.close()

    def test_server_passes_rdma_options(self):
        received = []
        def factory(**kwargs):
            received.append(kwargs)
            return FakeSession()
        server = load_server()(None, dict(device_id=0, rank_id=0, world_size=3,
            pool_gb=1, slot_size_mb=32, store_url="cpu-only", data_op_type="DEVICE_RDMA",
            device_nic="tcp://127.0.0.1:10005", enable_56bits_gva=True), session_factory=factory)
        try:
            server._wake.set()
            server._owner_thread.join(0.05)
            self.assertEqual(received[0]["data_op_type"], "DEVICE_RDMA")
            self.assertTrue(received[0]["enable_56bits_gva"])
            self.assertEqual(server.report_status()["protocol"], "DEVICE_RDMA")
            self.assertTrue(server.report_status()["enable_56bits_gva"])
            wrong = json.loads(server.register(json.dumps(dict(rank=1, member_id="P")).encode()))
            self.assertFalse(wrong["ok"])
            self.assertIn("transport contracts differ", wrong["error"])
            matched = json.loads(server.register(json.dumps(dict(rank=1, member_id="P",
                data_op_type="DEVICE_RDMA", enable_56bits_gva=True,
                transport_contract_version=1)).encode()))
            self.assertTrue(matched["ok"])
        finally:
            server.close()

    def test_pool_rejects_transport_mismatch_before_membership_changes(self):
        for operation, bits in (("SDMA", False), ("DEVICE_RDMA", True)):
            with self.subTest(operation=operation):
                pool = PoolIndex(pool_bytes=128, slot_size=64, storage_rank=0,
                    world_size=2, data_op_type=operation, enable_56bits_gva=bits)
                for wrong in (
                        {"data_op_type": "DEVICE_RDMA" if operation == "SDMA" else "SDMA",
                         "enable_56bits_gva": bits},
                        {"data_op_type": operation, "enable_56bits_gva": not bits},
                        {"data_op_type": operation, "enable_56bits_gva": bits,
                         "transport_contract_version": 2}):
                    with self.assertRaisesRegex(ValueError, "transport contracts differ"):
                        pool.register(rank=1, member_id="P", **wrong)
                    self.assertFalse(pool.status()["registration_ready"])
                pool.register(rank=1, member_id="P", data_op_type=operation,
                    enable_56bits_gva=bits)
                self.assertTrue(pool.status()["registration_ready"])
                self.assertEqual(pool.status()["protocol"], operation)

    def test_worker_surfaces_control_mismatch_and_closes_native_session(self):
        for owner_op, owner_bits, worker_op, worker_bits in (
                ("SDMA", False, "DEVICE_RDMA", False),
                ("DEVICE_RDMA", True, "SDMA", False),
                ("DEVICE_RDMA", False, "DEVICE_RDMA", True)):
            with self.subTest(owner=owner_op, worker=worker_op, bits=worker_bits):
                client, sessions = Client(), []
                client.index = PoolIndex(pool_bytes=1 << 30, slot_size=32 << 20,
                    storage_rank=0, world_size=2, data_op_type=owner_op,
                    enable_56bits_gva=owner_bits)
                def factory(**kwargs):
                    session = Session(**kwargs)
                    session.register_npu_tensor = lambda value: session.record("register_tensor")
                    sessions.append(session)
                    return session
                worker = DIRECT.MemFabricDirectTransferContext(instance_id=1, req_client=client,
                    tokens_per_chunk=128, mq_timeout=2, session_factory=factory, transfer_factory=Transfer,
                    extra_config={"lmcache.mp.direct": dict(device_id=0, rank_id=1, world_size=2,
                        pool_gb=1, store_url="cpu-only", data_op_type=worker_op,
                        device_nic="tcp://127.0.0.1:10005", enable_56bits_gva=worker_bits)})
                try:
                    with self.assertRaisesRegex(RuntimeError, "transport contracts differ"):
                        worker.register({}, "qwen3", 1, 1, 2)
                    self.assertTrue(sessions[0].closed)
                    self.assertFalse(client.index.status()["registration_ready"])
                finally:
                    worker.close()


if __name__ == "__main__":
    unittest.main()
