# SPDX-License-Identifier: Apache-2.0
"""CPU lease/configuration controls only; the native SDK is a fake boundary."""
import gc
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import weakref

import test_memfabric_rdma as rdma_cpu


class MemoryObj:
    def __init__(self, pointer=4096, size=18 << 20):
        self.data_ptr, self.size, self.refs = pointer, size, 1
        self.raw_tensor = rdma_cpu.tensor(pointer, size)
        self.raw_tensor.device.type = "cpu"
        self.released = False

    def get_size(self):
        return self.size

    def get_shapes(self):
        return [(self.size,)]

    def get_dtypes(self):
        return ["uint8"]

    def get_memory_format(self):
        return "binary"

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1
        self.released = self.refs == 0


class Bitmap:
    def __init__(self, size):
        self.bits = set()

    def set(self, index):
        self.bits.add(index)

    def popcount(self):
        return len(self.bits)


class Session:
    def __init__(self, **kwargs):
        self.owner = threading.get_ident()
        self.config = kwargs
        self.peer_departed = threading.Event()
        self.departed_rank = None
        self.failure = False
        self.operations = []

    def peer_address(self, rank):
        return 1 << 30

    def copy_cpu_object(self, obj, address, size, *, store):
        assert self.owner == threading.get_ident()
        self.operations.append("registered_cpu")
        if self.failure:
            raise RuntimeError("native CPU registration failed")

    def copy(self, *args, **kwargs):
        self.operations.append("legacy_sdma")

    def wait(self):
        pass

    def close(self):
        assert self.owner == threading.get_ident()


def load_adapter(factory):
    class Base:
        def __init__(self, **kwargs):
            pass
        def _notify_keys_stored(self, *args):
            pass
        def _notify_keys_deleted(self, *args):
            pass
    attrs = {
        "lmcache.lmcache_native": dict(Bitmap=Bitmap),
        "lmcache.logging": dict(init_logger=lambda name: MagicMock()),
        "lmcache.v1.distributed.api": dict(MemoryResidency=SimpleNamespace(LOCAL=1),
            MemorySegment=SimpleNamespace(DRAM=2), MemoryTags=lambda *a: a),
        "lmcache.v1.distributed.internal_api": dict(L2StoreResult=lambda success, size:
            SimpleNamespace(success=success, size=size)),
        "lmcache.v1.distributed.l2_adapters.base": dict(L2AdapterInterface=Base),
        "lmcache.v1.platform": dict(create_event_notifier=lambda: MagicMock()),
        "lmcache_ascend.v1.distributed.memfabric_bm": dict(BMSession=factory),
    }
    modules = {}
    for name, fields in attrs.items():
        modules[name] = ModuleType(name)
        modules[name].__dict__.update(fields)
    path = Path(__file__).parents[3] / "lmcache_ascend/v1/distributed/memfabric_l2_adapter.py"
    spec = importlib.util.spec_from_file_location("l2_rdma_cpu_adapter", path)
    module = importlib.util.module_from_spec(spec)
    modules[spec.name] = module
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
    return module.MemFabricL2Adapter


class CpuMemoryObjContracts(unittest.TestCase):
    def native_session(self):
        module, session, handle, bm = rdma_cpu.RdmaContracts.make_session(self)
        bm.BmCopyType.H2G, bm.BmCopyType.G2H = 5, 6
        session._join_barrier_complete = True  # Fake native I/O unit boundary, not readiness proof.
        return module, session, handle

    def test_exact_range_registration_copy_unregister_then_release_real_lease(self):
        module, session, handle = self.native_session()
        obj = MemoryObj()
        order = []
        def native(name):
            def operation(*args):
                self.assertEqual(obj.refs, 2)
                self.assertIs(session._registered_cpu_objects[4096][0], obj)
                order.append((name, args))
                return 0
            return operation
        handle.register.side_effect = native("register")
        handle.copy_data.side_effect = native("copy")
        handle.unregister.side_effect = native("unregister")
        try:
            session.copy_cpu_object(obj, 1 << 30, obj.size, store=True)
            session.copy_cpu_object(obj, 1 << 30, obj.size, store=False)
            self.assertEqual([name for name, _ in order], ["register", "copy", "unregister"] * 2)
            self.assertEqual(order[0][1], (4096, 18 << 20))
            self.assertEqual(order[1][1], (4096, 1 << 30, 18 << 20, 5, 0))
            self.assertEqual(order[4][1], (1 << 30, 4096, 18 << 20, 6, 0))
            self.assertEqual(obj.refs, 1)
            self.assertEqual(session._registered_cpu_objects, {})
        finally:
            session.close()

    def test_native_failure_keeps_memoryobj_lease_and_tensor_even_after_caller_release(self):
        for stage in ("register", "copy_data", "unregister"):
            with self.subTest(stage=stage):
                module, session, handle = self.native_session()
                obj = MemoryObj()
                obj_ref, raw_ref = weakref.ref(obj), weakref.ref(obj.raw_tensor)
                getattr(handle, stage).return_value = -1
                with self.assertRaisesRegex(RuntimeError, "failed: -1"):
                    session.copy_cpu_object(obj, 1 << 30, obj.size, store=True)
                self.assertEqual(obj.refs, 2)
                obj.ref_count_down()  # Simulate caller abandoning its own lease.
                del obj
                gc.collect()
                self.assertIsNotNone(obj_ref())
                self.assertIsNotNone(raw_ref())
                self.assertFalse(obj_ref().released)
                self.assertEqual(obj_ref().refs, 1)
                self.assertIn(session, module.BMSession._uncertain_sessions)
                with self.assertRaisesRegex(RuntimeError, "completion unknown"):
                    session.close()
                session_ref = weakref.ref(session)
                handle.reset_mock()
                del session
                gc.collect()
                self.assertIsNotNone(session_ref())
                self.assertIs(session_ref()._registered_cpu_objects[4096][0], obj_ref())

    def test_unaligned_range_rejected_without_registration_or_extra_padding(self):
        module, session, handle = self.native_session()
        obj = MemoryObj(pointer=4096 + 192, size=64)
        try:
            with self.assertRaisesRegex(ValueError, "modulo4096=192 bytes=64"):
                session.copy_cpu_object(obj, 1 << 30, 64, store=True)
            self.assertEqual(obj.refs, 1)
            handle.register.assert_not_called()
            handle.copy_data.assert_not_called()
        finally:
            session.close()

    def test_aligned_small_range_native_rejection_is_explicit_and_exact(self):
        module, session, handle = self.native_session()
        obj = MemoryObj(size=64)
        handle.register.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "bytes=64 modulo4096=0 failed: -1"):
            session.copy_cpu_object(obj, 1 << 30, 64, store=True)
        handle.register.assert_called_once_with(4096, 64)
        handle.copy_data.assert_not_called()
        self.assertEqual(obj.refs, 2)


class AdapterContracts(unittest.TestCase):
    def adapter(self, rdma=True):
        sessions = []
        def factory(**kwargs):
            sessions.append(Session(**kwargs))
            return sessions[-1]
        params = dict(device_id=0, world_size=2, pool_gb=1, storage_rank=1, slot_size_mb=32,
                      store_url="cpu-only")
        if rdma:
            params.update(data_op_type="DEVICE_RDMA", device_nic="tcp://127.0.0.1:10005",
                          enable_56bits_gva=True, join_coord_url="http://127.0.0.1:29595",
                          join_group_id="cpu-only-fresh")
        adapter = load_adapter(factory)(params)
        return adapter, sessions[0]

    def test_configuration_and_success_use_registered_entry_on_owner_thread(self):
        adapter, session = self.adapter()
        obj = MemoryObj()
        try:
            self.assertEqual(session.config["data_op_type"], "DEVICE_RDMA")
            self.assertTrue(session.config["enable_56bits_gva"])
            self.assertEqual(session.config["device_nic"], "tcp://127.0.0.1:10005")
            result = adapter._executor.submit(adapter._store, ["a"], [obj]).result()
            self.assertTrue(result.success)
            bits = adapter._executor.submit(adapter._load, ["a"], [MemoryObj()]).result()
            self.assertEqual(bits.popcount(), 1)
            self.assertEqual(session.operations, ["registered_cpu", "registered_cpu"])
            self.assertEqual(adapter.report_status()["protocol"], "DEVICE_RDMA")
        finally:
            adapter.close()

    def test_rdma_store_load_and_poisoned_lookup_raise_instead_of_becoming_misses(self):
        for first_store in (True, False):
            with self.subTest(first_store=first_store):
                adapter, session = self.adapter()
                try:
                    if not first_store:
                        adapter._executor.submit(adapter._store, ["a"], [MemoryObj()]).result()
                    session.failure = True
                    operation = adapter._store if first_store else adapter._load
                    with self.assertRaisesRegex(RuntimeError, "native CPU registration failed"):
                        adapter._executor.submit(operation, ["a"], [MemoryObj()]).result()
                    self.assertFalse(adapter.report_status()["is_healthy"])
                    with self.assertRaisesRegex(RuntimeError, "native CPU registration failed"):
                        adapter._executor.submit(adapter._lookup, ["a"], {}).result()
                finally:
                    adapter.close()

    def test_default_sdma_uses_unchanged_raw_copy_and_ref_count_path(self):
        adapter, session = self.adapter(rdma=False)
        obj = MemoryObj()
        try:
            self.assertNotIn("data_op_type", session.config)
            self.assertTrue(adapter._executor.submit(adapter._store, ["a"], [obj]).result().success)
            self.assertEqual(session.operations, ["legacy_sdma"])
            self.assertEqual(obj.refs, 1)
            self.assertEqual(adapter.report_status()["protocol"], "SDMA")
        finally:
            adapter.close()

    def test_rdma_close_failure_is_not_marked_closed_or_hidden_on_retry(self):
        adapter, session = self.adapter()
        original_close = session.close
        def fail():
            raise RuntimeError("native close completion unknown")
        session.close = fail
        try:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "native close completion unknown"):
                    adapter.close()
                self.assertFalse(adapter._closed)
                status = adapter.report_status()
                self.assertFalse(status["is_healthy"])
                self.assertEqual(status["last_error"], "native close completion unknown")
        finally:
            session.close = original_close
            adapter.close()

    def test_real_memoryobj_gate_defaults_to_plan_without_runtime_imports(self):
        path = Path(__file__).parents[3] / "examples/memfabric/l2_rdma_memoryobj_gate.py"
        spec = importlib.util.spec_from_file_location("l2_memoryobj_plan_cpu", path)
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            with patch.dict("sys.modules", {"torch": None}), contextlib.redirect_stdout(io.StringIO()):
                gate.main(["--store-url", "tcp://127.0.0.1:28580", "--device-nic",
                    "tcp://127.0.0.1:10005", "--enable-56bits-gva", "--join-coord-url",
                    "http://127.0.0.1:29595", "--join-group-id", "fresh-plan", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PLAN_ONLY")
            self.assertEqual(report["payload_bytes"], 18 << 20)
            self.assertFalse(report["fallback_allowed"])


if __name__ == "__main__":
    unittest.main()
