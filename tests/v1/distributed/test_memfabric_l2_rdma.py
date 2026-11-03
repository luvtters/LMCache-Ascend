# SPDX-License-Identifier: Apache-2.0
"""CPU lease/configuration controls only; the native SDK is a fake boundary."""
import gc
from concurrent.futures import wait as wait_futures
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
    def __init__(self, pointer=4096, size=18 << 20, parent_allocator=None):
        self.data_ptr, self.size, self.refs = pointer, size, 1
        self.raw_tensor = rdma_cpu.tensor(pointer, size)
        self.raw_tensor.device.type = "cpu"
        self.parent_allocator = parent_allocator
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
        self.close_failure = False
        self.operations = []
        self.arena_calls = []

    def peer_address(self, rank):
        return 1 << 30

    def set_cpu_registration_arena(self, pointer, size, chunk_size=0):
        assert self.owner == threading.get_ident()
        self.arena_calls.append((pointer, size, chunk_size))
        self.arena = (pointer, size)
        self.arena_chunk_size = chunk_size

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
        if self.close_failure:
            raise RuntimeError("fake session close failed")


class MixedMemoryAllocator:
    def __init__(self, pointer=4096, size=18 << 20, shm_name="lmcache_l1_pool_cpu"):
        self.buffer = rdma_cpu.tensor(pointer, size)
        self.buffer.device.type = "cpu"
        self.size = size
        self.shm_name = shm_name
        self._unregistered = False


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
        "lmcache": dict(__path__=[]),
        "lmcache.v1": dict(__path__=[]),
        "lmcache.v1.memory_allocators": dict(__path__=[]),
        "lmcache.v1.memory_allocators.mixed_memory_allocator": dict(
            MixedMemoryAllocator=MixedMemoryAllocator),
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
    module.MemFabricL2Adapter._cpu_import_modules = modules
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

    def test_arena_reuses_registration_and_unregisters_once_before_leave(self):
        _, session, handle = self.native_session()
        size = 18 << 20
        session.set_cpu_registration_arena(4096, size * 2)
        first, second = MemoryObj(), MemoryObj(pointer=4096 + size)
        for obj, store in ((first, True), (second, False), (first, False)):
            self.assertEqual(session.copy_cpu_object(obj, 1 << 30, size, store=store), "l1_arena")
            self.assertEqual(obj.refs, 1)
        handle.register.assert_called_once_with(4096, size * 2)
        handle.unregister.assert_not_called()
        session.close()
        handle.unregister.assert_called_once_with(4096)
        calls = [c[0] for c in handle.method_calls]
        self.assertLess(calls.index("unregister"), calls.index("leave"))

    def test_chunked_arena_registers_overlapped_regions_once_and_unregisters_each_once(self):
        _, session, handle = self.native_session()
        base, chunk, obj_size = 4096, 32 << 20, 18 << 20
        session.set_cpu_registration_arena(base, 64 << 20, chunk)
        first = MemoryObj(pointer=base, size=obj_size)
        second = MemoryObj(pointer=base + obj_size, size=obj_size)
        for obj, store in ((first, True), (second, False), (first, False)):
            self.assertEqual(session.copy_cpu_object(obj, 1 << 30, obj_size, store=store),
                             "l1_chunks")
            self.assertEqual(obj.refs, 1)
        regions = [(base, chunk), (base + chunk, chunk)]
        self.assertEqual([call.args for call in handle.register.call_args_list], regions)
        handle.unregister.assert_not_called()

        session.close()
        self.assertEqual([call.args[0] for call in handle.unregister.call_args_list],
                         [base, base + chunk])
        self.assertEqual(len(handle.unregister.call_args_list), 2)
        calls = [call[0] for call in handle.method_calls]
        self.assertLess(calls.index("unregister"), calls.index("leave"))

    def test_chunked_arena_splits_h2g_and_g2h_at_mr_boundary_with_contiguous_pool_offsets(self):
        _, session, handle = self.native_session()
        base, chunk, obj_size, pool = 4096, 32 << 20, 18 << 20, 1 << 30
        pointer = base + obj_size
        first_part, second_part = 14 << 20, 4 << 20
        session.set_cpu_registration_arena(base, 64 << 20, chunk)
        obj = MemoryObj(pointer=pointer, size=obj_size)

        self.assertEqual(session.copy_cpu_object(obj, pool, obj_size, store=True), "l1_chunks")
        self.assertEqual(session.copy_cpu_object(obj, pool, obj_size, store=False), "l1_chunks")
        self.assertEqual(obj.refs, 1)
        self.assertEqual([call.args for call in handle.copy_data.call_args_list], [
            (pointer, pool, first_part, 5, 0),
            (pointer + first_part, pool + first_part, second_part, 5, 0),
            (pool, pointer, first_part, 6, 0),
            (pool + first_part, pointer + first_part, second_part, 6, 0),
        ])
        self.assertEqual(first_part + second_part, obj_size)
        self.assertEqual(first_part, (base + chunk) - pointer)
        self.assertEqual([call.args for call in handle.register.call_args_list], [
            (base, chunk), (base + chunk, chunk)])
        session.close()

    def test_second_chunk_copy_failure_retains_object_and_all_regions_without_unregister(self):
        module, session, handle = self.native_session()
        base, chunk, obj_size, pool = 4096, 32 << 20, 18 << 20, 1 << 30
        pointer = base + obj_size
        session.set_cpu_registration_arena(base, 64 << 20, chunk)
        obj = MemoryObj(pointer=pointer, size=obj_size)
        handle.copy_data.side_effect = [0, -1]

        with self.assertRaisesRegex(RuntimeError, "failed: -1"):
            session.copy_cpu_object(obj, pool, obj_size, store=True)
        self.assertEqual([call.args for call in handle.copy_data.call_args_list], [
            (pointer, pool, 14 << 20, 5, 0),
            (pointer + (14 << 20), pool + (14 << 20), 4 << 20, 5, 0),
        ])
        self.assertEqual(obj.refs, 2)
        self.assertIs(session._registered_cpu_objects[pointer][0], obj)
        self.assertEqual(session._cpu_arena_ranges,
                         {base: chunk, base + chunk: chunk})
        self.assertTrue(session._completion_unknown)
        self.assertIn(session, module.BMSession._uncertain_sessions)
        with self.assertRaisesRegex(RuntimeError, "completion unknown"):
            session.close()
        handle.unregister.assert_not_called()
        handle.leave.assert_not_called()
        handle.destroy.assert_not_called()

    def test_chunked_arena_tail_registration_uses_only_stable_prefix_remainder(self):
        _, session, handle = self.native_session()
        base, chunk, stable_size = 4096, 32 << 20, 50 << 20
        tail_size = stable_size - chunk
        session.set_cpu_registration_arena(base, stable_size, chunk)
        obj = MemoryObj(pointer=base + chunk, size=tail_size)
        try:
            self.assertEqual(session.copy_cpu_object(obj, 1 << 30, tail_size, store=True),
                             "l1_chunks")
            self.assertEqual([call.args for call in handle.register.call_args_list],
                             [(base + chunk, tail_size)])
            session.close()
            self.assertEqual([call.args for call in handle.unregister.call_args_list],
                             [(base + chunk,)])
        finally:
            if not session.closed and not session._completion_unknown:
                session.close()

    def test_chunked_arena_second_region_register_failure_retains_lease_and_first_region(self):
        module, session, handle = self.native_session()
        base, chunk, obj_size = 4096, 32 << 20, 18 << 20
        session.set_cpu_registration_arena(base, 64 << 20, chunk)
        obj = MemoryObj(pointer=base + obj_size, size=obj_size)
        handle.register.side_effect = [0, -1]

        with self.assertRaisesRegex(RuntimeError, "BM register CPU L1 arena bytes=33554432 failed: -1"):
            session.copy_cpu_object(obj, 1 << 30, obj_size, store=True)
        self.assertEqual(obj.refs, 2)
        self.assertIs(session._registered_cpu_objects[obj.data_ptr][0], obj)
        self.assertEqual(session._cpu_arena_ranges, {base: chunk})
        self.assertTrue(session._completion_unknown)
        self.assertIn(session, module.BMSession._uncertain_sessions)

        with self.assertRaisesRegex(RuntimeError, "completion unknown"):
            session.close()
        handle.unregister.assert_not_called()
        handle.leave.assert_not_called()
        handle.destroy.assert_not_called()

    def test_arena_does_not_cover_outside_or_partial_objects(self):
        _, session, handle = self.native_session()
        session.set_cpu_registration_arena(4096, 18 << 20)
        outside = MemoryObj(pointer=4096 + (36 << 20))
        self.assertEqual(session.copy_cpu_object(outside, 1 << 30, outside.size, store=True), "per_object")
        handle.register.assert_called_once_with(outside.data_ptr, outside.size)
        handle.unregister.assert_called_once_with(outside.data_ptr)
        crossing = MemoryObj(pointer=8192)
        with self.assertRaisesRegex(ValueError, "straddles"):
            session.copy_cpu_object(crossing, 1 << 30, crossing.size, store=True)
        self.assertEqual(crossing.refs, 1)
        session.close()

    def test_arena_unknown_completion_keeps_lease_and_prevents_unregister(self):
        for stage in ("register", "copy_data"):
            with self.subTest(stage=stage):
                module, session, handle = self.native_session()
                session.set_cpu_registration_arena(4096, 18 << 20)
                obj = MemoryObj()
                getattr(handle, stage).return_value = -1
                with self.assertRaises(RuntimeError):
                    session.copy_cpu_object(obj, 1 << 30, obj.size, store=True)
                self.assertEqual(obj.refs, 2)
                with self.assertRaisesRegex(RuntimeError, "completion unknown"):
                    session.close()
                handle.unregister.assert_not_called()
                self.assertIn(session, module.BMSession._uncertain_sessions)

    def test_arena_unregister_failure_prevents_session_destruction(self):
        _, session, handle = self.native_session()
        session.set_cpu_registration_arena(4096, 18 << 20)
        obj = MemoryObj()
        session.copy_cpu_object(obj, 1 << 30, obj.size, store=True)
        handle.unregister.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "unregister CPU L1 arena"):
            session.close()
        handle.leave.assert_not_called()
        handle.destroy.assert_not_called()


class AdapterContracts(unittest.TestCase):
    def adapter(self, rdma=True, register_arena=False, desc=None):
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
        if register_arena:
            params["register_l1_arena"] = True
        adapter = load_adapter(factory)(params, desc)
        return adapter, sessions[0]

    @staticmethod
    def desc(pointer=4096, size=64 << 20):
        # Fixed official core descriptor: deliberately has no new field.
        return SimpleNamespace(ptr=pointer, size=size, align_bytes=4096)

    def store(self, adapter, key, obj):
        with patch.dict("sys.modules", adapter.__class__._cpu_import_modules):
            return adapter._executor.submit(adapter._store, [key], [obj]).result()

    def test_fixed_shm_first_copy_configures_once_and_reuses_owner(self):
        base, size = 4096, 64 << 20
        owner = MixedMemoryAllocator(base, size)
        adapter, session = self.adapter(register_arena=True, desc=self.desc(base, size))
        first = MemoryObj(base, 18 << 20, owner)
        second = MemoryObj(base + (18 << 20), 18 << 20, owner)
        try:
            self.assertTrue(self.store(adapter, "first", first).success)
            self.assertTrue(self.store(adapter, "second", second).success)
            self.assertEqual(session.arena_calls, [(base, size, 0)])
            self.assertEqual(session.arena, (base, size))
            self.assertIs(adapter._registration_owner, owner)
            status = adapter.report_status()
            self.assertEqual(status["registration_source"], "fixed_shm_allocator")
            self.assertEqual(status["registration_arena_bytes"], size)
        finally:
            adapter.close()

    def test_fixed_shm_owner_is_retained_until_session_close_succeeds(self):
        owner = MixedMemoryAllocator(size=64 << 20)
        adapter, session = self.adapter(register_arena=True, desc=self.desc())
        try:
            self.assertTrue(self.store(adapter, "key", MemoryObj(parent_allocator=owner)).success)
            owner_seen_at_close = []
            original_close = session.close
            def close_and_observe_owner():
                owner_seen_at_close.append(adapter._registration_owner)
                original_close()
            session.close = close_and_observe_owner

            session.close_failure = True
            with self.assertRaisesRegex(RuntimeError, "fake session close failed"):
                adapter.close()
            self.assertIs(adapter._registration_owner, owner)
            self.assertIs(owner_seen_at_close[0], owner)

            session.close_failure = False
            adapter.close()
            self.assertIs(owner_seen_at_close[1], owner)
            self.assertIsNone(adapter._registration_owner)
        finally:
            if not adapter._closed:
                session.close_failure = False
                adapter.close()

    def test_lazy_and_unknown_allocator_fall_back_to_per_object(self):
        class LazyMemoryAllocator:
            pass
        for parent in (LazyMemoryAllocator(), object()):
            with self.subTest(parent=type(parent).__name__):
                adapter, session = self.adapter(register_arena=True, desc=self.desc())
                try:
                    obj = MemoryObj(parent_allocator=parent)
                    self.assertTrue(self.store(adapter, "key", obj).success)
                    self.assertEqual(session.arena_calls, [])
                    status = adapter.report_status()
                    self.assertEqual(status["registration_source"], "per_object")
                    self.assertEqual(status["registration_fallback_reason"],
                                     "not_fixed_shm_allocator")
                finally:
                    adapter.close()

    def test_descriptor_backing_mismatch_falls_back_without_arena_registration(self):
        base, size = 4096, 64 << 20
        owner = MixedMemoryAllocator(base, size)
        for desc in (self.desc(base + 4096, size), self.desc(base, size - 4096)):
            with self.subTest(desc=desc):
                adapter, session = self.adapter(register_arena=True, desc=desc)
                try:
                    obj = MemoryObj(base, 18 << 20, owner)
                    self.assertTrue(self.store(adapter, "key", obj).success)
                    self.assertEqual(session.arena_calls, [])
                    status = adapter.report_status()
                    self.assertEqual(status["registration_source"], "per_object")
                    self.assertEqual(status["registration_fallback_reason"],
                                     "fixed_shm_bounds_not_proven")
                finally:
                    adapter.close()

    def test_registration_opt_out_keeps_per_object_path(self):
        owner = MixedMemoryAllocator()
        adapter, session = self.adapter(register_arena=False, desc=self.desc())
        try:
            self.assertTrue(self.store(adapter, "key", MemoryObj(parent_allocator=owner)).success)
            self.assertEqual(session.arena_calls, [])
            self.assertIsNone(adapter._registration_owner)
            self.assertEqual(adapter.report_status()["registration_source"], "per_object")
        finally:
            adapter.close()

    def test_chunk_option_requires_rdma_arena_and_forwards_bytes_on_owner_thread(self):
        factory = MagicMock()
        adapter_type = load_adapter(factory)
        desc = self.desc(4096, 64 << 20)
        common = dict(register_l1_arena=True, l1_registration_chunk_mb=32,
                      join_coord_url="http://127.0.0.1:29595", join_group_id="cpu-chunks")
        invalid = [
            dict(data_op_type="DEVICE_RDMA", register_l1_arena=False,
                 l1_registration_chunk_mb=32),
            dict(data_op_type="SDMA", **common),
            dict(data_op_type="DEVICE_RDMA", register_l1_arena=True,
                 l1_registration_chunk_mb=-1),
            dict(data_op_type="DEVICE_RDMA", register_l1_arena=True,
                 l1_registration_chunk_mb=1.5),
            dict(data_op_type="DEVICE_RDMA", register_l1_arena=True,
                 l1_registration_chunk_mb=True),
        ]
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                adapter_type(params, desc)
        factory.assert_not_called()

        sessions = []
        def session_factory(**kwargs):
            sessions.append(Session(**kwargs))
            return sessions[-1]
        params = dict(data_op_type="DEVICE_RDMA", **common)
        adapter = load_adapter(session_factory)(params, desc)
        owner = MixedMemoryAllocator(size=64 << 20)
        try:
            self.assertIsNone(getattr(sessions[0], "arena", None))
            self.assertTrue(self.store(adapter, "key", MemoryObj(parent_allocator=owner)).success)
            self.assertEqual(sessions[0].arena_calls, [(desc.ptr, desc.size, 32 << 20)])
            self.assertEqual(sessions[0].arena_chunk_size, 32 << 20)
            self.assertEqual(adapter.report_status()["registration_arena_bytes"], 64 << 20)
        finally:
            adapter.close()

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

    def test_failed_store_completion_is_consumed_and_next_success_is_returned(self):
        adapter, session = self.adapter()
        try:
            earlier_tid = adapter.submit_store_task(["earlier"], [MemoryObj(size=64)])
            earlier_future = adapter._tasks["store"][earlier_tid]
            done, _ = wait_futures([earlier_future], timeout=5)
            self.assertIn(earlier_future, done)

            failed_tid = adapter.submit_store_task(["bad"], [MemoryObj(size=0)])
            failed_future = adapter._tasks["store"][failed_tid]
            done, _ = wait_futures([failed_future], timeout=5)
            self.assertIn(failed_future, done)
            with self.assertRaisesRegex(ValueError, "contiguous CPU MemoryObj"):
                adapter.pop_completed_store_tasks()
            self.assertNotIn(failed_tid, adapter._tasks["store"])
            self.assertIn(earlier_tid, adapter._tasks["store"])

            earlier_result = adapter.pop_completed_store_tasks()
            self.assertEqual(set(earlier_result), {earlier_tid})
            self.assertTrue(earlier_result[earlier_tid].success)

            success_tid = adapter.submit_store_task(["good"], [MemoryObj(size=64)])
            success_future = adapter._tasks["store"][success_tid]
            done, _ = wait_futures([success_future], timeout=5)
            self.assertIn(success_future, done)
            completed = adapter.pop_completed_store_tasks()
            self.assertEqual(set(completed), {success_tid})
            self.assertTrue(completed[success_tid].success)
            self.assertEqual(session.operations, ["registered_cpu", "registered_cpu"])
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
                for action in (lambda: adapter.submit_unlock(["a"]),
                               lambda: adapter.delete(["a"]),
                               lambda: adapter.submit_store_task(["a"], [MemoryObj()]),
                               lambda: adapter._executor.submit(adapter._unlock, ["a"]).result(),
                               lambda: adapter._executor.submit(adapter._delete, ["a"]).result()):
                    with self.assertRaisesRegex(RuntimeError, "native close completion unknown"):
                        action()
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
