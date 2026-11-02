#!/usr/bin/env python3
"""Two-process real-NPU gate for native LMCache L1 backed by BM HOST memory.

No model, vLLM connector, DirectTransferContext, PoolIndex, or L2 adapter.
Files carry control metadata only; KV bytes move through BM L2G/G2L.
Run pool and worker with distinct logical devices and the same fresh run-dir.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def emit(root, name, **data):
    target = root / (name + ".json")
    temporary = target.with_suffix(".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps(dict(event=name, **data)), flush=True)


def wait(root, name, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        failed = list(root.glob("failed-*.json"))
        if failed:
            raise RuntimeError(failed[0].read_text(encoding="utf-8"))
        target = root / (name + ".json")
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise TimeoutError(name)


def run(args):
    sys.path[:0] = [str(args.ascend_source), str(args.core_source)]
    import torch
    import torch_npu  # noqa: F401
    from lmcache_ascend.v1.distributed.memfabric_bm import BMSession
    import lmcache
    import lmcache_ascend

    emit(args.run_dir, "source-" + args.role,
         core=lmcache.__file__, ascend=lmcache_ascend.__file__, argv=sys.argv,
         shape=args.shape, device=args.device)
    session = BMSession(
        device_id=args.device, rank_id=0 if args.role == "pool" else 1,
        world_size=2, pool_gb=1, store_url=args.store_url,
        pool_memory_type="host", data_op_type="DEVICE_RDMA",
        device_nic=args.device_nic,
        enable_56bits_gva=True, init_timeout=int(args.timeout),
    )
    storage = None
    safe_to_close = args.role == "worker"
    try:
        if not session._local_join_complete.wait(args.timeout):
            raise TimeoutError("BM own-rank after-mmap JOIN callback")
        emit(args.run_dir, "joined-" + args.role, after_mmap_join=True)
        wait(args.run_dir, "joined-worker" if args.role == "pool" else "joined-pool",
             args.timeout)
        session._join_barrier_complete = True
        if args.role == "pool":
            from lmcache.v1.distributed.api import (
                GroupedObjectKeys, MemoryLayoutDesc, ObjectKey, PrefetchTaskSpec,
            )
            from lmcache.v1.distributed.config import (
                EvictionConfig, L1ManagerConfig, L1MemoryManagerConfig,
                StorageManagerConfig,
            )
            from lmcache.v1.distributed.error import L1Error
            from lmcache.v1.distributed.l1_manager import L1Manager
            from lmcache.v1.distributed.storage_manager import StorageManager
            from lmcache_ascend.v1.distributed.memfabric_l1_memory import MemFabricL1MemoryManager

            memory = MemFabricL1MemoryManager(session)
            config = L1ManagerConfig(L1MemoryManagerConfig(
                size_in_bytes=memory.size_bytes, use_lazy=False, shm_name=""
            ))
            manager = L1Manager(config, memory_manager=memory)
            storage = StorageManager(
                StorageManagerConfig(config, EvictionConfig("noop")),
                _l1_managers=(manager,),
            )
            key = ObjectKey(b"native-bm-l1-gate", "synthetic-kv", 0)
            layout = MemoryLayoutDesc([torch.Size(args.shape)], [torch.bfloat16])
            objects = storage.reserve_write([key], layout)
            if key not in objects:
                raise RuntimeError("Native L1 reserve_write did not allocate")
            obj = objects[key]
            safe_to_close = False
            emit(args.run_dir, "reserved", gva=memory.global_address(obj),
                 bytes=obj.get_size(), offset=obj.tensor.data_ptr() - memory.local_base,
                 memory_obj_type=type(obj).__name__, owner=obj.get_l1_manager())
            # Uncommitted objects must remain invisible to native readers.
            if manager.reserve_read([key])[key][0] == L1Error.SUCCESS:
                raise RuntimeError("Native staging object became readable before commit")
            wait(args.run_dir, "write-done", args.timeout)
            expected = (torch.arange(obj.tensor.numel(), dtype=torch.int32) % 127 - 63)
            expected = expected.to(torch.bfloat16).reshape(args.shape)
            if not torch.equal(obj.tensor, expected):
                raise RuntimeError("Native MemoryObj does not contain the BM-written KV bytes")
            storage.finish_write([key])
            spec = PrefetchTaskSpec(
                [GroupedObjectKeys([key], 0, layout)],
                allowed_l2_memory_tags=frozenset(),
            )
            handle = storage.submit_prefetch_task(spec, "bm-l1-gate")
            deadline = time.monotonic() + args.timeout
            result = None
            while result is None and time.monotonic() < deadline:
                result = storage.query_prefetch_status(handle)
                if result is None:
                    time.sleep(0.01)
            if result is None or result.l1_hit_count != 1 or result.l2_hit_count != 0:
                raise RuntimeError("Native L1 prefetch did not return one L1-only hit")
            keys, readable = storage.unsafe_read([key])
            if keys != [key] or len(readable) != 1:
                raise RuntimeError("Native protected read did not return the object")
            if storage.delete_l1_keys([key]) != (0, 1):
                raise RuntimeError("Native read lock did not prevent eviction")
            emit(args.run_dir, "readable", gva=memory.global_address(readable[0]),
                 l1_hits=result.l1_hit_count, l2_hits=result.l2_hit_count,
                 read_lock_blocks_delete=True, host_content_equal=True)
            wait(args.run_dir, "read-done", args.timeout)
            storage.finish_read_prefetched([key])
            if storage.delete_l1_keys([key]) != (1, 0):
                raise RuntimeError("Native L1 object was not released after read completion")
            if storage.get_l1_usage()[0] != 0 or not storage.memcheck():
                raise RuntimeError("Native L1 allocation remained after deletion")
            emit(args.run_dir, "released", used_bytes=0, native_memcheck=True)
            wait(args.run_dir, "worker-closed", args.timeout)
            safe_to_close = True
        else:
            torch.npu.set_device(args.device)
            elements = 1
            for dim in args.shape:
                elements *= dim
            expected = (torch.arange(elements, dtype=torch.int32,
                                     device=f"npu:{args.device}") % 127 - 63)
            expected = expected.to(torch.bfloat16).reshape(args.shape)
            payload = expected.clone()
            torch.npu.synchronize(args.device)
            session.register_npu_tensor(payload)
            slot = wait(args.run_dir, "reserved", args.timeout)
            size = payload.numel() * payload.element_size()
            if slot["bytes"] != size:
                raise RuntimeError("NPU and native L1 object sizes disagree")
            session.copy_npu_to_global(payload.data_ptr(), slot["gva"], size)
            session.wait()
            emit(args.run_dir, "write-done", bytes=size, completion="synchronous-L2G")
            payload.fill_(-79)
            torch.npu.synchronize(args.device)
            read = wait(args.run_dir, "readable", args.timeout)
            session.copy_global_to_npu(read["gva"], payload.data_ptr(), size)
            session.wait()
            torch.npu.synchronize(args.device)
            if not torch.equal(payload, expected):
                raise RuntimeError("NPU roundtrip contents differ")
            emit(args.run_dir, "read-done", bytes=size, content_equal=True,
                 completion="synchronous-G2L")
            wait(args.run_dir, "released", args.timeout)
    except BaseException:
        emit(args.run_dir, "failed-" + args.role, error=traceback.format_exc())
        if not safe_to_close or session._completion_unknown:
            # A timeout is not device completion. Keep the native L1 arena,
            # MemoryObjs and BM session alive for the external run guard.
            # Never call L1Manager.close(), which frees outstanding objects.
            emit(args.run_dir, "retained-" + args.role,
                 reason="peer/device completion not established; external guard required")
            while True:
                time.sleep(1)
        raise
    finally:
        if safe_to_close and not session._completion_unknown:
            if storage is not None:
                storage.close()
            session.close()
    emit(args.run_dir, args.role + "-closed", clean_close=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("pool", "worker"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--store-url", default="tcp://127.0.0.1:28890")
    parser.add_argument("--device-nic", default="tcp://127.0.0.1:10005")
    parser.add_argument("--shape", type=int, nargs=4, default=[2, 36, 128, 1024])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args()
    if not args.allow_npu:
        parser.error("Device gate requires --allow-npu")
    if min(args.shape) <= 0 or args.device < 0 or not 1 <= args.timeout <= 3600:
        parser.error("Invalid shape, device or timeout")
    if 2 * __import__("math").prod(args.shape) > (1 << 30):
        parser.error("KV object must fit the 1 GiB L1 pool")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except BaseException:
        emit(args.run_dir, "failed-" + args.role, error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
