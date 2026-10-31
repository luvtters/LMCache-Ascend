#!/usr/bin/env python3
"""Actual L2 adapter / allocator MemoryObj gate; PLAN_ONLY unless --allow-npu."""
import argparse
import json
from pathlib import Path
import time
import traceback


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--storage-rank", type=int, default=1)
    parser.add_argument("--pool-gb", type=int, default=1)
    parser.add_argument("--store-url", required=True)
    parser.add_argument("--device-nic", required=True)
    parser.add_argument("--enable-56bits-gva", action="store_true")
    parser.add_argument("--payload-bytes", type=int, default=18 << 20)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args(argv)
    if (args.device < 0 or args.world_size < 2 or not 0 < args.storage_rank < args.world_size
            or not 0 < args.payload_bytes <= 32 << 20 or args.pool_gb < 1 or args.timeout <= 0):
        parser.error("Invalid device, fixed group, pool, payload or timeout")
    return args


def wait(query, timeout):
    deadline = time.monotonic() + timeout
    while True:
        value = query()
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError("L2 adapter task did not complete; completion remains unknown")
        time.sleep(0.01)


def run(args, report):
    import torch
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter

    torch.set_num_threads(1)
    size = args.payload_bytes
    shape = torch.Size((size,))
    allocator = HostMemoryAllocator(size=2 * size, use_paging=True,
        shapes=[shape], dtypes=[torch.uint8], fmt=MemoryFormat.BINARY)
    source = allocator.allocate([shape], [torch.uint8], MemoryFormat.BINARY)
    target = allocator.allocate([shape], [torch.uint8], MemoryFormat.BINARY)
    if source is None or target is None:
        raise RuntimeError("Actual HostMemoryAllocator failed to allocate two MemoryObjs")
    report["allocator"] = type(allocator).__name__
    report["memory_objects"] = [dict(role=role, type=type(obj).__name__, pointer=obj.data_ptr,
        modulo4096=obj.data_ptr % 4096, bytes=obj.get_size(), refs=obj.get_ref_count(),
        parent_allocator=type(obj.parent_allocator).__name__)
        for role, obj in (("source", source), ("target", target))]
    print(json.dumps(dict(event="ACTUAL_MEMORYOBJ", objects=report["memory_objects"])), flush=True)
    # Do not adjust addresses or extend registration to padding outside an object.
    adapter = MemFabricL2Adapter(report["adapter_params"])
    try:
        expected = (torch.arange(size, dtype=torch.int32) % 251).to(torch.uint8)
        for index, label in enumerate(("A", "B")):
            if index:
                expected = (expected.to(torch.int32) + 73).remainder(251).to(torch.uint8)
            source.raw_tensor.copy_(expected)
            key = ObjectKey(bytes([index + 1]) * 32, "registered-rdma-l2-memoryobj", 0)
            task = adapter.submit_store_task([key], [source])
            stored = wait(lambda: adapter.pop_completed_store_tasks().get(task), args.timeout)
            if not stored.is_successful() or stored.bytes_transferred() != size:
                raise AssertionError("L2 store did not complete every requested byte")
            source.raw_tensor.fill_(51)
            target.raw_tensor.fill_(90)
            task = adapter.submit_lookup_and_lock_task([key], {})
            found = wait(lambda: adapter.query_lookup_and_lock_result(task), args.timeout)
            if not found.test(0):
                raise AssertionError("Newly stored L2 key was not found")
            task = adapter.submit_load_task([key], [target])
            loaded = wait(lambda: adapter.query_load_result(task), args.timeout)
            equal = bool(loaded.test(0) and torch.equal(target.raw_tensor, expected))
            check = dict(pattern=label, full_equal=equal, verified_bytes=size if equal else 0,
                source_destroyed_before_load=True, source_refs=source.get_ref_count(),
                target_refs=target.get_ref_count())
            report.setdefault("content_checks", []).append(check)
            print(json.dumps(dict(event="L2_MEMORYOBJ_CONTENT", **check)), flush=True)
            if not equal or source.get_ref_count() != 1 or target.get_ref_count() != 1:
                raise AssertionError("L2 content or actual MemoryObj lease mismatch")
            adapter.submit_unlock([key])
        # Fence queued unlocks on the same I/O owner before reporting pins.
        report["adapter_status"] = adapter._executor.submit(adapter.report_status).result()
        if report["adapter_status"]["total_pins"]:
            raise AssertionError("L2 gate left lookup pins after explicit unlocks")
    finally:
        # A failed drain raises and retains uncertain leases in the BM session.
        adapter.close()
    source.ref_count_down()
    target.ref_count_down()
    report["allocator_memcheck"] = allocator.memcheck()
    if not report["allocator_memcheck"]:
        raise AssertionError("Allocator consistency check failed after successful teardown")
    report["status"] = "L2_MEMORYOBJ_CONTENT_AND_LOCAL_CLOSE_PASSED_EXTERNAL_WAIT_REQUIRED"


def main(argv=None):
    args = parse_args(argv)
    report = dict(status="PLAN_ONLY", scope="actual_L2_adapter_and_HostMemoryAllocator_MemoryObj",
        evidence_boundary="not_worker_direct_not_inference_not_performance",
        independent_owner_content_check=False, external_process_wait_required=True,
        payload_bytes=args.payload_bytes, fallback_allowed=False,
        adapter_params=dict(device_id=args.device, world_size=args.world_size,
            storage_rank=args.storage_rank, pool_gb=args.pool_gb, slot_size_mb=32,
            store_url=args.store_url, data_op_type="DEVICE_RDMA", device_nic=args.device_nic,
            enable_56bits_gva=args.enable_56bits_gva))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if not args.allow_npu:
        print(json.dumps(report), flush=True)
        return
    report["status"] = "RUNNING"
    try:
        run(args, report)
    except BaseException:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["ended_unix_ns"] = time.time_ns()
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
