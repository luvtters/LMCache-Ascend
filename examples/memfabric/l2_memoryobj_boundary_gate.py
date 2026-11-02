#!/usr/bin/env python3
"""L2 adapter boundary gate using LMCache's real HostMemoryAllocator MemoryObjs.

By default this writes a plan only. Use --allow-npu to run real MemFabric BM
operations. The capacity case uses one logical L2 slot; never run alongside
another service using the same fresh join group.
"""
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
    parser.add_argument("--data-op-type", choices=("SDMA", "DEVICE_RDMA"), default="DEVICE_RDMA")
    parser.add_argument("--device-nic", required=True)
    parser.add_argument("--enable-56bits-gva", action="store_true")
    parser.add_argument("--join-coord-url", required=True)
    parser.add_argument("--join-group-id", required=True)
    parser.add_argument("--payload-bytes", type=int, default=18 << 20)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args(argv)
    if (args.device < 0 or args.world_size < 2 or not 0 < args.storage_rank < args.world_size
            or not 0 < args.payload_bytes <= 32 << 20 or args.pool_gb < 1 or args.timeout <= 0):
        parser.error("Invalid device, group, pool, payload or timeout")
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


def store(adapter, key, obj, timeout):
    task = adapter.submit_store_task([key], [obj])
    return wait(lambda: adapter.pop_completed_store_tasks().get(task), timeout)


def lookup(adapter, key, timeout):
    task = adapter.submit_lookup_and_lock_task([key], {})
    return wait(lambda: adapter.query_lookup_and_lock_result(task), timeout).test(0)


def load(adapter, key, obj, timeout):
    task = adapter.submit_load_task([key], [obj])
    return wait(lambda: adapter.query_load_result(task), timeout).test(0)


def fence(adapter):
    return adapter._executor.submit(adapter.report_status).result()


def assert_case(report, name, passed, **detail):
    item = {"case": name, "passed": bool(passed), **detail}
    report.setdefault("cases", []).append(item)
    print(json.dumps(item), flush=True)
    if not passed:
        raise AssertionError(f"case failed: {name}: {detail}")


def run(args, report):
    import torch
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter

    torch.set_num_threads(1)
    size = args.payload_bytes
    wrong_size = size - 4096 if size > 4096 else size // 2
    if size % 4096 or wrong_size <= 0 or wrong_size % 4096:
        raise ValueError("payload-bytes and the different-size object must be positive 4096-byte multiples")
    shape, wrong_shape = torch.Size((size,)), torch.Size((wrong_size,))
    allocator = HostMemoryAllocator(size=4 * size, use_paging=True,
        shapes=[shape], dtypes=[torch.uint8], fmt=MemoryFormat.BINARY)
    objects = [allocator.allocate([shape], [torch.uint8], MemoryFormat.BINARY) for _ in range(4)]
    wrong_allocator = HostMemoryAllocator(size=wrong_size, use_paging=True,
        shapes=[wrong_shape], dtypes=[torch.uint8], fmt=MemoryFormat.BINARY)
    wrong = wrong_allocator.allocate([wrong_shape], [torch.uint8], MemoryFormat.BINARY)
    if any(obj is None for obj in objects) or wrong is None:
        raise RuntimeError("Actual HostMemoryAllocator could not allocate MemoryObjs")
    source, rewrite, target, target_again = objects
    report["allocator"] = type(allocator).__name__
    report["memory_objects"] = [
        {"role": role, "type": type(obj).__name__, "bytes": obj.get_size(),
         "pointer_modulo_4096": obj.data_ptr % 4096,
         "parent_allocator": type(obj.parent_allocator).__name__}
        for role, obj in [("source", source), ("rewrite", rewrite), ("target", target),
                          ("target_again", target_again), ("wrong_size", wrong)]]
    report["slot_size_mb"] = args.pool_gb * 1024  # one slot makes locked-capacity rejection practical
    print(json.dumps({"event": "ACTUAL_MEMORYOBJ", "objects": report["memory_objects"]}), flush=True)

    params = dict(device_id=args.device, world_size=args.world_size,
        storage_rank=args.storage_rank, pool_gb=args.pool_gb, slot_size_mb=args.pool_gb * 1024,
        store_url=args.store_url, data_op_type=args.data_op_type, device_nic=args.device_nic,
        enable_56bits_gva=args.enable_56bits_gva, join_coord_url=args.join_coord_url,
        join_group_id=args.join_group_id)
    adapter = MemFabricL2Adapter(params)
    key_a = ObjectKey(b"A" * 32, "m8-l2-boundary-gate", 0)
    key_b = ObjectKey(b"B" * 32, "m8-l2-boundary-gate", 0)
    key_missing = ObjectKey(b"M" * 32, "m8-l2-boundary-gate", 0)
    try:
        source.raw_tensor.fill_(17)
        source.raw_tensor[-4096:].fill_(241)
        first = store(adapter, key_a, source, args.timeout)
        assert_case(report, "store_initial", first.is_successful() and first.bytes_transferred() == size,
                    bytes=first.bytes_transferred(), requested=size)

        assert_case(report, "lookup_existing_and_pin", lookup(adapter, key_a, args.timeout))
        full_task = adapter.submit_store_task([key_b], [source])
        try:
            full_result = wait(lambda: adapter.pop_completed_store_tasks().get(full_task), args.timeout)
        except RuntimeError as exc:
            full_mode = "raised_exception"
            full_error = str(exc)
            full_bytes = None
            full_rejected = (args.data_op_type == "DEVICE_RDMA"
                and "pool is full and all slots are locked" in full_error)
        else:
            full_mode = "result"
            full_error = ""
            full_bytes = full_result.bytes_transferred()
            full_rejected = (args.data_op_type == "SDMA"
                and not full_result.is_successful() and full_bytes == 0)
        failed_tid_consumed = full_task not in adapter._tasks["store"]
        assert_case(report, "capacity_full_all_locked_protocol_outcome",
                    full_rejected and failed_tid_consumed, error=full_error,
                    protocol=args.data_op_type, outcome=full_mode,
                    store_result_success=(full_result.is_successful() if full_mode == "result" else None),
                    bytes_transferred=full_bytes,
                    failed_task_consumed_by_adapter=failed_tid_consumed,
                    harness_workaround_used=False)

        assert_case(report, "missing_key_is_miss", not lookup(adapter, key_missing, args.timeout))
        assert_case(report, "different_size_load_is_miss",
                    not load(adapter, key_a, wrong, args.timeout), requested=wrong_size, stored=size)

        adapter.delete([key_a])
        fence(adapter)
        assert_case(report, "delete_pending_key_not_found", not lookup(adapter, key_a, args.timeout))
        adapter.submit_unlock([key_a])
        fence(adapter)
        rewrite.raw_tensor.fill_(89)
        rewrite.raw_tensor[-4096:].fill_(13)
        rewritten = store(adapter, key_a, rewrite, args.timeout)
        assert_case(report, "delete_then_rewrite_same_key",
                    rewritten.is_successful() and rewritten.bytes_transferred() == size,
                    bytes=rewritten.bytes_transferred())
        assert_case(report, "store_resumes_after_capacity_rejection",
                    rewritten.is_successful() and rewritten.bytes_transferred() == size,
                    protocol=args.data_op_type, prior_capacity_outcome=full_mode,
                    bytes=rewritten.bytes_transferred())
        assert_case(report, "lookup_rewritten", lookup(adapter, key_a, args.timeout))
        assert_case(report, "tail_bytes_round_trip",
                    load(adapter, key_a, target, args.timeout)
                    and torch.equal(target.raw_tensor, rewrite.raw_tensor),
                    bytes=size, tail_bytes=int(torch.count_nonzero(target.raw_tensor[-4096:])))
        adapter.submit_unlock([key_a])
        fence(adapter)
        assert_case(report, "eviction_after_unlock_reuses_capacity",
                    store(adapter, key_b, source, args.timeout).is_successful())
        assert_case(report, "evicted_key_is_miss", not lookup(adapter, key_a, args.timeout))
        adapter.submit_unlock([key_b])
        before_close = fence(adapter)
        assert_case(report, "queued_unlocks_drained_before_close",
                    before_close["total_pins"] == 0,
                    total_pins=before_close["total_pins"],
                    remaining_objects=before_close["objects"])

        adapter.close()
        adapter.close()
        try:
            adapter.submit_store_task([key_a], [source])
        except RuntimeError as exc:
            closed_rejected = "closed" in str(exc).lower()
        else:
            closed_rejected = False
        assert_case(report, "close_idempotent_and_rejects_new_work", closed_rejected)
        report["final_adapter_status"] = adapter.report_status()
        report["status"] = "DEVICE_ADAPTER_MEMORYOBJ_BOUNDARY_CASES_PASSED"
    finally:
        if not adapter._closed:
            adapter.close()
        for obj in objects:
            obj.ref_count_down()
        wrong.ref_count_down()
        report["allocator_memcheck"] = allocator.memcheck() and wrong_allocator.memcheck()
        if not report["allocator_memcheck"]:
            raise AssertionError("Allocator consistency check failed after teardown")


def main(argv=None):
    args = parse_args(argv)
    report = {"status": "PLAN_ONLY", "scope": "actual_L2_adapter_and_HostMemoryAllocator_MemoryObj",
        "io_boundary": "real_MemFabric_BMSession_when_allow_npu_is_set",
        "evidence_boundary": "CPU_plan_is_not_adapter_or_device_validation",
        "payload_bytes": args.payload_bytes, "capacity_slots": 1,
        "fallback_allowed": False,
        "adapter_params": {"device_id": args.device, "world_size": args.world_size,
            "storage_rank": args.storage_rank, "pool_gb": args.pool_gb,
            "slot_size_mb": args.pool_gb * 1024, "store_url": args.store_url,
            "data_op_type": args.data_op_type, "device_nic": args.device_nic,
            "enable_56bits_gva": args.enable_56bits_gva,
            "join_coord_url": args.join_coord_url, "join_group_id": args.join_group_id}}
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
