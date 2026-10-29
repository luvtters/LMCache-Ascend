"""Opt-in real L2 adapter API matrix; CPU preparation does not initialize NPU."""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import time
import traceback

from l2_config import adapter_config, read_config, write_json

MIB = 1 << 20


def run_matrix(adapter, run_id, emit, timeout=30):
    """One 1-MiB MemoryObj per key, four 256-MiB slots in the selected pool."""
    import torch
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj

    def memory(raw):
        meta = MemoryObjMetadata(shape=raw.shape, dtype=torch.uint8,
            address=raw.data_ptr(), phy_size=raw.numel(), ref_count=1,
            fmt=MemoryFormat.BINARY, shapes=[raw.shape], dtypes=[torch.uint8])
        return TensorMemoryObj(raw, meta, None)

    keys = [ObjectKey(bytes([index]) * 32, run_id, 0) for index in range(7)]
    base = torch.arange(MIB, dtype=torch.int32)
    expected = [(base + index * 17).remainder(251).to(torch.uint8) for index in range(7)]
    sources = [memory(pattern.clone()) for pattern in expected]

    def task(kind, selected, objects=None):
        method = "lookup_and_lock" if kind == "lookup" else kind
        args = (selected, {}) if kind == "lookup" else (selected, objects)
        tid = getattr(adapter, "submit_" + method + "_task")(*args)
        emit("submitted", kind=kind, task_id=tid)
        fd = getattr(adapter, "get_" + method + "_event_fd")()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not poller.poll(50):
                continue
            os.read(fd, 8)
            result = (adapter.pop_completed_store_tasks().get(tid) if kind == "store"
                else getattr(adapter, "query_" + method + "_result")(tid))
            if result is not None:
                observed = (dict(success=result.is_successful(), bytes=result.bytes_transferred())
                    if kind == "store" else dict(bitmap=[result.test(i) for i in range(len(selected))]))
                emit("completed", kind=kind, task_id=tid, result=observed)
                return result
        raise TimeoutError(f"{kind} task {tid} did not complete")

    def state(label, *, objects, pins, free, healthy=True):
        status = adapter.report_status()
        emit("status", label=label, status=status)
        assert status["objects"] == objects and status["total_pins"] == pins, status
        assert status["free_slots"] == free and status["is_healthy"] == healthy, status
        return status

    def store(index, *, success=True, transferred=MIB, source=None):
        result = task("store", [keys[index]], [sources[index] if source is None else source])
        assert result.is_successful() == success, result
        assert result.bytes_transferred() == transferred, result

    def lookup(indices, wanted):
        result = task("lookup", [keys[index] for index in indices])
        assert [result.test(i) for i in range(len(indices))] == wanted

    comparisons = []
    def read(index):
        target = memory(torch.full((MIB,), 255, dtype=torch.uint8))
        assert task("load", [keys[index]], [target]).test(0)
        difference = torch.nonzero(target.raw_tensor != expected[index]).reshape(-1)
        record = dict(key=index, compared_bytes=MIB, mismatch_count=difference.numel(),
                      first_mismatch_offsets=difference[:16].tolist())
        emit("full_comparison", **record)
        assert difference.numel() == 0, record
        assert target.get_ref_count() == 1
        comparisons.append(record)

    state("initial", objects=0, pins=0, free=4)
    lookup([6], [False])
    for index in range(4):
        store(index)
    state("filled", objects=4, pins=0, free=0)
    lookup([0], [True])
    replacement = memory(torch.full((MIB,), 254, dtype=torch.uint8))
    store(0, transferred=0, source=replacement)
    store(4)
    lookup([1], [False])
    read(0)
    state("evicted_unlocked_key1", objects=4, pins=1, free=0)
    lookup([2, 3, 4], [True] * 3)
    state("all_slots_pinned", objects=4, pins=4, free=0)
    store(5, success=False, transferred=0)
    state("capacity_failure_stays_healthy", objects=4, pins=4, free=0)
    adapter.delete([keys[0]])
    lookup([0], [False])  # FIFO barrier after delete; must not add another pin.
    state("pinned_delete_hidden_not_reclaimed", objects=4, pins=4, free=0)
    store(5, success=False, transferred=0)
    adapter.submit_unlock([keys[0]])
    lookup([0], [False])
    state("unlock_reclaims_deleted_slot", objects=3, pins=3, free=1)
    store(5)
    lookup([5], [True])
    for index in (2, 3, 4, 5):
        read(index)
    adapter.submit_unlock([keys[index] for index in (2, 3, 4, 5)])
    adapter.delete(keys)
    lookup(list(range(7)), [False] * 7)
    final = state("drained_deleted", objects=0, pins=0, free=4)
    assert final["store_bytes"] == 6 * MIB and final["load_bytes"] == 5 * MIB, final
    assert final["evicted_objects"] == 1, final
    assert all(obj.get_ref_count() == 1 for obj in [*sources, replacement])
    emit("close_enter")
    adapter.close()
    emit("close_return")
    adapter.close()
    emit("repeat_close_return")
    try:
        adapter.submit_lookup_and_lock_task([keys[6]], {})
    except RuntimeError as error:
        assert "closed" in str(error).lower(), error
        emit("closed_submit_rejected", error=str(error))
    else:
        raise AssertionError("Closed adapter accepted a task")
    return dict(status="API_MATRIX_FUNCTION_PASS", payload_bytes=MIB, slots=4,
        comparisons=comparisons, store_bytes=6*MIB, load_bytes=5*MIB, final_status=final,
        external_real_wait_required=True, npu_verified="OUTER_EXECUTOR_MUST_CLASSIFY")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    params = adapter_config(cfg, driver=True)["adapter_params"]
    if (params["pool_gb"], params["slot_size_mb"], params["world_size"]) != (1, 256, 2):
        parser.error("This matrix requires pool_gb=1, slot_size_mb=256, world_size=2")
    if not args.run:
        print(json.dumps(dict(status="PLAN_ONLY", adapter=params, payload_bytes=MIB,
            source_devices=cfg["driver_devices"], member_devices=cfg.get("member_devices", [7]),
            automatic_device_launch=False), indent=2))
        return
    if not args.allow_npu:
        parser.error("--run requires --allow-npu and an authorized outer executor")
    out = Path(cfg["case_root"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, cfg["driver_devices"]))
    sys.path[:0] = [cfg["core_path"], cfg["ascend_path"]]
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(cfg["driver_bm_device"])
    import ctypes
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    acl.aclrtGetPhyDevIdByLogicDevId.restype = ctypes.c_int
    physical = ctypes.c_int(-1)
    logical = torch.npu.current_device()
    code = acl.aclrtGetPhyDevIdByLogicDevId(logical, ctypes.byref(physical))
    selection = dict(logical=logical, physical=physical.value, query_returncode=code,
                     visible_devices=cfg["driver_devices"])
    write_json(out / "device_selection.json", selection)
    assert code == 0 and logical == cfg["driver_bm_device"], selection
    assert physical.value == cfg["driver_devices"][logical], selection
    from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter
    adapter = None
    with (out / "events.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        def emit(event, **fields):
            row = dict(event=event, unix_ns=time.time_ns(), mono_ns=time.monotonic_ns(), **fields)
            events.write(json.dumps(row, default=str) + "\n")
            print(json.dumps(row, default=str), flush=True)
        try:
            adapter = MemFabricL2Adapter(params)
            actual_copy = adapter._session.copy
            calls = []
            def copy(src, dst, size, *, store):
                calls.append(dict(store=store, bytes=size))
                emit("BM_COPY_ENTER", src=src, dst=dst, bytes=size, store=store)
                value = actual_copy(src, dst, size, store=store)
                emit("BM_COPY_RETURN", src=src, dst=dst, bytes=size, store=store)
                return value
            adapter._session.copy = copy
            result = run_matrix(adapter, cfg["run_id"], emit, cfg.get("task_timeout", 30))
            assert len(calls) == 11
            assert sum(row["bytes"] for row in calls if row["store"]) == 6 * MIB
            assert sum(row["bytes"] for row in calls if not row["store"]) == 5 * MIB
            result["completed_bm_calls"] = calls
            write_json(out / "functional_result.json", result)
        except BaseException:
            write_json(out / "failure.json", dict(traceback=traceback.format_exc()))
            raise
        finally:
            if adapter is not None:
                adapter.close()


if __name__ == "__main__":
    main()
