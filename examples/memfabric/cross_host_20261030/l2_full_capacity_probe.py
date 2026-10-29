"""Opt-in full 1-GiB selected pool and 32/33-MiB object boundary probe."""
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
SLOT = 32 * MIB
POOL = 1 << 30
SLOTS = 32


def run_capacity_matrix(adapter, run_id, emit, timeout=30):
    import torch
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj

    def memory(raw):
        meta = MemoryObjMetadata(shape=raw.shape, dtype=torch.uint8,
            address=raw.data_ptr(), phy_size=raw.numel(), ref_count=1,
            fmt=MemoryFormat.BINARY, shapes=[raw.shape], dtypes=[torch.uint8])
        return TensorMemoryObj(raw, meta, None)

    pattern = torch.arange(256, dtype=torch.uint8).repeat(SLOT // 256)
    source = memory(torch.empty(SLOT, dtype=torch.uint8))
    target = memory(torch.empty(SLOT, dtype=torch.uint8))
    keys = [ObjectKey(bytes([index]) * 32, run_id, 0) for index in range(34)]
    calls = []
    actual_copy = adapter._session.copy

    def traced_copy(src, dst, size, *, store):
        row = dict(src=src, dst=dst, bytes=size, store=store, completed=False)
        calls.append(row)
        emit("BM_COPY_ENTER", **row)
        value = actual_copy(src, dst, size, store=store)
        row["completed"] = True
        emit("BM_COPY_RETURN", **row)
        return value

    adapter._session.copy = traced_copy

    def task(kind, indices, objects=None):
        method = "lookup_and_lock" if kind == "lookup" else kind
        selected = [keys[index] for index in indices]
        args = (selected, {}) if kind == "lookup" else (selected, objects)
        tid = getattr(adapter, "submit_" + method + "_task")(*args)
        emit("submitted", kind=kind, indices=indices, task_id=tid)
        fd = getattr(adapter, "get_" + method + "_event_fd")()
        poller = select.poll(); poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not poller.poll(50):
                continue
            os.read(fd, 8)
            result = (adapter.pop_completed_store_tasks().get(tid) if kind == "store"
                else getattr(adapter, "query_" + method + "_result")(tid))
            if result is not None:
                observed = (dict(success=result.is_successful(), bytes=result.bytes_transferred())
                    if kind == "store" else dict(bitmap=[result.test(i) for i in range(len(indices))]))
                emit("completed", kind=kind, indices=indices, task_id=tid, result=observed)
                return result
        raise TimeoutError(f"{kind} task {tid} did not complete")

    def state(label, objects, pins, free):
        status = adapter.report_status()
        emit("status", label=label, status=status)
        assert status["is_healthy"] and status["objects"] == objects, status
        assert status["total_pins"] == pins and status["free_slots"] == free, status
        assert status["local_selected_pool_bytes"] == POOL, status
        return status

    def lookup(indices, wanted):
        result = task("lookup", indices)
        assert [result.test(i) for i in range(len(indices))] == wanted

    def store(index):
        # This write happens only after the previous task's SDK copy+wait has
        # returned and its result was consumed. No batch shares mutable contents.
        assert source.get_ref_count() == 1
        torch.bitwise_xor(pattern, index, out=source.raw_tensor)
        emit("source_pattern_ready", key=index, address=source.data_ptr, bytes=SLOT)
        result = task("store", [index], [source])
        assert result.is_successful() and result.bytes_transferred() == SLOT
        assert source.get_ref_count() == 1

    comparisons = []
    def read(index):
        target.raw_tensor.fill_(255)
        assert task("load", [index], [target]).test(0)
        expected = torch.bitwise_xor(pattern, index)
        mismatches = int(torch.count_nonzero(target.raw_tensor != expected).item())
        row = dict(key=index, compared_bytes=SLOT, mismatches=mismatches)
        emit("full_comparison", **row)
        assert mismatches == 0, row
        assert target.get_ref_count() == 1
        comparisons.append(row)

    initial = state("initial", 0, 0, SLOTS)
    for index in range(SLOTS):
        store(index)
    full = state("actual_1gib_written", SLOTS, 0, 0)
    assert full["store_bytes"] == POOL
    stores = [row for row in calls if row["store"]]
    first = initial["storage_peer_gva"]
    assert len(stores) == SLOTS and all(row["completed"] for row in stores)
    with adapter._state_lock:
        before_entries = {str(index): dict(slot=adapter._entries[keys[index]].slot,
            gva=first + adapter._entries[keys[index]].slot * SLOT,
            bytes=adapter._entries[keys[index]].size) for index in range(SLOTS)}
    emit("slot_snapshot", label="full_pool_before_lru", key_entries=before_entries)
    assert {row["slot"] for row in before_entries.values()} == set(range(SLOTS))
    assert [row["dst"] for row in stores] == [before_entries[str(i)]["gva"] for i in range(SLOTS)]
    assert max(row["dst"] for row in stores) + SLOT == first + POOL
    lookup(list(range(SLOTS)), [True] * SLOTS)
    for index in range(SLOTS):
        read(index)
    adapter.submit_unlock(keys[:SLOTS])
    lookup([33], [False])  # FIFO barrier after unlock; miss does not alter LRU.
    state("all_32_objects_fully_verified", SLOTS, 0, 0)
    # Sequentially reading every object leaves key0 the oldest unlocked entry.
    store(32)
    lookup([0, 32], [False, True])
    read(32)
    adapter.submit_unlock([keys[32]])
    lookup([33], [False])
    after_lru = state("33rd_object_evicts_key0", SLOTS, 0, 0)
    assert after_lru["evicted_objects"] == 1
    with adapter._state_lock:
        replacement = dict(slot=adapter._entries[keys[32]].slot,
            gva=first + adapter._entries[keys[32]].slot * SLOT,
            bytes=adapter._entries[keys[32]].size)
    emit("slot_snapshot", label="replacement_after_lru", key=32, entry=replacement,
         evicted_key=0, evicted_entry=before_entries["0"])
    assert replacement == before_entries["0"]
    assert [row for row in calls if row["store"]][-1]["dst"] == replacement["gva"]
    oversize = memory(torch.zeros(SLOT + MIB, dtype=torch.uint8))
    before_calls = len(calls)
    result = task("store", [33], [oversize])
    assert not result.is_successful() and result.bytes_transferred() == 0
    assert len(calls) == before_calls, "Oversized object submitted a BM copy"
    after_oversize = state("33mib_rejected_without_bm_copy", SLOTS, 0, 0)
    assert after_oversize["evicted_objects"] == 1
    adapter.delete(keys)
    lookup(list(range(34)), [False] * 34)
    final = state("drained_deleted", 0, 0, SLOTS)
    assert final["store_bytes"] == 33 * SLOT and final["load_bytes"] == 33 * SLOT
    assert len(calls) == 66 and all(row["completed"] for row in calls)
    assert all(row["bytes"] == SLOT for row in calls)
    assert source.get_ref_count() == target.get_ref_count() == oversize.get_ref_count() == 1
    return dict(status="FULL_1GIB_CAPACITY_FUNCTION_PASS", selected_pool_bytes=POOL,
        slot_bytes=SLOT, initial_filled_objects=SLOTS, initial_written_bytes=POOL,
        total_store_bytes=33*SLOT, total_load_bytes=33*SLOT,
        first_slot_address=first, last_slot_address=first+31*SLOT,
        verified_last_slot_end=first+POOL, evicted_key=0, new_key=32,
        before_lru_key_entries=before_entries, replacement_entry=replacement,
        comparisons=comparisons, oversize_bytes=SLOT+MIB, oversize_completed_bytes=0,
        oversize_bm_calls=0, completed_bm_calls=calls, final_status=final,
        external_real_wait_required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-npu", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    params = adapter_config(cfg, driver=True)["adapter_params"]
    if (params["pool_gb"], params["slot_size_mb"], params["world_size"]) != (1, 32, 2):
        parser.error("Requires pool_gb=1, slot_size_mb=32, world_size=2")
    if not args.run:
        print(json.dumps(dict(status="PLAN_ONLY", adapter=params, object_bytes=SLOT,
            oversize_bytes=SLOT+MIB, initial_write_bytes=POOL, no_device_launch=True), indent=2))
        return
    if not args.allow_npu:
        parser.error("--run requires --allow-npu and an authorized outer executor")
    out = Path(cfg["case_root"]); out.mkdir(parents=True, exist_ok=False)
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
    logical = torch.npu.current_device(); physical = ctypes.c_int(-1)
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
            result = run_capacity_matrix(adapter, cfg["run_id"], emit, cfg.get("task_timeout", 30))
            write_json(out / "functional_result.json", result)
        except BaseException:
            write_json(out / "failure.json", dict(traceback=traceback.format_exc()))
            raise
        finally:
            if adapter is not None:
                emit("close_enter")
                try:
                    adapter.close()
                except BaseException as error:
                    write_json(out / "close_result.json", dict(success=False, error=repr(error)))
                    raise
                emit("close_return")
                write_json(out / "close_result.json", dict(success=True))


if __name__ == "__main__":
    main()
