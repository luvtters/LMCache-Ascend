"""Rank0: real 18 MiB CPU MemoryObj -> remote rank1 -> fresh CPU MemoryObj.

Run with the matching, independently supervised pool_member.py on the other host.
The outer supervisor must retain this process's actual wait result.
"""
import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import select
import sys
import time
import traceback

from l2_config import MIB, adapter_config, read_config, write_json


def run(cfg):
    out = Path(cfg["case_root"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, cfg["driver_devices"]))
    for path in (cfg["core_path"], cfg["ascend_path"]):
        sys.path.insert(0, path)
    import torch
    import torch_npu  # noqa: F401
    # LMCache selects its platform at first import. Match the successful
    # shared-L1 diagnostic's ordering and establish this thread's ACL device.
    torch.npu.set_device(cfg["driver_bm_device"])
    from lmcache.v1.distributed.api import ObjectKey
    from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
    from lmcache.v1.platform import current_device_spec
    from lmcache.v1.platform.torch_ops import mem_alloc
    from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter

    events = (out / "events.jsonl").open("x", encoding="utf-8", buffering=1)

    def emit(event, **fields):
        row = dict(event=event, pid=os.getpid(), unix_ns=time.time_ns(),
                   monotonic_ns=time.monotonic_ns(), **fields)
        events.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)

    def memory(raw):
        size = raw.numel()
        meta = MemoryObjMetadata(shape=torch.Size([size]), dtype=torch.uint8,
                    address=raw.data_ptr(), phy_size=size, ref_count=1,
                    fmt=MemoryFormat.BINARY, shapes=[torch.Size([size])], dtypes=[torch.uint8])
        return TensorMemoryObj(raw, meta, None)

    adapter = None
    try:
        logical = torch.npu.current_device()
        acl = ctypes.CDLL("libascendcl.so")
        acl.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        acl.aclrtGetPhyDevIdByLogicDevId.restype = ctypes.c_int
        physical = ctypes.c_int(-1)
        rc = acl.aclrtGetPhyDevIdByLogicDevId(logical, ctypes.byref(physical))
        selection = dict(visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                         logical_device=logical, physical_query_returncode=rc,
                         physical_device=physical.value, lmcache_device_type=current_device_spec.device_type,
                         lmcache_device_spec=type(current_device_spec).__module__ + "." + type(current_device_spec).__name__)
        write_json(out / "device_selection.json", selection)
        emit("DEVICE_SELECTION", **selection)
        assert logical == cfg["driver_bm_device"] and rc == 0, selection
        assert physical.value == cfg["driver_devices"][logical], selection
        assert current_device_spec.device_type == "npu", selection
        emit("initializing", store_url=cfg["store_url"], bm_rank=0, storage_rank=1)
        adapter = MemFabricL2Adapter(adapter_config(cfg, driver=True)["adapter_params"])
        write_json(out / "loaded_modules.json", {name: importlib.import_module(name).__file__
                   for name in ("torch", "torch_npu", "memfabric_hybrid", "lmcache",
                                "lmcache_ascend.v1.distributed.memfabric_l2_adapter")})
        copy_actual = adapter._session.copy

        def traced_copy(src, dst, size, *, store):
            emit("BM_COPY_ENTER", direction="STORE" if store else "LOAD", bytes=size,
                 src=src, dst=dst)
            result = copy_actual(src, dst, size, store=store)
            emit("BM_COPY_RETURN", direction="STORE" if store else "LOAD", bytes=size)
            return result

        adapter._session.copy = traced_copy

        def status(name):
            raw = adapter.report_status()
            write_json(out / (name + ".json"), raw)
            return raw

        def submit(kind, keys, objects=None):
            method = "lookup_and_lock" if kind == "lookup" else kind
            args = (keys, {}) if kind == "lookup" else (keys, objects)
            tid = getattr(adapter, "submit_" + method + "_task")(*args)
            emit("task_submitted", kind=kind, task_id=tid)
            fd = getattr(adapter, "get_" + method + "_event_fd")()
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            deadline = time.monotonic() + cfg.get("task_timeout", 120)
            while time.monotonic() < deadline:
                if not poller.poll(100):
                    continue
                os.read(fd, 8)
                result = (adapter.pop_completed_store_tasks().get(tid) if kind == "store"
                          else getattr(adapter, "query_" + method + "_result")(tid))
                if result is not None:
                    observed = (dict(success=result.is_successful(), bytes=result.bytes_transferred())
                                if kind == "store" else dict(bitmap=[bool(result.test(i)) for i in range(len(keys))]))
                    emit("task_completed", kind=kind, task_id=tid, result=observed)
                    return result
            raise TimeoutError((kind, tid))

        status("initial")
        emit("L2_DRIVER_READY")
        size = 18 * MIB
        if cfg["slot_size_mb"] * MIB < size:
            raise ValueError("18 MiB payload must fit slot")
        # Every byte participates; a non-uniform pattern catches interior damage.
        expected = torch.arange(size, dtype=torch.int32).remainder_(251).to(torch.uint8)
        source = memory(expected.clone())
        target = memory(torch.zeros(size, dtype=torch.uint8))
        # These are raw diagnostics, not claims that absence from one registry
        # proves absence of all possible native registrations.
        backend = getattr(current_device_spec, "_pin_backend_cache", None)
        allocation = dict(memory_kind="torch", shared_l1=False,
            allocator="torch CPU tensor; TensorMemoryObj parent_allocator=None",
            pinned_ptr_registry=dict(getattr(mem_alloc, "_pinned_ptr_registry", {})),
            pin_backend_class=None if backend is None else type(backend).__module__ + "." + type(backend).__name__,
            pin_backend_registered_bases=dict(getattr(backend, "_registered_bases", {})),
            objects=[dict(role=role, address=obj.data_ptr, bytes=obj.get_size(),
                          format=str(obj.get_memory_format()), shapes=[list(s) for s in obj.get_shapes()],
                          dtypes=[str(t) for t in obj.get_dtypes()],
                          torch_is_pinned=bool(obj.raw_tensor.is_pinned()),
                          parent_allocator=None if obj.parent_allocator is None else type(obj.parent_allocator).__name__)
                     for role, obj in (("source", source), ("target", target))])
        write_json(out / "allocation_query.json", allocation)
        emit("ALLOCATION_QUERY", **allocation)
        key = ObjectKey(bytes(range(32)), "cross-host-" + cfg["run_id"], 0)
        torch.save(expected, out / "expected.pt")
        result = submit("store", [key], [source])
        assert result.is_successful() and result.bytes_transferred() == size
        status("after_store")
        source.raw_tensor.zero_()
        assert submit("lookup", [key]).test(0)
        assert submit("load", [key], [target]).test(0)
        torch.save(target.raw_tensor, out / "actual.pt")
        differences = torch.nonzero(expected != target.raw_tensor, as_tuple=False).reshape(-1)
        first = differences[:64]
        comparison = dict(compared_bytes=size, mismatch_count=differences.numel(),
             first_mismatch_offsets=first.tolist(), expected_values=expected[first].tolist(),
             actual_values=target.raw_tensor[first].tolist(), observed_head=target.raw_tensor[:32].tolist(),
             observed_tail=target.raw_tensor[-32:].tolist(), full_comparison=True)
        write_json(out / "comparison.json", comparison)
        emit("PAYLOAD_FULL_COMPARISON", **comparison)
        assert differences.numel() == 0, comparison
        assert source.get_ref_count() == target.get_ref_count() == 1
        adapter.submit_unlock([key])
        adapter.delete([key])
        # Lookup is queued behind unlock/delete on the stock single I/O worker.
        assert not submit("lookup", [key]).test(0)
        final = status("after_delete")
        assert final["objects"] == final["total_pins"] == final["locked_objects"] == 0
        assert final["store_bytes"] == final["load_bytes"] == size
        emit("BM_WAIT_ENTER")
        adapter._executor.submit(adapter._session.wait).result()
        emit("BM_WAIT_RETURN")
        write_json(out / "functional_result.json", dict(status="PASS_FUNCTION_ONLY", bytes=size,
                   remote_rank=1, process_exit="EXTERNAL_SUPERVISOR_MUST_WAIT"))
    except BaseException:
        write_json(out / "failure.json", dict(traceback=traceback.format_exc(), unix_ns=time.time_ns()))
        if adapter is not None:
            status("failure_status")
        raise
    finally:
        if adapter is not None:
            emit("ADAPTER_CLOSE_ENTER")
            adapter.close()
            emit("ADAPTER_CLOSE_RETURN")
        events.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(read_config(parser.parse_args().config))
