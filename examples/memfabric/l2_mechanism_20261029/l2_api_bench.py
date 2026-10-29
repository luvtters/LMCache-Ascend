#!/usr/bin/env python3
"""Opt-in, no-model benchmark of real LMCache L2 adapters and MemoryObj objects.

Run in a fresh externally supervised process. Native teardown and the actual
process wait code must be retained by that supervisor, including crashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import select
import sys
import threading
import time
import uuid

MIB = 1024 * 1024
CASES = ([dict(id=f"e01_{n}x18", objects=n, object_bytes=18*MIB, experiment="E01a")
          for n in [1, 16, 32, 64, 115, 128]] +
         [dict(id=f"e02_total144_{n}", objects=n, object_bytes=144*MIB//n, experiment="E02")
          for n in [8, 16, 32, 128]] +
         [dict(id=f"e02_count32_{s}", objects=32, object_bytes=s*MIB, experiment="E02")
          for s in [1, 4, 18]])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf8")
    temporary.replace(path)


class Recorder:
    def __init__(self, path, backend, run_id, role="client"):
        self.stream = Path(path).open("x", encoding="utf8", buffering=1)
        self.lock = threading.Lock()
        self.common = dict(backend=backend, run_id=run_id, role=role)

    def emit(self, event, **fields):
        row = dict(self.common, event=event, pid=os.getpid(), tid=threading.get_native_id(),
                   unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
        row.update(fields)
        with self.lock:
            self.stream.write(json.dumps(row, default=str) + "\n")


class TracedNativeClient:
    """Observe existing SDK submit/drain boundaries; SDK execution stays opaque."""
    def __init__(self, actual, recorder):
        self.actual = actual
        self.recorder = recorder
        self.last_submission = None

    def __getattr__(self, name):
        target = getattr(self.actual, name)
        if name not in ("submit_batch_set", "submit_batch_get", "submit_batch_exists",
                        "submit_batch_delete", "drain_completions"):
            return target

        def invoke(*args, **kwargs):
            started = time.monotonic_ns()
            result = target(*args, **kwargs)
            if name.startswith("submit_batch_"):
                self.last_submission = dict(api=name, native_future_id=result)
            self.recorder.emit("native_api_return", api=name, started_monotonic_ns=started,
                               key_count=len(args[0]) if args else None, result=result,
                               sdk_worker_queue="UNKNOWN", sdk_transfer_bytes="UNKNOWN")
            return result
        return invoke


class TracedExecutor:
    """Keep the original Future and worker; record submit and dequeue boundaries."""
    def __init__(self, actual, recorder):
        self.actual, self.recorder = actual, recorder
        self.serial = 0

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def submit(self, fn, *args, **kwargs):
        self.serial += 1
        op_id = self.serial
        self.recorder.emit("executor_submit", op_id=op_id, operation=getattr(fn, "__name__", str(fn)),
                           queue_depth=self.actual._work_queue.qsize())

        def work():
            self.recorder.emit("executor_dequeue", op_id=op_id,
                               queue_depth=self.actual._work_queue.qsize())
            try:
                result = fn(*args, **kwargs)
            except BaseException as error:
                self.recorder.emit("executor_end", op_id=op_id, error=repr(error))
                raise
            self.recorder.emit("executor_end", op_id=op_id, error=None)
            return result
        future = self.actual.submit(work)
        self.recorder.emit("executor_submit_return", op_id=op_id, python_future_object_id=id(future))
        return future


def attach_mf_trace(adapter, recorder):
    adapter._executor = TracedExecutor(adapter._executor, recorder)
    actual_copy = adapter._session.copy

    def copy(src, dst, size, *, store):
        started = time.monotonic_ns()
        try:
            result = actual_copy(src, dst, size, store=store)
        except BaseException as error:
            recorder.emit("bm_copy_return", started_monotonic_ns=started, direction="store" if store else "load",
                          requested_bytes=size, success_bytes=0, error=repr(error))
            raise
        recorder.emit("bm_copy_return", started_monotonic_ns=started, direction="store" if store else "load",
                      requested_bytes=size, success_bytes=size, error=None)
        return result
    adapter._session.copy = copy


def owner_command(owner, op, entries, output, label, timeout=90):
    request_id = uuid.uuid4().hex
    folder = Path(owner["query_dir"])
    request = dict(request_id=request_id, op=op, entries=entries)
    atomic_json(output / (label + ".request.json"), request)
    atomic_json(folder / (request_id + ".request.json"), request)
    reply = folder / (request_id + ".reply.json")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stat = Path("/proc") / str(owner["pid"]) / "stat"
        if not stat.exists() or stat.read_text().split()[21] != owner["proc_start_ticks"]:
            raise RuntimeError("Independent owner exited or PID changed")
        if reply.exists():
            raw = reply.read_bytes()
            (output / (label + ".reply.json")).write_bytes(raw)
            result = json.loads(raw)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        time.sleep(0.01)
    raise TimeoutError((op, request_id))


def verify_placement(result, owner, client_endpoint):
    rows = result["objects"]
    for entry in rows:
        replicas = entry["replicas"]
        if len(replicas) != 1:
            raise RuntimeError(("Expected one unsplit replica", entry))
        r = replicas[0]
        if (r["status"] != "ReplicaStatus.COMPLETE" or not r["is_memory"]
                or r["transport_endpoint"] != owner["hostname"]
                or r["transport_endpoint"] == client_endpoint
                or r["size"] < entry["size"] or r["buffer_address"] <= 0):
            raise RuntimeError(("Object is not fully in the independent owner", entry))


def bitmap_values(bitmap, count):
    return [bool(bitmap.test(i)) for i in range(count)]


def wait_task(adapter, kind, task_id, recorder, key_count, timeout):
    fd = getattr(adapter, "get_" + ("lookup_and_lock" if kind == "lookup" else kind) + "_event_fd")()
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Follow the public eventfd contract before consuming its result. A
        # query-first fast path could leave this task's notification unread
        # and incorrectly associate it with the next task in the trace.
        if not poller.poll(100):
            continue
        os.read(fd, 8)
        recorder.emit("adapter_event_consumed", operation=kind, task_id=task_id)
        if kind == "store":
            result = adapter.pop_completed_store_tasks().get(task_id)
        else:
            result = getattr(adapter, "query_" + ("lookup_and_lock" if kind == "lookup" else kind) + "_result")(task_id)
        if result is not None:
            raw = (dict(success=result.is_successful(), bytes=result.bytes_transferred())
                   if kind == "store" else dict(bitmap=bitmap_values(result, key_count)))
            recorder.emit("adapter_result_observed", operation=kind, task_id=task_id,
                          key_count=key_count, result=raw)
            return result
    raise TimeoutError((kind, task_id))


def submit_wait(adapter, kind, keys, objects, recorder, timeout):
    started = time.monotonic_ns()
    recorder.emit("adapter_submit_enter", operation=kind, key_count=len(keys))
    if kind == "lookup":
        task_id = adapter.submit_lookup_and_lock_task(keys, {})
    else:
        task_id = getattr(adapter, "submit_" + kind + "_task")(keys, objects)
    submitted = time.monotonic_ns()
    mapping = {}
    if hasattr(adapter, "_tasks"):
        future = adapter._tasks[kind].get(task_id)
        mapping["python_future_object_id"] = id(future) if future is not None else None
    if isinstance(getattr(adapter,"_client",None),TracedNativeClient):
        mapping.update(adapter._client.last_submission or {})
    recorder.emit("adapter_submit_return", operation=kind, task_id=task_id, key_count=len(keys),
                  started_monotonic_ns=started, **mapping)
    result = wait_task(adapter, kind, task_id, recorder, len(keys), timeout)
    return result, dict(submit_ns=submitted-started, observed_completion_ns=time.monotonic_ns()-started)


def wait_operations(adapter, tasks, recorder, timeout):
    """Observe both real completion fds; a shared poll batch has unknown order."""
    poller=select.poll()
    fd_to_kind={}
    for kind in tasks:
        fd=getattr(adapter,"get_"+kind+"_event_fd")()
        fd_to_kind[fd]=kind
        poller.register(fd,select.POLLIN)
    end=time.monotonic()+timeout
    completed={}
    while len(completed)<len(tasks) and time.monotonic()<end:
        ready=poller.poll(100)
        observed=time.monotonic_ns()
        if not ready:continue
        recorder.emit("rw_completion_poll",ready_operations=[fd_to_kind[fd] for fd,_ in ready],
                      same_poll_batch_order="UNKNOWN" if len(ready)>1 else "one fd ready")
        for fd,_ in ready:
            kind=fd_to_kind[fd]
            os.read(fd,8)
            task_id,count=tasks[kind]
            result=(adapter.pop_completed_store_tasks().get(task_id) if kind=="store"
                    else adapter.query_load_result(task_id))
            if result is None:continue
            raw=(dict(success=result.is_successful(),adapter_accounted_bytes=result.bytes_transferred())
                 if kind=="store" else dict(bitmap=bitmap_values(result,count)))
            recorder.emit("rw_completion_observed",operation=kind,task_id=task_id,result=raw,
                          poll_observed_monotonic_ns=observed)
            completed[kind]=(result,time.monotonic_ns())
            poller.unregister(fd)
    if len(completed)!=len(tasks):raise TimeoutError("Read/write completion is incomplete")
    return completed


def rw_probe(a,adapter,buffers,torch,ObjectKey,recorder,owner,config,run_id):
    """E05: 64 immutable reads + at most 32 distinct writes in a 128-slot pool."""
    folder=a.output/"e05_rw"
    folder.mkdir()
    count,size=64,18*MIB
    keys=[ObjectKey(bytes.fromhex(run_id)+i.to_bytes(16,"big"),"l2-rw-read",0) for i in range(count)]
    values=[i+1 for i in range(count)]
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import _object_key_to_string
    entries=[dict(key=_object_key_to_string(k),size=size,value=v) for k,v in zip(keys,values)]
    if owner:
        verify_placement(owner_command(owner,"seed",entries,folder,"seed_read",a.timeout),
                         owner,config["local_hostname"])
    else:
        source=buffers.allocate(size,values)
        result,_=submit_wait(adapter,"store",keys,source,recorder,a.timeout)
        if not result.is_successful():raise RuntimeError("Read-set seed failed")
        buffers.free(source)
    targets=buffers.allocate(size,[0]*count)
    # Keep identical addresses and allocation pressure in both conditions.
    # Baseline owns these prepared buffers too; only its SDK SET is omitted.
    write_source=buffers.allocate(size,[i+129 for i in range(32)])
    recorder.emit("rw_buffers_prepared",read_target_addresses=[obj.data_ptr for obj in targets],
                  write_source_addresses=[obj.data_ptr for obj in write_source],
                  allocation_pressure_same_for_both_conditions=True)
    save_readback(folder/"write_sources.raw.json",write_source,[i+129 for i in range(32)])
    if owner:
        verify_placement(owner_command(owner,"query",entries,folder,"read_placement_before",a.timeout),
                         owner,config["local_hostname"])
    with (folder/"samples.raw.jsonl").open("x",encoding="utf8",buffering=1) as output:
        for block in range(a.blocks):
            for iteration in range(a.warmups+a.repeats):
                conditions=("read_only","store_before_load") if iteration%2==0 else ("store_before_load","read_only")
                for condition in conditions:
                    label=f"b{block:02d}_i{iteration:03d}_{condition}"
                    before=adapter.report_status()
                    atomic_json(folder/(label+".before.raw.json"),before)
                    if a.backend=="memfabric" and (before["objects"]!=64 or before["evicted_objects"]!=0):
                        raise RuntimeError("E05 requires exactly 64 read objects and no eviction")
                    if a.backend=="memfabric" and before["free_slots"]<32:
                        raise RuntimeError("E05 requires 32 additional free slots; do not confound eviction")
                    for obj in targets:obj.raw_tensor.zero_()
                    write_keys=[]
                    write_objects=[]
                    if condition=="store_before_load":
                        write_keys=[ObjectKey(bytes.fromhex(run_id)+i.to_bytes(16,"big"),
                                     f"l2-rw-write-{block}-{iteration}",0) for i in range(32)]
                        write_objects=write_source
                    found,_=submit_wait(adapter,"lookup",keys,None,recorder,a.timeout)
                    if not all(bitmap_values(found,count)):
                        adapter.submit_unlock(keys)
                        raise RuntimeError("Read set missing before queue probe")
                    tasks={}
                    store_started=store_submitted=None
                    try:
                        if write_keys:
                            store_started=time.monotonic_ns()
                            recorder.emit("rw_submit_enter",condition=condition,operation="store",key_count=32)
                            store_id=adapter.submit_store_task(write_keys,write_objects)
                            store_submitted=time.monotonic_ns()
                            recorder.emit("rw_submit_return",condition=condition,operation="store",task_id=store_id,key_count=32)
                            tasks["store"]=(store_id,32)
                        load_started=time.monotonic_ns()
                        recorder.emit("rw_submit_enter",condition=condition,operation="load",key_count=count)
                        load_id=adapter.submit_load_task(keys,targets)
                        load_submitted=time.monotonic_ns()
                        recorder.emit("rw_submit_return",condition=condition,operation="load",task_id=load_id,key_count=count)
                        tasks["load"]=(load_id,count)
                        completed=wait_operations(adapter,tasks,recorder,a.timeout)
                        if not all(bitmap_values(completed["load"][0],count)):
                            raise RuntimeError("Queue-probe load incomplete")
                        if write_keys and not completed["store"][0].is_successful():
                            raise RuntimeError("Queue-probe store failed")
                    finally:
                        adapter.submit_unlock(keys)
                    check_memory(torch,targets,values)
                    if block==0 and iteration==0:
                        save_readback(folder/(condition+".readback.raw.json"),targets,values)
                    row=dict(block=block,iteration=iteration,condition=condition,
                             phase="warmup" if iteration<a.warmups else "timed",
                             load_objects=64,load_payload_bytes=64*size,store_objects=len(write_keys),
                             store_payload_bytes=len(write_keys)*size,
                             load_started_monotonic_ns=load_started,load_submitted_monotonic_ns=load_submitted,
                             load_completion_observed_monotonic_ns=completed["load"][1],
                             store_started_monotonic_ns=store_started,store_submitted_monotonic_ns=store_submitted,
                             store_completion_observed_monotonic_ns=completed["store"][1] if write_keys else None,
                             full_content_checked=True,store_sdk_internal_dequeue="UNKNOWN",
                             transport_bytes="see MF counters; Store SDK bytes UNKNOWN")
                    output.write(json.dumps(row)+"\n")
                    if write_keys:
                        if owner:
                            write_entries=[dict(key=_object_key_to_string(k),size=size) for k in write_keys]
                            verify_placement(owner_command(owner,"query",entries+write_entries,folder,
                                              label+".placement",a.timeout),owner,config["local_hostname"])
                        adapter.delete(write_keys)
                    if a.backend=="memfabric":
                        adapter._executor.submit(lambda:None).result(timeout=a.timeout)
                    after=adapter.report_status()
                    atomic_json(folder/(label+".after.raw.json"),after)
                    if a.backend=="memfabric" and (after["objects"]!=64 or after["evicted_objects"]!=0 or after["total_pins"]!=0):
                        raise RuntimeError("Queue probe changed read-set residency or left pins")
    if owner:
        verify_placement(owner_command(owner,"query",entries,folder,"read_placement_after",a.timeout),
                         owner,config["local_hostname"])
    adapter.delete(keys)
    if a.backend=="memfabric":adapter._executor.submit(lambda:None).result(timeout=a.timeout)
    atomic_json(folder/"final_adapter_status.raw.json",adapter.report_status())
    buffers.free(targets)
    buffers.free(write_source)


def make_memory(torch, memory_classes, size, value):
    fmt, metadata, tensor_obj = memory_classes
    raw = torch.empty(size, dtype=torch.uint8, device="cpu")
    raw.fill_(value)
    raw[0] = value ^ 0x55
    raw[-1] = value ^ 0xAA
    meta = metadata(shape=torch.Size([size]), dtype=torch.uint8, address=raw.data_ptr(),
                    phy_size=size, ref_count=1, fmt=fmt.BINARY,
                    shapes=[torch.Size([size])], dtypes=[torch.uint8])
    return tensor_obj(raw, meta, None)


def check_memory(torch, objects, values):
    for obj, value in zip(objects, values):
        raw = obj.raw_tensor.reshape(-1)
        if raw[0].item() != value ^ 0x55 or raw[-1].item() != value ^ 0xAA:
            raise RuntimeError("Payload endpoints differ")
        if not bool(torch.all(raw[1:-1] == value)):
            raise RuntimeError("Payload body differs")


def save_readback(path, objects, values):
    """Save direct MemoryObj byte queries; full-body validation is separate."""
    rows=[]
    for index,(obj,value) in enumerate(zip(objects,values)):
        raw=obj.raw_tensor.reshape(-1)
        rows.append(dict(object_index=index,payload_bytes=obj.get_size(),data_ptr=obj.data_ptr,
                         observed_head_u8=raw[:16].tolist(),observed_tail_u8=raw[-16:].tolist(),
                         expected_head_u8=[value^0x55]+[value]*15,
                         expected_tail_u8=[value]*15+[value^0xAA]))
    atomic_json(path,dict(query="MemoryObj.raw_tensor head/tail 16 uint8 values",
                         unix_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),objects=rows))


class MemoryBuffers:
    """Use the stock MP L1 allocation path, or an explicitly separate diagnostic."""
    def __init__(self, torch, memory_classes, kind, capacity, run_id, recorder):
        self.torch, self.memory_classes, self.kind = torch, memory_classes, kind
        self.manager = None
        self.active = []
        started = time.monotonic_ns()
        if kind == "shared":
            from lmcache.v1.distributed.config import L1MemoryManagerConfig
            from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
            config = L1MemoryManagerConfig(size_in_bytes=capacity, use_lazy=False,
                         align_bytes=4096, shm_name="lmcache_l1_pool_mech_"+run_id)
            self.manager = L1MemoryManager(config)
            descriptor = self.manager.get_l1_memory_desc()
            recorder.emit("memory_arena_initialized", started_monotonic_ns=started,
                          allocator=type(self.manager._allocator).__module__+"."+type(self.manager._allocator).__name__,
                          allocation_path="L1MemoryManager -> MixedMemoryAllocator -> SHM pinned arena -> TensorMemoryAllocator",
                          memory_kind=kind, capacity_bytes=capacity, shm_name=config.shm_name,
                          l1_memory_desc=vars(descriptor), use_lazy=False, use_hugepages=False)
        else:
            recorder.emit("memory_arena_initialized", started_monotonic_ns=started,
                          allocator="torch.empty CPU; parent_allocator=None; synthetic diagnostic only",
                          memory_kind=kind, production_l1_allocator=False)

    def allocate(self, size, values):
        if self.manager is None:
            objects = [make_memory(self.torch, self.memory_classes, size, value) for value in values]
        else:
            from lmcache.v1.distributed.api import MemoryLayoutDesc
            from lmcache.v1.distributed.error import L1Error
            # KV_2LTD is the stock manager's default; this is a legal 4D
            # synthetic layout with exactly size bytes, not a model chunk.
            layout = MemoryLayoutDesc(shapes=[self.torch.Size([2,1,1,size//2])], dtypes=[self.torch.uint8])
            error, objects = self.manager.allocate(layout, len(values))
            if error != L1Error.SUCCESS or len(objects) != len(values):
                raise RuntimeError(("Stock L1 allocation failed",str(error),len(objects)))
            for obj,value in zip(objects,values):
                raw=obj.raw_tensor.reshape(-1)
                raw.fill_(value)
                raw[0]=value^0x55
                raw[-1]=value^0xAA
        self.active.append(objects)
        return objects

    def free(self, objects):
        if self.manager is not None:
            self.manager.free(objects)
        self.active = [group for group in self.active if group is not objects]

    def close(self):
        for group in list(self.active):
            self.free(group)
        if self.manager is not None:
            self.manager.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--allow-npu", action="store_true")
    p.add_argument("--backend", choices=["memfabric", "store"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--core-source", type=Path, required=True)
    p.add_argument("--ascend-source", type=Path, required=True)
    p.add_argument("--native-so", type=Path)
    p.add_argument("--owner-receipt", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cases", default="all", help="all, smoke, or comma-separated case ids")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--rw-probe",action="store_true",help="E05 read64x18MiB with/without preceding new-key store32x18MiB")
    p.add_argument("--memory-kind", choices=["shared","torch"], default="shared",
                   help="shared uses the stock MP L1 allocator; torch is a separate synthetic diagnostic")
    a = p.parse_args()
    cases = ([dict(id="e00_1x18", objects=1, object_bytes=18*MIB, experiment="E00")]
             if a.cases == "smoke" else CASES if a.cases == "all" else
             [c for name in a.cases.split(",") for c in CASES if c["id"] == name])
    if a.rw_probe:
        cases=[dict(id="e05_rw",objects=64,object_bytes=18*MIB,experiment="E05")]
    if not a.rw_probe and a.cases not in ("all","smoke") and set(a.cases.split(",")) != {c["id"] for c in cases}:
        p.error("Unknown case id")
    if not cases or min(a.repeats, a.blocks) < 1 or a.warmups < 0:
        p.error("Invalid cases or repetition counts")
    config = json.loads(a.config.read_text(encoding="utf8"))
    if not a.run:
        print(json.dumps(dict(mode="PLAN_ONLY", cases=cases, backend=a.backend,
                              trace=a.trace, memory_kind=a.memory_kind,native_initialized=False), indent=2))
        return
    if not a.allow_npu or not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        p.error("Execution requires --allow-npu and explicit ASCEND_RT_VISIBLE_DEVICES")
    a.output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    recorder = Recorder(a.output / "events.raw.jsonl", a.backend, run_id)
    recorder.emit("process_start", argv=sys.argv, cases=cases, config=config,
                  visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                  trace=a.trace, memory_kind=a.memory_kind,cpu_affinity=sorted(os.sched_getaffinity(0)))
    sys.path[:0] = [str(a.ascend_source), str(a.core_source)]
    # Platform selection must happen only after the registered Ascend backend.
    import torch
    import torch_npu
    torch.set_num_threads(1)
    from lmcache.v1.distributed.api import ObjectKey, MemoryTags, MemorySegment, MemoryResidency
    from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
    memory_classes = (MemoryFormat, MemoryObjMetadata, TensorMemoryObj)
    initialized = time.monotonic_ns()
    owner = None
    if a.backend == "memfabric":
        from lmcache_ascend.v1.distributed.memfabric_l2_adapter import MemFabricL2Adapter
        adapter = MemFabricL2Adapter(config.get("params", config))
        if a.trace:
            attach_mf_trace(adapter, recorder)
    else:
        if not a.native_so or not a.owner_receipt:
            p.error("Store requires --native-so and --owner-receipt")
        owner = json.loads(a.owner_receipt.read_text(encoding="utf8"))
        own_mask = set(owner["visible_devices"].split(","))
        if own_mask & set(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")):
            raise ValueError("Owner and client device masks overlap")
        acl = ctypes.CDLL("libascendcl.so")
        for name, args in [("aclInit", [None]), ("aclrtSetDevice", [a.device])]:
            rc = getattr(acl, name)(*args)
            if rc:
                raise RuntimeError((name, rc))
        spec = importlib.util.spec_from_file_location("lmcache_mooncake", a.native_so)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        filtered = {k: str(v) for k,v in config.items() if k not in {"type", "num_workers", "per_op_workers", "eviction"}}
        client = native.LMCacheMooncakeClient(filtered, int(config.get("num_workers", 1)))
        if a.trace:
            client = TracedNativeClient(client, recorder)
        from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import NativeConnectorL2Adapter
        adapter = NativeConnectorL2Adapter(client, MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE))
    recorder.emit("adapter_initialized", started_monotonic_ns=initialized,
                  sdk_worker_queue="UNKNOWN" if a.backend == "store" else "traced when enabled")
    capacity=max(c["objects"]*((c["object_bytes"]+4095)//4096*4096) for c in cases)+32*MIB
    if a.rw_probe:capacity=96*18*MIB+32*MIB
    buffers=MemoryBuffers(torch,memory_classes,a.memory_kind,capacity,run_id,recorder)
    recorder.emit("loaded_python_modules", paths={name:getattr(sys.modules.get(name),"__file__",None)
                  for name in ["torch","torch_npu","lmcache","lmcache_ascend",
                   "lmcache.v1.distributed.memory_manager.l1_memory_manager",
                   "lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter",
                   "lmcache_ascend.v1.distributed.memfabric_l2_adapter"]})
    (a.output/"process_maps.raw.txt").write_bytes(Path("/proc/self/maps").read_bytes())
    try:
        if a.rw_probe:
            return rw_probe(a,adapter,buffers,torch,ObjectKey,recorder,owner,config,run_id)
        for case in cases:
            folder = a.output / case["id"]
            folder.mkdir()
            count, size = case["objects"], case["object_bytes"]
            keys = [ObjectKey(bytes.fromhex(run_id)+i.to_bytes(16,"big"), "l2-mechanism-"+case["id"], 0) for i in range(count)]
            values = [(i % 200)+1 for i in range(count)]
            from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import _object_key_to_string
            entries = [dict(key=_object_key_to_string(k), size=size, value=v) for k,v in zip(keys, values)]
            recorder.emit("case_begin", **case)
            if a.backend == "store" and case["experiment"] != "E00":
                seed = owner_command(owner, "seed", entries, folder, "owner_seed", a.timeout)
                verify_placement(seed, owner, config["local_hostname"])
            else:
                start = time.monotonic_ns()
                sources = buffers.allocate(size,values)
                recorder.emit("source_allocated_pretouched", started_monotonic_ns=start,
                              key_count=count, payload_bytes=count*size)
                result, times = submit_wait(adapter, "store", keys, sources, recorder, a.timeout)
                if not result.is_successful():
                    raise RuntimeError("Actual adapter store failed")
                buffers.free(sources)
                del sources
            if owner:
                placement = owner_command(owner, "query", entries, folder, "placement_before", a.timeout)
                verify_placement(placement, owner, config["local_hostname"])
            start = time.monotonic_ns()
            targets = buffers.allocate(size,[0]*count)
            recorder.emit("target_allocated_pretouched", started_monotonic_ns=start,
                          key_count=count, payload_bytes=count*size,
                          addresses=[x.data_ptr for x in targets])
            atomic_json(folder / "adapter_before.raw.json", adapter.report_status())
            with (folder / "samples.raw.jsonl").open("x", encoding="utf8", buffering=1) as samples:
                for block in range(a.blocks):
                    for iteration in range(a.warmups+a.repeats):
                        phase = "warmup" if iteration<a.warmups else "timed"
                        for obj in targets:
                            obj.raw_tensor.zero_()
                        bits, lookup = submit_wait(adapter, "lookup", keys, None, recorder, a.timeout)
                        if not all(bitmap_values(bits,count)):
                            adapter.submit_unlock(keys)
                            raise RuntimeError("Missing object during lookup")
                        try:
                            bits, load = submit_wait(adapter, "load", keys, targets, recorder, a.timeout)
                            if not all(bitmap_values(bits,count)):
                                raise RuntimeError("Actual adapter load incomplete")
                        finally:
                            adapter.submit_unlock(keys)
                        if block==0 and iteration==0:
                            save_readback(folder/"readback_first_load.raw.json",targets,values)
                        check_started = time.monotonic_ns()
                        check_memory(torch, targets, values)
                        row = dict(case=case["id"], block=block, iteration=iteration, phase=phase,
                                   objects=count, object_bytes=size, lookup=lookup, load=load,
                                   full_content_checked=True, validation_ns=time.monotonic_ns()-check_started,
                                   successful_payload_bytes=count*size,
                                   sdk_transport_bytes="UNKNOWN" if owner else "see adapter status and BM trace",
                                   unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
                        samples.write(json.dumps(row)+"\n")
                        recorder.emit("sample_completed", **row)
            if a.backend == "memfabric":
                # submit_unlock is fire-and-forget. This owner-queue barrier is
                # outside all samples and waits rather than editing pin state.
                adapter._executor.submit(lambda: None).result(timeout=a.timeout)
            atomic_json(folder / "adapter_after.raw.json", adapter.report_status())
            if owner:
                placement = owner_command(owner, "query", entries, folder, "placement_after", a.timeout)
                verify_placement(placement, owner, config["local_hostname"])
            adapter.delete(keys)
            # MF unlock/delete share the same owner queue. A completed lookup
            # orders both before the deletion check without direct state edits.
            bits, _ = submit_wait(adapter, "lookup", keys, None, recorder, a.timeout)
            if any(bitmap_values(bits,count)):
                raise RuntimeError("Objects remain visible after delete")
            buffers.free(targets)
            del targets
            recorder.emit("case_finished", **case)
    finally:
        recorder.emit("adapter_close_enter")
        adapter.close()
        recorder.emit("adapter_close_return")
        buffers.close()
        recorder.emit("memory_arena_close_return")
    recorder.emit("process_main_return", sdk_transport="require native logs; no inferred HCCS bytes")


if __name__ == "__main__":
    main()
