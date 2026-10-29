#!/usr/bin/env python3
"""E01b: no-model prefetch through the real StorageManager and controllers.

Requires an external process supervisor and independent pool owner/member.
Never edits core/controller state or consumes a controller-owned adapter fd.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from l2_api_bench import (MIB, Recorder, TracedNativeClient, atomic_json,
                          attach_mf_trace, check_memory, owner_command, verify_placement,
                          save_readback)


def adapter_config(backend, raw):
    if backend == "store":
        return raw
    if raw.get("type") == "plugin":
        return raw
    return dict(type="plugin", module_path="lmcache_ascend.v1.distributed.memfabric_l2_adapter",
                class_name="MemFabricL2Adapter",
                adapter_params=raw.get("params", raw))


def observe(instance, method, recorder, label):
    """Wrap one existing call; return its original result and exception unchanged."""
    actual = getattr(instance, method)
    def call(*args, **kwargs):
        fields = dict(method=method)
        for value in args:
            if hasattr(value, "request_id"):
                fields["prefetch_request_id"] = value.request_id
            if isinstance(value, list):
                fields["key_count"] = len(value)
        if method == "_start_lookup_phase" and args:
            fields["prefetch_request_id"] = args[0]
        if method == "_publish_result" and len(args) > 1:
            fields.update(l1_hit_count=args[1].l1_hit_count, l2_hit_count=args[1].l2_hit_count)
        if "tag" in kwargs:
            fields["write_tag"] = kwargs["tag"]
        started = time.monotonic_ns()
        recorder.emit(label+"_enter", **fields)
        try:
            result = actual(*args, **kwargs)
        except BaseException as error:
            recorder.emit(label+"_return", started_monotonic_ns=started, error=repr(error), **fields)
            raise
        if isinstance(result,bool):fields["returned_value"]=result
        recorder.emit(label+"_return", started_monotonic_ns=started, error=None, **fields)
        return result
    setattr(instance, method, call)


def drained(manager):
    raw = manager.report_status()
    l1 = raw["l1_manager"]
    store = raw["store_controller"]
    prefetch = raw["prefetch_controller"]
    okay = (all(l1[k] == 0 for k in ["write_locked_count","read_locked_count",
                 "temporary_count","staging_object_count"])
            and all(store[k] == 0 for k in ["pending_keys_count","in_flight_task_count",
                                           "deferred_l1_deletion_count"])
            and all(prefetch[k] == 0 for k in ["in_flight_request_count","submission_queue_size",
                                              "pending_queue_size","completed_results_count"]))
    protections=[]
    for _descriptor,adapter in manager.l2_adapters():
        if hasattr(adapter,"_locked_keys"):
            with adapter._lock:
                count=sum(adapter._locked_keys.values())
            protections.append(dict(query="NativeConnectorL2Adapter._locked_keys",read_locks=count))
            okay=okay and count==0
        elif hasattr(adapter,"report_status"):
            status=adapter.report_status()
            protections.append(dict(query="L2Adapter.report_status",total_pins=status.get("total_pins")))
            if "total_pins" in status:okay=okay and status["total_pins"]==0
    raw["diagnostic_protection_queries"]=protections
    return okay, raw


def wait_drained(manager, output, name, timeout):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        okay,raw=drained(manager)
        if okay:
            atomic_json(output/(name+".status.raw.json"),raw)
            return raw
        time.sleep(.01)
    atomic_json(output/(name+".timeout.raw.json"),manager.report_status())
    raise TimeoutError("Controller/locks did not drain: "+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",action="store_true")
    p.add_argument("--allow-npu",action="store_true")
    p.add_argument("--backend",choices=["memfabric","store"],required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--core-source",type=Path,required=True)
    p.add_argument("--ascend-source",type=Path,required=True)
    p.add_argument("--native-so",type=Path)
    p.add_argument("--owner-receipt",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--objects",type=int,default=128)
    p.add_argument("--object-bytes",type=int,default=18*MIB)
    p.add_argument("--warmups",type=int,default=1)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--l1-bytes",type=int,default=3*1024**3)
    p.add_argument("--timeout",type=float,default=120)
    p.add_argument("--trace",action="store_true")
    p.add_argument("--release-native-after-close",action="store_true",help="Diagnostic: release this stopped adapter SDK before device runtime teardown")
    a=p.parse_args()
    if a.release_native_after_close and a.backend!="store":
        p.error("--release-native-after-close is a Store-only diagnostic")
    if not 0<a.objects<=128 or a.object_bytes<=0 or a.object_bytes%4096:
        p.error("Require 1..128 objects with a positive 4KiB-multiple payload")
    if a.objects*a.object_bytes>=a.l1_bytes or a.repeats<1 or a.warmups<0:
        p.error("Require spare L1 capacity and valid repeat counts")
    raw_config=json.loads(a.config.read_text(encoding="utf8"))
    if not a.run:
        print(json.dumps(dict(mode="PLAN_ONLY",experiment="E01b",backend=a.backend,
                   objects=a.objects,object_bytes=a.object_bytes,l1_bytes=a.l1_bytes,
                   allocator="stock L1Manager",native_initialized=False),indent=2))
        return
    if not a.allow_npu or not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        p.error("Execution requires --allow-npu and an explicit visible-device mask")
    if a.backend=="store" and (not a.native_so or not a.owner_receipt):
        p.error("Store requires native-so and independent owner-receipt")
    a.output.mkdir(parents=True,exist_ok=False)
    run_id=uuid.uuid4().hex
    recorder=Recorder(a.output/"controller_events.raw.jsonl",a.backend,run_id)
    recorder.emit("process_start",argv=sys.argv,visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                  cpu_affinity=sorted(os.sched_getaffinity(0)),trace=a.trace)
    sys.path[:0]=[str(a.ascend_source),str(a.core_source)]
    import torch
    import torch_npu
    torch.set_num_threads(1)
    # Match the service transport device before stock L1/native construction.
    torch.npu.set_device(a.device)
    if a.backend=="store":
        spec=importlib.util.spec_from_file_location("lmcache.lmcache_mooncake",a.native_so)
        native=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sys.modules["lmcache.lmcache_mooncake"]=native
    from lmcache.v1.distributed.api import ObjectKey,MemoryLayoutDesc,GroupedObjectKeys,PrefetchTaskSpec
    from lmcache.v1.distributed.config import (L1MemoryManagerConfig,L1ManagerConfig,
                                             EvictionConfig,StorageManagerConfig)
    from lmcache.v1.distributed.l2_adapters.config import parse_args_to_l2_adapters_config
    from lmcache.v1.distributed.storage_manager import StorageManager
    config=StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(L1MemoryManagerConfig(a.l1_bytes,False,
                         align_bytes=4096,shm_name="lmcache_l1_pool_controller_"+run_id)),
        eviction_config=EvictionConfig("LRU"),
        l2_adapter_config=parse_args_to_l2_adapters_config(argparse.Namespace(
            l2_adapter=[json.dumps(adapter_config(a.backend,raw_config))])),
        store_policy="default",prefetch_policy="default",prefetch_inflight_watermark=.5)
    manager=StorageManager(config)
    adapter=manager.l2_adapters()[0][1]
    recorder.emit("storage_manager_initialized",l1_memory_desc=vars(manager.l1_memory_desc),
                  allocator="stock L1Manager -> L1MemoryManager -> MixedMemoryAllocator SHM pinned arena")
    if a.trace:
        if a.backend=="memfabric":
            attach_mf_trace(adapter,recorder)
        else:
            adapter._client=TracedNativeClient(adapter._client,recorder)
        controller=manager._prefetch_controller
        for name in ["_can_admit_new_prefetch","_start_lookup_phase","_transition_to_load_phase",
                     "_submit_load_tasks","_poll_load_results","_publish_result"]:
            observe(controller,name,recorder,"controller"+name)
        for name in ["reserve_write","finish_write_and_reserve_read","finish_read","delete"]:
            observe(manager._l1_manager,name,recorder,"l1"+name)
    # Observe the real store-controller completion. Never submit an unrelated
    # adapter task whose eventfd/result would compete with this controller.
    seed_lock=threading.Lock()
    seed_seen=set()
    seed_errors=[]
    original_finalize=manager._store_controller._finalize_store
    def finalized(task_key,task):
        result=original_finalize(task_key,task)
        recorder.emit("seed_store_controller_finalized",task_key=task_key,key_count=len(task.keys),
                      success=task.l2_store_result,adapter_reported_bytes=task.l2_bytes_transferred)
        with seed_lock:
            if task.l2_store_result:seed_seen.update(task.keys)
            else:seed_errors.append(task_key)
        return result
    manager._store_controller._finalize_store=finalized
    keys=[ObjectKey(bytes.fromhex(run_id)+i.to_bytes(16,"big"),"l2-controller",0) for i in range(a.objects)]
    values=[i%200+1 for i in range(a.objects)]
    layout=MemoryLayoutDesc([torch.Size([2,1,1,a.object_bytes//2])],[torch.uint8])
    owner=json.loads(a.owner_receipt.read_text(encoding="utf8")) if a.owner_receipt else None
    try:
        reserved=manager.reserve_write(keys,layout)
        if len(reserved)!=len(keys):raise RuntimeError("Could not reserve all seed objects")
        for key,value in zip(keys,values):
            raw=reserved[key].raw_tensor.reshape(-1)
            raw.fill_(value);raw[0]=value^0x55;raw[-1]=value^0xAA
        del raw
        manager.finish_write(keys)
        del reserved
        end=time.monotonic()+a.timeout
        while time.monotonic()<end:
            with seed_lock:
                if seed_errors:raise RuntimeError(("Seed store failed",seed_errors))
                if len(seed_seen)==len(keys):break
            time.sleep(.01)
        else:raise TimeoutError("Actual store controller did not finalize the complete seed")
        wait_drained(manager,a.output,"seed_drained",a.timeout)
        if owner:
            from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import _object_key_to_string
            entries=[dict(key=_object_key_to_string(k),size=a.object_bytes) for k in keys]
            verify_placement(owner_command(owner,"query",entries,a.output,"placement_before",a.timeout),
                             owner,raw_config["local_hostname"])
        with (a.output/"controller_samples.raw.jsonl").open("x",encoding="utf8",buffering=1) as samples:
            for iteration in range(a.warmups+a.repeats):
                deleted,skipped=manager.delete_l1_keys(keys,force=False)
                if skipped:raise RuntimeError(("Normal L1 delete skipped locked objects",skipped))
                before=wait_drained(manager,a.output,f"sample_{iteration:03d}_before",a.timeout)
                if before["l1_manager"]["total_object_count"]!=0:
                    raise RuntimeError("L1 is not empty before prefetch")
                # A freed SHM allocation can reuse the seed's old address.
                # Prepare only this wholly empty, drained private arena before
                # timing so stale seed bytes cannot satisfy the content check.
                prepared=time.monotonic_ns()
                arena=manager._l1_manager._memory_manager._allocator.buffer
                arena.zero_()
                recorder.emit("empty_private_l1_arena_pretouched",started_monotonic_ns=prepared,
                              bytes=arena.numel(),value=0)
                spec=PrefetchTaskSpec([GroupedObjectKeys(keys,0,layout)],num_kv_readers=1,fetching_policy="prefix")
                started=time.monotonic_ns()
                handle=manager.submit_prefetch_task(spec,external_request_id=f"e01b-{run_id}-{iteration}")
                if not manager.wait_prefetch_status(handle,a.timeout):
                    raise TimeoutError("Controller prefetch did not complete")
                result=manager.query_prefetch_status(handle)
                ended=time.monotonic_ns()
                if result is None:raise RuntimeError("Missing controller result after wait")
                row=dict(iteration=iteration,phase="warmup" if iteration<a.warmups else "timed",
                         prefetch_request_id=handle.prefetch_request_id,external_request_id=handle.external_request_id,
                         started_monotonic_ns=started,finished_monotonic_ns=ended,
                         hit_cells=[[bool(b.test(i)) for i in range(a.objects)] for b in result.hit_cells],
                         l1_hit_count=result.l1_hit_count,l2_hit_count=result.l2_hit_count)
                samples.write(json.dumps(row)+"\n")
                recorder.emit("prefetch_public_result",**row)
                if result.l1_hit_count!=0 or result.l2_hit_count!=len(keys):
                    retained=[k for i,k in enumerate(keys) if result.hit_cells[0].test(i)]
                    manager.finish_read_prefetched(retained,l1_owners=result.l1_owners)
                    raise RuntimeError("Prefetch is not complete zero-L1/full-L2")
                with manager.read_prefetched_results(keys,l1_owners=result.l1_owners) as objects:
                    if objects is None:raise RuntimeError("Prefetched objects are unavailable for read")
                    if iteration==0:
                        save_readback(a.output/"readback_first_prefetch.raw.json",objects,values)
                    check_memory(torch,objects,values)
                manager.finish_read_prefetched(keys,l1_owners=result.l1_owners)
                recorder.emit("full_content_checked_and_read_finished",prefetch_request_id=handle.prefetch_request_id)
                deleted,skipped=manager.delete_l1_keys(keys,force=False)
                if skipped:raise RuntimeError("Read locks did not release before ordinary L1 deletion")
                after=wait_drained(manager,a.output,f"sample_{iteration:03d}_after",a.timeout)
                if after["l1_manager"]["total_object_count"]!=0:
                    raise RuntimeError("L1 objects remain after release/delete")
        if owner:
            verify_placement(owner_command(owner,"query",entries,a.output,"placement_after",a.timeout),
                             owner,raw_config["local_hostname"])
        adapter.delete(keys)
        if a.backend=="memfabric":
            # Only order already-issued deletion on its existing owner queue;
            # do not consume controller-owned eventfds or query their tasks.
            adapter._executor.submit(lambda:None).result(timeout=a.timeout)
        wait_drained(manager,a.output,"final_deleted",a.timeout)
        usage=adapter.get_usage()
        atomic_json(a.output/"final_adapter_usage.raw.json",dict(
            query="L2Adapter.get_usage",total_bytes_used=usage.total_bytes_used,
            total_capacity_bytes=usage.total_capacity_bytes,
            bytes_by_cache_salt=dict(usage.bytes_by_cache_salt)))
        if usage.total_bytes_used!=0:
            raise RuntimeError("Tracked L2 bytes remain after final ordinary deletion")
        status=adapter.report_status()
        if "objects" in status and status["objects"]!=0:
            raise RuntimeError("MemFabric objects remain after final deletion")
        recorder.emit("controller_benchmark_body_return")
    finally:
        recorder.emit("storage_manager_close_enter")
        manager.close()
        recorder.emit("storage_manager_close_return")
        if a.release_native_after_close:
            # Diagnostic-only: close() joined workers, but the stopped adapter can
            # remain rooted by metrics/controllers. Release this owned SDK instance
            # while its current ACL context is still valid. No os._exit shortcut.
            import ctypes, gc
            assert adapter._stop.is_set() and not adapter._demux_thread.is_alive()
            acl=ctypes.CDLL("libascendcl.so")
            context=ctypes.c_void_p()
            rc=acl.aclrtGetCurrentContext(ctypes.byref(context))
            recorder.emit("native_release_before",context_returncode=rc,context=context.value,
                          demux_alive=adapter._demux_thread.is_alive())
            if rc or not context.value:
                raise RuntimeError("Refuse native release without a live current context")
            adapter._client=None
            collected=gc.collect()
            after=ctypes.c_void_p()
            rc_after=acl.aclrtGetCurrentContext(ctypes.byref(after))
            recorder.emit("native_release_after",collected=collected,context_returncode=rc_after,context=after.value)


if __name__=="__main__":
    main()
