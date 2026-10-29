#!/usr/bin/env python3
"""Independent real Store owner supporting explicit seed and replica queries."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n", encoding="utf8")
    temporary.replace(path)


def replicas(store, key):
    result=[]
    for r in store.get_replica_desc(key):
        value=dict(status=str(r.status), is_memory=r.is_memory_replica())
        if value["is_memory"]:
            d=r.get_memory_descriptor().buffer_descriptor
            value.update(size=int(d.size), buffer_address=int(d.buffer_address), transport_endpoint=d.transport_endpoint)
        result.append(value)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--allow-npu", action="store_true")
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--hostname",required=True)
    p.add_argument("--metadata",required=True)
    p.add_argument("--master",required=True)
    p.add_argument("--segment-bytes",type=int,default=4*1024**3)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--query-dir",type=Path,required=True)
    a=p.parse_args()
    if not a.run:
        print(json.dumps(dict(mode="PLAN_ONLY", native_initialized=False)))
        return
    if not a.allow_npu or not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        p.error("Execution requires explicit --allow-npu and device mask")
    if a.receipt.exists():
        raise FileExistsError(a.receipt)
    a.query_dir.mkdir(parents=True,exist_ok=False)
    acl=ctypes.CDLL("libascendcl.so")
    for name,args in [("aclInit",[None]),("aclrtSetDevice",[a.device])]:
        rc=getattr(acl,name)(*args)
        if rc: raise RuntimeError((name,rc))
    import mooncake.store as sdk
    store=sdk.MooncakeDistributedStore()
    rc=store.setup(a.hostname,a.metadata,a.segment_bytes,64*1024**2,"ascend","",a.master)
    if rc: raise RuntimeError(("setup",rc))
    receipt=dict(pid=os.getpid(), proc_start_ticks=Path("/proc/self/stat").read_text().split()[21],
                 created_unix=time.time(),hostname=a.hostname,metadata=a.metadata,master=a.master,
                 protocol="ascend",global_segment_size=a.segment_bytes,query_dir=str(a.query_dir.resolve()),
                 visible_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],store_module=sdk.__file__)
    atomic_json(a.receipt,receipt)
    print("OWNER_READY",json.dumps(receipt),flush=True)
    stop=False
    def stopped(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopped)
    signal.signal(signal.SIGINT,stopped)
    try:
        while not stop:
            for request_path in a.query_dir.glob("*.request.json"):
                reply=request_path.with_name(request_path.name.replace(".request.json",".reply.json"))
                if reply.exists():continue
                request=json.loads(request_path.read_text(encoding="utf8"))
                result=dict(request_id=request["request_id"],owner_pid=os.getpid(),owner_endpoint=a.hostname,
                            started_unix_ns=time.time_ns(),objects=[])
                try:
                    for e in request["entries"]:
                        if request["op"]=="seed":
                            data=bytearray([e["value"]])*int(e["size"])
                            data[0]=e["value"]^0x55
                            data[-1]=e["value"]^0xAA
                            config=sdk.ReplicateConfig()
                            config.replica_num=1
                            config.preferred_segment=a.hostname
                            rc=store.put(e["key"],data,config)
                            # This SDK preference can fall back. The client
                            # still validates every actual replica descriptor.
                            print("OWNER_PUT_RETURN",json.dumps(dict(key=e["key"],size=e["size"],returncode=rc,
                                  preferred_segment=a.hostname),default=str),flush=True)
                            if rc!=0:raise RuntimeError(("put",rc,e["key"]))
                            del data
                        elif request["op"]!="query":
                            raise ValueError("Unknown owner operation")
                        result["objects"].append(dict(key=e["key"],size=e["size"],replicas=replicas(store,e["key"])))
                except Exception as error:
                    result["error"]=repr(error)
                result["returned_unix_ns"]=time.time_ns()
                atomic_json(reply,result)
                print("OWNER_QUERY",json.dumps(result,default=str),flush=True)
            time.sleep(.01)
    finally:
        rc=store.close()
        print("OWNER_CLOSE_RETURNED",rc,flush=True)
        if rc!=0:raise RuntimeError(("owner close",rc))


if __name__=="__main__":
    main()
