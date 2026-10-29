#!/usr/bin/env python3
"""Independent 4 GiB e9 Store owner; native initialization requires --allow-npu."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def describe_replica(replica):
    result = {"status": str(replica.status), "is_memory": replica.is_memory_replica()}
    if result["is_memory"]:
        desc = replica.get_memory_descriptor().buffer_descriptor
        result.update(size=int(desc.size), buffer_address=int(desc.buffer_address),
                      transport_endpoint=desc.transport_endpoint)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-npu", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--segment-bytes", type=int, default=4 * 1024**3)
    p.add_argument("--local-buffer-bytes", type=int, default=64 * 1024**2)
    p.add_argument("--hostname", default="127.0.0.1:28505")
    p.add_argument("--metadata", default="http://127.0.0.1:28502/metadata")
    p.add_argument("--master", default="127.0.0.1:28501")
    p.add_argument("--receipt", type=Path)
    p.add_argument("--query-dir", type=Path)
    a = p.parse_args()
    if a.segment_bytes != 4 * 1024**3:
        p.error("This profile requires the independent 4 GiB owner")
    if not a.allow_npu:
        print(json.dumps({"mode": "PLAN_ONLY", "owner_endpoint": a.hostname,
                          "global_segment_size": a.segment_bytes,
                          "native_initialized": False}, indent=2))
        return
    if not a.receipt or not a.query_dir:
        p.error("Execution requires --receipt and --query-dir")
    if a.receipt.exists():
        p.error("Refuse an existing owner receipt; use a fresh run directory")
    a.query_dir.mkdir(parents=True, exist_ok=False)

    import ctypes
    import importlib.metadata
    acl = ctypes.CDLL("libascendcl.so")
    for name, args in (("aclInit", [None]), ("aclrtSetDevice", [a.device])):
        rc = getattr(acl, name)(*args)
        if rc:
            raise RuntimeError((name, rc))
    import mooncake.store as sdk
    # Fixed e9 store_py.cpp:1836 exposes MooncakeDistributedStore, not a guessed alias.
    store = sdk.MooncakeDistributedStore()
    rc = store.setup(a.hostname, a.metadata, a.segment_bytes, a.local_buffer_bytes,
                     "ascend", "", a.master)
    if rc:
        raise RuntimeError(("store.setup", rc))
    receipt = {
        "pid": os.getpid(), "proc_start_ticks": Path("/proc/self/stat").read_text().split()[21],
        "created_unix": time.time(), "hostname": a.hostname, "metadata": a.metadata,
        "master": a.master, "protocol": "ascend", "global_segment_size": a.segment_bytes,
        "local_buffer_size": a.local_buffer_bytes,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": a.device, "store_module": sdk.__file__,
        "store_class": "mooncake.store.MooncakeDistributedStore",
        "sdk_distribution": "mooncake-transfer-engine-npu",
        "sdk_version": importlib.metadata.version("mooncake-transfer-engine-npu"),
        "query_dir": str(a.query_dir.resolve()),
    }
    if receipt["sdk_version"] != "0.3.11.post1":
        raise RuntimeError(("Unexpected existing SDK version", receipt["sdk_version"]))
    write_json(a.receipt, receipt)
    print("OWNER_READY", json.dumps(receipt), flush=True)
    stop = False

    def stopped(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        while not stop:
            for request_path in a.query_dir.glob("*.request.json"):
                response_path = request_path.with_name(request_path.name.replace(".request.json", ".reply.json"))
                if response_path.exists():
                    continue
                request = json.loads(request_path.read_text(encoding="utf-8"))
                try:
                    rows = [describe_replica(r) for r in store.get_replica_desc(request["key"])]
                    response = {"request_id": request["request_id"], "key": request["key"],
                                "owner_pid": os.getpid(), "owner_endpoint": a.hostname,
                                "queried_unix": time.time(), "replicas": rows,
                                "query_api": "MooncakeDistributedStore.get_replica_desc(key)"}
                except Exception as exc:
                    response = {"request_id": request["request_id"], "error": repr(exc)}
                write_json(response_path, response)
                print("OWNER_QUERY", json.dumps(response), flush=True)
            time.sleep(0.05)
    finally:
        rc = store.close()
        print("OWNER_CLOSE_RETURNED", rc, flush=True)
        if rc != 0:
            raise RuntimeError(("owner close", rc))
        del store


if __name__ == "__main__":
    main()
