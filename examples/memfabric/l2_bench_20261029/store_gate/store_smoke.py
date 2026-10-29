#!/usr/bin/env python3
"""Plan by default; fixed e9 native Store content/placement/exit gate, no timing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

MIB = 1024**2
SLAB = 16 * MIB
PAYLOADS = (18 * MIB, 36 * MIB)
LOCAL_KEYS = {"type", "num_workers", "per_op_workers", "eviction"}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("type") != "mooncake_store":
        raise ValueError("Expected the existing mooncake_store adapter")
    if int(config["global_segment_size"]) != SLAB:
        raise ValueError("e9 requires exactly one 16 MiB client slab for this profile")
    if int(config["local_buffer_size"]) < max(PAYLOADS):
        raise ValueError("Client local transfer buffer is too small")
    if config.get("protocol") != "tcp":
        raise ValueError("Keep the fixed e9 ConfigDict; actual native transport needs logs")
    if not config.get("metadata_server", "").endswith("/metadata"):
        raise ValueError("Expected HTTP metadata URL ending in /metadata")
    return config


def checked_completion(rows, request_id, expected_key=None):
    if len(rows) != 1 or rows[0][0] != request_id:
        raise RuntimeError(f"Unexpected native completions: {rows!r}")
    row = rows[0]
    if row[1] is not True or row[2]:
        raise RuntimeError(f"Native operation failed: {row!r}")
    if expected_key is not None and (row[3] is None or list(row[3]) != [expected_key]):
        raise RuntimeError(f"Per-key operation failed: {row!r}")
    return row


def check_owner_alive(owner):
    path = Path(f"/proc/{int(owner['pid'])}/stat")
    if path.read_text().split()[21] != owner["proc_start_ticks"]:
        raise RuntimeError("Owner PID was reused")


def check_placement(response, owner, client_endpoint, size):
    if response.get("error"):
        raise RuntimeError(response["error"])
    if response.get("owner_pid") != owner["pid"] or response.get("owner_endpoint") != owner["hostname"]:
        raise RuntimeError("Placement response came from a different owner")
    rows = response.get("replicas", [])
    if len(rows) != 1:
        raise RuntimeError(f"Expected one unsplit replica: {rows!r}")
    row = rows[0]
    if not row.get("is_memory") or row.get("status") != "ReplicaStatus.COMPLETE":
        raise RuntimeError(f"Expected one complete memory replica: {row!r}")
    if row.get("transport_endpoint") != owner["hostname"] or row["transport_endpoint"] == client_endpoint:
        raise RuntimeError(f"Object not placed at the independent owner: {row!r}")
    if int(row.get("size", 0)) < size or int(row.get("buffer_address", 0)) <= 0:
        raise RuntimeError(f"Invalid complete-object descriptor: {row!r}")
    return response


def owner_query(owner, key, client_endpoint, size):
    request_id = uuid.uuid4().hex
    root = Path(owner["query_dir"])
    request_path = root / f"{request_id}.request.json"
    reply_path = root / f"{request_id}.reply.json"
    write_json(request_path, {"request_id": request_id, "key": key})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        check_owner_alive(owner)
        if reply_path.exists():
            response = json.loads(reply_path.read_text(encoding="utf-8"))
            if response.get("request_id") != request_id or response.get("key") != key:
                raise RuntimeError("Unexpected placement response key/id")
            return check_placement(response, owner, client_endpoint, size)
        time.sleep(0.05)
    raise TimeoutError(f"Independent owner query timed out: {key}")


def master_allocator_evidence(path):
    lines = [line for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
             if "Master service started on port" in line]
    if not lines or not re.search(r"\bmemory_allocator=offset(?:,|\s|$)", lines[-1]):
        raise ValueError("Fresh master startup log must confirm memory_allocator=offset")
    return lines[-1]


def worker(a, config):
    # All native imports and device initialization stay behind --allow-npu.
    import ctypes
    import select

    owner = json.loads(a.owner_receipt.read_text(encoding="utf-8"))
    check_owner_alive(owner)
    if owner["master"] != config["master_server_addr"] or owner["metadata"] != config["metadata_server"]:
        raise ValueError("Owner and client use different master/metadata endpoints")
    if owner["protocol"] != "ascend" or int(owner["global_segment_size"]) != 4 * 1024**3:
        raise ValueError("Expected the independent 4 GiB Ascend owner")
    if owner["hostname"] == config["local_hostname"]:
        raise ValueError("Owner and client endpoints collide")
    client_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not client_devices or not owner.get("visible_devices"):
        raise ValueError("Both owner/client require explicit device masks")
    if set(client_devices.split(",")) & set(owner["visible_devices"].split(",")):
        raise ValueError("Owner and client device masks overlap")
    master_line = master_allocator_evidence(a.master_log)
    acl = ctypes.CDLL("libascendcl.so")
    for name, args in (("aclInit", [None]), ("aclrtSetDevice", [a.device])):
        rc = getattr(acl, name)(*args)
        if rc:
            raise RuntimeError((name, rc))
    spec = importlib.util.spec_from_file_location("lmcache_mooncake", a.native_so)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native_config = {k: str(v) for k, v in config.items() if k not in LOCAL_KEYS}
    client = native.LMCacheMooncakeClient(native_config, config["num_workers"])
    poll = select.poll()
    poll.register(client.event_fd(), select.POLLIN)
    events, placements = [], []

    def wait(request_id, operation, expected_key=None):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not poll.poll(200):
                continue
            os.read(client.event_fd(), 8)
            rows = client.drain_completions()
            if rows:
                row = checked_completion(rows, request_id, expected_key)
                events.append({"request_id": request_id, "operation": operation, "result": repr(row)})
                print("COMPLETION", operation, row, flush=True)
                return
        raise TimeoutError((operation, request_id))

    run_id = uuid.uuid4().hex
    try:
        for size in PAYLOADS:
            key = f"l2_remote_{run_id}_{size}"
            expected = (bytes(range(251)) * (size // 251 + 1))[:size]
            buf = bytearray(expected)
            view = memoryview(buf)
            wait(client.submit_batch_set([key], [view]), "SET")
            wait(client.submit_batch_exists([key]), "EXISTS", True)
            placement = owner_query(owner, key, config["local_hostname"], size)
            placements.append(placement)
            write_json(a.output / "placements.json", placements)
            buf[:] = bytes(size)
            wait(client.submit_batch_get([key], [view]), "GET", True)
            if buf != expected:
                raise RuntimeError(f"Full payload mismatch: {size}")
            wait(client.submit_batch_delete([key]), "DELETE", True)
            wait(client.submit_batch_exists([key]), "EXISTS_AFTER_DELETE", False)
            del view, buf, expected
    finally:
        client.close()
        del client
        print("NATIVE_CLOSE_RETURNED", flush=True)
    check_owner_alive(owner)
    result = {
        "content_passed": True, "placement_passed": True, "config": native_config,
        "payload_bytes": list(PAYLOADS), "events": events, "placements": placements,
        "owner": owner, "client": {"pid": os.getpid(), "endpoint": config["local_hostname"],
                                   "visible_devices": client_devices, "logical_device": a.device},
        "native_so": str(Path(a.native_so).resolve()), "master_startup_line": master_line,
        "actual_transport": "NOT_VERIFIED; correlate native/HCCL logs with these keys/endpoints",
    }
    write_json(a.output / "content.json", result)
    print("NATIVE_CONTENT_PLACEMENT_PASS", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("adapter.json"))
    p.add_argument("--allow-npu", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--native-so")
    p.add_argument("--owner-receipt", type=Path)
    p.add_argument("--master-log", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    config = read_config(a.config)
    if not a.allow_npu:
        print(json.dumps({"mode": "PLAN_ONLY", "config": config, "payload_bytes": PAYLOADS,
                          "owner_segment_bytes": 4 * 1024**3, "master_allocator_required": "offset",
                          "placement_query": "independent owner: MooncakeDistributedStore.get_replica_desc(key)",
                          "native_initialized": False}, indent=2))
        return
    if not all((a.native_so, a.owner_receipt, a.master_log, a.output)):
        p.error("Execution requires --native-so, --owner-receipt, --master-log and a fresh --output")
    if a.worker:
        return worker(a, config)
    a.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    with (a.output / "native.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = child.wait(timeout=240)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    content = json.loads((a.output / "content.json").read_text(encoding="utf-8")) if (a.output / "content.json").exists() else {}
    passed = code == 0 and not timed_out and content.get("content_passed") is True and content.get("placement_passed") is True
    outcome = {"exit_code": code, "timed_out": timed_out, "native_content_placement_and_exit_passed": passed,
               "command": command, "transport_passed": False, "inference_passed": False}
    write_json(a.output / "outcome.json", outcome)
    print(json.dumps(outcome, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
