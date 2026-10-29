#!/usr/bin/env python3
"""Small stdlib checks for the gate only; no native imports or device operations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import store_smoke as smoke


def rejects(function, *args):
    try:
        function(*args)
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid gate evidence was accepted")


def main():
    root = Path(__file__).resolve().parent
    config = smoke.read_config(root / "adapter.json")
    assert all(size > int(config["global_segment_size"]) for size in smoke.PAYLOADS)
    smoke.checked_completion([(3, True, "", [False])], 3, False)
    rejects(smoke.checked_completion, [(3, True, "", [True])], 3, False)
    owner = {"pid": 19, "hostname": "127.0.0.1:28505"}
    row = {"status": "ReplicaStatus.COMPLETE", "is_memory": True,
           "transport_endpoint": owner["hostname"], "size": smoke.PAYLOADS[0],
           "buffer_address": 12345}
    response = {"owner_pid": 19, "owner_endpoint": owner["hostname"], "replicas": [row]}
    client = config["local_hostname"]
    smoke.check_placement(response, owner, client, smoke.PAYLOADS[0])
    for patch in ({"transport_endpoint": client}, {"size": smoke.SLAB},
                  {"status": "ReplicaStatus.PROCESSING"}, {"is_memory": False}):
        rejects(smoke.check_placement, dict(response, replicas=[dict(row, **patch)]),
                owner, client, smoke.PAYLOADS[0])
    rejects(smoke.check_placement, dict(response, replicas=[row, row]),
            owner, client, smoke.PAYLOADS[0])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "master.log"
        path.write_text("Master service started on port 28501, memory_allocator=offset, rpc_port=28501\n", encoding="utf-8")
        smoke.master_allocator_evidence(path)
        path.write_text("Master service started on port 28501, memory_allocator=cachelib\n", encoding="utf-8")
        rejects(smoke.master_allocator_evidence, path)
    for script in ("store_owner.py", "store_smoke.py"):
        result = subprocess.run([sys.executable, str(root / script)], check=True,
                                capture_output=True, text=True)
        plan = json.loads(result.stdout)
        assert plan["mode"] == "PLAN_ONLY" and plan["native_initialized"] is False
    assert not any(name.split(".")[0] in {"torch", "torch_npu", "mooncake", "ctypes"}
                   for name in sys.modules)
    print("PASS_CPU_ONLY: 16 MiB profile, absent completion, real descriptor constraints, offset gate, native-free plans")


if __name__ == "__main__":
    main()
