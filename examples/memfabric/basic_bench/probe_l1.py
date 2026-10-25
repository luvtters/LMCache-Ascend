# SPDX-License-Identifier: Apache-2.0
"""Clear service L1, preserve warm L2, then run the existing basic benchmark.

This is a controlled diagnostic, separate from ordinary A/B runs. It never
starts or stops services and never imports device libraries.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request


CASES = ("qwen_2560_unique", "qwen_2560_prefix50", "qwen_2560_prefix90")


def check_arm(status, arm):
    adapters = status["storage_manager"]["l2_adapters"]
    if arm == "A" and adapters:
        raise ValueError("A requires no L2 adapters")
    if arm == "B" and (len(adapters) != 1 or adapters[0].get("protocol") != "SDMA"
                       or not adapters[0].get("is_healthy")):
        raise ValueError("B requires exactly one healthy SDMA adapter")
    return adapters


def l1_empty(status):
    manager = status["storage_manager"]["l1_manager"]
    return manager.get("total_object_count") == 0 and manager.get("memory_used_bytes") == 0


def stores_drained(status):
    controller = status["storage_manager"]["store_controller"]
    return controller.get("pending_keys_count") == 0 and controller.get("in_flight_task_count") == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("A", "B"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--case", choices=CASES, default="qwen_2560_prefix90")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("tag may contain only letters, digits, underscores and hyphens")
    here = Path(__file__).resolve().parent
    root = Path(os.environ.get("MF_RESULT_ROOT", str(here / "results"))).resolve()
    run_name = f"{args.tag}_{args.arm}_{args.case}"
    output = root / (run_name + "_l1_probe")
    if output.exists() or (root / run_name).exists():
        parser.error("probe or benchmark result directory already exists; use a new tag")
    data_root = Path(os.environ.get("MF_DATA_ROOT", str(here / "generated/basic/dataset")))
    if not (data_root / args.case / "custom.jsonl").is_file():
        parser.error("case data is missing; no cache clear was sent")
    bash = shutil.which("bash")
    if not bash:
        parser.error("bash is required for the existing bench.sh entrypoint")
    service = os.environ.get("MF_SERVICE_URL", "http://127.0.0.1:28080").rstrip("/")

    def request(path, body=None, timeout=10):
        data = None if body is None else json.dumps(body).encode("utf-8")
        call = urllib.request.Request(service + path, data=data,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(call, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def save(name, value):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def settled(require_empty=False):
        deadline = time.monotonic() + 30
        status = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                save("timeout_status", status)
                raise TimeoutError("Service stores or L1 did not settle within 30 seconds")
            status = request("/status", timeout=min(10, remaining))
            check_arm(status, args.arm)
            if stores_drained(status) and (not require_empty or l1_empty(status)):
                return status
            if time.monotonic() >= deadline:
                save("timeout_status", status)
                raise TimeoutError("Service stores or L1 did not settle within 30 seconds")
            time.sleep(0.2)

    initial = request("/status")
    check_arm(initial, args.arm)
    output.mkdir(parents=True, exist_ok=False)
    before = settled()
    if args.arm == "B" and before["storage_manager"]["l2_adapters"][0].get("objects", 0) <= 0:
        save("before_clear", before)
        raise ValueError("B pool has no objects; warm the selected case before this diagnostic")
    save("before_clear", before)
    save("clear_response", request("/cache/clear", {"tier": "l1", "force": False}))
    after = settled(require_empty=True)
    save("after_clear", after)
    if args.arm == "B" and after["storage_manager"]["l2_adapters"][0].get("objects", 0) <= 0:
        raise ValueError("L2 pool objects disappeared; benchmark was not started")
    environment = os.environ.copy()
    environment["MF_CACHE_STATE"] = "l1_cleared_before_run"
    command = [bash, str(here / "bench.sh"), args.arm, args.case, args.tag]
    save("probe", {"arm": args.arm, "case": args.case,
                   "cache_state_before_case": "l1_cleared_before_run",
                   "benchmark_command": command,
                   "note": "Controlled L1-clear diagnostic; not an ordinary three-case run"})
    result = subprocess.run(command, env=environment)
    save("benchmark_exit", {"returncode": result.returncode})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
