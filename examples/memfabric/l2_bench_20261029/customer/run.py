#!/usr/bin/env python3
"""Run seed + one timed native vLLM client against already-running services."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def drain_check(status):
    manager = status.get("storage_manager", {})
    l1 = manager.get("l1_managers") or {"_default": manager.get("l1_manager", {})}
    fields = {}
    for name, values in l1.items():
        for key in ("total_object_count", "memory_used_bytes", "read_locked_count", "write_locked_count",
                    "temporary_count", "staging_object_count", "staging_bytes"):
            fields["l1." + name + "." + key] = values.get(key)
    for controller, keys in (("store_controller", ("pending_keys_count", "in_flight_task_count", "deferred_l1_deletion_count")),
                              ("prefetch_controller", ("in_flight_request_count",))):
        for key in keys:
            fields[controller + "." + key] = manager.get(controller, {}).get(key)
    for index, adapter in enumerate(manager.get("l2_adapters", [])):
        for key in ("locked_objects", "total_pins"):
            if key in adapter:
                fields[f"l2[{index}].{key}"] = adapter[key]
    return {"fields": fields, "all_observed_zero": bool(fields) and all(value == 0 for value in fields.values())}


def snapshot(observer, folder, phase, manifest, phase_name):
    deadline = time.monotonic() + float(os.environ.get("CUSTOMER_DRAIN_SECONDS", "60"))
    expected = manifest["source_expectations"][phase_name]
    while True:
        args = SimpleNamespace(folder=folder, phase=phase, wait_seconds=0,
            expected_requests=2 if phase_name == "seed" else manifest["requests"], data_root=folder,
            require_source_alignment=False, wait_source_alignment=False,
            service_url=os.environ.get("AB_SERVICE_URL", "http://127.0.0.1:28185"),
            p_url=os.environ.get("AB_P_URL", "http://127.0.0.1:28301"),
            d_url=os.environ.get("AB_D_URL", "http://127.0.0.1:28302"),
            backend_metrics_url=os.environ.get("AB_BACKEND_METRICS_URL", "http://127.0.0.1:29185/metrics"))
        snapshot_error = None
        try:
            observer.snapshot(args)
        except SystemExit as exc:
            # The shared observer rejects an undrained after-snapshot. This
            # customer loop must keep polling until its stronger drain gate.
            snapshot_error = str(exc)
        status_path = folder / (phase + "_status.json")
        status = read(status_path) if status_path.exists() else {}
        drain = drain_check(status)
        report = read(folder / (phase + "_http.json"))
        endpoints = report.get("endpoints", {})
        http_ok = bool(endpoints) and all(value.get("http_status") == 200 and "error" not in value
                                         for value in endpoints.values())
        aligned = True
        actual = None
        if phase == "after":
            alignment = report.get("source_alignment") or {}
            actual = {role: {key: alignment.get(role, {}).get("sources", {}).get(key, {}).get("delta")
                            for key in expected[role]} for role in ("p", "d")}
            aligned = actual == expected
        dump(folder / (phase + "_customer_drain.json"), {"drain": drain,
             "expected_sources": expected if phase == "after" else None, "actual_sources": actual,
             "source_expectations_met": aligned if phase == "after" else None,
             "http_endpoints_ok": http_ok, "snapshot_error": snapshot_error})
        if http_ok and drain["all_observed_zero"] and aligned:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("L1/protection drain or source fence not satisfied; see customer_drain.json")
        time.sleep(.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--arm", choices=["store", "memfabric"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--phase", choices=["seed", "measure", "both"], default="both")
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in a.tag):
        p.error("tag must use letters, digits, _ or -")
    manifest = read(a.data_root / "manifest.json")
    if a.requests != manifest["requests"]:
        p.error("requests must match the token-verified manifest")
    observer_path = Path(__file__).resolve().parent.parent / "observe.py"
    spec = importlib.util.spec_from_file_location("customer_observer", observer_path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    phases = ("seed", "timed") if a.phase == "both" else ("seed" if a.phase == "seed" else "timed",)
    for phase in phases:
        count = 2 if phase == "seed" else a.requests
        output = 1 if phase == "seed" else manifest["output_tokens"]
        folder = a.run_root / (a.tag + "_" + a.arm) / manifest["case_id"] / phase
        command = [os.environ.get("VLLM_BIN", "vllm"), "bench", "serve", "--backend", "openai",
            "--base-url", os.environ.get("AB_BASE_URL", "http://127.0.0.1:28300"), "--endpoint", "/v1/completions",
            "--model", os.environ.get("AB_MODEL", "mf-l2-ab"), "--served-model-name", os.environ.get("AB_MODEL", "mf-l2-ab"),
            "--tokenizer", os.environ.get("AB_TOKENIZER", manifest["model_directory"]),
            "--dataset-name", "custom", "--dataset-path", str(a.data_root / (phase + ".jsonl")),
            "--custom-output-len", str(output), "--num-prompts", str(count), "--max-concurrency", "1", "--request-rate", "inf",
            "--temperature", "0", "--num-warmups", "0", "--ready-check-timeout-sec", "0", "--ignore-eos",
            "--disable-shuffle", "--no-oversample", "--seed", "0", "--skip-chat-template",
            "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,95", "--save-result", "--save-detailed",
            "--result-dir", str(folder), "--result-filename", "benchmark.json"]
        print(shlex.join(command), flush=True)
        if a.dry_run:
            continue
        if folder.exists():
            raise RuntimeError("Refusing existing result directory: " + str(folder))
        folder.mkdir(parents=True)
        dump(folder / "manifest.json", manifest)
        (folder / "requests.jsonl").write_text((a.data_root / (phase + ".jsonl")).read_text(encoding="utf-8"), encoding="utf-8")
        (folder / "command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
        snapshot(observer, folder, "before", manifest, phase)
        with (folder / "benchmark.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            dump(folder / "client_process.json", {"pid": child.pid, "started_unix": time.time()})
            result = child.wait()
        after_error = None
        try:
            snapshot(observer, folder, "after", manifest, phase)
        except (RuntimeError, SystemExit) as exc:
            after_error = str(exc)
        with (folder / "validation.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            observer.finish(SimpleNamespace(folder=folder, data_root=folder, requests=count,
                output_tokens=output, exit_code=result, snapshot_exit=1 if after_error else 0,
                arm=a.arm, phase=phase, concurrency=1, round=1))
        native = read(folder / "benchmark.json")
        checks = {"input_lengths": native.get("input_lens") == [manifest["input_tokens"]] * count,
                  "generated_texts_saved": isinstance(native.get("generated_texts"), list) and len(native["generated_texts"]) == count,
                  "drain_and_source_fence": after_error is None}
        dump(folder / "customer_validation.json", {"checks": checks, "passed": all(checks.values()),
             "after_error": after_error, "expected_source_deltas": manifest["source_expectations"][phase],
             "exit_gate": "NOT_ESTABLISHED_BY_CLIENT_COUNTS"})
        if not all(checks.values()):
            raise RuntimeError("Customer count/source validation failed")
        print(json.dumps({"phase": phase, "folder": str(folder), "checks_passed": True}), flush=True)


if __name__ == "__main__":
    main()
