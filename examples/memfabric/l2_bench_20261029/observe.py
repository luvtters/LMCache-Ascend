#!/usr/bin/env python3
"""Read-only HTTP snapshots and native benchmark result validation/summary."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


TOKEN_SOURCES = ("local_compute", "external_kv_transfer", "local_cache_hit")


def source_counts(text):
    found = {source: None for source in TOKEN_SOURCES}
    for line in text.splitlines():
        match = re.match(r'^vllm:prompt_tokens_by_source_total\{([^}]*)\}\s+(\S+)', line)
        label = re.search(r'(?:^|,)\s*source="([^"]*)"(?:\s*,|\s*$)', match.group(1)) if match else None
        if label and label.group(1) in found:
            source = label.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                return {key: None for key in TOKEN_SOURCES}
            if not finite(value) or value < 0:
                return {key: None for key in TOKEN_SOURCES}
            found[source] = value if found[source] is None else found[source] + value
    return found


def source_alignment(folder, after_bodies, expected):
    result = {}
    for role in ("p", "d"):
        path = folder / f"before_{role}_metrics.txt"
        before = source_counts(path.read_text(encoding="utf-8")) if path.exists() else {key: None for key in TOKEN_SOURCES}
        after = source_counts(after_bodies.get(role + "_metrics", ""))
        changes = deltas(before, after)
        values = [changes[key]["delta"] for key in TOKEN_SOURCES]
        known = all(finite(value) for value in values)
        observed = sum(values) if known else None
        result[role] = {"sources": changes, "expected_prompt_tokens": expected,
                        "observed_prompt_tokens": observed,
                        "aligned": known and observed == expected,
                        "state": "known" if known else "unknown"}
    return result


def drain_state(status):
    manager = status.get("storage_manager", {})
    fields = {}
    for controller, names in (("store_controller", ("pending_keys_count", "in_flight_task_count")),
                              ("prefetch_controller", ("in_flight_request_count",))):
        values = manager.get(controller, {})
        for name in names:
            value = values.get(name)
            fields[controller + "." + name] = value if finite(value) else None
    known = [value for value in fields.values() if finite(value)]
    # Missing fields remain unknown; only observed nonzero fields block draining.
    drained = all(value == 0 for value in known) if known else None
    return {"fields": fields, "drained": drained,
            "all_fields_observed": all(finite(value) for value in fields.values())}


def snapshot(a):
    folder = a.folder
    folder.mkdir(parents=True, exist_ok=True)
    endpoints = [("status", a.service_url.rstrip("/") + "/status"),
                 ("p_metrics", a.p_url.rstrip("/") + "/metrics"),
                 ("d_metrics", a.d_url.rstrip("/") + "/metrics")]
    if a.backend_metrics_url:
        endpoints.append(("backend_metrics", a.backend_metrics_url))
    expected = None
    if a.expected_requests is not None and a.data_root:
        expected = a.expected_requests * read(a.data_root / "manifest.json")["input_tokens"]
    deadline = time.monotonic() + a.wait_seconds
    attempts = 0
    while True:
        attempts += 1
        report = {"unix_time": time.time(), "endpoints": {}, "poll_attempts": attempts}
        bodies, status = {}, {}
        for name, url in endpoints:
            record = {"url": url, "http_status": None}
            try:
                timeout = min(3, max(.05, deadline - time.monotonic())) if a.wait_seconds else 10
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    body = response.read().decode("utf-8")
                    record["http_status"] = response.status
                bodies[name] = body
                (folder / f"{a.phase}_{name}.txt").write_text(body, encoding="utf-8")
                if name == "status":
                    status = json.loads(body)
                    dump(folder / f"{a.phase}_status.json", status)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                record["error"] = str(exc)
            report["endpoints"][name] = record
        failures = [k for k, v in report["endpoints"].items() if v.get("http_status") != 200 or "error" in v]
        report["drain"] = drain_state(status)
        alignment = source_alignment(folder, bodies, expected) if a.phase == "after" else None
        report["source_alignment"] = alignment
        sources_ok = alignment is not None and all(value["aligned"] for value in alignment.values())
        drain_ok = report["drain"]["drained"] is not False
        complete = not failures and drain_ok and (not a.require_source_alignment or sources_ok)
        report["post_wait_complete"] = complete
        report["wait_seconds_limit"] = a.wait_seconds
        sources_settled = sources_ok or not (a.require_source_alignment or a.wait_source_alignment)
        if (complete and sources_settled) or not a.wait_seconds or time.monotonic() >= deadline:
            break
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    dump(folder / f"{a.phase}_http.json", report)
    failures = [k for k, v in report["endpoints"].items() if v.get("http_status") != 200 or "error" in v]
    if failures:
        raise SystemExit("Snapshot missing/invalid endpoints: " + ", ".join(failures))
    if a.phase == "after" and not report["post_wait_complete"]:
        raise SystemExit("Post-benchmark drain/source alignment incomplete; native timing was not changed")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_metrics(metrics):
    selected = {k: metrics.get(k) for k in ("output_throughput", "request_throughput", "duration")}
    sources = {}
    # Native detailed arrays are seconds; resulting fields are milliseconds.
    arrays = {"ttft": metrics.get("ttfts")}
    if all(isinstance(metrics.get(k), list) for k in ("latencies", "ttfts", "output_lens")):
        arrays["tpot"] = [(lat - ttft) / (length - 1)
                          for lat, ttft, length in zip(metrics["latencies"], metrics["ttfts"], metrics["output_lens"])
                          if finite(lat) and finite(ttft) and finite(length) and length > 1]
    for name in ("ttft", "tpot"):
        values = arrays.get(name)
        usable = isinstance(values, list) and bool(values) and all(finite(v) for v in values)
        for statistic, native_key in (("mean", f"mean_{name}_ms"), ("p50", f"p50_{name}_ms"), ("p95", f"p95_{name}_ms")):
            value = metrics.get(native_key)
            if statistic == "p50" and not finite(value):
                value = metrics.get(f"median_{name}_ms")
            if finite(value):
                selected[f"{statistic}_{name}_ms"] = value
                sources[f"{statistic}_{name}_ms"] = "native_vllm_result"
            elif usable:
                result = statistics.mean(values) if statistic == "mean" else percentile(values, .5 if statistic == "p50" else .95)
                selected[f"{statistic}_{name}_ms"] = result * 1000
                sources[f"{statistic}_{name}_ms"] = "derived_from_native_detailed_arrays_linear_percentile"
            else:
                selected[f"{statistic}_{name}_ms"] = None
                sources[f"{statistic}_{name}_ms"] = "missing"
    return selected, sources


def flatten_counters(value, path=""):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            if finite(item) and re.search(r"(?:bytes|hits|misses|requests|operations|objects|_total|_count)(?:$|_)", key):
                found[child] = item
            else:
                found.update(flatten_counters(item, child))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.update(flatten_counters(item, f"{path}[{i}]"))
    return found


def metric_counters(text):
    found = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(\S+(?:\{.*\})?)\s+([-+0-9.eE]+)(?:\s|$)", line)
        if match and any(s in match.group(1) for s in ("prompt_tokens_by_source_total", "bytes", "_total", "_count")):
            try:
                value = float(match.group(2))
                if finite(value):
                    found[match.group(1)] = value
            except ValueError:
                pass
    return found


def deltas(first, last):
    result = {}
    for key in sorted(first.keys() | last.keys()):
        start, end = first.get(key), last.get(key)
        valid = finite(start) and finite(end) and end >= start
        result[key] = {"before": start, "after": end, "delta": end - start if valid else None,
                       "note": None if valid else "missing endpoint/series or counter decreased"}
    return result


def finish(a):
    folder = a.folder
    manifest = read(a.data_root / "manifest.json")
    metrics = read(folder / "benchmark.json") if (folder / "benchmark.json").exists() else {}
    count = a.requests
    expected_input = count * manifest["input_tokens"]
    # run.sh passes the exact --custom-output-len here, including an explicit
    # AB_OUTPUT_TOKENS override. Seed always requests one output token.
    expected_output = count * a.output_tokens
    checks = {"completed": metrics.get("completed") == count,
              "failed": metrics.get("failed", 0) == 0,
              "input_tokens": metrics.get("total_input_tokens") == expected_input,
              "output_tokens": metrics.get("total_output_tokens") == expected_output,
              "output_lengths": metrics.get("output_lens") == [a.output_tokens] * count,
              "benchmark_exit": a.exit_code == 0, "after_snapshot_exit": a.snapshot_exit == 0}
    selected, sources = latency_metrics(metrics)
    after_report_path = folder / "after_http.json"
    after_report = read(after_report_path) if after_report_path.exists() else {}
    alignment = after_report.get("source_alignment")
    aligned = isinstance(alignment, dict) and all(alignment.get(role, {}).get("aligned") is True for role in ("p", "d"))
    counters = {}
    for role in ("status", "p_metrics", "d_metrics", "backend_metrics"):
        extension = "json" if role == "status" else "txt"
        paths = [folder / f"{phase}_{role}.{extension}" for phase in ("before", "after")]
        if all(path.exists() for path in paths):
            parser = flatten_counters if role == "status" else metric_counters
            values = [parser(read(path) if role == "status" else path.read_text(encoding="utf-8")) for path in paths]
            counters[role] = deltas(*values)
        else:
            counters[role] = None
    result = {"arm": a.arm, "phase": a.phase, "concurrency": a.concurrency, "round": a.round,
              "dataset_id": manifest["dataset_id"], "input_tokens_per_request": manifest["input_tokens"],
              "output_tokens_per_request": a.output_tokens, "requests": count,
              "manifest_output_tokens": manifest["output_tokens"],
              "output_tokens_source": "seed_fixed_one" if a.phase == "seed" else
                  ("manifest" if a.output_tokens == manifest["output_tokens"] else "explicit_override"),
              "completed": metrics.get("completed"), "failed": metrics.get("failed"),
              "total_input_tokens": metrics.get("total_input_tokens"), "total_output_tokens": metrics.get("total_output_tokens"),
              "checks": checks, "valid_counts": all(checks.values()), "metrics": selected,
              "latency_metric_sources": sources, "observed_counter_deltas": counters,
              "missing_counters_are_not_zero": True,
              "post_benchmark_drain": after_report.get("drain"), "prompt_source_alignment": alignment,
              "accepted_for_comparison": a.phase == "timed" and all(checks.values()) and aligned,
              "l2_path_acceptance": "REVIEW_SUCCESSFUL_CHUNK_COUNTERS_AND_SERVER_LOGS"}
    dump(folder / "summary.json", result)
    print(json.dumps(result, indent=2))
    if not result["valid_counts"]:
        raise SystemExit("Benchmark count/length/exit validation failed")


def summary(a):
    rows = []
    for path in sorted(a.root.rglob("summary.json")):
        row = read(path)
        if row.get("phase") == "timed":
            row["result_directory"] = str(path.parent)
            rows.append(row)
    dump(a.output, {"runs": rows, "aggregation": "individual rounds; no failed/mismatched runs accepted for comparison"})
    print("arm,C,round,valid,output_token_s,mean_ttft_ms,p50_ttft_ms,p95_ttft_ms,mean_tpot_ms,p50_tpot_ms,p95_tpot_ms")
    for row in rows:
        values = [row.get(k) for k in ("arm", "concurrency", "round", "accepted_for_comparison")]
        values += [row["metrics"].get(k) for k in ("output_throughput", "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms")]
        print(",".join("" if v is None else str(v) for v in values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--folder", type=Path, required=True)
    snap.add_argument("--phase", choices=["before", "after"], required=True)
    snap.add_argument("--wait-seconds", type=float, default=0)
    snap.add_argument("--expected-requests", type=int)
    snap.add_argument("--data-root", type=Path)
    snap.add_argument("--require-source-alignment", action="store_true")
    snap.add_argument("--wait-source-alignment", action="store_true")
    for key, default in (("service-url", "http://127.0.0.1:28185"), ("p-url", "http://127.0.0.1:28301"), ("d-url", "http://127.0.0.1:28302"), ("backend-metrics-url", "")):
        snap.add_argument("--" + key, default=default)
    done = sub.add_parser("finish")
    done.add_argument("--folder", type=Path, required=True)
    done.add_argument("--data-root", type=Path, required=True)
    done.add_argument("--arm", required=True)
    done.add_argument("--phase", choices=["seed", "warmup", "timed"], required=True)
    for key in ("concurrency", "round", "requests", "output-tokens", "exit-code", "snapshot-exit"):
        done.add_argument("--" + key, type=int, required=True)
    collect = sub.add_parser("summary")
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    {"snapshot": snapshot, "finish": finish, "summary": summary}[a.operation](a)


if __name__ == "__main__":
    main()
