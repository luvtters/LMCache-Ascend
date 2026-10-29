#!/usr/bin/env python3
"""D-only official vLLM benchmark; default is a plan, never starts services.

Rendering uses only a local tokenizer. --run --prepare-only writes the common
dataset without sending requests. --run executes one named case/round. Native
benchmark JSON, its log/return code, and HTTP evidence are preserved separately.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request


TOKEN_SOURCES = ("local_compute", "external_kv_transfer", "local_cache_hit")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def source_counts(text):
    """Sum source-labelled series; absent or invalid series stay unknown."""
    result = dict.fromkeys(TOKEN_SOURCES)
    for line in text.splitlines():
        match = re.match(r'^vllm:prompt_tokens_by_source_total\{([^}]*)\}\s+(\S+)', line)
        if not match:
            continue
        label = re.search(r'(?:^|,)\s*source="([^"]*)"(?:\s*,|\s*$)', match[1])
        if not label or label[1] not in result:
            continue
        try:
            value = float(match[2])
        except ValueError:
            return dict.fromkeys(TOKEN_SOURCES)
        if not finite(value) or value < 0:
            return dict.fromkeys(TOKEN_SOURCES)
        result[label[1]] = (result[label[1]] or 0) + value
    return result


def delta_values(before, after):
    result = {}
    for key in sorted(before.keys() | after.keys()):
        start, end = before.get(key), after.get(key)
        valid = finite(start) and finite(end) and end >= start
        result[key] = dict(before=start, after=end,
                           delta=end - start if valid else None,
                           state="known" if valid else "missing_or_counter_reset")
    return result


def numeric_fields(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if finite(item):
                result[name] = item
            else:
                result.update(numeric_fields(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(numeric_fields(item, f"{prefix}[{index}]"))
    return result


def endpoint(base, suffix):
    return base.rstrip("/") + suffix


def take_snapshot(args):
    urls = {"d_metrics": args.metrics_url or endpoint(args.base_url, "/metrics"),
            "status": endpoint(args.service_url, "/status")}
    if args.p_url:
        urls["p_metrics"] = endpoint(args.p_url, "/metrics")
    if args.counter_url:
        urls["counter"] = args.counter_url
    snapshot = {"started_unix": time.time(), "endpoints": {}, "bodies": {}, "json": {}}
    for name, url in urls.items():
        record = {"url": url}
        try:
            with urllib.request.urlopen(url, timeout=args.http_timeout) as response:
                body = response.read().decode("utf-8")
                record["http_status"] = response.status
            snapshot["bodies"][name] = body
            if name in ("status", "counter"):
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise ValueError("Expected JSON object")
                snapshot["json"][name] = value
        except (OSError, ValueError) as error:
            record["error"] = str(error)
        snapshot["endpoints"][name] = record
    snapshot["finished_unix"] = time.time()
    return snapshot


def save_snapshot(folder, phase, snapshot):
    report = {key: value for key, value in snapshot.items() if key not in ("bodies", "json")}
    for name, body in snapshot["bodies"].items():
        (folder / f"{phase}_{name}.txt").write_text(body, encoding="utf-8")
    for name, value in snapshot["json"].items():
        dump(folder / f"{phase}_{name}.json", value)
    dump(folder / f"{phase}_http.json", report)


def endpoint_success(snapshot):
    return all(record.get("http_status") == 200 and "error" not in record
               for record in snapshot["endpoints"].values())


def alignment(before, after, requests, input_tokens):
    result = {}
    for role, expected in (("d", requests * input_tokens), ("p", 0)):
        key = role + "_metrics"
        if key not in before["bodies"] and key not in after["bodies"]:
            continue
        changes = delta_values(source_counts(before["bodies"].get(key, "")),
                               source_counts(after["bodies"].get(key, "")))
        values = [item["delta"] for item in changes.values()]
        known = all(finite(value) for value in values)
        total = sum(values) if known else None
        result[role] = dict(sources=changes, observed_prompt_tokens=total,
                            expected_prompt_tokens=expected, aligned=known and total == expected)
    return result


def drain_state(snapshot, case=None):
    """Observe zero work/leases; required missing gauges remain unknown.

    L1 temporary objects are deleted after their final read lock is released;
    with this experiment's default prefetch policy their idle count is zero.
    """
    names = {"pending_writes", "total_pins", "quarantined_slots", "pending_keys_count", "in_flight_task_count",
             "in_flight_request_count", "locked_objects", "read_locked_count", "write_locked_count",
             "staging_object_count", "temporary_count"}
    fields = {key: value for key, value in numeric_fields(snapshot["json"].get("status", {})).items()
              if key.rsplit(".", 1)[-1] in names}
    counter = snapshot["json"].get("counter", {})
    fields.update({"counter." + key: value for key, value in numeric_fields(counter).items()
                   if key.endswith(".in_flight")})
    required = set()
    if case in ("A", "B", "C"):
        required.update(("read_locked_count", "write_locked_count", "staging_object_count", "temporary_count"))
    if case in ("B", "C"):
        required.add("total_pins")
    if case == "C":
        required.update(("pending_writes", "quarantined_slots"))
    observed = {key.rsplit(".", 1)[-1] for key in fields if not key.startswith("counter.")}
    missing = required - observed
    drained = (False if any(value != 0 for value in fields.values())
               else None if missing or not fields else True)
    return dict(fields=fields, required_names=sorted(required), missing_required=sorted(missing),
                drained=drained)


def observed_bytes(before, after):
    """Actual completed-copy bytes only; never infer traffic from restored tokens."""
    status_delta = delta_values(numeric_fields(before["json"].get("status", {})),
                                numeric_fields(after["json"].get("status", {})))
    byte_fields = {key: value for key, value in status_delta.items()
                   if key.rsplit(".", 1)[-1] in ("load_bytes", "store_bytes")}
    first, last = before["json"].get("counter"), after["json"].get("counter")
    counter_delta = None
    same_session = None
    read_bytes = write_bytes = cpu_copy_attempts = None
    source = "not_observed"
    if first is not None and last is not None:
        same_session = all(first.get(key) is not None and first.get(key) == last.get(key)
                           for key in ("pid", "session_id"))
        if same_session:
            counter_delta = delta_values(numeric_fields(first.get("operations", {})),
                                          numeric_fields(last.get("operations", {})))
            read_bytes = counter_delta.get("G2L.completed_bytes", {}).get("delta")
            write_bytes = counter_delta.get("L2G.completed_bytes", {}).get("delta")
            cpu_copy_attempts = delta_values(
                {"cpu_copy_attempts": first.get("cpu_copy_attempts")},
                {"cpu_copy_attempts": last.get("cpu_copy_attempts")},
            )["cpu_copy_attempts"]["delta"]
            source = "worker_completed_bm_copy_counters"
    else:
        reads = [item["delta"] for key, item in byte_fields.items() if key.endswith(".load_bytes")]
        writes = [item["delta"] for key, item in byte_fields.items() if key.endswith(".store_bytes")]
        if reads and all(finite(value) for value in reads):
            read_bytes = sum(reads)
            source = "l2_adapter_completed_load_bytes"
        if writes and all(finite(value) for value in writes):
            write_bytes = sum(writes)
    return dict(read_bytes=read_bytes, write_bytes=write_bytes, source=source,
                counter_same_session=same_session, cpu_copy_attempts=cpu_copy_attempts,
                worker_operation_deltas=counter_delta,
                status_byte_deltas=byte_fields, tokens_used_to_estimate_bytes=False)


def prepare_data(args):
    if args.dataset:
        if args.input_tokens is None:
            raise ValueError("--dataset requires --input-tokens from the verified manifest")
        if not args.dataset.is_file():
            raise ValueError("Dataset does not exist")
        return args.dataset, args.input_tokens, {"reused_dataset": str(args.dataset),
                                                "input_tokens": args.input_tokens}
    if not args.request_json or not args.tokenizer:
        raise ValueError("Use --request-json with --tokenizer, or --dataset with --input-tokens")
    if not Path(args.tokenizer).is_dir():
        raise ValueError("Tokenizer must be an existing local directory; downloads are forbidden")
    from transformers import AutoTokenizer
    request = read(args.request_json)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    with_special = len(tokenizer.encode(prompt, add_special_tokens=True))
    if args.input_tokens is not None and input_tokens != args.input_tokens:
        raise ValueError(f"Rendered input {input_tokens} differs from expected {args.input_tokens}")
    dataset = args.output_root / "data" / "custom.jsonl"
    if dataset.exists():
        raise ValueError("Refuse to overwrite an existing rendered dataset; use --dataset to reuse it")
    dataset.parent.mkdir(parents=True, exist_ok=True)
    row = json.dumps({"prompt": prompt, "output_tokens": args.output_tokens}, ensure_ascii=False)
    dataset.write_text((row + "\n") * args.requests, encoding="utf-8")
    manifest = dict(request_json=str(args.request_json), tokenizer=args.tokenizer,
                    input_tokens=input_tokens, input_tokens_with_special_tokens=with_special,
                    output_tokens=args.output_tokens, requests=args.requests,
                    identical_prompt_rows=True, add_generation_prompt=True, enable_thinking=False,
                    token_count_verified_by="local_tokenizer_only_until_native_benchmark_result",
                    dataset=str(dataset))
    dump(dataset.parent / "manifest.json", manifest)
    (dataset.parent / "prompt.txt").write_text(prompt, encoding="utf-8")
    return dataset, input_tokens, manifest


def command(args, folder, dataset):
    return [args.python, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai", "--base-url", args.base_url,
            "--endpoint", args.endpoint, "--model", args.model, "--served-model-name", args.model,
            "--tokenizer", args.tokenizer, "--dataset-name", "custom", "--dataset-path", str(dataset),
            "--custom-output-len", str(args.output_tokens), "--num-prompts", str(args.requests),
            "--max-concurrency", str(args.concurrency), "--request-rate", "inf", "--temperature", "0",
            "--num-warmups", "0", "--ready-check-timeout-sec", "0", "--ignore-eos",
            "--disable-shuffle", "--no-oversample", "--seed", "0", "--skip-chat-template",
            "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,95",
            "--save-result", "--save-detailed", "--result-dir", str(folder),
            "--result-filename", "benchmark.json"]


def summarize(args, metrics, before, after, input_tokens, returncode):
    sources = alignment(before, after, args.requests, input_tokens)
    traffic = observed_bytes(before, after)
    d_values = {key: value["delta"] for key, value in sources.get("d", {}).get("sources", {}).items()}
    before_drain, after_drain = drain_state(before, args.case), drain_state(after, args.case)
    checks = dict(benchmark_exit=returncode == 0,
                  completed=metrics.get("completed") == args.requests,
                  no_failed_requests=metrics.get("failed") == 0,
                  input_tokens=metrics.get("total_input_tokens") == args.requests * input_tokens,
                  output_tokens=metrics.get("total_output_tokens") == args.requests * args.output_tokens,
                  output_lengths=metrics.get("output_lens") == [args.output_tokens] * args.requests,
                  http_before=endpoint_success(before), http_after=endpoint_success(after),
                  d_source_alignment=sources.get("d", {}).get("aligned") is True,
                  local_prefix_zero=d_values.get("local_cache_hit") == 0,
                  before_drained=before_drain["drained"] is True,
                  after_drained=after_drain["drained"] is True)
    if args.p_url:
        checks["p_no_new_requests"] = sources.get("p", {}).get("aligned") is True
    external = d_values.get("external_kv_transfer")
    if args.case == "A":
        checks["all_recomputed"] = external == 0 and d_values.get("local_compute") == args.requests * input_tokens
    else:
        checks["external_restored"] = finite(external) and external > 0
        checks["actual_pool_read_observed"] = finite(traffic["read_bytes"]) and traffic["read_bytes"] > 0
    if args.expected_external_tokens is not None:
        checks["expected_external_tokens"] = external == args.requests * args.expected_external_tokens
    if args.counter_url:
        checks["counter_same_session"] = traffic["counter_same_session"] is True
        checks["cpu_copy_attempts_zero"] = traffic["cpu_copy_attempts"] == 0
        errors = traffic.get("worker_operation_deltas") or {}
        checks["no_copy_errors"] = all(item["delta"] == 0 for key, item in errors.items()
                                        if key.endswith(".failed_calls")) and bool(errors)
    selected = {key: metrics.get(key) for key in (
        "duration", "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p95_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms", "p95_tpot_ms")}
    return dict(case=args.case, concurrency=args.concurrency, round=args.round,
                requests=args.requests, input_tokens_per_request=input_tokens,
                output_tokens_per_request=args.output_tokens, subprocess_returncode=returncode,
                completed=metrics.get("completed"), failed=metrics.get("failed"),
                checks=checks, accepted_for_comparison=all(checks.values()), metrics=selected,
                prompt_sources=sources, p_metrics_observed=bool(args.p_url),
                observed_transfer=traffic, before_drain=before_drain, after_drain=after_drain,
                scope="D-only recovery or recomputation; seed and warmup excluded",
                same_prefix_concurrency_can_deduplicate=True,
                missing_counters_are_unknown=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("A", "B", "C"), required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=1)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--expected-external-tokens", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:28202")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--service-url", default="http://127.0.0.1:28175")
    parser.add_argument("--p-url", default="")
    parser.add_argument("--counter-url", default="")
    parser.add_argument("--model", default="mf-direct")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--http-timeout", type=float, default=3)
    parser.add_argument("--after-wait-seconds", type=float, default=15)
    parser.add_argument("--benchmark-timeout", type=float, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.requests, args.output_tokens, args.round) < 1:
        parser.error("requests/output-tokens/round must be positive")
    folder = args.output_root / f"{args.case}_c{args.concurrency}_r{args.round}"
    planned_dataset = args.dataset or args.output_root / "data" / "custom.jsonl"
    plan = dict(mode="RUN" if args.run and not args.dry_run else "PLAN_ONLY",
                output_directory=str(folder), prepare_only=args.prepare_only,
                command=command(args, folder, planned_dataset),
                request_json=str(args.request_json) if args.request_json else None,
                network_requests_in_plan=False, services_started=False)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.run or args.dry_run:
        return 0
    if not args.prepare_only:
        if not args.p_url.strip():
            parser.error("Real benchmark runs require --p-url to verify D-only work")
        if args.case in ("B", "C") and args.expected_external_tokens is None:
            parser.error("B/C benchmark runs require --expected-external-tokens")
        if args.case == "C" and not args.counter_url.strip():
            parser.error("C benchmark runs require --counter-url for completed-copy evidence")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only and folder.exists():
        raise ValueError("Refuse existing result directory; choose a new round/output-root")
    dataset, input_tokens, manifest = prepare_data(args)
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    folder.mkdir()
    argv = command(args, folder, dataset)
    dump(folder / "command.json", argv)
    (folder / "command.sh").write_text(shlex.join(argv) + "\n", encoding="utf-8")
    dump(folder / "dataset_receipt.json", manifest)
    before = take_snapshot(args)
    save_snapshot(folder, "before", before)
    if not endpoint_success(before):
        dump(folder / "execution.json", {"status": "BEFORE_SNAPSHOT_FAILED", "benchmark_started": False})
        return 2
    before_drain = drain_state(before, args.case)
    if before_drain["drained"] is not True:
        dump(folder / "execution.json", {
            "status": "BEFORE_NOT_DRAINED" if before_drain["drained"] is False else "BEFORE_DRAIN_UNVERIFIED",
            "benchmark_started": False, "before_drain": before_drain,
        })
        return 2
    start = time.time()
    returncode, error = None, None
    try:
        with (folder / "benchmark.log").open("wb") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=args.benchmark_timeout)
            except subprocess.TimeoutExpired:
                # This is only the benchmark client we own, never the services.
                process.terminate()
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
                error = "benchmark_client_timeout"
    except OSError as exception:
        error = str(exception)
    execution = dict(started_unix=start, finished_unix=time.time(),
                     subprocess_returncode=returncode, error=error,
                     recorded_at=datetime.now(timezone.utc).isoformat())
    dump(folder / "execution.json", execution)
    deadline = time.monotonic() + args.after_wait_seconds
    while True:
        after = take_snapshot(args)
        sources = alignment(before, after, args.requests, input_tokens)
        settled = endpoint_success(after) and sources.get("d", {}).get("aligned") is True
        settled = settled and (not args.p_url or sources.get("p", {}).get("aligned") is True)
        settled = settled and drain_state(after, args.case)["drained"] is True
        if settled or time.monotonic() >= deadline:
            break
        time.sleep(.2)
    save_snapshot(folder, "after", after)
    result_path = folder / "benchmark.json"
    metrics = read(result_path) if result_path.exists() else {}
    summary = summarize(args, metrics, before, after, input_tokens, returncode)
    if error:
        summary["checks"]["client_execution_error_absent"] = False
        summary["accepted_for_comparison"] = False
    dump(folder / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["accepted_for_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
