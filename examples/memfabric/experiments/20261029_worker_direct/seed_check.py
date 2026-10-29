#!/usr/bin/env python3
"""Seed B/C through P, then verify two D consumer requests; plan by default.

Only HTTP calls to already-running services. No SSH, process launch, cache
clearing, model changes, or tokenizer imports. Each run gets a fresh evidence
directory so a failed/interrupted check cannot become a later PASS.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import urllib.error
import urllib.request

import bench_d


ROOT = Path(os.environ.get("MF_EXPERIMENT_ROOT", "/path/to/private/memfabric-worker-direct"))


def http_post(url, payload, timeout=90):
    record = dict(url=url, started_unix=time.time(), http_status=None)
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            record["http_status"] = response.status
        record["body"] = body
        record["response"] = json.loads(body)
    except urllib.error.HTTPError as error:
        record.update(http_status=error.code, body=error.read().decode("utf-8", errors="replace"),
                      error=str(error))
    except (OSError, ValueError) as error:
        record["error"] = str(error)
    record["finished_unix"] = time.time()
    return record


def get_json(url):
    record = dict(url=url, http_status=None)
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read().decode("utf-8")
            record["http_status"] = response.status
        record.update(body=body, json=json.loads(body))
    except (OSError, ValueError) as error:
        record["error"] = str(error)
    return record


def pool_state(value):
    if isinstance(value, dict):
        if {"objects", "pending_writes", "total_pins", "quarantined_slots"} <= value.keys():
            return value
        for item in value.values():
            result = pool_state(item)
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = pool_state(item)
            if result is not None:
                return result
    return None


def adapter_state(value):
    if isinstance(value, dict):
        if {"objects", "store_bytes", "load_bytes", "total_pins"} <= value.keys():
            return value
        for item in value.values():
            result = adapter_state(item)
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = adapter_state(item)
            if result is not None:
                return result
    return None


def dataset_prompts(dataset, seed_unique):
    with dataset.open(encoding="utf-8") as stream:
        prompts = [json.loads(line)["prompt"] for line in stream if line.strip()]
    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Expected nonempty raw text prompts")
    return list(dict.fromkeys(prompts)) if seed_unique else prompts[:1]


def l1_empty(status):
    manager = status.get("storage_manager", {}).get("l1_manager", {})
    objects, used = manager.get("total_object_count"), manager.get("memory_used_bytes")
    return objects == 0 and used == 0


def completion_text(record):
    value = record.get("response", {})
    choices = value.get("choices", [])
    return choices[0].get("text", "") if choices else ""


def counter_write_delta(before, after):
    first, last = before.get("json", {}), after.get("json", {})
    same = all(first.get(key) is not None and first.get(key) == last.get(key)
               for key in ("pid", "session_id"))
    changes = bench_d.delta_values(bench_d.numeric_fields(first.get("operations", {})),
                                   bench_d.numeric_fields(last.get("operations", {}))) if same else {}
    return dict(same_session=same, completed_bytes=changes.get("L2G.completed_bytes", {}).get("delta"),
                operations=changes)


def request_checks(args, response, before, after):
    alignment = bench_d.alignment(before, after, 1, args.input_tokens)
    sources = {key: item["delta"] for key, item in alignment.get("d", {}).get("sources", {}).items()}
    expected_external = 0 if args.case == "A" else args.expected_external_tokens
    traffic = bench_d.observed_bytes(before, after)
    checks = dict(http_success=response.get("http_status") == 200 and "error" not in response,
                  usage_prompt_tokens=response.get("response", {}).get("usage", {}).get("prompt_tokens") == args.input_tokens,
                  answer_orchid="ORCHID" in completion_text(response),
                  d_source_alignment=alignment.get("d", {}).get("aligned") is True,
                  p_no_new_requests=alignment.get("p", {}).get("aligned") is True,
                  external_tokens=sources.get("external_kv_transfer") == expected_external,
                  local_compute=sources.get("local_compute") == args.input_tokens - expected_external,
                  local_prefix_zero=sources.get("local_cache_hit") == 0,
                  before_http=bench_d.endpoint_success(before), after_http=bench_d.endpoint_success(after),
                  protection_drained=bench_d.drain_state(after)["drained"] is not False)
    if args.case == "A":
        checks["ordinary_l1_empty_before"] = l1_empty(before["json"].get("status", {}))
        checks["ordinary_l1_empty_after"] = l1_empty(after["json"].get("status", {}))
    else:
        checks["actual_pool_read"] = bench_d.finite(traffic["read_bytes"]) and traffic["read_bytes"] > 0
    if args.case == "C":
        checks["counter_session_unchanged"] = traffic["counter_same_session"] is True
        checks["cpu_copy_attempts_zero"] = traffic["cpu_copy_attempts"] == 0
        operations = traffic["worker_operation_deltas"] or {}
        checks["no_copy_errors"] = bool(operations) and all(
            item["delta"] == 0 for key, item in operations.items() if key.endswith(".failed_calls"))
    return dict(checks=checks, passed=all(checks.values()), answer=completion_text(response),
                prompt_sources=alignment, observed_transfer=traffic,
                after_drain=bench_d.drain_state(after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("A", "B", "C"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "bench1373/data/custom.jsonl")
    parser.add_argument("--input-tokens", type=int, default=1373)
    parser.add_argument("--expected-external-tokens", type=int, default=1280)
    parser.add_argument("--model", default="mf-night")
    parser.add_argument("--p-url", default="http://127.0.0.1:28401")
    parser.add_argument("--d-url", default="http://127.0.0.1:28402")
    parser.add_argument("--service-url", default="http://127.0.0.1:28195")
    parser.add_argument("--p-counter-url", default="http://127.0.0.1:29401/counters")
    parser.add_argument("--d-counter-url", default="http://127.0.0.1:29402/counters")
    parser.add_argument("--wait-seconds", type=float, default=90)
    parser.add_argument("--seed-unique", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    folder = args.run_dir / "consumer_check"
    plan = dict(mode="RUN" if args.run else "PLAN_ONLY", case=args.case, folder=str(folder),
                dataset=str(args.dataset), seed_unique=args.seed_unique,
                seed_requests=0 if args.case == "A" else "dataset_unique_prompts" if args.seed_unique else 1,
                d_requests=2, d_max_tokens=8, services_started=False,
                expected_input_tokens=args.input_tokens,
                expected_external_tokens=0 if args.case == "A" else args.expected_external_tokens)
    print(json.dumps(plan, indent=2))
    if not args.run:
        return 0
    if not 0 <= args.expected_external_tokens < args.input_tokens:
        raise ValueError("Expected external tokens must fit the prompt")
    if folder.exists():
        raise FileExistsError("Use a fresh backend case; refusing to overwrite consumer_check")
    prompts = dataset_prompts(args.dataset, args.seed_unique)
    prompt = prompts[0]
    plan["seed_requests"] = 0 if args.case == "A" else len(prompts)
    expected_objects = len(prompts) * (args.expected_external_tokens // 128)
    expected_seed_prompt_tokens = len(prompts) * args.input_tokens
    folder.mkdir(parents=True)
    bench_d.dump(folder / "plan.json", plan)
    snap_args = SimpleNamespace(base_url=args.d_url, service_url=args.service_url,
                                p_url=args.p_url, counter_url=args.d_counter_url if args.case == "C" else "",
                                metrics_url="", http_timeout=3)
    initial = bench_d.take_snapshot(snap_args)
    bench_d.save_snapshot(folder, "initial", initial)
    seed, seed_responses = None, []
    ready = args.case == "A"
    seed_evidence = {}
    if args.case != "A":
        p_before = get_json(args.p_counter_url) if args.case == "C" else None
        if p_before is not None:
            bench_d.dump(folder / "seed_p_counter_before.json", p_before)
        for index, seed_prompt in enumerate(prompts, 1):
            response = http_post(args.p_url.rstrip("/") + "/v1/completions",
                                 dict(model=args.model, prompt=seed_prompt, max_tokens=1, temperature=0))
            response["telemetry_delivery"] = "NOT_OBSERVED_BY_CLIENT; HTTP success does not prove store notification delivery"
            response["seed_index"] = index
            seed_responses.append(response)
            bench_d.dump(folder / f"seed_response_{index:02d}.json", response)
        seed = seed_responses[0]
        bench_d.dump(folder / "seed_response.json", seed)
        deadline = time.monotonic() + min(args.wait_seconds, 90)
        while True:
            current = bench_d.take_snapshot(snap_args)
            if args.case == "C":
                p_after = get_json(args.p_counter_url)
                writes = counter_write_delta(p_before, p_after)
                completed_calls = writes["operations"].get("L2G.completed_calls", {}).get("delta")
                pool = pool_state(current["json"].get("status", {}))
                ready = (bench_d.endpoint_success(current) and writes["same_session"]
                         and bench_d.finite(writes["completed_bytes"]) and writes["completed_bytes"] > 0
                         and pool is not None and pool["objects"] >= max(1, expected_objects)
                         and bench_d.finite(completed_calls) and completed_calls >= expected_objects
                         and pool["pending_writes"] == pool["quarantined_slots"] == 0)
                seed_evidence = dict(p_write=writes, pool=pool, p_counter=p_after)
            else:
                traffic = bench_d.observed_bytes(initial, current)
                adapter = adapter_state(current["json"].get("status", {}))
                ready = (bench_d.endpoint_success(current) and bench_d.finite(traffic["write_bytes"])
                         and traffic["write_bytes"] > 0 and adapter is not None
                         and adapter["objects"] >= max(1, expected_objects)
                         and bench_d.drain_state(current)["drained"] is not False)
                seed_evidence = dict(adapter_transfer=traffic, adapter=adapter)
            # Let P's asynchronous source metric settle before the D-only
            # windows, so a delayed seed statistic is not mistaken for new P work.
            p_counts = bench_d.source_counts(current["bodies"].get("p_metrics", ""))
            p_initial = bench_d.source_counts(initial["bodies"].get("p_metrics", ""))
            p_known = all(bench_d.finite(value) for value in p_counts.values())
            p_initial_known = all(bench_d.finite(value) for value in p_initial.values())
            p_total = sum(p_counts.values()) if p_known else None
            p_delta = p_total - sum(p_initial.values()) if p_known and p_initial_known else None
            p_settled = (p_delta == expected_seed_prompt_tokens if p_delta is not None
                         else p_known and p_total == expected_seed_prompt_tokens)
            seed_evidence["p_seed_source_metric"] = dict(total=p_total, delta=p_delta, settled=p_settled,
                                                        expected=expected_seed_prompt_tokens)
            ready = ready and p_settled
            if ready or time.monotonic() >= deadline:
                break
            time.sleep(.2)
        bench_d.save_snapshot(folder, "seed_ready", current)
        bench_d.dump(folder / "seed_evidence.json", dict(ready=ready, **seed_evidence))
    results = []
    if ready:
        for index in (1, 2):
            before = bench_d.take_snapshot(snap_args)
            bench_d.save_snapshot(folder, f"d{index}_before", before)
            response = http_post(args.d_url.rstrip("/") + "/v1/completions",
                                 dict(model=args.model, prompt=prompt, max_tokens=8, temperature=0))
            bench_d.dump(folder / f"d{index}_response.json", response)
            deadline = time.monotonic() + 15
            while True:
                after = bench_d.take_snapshot(snap_args)
                aligned = bench_d.alignment(before, after, 1, args.input_tokens)
                if (aligned.get("d", {}).get("aligned") is True
                        and aligned.get("p", {}).get("aligned") is True
                        and bench_d.drain_state(after)["drained"] is not False) or time.monotonic() >= deadline:
                    break
                time.sleep(.2)
            bench_d.save_snapshot(folder, f"d{index}_after", after)
            result = request_checks(args, response, before, after)
            result["request_index"] = index
            bench_d.dump(folder / f"d{index}_checks.json", result)
            results.append(result)
    summary = dict(case=args.case, seed_ready=ready, seed_response=seed,
                   seed_responses=seed_responses, seed_unique=args.seed_unique,
                   unique_seed_prompts=0 if args.case == "A" else len(prompts),
                   expected_seed_objects=0 if args.case == "A" else expected_objects,
                   d_requests=results, passed=ready and len(results) == 2 and all(row["passed"] for row in results),
                   payload_source="first raw prompt in common dataset", no_cache_clear=True,
                   note="Consumer functional preflight only, not a performance result")
    bench_d.dump(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
