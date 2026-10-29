#!/usr/bin/env python3
"""Functional P-write/D-read client for an existing diagnostic group.

Plans by default. No service/process control, cache clear, direct pool release,
or native transport replacement. HTTP interval overlap is not DMA overlap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

import bench_d
import seed_check


CHUNK_BYTES = 18 << 20
OLD_OBJECTS = 40
FINAL_OBJECTS = 60
OPERATIONS = ("L2G", "G2L", "gather", "scatter")


def prepare_prompts(prompts, tokenizer):
    if len(prompts) != 4 or "Record 000" not in prompts[0]:
        raise ValueError("Expected the original four-prefix dataset, starting with Record 000")
    new = [prompts[0].replace("Record 000", "Record " + marker, 1) for marker in ("400", "500")]
    items = []
    for index, prompt in enumerate([*prompts, *new]):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) != 1373:
            raise ValueError(f"Variant {index} has {len(tokens)} tokens instead of 1373")
        items.append(dict(index=index, input_tokens=len(tokens), first_chunk_tokens=list(tokens[:128])))
    if len({tuple(item["first_chunk_tokens"]) for item in items}) != 6 or any(prompt in prompts for prompt in new):
        raise ValueError("All six prefixes must differ in the first complete chunk")
    return new, dict(old_prefixes=4, new_prefixes=2, input_tokens=1373, chunk_tokens=128,
                     final_objects=FINAL_OBJECTS, total_slots=64, variants=items,
                     first_complete_chunks_distinct=True, tokenizer_only_until_http_usage=True)


def take_snapshot(args):
    snap_args = SimpleNamespace(base_url=args.d_url, service_url=args.service_url,
                                p_url=args.p_url, counter_url=args.d_counter_url,
                                metrics_url="", http_timeout=3)
    snapshot = bench_d.take_snapshot(snap_args)
    producer = seed_check.get_json(args.p_counter_url)
    snapshot["endpoints"]["p_counter"] = {key: producer[key] for key in ("url", "http_status", "error") if key in producer}
    if "body" in producer:
        snapshot["bodies"]["p_counter"] = producer["body"]
    if isinstance(producer.get("json"), dict):
        snapshot["json"]["p_counter"] = producer["json"]
    return snapshot


def idle_state(snapshot):
    drain = bench_d.drain_state(snapshot, "C")
    work = {}
    for role, counter_name in (("p", "p_counter"), ("d", "counter")):
        fields = bench_d.numeric_fields(snapshot["json"].get(counter_name, {}))
        required = ["operations." + name + ".in_flight" for name in OPERATIONS]
        required += ["retrieve.in_flight", *["rpc." + name + ".active_result_calls"
                                          for name in ("lookup", "acquire", "release", "other")]]
        work.update({role + "." + name: fields.get(name) for name in required})
        text = snapshot["bodies"].get(role + "_metrics", "")
        for name in ("num_requests_running", "num_requests_waiting"):
            matches = re.findall(r"^vllm:" + name + r"(?:\{[^}]*\})?\s+(\S+)", text, re.M)
            try:
                values = [float(value) for value in matches]
            except ValueError:
                values = []
            work[role + "." + name] = sum(values) if values and all(bench_d.finite(value) and value >= 0 for value in values) else None
    known = bench_d.endpoint_success(snapshot) and drain["drained"] is not None and all(bench_d.finite(value) for value in work.values())
    return dict(known=known, idle=known and drain["drained"] is True and all(value == 0 for value in work.values()),
                pool_and_l1=drain, worker_and_request_gauges=work)


def pool(snapshot):
    return seed_check.pool_state(snapshot["json"].get("status", {}))


def source_delta(before, after, role):
    changes = bench_d.delta_values(bench_d.source_counts(before["bodies"].get(role + "_metrics", "")),
                                   bench_d.source_counts(after["bodies"].get(role + "_metrics", "")))
    return {key: item["delta"] for key, item in changes.items()}


def source_total(before, after, role):
    values = source_delta(before, after, role).values()
    return sum(values) if all(bench_d.finite(value) for value in values) else None


def transfers(before, after, role):
    def view(snapshot):
        return dict(json=dict(status=snapshot["json"].get("status", {}),
                              counter=snapshot["json"].get("p_counter" if role == "p" else "counter")))
    return bench_d.observed_bytes(view(before), view(after))


def native_ok(before, after, role):
    traffic = transfers(before, after, role)
    changes = traffic.get("worker_operation_deltas") or {}
    return (traffic["counter_same_session"] is True and traffic["cpu_copy_attempts"] == 0
            and all(changes.get(name + ".failed_calls", {}).get("delta") == 0 for name in OPERATIONS))


def wait_for(args, predicate):
    deadline = time.monotonic() + min(args.wait_seconds, 90)
    quiet_since = None
    while True:
        current = take_snapshot(args)
        state = idle_state(current)
        ready = predicate(current) and state["idle"]
        now = time.monotonic()
        quiet_since = (quiet_since or now) if ready else None
        if quiet_since is not None and now - quiet_since >= .5:
            return current, dict(passed=True, idle=state)
        if now >= deadline:
            return current, dict(passed=False, idle=state, status="TIMEOUT_OR_UNKNOWN")
        time.sleep(.2)


def post(args, role, prompt, maximum):
    return seed_check.http_post((args.p_url if role == "p" else args.d_url).rstrip("/") + "/v1/completions",
                               dict(model=args.model, prompt=prompt, max_tokens=maximum, temperature=0), args.request_timeout)


def response_ok(response, *, decode):
    value = response.get("response", {})
    return (response.get("http_status") == 200 and "error" not in response
            and value.get("usage", {}).get("prompt_tokens") == 1373
            and ("ORCHID" in seed_check.completion_text(response) if decode
                 else value.get("usage", {}).get("completion_tokens") == 1))


def run_mixed_http(args, old_prompts, new_prompts):
    barrier_record = dict(created_monotonic_ns=time.monotonic_ns())
    barrier = threading.Barrier(4, action=lambda: barrier_record.update(release_monotonic_ns=time.monotonic_ns()))

    def worker(role, prompts):
        ready = time.monotonic_ns()
        barrier.wait(timeout=10)
        released = time.monotonic_ns()
        responses = []
        for index, prompt in enumerate(prompts):
            started = time.monotonic_ns()
            response = post(args, role, prompt, 1 if role == "p" else 8)
            response.update(role=role, request_index=index, client_started_monotonic_ns=started,
                            client_finished_monotonic_ns=time.monotonic_ns())
            responses.append(response)
            if response.get("http_status") != 200 or "error" in response:
                break
        return dict(role=role, ready_monotonic_ns=ready, barrier_return_monotonic_ns=released,
                    responses=responses)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, "p", [new_prompts[0]]),
                   executor.submit(worker, "p", [new_prompts[1]]),
                   executor.submit(worker, "d", [old_prompts[index % 4] for index in range(8)])]
        barrier_record["caller_enter_monotonic_ns"] = time.monotonic_ns()
        barrier.wait(timeout=10)
        results = [future.result() for future in futures]
    reads = [response for item in results if item["role"] == "d" for response in item["responses"]]
    writes = [response for item in results if item["role"] == "p" for response in item["responses"]]
    overlap = []
    for pi, producer in enumerate(writes):
        for di, decoder in enumerate(reads):
            start = max(producer["client_started_monotonic_ns"], decoder["client_started_monotonic_ns"])
            end = min(producer["client_finished_monotonic_ns"], decoder["client_finished_monotonic_ns"])
            if end > start:
                overlap.append(dict(p_index=pi, d_index=di, overlap_ns=end - start))
    return dict(barrier=barrier_record, workers=results, p_responses=writes, d_responses=reads,
                client_http_interval_overlap_observed=bool(overlap), overlaps=overlap,
                native_dma_overlap_verified=False)


def mixed_checks(before, after, requests):
    producer, decoder = transfers(before, after, "p"), transfers(before, after, "d")
    p_source, d_source = source_delta(before, after, "p"), source_delta(before, after, "d")
    first, last = pool(before), pool(after)
    checks = dict(p_http=len(requests["p_responses"]) == 2 and all(response_ok(row, decode=False) for row in requests["p_responses"]),
                  d_http=len(requests["d_responses"]) == 8 and all(response_ok(row, decode=True) for row in requests["d_responses"]),
                  idle_before=idle_state(before)["idle"], idle_after=idle_state(after)["idle"],
                  p_native=native_ok(before, after, "p"), d_native=native_ok(before, after, "d"),
                  p_wrote_360_mib=producer["write_bytes"] == 20 * CHUNK_BYTES,
                  p_read_zero=producer["read_bytes"] == 0,
                  d_read_1440_mib=decoder["read_bytes"] == 80 * CHUNK_BYTES,
                  d_wrote_zero=decoder["write_bytes"] == 0,
                  p_sources=p_source == dict(local_compute=2746.0, external_kv_transfer=0.0, local_cache_hit=0.0),
                  d_sources=d_source == dict(local_compute=744.0, external_kv_transfer=10240.0, local_cache_hit=0.0),
                  initial_objects_40=first is not None and first["objects"] == 40,
                  final_objects_60=last is not None and last["objects"] == 60,
                  final_free_slots_4=last is not None and last.get("free_slots") == 4,
                  no_eviction=(first is not None and last is not None and first.get("evicted_objects") is not None
                               and last.get("evicted_objects") == first["evicted_objects"]))
    return dict(passed=all(checks.values()), checks=checks, p_sources=p_source, d_sources=d_source,
                p_actual_transfer=producer, d_actual_transfer=decoder,
                pool_before=first, pool_after=last,
                client_http_interval_overlap_observed=requests["client_http_interval_overlap_observed"],
                overlap_status="OBSERVED" if requests["client_http_interval_overlap_observed"] else "NOT_OBSERVED",
                dma_overlap_status="NOT_VERIFIED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--p-url", default="http://127.0.0.1:28401")
    parser.add_argument("--d-url", default="http://127.0.0.1:28402")
    parser.add_argument("--service-url", default="http://127.0.0.1:28195")
    parser.add_argument("--p-counter-url", default="http://127.0.0.1:29401/counters")
    parser.add_argument("--d-counter-url", default="http://127.0.0.1:29402/counters")
    parser.add_argument("--model", default="mf-night")
    parser.add_argument("--wait-seconds", type=float, default=90)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.wait_seconds <= 90:
        parser.error("wait-seconds must be in (0,90]")
    print(json.dumps(dict(mode="RUN" if args.run else "PLAN_ONLY", output_root=str(args.output_root),
                          initial_prefixes=4, p_new_requests=2, d_old_requests=8,
                          expected_p_write_bytes=20 * CHUNK_BYTES, expected_d_read_bytes=80 * CHUNK_BYTES,
                          total_objects=60, available_slots=64, no_fault_injection=True,
                          client_concurrency_is_not_dma_overlap=True), indent=2))
    if not args.run:
        return 0
    if args.output_root.exists():
        raise FileExistsError("Use a fresh mixed result directory")
    if not Path(args.tokenizer).is_dir():
        raise ValueError("Use an existing local tokenizer directory; downloads are not allowed")
    old = seed_check.dataset_prompts(args.dataset, True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    new, manifest = prepare_prompts(old, tokenizer)
    args.output_root.mkdir(parents=True)
    bench_d.dump(args.output_root / "data_manifest.json", manifest)
    (args.output_root / "new_prompts.jsonl").write_text('\n'.join(json.dumps(dict(prompt=value), ensure_ascii=False) for value in new) + '\n', encoding="utf-8")
    initial, initial_idle = wait_for(args, lambda _snapshot: True)
    bench_d.save_snapshot(args.output_root, "initial", initial)
    node = pool(initial)
    valid = (initial_idle["passed"] and node is not None and node["pool_bytes"] == 2 << 30
             and node["slot_size"] == 32 << 20 and node["objects"] in (0, 10, 20, 30, 40))
    if not valid:
        bench_d.dump(args.output_root / "summary.json", dict(passed=False, status="INITIAL_STATE_UNSUITABLE", idle=initial_idle, pool=node))
        return 2
    # Filling missing old prefixes is setup, outside the mixed measurement.
    if node["objects"] < OLD_OBJECTS:
        seeded = []
        for index, prompt in enumerate(old):
            response = post(args, "p", prompt, 1)
            seeded.append(response)
            bench_d.dump(args.output_root / f"seed_{index:02d}.json", response)
            if not response_ok(response, decode=False):
                break
        expected_seed_bytes = (OLD_OBJECTS - node["objects"]) * CHUNK_BYTES
        ready, ready_state = wait_for(args, lambda snap: (
            pool(snap) is not None and pool(snap)["objects"] == OLD_OBJECTS
            and transfers(initial, snap, "p")["write_bytes"] == expected_seed_bytes
            and source_total(initial, snap, "p") == len(seeded) * 1373))
        bench_d.save_snapshot(args.output_root, "seeded", ready)
        if len(seeded) != 4 or not all(response_ok(row, decode=False) for row in seeded) or not ready_state["passed"]:
            bench_d.dump(args.output_root / "summary.json", dict(passed=False, status="OLD_PREFIX_SEED_FAILED", ready=ready_state))
            return 2
    preflight_before, preflight_ready = wait_for(args, lambda snap: pool(snap) is not None and pool(snap)["objects"] == OLD_OBJECTS)
    bench_d.save_snapshot(args.output_root, "preflight_before", preflight_before)
    if not preflight_ready["passed"]:
        bench_d.dump(args.output_root / "summary.json", dict(passed=False, status="OLD_PREFIX_NOT_IDLE", ready=preflight_ready))
        return 2
    preflight = []
    for index, prompt in enumerate(old):
        response = post(args, "d", prompt, 8)
        preflight.append(response)
        bench_d.dump(args.output_root / f"preflight_d_{index:02d}.json", response)
        if not response_ok(response, decode=True):
            break
    before, preflight_state = wait_for(args, lambda snap: (
        source_delta(preflight_before, snap, "d") == dict(local_compute=372.0, external_kv_transfer=5120.0, local_cache_hit=0.0)
        and transfers(preflight_before, snap, "d")["read_bytes"] == 40 * CHUNK_BYTES))
    bench_d.save_snapshot(args.output_root, "before", before)
    if (not preflight_state["passed"] or len(preflight) != 4 or not all(response_ok(row, decode=True) for row in preflight)
            or pool(before)["objects"] != OLD_OBJECTS or not native_ok(preflight_before, before, "d")
            or not native_ok(preflight_before, before, "p") or source_total(preflight_before, before, "p") != 0):
        bench_d.dump(args.output_root / "summary.json", dict(passed=False, status="OLD_PREFIX_PREFLIGHT_FAILED", ready=preflight_state))
        return 2
    requests = run_mixed_http(args, old, new)
    bench_d.dump(args.output_root / "requests.json", requests)
    after, final_idle = wait_for(args, lambda snap: (
        pool(snap) is not None and pool(snap)["objects"] == FINAL_OBJECTS
        and transfers(before, snap, "p")["write_bytes"] == 20 * CHUNK_BYTES
        and transfers(before, snap, "d")["read_bytes"] == 80 * CHUNK_BYTES
        and source_delta(before, snap, "p") == dict(local_compute=2746.0, external_kv_transfer=0.0, local_cache_hit=0.0)
        and source_delta(before, snap, "d") == dict(local_compute=744.0, external_kv_transfer=10240.0, local_cache_hit=0.0)))
    bench_d.save_snapshot(args.output_root, "after", after)
    result = mixed_checks(before, after, requests)
    result.update(final_wait=final_idle, passed=result["passed"] and final_idle["passed"],
                  services_started_or_stopped=False, pool_protections_released_by_client=False,
                  scope="Mixed HTTP requests, actual completed copies and idle cleanup; no native overlap or performance claim")
    bench_d.dump(args.output_root / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
