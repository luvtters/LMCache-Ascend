#!/usr/bin/env python3
"""Opt-in 500-request TP1 HTTP correctness check against a resident D.

250 serial requests then 250 with two client threads; engine max_num_seqs may
remain one. No engine launch, cache mutation, accelerator import or performance
claim. The outer executor owns the 600-second hard stop and real role waits.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import direct_followups as follow


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    def remaining(self):
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Absolute repeat-probe budget exhausted")
        return value


def schedule(send, checkpoint, budget):
    """Only 50 futures are queued at once; never more than two HTTP threads."""
    rows = []
    for index in range(250):
        budget.remaining()
        rows.append(send(index))
        if len(rows) % 50 == 0:
            checkpoint(len(rows), 1)
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        for start in range(250, 500, 50):
            budget.remaining()
            jobs = [executor.submit(send, index) for index in range(start, start + 50)]
            rows.extend(job.result(timeout=budget.remaining()) for job in jobs)
            assert all(job.done() for job in jobs), "Client futures remain pending"
            checkpoint(len(rows), 2)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return rows


def validate_response(response, expected_usage, expected_answer):
    assert response["choices"][0]["message"]["content"] == expected_answer, "D answer differs"
    assert response["usage"] == expected_usage, "Usage differs from original strong gate"
    assert isinstance(response.get("id"), str) and response["id"], "Missing response ID"
    return response["id"]


def native_counts(before, after, chunks, pointer):
    active = [r for r in after["records"] if r["mono_ns"] >= before["mono_ns"] and r["role"] == "D"]
    assert not any(r["phase"] == "error" for r in active)
    native = [r for r in active if r["kind"] == "native_transfer"]
    assert all(r["object_pointers"] == [pointer] for r in native), "Wrong native scratch"
    identity = lambda r: (r["pid"], r["thread_id"], tuple(r["object_pointers"]), r["direction"])
    begins = Counter(identity(r) for r in native if r["phase"] == "begin")
    returns = [r for r in native if r["phase"] == "returned"]
    assert begins == Counter(identity(r) for r in returns), "Native transfer has not returned"
    stages = [r for r in active if r["kind"] == "kv_stage" and r["phase"] == "complete"]
    assert len(returns) == len(stages) == chunks, "Missing native/scatter completion"
    assert all(r["direction"] == "scatter" and r["local_ptr"] == pointer for r in stages)
    return dict(native_returned=len(returns), scatter_completed=len(stages))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.duration_limit)
    lock = threading.Lock()
    state = dict(submitted=0, completed=0, pending=0, peak_pending=0, failed=0)
    response_ids = set()
    started_unix_ns = time.time_ns()
    follow.save(args.output / "run_config.json", dict(
        decoder=args.decoder, service_http=args.service_http, trace_url=args.trace_url,
        session_id=args.session_id, gate_dir=str(args.gate_dir), started_unix_ns=started_unix_ns,
        duration_limit_seconds=args.duration_limit, requests=500, first_serial=250,
        last_client_concurrency=2, outer_hard_stop_seconds=600,
        pending_scope="This HTTP probe only; internal engine RPC pending map is not exposed"))

    def get(url, folder, body=None, request_id=None):
        raw, meta = follow.http(url, folder, body, min(args.timeout, budget.remaining()), request_id)
        budget.remaining()
        return raw, meta

    def snapshot(folder, with_trace=False):
        metrics, _ = get(args.decoder + "/metrics", folder / "metrics")
        status, _ = get(args.service_http + "/status", folder / "status")
        result = dict(metrics=metrics.decode("utf-8"), status=json.loads(status))
        if with_trace:
            trace, _ = get(args.trace_url, folder / "trace")
            result["trace"] = json.loads(trace)
        return result

    def status_ok(before, after):
        follow.check_status(args, before, after)
        assert after["is_healthy"] and after["objects"] == 10, "Expected fixed ten-object healthy pool"
        assert after.get("active_prefetch_jobs", 0) == after.get("active_sessions", 0) == 0
        assert after.get("active_p2p_lookup_jobs", 0) == 0

    try:
        body = json.loads((args.gate_dir / "request.json").read_text(encoding="utf-8"))
        reference = json.loads((args.gate_dir / "cached_response/response.body.bin").read_bytes())
        usage = reference["usage"]
        validate_response(reference, usage, args.expected_answer)
        assert body["temperature"] == 0, "Repeat check requires the original greedy request"
        token_args = SimpleNamespace(decoder=args.decoder, timeout=min(args.timeout, budget.remaining()))
        tokens = follow.tokenize(token_args, body, args.output / "tokenize")
        budget.remaining()
        assert len(tokens) == usage["prompt_tokens"], "Actual tokenize differs from gate usage"
        chunks = len(tokens) // args.chunk_size
        assert chunks == 10 and len(tokens) % args.chunk_size, "Requires the ten-chunk strong gate"
        before = snapshot(args.output / "before", with_trace=True)
        status_ok(before["status"], before["status"])
        roster = follow.d_roster(before["trace"], args)
        checkpoints = []

        def send(index):
            budget.remaining()
            client_id = "repeat-" + uuid.uuid4().hex
            with lock:
                state["submitted"] += 1
                state["pending"] += 1
                state["peak_pending"] = max(state["peak_pending"], state["pending"])
            success = False
            try:
                raw, meta = get(args.decoder + "/v1/chat/completions",
                    args.output / "requests" / f"{index:04d}", body, client_id)
                response = json.loads(raw)
                response_id = validate_response(response, usage, args.expected_answer)
                with lock:
                    assert response_id not in response_ids, "Repeated response ID"
                    response_ids.add(response_id)
                success = True
                return dict(index=index, client_request_id=client_id, response_id=response_id,
                    usage=response["usage"], started_mono_ns=meta["started_mono_ns"],
                    returned_mono_ns=meta["returned_mono_ns"])
            finally:
                with lock:
                    state["pending"] -= 1
                    state["completed" if success else "failed"] += 1

        def checkpoint(count, concurrency):
            with lock:
                client = dict(state)
            assert client["pending"] == client["failed"] == 0
            assert client["submitted"] == client["completed"] == count
            assert client["peak_pending"] <= concurrency
            deadline = min(budget.deadline, time.monotonic() + 5)
            attempt = 0
            while True:
                folder = args.output / "checkpoints" / f"{count:04d}" / str(attempt)
                after = snapshot(folder)
                try:
                    status_ok(before["status"], after["status"])
                    counts = follow.metric_deltas(before["metrics"], after["metrics"])
                    external = count * chunks * args.chunk_size
                    assert counts == dict(prompt=count * len(tokens), external=external,
                        compute=count * len(tokens) - external, local_prefix=0), counts
                    break
                except AssertionError as error:
                    follow.save(folder / "not_settled.derived.json", dict(error=repr(error)))
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(min(.1, budget.remaining()))
                    attempt += 1
            row = dict(completed_requests=count, client_concurrency=concurrency,
                       client=client, metrics=counts, raw_folder=str(folder))
            follow.save(folder / "checkpoint.derived.json", row)
            checkpoints.append(row)

        rows = schedule(send, checkpoint, budget)
        after = snapshot(args.output / "after", with_trace=True)
        status_ok(before["status"], after["status"])
        counts = follow.metric_deltas(before["metrics"], after["metrics"])
        external = 500 * chunks * args.chunk_size
        assert counts == dict(prompt=500 * len(tokens), external=external,
                              compute=500 * len(tokens) - external, local_prefix=0), counts
        copies = follow.trace_counts(args, before["trace"], after["trace"], 500 * chunks)
        native = native_counts(before["trace"], after["trace"], 500 * chunks, roster[0][1])
        budget.remaining()
        result = dict(status="SHORT_REPEAT_FUNCTION_PASS", completed_requests=len(rows),
            clients=state, request_identities=rows, checkpoints=checkpoints, metrics=counts,
            d_bm_copies=copies, d_native=native, elapsed_seconds=time.monotonic() - budget.started,
            ended_unix_ns=time.time_ns(), engine_concurrency="NOT_INFERRED; max_num_seqs may be one",
            scope="Short HTTP correctness/protection sample; not long-term stability or performance",
            raw_trace_jsonl="Outer executor must retain original worker JSONL and real role waits")
        assert result["completed_requests"] == len(response_ids) == 500
        follow.save(args.output / "summary.derived.json", result)
        return result
    except BaseException as error:
        follow.save(args.output / "failure.derived.json", dict(status="FAIL", error=repr(error),
            client=state, elapsed_seconds=time.monotonic() - budget.started,
            started_unix_ns=started_unix_ns, ended_unix_ns=time.time_ns()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--service-http", required=True)
    parser.add_argument("--trace-url", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-answer", default="VELM42")
    parser.add_argument("--expected-pool-memory-type", choices=("host", "device"), default="device")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--duration-limit", type=float, default=570)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 15 or not 0 < args.duration_limit <= 570:
        parser.error("HTTP timeout must be <=15s; absolute budget <=570s within outer 600s guard")
    args.tp_size, args.chunk_size = 1, 128
    if not args.run:
        print(json.dumps(dict(status="PLAN_ONLY", requests=500, serial_requests=250,
            remaining_client_threads=2, checkpoint_every=50, total_budget=args.duration_limit,
            decoder=args.decoder, service_http=args.service_http, trace_url=args.trace_url,
            no_device_or_service_launch=True, original_worker_jsonl_required=True), indent=2))
        return
    print(json.dumps({k: v for k, v in run(args).items()
                      if k in ("status", "completed_requests", "metrics", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
