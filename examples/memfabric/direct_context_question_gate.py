#!/usr/bin/env python3
"""Experimental context question + optional cold-reference gate.

Extra options: --request-json PATH --expected-answer CODE --cold-reference.
Without these options, delegate to the original gate unchanged. Custom requests
use actual tokenizer/usage/scratch geometry, never frozen 1373/180MiB values.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import threading
import time

import direct_pd_gate as stock
from direct_followups import http, metric_deltas, save


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--request-json", type=Path)
    extra_parser.add_argument("--expected-answer", default="ORCHID")
    extra_parser.add_argument("--cold-reference", action="store_true")
    extra, remaining = extra_parser.parse_known_args(argv)
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        args = stock.parse_args()
    finally:
        sys.argv = previous
    custom = extra.request_json is not None or extra.cold_reference or extra.expected_answer != "ORCHID"
    if custom:
        if extra.request_json is None or not extra.expected_answer:
            extra_parser.error("Experimental mode requires --request-json and a nonempty expected answer")
        if not args.p_trace_url or not args.d_trace_url or not args.session_id:
            extra_parser.error("Experimental mode requires both host-local trace URLs and --session-id")
        # Stock role launchers may still supply frozen-seed expectations. They
        # do not apply to the custom request and are explicitly discarded.
        args.expected_prompt_tokens = args.expected_copy_bytes = None
    return args, extra, custom


def validate_request(body, answer):
    contents = [message["content"] for message in body["messages"]]
    assert all(isinstance(content, str) for content in contents), "Text messages required"
    assert sum(content.count(answer) for content in contents) == 1, "Expected code must occur exactly once"
    assert not any(answer in message["content"] for message in body["messages"]
                   if message["role"] == "system"), "Code must not appear in the system instruction"
    assert body.get("temperature", 0) == 0, "Greedy request required"
    assert body.get("max_tokens", 0) > 0


def check_capacity(status, prompt_tokens, chunk_size, tp_size):
    chunks = prompt_tokens // chunk_size
    required = chunks * tp_size
    assert chunks >= 2, "Context question requires at least two complete cached chunks"
    if required > status["free_slots"]:
        raise ValueError("Custom prompt needs " + str(required) + " slots but only " +
                         str(status["free_slots"]) + " are free; shorten it before starting P")
    return chunks


def verify_code_chunks(first_decoded, later_decoded, answer):
    assert answer in first_decoded, "Entire unique code must be inside the first cached chunk"
    assert answer not in later_decoded, "Code must not be repeated in later input/tail"


def run_custom(args, extra):
    from fastapi import FastAPI, HTTPException, Request
    import uvicorn

    body = json.loads(extra.request_json.read_text(encoding="utf-8"))
    validate_request(body, extra.expected_answer)
    save(args.output / "request.json", body)

    def get_json(url, path, payload=None):
        raw, _ = http(url, path, payload, args.timeout)
        return json.loads(raw)

    def capture(label):
        folder = args.output / label
        metrics = {}
        records, hosts = [], {}
        for role, engine, trace_url in (("P", args.prefiller, args.p_trace_url),
                                         ("D", args.decoder, args.d_trace_url)):
            raw, _ = http(engine + "/metrics", folder / (role + "_metrics"), timeout=args.timeout)
            metrics[role] = raw.decode("utf-8")
            trace = get_json(trace_url, folder / (role + "_trace"))
            assert trace["run_id"] == args.session_id
            hosts[role] = {key: trace[key] for key in ("host_id", "run_id", "mono_ns")}
            records.extend(record for record in trace["records"] if record["role"] == role)
        status = get_json(args.service_http + "/status", folder / "status")
        return dict(metrics=metrics, records=records, hosts=hosts, status=status)

    def status_ok(status, *, empty=False):
        assert status["chunk_size"] == args.chunk_size
        assert status["world_size"] == 1 + 2 * args.tp_size
        assert status["transfer_ready"] and status["registration_ready"]
        assert status.get("pool_memory_type", "host") == args.expected_pool_memory_type
        assert stock.l1_empty(status) and status["storage_manager"]["l2_adapters"] == []
        assert status["total_pins"] == status["pending_writes"] == status["quarantined_slots"] == 0
        if empty:
            assert status["objects"] == 0, "Cold reference/preflight requires an empty pool"

    token_request = {key: body[key] for key in ("model", "messages", "chat_template_kwargs") if key in body}
    token_request["add_generation_prompt"] = True
    encoded = get_json(args.decoder + "/tokenize", args.output / "tokenize", token_request)
    tokens = encoded["tokens"]
    assert isinstance(tokens, list) and all(type(token) is int for token in tokens)
    assert encoded.get("count", len(tokens)) == len(tokens)
    first = get_json(args.decoder + "/detokenize", args.output / "first_chunk_decode",
                     dict(model=body["model"], tokens=tokens[:args.chunk_size]))
    later = get_json(args.decoder + "/detokenize", args.output / "later_chunks_decode",
                     dict(model=body["model"], tokens=tokens[args.chunk_size:]))
    verify_code_chunks(first["prompt"], later["prompt"], extra.expected_answer)
    preflight = capture("preflight")
    status_ok(preflight["status"], empty=True)
    chunks = check_capacity(preflight["status"], len(tokens), args.chunk_size, args.tp_size)
    save(args.output / "preflight.derived.json", dict(prompt_tokens=len(tokens), full_chunks=chunks,
        required_slots=chunks * args.tp_size, free_slots=preflight["status"]["free_slots"],
        code_in_first_chunk_only=True, frozen_seed_expectations_used=False))
    cold_answer = None
    causal = []
    if extra.cold_reference:
        cold = get_json(args.decoder + "/v1/chat/completions", args.output / "cold_reference" / "response", body)
        cold_answer = cold["choices"][0]["message"]["content"].strip()
        assert cold_answer == extra.expected_answer, "Cold computation did not answer the context question"
        assert cold["usage"]["prompt_tokens"] == len(tokens), "Actual cold usage differs from tokenizer"
        deadline, attempt = time.monotonic() + 25, 0
        while True:
            after_cold = capture("cold_reference/after_" + str(attempt))
            try:
                status_ok(after_cold["status"], empty=True)
                assert after_cold["status"]["epoch"] == preflight["status"]["epoch"]
                delta = metric_deltas(preflight["metrics"]["D"], after_cold["metrics"]["D"])
                assert delta == dict(prompt=len(tokens), external=0, compute=len(tokens), local_prefix=0), delta
                active = stock.active_trace(after_cold["records"], preflight["hosts"])
                assert not any(record["kind"] == "bm_copy" for record in active), "Cold reference used a BM transfer"
                assert not any(record["phase"] == "error" for record in active)
                break
            except AssertionError as error:
                save(args.output / "cold_reference" / ("settle_" + str(attempt) + ".derived.json"), dict(error=repr(error)))
                if time.monotonic() >= deadline:
                    raise
                attempt += 1
                time.sleep(.25)
        save(args.output / "cold_reference" / "summary.derived.json", dict(request_id=cold["id"],
            answer=cold_answer, usage=cold["usage"], metrics=delta, objects=0,
            bm_copies=0, boundary="Full computation reference, not cache reuse"))
        causal.append(dict(event="cold_reference_completed", request_id=cold["id"]))

    # Reset BOTH engine metrics and each host's independent monotonic cutoff.
    # The cold request is never subtracted using a guessed token count.
    baseline = capture("baseline_after_cold")
    status_ok(baseline["status"], empty=True)
    epoch = baseline["status"]["epoch"]
    stock.worker_roster(baseline["records"], args.tp_size)
    scratch = {}
    for record in baseline["records"]:
        if record["kind"] == "npu_scratch" and record["phase"] == "complete":
            scratch[(record["role"], record.get("worker_shard", 0))] = record["bytes"]
    assert set(scratch) == {(role, shard) for role in ("P", "D") for shard in range(args.tp_size)}
    assert all(type(size) is int and size > 0 for size in scratch.values())
    assert len(set(scratch.values())) == 1, "Homogeneous actual P/D scratch geometry required"
    expected_bytes = chunks * sum(scratch[("P", shard)] for shard in range(args.tp_size))
    causal.append(dict(event="metric_and_host_trace_baselines_reset", prompt_tokens=len(tokens),
                       chunks=chunks, expected_copy_bytes=expected_bytes))
    save(args.output / "baseline_geometry.derived.json", dict(
        scratch_bytes={role + str(shard): size for (role, shard), size in scratch.items()},
        expected_bytes_per_direction=expected_bytes, trace_baselines=baseline["hosts"]))

    app, lock = FastAPI(), asyncio.Lock()
    receipts, gate = {}, {}
    receipt_count = 0

    @app.post("/api/v1/telemetry")
    async def store_finished(request: Request):
        nonlocal receipt_count
        async with lock:
            raw = await request.body()
            folder = args.output / ("p_receipt_" + str(receipt_count))
            folder.mkdir()
            (folder / "request.body.bin").write_bytes(raw)
            receipt_count += 1
            payload = json.loads(raw)
            try:
                complete = stock.accept_store_receipt(receipts, payload, args.tp_size)
                save(args.output / "p_store_receipts.json", receipts)
                if gate:
                    return gate["proxy_notification"]
                if not complete:
                    return dict(notified=0, pending_worker_shards=sorted(set(range(args.tp_size)) - set(receipts)))
                deadline, attempt = time.monotonic() + 45, 0
                while True:
                    current = await asyncio.to_thread(capture, "before_decode_" + str(attempt))
                    try:
                        status_ok(current["status"])
                        assert current["status"]["epoch"] == epoch
                        assert current["status"]["objects"] == chunks * args.tp_size
                        active = stock.active_trace(current["records"], baseline["hosts"])
                        assert not any(r["role"] == "D" and r["kind"] == "bm_copy" for r in active)
                        counts = stock.rank_copy_summary(current["records"], baseline["hosts"], args.tp_size, "P", "L2G")
                        stock.verify_rank_counts(counts, chunks, expected_bytes, args.tp_size)
                        break
                    except AssertionError:
                        if time.monotonic() >= deadline:
                            raise
                        attempt += 1
                        await asyncio.sleep(.1)
                causal.append(dict(event="all_p_ranks_committed_before_d", epoch=epoch,
                                   objects=current["status"]["objects"], request_id=payload["request_ids_set"][0]))
                notification = await asyncio.to_thread(get_json, args.proxy_telemetry,
                                                       args.output / "proxy_notification", payload)
                assert notification.get("notified", 0) >= 1
                gate.update(p_request_id=payload["request_ids_set"][0], proxy_notification=notification)
                causal.append(dict(event="proxy_released_once_after_full_p_completion"))
                save(args.output / "causal_events.derived.json", causal)
                return notification
            except Exception as error:
                save(args.output / "callback_failure.derived.json", dict(error=repr(error)))
                raise HTTPException(502, str(error)) from error

    server = uvicorn.Server(uvicorn.Config(app, host=args.telemetry_bind,
                                          port=args.telemetry_port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    try:
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(.05)
        response = get_json(args.proxy + "/v1/chat/completions", args.output / "cached_response", body)
        answer = response["choices"][0]["message"]["content"].strip()
        assert answer == extra.expected_answer
        if extra.cold_reference:
            assert answer == cold_answer
        assert gate and response["usage"]["prompt_tokens"] == len(tokens)
        deadline, attempt = time.monotonic() + 25, 0
        while True:
            final = capture("after_decode_" + str(attempt))
            try:
                status_ok(final["status"])
                assert final["status"]["epoch"] == epoch and final["status"]["objects"] == chunks * args.tp_size
                d = metric_deltas(baseline["metrics"]["D"], final["metrics"]["D"])
                external = min(chunks * args.chunk_size, len(tokens) - 1)
                assert d == dict(prompt=len(tokens), external=external, compute=len(tokens) - external, local_prefix=0), d
                p = metric_deltas(baseline["metrics"]["P"], final["metrics"]["P"])
                assert p == dict(prompt=len(tokens), external=0, compute=len(tokens), local_prefix=0), p
                trace = stock.verify_trace(final["records"], baseline["hosts"], args.tp_size, chunks, expected_bytes)
                break
            except AssertionError as error:
                save(args.output / ("settle_" + str(attempt) + ".derived.json"), dict(error=repr(error)))
                if time.monotonic() >= deadline:
                    raise
                attempt += 1
                time.sleep(.25)
        save(args.output / "summary.derived.json", dict(status="CONTEXT_QUESTION_FUNCTION_PASS",
            expected_answer=extra.expected_answer, answer=answer, cold_reference_answer=cold_answer,
            cold_reference_requested=extra.cold_reference, prompt_tokens=response["usage"]["prompt_tokens"],
            usage=response["usage"], d_metrics=d, p_metrics=p, chunks=chunks,
            actual_expected_bytes=expected_bytes, direct_trace=trace, tp_size=args.tp_size,
            pool_memory_type=final["status"].get("pool_memory_type", "host"),
            p_request_id=gate["p_request_id"], d_request_id=response["id"],
            lifecycle="Outer executor must retain real P/D/service wait and release evidence",
            performance="NOT_MEASURED"))
    finally:
        server.should_exit = True
        if thread.ident is not None:
            thread.join(timeout=10)


def main(argv=None):
    args, extra, custom = parse(argv)
    if not custom:
        return stock.run(args)
    # Own a fresh directory before the exception recorder is enabled. A reused
    # output path must never overwrite evidence from an earlier attempt.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        return run_custom(args, extra)
    except BaseException as error:
        if args.output.exists():
            save(args.output / "failure.derived.json", dict(status="FAIL", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
