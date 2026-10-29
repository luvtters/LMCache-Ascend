#!/usr/bin/env python3
"""Opt-in HTTP follow-ups against an already resident kv_consumer D engine.

Never starts P/proxy/services, edits cache state, or imports an accelerator SDK.
Every HTTP response body is retained verbatim; summaries are derived evidence.
Client concurrency two does not imply concurrent engine execution.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

from direct_pd_gate import l1_empty, metric


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def variants(base):
    partial = deepcopy(base)
    user = next(message for message in reversed(partial["messages"]) if message["role"] == "user")
    lines = user["content"].splitlines()
    if len(lines) < 4:
        raise ValueError("Gate user message needs multiple lines for a latter-half modification")
    user["content"] = "\n".join(lines[:len(lines) // 2] + [
        "Replacement appendix: " + str(i) + " lavender glass silver. The answer remains ORCHID."
        for i in range(len(lines) // 2)] + ["Reply with exactly ORCHID."])
    cold = deepcopy(base)
    cold["messages"] = [
        dict(role="system", content="Independent new document: return only ORCHID without commentary."),
        dict(role="user", content="\n".join(
            "Fresh amber document " + str(i) + ": orange silver lavender; access word ORCHID."
            for i in range(64)) + "\nReturn exactly ORCHID.")]
    return dict(warm=deepcopy(base), partial=partial, cold=cold)


def common_chunks(left, right, chunk_size):
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    return common // chunk_size


def http(url, directory, body=None, timeout=180, request_id=None):
    directory.mkdir(parents=True, exist_ok=True)
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    if raw is not None:
        (directory / "request.body.bin").write_bytes(raw)
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    started = time.monotonic_ns()
    metadata = dict(url=url, method="GET" if raw is None else "POST", request_headers=headers,
                    started_mono_ns=started, client_request_id=request_id)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        try:
            response = opener.open(urllib.request.Request(url, data=raw, headers=headers), timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            payload = response.read()
            metadata.update(status=response.status, response_headers=list(response.headers.items()),
                            returned_mono_ns=time.monotonic_ns(), response_bytes=len(payload))
            (directory / "response.body.bin").write_bytes(payload)
        save(directory / "http.raw.json", metadata)
        if not 200 <= metadata["status"] < 300:
            raise RuntimeError("HTTP failure at " + url + ": " + str(metadata["status"]))
        return payload, metadata
    except BaseException as error:
        metadata.update(error=repr(error), ended_mono_ns=time.monotonic_ns())
        save(directory / "http.raw.json", metadata)
        raise


def tokenize(args, body, folder):
    request = {key: body[key] for key in ("model", "messages", "chat_template_kwargs") if key in body}
    request["add_generation_prompt"] = True
    raw, _ = http(args.decoder + "/tokenize", folder, request, args.timeout)
    result = json.loads(raw)
    tokens = result["tokens"]
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int for token in tokens)
            or result.get("count", len(tokens)) != len(tokens)):
        raise ValueError("D /tokenize did not return exact token IDs and matching count")
    return tokens


def snapshot(args, folder):
    metrics, _ = http(args.decoder + "/metrics", folder / "metrics", timeout=args.timeout)
    status, _ = http(args.service_http + "/status", folder / "status", timeout=args.timeout)
    trace, _ = http(args.trace_url, folder / "trace", timeout=args.timeout)
    return dict(metrics=metrics.decode("utf-8"), status=json.loads(status), trace=json.loads(trace))


def check_status(args, before, after):
    for state in (before, after):
        assert state["chunk_size"] == args.chunk_size
        assert state["world_size"] == 1 + 2 * args.tp_size
        assert state["storage_manager"]["l2_adapters"] == [] and l1_empty(state)
        assert state["total_pins"] == state["pending_writes"] == state["quarantined_slots"] == 0
        assert state["transfer_ready"] and state["registration_ready"]
        if args.expected_pool_memory_type:
            assert state.get("pool_memory_type", "host") == args.expected_pool_memory_type
    assert before["epoch"] == after["epoch"] and before["objects"] == after["objects"]


def d_roster(trace, args):
    assert trace["run_id"] == args.session_id, "Trace session differs"
    records = [r for r in trace["records"] if r["role"] == "D"]
    assert all(r["host_id"] == trace["host_id"] and r["session_id"] == args.session_id for r in records)
    result = {}
    for record in records:
        if record["kind"] == "npu_scratch" and record["phase"] == "complete":
            shard = record.get("worker_shard", 0)
            identity = (record["pid"], record["local_ptr"], record["bytes"])
            assert shard not in result or result[shard] == identity, "Scratch changed within fixed group"
            assert record["memory_type"] == "npu" and record.get("tp_size", 1) == args.tp_size
            assert identity[2] > 0
            result[shard] = identity
    assert set(result) == set(range(args.tp_size)), "Missing D shard scratch identity"
    assert len({identity[0] for identity in result.values()}) == args.tp_size, "D shards are not distinct worker processes"
    return result


def trace_counts(args, before, after, chunks):
    assert before["host_id"] == after["host_id"], "Trace host changed"
    roster = d_roster(before, args)
    assert d_roster(after, args) == roster
    active = [r for r in after["records"] if r["mono_ns"] >= before["mono_ns"]]
    assert not any(r["role"] == "D" and r["phase"] == "error" for r in active)
    assert not any(r["role"] == "P" and r["kind"] == "bm_copy" for r in active), "P unexpectedly transferred"
    assert all(r.get("worker_shard", 0) in roster for r in active
               if r["role"] == "D" and r["kind"] == "bm_copy"), "Unexpected D shard transferred"
    result = {}
    for shard, (pid, pointer, chunk_bytes) in roster.items():
        copies = [r for r in active if r["role"] == "D" and r["kind"] == "bm_copy"
                  and r.get("worker_shard", 0) == shard]
        assert all(r["pid"] == pid and r["local_ptr"] == pointer and r["bytes"] == chunk_bytes
                   and r["direction"] == "G2L" and r.get("bm_rank", 2) == 1 + args.tp_size + shard
                   and r["memory_type"] == "npu"
                   for r in copies), "Wrong shard, pointer, direction, BM rank or actual chunk bytes"
        identity = lambda r: (r["pid"], r["thread_id"], r["src"], r["dst"], r["bytes"])
        begins = Counter(identity(r) for r in copies if r["phase"] == "begin")
        completed = [r for r in copies if r["phase"] == "complete"]
        assert begins == Counter(identity(r) for r in completed), "D BM transfer remains in flight"
        assert len(completed) == chunks, "Per-shard copy count does not match exact shared chunks"
        result[str(shard)] = dict(completed_copies=len(completed),
            completed_bytes=sum(r["bytes"] for r in completed), actual_chunk_bytes=chunk_bytes,
            pid=pid, bm_rank=1 + args.tp_size + shard)
    return result


def metric_deltas(before, after):
    source = "vllm:prompt_tokens_by_source_total"
    return dict(prompt=metric(after, "vllm:prompt_tokens_total") - metric(before, "vllm:prompt_tokens_total"),
        **{name: metric(after, source, 'source="' + label + '"') -
                  metric(before, source, 'source="' + label + '"')
           for name, label in (("external", "external_kv_transfer"), ("compute", "local_compute"),
                               ("local_prefix", "local_cache_hit"))})


def run_case(args, name, requests, folder):
    before = snapshot(args, folder / "before")
    check_status(args, before["status"], before["status"])
    d_roster(before["trace"], args)
    barrier = threading.Barrier(len(requests)) if len(requests) > 1 else None

    def send(index, item):
        if barrier:
            barrier.wait(timeout=10)
        raw, timing = http(args.decoder + "/v1/chat/completions", folder / ("request_" + str(index)),
                           item["body"], args.timeout, "followup-" + uuid.uuid4().hex)
        response = json.loads(raw)
        assert response["choices"][0]["message"]["content"].strip() == "ORCHID", "Unexpected D answer"
        assert response["usage"]["prompt_tokens"] == len(item["tokens"]), "D /tokenize and actual usage differ"
        return dict(response_request_id=response["id"], client_request_id=timing["client_request_id"],
            prompt_tokens=response["usage"]["prompt_tokens"], usage=response["usage"],
            started_mono_ns=timing["started_mono_ns"], returned_mono_ns=timing["returned_mono_ns"],
            shared_full_chunks=item["chunks"], variant=item["variant"], answer="ORCHID")

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        jobs = [executor.submit(send, index, item) for index, item in enumerate(requests)]
        responses = [job.result() for job in jobs]
    if len(requests) > 1:
        assert max(r["started_mono_ns"] for r in responses) < min(r["returned_mono_ns"] for r in responses), "Client intervals did not overlap"
    expected_chunks = sum(item["chunks"] for item in requests)
    expected_prompt = sum(len(item["tokens"]) for item in requests)
    expected_external = sum(min(item["chunks"] * args.chunk_size, len(item["tokens"]) - 1) for item in requests)
    save(folder / "response_identities.derived.json", responses)
    deadline = time.monotonic() + args.settle_timeout
    attempt = 0
    while True:
        after = snapshot(args, folder / ("after_" + str(attempt)))
        try:
            check_status(args, before["status"], after["status"])
            counts = metric_deltas(before["metrics"], after["metrics"])
            assert counts == dict(prompt=expected_prompt, external=expected_external,
                                  compute=expected_prompt - expected_external, local_prefix=0), counts
            per_shard = trace_counts(args, before["trace"], after["trace"], expected_chunks)
            break
        except AssertionError as error:
            save(folder / ("settle_" + str(attempt) + ".derived.json"), dict(error=repr(error)))
            if time.monotonic() >= deadline:
                raise
            attempt += 1
            time.sleep(.25)
    result = dict(status="FUNCTIONAL_FOLLOWUP_PASS", case=name, requests=responses,
        metrics=counts, per_d_shard=per_shard, pool_memory_type=after["status"].get("pool_memory_type", "host"),
        client_concurrency=len(requests), engine_concurrency="NOT_INFERRED_FROM_CLIENT_CONCURRENCY",
        evidence="Raw HTTP bodies, metrics, trace and status saved separately", model_lifecycle="NOT_TESTED")
    save(folder / "summary.derived.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--service-http", required=True)
    parser.add_argument("--trace-url", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--expected-pool-memory-type", choices=("host", "device"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--settle-timeout", type=float, default=25)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or args.timeout <= 0 or args.settle_timeout <= 0:
        parser.error("chunk size and timeouts must be positive")
    base = json.loads((args.gate_dir / "request.json").read_text(encoding="utf-8"))
    bodies = variants(base)
    plan = dict(status="DRY_RUN", scenarios=["warm_repeat", "partial_prefix", "cold_miss", "client_concurrent_2"],
        decoder=args.decoder, no_prefiller_or_proxy_requests=True,
        tokens="Require actual D /tokenize IDs at runtime; no frozen token-length substitution",
        engine_concurrency="Client concurrency only; D may have max_num_seqs=1")
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        token_ids = {name: tokenize(args, body, args.output / "tokenize" / name)
                     for name, body in bodies.items()}
        reference = token_ids["warm"]
        items = {name: dict(variant=name, body=body, tokens=token_ids[name],
                           chunks=common_chunks(reference, token_ids[name], args.chunk_size))
                 for name, body in bodies.items()}
        assert 0 < items["partial"]["chunks"] < items["warm"]["chunks"], "Modified suffix did not produce a partial full-chunk prefix"
        assert items["cold"]["chunks"] == 0, "Cold request shares a stored complete chunk"
        save(args.output / "token_comparison.derived.json", {name: dict(prompt_tokens=len(item["tokens"]),
            shared_full_chunks=item["chunks"]) for name, item in items.items()})
        results = []
        for name, selected in (("warm_repeat", ["warm"]), ("partial_prefix", ["partial"]),
                               ("cold_miss", ["cold"]), ("client_concurrent_2", ["warm", "partial"])):
            results.append(run_case(args, name, [items[v] for v in selected], args.output / name))
        save(args.output / "summary.derived.json", dict(status="FOLLOWUPS_FUNCTION_PASS", cases=results,
            lifecycle="Resident engines remain running; outer executor owns shutdown and real wait"))
    except BaseException as error:
        save(args.output / "failure.derived.json", dict(status="FAIL", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
