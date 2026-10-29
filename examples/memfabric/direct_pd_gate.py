# SPDX-License-Identifier: Apache-2.0
"""Single-request TP1 direct-pool gate for the existing upstream P/D proxy.

This sends real inference only when run. P telemetry must point at this
relay (:25769), while the upstream proxy telemetry remains at :25768.
It never clears L1, starts engines, stops services, or changes pool objects.
Pool observations use HTTP only; inference workers still use their real RPCs.
Use --require-direct-trace with traced_direct_connector.py for path evidence.
"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-http", default="http://127.0.0.1:28080")
    parser.add_argument("--prefiller", default="http://127.0.0.1:28001")
    parser.add_argument("--decoder", default="http://127.0.0.1:28002")
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--proxy-telemetry",
                        default="http://127.0.0.1:25768/api/v1/telemetry")
    parser.add_argument("--telemetry-port", type=int, default=25769)
    parser.add_argument("--telemetry-bind", default="127.0.0.1")
    parser.add_argument("--telemetry-advertise", help="Reachable full HTTP callback URL configured on P")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--p-trace-url", help="P host's CPU /snapshot endpoint")
    parser.add_argument("--d-trace-url", help="D host's CPU /snapshot endpoint")
    parser.add_argument("--session-id", help="Required for remote host-local trace snapshots")
    parser.add_argument("--expected-prompt-tokens", type=int)
    parser.add_argument("--expected-copy-bytes", type=int)
    parser.add_argument("--expected-pool-memory-type", choices=("host", "device"), default="host")
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--model", default="mf-direct")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--require-direct-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    remote = bool(args.p_trace_url or args.d_trace_url)
    if remote and not (args.p_trace_url and args.d_trace_url and args.session_id):
        parser.error("remote trace needs both trace URLs and --session-id")
    if args.require_direct_trace and args.trace_dir is None and not remote:
        parser.error("--require-direct-trace needs --trace-dir or both remote trace URLs")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.telemetry_advertise:
        parsed = urlparse(args.telemetry_advertise)
        if parsed.scheme != "http" or not parsed.hostname or parsed.hostname in ("0.0.0.0", "::"):
            parser.error("telemetry-advertise must be a reachable HTTP URL, not a wildcard")
    return args


def metric(text, name, label=None):
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if (line.startswith(name + "{") or line.startswith(name + " "))
              and (label is None or label in line)]
    assert values, ("Missing required metric", name, label)
    return sum(values)


def l1_empty(status):
    manager = status["storage_manager"]["l1_manager"]
    return manager["total_object_count"] == manager["memory_used_bytes"] == 0


def expected_external_tokens(prompt_tokens, chunk_size):
    """The scheduler leaves one prompt token for compute on an exact hit.

    BM still restores full chunks; the external-token metric reports the
    usable prefix after that final-token trim, not the number of bytes read.
    """
    chunks = prompt_tokens // chunk_size
    return min(chunks * chunk_size, prompt_tokens - 1)


def trace_records(directory):
    if directory is None:
        return []
    records = []
    for path in sorted(directory.glob("[PD]-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def active_trace(records, since_ns):
    """A dictionary contains independent cutoffs sampled on each worker host."""
    if isinstance(since_ns, dict):
        for record in records:
            baseline = since_ns[record["role"]]
            assert record.get("host_id") == baseline["host_id"], "Trace host changed"
            assert record.get("session_id") == baseline["run_id"], "Trace session changed"
        return [record for record in records
                if record["mono_ns"] >= since_ns[record["role"]]["mono_ns"]]
    return [record for record in records if record["mono_ns"] >= since_ns]


def worker_roster(records, tp_size):
    roster = {}
    for record in records:
        if record["kind"] != "trace_installed":
            continue
        shard = record.get("worker_shard", 0)
        assert record.get("tp_size", 1) == tp_size, "Worker TP size differs"
        assert record.get("vllm_worker_id", shard) == shard, "vLLM rank differs from shard"
        identity = (record.get("host_id"), record["pid"])
        key = (record["role"], shard)
        assert key not in roster or roster[key] == identity, "Duplicate worker for one shard"
        roster[key] = identity
    assert set(roster) == {(role, shard) for role in ("P", "D")
                          for shard in range(tp_size)}, "Missing real P/D worker shard"
    for role in ("P", "D"):
        assert len({identity for (worker_role, _), identity in roster.items()
                    if worker_role == role}) == tp_size, "One process cannot represent multiple TP shards"
    for record in records:
        if record["kind"] == "scheduler_step":
            continue  # Actual scheduler PID is not a TP worker identity.
        if tp_size > 1:
            assert all(field in record for field in ("worker_shard", "tp_size", "vllm_worker_id")), "Missing actual TP worker identity"
        key = (record["role"], record.get("worker_shard", 0))
        assert key in roster and roster[key] == (record.get("host_id"), record["pid"]), "Worker identity changed"
        if record["kind"] == "bm_copy":
            base = 1 if record["role"] == "P" else 1 + tp_size
            if tp_size > 1:
                assert "bm_rank" in record, "Missing actual BM rank"
            # TP1 old traces already carry bm_rank; retain compatibility with
            # standalone historical CPU trace fixtures that omitted it.
            assert record.get("bm_rank", base) == base + key[1], "BM rank differs from real worker shard"
    return roster


def rank_copy_summary(records, since_ns, tp_size, role, direction):
    worker_roster(records, tp_size)
    active = active_trace(records, since_ns)
    result = {}
    for shard in range(tp_size):
        copies = [record for record in active if record["kind"] == "bm_copy"
                  and record["role"] == role and record.get("worker_shard", 0) == shard]
        assert not any(record["phase"] == "error" for record in copies), "Rank transfer error"
        assert all(record["direction"] == direction for record in copies), "Unexpected rank direction"
        identity = lambda record: (record["local_ptr"], record["src"], record["dst"], record["bytes"])
        begins = Counter(identity(record) for record in copies if record["phase"] == "begin")
        done = [record for record in copies if record["phase"] == "complete"]
        assert begins == Counter(identity(record) for record in done), "Rank copy still in flight"
        assert done, "Worker shard completed no transfer"
        result[str(shard)] = dict(copies=len(done), bytes=sum(record["bytes"] for record in done),
                                  chunk_bytes=sorted({record["bytes"] for record in done}))
    return result


def verify_rank_counts(summary, chunks, total_bytes, tp_size):
    assert set(summary) == {str(shard) for shard in range(tp_size)}, "Missing rank counters"
    for item in summary.values():
        assert item["copies"] == chunks, "Per-rank chunk count differs"
        if total_bytes is not None:
            assert total_bytes % tp_size == 0 and item["bytes"] == total_bytes // tp_size, "Per-rank bytes differ"
            assert item["chunk_bytes"] == [total_bytes // tp_size // chunks], "Actual rank scratch/chunk bytes differ"


def accept_store_receipt(receipts, payload, tp_size):
    """Only a complete set of immutable real-rank receipts can open the gate."""
    if (payload.get("event") != "request_store_finished"
            or len(payload.get("request_ids_set", [])) != 1
            or payload.get("world_size") != tp_size
            or type(payload.get("kv_rank")) is not int
            or not 0 <= payload["kv_rank"] < tp_size):
        raise ValueError("Expected one request from a real TP worker shard")
    shard = payload["kv_rank"]
    for prior in receipts.values():
        if prior["request_ids_set"] != payload["request_ids_set"]:
            raise RuntimeError("Only one request is accepted")
    if shard in receipts and receipts[shard] != payload:
        raise RuntimeError("Conflicting immutable rank store receipt")
    receipts[shard] = payload
    return set(receipts) == set(range(tp_size))


def verify_trace(records, since_ns, tp_size=1, expected_chunks=None, expected_bytes=None):
    """Check actual NPU pointers and completed copy/stage pairs, not API names."""
    installed = {r["role"] for r in records if r["kind"] == "trace_installed"}
    assert installed == {"P", "D"}, "Both real worker trace hooks must be active"
    worker_roster(records, tp_size)
    scratch = {(r["role"], r["pid"], r["local_ptr"]): r for r in records
               if r["kind"] == "npu_scratch" and r["phase"] == "complete"}
    active = active_trace(records, since_ns)
    assert not any(r["kind"] == "cpu_bm_copy" for r in records), "CPU BM staging observed"
    assert not any(r["phase"] == "error" for r in active), "Transfer error in worker trace"
    copies = [r for r in active if r["kind"] == "bm_copy"]
    identity = lambda r: (r["role"], r["pid"], r["thread_id"], r["local_ptr"],
                          r["direction"], r["src"], r["dst"], r["bytes"])
    assert Counter(identity(r) for r in copies if r["phase"] == "begin") == Counter(
        identity(r) for r in copies if r["phase"] == "complete"), "BM copy still in flight"
    completed = [r for r in copies if r["phase"] == "complete"]
    for record in completed:
        allocation = scratch[(record["role"], record["pid"], record["local_ptr"])]
        assert allocation["memory_type"] == record["memory_type"] == "npu"
        assert allocation["bytes"] == record["bytes"] > 0
    # The first implementation is synchronous and uses one scratch allocation
    # on one owner thread. Match each nested call within that exact stream;
    # equal totals from unrelated workers or different chunks are insufficient.
    streams = {}
    events = sorted((r for r in active
                     if r["kind"] in ("kv_stage", "native_transfer", "bm_copy")),
                    key=lambda r: (r["role"], r["pid"], r["thread_id"], r["mono_ns"]))
    for record in events:
        stream = (record["role"], record["pid"], record["thread_id"])
        state = streams.setdefault(stream, dict(
            stage=None, native=None, copy=None, gathered=None, loaded=None))
        kind, phase = record["kind"], record["phase"]
        if kind == "native_transfer":
            stage = state["stage"]
            assert stage is not None and state["copy"] is None, "Native outside KV stage"
            expected = "D2H" if stage["direction"] == "gather" else "H2D"
            native_id = (tuple(record["object_pointers"]), record["direction"],
                         record["device"])
            assert record["object_pointers"] == [stage["local_ptr"]], "Native pointer differs"
            assert record["direction"].rsplit(".", 1)[-1] == expected, "Wrong native direction"
            assert record["device"] == stage["device"], "Native device differs"
            if phase == "begin":
                assert state["native"] is None and not stage["native_done"], "Duplicate native call"
                assert record["mono_ns"] > stage["mono_ns"], "Native precedes stage"
                state["native"] = (native_id, record["mono_ns"])
            else:
                assert phase == "returned" and state["native"] is not None, "Native return without begin"
                assert state["native"][0] == native_id, "Native return identity differs"
                assert record["mono_ns"] > state["native"][1], "Native return precedes call"
                state["native"] = None
                stage["native_done"] = True
            continue

        allocation = scratch[(record["role"], record["pid"], record["local_ptr"])]
        buffer = (record["local_ptr"], record["bytes"])
        assert allocation["memory_type"] == "npu" and allocation["bytes"] == record["bytes"]
        assert allocation["device"] == record["device"], "Stage/BM device differs"
        assert allocation["mono_ns"] < record["mono_ns"], "Copy precedes NPU allocation"
        if kind == "kv_stage":
            direction = record["direction"]
            assert direction in ("gather", "scatter"), direction
            if phase == "begin":
                assert state["stage"] is state["native"] is state["copy"] is None, "Overlapping stage/copy"
                assert state["gathered"] is None, "Gather overwritten before L2G"
                if direction == "scatter":
                    assert state["loaded"] is not None, "Scatter without completed G2L"
                    assert state["loaded"][0] == buffer, "Scatter uses another pointer"
                    assert record["mono_ns"] > state["loaded"][1], "Scatter precedes G2L completion"
                    state["loaded"] = None
                else:
                    assert state["loaded"] is None, "Loaded scratch overwritten before scatter"
                state["stage"] = dict(record, native_done=False)
            else:
                stage = state["stage"]
                assert phase == "complete" and stage is not None, "Stage completion without begin"
                assert state["native"] is None and stage["native_done"], "Native call did not return"
                assert (stage["direction"], stage["local_ptr"], stage["bytes"]) == (
                    direction, record["local_ptr"], record["bytes"]), "Stage identity differs"
                assert record["mono_ns"] > stage["mono_ns"], "Stage completion precedes begin"
                if direction == "gather":
                    state["gathered"] = (buffer, record["mono_ns"])
                state["stage"] = None
        else:
            direction = record["direction"]
            assert direction in ("L2G", "G2L"), direction
            if phase == "begin":
                assert state["stage"] is state["native"] is state["copy"] is None, "Overlapping BM/stage"
                assert state["loaded"] is None, "G2L scratch not yet scattered"
                if direction == "L2G":
                    assert state["gathered"] is not None, "L2G without completed gather"
                    assert state["gathered"][0] == buffer, "L2G uses another pointer"
                    assert record["mono_ns"] > state["gathered"][1], "L2G precedes gather completion"
                    state["gathered"] = None
                else:
                    assert state["gathered"] is None, "Gathered scratch overwritten before L2G"
                state["copy"] = record
            else:
                copy = state["copy"]
                assert phase == "complete" and copy is not None, "BM completion without begin"
                assert identity(copy) == identity(record), "BM completion identity differs"
                assert record["mono_ns"] > copy["mono_ns"], "BM completion precedes begin"
                if direction == "G2L":
                    state["loaded"] = (buffer, record["mono_ns"])
                state["copy"] = None
    assert all(all(value is None for value in state.values())
               for state in streams.values()), "Unpaired stage/native/copy remains"
    stores = [r for r in completed if r["role"] == "P" and r["direction"] == "L2G"]
    loads = [r for r in completed if r["role"] == "D" and r["direction"] == "G2L"]
    store_bytes, load_bytes = (sum(r["bytes"] for r in group) for group in (stores, loads))
    assert store_bytes == load_bytes > 0, (store_bytes, load_bytes)
    assert not any(r["role"] == "P" and r["direction"] == "G2L" for r in completed)
    stages = [r for r in active if r["kind"] == "kv_stage" and r["phase"] == "complete"]
    gathers = [r for r in stages if r["role"] == "P" and r["direction"] == "gather"]
    scatters = [r for r in stages if r["role"] == "D" and r["direction"] == "scatter"]
    assert len(gathers) == len(stores) and len(scatters) == len(loads)
    native = [r for r in active if r["kind"] == "native_transfer"]
    assert len([r for r in native if r["phase"] == "begin"]) == len(
        [r for r in native if r["phase"] == "returned"]) >= len(gathers) + len(scatters)
    per_rank = {}
    for role, direction in (("P", "L2G"), ("D", "G2L")):
        per_rank[role] = rank_copy_summary(records, since_ns, tp_size, role, direction)
        chunks = expected_chunks if expected_chunks is not None else len(stores) // tp_size
        verify_rank_counts(per_rank[role], chunks, expected_bytes, tp_size)
    return dict(p_l2g_copies=len(stores), d_g2l_copies=len(loads),
                p_l2g_bytes=store_bytes, d_g2l_bytes=load_bytes,
                p_gather_completions=len(gathers), d_scatter_completions=len(scatters),
                actual_local_memory_type="npu", cpu_bm_copy_calls=0,
                tp_size=tp_size, per_rank=per_rank,
                boundary="Python/native-call trace; not a hardware DMA profiler")


def run(args):
    # Delay runtime imports so --help works on a local CPU documentation host.
    from fastapi import FastAPI, HTTPException, Request
    import httpx
    import uvicorn

    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def pool_status():
        # MPCacheServer flattens each module's report_status into /status.
        # Avoid loading optional native transfer backends in this observer.
        with httpx.Client(timeout=5, trust_env=False) as observer:
            response = observer.get(args.service_http + "/status")
            response.raise_for_status()
            status = response.json()
        required = ("epoch", "transfer_ready", "registration_ready", "storage_rank",
                    "world_size", "objects", "pending_writes", "total_pins",
                    "quarantined_slots")
        assert all(name in status for name in required), status
        assert status["transfer_ready"] and status["registration_ready"], status
        assert status["storage_rank"] == 0 and status["world_size"] == 1 + 2 * args.tp_size, status
        assert status.get("pool_memory_type", "host") == args.expected_pool_memory_type, "Unexpected pool medium"
        if args.tp_size > 1:
            assert status["tp_size"] == args.tp_size, status
        return status

    def check_http_status(status):
        assert status["chunk_size"] == args.chunk_size, status["chunk_size"]
        assert status["storage_manager"]["l2_adapters"] == [], "Direct test must not configure L2"
        assert l1_empty(status), "Ordinary L1 contains payload objects"

    app = FastAPI()
    gate = {}
    gate_lock = asyncio.Lock()
    started_ns = time.monotonic_ns()
    epoch = None
    trace_baseline = started_ns
    trace_hosts = {}
    causal_events = []
    store_receipts = {}

    def read_traces(*, baseline=False):
        nonlocal trace_baseline
        if not args.p_trace_url:
            return trace_records(args.trace_dir)
        from direct_coordination import http_json
        records = []
        for role, url in (("P", args.p_trace_url), ("D", args.d_trace_url)):
            snapshot = http_json(url)
            assert snapshot["run_id"] == args.session_id, "Remote trace session mismatch"
            if baseline:
                trace_hosts[role] = {key: snapshot[key] for key in ("host_id", "run_id", "mono_ns")}
            else:
                assert snapshot["host_id"] == trace_hosts[role]["host_id"], "Remote host changed"
            records.extend(record for record in snapshot["records"] if record["role"] == role)
            save("trace_snapshot_" + role + ("_baseline" if baseline else "_latest"), snapshot)
        trace_baseline = trace_hosts
        return records

    @app.post("/api/v1/telemetry")
    async def on_store_finished(request: Request):
        async with gate_lock:
            payload = await request.json()
            try:
                all_receipts = accept_store_receipt(store_receipts, payload, args.tp_size)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
            except RuntimeError as error:
                raise HTTPException(409, str(error)) from error
            save("p_store_receipts", store_receipts)
            if gate:
                if payload["request_ids_set"][0] != gate["p_request_id"]:
                    raise HTTPException(409, "Only one request is accepted")
                return gate["proxy_notification"]
            # The upstream proxy wakes on ANY telemetry, without TP aggregation.
            # Acknowledge early ranks locally; forward exactly once only after
            # every distinct real shard completed and the pool gate below passes.
            if not all_receipts:
                return dict(notified=0, pending_worker_shards=sorted(set(range(args.tp_size)) - set(store_receipts)))
            save("p_store_event", payload)
            causal_events.append(dict(sequence=len(causal_events), event="p_store_finished_received",
                                      request_id=payload["request_ids_set"][0]))
            try:
                deadline = time.monotonic() + 45
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    while True:
                        bm = await asyncio.to_thread(pool_status)
                        response = await client.get(args.service_http + "/status")
                        response.raise_for_status()
                        status = response.json()
                        check_http_status(status)
                        assert bm["epoch"] == epoch
                        if (bm["objects"] > 0 and bm["pending_writes"] == 0
                                and bm["quarantined_slots"] == 0):
                            break
                        if time.monotonic() >= deadline:
                            save("gate_timeout", dict(pool=bm, service=status))
                            raise TimeoutError("Direct commits did not settle")
                        await asyncio.sleep(0.05)
                    before_ns = time.monotonic_ns()
                    records = await asyncio.to_thread(read_traces)
                    active = active_trace(records, trace_baseline)
                    assert not any(r["kind"] == "bm_copy" and r["role"] == "D"
                                   for r in active), "D transferred before the gate opened"
                    if args.require_direct_trace:
                        p_counts = rank_copy_summary(records, trace_baseline, args.tp_size, "P", "L2G")
                        expected_chunks = (args.expected_prompt_tokens // args.chunk_size
                                           if args.expected_prompt_tokens is not None
                                           else p_counts["0"]["copies"])
                        verify_rank_counts(p_counts, expected_chunks, args.expected_copy_bytes, args.tp_size)
                        assert bm["objects"] == expected_chunks * args.tp_size, "Pool missing committed shards"
                    save("before_decode", dict(pool=bm, service=status,
                                               mono_ns=before_ns, trace=records))
                    causal_events.append(dict(sequence=len(causal_events), event="p_commits_confirmed",
                                              epoch=bm["epoch"], objects=bm["objects"],
                                              pending_writes=bm["pending_writes"]))
                    gate.update(p_request_id=payload["request_ids_set"][0],
                                objects_before_d=bm["objects"],
                                worker_shards=sorted(store_receipts),
                                gate_open_mono_ns=before_ns)
                    print("MEMFABRIC_DIRECT_GATE_READY", json.dumps(gate), flush=True)
                    causal_events.append(dict(sequence=len(causal_events), event="release_d_notification_sent"))
                    save("causal_gate_events", causal_events)
                    response = await client.post(args.proxy_telemetry, json=payload)
                    response.raise_for_status()
                    notification = response.json()
                    save("proxy_notification", notification)
                    assert notification.get("notified") == 1, notification
                    gate["proxy_notification"] = notification
                    return notification
            except Exception as exc:
                save("gate_error", dict(error=repr(exc)))
                raise HTTPException(502, str(exc)) from exc

    server = uvicorn.Server(uvicorn.Config(
        app, host=args.telemetry_bind, port=args.telemetry_port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    before, after = {}, {}
    try:
        with httpx.Client(timeout=args.timeout, trust_env=False) as client:
            for engine in (args.prefiller, args.decoder):
                response = client.get(engine + "/health")
                response.raise_for_status()
            response = client.get(args.service_http + "/status")
            response.raise_for_status()
            initial_http, initial_pool = response.json(), pool_status()
            check_http_status(initial_http)
            assert initial_pool["objects"] == initial_pool["pending_writes"] == 0
            assert initial_pool["total_pins"] == initial_pool["quarantined_slots"] == 0
            epoch = initial_pool["epoch"]
            initial_records = read_traces(baseline=True)
            save("initial", dict(pool=initial_pool, service=initial_http,
                                 trace=initial_records, trace_baseline=trace_baseline,
                                 telemetry_bind=args.telemetry_bind,
                                 telemetry_advertise=args.telemetry_advertise,
                                 max_tokens=args.max_tokens))
            for role, engine in (("p", args.prefiller), ("d", args.decoder)):
                response = client.get(engine + "/metrics")
                response.raise_for_status()
                before[role] = response.text
                (args.output / f"{role}_metrics_before.txt").write_text(
                    response.text, encoding="utf-8")
            body = dict(model=args.model, temperature=0, max_tokens=args.max_tokens,
                        chat_template_kwargs={"enable_thinking": False}, messages=[
                            dict(role="system", content="Reply with exactly ORCHID."),
                            dict(role="user", content="\n".join(
                                f"Record {i:03d}: the access word is ORCHID. "
                                "This line exists for cache validation." for i in range(64))
                                 + "\nGive the access word.")])
            save("request", body)
            thread.start()
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.05)
            response = client.post(args.proxy + "/v1/chat/completions", json=body)
            response.raise_for_status()
            result = response.json()
            save("response", result)
            assert gate, "P telemetry never passed through the direct gate"
            assert result["choices"][0]["message"]["content"].strip() == "ORCHID", result
            prompt_tokens = result["usage"]["prompt_tokens"]
            if args.expected_prompt_tokens is not None:
                assert prompt_tokens == args.expected_prompt_tokens, (prompt_tokens, args.expected_prompt_tokens)
            chunks = prompt_tokens // args.chunk_size
            assert gate["objects_before_d"] == chunks * args.tp_size > 0
            def delta(role, name, label=None):
                return metric(after[role], name, label) - metric(before[role], name, label)

            source = "vllm:prompt_tokens_by_source_total"
            deadline = time.monotonic() + 20
            snapshots = []
            while True:
                for role, engine in (("p", args.prefiller), ("d", args.decoder)):
                    response = client.get(engine + "/metrics")
                    response.raise_for_status()
                    after[role] = response.text
                response = client.get(args.service_http + "/status")
                response.raise_for_status()
                status, bm = response.json(), pool_status()
                check_http_status(status)
                snapshots.append(dict(pool=bm, service=status))
                try:
                    counts = dict(
                        d_prompt=delta("d", "vllm:prompt_tokens_total"),
                        p_computed=delta("p", source, 'source="local_compute"'),
                        p_external=delta("p", source, 'source="external_kv_transfer"'),
                        p_generation=delta("p", "vllm:generation_tokens_total"),
                        d_external=delta("d", source, 'source="external_kv_transfer"'),
                        d_computed=delta("d", source, 'source="local_compute"'),
                        d_prefix=delta("d", source, 'source="local_cache_hit"'),
                    )
                    metrics_ready = (
                        counts["d_prompt"] == counts["p_computed"] == prompt_tokens
                        and counts["p_external"] == counts["d_prefix"] == 0
                        and counts["p_generation"] == 1
                        and counts["d_external"] == expected_external_tokens(
                            prompt_tokens, args.chunk_size)
                        and counts["d_computed"] + counts["d_external"] == prompt_tokens)
                except AssertionError as exc:
                    counts, metrics_ready = dict(not_yet_published=str(exc)), False
                if metrics_ready and bm["total_pins"] == bm["pending_writes"] == 0:
                    break
                if time.monotonic() >= deadline:
                    save("settle_timeout", dict(metrics=counts, pool=bm))
                    raise TimeoutError("P/D metrics or pool protections did not settle")
                time.sleep(0.2)
            save("after_decode", snapshots)
            for role, text in after.items():
                (args.output / f"{role}_metrics_after.txt").write_text(text, encoding="utf-8")

            external = delta("d", source, 'source="external_kv_transfer"')
            computed = delta("d", source, 'source="local_compute"')
            prefix = delta("d", source, 'source="local_cache_hit"')
            assert delta("p", source, 'source="local_compute"') == prompt_tokens
            assert delta("p", source, 'source="external_kv_transfer"') == 0
            assert delta("p", "vllm:generation_tokens_total") == 1
            assert external == expected_external_tokens(prompt_tokens, args.chunk_size) > 0
            assert computed + external == prompt_tokens and prefix == 0
            records = read_traces()
            save("worker_trace", records)
            trace_summary = verify_trace(records, trace_baseline, args.tp_size, chunks,
                                         args.expected_copy_bytes) if args.require_direct_trace else None
            if trace_summary:
                assert trace_summary["p_l2g_copies"] == trace_summary["d_g2l_copies"] == chunks * args.tp_size
                if args.expected_copy_bytes is not None:
                    assert trace_summary["p_l2g_bytes"] == args.expected_copy_bytes, trace_summary
            summary = dict(
                status="DIRECT_POOL_PATH_PASS" if trace_summary else "FUNCTIONAL_POOL_REUSE_PASS",
                p_request_id=gate["p_request_id"], d_request_id=result["id"],
                prompt_tokens=prompt_tokens, d_external_tokens=external,
                d_computed_tokens=computed, d_local_prefix_tokens=prefix,
                pool_objects_before_d=gate["objects_before_d"], answer="ORCHID",
                final_pool_pins=bm["total_pins"], manual_l1_clear=False,
                pool_observation_transport="http_status",
                direct_trace=trace_summary,
                causal_gate_events=causal_events,
                trace_time_domain="per_host_baselines" if args.p_trace_url else "local_host",
                max_tokens=args.max_tokens,
                tp_size=args.tp_size,
                pool_memory_type=bm.get("pool_memory_type", "host"),
                pool_selected_sdk_memory_type=bm.get("pool_selected_sdk_memory_type", "HOST"),
                bm_declared_local_dram_bytes=bm.get("bm_declared_local_dram_bytes"),
                bm_declared_local_hbm_bytes=bm.get("bm_declared_local_hbm_bytes"),
                direct_path_without_trace="NOT_VERIFIED" if trace_summary is None else None,
                note="Single-request gated correctness check; not a performance sample",
            )
            save("summary", summary)
            print("MEMFABRIC_DIRECT_PD_PASS", json.dumps(summary), flush=True)
    except Exception as exc:
        save("failure", dict(error=repr(exc), gate=gate, worker_store_receipts=store_receipts,
                             missing_worker_shards=sorted(set(range(args.tp_size)) - set(store_receipts))))
        raise
    finally:
        server.should_exit = True
        if thread.ident is not None:
            thread.join(timeout=10)


if __name__ == "__main__":
    run(parse_args())
