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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-http", default="http://127.0.0.1:28080")
    parser.add_argument("--prefiller", default="http://127.0.0.1:28001")
    parser.add_argument("--decoder", default="http://127.0.0.1:28002")
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--proxy-telemetry",
                        default="http://127.0.0.1:25768/api/v1/telemetry")
    parser.add_argument("--telemetry-port", type=int, default=25769)
    parser.add_argument("--model", default="mf-direct")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--require-direct-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.require_direct_trace and args.trace_dir is None:
        parser.error("--require-direct-trace needs --trace-dir")
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


def verify_trace(records, since_ns):
    """Check actual NPU pointers and completed copy/stage pairs, not API names."""
    installed = {r["role"] for r in records if r["kind"] == "trace_installed"}
    assert installed == {"P", "D"}, "Both real worker trace hooks must be active"
    scratch = {(r["role"], r["pid"], r["local_ptr"]): r for r in records
               if r["kind"] == "npu_scratch" and r["phase"] == "complete"}
    active = [r for r in records if r["mono_ns"] >= since_ns]
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
                    key=lambda r: r["mono_ns"])
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
    return dict(p_l2g_copies=len(stores), d_g2l_copies=len(loads),
                p_l2g_bytes=store_bytes, d_g2l_bytes=load_bytes,
                p_gather_completions=len(gathers), d_scatter_completions=len(scatters),
                actual_local_memory_type="npu", cpu_bm_copy_calls=0,
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
        assert status["storage_rank"] == 0 and status["world_size"] == 3, status
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

    @app.post("/api/v1/telemetry")
    async def on_store_finished(request: Request):
        async with gate_lock:
            payload = await request.json()
            if (payload.get("event") != "request_store_finished"
                    or len(payload.get("request_ids_set", [])) != 1
                    or payload.get("world_size") != 1 or payload.get("kv_rank") != 0):
                raise HTTPException(400, "Expected one TP1 store-finished request")
            if gate:
                if payload["request_ids_set"][0] != gate["p_request_id"]:
                    raise HTTPException(409, "Only one request is accepted")
                return gate["proxy_notification"]
            save("p_store_event", payload)
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
                    records = trace_records(args.trace_dir)
                    active = [r for r in records if r["mono_ns"] >= started_ns]
                    assert not any(r["kind"] == "bm_copy" and r["role"] == "D"
                                   for r in active), "D transferred before the gate opened"
                    if args.require_direct_trace:
                        assert any(r["kind"] == "bm_copy" and r["role"] == "P"
                                   and r["direction"] == "L2G" and r["phase"] == "complete"
                                   for r in active), "No actual P L2G completion"
                    save("before_decode", dict(pool=bm, service=status,
                                               mono_ns=before_ns, trace=records))
                    gate.update(p_request_id=payload["request_ids_set"][0],
                                objects_before_d=bm["objects"],
                                gate_open_mono_ns=before_ns)
                    print("MEMFABRIC_DIRECT_GATE_READY", json.dumps(gate), flush=True)
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
        app, host="127.0.0.1", port=args.telemetry_port, log_level="info"))
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
            save("initial", dict(pool=initial_pool, service=initial_http,
                                 trace=trace_records(args.trace_dir)))
            for role, engine in (("p", args.prefiller), ("d", args.decoder)):
                response = client.get(engine + "/metrics")
                response.raise_for_status()
                before[role] = response.text
                (args.output / f"{role}_metrics_before.txt").write_text(
                    response.text, encoding="utf-8")
            body = dict(model=args.model, temperature=0, max_tokens=32,
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
            chunks = prompt_tokens // args.chunk_size
            assert gate["objects_before_d"] == chunks > 0
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
            records = trace_records(args.trace_dir)
            save("worker_trace", records)
            trace_summary = verify_trace(records, started_ns) if args.require_direct_trace else None
            if trace_summary:
                assert trace_summary["p_l2g_copies"] == trace_summary["d_g2l_copies"] == chunks
            summary = dict(
                status="DIRECT_POOL_PATH_PASS" if trace_summary else "FUNCTIONAL_POOL_REUSE_PASS",
                p_request_id=gate["p_request_id"], d_request_id=result["id"],
                prompt_tokens=prompt_tokens, d_external_tokens=external,
                d_computed_tokens=computed, d_local_prefix_tokens=prefix,
                pool_objects_before_d=gate["objects_before_d"], answer="ORCHID",
                final_pool_pins=bm["total_pins"], manual_l1_clear=False,
                pool_observation_transport="http_status",
                direct_trace=trace_summary,
                direct_path_without_trace="NOT_VERIFIED" if trace_summary is None else None,
                note="Single-request gated correctness check; not a performance sample",
            )
            save("summary", summary)
            print("MEMFABRIC_DIRECT_PD_PASS", json.dumps(summary), flush=True)
    except Exception as exc:
        save("failure", dict(error=repr(exc), gate=gate))
        raise
    finally:
        server.should_exit = True
        if thread.ident is not None:
            thread.join(timeout=10)


if __name__ == "__main__":
    run(parse_args())
