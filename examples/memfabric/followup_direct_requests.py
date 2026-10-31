# SPDX-License-Identifier: Apache-2.0
"""After strict gate PASS, replay its prompt to D: twice serial, then two concurrent.

This experiment sends four real inference requests to an existing D. It never
starts/stops services, clears caches, issues pool-control mutations, or uses SSH.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from direct_pd_gate import (active_trace, l1_empty, metric, rank_copy_summary,
                            trace_records, verify_rank_counts)


def validate_snapshot(snapshot, role, session_id, baseline=None):
    assert snapshot['run_id'] == session_id, 'Remote followup trace session mismatch'
    assert isinstance(snapshot['host_id'], str) and snapshot['host_id'], 'Missing remote host identity'
    assert type(snapshot['mono_ns']) is int and snapshot['mono_ns'] > 0
    if baseline is not None:
        assert snapshot['host_id'] == baseline['host_id'], 'Remote followup trace host changed'
        assert snapshot['run_id'] == baseline['run_id']
        assert snapshot['mono_ns'] >= baseline['mono_ns'], 'Remote trace clock moved backwards'
    records = [r for r in snapshot['records'] if r['role'] == role]
    assert all(r.get('host_id') == snapshot['host_id'] and r.get('session_id') == session_id
               for r in records), 'Remote record host/session mismatch'
    return records


def collect_remote_snapshots(fetch, urls, session_id, baselines=None):
    records, snapshots = [], {}
    cutoffs = {} if baselines is None else dict(baselines)
    for role in ('P', 'D'):
        snapshot = fetch(urls[role])
        records.extend(validate_snapshot(snapshot, role, session_id,
                       None if baselines is None else baselines[role]))
        snapshots[role] = snapshot
        if baselines is None:
            cutoffs[role] = {key: snapshot[key] for key in ('host_id', 'run_id', 'mono_ns')}
    return records, cutoffs, snapshots


def assert_no_new_p_traffic(records, since_ns):
    active = active_trace(records, since_ns)
    assert not any(r['role'] == 'P' and r['kind'] in ('bm_copy', 'kv_stage', 'native_transfer')
                   for r in active), 'Unexpected P traffic after followup P-host baseline'
    return 0


def expected_followup_trace(prompt_tokens, chunk, requests, tp_size, bytes_per_request):
    assert tp_size in (1, 2, 4, 8) and requests > 0 and chunk > 0
    chunks = prompt_tokens // chunk
    assert chunks > 0 and bytes_per_request > 0 and bytes_per_request % (chunks * tp_size) == 0
    return dict(copies=chunks * requests * tp_size, bytes=bytes_per_request * requests,
                per_rank_copies=chunks * requests,
                per_rank_bytes=bytes_per_request * requests // tp_size)


def verify_d_trace(records, since_ns, expected_copies, tp_size=1, expected_bytes=None):
    """Match completed D G2L-to-scatter chains on the same owner and NPU pointer."""
    scratch = {(r["pid"], r["local_ptr"]): r for r in records
               if r["role"] == "D" and r["kind"] == "npu_scratch"
               and r["phase"] == "complete"}
    active = active_trace(records, since_ns)
    assert not any(r["kind"] == "cpu_bm_copy" for r in active), "CPU BM copy observed"
    assert not any(r["phase"] == "error" for r in active), "Worker trace contains an error"
    assert not any(r["role"] == "P" and r["kind"] in ("bm_copy", "kv_stage")
                   for r in active), "Unexpected P traffic during D-only replay"
    events = sorted((r for r in active if r["role"] == "D"
                     and r["kind"] in ("bm_copy", "kv_stage", "native_transfer")),
                    key=lambda r: r["mono_ns"])
    streams, completed, read_bytes = {}, 0, 0
    for record in events:
        key = (record["pid"], record["thread_id"])
        state = streams.setdefault(key, dict(copy=None, loaded=None,
                                             stage=None, native=None))
        kind, phase = record["kind"], record["phase"]
        if kind == "native_transfer":
            stage = state["stage"]
            assert stage is not None, "Native call outside scatter"
            identity = (tuple(record["object_pointers"]), record["direction"],
                        record["device"])
            assert record["object_pointers"] == [stage["local_ptr"]]
            assert record["direction"].rsplit(".", 1)[-1] == "H2D"
            assert record["device"] == stage["device"]
            if phase == "begin":
                assert state["native"] is None and not stage["native_done"]
                assert record["mono_ns"] > stage["mono_ns"]
                state["native"] = (identity, record["mono_ns"])
            else:
                assert phase == "returned" and state["native"] is not None
                assert state["native"][0] == identity
                assert record["mono_ns"] > state["native"][1]
                stage["native_done"], state["native"] = True, None
            continue

        allocation = scratch[(record["pid"], record["local_ptr"])]
        buffer = (record["local_ptr"], record["bytes"], record["device"])
        assert allocation["memory_type"] == "npu"
        assert allocation["bytes"] == record["bytes"] > 0
        assert allocation["device"] == record["device"]
        assert allocation["mono_ns"] < record["mono_ns"]
        if kind == "bm_copy":
            assert record["direction"] == "G2L", "Warm D unexpectedly wrote to pool"
            assert record["memory_type"] == "npu"
            identity = (buffer, record["src"], record["dst"], record["bm_rank"])
            if phase == "begin":
                assert all(value is None for value in state.values())
                state["copy"] = (identity, record["mono_ns"])
            else:
                assert phase == "complete" and state["copy"] is not None
                assert state["copy"][0] == identity
                assert record["mono_ns"] > state["copy"][1]
                state["copy"] = None
                state["loaded"] = (buffer, record["mono_ns"])
                read_bytes += record["bytes"]
        else:
            assert record["direction"] == "scatter", "Warm D unexpectedly gathered KV"
            if phase == "begin":
                assert state["stage"] is state["copy"] is state["native"] is None
                assert state["loaded"] is not None and state["loaded"][0] == buffer
                assert record["mono_ns"] > state["loaded"][1]
                state["loaded"] = None
                state["stage"] = dict(record, native_done=False)
            else:
                stage = state["stage"]
                assert phase == "complete" and stage is not None
                assert state["native"] is None and stage["native_done"]
                assert buffer == (stage["local_ptr"], stage["bytes"], stage["device"])
                assert record["mono_ns"] > stage["mono_ns"]
                state["stage"] = None
                completed += 1
    assert all(all(value is None for value in state.values())
               for state in streams.values()), "Incomplete D copy/scatter chain"
    assert completed == expected_copies > 0, (completed, expected_copies)
    per_rank = None
    if expected_bytes is not None:
        assert read_bytes == expected_bytes, (read_bytes, expected_bytes)
        assert expected_copies % tp_size == 0
        per_rank = rank_copy_summary(records, since_ns, tp_size, 'D', 'G2L')
        verify_rank_counts(per_rank, expected_copies // tp_size, expected_bytes, tp_size)
    return dict(g2l_copies=completed, scatter_completions=completed,
                g2l_bytes=read_bytes, cpu_bm_copy_calls=0,
                local_memory_type="npu", new_p_copies=0, tp_size=tp_size, per_rank=per_rank)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--decoder", default="http://127.0.0.1:28202")
    parser.add_argument("--service-http", default="http://127.0.0.1:28175")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--p-trace-url", help="P host CPU /snapshot endpoint")
    parser.add_argument("--d-trace-url", help="D host CPU /snapshot endpoint")
    parser.add_argument("--session-id", help="Exact existing strict-gate session for remote snapshots")
    parser.add_argument("--model", default="mf-direct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    remote = bool(args.p_trace_url or args.d_trace_url)
    if remote and not (args.p_trace_url and args.d_trace_url and args.session_id):
        parser.error('remote trace requires paired --p-trace-url/--d-trace-url and --session-id')
    if not remote and args.trace_dir is None:
        parser.error('local trace requires --trace-dir')
    if remote and args.trace_dir is not None:
        parser.error('choose local trace-dir or paired remote snapshots')
    return args


def main(argv=None):
    args = arguments(argv)
    gate = json.loads((args.gate_dir / "summary.json").read_text(encoding="utf-8"))
    assert gate["status"] == "DIRECT_POOL_PATH_PASS", "Strict gate has not passed"
    initial = json.loads((args.gate_dir / "initial.json").read_text(encoding="utf-8"))
    epoch, chunk = initial["pool"]["epoch"], initial["service"]["chunk_size"]
    body = json.loads((args.gate_dir / "request.json").read_text(encoding="utf-8"))
    body.update(model=args.model, stream=False)
    body.pop("stream_options", None)
    expected = dict(prompt=gate["prompt_tokens"], external=gate["d_external_tokens"],
                    computed=gate["d_computed_tokens"], prefix=0)
    tp_size = gate.get('tp_size', 1)
    bytes_per_request = gate['direct_trace']['d_g2l_bytes']
    assert bytes_per_request == gate['direct_trace']['p_l2g_bytes'] > 0
    assert expected["external"] > 0 and gate["d_local_prefix_tokens"] == 0
    assert expected["computed"] + expected["external"] == expected["prompt"]
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import httpx

    def get(client, path):
        response = client.get(path)
        response.raise_for_status()
        return response

    def check_service(status):
        assert status["epoch"] == epoch and status["transfer_ready"]
        assert status["registration_ready"]
        assert status["chunk_size"] == chunk
        assert status["storage_manager"]["l2_adapters"] == []
        assert l1_empty(status), "L1 contains payload objects"
        assert status["quarantined_slots"] == 0
        return status["pending_writes"] == status["total_pins"] == 0

    def sources(text):
        source = "vllm:prompt_tokens_by_source_total"
        return dict(
            prompt=metric(text, "vllm:prompt_tokens_total"),
            external=metric(text, source, 'source="external_kv_transfer"'),
            computed=metric(text, source, 'source="local_compute"'),
            prefix=metric(text, source, 'source="local_cache_hit"'),
        )

    def request(index):
        started = time.monotonic_ns()
        with httpx.Client(timeout=180, trust_env=False) as client:
            response = client.post(args.decoder + "/v1/chat/completions", json=body)
            response.raise_for_status()
            result = response.json()
        assert result["choices"][0]["message"]["content"].strip() == "ORCHID", result
        assert result["usage"]["prompt_tokens"] == expected["prompt"], result["usage"]
        return dict(index=index, started_ns=started, completed_ns=time.monotonic_ns(),
                    request_id=result["id"], answer="ORCHID", usage=result["usage"],
                    response=result)

    receipt = dict(status="RUNNING", gate_dir=str(args.gate_dir), expected=expected,
                   decoder=args.decoder, service_http=args.service_http, phases=[], tp_size=tp_size,
                   trace_time_domain='per_host_baselines' if args.p_trace_url else 'local_client_monotonic',
                   session_id=args.session_id, bytes_per_request=bytes_per_request)
    save("request", body)
    save("receipt", receipt)
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            for phase, count in (("sequential_1", 1), ("sequential_2", 1), ("concurrent_2", 2)):
                status = get(client, args.service_http + "/status").json()
                assert check_service(status), "Start requires an idle pool"
                before_metrics = get(client, args.decoder + "/metrics").text
                baseline = sources(before_metrics)
                save(phase + "_before_service", status)
                (args.output / f"{phase}_metrics_before.txt").write_text(
                    before_metrics, encoding="utf-8")
                if args.p_trace_url:
                    _, since_ns, snapshots = collect_remote_snapshots(
                        lambda url: get(client, url).json(), {'P':args.p_trace_url,'D':args.d_trace_url}, args.session_id)
                    for role, snapshot in snapshots.items():
                        save(phase + '_trace_snapshot_' + role + '_baseline', snapshot)
                else:
                    since_ns = time.monotonic_ns()
                if count == 1:
                    results = [request(0)]
                else:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = [executor.submit(request, i) for i in range(count)]
                        results = [future.result() for future in futures]
                save(phase + "_responses", results)
                deadline, observed = time.monotonic() + 20, {}
                while True:
                    status = get(client, args.service_http + "/status").json()
                    idle = check_service(status)
                    after_metrics = get(client, args.decoder + "/metrics").text
                    try:
                        observed = {key: value - baseline[key]
                                    for key, value in sources(after_metrics).items()}
                        ready = all(observed[key] == value * count
                                    for key, value in expected.items())
                    except AssertionError as exc:
                        observed, ready = dict(not_yet_published=str(exc)), False
                    if idle and ready:
                        break
                    if time.monotonic() >= deadline:
                        save(phase + "_timeout", dict(service=status, observed=observed))
                        raise TimeoutError("D source metrics or pool protections did not settle")
                    time.sleep(0.2)
                if args.p_trace_url:
                    records, _, snapshots = collect_remote_snapshots(
                        lambda url: get(client, url).json(), {'P':args.p_trace_url,'D':args.d_trace_url},
                        args.session_id, since_ns)
                    for role, snapshot in snapshots.items():
                        save(phase + '_trace_snapshot_' + role + '_latest', snapshot)
                else:
                    records = trace_records(args.trace_dir)
                assert_no_new_p_traffic(records, since_ns)
                counts = expected_followup_trace(expected['prompt'], chunk, count, tp_size, bytes_per_request)
                trace = verify_d_trace(records, since_ns, counts['copies'], tp_size, counts['bytes'])
                save(phase + '_trace_full_with_scratch', records)
                save(phase + "_trace", active_trace(records, since_ns))
                save(phase + "_after_service", status)
                (args.output / f"{phase}_metrics_after.txt").write_text(
                    after_metrics, encoding="utf-8")
                summary = dict(name=phase, requests=count, answers=[r["answer"] for r in results],
                               source_delta=observed, direct_trace=trace,
                               final_pins=status["total_pins"], pending_writes=status["pending_writes"],
                               expected_trace=counts,
                               note="Client concurrency; source counters aggregate. Scheduler batching/copy overlap not established.")
                receipt["phases"].append(summary)
                save("receipt", receipt)
                print("MEMFABRIC_DIRECT_FOLLOWUP_PHASE_PASS", json.dumps(summary), flush=True)
        receipt.update(status="PASS", exit_code=0, completed_ns=time.monotonic_ns(),
                       total_requests=4,
                       note="Functional replay/client concurrency only; not scheduler batching, copy overlap or a performance sample")
        save("receipt", receipt)
        print("MEMFABRIC_DIRECT_FOLLOWUP_PASS", json.dumps(receipt), flush=True)
    except Exception as exc:
        receipt.update(status="FAIL", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        save("receipt", receipt)
        raise


if __name__ == "__main__":
    main()
