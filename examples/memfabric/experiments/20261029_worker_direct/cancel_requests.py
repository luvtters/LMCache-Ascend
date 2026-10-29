#!/usr/bin/env python3
"""Close real completion streams, then observe cleanup; plan by default.

This CPU client never starts/stops a service, clears a cache, or releases a
pool protection. Closing headers-only is not native DMA fault injection and
does not prove that the model or the network produced no output tokens.
"""
import argparse
import http.client
import json
from pathlib import Path
import re
import socket
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request
import uuid

import bench_d


OPERATIONS = ("L2G", "G2L", "gather", "scatter")


def metric_sum(text, name, labels=None):
    values = []
    for line in text.splitlines():
        match = re.match(r"^" + re.escape(name) + r"(?:\{([^}]*)\})?\s+(\S+)", line)
        if not match:
            continue
        found = dict(re.findall(r'(\w+)="([^"\\]*(?:\\.[^"\\]*)*)"', match[1] or ""))
        if labels and any(found.get(key) != value for key, value in labels.items()):
            continue
        try:
            value = float(match[2])
        except ValueError:
            return None
        if not bench_d.finite(value) or value < 0:
            return None
        values.append(value)
    return sum(values) if values else None


def metric_change(before, after, name, labels=None):
    first = metric_sum(before["bodies"].get("d_metrics", ""), name, labels)
    last = metric_sum(after["bodies"].get("d_metrics", ""), name, labels)
    return last - first if bench_d.finite(first) and bench_d.finite(last) and last >= first else None


def idle_state(snapshot):
    drain = bench_d.drain_state(snapshot, "C")
    text = snapshot["bodies"].get("d_metrics", "")
    requests = {name: metric_sum(text, "vllm:" + name)
                for name in ("num_requests_running", "num_requests_waiting")}
    counter = bench_d.numeric_fields(snapshot["json"].get("counter", {}))
    required = ["operations." + name + ".in_flight" for name in OPERATIONS]
    required += ["retrieve.in_flight"]
    required += ["rpc." + name + ".active_result_calls" for name in ("lookup", "acquire", "release", "other")]
    diagnostic_work = {name: counter.get(name) for name in required}
    known = (bench_d.endpoint_success(snapshot) and drain["drained"] is not None
             and all(bench_d.finite(value) for value in (*requests.values(), *diagnostic_work.values())))
    idle = (known and drain["drained"] is True
            and all(value == 0 for value in (*requests.values(), *diagnostic_work.values())))
    return dict(known=known, idle=idle, pool_and_l1=drain,
                request_gauges=requests, worker_diagnostic_work=diagnostic_work)


def window_evidence(before, after):
    traffic = bench_d.observed_bytes(before, after)
    fields = ("rpc", "retrieve")
    first = {name: before["json"].get("counter", {}).get(name, {}) for name in fields}
    last = {name: after["json"].get("counter", {}).get(name, {}) for name in fields}
    diagnostic = bench_d.delta_values(bench_d.numeric_fields(first), bench_d.numeric_fields(last))
    source_changes = {role: bench_d.delta_values(
        bench_d.source_counts(before["bodies"].get(role + "_metrics", "")),
        bench_d.source_counts(after["bodies"].get(role + "_metrics", ""))) for role in ("p", "d")}
    p_idle = all(item["delta"] == 0 for item in source_changes["p"].values())
    errors = traffic.get("worker_operation_deltas") or {}
    native_error_names = [name + ".failed_calls" for name in OPERATIONS]
    no_native_errors = all(errors.get(name, {}).get("delta") == 0 for name in native_error_names)
    return dict(actual_transfer=traffic, rpc_and_retrieve_deltas=diagnostic,
                prompt_source_deltas=source_changes, p_no_new_requests=p_idle,
                no_new_native_errors=no_native_errors,
                abort_count=metric_change(before, after, "vllm:request_success_total", {"finished_reason": "abort"}),
                generated_tokens=metric_change(before, after, "vllm:generation_tokens_total"))


def wait_idle(args, *, after_close=None, before=None, for_completion=False):
    deadline = time.monotonic() + args.wait_seconds
    quiet_since = None
    attempts = 0
    while True:
        remaining = max(.01, deadline - time.monotonic())
        # Four endpoints share the remaining poll budget, rather than each
        # consuming a full new timeout after the 30-second deadline.
        limited = SimpleNamespace(**dict(vars(args), http_timeout=min(args.http_timeout, remaining / 4)))
        current = bench_d.take_snapshot(limited)
        state = idle_state(current)
        now = time.monotonic()
        attempts += 1
        if state["idle"]:
            quiet_since = quiet_since or now
        else:
            quiet_since = None
        quiet = (quiet_since is not None and now - quiet_since >= args.quiet_seconds
                 and (after_close is None or now - after_close >= args.quiet_seconds))
        observed = window_evidence(before, current) if before is not None else None
        if for_completion:
            aligned = bench_d.alignment(before, current, 1, args.input_tokens)
            acknowledged = aligned.get("d", {}).get("aligned") is True and aligned.get("p", {}).get("aligned") is True
        else:
            acknowledged = (observed is None or (bench_d.finite(observed["abort_count"])
                            and observed["abort_count"] >= 1))
        full_generation = (not for_completion and observed is not None and bench_d.finite(observed["generated_tokens"])
                           and observed["generated_tokens"] >= args.max_tokens)
        if quiet and (acknowledged or full_generation):
            break
        if now >= deadline:
            break
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    return current, dict(state=state, stable_idle=bool(quiet), polls=attempts,
                         wait_seconds=args.wait_seconds - max(0, deadline - time.monotonic()),
                         abort_acknowledged=acknowledged if before is not None and not for_completion else None,
                         completion_source_alignment=acknowledged if for_completion else None)


def cancel_http(url, payload, delay, timeout):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Expected an HTTP(S) completion URL")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    hint = "mf-cancel-" + uuid.uuid4().hex
    record = dict(url=url, request_id_hint=hint, requested_delay_seconds=delay,
                  started_unix=time.time(), started_monotonic_ns=time.monotonic_ns(),
                  request_body_bytes_submitted=len(body), application_body_bytes_read=0,
                  network_body_bytes_read="UNKNOWN; the HTTP header parser may buffer body bytes",
                  http_status=None, headers=[], response_close_called=False, connection_close_called=False)
    response = transport = None
    try:
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request("POST", target, body=body, headers={
            "Content-Type": "application/json", "Accept": "text/event-stream", "X-Request-Id": hint})
        record["request_sent_monotonic_ns"] = time.monotonic_ns()
        transport = connection.sock
        response = connection.getresponse()
        record.update(http_status=response.status, reason=response.reason,
                      headers=list(response.getheaders()), headers_received_unix=time.time(),
                      headers_received_monotonic_ns=time.monotonic_ns())
        time.sleep(delay)
    except (OSError, http.client.HTTPException) as error:
        record["error"] = repr(error)
    finally:
        record["close_started_unix"] = time.time()
        record["close_started_monotonic_ns"] = time.monotonic_ns()
        if transport is not None and transport.fileno() >= 0:
            try:
                transport.shutdown(socket.SHUT_RDWR)
                record["socket_shutdown_called"] = True
            except OSError as error:
                record["socket_shutdown_error"] = repr(error)
        if response is not None:
            response.close()
            record["response_close_called"] = True
        connection.close()
        record["connection_close_called"] = True
        record["closed_unix"] = time.time()
        record["closed_monotonic_ns"] = time.monotonic_ns()
    return record


def complete_http(url, payload, timeout):
    record = dict(url=url, started_unix=time.time())
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            record["body"] = response.read().decode("utf-8")
        record["response"] = json.loads(record["body"])
    except (OSError, ValueError) as error:
        record["error"] = repr(error)
    record["finished_unix"] = time.time()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--d-url", "--base-url", dest="base_url", default="http://127.0.0.1:28402")
    parser.add_argument("--p-url", default="http://127.0.0.1:28401")
    parser.add_argument("--service-url", default="http://127.0.0.1:28195")
    parser.add_argument("--counter-url", default="http://127.0.0.1:29402/counters")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--model", default="mf-night")
    parser.add_argument("--input-tokens", type=int, default=1373)
    parser.add_argument("--expected-external-tokens", type=int, default=1280)
    parser.add_argument("--delays-seconds", default="0,0.005,0.02,0.05")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--wait-seconds", type=float, default=30)
    parser.add_argument("--quiet-seconds", type=float, default=1)
    parser.add_argument("--http-timeout", type=float, default=3)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    delays = [float(value) for value in args.delays_seconds.split(",")]
    if not delays or any(not 0 <= value <= 1 for value in delays):
        parser.error("Use nonnegative close delays no greater than one second")
    if not 0 < args.wait_seconds <= 30 or not 0 < args.quiet_seconds < args.wait_seconds:
        parser.error("Need 0 < quiet-seconds < wait-seconds <= 30")
    if not 0 <= args.expected_external_tokens < args.input_tokens or args.max_tokens < 1:
        parser.error("Invalid token lengths")
    print(json.dumps(dict(mode="RUN" if args.run else "PLAN_ONLY", output_root=str(args.output_root),
                          delay_seconds=delays, stream=True, ignore_eos=True,
                          application_body_reads=False, services_started=False,
                          boundary="Real HTTP client close and cleanup; not native DMA fault injection"), indent=2))
    if not args.run:
        return 0
    if args.output_root.exists():
        raise FileExistsError("Use a fresh cancellation output directory")
    with args.dataset.open(encoding="utf-8") as stream:
        prompt = json.loads(next(line for line in stream if line.strip()))["prompt"]
    if not isinstance(prompt, str):
        raise ValueError("Expected a raw text prompt")
    args.output_root.mkdir(parents=True)
    trials, safe = [], True
    url = args.base_url.rstrip("/") + "/v1/completions"
    for index, delay in enumerate(delays, 1):
        folder = args.output_root / f"cancel_{index:02d}"
        folder.mkdir()
        before, before_idle = wait_idle(args)
        bench_d.save_snapshot(folder, "before", before)
        bench_d.dump(folder / "before_idle.json", before_idle)
        known_abort = metric_sum(before["bodies"].get("d_metrics", ""), "vllm:request_success_total", {"finished_reason": "abort"})
        known_generation = metric_sum(before["bodies"].get("d_metrics", ""), "vllm:generation_tokens_total")
        if not before_idle["stable_idle"] or not bench_d.finite(known_abort) or not bench_d.finite(known_generation):
            trials.append(dict(index=index, passed=False, status="BEFORE_NOT_IDLE_OR_METRICS_UNKNOWN"))
            safe = False
            break
        response = cancel_http(url, dict(model=args.model, prompt=prompt, max_tokens=args.max_tokens,
                                        temperature=0, stream=True, ignore_eos=True), delay, args.request_timeout)
        bench_d.dump(folder / "connection.json", response)
        after, after_idle = wait_idle(args, after_close=response["closed_monotonic_ns"] / 1e9, before=before)
        bench_d.save_snapshot(folder, "after", after)
        evidence = window_evidence(before, after)
        content_type = dict((key.lower(), value) for key, value in response["headers"]).get("content-type", "")
        checks = dict(http_200=response["http_status"] == 200 and "error" not in response,
                      event_stream=content_type.startswith("text/event-stream"),
                      closed_without_application_body_read=(response["response_close_called"] and response["connection_close_called"]
                                                            and response["application_body_bytes_read"] == 0),
                      idle_before=before_idle["stable_idle"], idle_after=after_idle["stable_idle"],
                      counter_same_session=evidence["actual_transfer"]["counter_same_session"] is True,
                      cpu_copy_attempts_zero=evidence["actual_transfer"]["cpu_copy_attempts"] == 0,
                      no_new_native_errors=evidence["no_new_native_errors"],
                      p_no_new_requests=evidence["p_no_new_requests"],
                      server_abort_observed=evidence["abort_count"] == 1,
                      full_generation_not_completed=(bench_d.finite(evidence["generated_tokens"])
                                                     and 0 <= evidence["generated_tokens"] < args.max_tokens))
        row = dict(index=index, delay_seconds=delay, passed=all(checks.values()), checks=checks,
                   evidence=evidence, after_idle=after_idle,
                   fixed_external_token_count_required=False,
                   dma_interruption_verified=False, network_fault_injected=False)
        bench_d.dump(folder / "summary.json", row)
        trials.append(row)
        safe = (after_idle["stable_idle"] and checks["counter_same_session"] and checks["no_new_native_errors"]
                and checks["cpu_copy_attempts_zero"] and checks["p_no_new_requests"]
                and checks["http_200"] and checks["closed_without_application_body_read"])
        if not safe:
            break
    followup = dict(sent=False, passed=False)
    if safe:
        folder = args.output_root / "followup"
        folder.mkdir()
        before, before_idle = wait_idle(args)
        bench_d.save_snapshot(folder, "before", before)
        if before_idle["stable_idle"]:
            response = complete_http(url, dict(model=args.model, prompt=prompt, max_tokens=8, temperature=0), args.request_timeout)
            bench_d.dump(folder / "response.json", response)
            after, after_idle = wait_idle(args, before=before, for_completion=True)
            bench_d.save_snapshot(folder, "after", after)
            alignment = bench_d.alignment(before, after, 1, args.input_tokens)
            sources = {key: item["delta"] for key, item in alignment.get("d", {}).get("sources", {}).items()}
            evidence = window_evidence(before, after)
            value = response.get("response", {})
            answer = (value.get("choices") or [{}])[0].get("text", "")
            checks = dict(http_200=response.get("http_status") == 200 and "error" not in response,
                          usage=value.get("usage", {}).get("prompt_tokens") == args.input_tokens,
                          answer_orchid="ORCHID" in answer,
                          source_alignment=alignment.get("d", {}).get("aligned") is True,
                          restored=sources.get("external_kv_transfer") == args.expected_external_tokens,
                          computed=sources.get("local_compute") == args.input_tokens - args.expected_external_tokens,
                          prefix_zero=sources.get("local_cache_hit") == 0,
                          actual_pool_read=bench_d.finite(evidence["actual_transfer"]["read_bytes"]) and evidence["actual_transfer"]["read_bytes"] > 0,
                          counter_same_session=evidence["actual_transfer"]["counter_same_session"] is True,
                          cpu_copy_attempts_zero=evidence["actual_transfer"]["cpu_copy_attempts"] == 0,
                          no_new_native_errors=evidence["no_new_native_errors"],
                          p_no_new_requests=evidence["p_no_new_requests"], idle_after=after_idle["stable_idle"])
            followup = dict(sent=True, passed=all(checks.values()), checks=checks, answer=answer,
                            prompt_sources=alignment, evidence=evidence, after_idle=after_idle)
        bench_d.dump(folder / "summary.json", followup)
    result = dict(passed=len(trials) == len(delays) and all(row["passed"] for row in trials) and followup["passed"],
                  trials=trials, followup=followup, services_started_or_stopped=False,
                  protection_released_by_client=False,
                  boundary="HTTP headers-only close, server abort counters and cleanup; not native DMA or network fault injection")
    bench_d.dump(args.output_root / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
