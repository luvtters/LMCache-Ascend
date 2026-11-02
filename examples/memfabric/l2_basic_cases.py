#!/usr/bin/env python3
"""Run extra, source-accounted L2 cases on an already-running P/D service.

This script never starts or stops a model, service, pool member, or device.
Run it only after the strict gate succeeds and while that same group is resident.
"""
import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


SOURCE = "vllm:prompt_tokens_by_source_total"
SOURCE_NAMES = ("external_kv_transfer", "local_compute", "local_cache_hit")


def _get(url, timeout=10):
    request = urllib.request.Request(url)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return response.status, response.read()


def _post(url, body, timeout=240):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data,
                                    headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)
        return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _metric(text, name, source=None):
    total = 0.0
    found = False
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        if source is not None and f'source="{source}"' not in line:
            continue
        total += float(line.rsplit(" ", 1)[1])
        found = True
    if not found:
        return 0.0
    return total


def _metrics(base, timeout=10):
    result = {}
    for role in ("p", "d"):
        status, raw = _get(base[role] + "/metrics", timeout)
        if status != 200:
            raise RuntimeError(f"{role} metrics HTTP {status}")
        result[role] = raw.decode("utf-8", errors="replace")
    return result


def _service_status(url):
    status, raw = _get(url + "/status")
    text = raw.decode("utf-8", errors="replace")
    if status != 200:
        raise RuntimeError(f"service status HTTP {status}: {text}")
    return json.loads(text)


def _store_state(status):
    storage = status.get("storage_manager", {})
    controller = storage.get("store_controller", {})
    adapters = storage.get("l2_adapters", [])
    if len(adapters) != 1:
        raise RuntimeError(f"Expected one L2 adapter while checking store completion: {adapters}")
    return {
        "pending_keys_count": int(controller["pending_keys_count"]),
        "in_flight_task_count": int(controller["in_flight_task_count"]),
        "store_bytes": int(adapters[0]["store_bytes"]),
        "free_slots": int(adapters[0]["free_slots"]),
    }


def _write_status_sample(stream, status=None, error=None):
    sample = {"time_unix": time.time()}
    if status is not None:
        sample["status"] = status
        sample["store_state"] = _store_state(status)
    if error is not None:
        sample["error"] = error
    stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
    stream.flush()
    return sample


def _wait_store_idle(service_url, out, name, timeout=45):
    started = time.monotonic()
    deadline = started + timeout
    samples_path = out / "store_idle_before_statuses.jsonl"
    last_state = None
    last_status = None
    idle_streak = 0
    with samples_path.open("w", encoding="utf-8") as stream:
        while time.monotonic() < deadline:
            try:
                status = _service_status(service_url)
                sample = _write_status_sample(stream, status=status)
                last_status = status
                state = sample["store_state"]
                if state["pending_keys_count"] == 0 and state["in_flight_task_count"] == 0:
                    idle_streak = idle_streak + 1 if last_state == state["store_bytes"] else 1
                    if idle_streak >= 2:
                        (out / "store_idle_before.json").write_text(json.dumps({
                            "case": name, "complete": True, "stable_idle_samples": idle_streak,
                            "store_bytes": state["store_bytes"], "status_samples": samples_path.name,
                        }, indent=2) + "\n", encoding="utf-8")
                        return status
                    last_state = state["store_bytes"]
                else:
                    idle_streak = 0
                    last_state = state["store_bytes"]
            except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
                _write_status_sample(stream, error=repr(exc))
                idle_streak = 0
                last_state = None
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    details = {
        "case": name, "complete": False, "timeout_seconds": timeout,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "last_status": last_status, "status_samples": samples_path.name,
    }
    (out / "store_idle_before.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    raise TimeoutError(f"{name}: L2 store controller did not remain idle before the cold request for {timeout}s")


def _wait_store_completion(service_url, out, name, base_store_bytes,
                           target_new_bytes, timeout=45, initial_status=None):
    started = time.monotonic()
    deadline = started + timeout
    samples_path = out / "store_completion_statuses.jsonl"
    last_status = None
    last_state = None
    with samples_path.open("w", encoding="utf-8") as stream:
        status = initial_status
        while time.monotonic() < deadline:
            try:
                if status is None:
                    status = _service_status(service_url)
                sample = _write_status_sample(stream, status=status)
                last_status = status
                last_state = sample["store_state"]
                complete = (
                    last_state["pending_keys_count"] == 0
                    and last_state["in_flight_task_count"] == 0
                    and last_state["store_bytes"] - base_store_bytes >= target_new_bytes
                )
                status = None
                if complete:
                    details = {
                        "case": name, "complete": True, "timeout_seconds": timeout,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "base_store_bytes": base_store_bytes,
                        "target_new_bytes": target_new_bytes,
                        "target_store_bytes": base_store_bytes + target_new_bytes,
                        "final_store_state": last_state,
                        "status_samples": samples_path.name,
                    }
                    (out / "store_completion.json").write_text(
                        json.dumps(details, indent=2) + "\n", encoding="utf-8")
                    return last_status
            except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
                _write_status_sample(stream, error=repr(exc))
                status = None
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    details = {
        "case": name, "complete": False, "timeout_seconds": timeout,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "base_store_bytes": base_store_bytes,
        "target_new_bytes": target_new_bytes,
        "target_store_bytes": base_store_bytes + target_new_bytes,
        "last_store_state": last_state, "last_status": last_status,
        "status_samples": samples_path.name,
    }
    if last_status is not None:
        (out / "service_status_after.json").write_text(
            json.dumps(last_status, indent=2) + "\n", encoding="utf-8")
    (out / "store_completion.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    raise TimeoutError(
        f"{name}: L2 write did not settle at {target_new_bytes} new bytes with idle store controller within {timeout}s"
    )


def _request(url, body):
    status, raw = _post(url, body)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    return {"http_status": status, "body_raw": raw, "body_json": parsed}


def _token_ids(tokenizer, body):
    ids = tokenizer.apply_chat_template(
        body["messages"], tokenize=True, add_generation_prompt=True,
        **body.get("chat_template_kwargs", {}),
    )
    if isinstance(ids, Mapping):
        if "input_ids" not in ids:
            raise ValueError("tokenizer chat template mapping has no input_ids")
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _lcp(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _answer(response, label):
    body = response["body_json"]
    if response["http_status"] != 200 or not isinstance(body, dict):
        raise RuntimeError(f"{label} request failed: HTTP {response['http_status']} {response['body_raw']}")
    answer = body["choices"][0]["message"]["content"].strip()
    if answer != "ORCHID":
        raise RuntimeError(f"{label} returned {answer!r}, expected ORCHID")
    return body


def _case_output(root, name):
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run_case(name, bodies, token_ids, base_tokens, urls, service_url, service_log,
              tokenizer_meta, output_root, expected_source_deltas=None,
              write_completion_floor_bytes=0, wait_for_store_idle_before=False,
              minimum_free_slots_before=None):
    out = _case_output(output_root, name)
    before_offset = service_log.stat().st_size
    try:
        return _run_case_body(name, bodies, token_ids, base_tokens, urls, service_url,
                              service_log, out, tokenizer_meta, before_offset,
                              expected_source_deltas, write_completion_floor_bytes,
                              wait_for_store_idle_before, minimum_free_slots_before)
    finally:
        try:
            with service_log.open("rb") as stream:
                stream.seek(before_offset)
                log_delta = stream.read().decode("utf-8", errors="replace")
            (out / "service_log_delta.txt").write_text(log_delta, encoding="utf-8")
        except OSError as exc:
            (out / "service_log_capture_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")


def _run_case_body(name, bodies, token_ids, base_tokens, urls, service_url,
                   service_log, out, tokenizer_meta, before_offset,
                   expected_source_deltas=None, write_completion_floor_bytes=0,
                   wait_for_store_idle_before=False, minimum_free_slots_before=None):
    status_before = (_wait_store_idle(service_url, out, name)
                     if wait_for_store_idle_before else _service_status(service_url))
    if minimum_free_slots_before is not None:
        free_slots = _store_state(status_before)["free_slots"]
        if free_slots < minimum_free_slots_before:
            raise RuntimeError(
                f"{name}: only {free_slots} L2 slots are free before the cold write; "
                f"need {minimum_free_slots_before} to avoid eviction"
            )
    metrics_before = _metrics(urls)
    (out / "service_status_before.json").write_text(json.dumps(status_before, indent=2) + "\n", encoding="utf-8")
    for role, text in metrics_before.items():
        (out / f"{role}_metrics_before.txt").write_text(text, encoding="utf-8")
    (out / "requests.json").write_text(json.dumps(bodies, indent=2) + "\n", encoding="utf-8")
    meta = {"tokenizer": tokenizer_meta, "requests": []}
    for index, ids in enumerate(token_ids):
        common = _lcp(base_tokens, ids)
        meta["requests"].append({
            "index": index,
            "prompt_tokens_by_cpu_tokenizer": len(ids),
            "common_prefix_tokens_with_base": common,
            "maximum_reusable_full_chunks_128": common // 128,
            "maximum_reusable_tokens_128": (common // 128) * 128,
        })
    (out / "token_prefixes.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "token_ids.json").write_text(json.dumps({
        "base": base_tokens,
        "requests": token_ids,
    }, separators=(",", ":")) + "\n", encoding="utf-8")

    if len(bodies) == 1:
        responses = [_request(urls["d"] + "/v1/chat/completions", bodies[0])]
    else:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_request, urls["d"] + "/v1/chat/completions", body)
                       for body in bodies]
            responses = [future.result() for future in futures]
    (out / "http_responses.json").write_text(json.dumps(responses, indent=2) + "\n", encoding="utf-8")
    parsed = [_answer(response, f"{name}[{i}]") for i, response in enumerate(responses)]

    expected_prompt_tokens = 0
    for i, body in enumerate(parsed):
        prompt_tokens = int(body["usage"]["prompt_tokens"])
        expected_prompt_tokens += prompt_tokens
        expected_cpu_tokens = meta["requests"][i]["prompt_tokens_by_cpu_tokenizer"]
        if prompt_tokens != expected_cpu_tokens:
            raise RuntimeError(f"{name}[{i}] tokenizer/API prompt token mismatch: {expected_cpu_tokens} != {prompt_tokens}")

    deadline = time.monotonic() + 40
    while True:
        metrics_after = _metrics(urls)
        before_total = _metric(metrics_before["d"], "vllm:prompt_tokens_total")
        after_total = _metric(metrics_after["d"], "vllm:prompt_tokens_total")
        if after_total - before_total >= expected_prompt_tokens:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{name}: D prompt metrics did not reach {expected_prompt_tokens}")
        time.sleep(0.2)
    for role, text in metrics_after.items():
        (out / f"{role}_metrics_after.txt").write_text(text, encoding="utf-8")
    if write_completion_floor_bytes:
        status_after_response = _service_status(service_url)
        (out / "service_status_after_response.json").write_text(
            json.dumps(status_after_response, indent=2) + "\n", encoding="utf-8")
        base_store_bytes = _store_state(status_before)["store_bytes"]
        _wait_store_completion(
            service_url, out, name, base_store_bytes, write_completion_floor_bytes,
            initial_status=status_after_response,
        )
    status_after = _service_status(service_url)
    (out / "service_status_after.json").write_text(json.dumps(status_after, indent=2) + "\n", encoding="utf-8")

    source_deltas = {
        source: _metric(metrics_after["d"], SOURCE, source) - _metric(metrics_before["d"], SOURCE, source)
        for source in SOURCE_NAMES
    }
    prompt_delta = _metric(metrics_after["d"], "vllm:prompt_tokens_total") - _metric(metrics_before["d"], "vllm:prompt_tokens_total")
    if int(round(sum(source_deltas.values()))) != expected_prompt_tokens:
        raise RuntimeError(f"{name}: source counters {source_deltas} do not sum to prompt tokens {expected_prompt_tokens}")
    if int(round(prompt_delta)) != expected_prompt_tokens:
        raise RuntimeError(f"{name}: D prompt counter delta {prompt_delta} != {expected_prompt_tokens}")
    if expected_source_deltas is not None:
        for source, expected in expected_source_deltas.items():
            actual = int(round(source_deltas[source]))
            if actual != expected:
                raise RuntimeError(f"{name}: {source} delta {actual} != expected {expected}")

    with service_log.open("rb") as stream:
        stream.seek(before_offset)
        log_delta = stream.read().decode("utf-8", errors="replace")
    prefetch = [line for line in log_delta.splitlines() if "Prefetch request completed (L1+L2):" in line]
    summary = {
        "case": name,
        "request_count": len(bodies),
        "answer_each_orchid": True,
        "prompt_tokens": expected_prompt_tokens,
        "d_prompt_tokens_delta": prompt_delta,
        "d_prompt_source_deltas": source_deltas,
        "source_delta_sum_equals_prompt": True,
        "service_status_before_path": "service_status_before.json",
        "service_status_after_path": "service_status_after.json",
        "prefetch_log_lines": prefetch,
        "note": ("L1/L2 ownership is observed from raw status/logs; no fixed cache-layer hit expectation is imposed."
                 if expected_source_deltas is None else
                 "Cold multichunk source deltas are checked against the explicit cold/repeat expectation."),
    }
    if expected_source_deltas is not None:
        summary["expected_d_prompt_source_deltas"] = expected_source_deltas
    if write_completion_floor_bytes:
        summary["store_completion_floor_bytes"] = write_completion_floor_bytes
    if wait_for_store_idle_before:
        summary["store_idle_before_required"] = True
    if minimum_free_slots_before is not None:
        summary["minimum_free_slots_before"] = minimum_free_slots_before
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="same config.json used for the completed strict gate")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    case_root = Path(cfg["case_root"])
    gate_root = case_root / "gate"
    base_request_path = gate_root / "request.json"
    if not base_request_path.is_file():
        raise FileNotFoundError(f"Strict gate request not found: {base_request_path}")
    base_body = json.loads(base_request_path.read_text(encoding="utf-8"))
    gate_response_path = gate_root / "response.json"
    gate_response = json.loads(gate_response_path.read_text(encoding="utf-8")) if gate_response_path.is_file() else None

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    base_ids = _token_ids(tokenizer, base_body)
    if gate_response is not None:
        gate_tokens = int(gate_response["usage"]["prompt_tokens"])
        if len(base_ids) != gate_tokens:
            raise RuntimeError(f"Base CPU tokenizer count {len(base_ids)} differs from strict gate usage {gate_tokens}")
    tokenizer_meta = {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "base_prompt_tokens": len(base_ids),
        "chat_template_kwargs": base_body.get("chat_template_kwargs", {}),
        "cpu_only": True,
    }

    partial_body = json.loads(json.dumps(base_body))
    original_lines = partial_body["messages"][1]["content"].splitlines()
    record_lines = [line for line in original_lines if re.match(r"^Record \d{3}:", line)]
    if len(record_lines) < 48:
        raise RuntimeError(f"Base prompt has only {len(record_lines)} Record lines")
    partial_body["messages"][1]["content"] = "\n".join(
        record_lines[:48] + [
            "Record 900: the access word is ORCHID. This is a changed suffix for partial-prefix validation.",
            "Give the access word.",
        ]
    )
    cold_body = json.loads(json.dumps(base_body))
    cold_body["messages"] = [
        {"role": "system", "content": "This is an isolated cache case named COLD-20261102-A7. Reply with exactly ORCHID."},
        {"role": "user", "content": "New topic COLD-20261102-A7. What is the access word? Reply exactly ORCHID."},
    ]
    cold_multichunk_body = json.loads(json.dumps(base_body))
    locations = ("ridge station", "coastal array", "desert dome", "valley receiver")
    colors = ("violet", "amber", "indigo", "silver")
    entries = [
        f"Sky log {index:03d}: {locations[index % len(locations)]} saw a "
        f"{colors[index % len(colors)]} pulse in cycle {4100 + index}; marker ORCHID."
        for index in range(64)
    ]
    cold_multichunk_body["messages"] = [
        {"role": "system", "content": (
            "You are reviewing a separate astronomy archive with a new cache prefix. "
            "Answer the user's question using its archive entries and reply exactly ORCHID."
        )},
        {"role": "user", "content": (
            "The following 64 observations describe the independent ORION-SPECTRUM topic.\n"
            + "\n".join(entries)
            + "\nWhat checksum label is repeated in this archive? Reply exactly ORCHID."
        )},
    ]
    cases = [
        ("repeat_same", [base_body]),
        ("partial_prefix", [partial_body]),
        ("cold_request", [cold_body]),
        ("concurrent_same_base", [json.loads(json.dumps(base_body)), json.loads(json.dumps(base_body))]),
    ]
    tokenized = {name: [_token_ids(tokenizer, body) for body in bodies] for name, bodies in cases}
    cold_common = _lcp(base_ids, tokenized["cold_request"][0])
    if cold_common >= 128:
        raise RuntimeError(f"Cold request shares {cold_common} initial prompt tokens with the base; case is not cold")
    cold_multichunk_ids = _token_ids(tokenizer, cold_multichunk_body)
    cold_multichunk_tokens = len(cold_multichunk_ids)
    cold_multichunk_common = _lcp(base_ids, cold_multichunk_ids)
    cold_multichunk_chunks = cold_multichunk_tokens // 128
    base_full_chunks = len(base_ids) // 128
    params = cfg["l2"]["adapter_params"]
    if int(params["pool_gb"]) != 1 or int(params["slot_size_mb"]) != 32:
        raise RuntimeError("cold_multichunk capacity guard expects the current 1 GiB / 32-slot pool")
    if base_full_chunks != 10:
        raise RuntimeError(f"Expected the strict base request to occupy 10 full chunks, got {base_full_chunks}")
    if cold_multichunk_tokens < 256:
        raise RuntimeError(f"cold_multichunk has only {cold_multichunk_tokens} tokenizer tokens; need at least 256")
    if cold_multichunk_common >= 128:
        raise RuntimeError(
            f"cold_multichunk shares {cold_multichunk_common} initial prompt tokens with the base; case is not cold"
        )
    if cold_multichunk_chunks > 16 or base_full_chunks + cold_multichunk_chunks > 32:
        raise RuntimeError(
            f"cold_multichunk capacity guard failed: base={base_full_chunks}, new={cold_multichunk_chunks}, slots=32"
        )

    output_root = case_root / "basic_cases"
    output_root.mkdir(parents=False, exist_ok=False)
    urls = {key: f"http://127.0.0.1:{port}" for key, port in cfg["ports"].items()}
    service_url = urls["http"]
    service_log = case_root / "service.log"
    summaries = []
    for name, bodies in cases:
        summary = _run_case(name, bodies, tokenized[name], base_ids, urls, service_url,
                            service_log, tokenizer_meta, output_root)
        summaries.append(summary)
    cold_full_tokens = cold_multichunk_chunks * 128
    cold_write_bytes = cold_multichunk_chunks * 18 * 1024 * 1024
    cold_multichunk = _run_case(
        "cold_multichunk", [cold_multichunk_body], [cold_multichunk_ids], base_ids,
        urls, service_url, service_log, tokenizer_meta, output_root,
        expected_source_deltas={
            "external_kv_transfer": 0,
            "local_compute": cold_multichunk_tokens,
            "local_cache_hit": 0,
        },
        write_completion_floor_bytes=cold_write_bytes,
        wait_for_store_idle_before=True,
        minimum_free_slots_before=cold_multichunk_chunks,
    )
    summaries.append(cold_multichunk)
    cold_multichunk_repeat = _run_case(
        "cold_multichunk_repeat", [cold_multichunk_body], [cold_multichunk_ids], base_ids,
        urls, service_url, service_log, tokenizer_meta, output_root,
        expected_source_deltas={
            "external_kv_transfer": cold_full_tokens,
            "local_compute": cold_multichunk_tokens - cold_full_tokens,
            "local_cache_hit": 0,
        },
        wait_for_store_idle_before=True,
    )
    summaries.append(cold_multichunk_repeat)
    # Both prefixes have completed their own L2 writes. Distinct first chunks
    # avoid the same-key L1 reservation race observed in concurrent_same_base.
    # This checks two clients can both restore, not that BM copies overlap.
    concurrent_different = _run_case(
        "concurrent_different_prefixes", [base_body, cold_multichunk_body],
        [base_ids, cold_multichunk_ids], base_ids, urls, service_url,
        service_log, tokenizer_meta, output_root,
        expected_source_deltas={
            "external_kv_transfer": base_full_chunks * 128 + cold_full_tokens,
            "local_compute": len(base_ids) + cold_multichunk_tokens
                - base_full_chunks * 128 - cold_full_tokens,
            "local_cache_hit": 0,
        },
        wait_for_store_idle_before=True,
    )
    summaries.append(concurrent_different)
    result = {"status": "PASS_SOURCE_ACCOUNTED_BASIC_CASES", "cases": summaries,
              "tokenizer": tokenizer_meta,
              "partial_prefix_max_reuse_tokens":
                  (_lcp(base_ids, tokenized["partial_prefix"][0]) // 128) * 128,
              "cold_common_prefix_tokens": cold_common,
              "cold_multichunk_prompt_tokens": cold_multichunk_tokens,
              "cold_multichunk_common_prefix_tokens": cold_multichunk_common,
              "cold_multichunk_full_chunks": cold_multichunk_chunks,
              "cold_multichunk_store_bytes_floor": cold_write_bytes,
              "base_full_chunks": base_full_chunks,
              "pool_slot_capacity": 32,
              "concurrency_scope": "Two D HTTP clients overlap; this does not claim concurrent BM copies."}
    (output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
