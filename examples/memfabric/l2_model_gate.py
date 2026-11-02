#!/usr/bin/env python3
"""Single-request TP1/TP2/TP4/TP8 P/D L2 correctness gate, using unchanged official core.

Print the plan by default. The caller owns process startup, logs and bounded
shutdown; this entry point neither SSHes nor retries/restarts failed roles.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import threading
import time


class _StoreCompletionBarrier:
    """Collect one request's completion callback from every P TP rank."""

    def __init__(self, tensor_parallel_size):
        self.tensor_parallel_size = tensor_parallel_size
        self._model_name = None
        self._request_ids = None
        self._events = {}
        self._released = False

    def accept(self, event):
        if event.get("event") != "request_store_finished":
            raise ValueError("Unexpected telemetry event")
        world_size = event.get("world_size")
        if type(world_size) is not int or world_size != self.tensor_parallel_size:
            raise ValueError("Telemetry world_size does not match tensor_parallel_size")
        rank = event.get("kv_rank")
        if type(rank) is not int or not 0 <= rank < self.tensor_parallel_size:
            raise ValueError("Telemetry kv_rank is outside the tensor-parallel group")
        ids = event.get("request_ids_set")
        if (not isinstance(ids, list) or len(ids) != 1
                or not isinstance(ids[0], str) or not ids[0]):
            raise ValueError("Expected exactly one request ID per rank event")
        model_name = event.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Telemetry model_name is required")

        request_ids = tuple(ids)
        if self._released:
            if (model_name == self._model_name and request_ids == self._request_ids
                    and self._events.get(rank) == event):
                return {"complete": True, "duplicate": True, "events": self.events}
            raise ValueError("Conflicting or additional telemetry after request completion")
        if self._model_name is None:
            self._model_name = model_name
            self._request_ids = request_ids
        elif model_name != self._model_name or request_ids != self._request_ids:
            raise ValueError("TP rank telemetry does not describe the same request/model")

        previous = self._events.get(rank)
        if previous is not None:
            if previous != event:
                raise ValueError("Conflicting duplicate telemetry for one TP rank")
            return {"complete": False, "duplicate": True, "events": self.events}
        self._events[rank] = dict(event)
        if len(self._events) < self.tensor_parallel_size:
            return {"complete": False, "duplicate": False, "events": self.events}
        self._released = True
        return {"complete": True, "duplicate": False, "events": self.events}

    @property
    def events(self):
        return [self._events[rank] for rank in sorted(self._events)]

    @property
    def proxy_event(self):
        if not self._released:
            raise RuntimeError("TP store completion barrier is not complete")
        # The unchanged proxy consumes request_ids_set; use rank 0 as the
        # canonical notification after every rank has reported completion.
        return self._events[0]


def _protocol(cfg):
    params = cfg["l2"]["adapter_params"]
    protocol = params.get("data_op_type")
    if protocol not in ("SDMA", "DEVICE_RDMA"):
        raise ValueError(f"Unsupported L2 data_op_type: {protocol!r}")
    enable_56bits_gva = params.get("enable_56bits_gva")
    expected_gva = protocol == "DEVICE_RDMA"
    if type(enable_56bits_gva) is not bool or enable_56bits_gva is not expected_gva:
        raise ValueError(
            f"{protocol} requires enable_56bits_gva={expected_gva}"
        )
    return protocol


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    ports = cfg["ports"]
    assert len(set(ports.values())) == len(ports)
    tp_size = cfg.get("tensor_parallel_size", 1)
    if type(tp_size) is not int or tp_size not in (1, 2, 4, 8):
        raise ValueError("tensor_parallel_size must be 1, 2, 4, or 8")
    chunk_size = cfg.get("chunk_size", 128)
    if type(chunk_size) is not int or chunk_size < 128 or chunk_size % 128:
        raise ValueError("chunk_size must be a positive multiple of block size 128")
    devices = cfg["devices"]
    for role in ("p", "d"):
        role_devices = devices[role]
        if (not isinstance(role_devices, list) or len(role_devices) != tp_size
                or any(type(device) is not int or device < 0 for device in role_devices)
                or len(set(role_devices)) != tp_size):
            raise ValueError(f"devices.{role} must list {tp_size} distinct physical devices")
    service_devices = devices["service"]
    if (not isinstance(service_devices, list) or not service_devices
            or any(type(device) is not int or device < 0 for device in service_devices)
            or len(set(service_devices)) != len(service_devices)):
        raise ValueError("devices.service must list distinct physical devices")
    params = cfg["l2"]["adapter_params"]
    max_tokens = cfg.get("max_tokens", 32)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    device_id = params["device_id"]
    if type(device_id) is not int or not 0 <= device_id < len(service_devices):
        raise ValueError("l2.adapter_params.device_id must index devices.service")
    service_context = cfg.get("service_context_logical_device", 0)
    if type(service_context) is not int or not 0 <= service_context < len(service_devices):
        raise ValueError("service_context_logical_device must index devices.service")
    _protocol(cfg)
    return cfg


def commands(cfg, config_path):
    _protocol(cfg)
    py = cfg["python"]
    p = cfg["ports"]
    shutdown_timeout = cfg.get("shutdown_timeout")
    if "shutdown_timeout" in cfg and (type(shutdown_timeout) is not int or shutdown_timeout < 0):
        raise ValueError("shutdown_timeout must be a nonnegative integer (not bool)")
    connector = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
                 "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                 "kv_connector_extra_config": {"lmcache.mp.host": "tcp://127.0.0.1",
                    "lmcache.mp.port": p["rpc"], "lmcache.mp.mp_transfer_mode": "lmcache_driven"}}
    engine = [py, "-m", "vllm.entrypoints.cli.main", "serve", cfg["model_path"],
              "--served-model-name", "mf-l2-gate", "--tensor-parallel-size", str(cfg.get("tensor_parallel_size", 1)),
              "--dtype", "bfloat16", "--seed", "1024", "--max-model-len", "4096",
              "--max-num-batched-tokens", "1024", "--max-num-seqs", "1",
              "--gpu-memory-utilization", "0.7", "--block-size", "128",
              "--no-enable-prefix-caching", "--enforce-eager", "--host", "127.0.0.1",
              "--kv-transfer-config", json.dumps(connector)]
    if "shutdown_timeout" in cfg:
        engine += ["--shutdown-timeout", str(shutdown_timeout)]
    return {
        "service": [py, "-u", str(Path(__file__).resolve()), "--config", config_path,
                    "--role", "service"],
        "service_cli": ["server", "--host", "127.0.0.1", "--port", str(p["rpc"]),
                        "--http-host", "127.0.0.1", "--http-port", str(p["http"]),
                        "--prometheus-port", str(p["metrics"]), "--chunk-size", str(cfg.get("chunk_size", 128)),
                        "--l1-size-gb", "2", "--max-gpu-workers", str(2 * cfg.get("tensor_parallel_size", 1)),
                        "--eviction-policy", "LRU", "--supported-transfer-mode", "lmcache_driven",
                        "--l2-store-policy", "skip_l1", "--l2-prefetch-policy", "default",
                        "--l2-adapter", json.dumps(cfg["l2"])],
        "p": engine + ["--port", str(p["p"])],
        "d": engine + ["--port", str(p["d"])],
        "proxy": [py, "-u", str(Path(cfg["core_path"]) / "examples/disagg_prefill_mp/disagg_proxy_server.py"),
                  "--host", "127.0.0.1", "--port", str(p["proxy"]),
                  "--prefiller-host", "127.0.0.1", "--prefiller-port", str(p["p"]),
                  "--decoder-host", "127.0.0.1", "--decoder-port", str(p["d"]),
                  "--telemetry-port", str(p["telemetry"])],
    }


def pool(status, cfg):
    adapters = status["storage_manager"]["l2_adapters"]
    protocol = _protocol(cfg)
    assert len(adapters) == 1 and adapters[0]["protocol"] == protocol, adapters
    assert adapters[0]["is_healthy"], adapters[0]
    return adapters[0]


def metric(text, name, label=None):
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith(name + "{") and (label is None or label in line)]
    assert values, (name, label)
    return sum(values)


def _validate_prefetch_rows(log, public_request_id, tp_size, expected_chunks, expected_objects):
    pattern = re.compile(
        r"Prefetch request completed \(L1\+L2\): (\d+)/(\d+) retained keys "
        r"\((\d+) L1, (\d+) L2\).*?external_request_id=([^,]+), "
        r"prefetch_request_id=(\d+)"
    )
    rows = []
    for line in log.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        retained, requested, l1, l2, request_id, prefetch_id = match.groups()
        rows.append({"retained": int(retained), "requested": int(requested),
                     "l1": int(l1), "l2": int(l2),
                     "external_request_id": request_id,
                     "prefetch_request_id": int(prefetch_id), "raw_line": line})
    if len(rows) not in (1, tp_size):
        raise AssertionError(f"Expected one aggregate or {tp_size} per-rank prefetch rows, got {len(rows)}: {rows}")
    public_id = public_request_id.removeprefix("chatcmpl-")
    for row in rows:
        internal_id = row["external_request_id"].removeprefix("chatcmpl-")
        if not (internal_id == public_id or internal_id.startswith(public_id + "-")):
            raise AssertionError(f"Prefetch row belongs to another response: {row}")
    if len(rows) > 1 and len({row["prefetch_request_id"] for row in rows}) != tp_size:
        raise AssertionError(f"Per-rank prefetch rows must have distinct prefetch IDs: {rows}")
    if len(rows) == 1:
        valid = (rows[0]["retained"], rows[0]["requested"], rows[0]["l1"], rows[0]["l2"])
        expected = (expected_objects, expected_objects, 0, expected_objects)
        form = "aggregate"
    else:
        valid = (sum(row["retained"] for row in rows),
                 sum(row["requested"] for row in rows),
                 sum(row["l1"] for row in rows),
                 sum(row["l2"] for row in rows))
        expected = (expected_objects, expected_objects, 0, expected_objects)
        if any((row["retained"], row["requested"], row["l1"], row["l2"])
               != (expected_chunks, expected_chunks, 0, expected_chunks) for row in rows):
            raise AssertionError(f"Each per-rank prefetch row must cover {expected_chunks} L2-only chunks: {rows}")
        form = "per_rank"
    if valid != expected:
        raise AssertionError(f"Prefetch totals {valid} do not match expected {expected}: {rows}")
    return {"form": form, "rows": rows, "totals": dict(zip(("retained", "requested", "l1", "l2"), valid)),
            "status": "PASS"}


def _source_metric_scale(external, computed, tokens, expected_external, tp_size):
    """Accept API-level or per-TP-rank prompt counters, but no other totals."""
    for scale in sorted({1, tp_size}):
        if (external == expected_external * scale
                and computed + external == tokens * scale):
            return scale
    raise AssertionError(
        "Prompt source counters must account for all tokens at API or per-rank scale: "
        f"external={external}, computed={computed}, tokens={tokens}, tp={tp_size}"
    )


def _decode_tp_kv_rank(value, tp_size):
    """Decode ObjectKey.kv_rank for a single-node TP group, or direct rank IDs."""
    if 0 <= value < tp_size:
        return value
    world_size = (value >> 24) & 0xFF
    global_rank = (value >> 16) & 0xFF
    local_world_size = (value >> 8) & 0xFF
    local_rank = value & 0xFF
    if (world_size == tp_size and local_world_size == tp_size
            and global_rank < tp_size and local_rank < tp_size
            and global_rank == local_rank):
        return global_rank
    return None


def _review_decode_shard_log(log, tp_size, expected_chunks,
                             aggregate_store_bytes=None, aggregate_load_bytes=None):
    """Match per-key STORE/LOAD evidence by chunk and TP rank from raw logs."""
    if tp_size == 1:
        return {"status": "PASS", "required": False,
                "reason": "per-rank log gate is required only for TP2"}
    operations = {
        direction: {rank: {"keys": {}, "bytes": 0} for rank in range(tp_size)}
        for direction in ("STORE", "LOAD")
    }
    invalid_lines = []
    key_pattern = re.compile(
        r"MEMFABRIC_L2_OBJECT_COMPLETE direction=(STORE|LOAD) "
        r"key=(ObjectKey\(.*?\)) bytes=(\d+)"
    )
    rank_pattern = re.compile(r"\bkv_rank\s*=\s*(\d+)")
    chunk_pattern = re.compile(r"\bchunk_hash=(b'(?:\\.|[^'])*'|b\"(?:\\.|[^\"])*\")")
    group_pattern = re.compile(r"\bobject_group_id=(\d+)")
    for line_number, line in enumerate(log.splitlines(), start=1):
        if "MEMFABRIC_L2_OBJECT_COMPLETE" not in line:
            continue
        match = key_pattern.search(line)
        if match is None:
            invalid_lines.append({"line_number": line_number, "line": line})
            continue
        direction, key_repr, byte_text = match.groups()
        rank_match = rank_pattern.search(key_repr)
        chunk_match = chunk_pattern.search(key_repr)
        group_match = group_pattern.search(key_repr)
        if rank_match is None or chunk_match is None or group_match is None:
            invalid_lines.append({"line_number": line_number, "line": line})
            continue
        raw_rank = int(rank_match.group(1))
        rank = _decode_tp_kv_rank(raw_rank, tp_size)
        if rank is None:
            invalid_lines.append({"line_number": line_number, "line": line})
            continue
        chunk_key = (chunk_match.group(1), int(group_match.group(1)))
        row = operations[direction][rank]
        if chunk_key in row["keys"]:
            invalid_lines.append({"line_number": line_number, "line": line,
                                  "reason": "duplicate direction/rank/chunk record"})
            continue
        row["keys"][chunk_key] = int(byte_text)
        row["bytes"] += int(byte_text)

    rank_results = {}
    all_pass = not invalid_lines
    canonical_chunks = None
    for rank in range(tp_size):
        stored = operations["STORE"][rank]
        loaded = operations["LOAD"][rank]
        store_chunks = set(stored["keys"])
        load_chunks = set(loaded["keys"])
        if canonical_chunks is None:
            canonical_chunks = store_chunks
        rank_ok = (
            len(store_chunks) == expected_chunks
            and len(load_chunks) == expected_chunks
            and store_chunks == load_chunks
            and (canonical_chunks is None or store_chunks == canonical_chunks)
            and stored["bytes"] > 0
            and stored["bytes"] == loaded["bytes"]
        )
        all_pass = all_pass and rank_ok
        rank_results[str(rank)] = {
            "store_objects": len(store_chunks),
            "load_objects": len(load_chunks),
            "store_bytes": stored["bytes"],
            "load_bytes": loaded["bytes"],
            "store_load_key_sets_match": store_chunks == load_chunks,
            "chunks_match_other_ranks": store_chunks == canonical_chunks,
            "pass": rank_ok,
        }
    all_pass = all_pass and len(canonical_chunks or ()) == expected_chunks
    store_sum = sum(row["bytes"] for row in operations["STORE"].values())
    load_sum = sum(row["bytes"] for row in operations["LOAD"].values())
    if aggregate_store_bytes is not None and store_sum != aggregate_store_bytes:
        all_pass = False
    if aggregate_load_bytes is not None and load_sum != aggregate_load_bytes:
        all_pass = False
    return {
        "status": "PASS" if all_pass else "PENDING_SHARD_LOG_REVIEW",
        "required": True,
        "expected_ranks": list(range(tp_size)),
        "expected_chunks_per_rank": expected_chunks,
        "store_bytes_sum": store_sum,
        "load_bytes_sum": load_sum,
        "aggregate_store_bytes": aggregate_store_bytes,
        "aggregate_load_bytes": aggregate_load_bytes,
        "per_rank": rank_results,
        "invalid_or_duplicate_records": invalid_lines,
        "reason": ("STORE and LOAD logs match per rank, chunk, and byte count"
                   if all_pass else "raw per-key STORE/LOAD logs do not prove every expected TP shard"),
    }


def inference_gate(cfg):
    import asyncio
    from fastapi import FastAPI, HTTPException, Request
    import httpx
    import uvicorn

    p = cfg["ports"]
    base = {key: "http://127.0.0.1:" + str(value) for key, value in p.items()}
    out = Path(cfg["case_root"]) / "gate"
    out.mkdir(parents=True, exist_ok=False)
    gate = {}
    tp_size = cfg.get("tensor_parallel_size", 1)
    store_barrier = _StoreCompletionBarrier(tp_size)
    app = FastAPI()
    service_log = Path(cfg["case_root"]) / "service.log"
    log_offset = service_log.stat().st_size

    def save(name, data):
        (out / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def status(client):
        response = client.get(base["http"] + "/status")
        response.raise_for_status()
        return response.json()

    @app.post("/api/v1/telemetry")
    async def finished(request: Request):
        event = await request.json()
        try:
            accepted = store_barrier.accept(event)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        save("p_store_events", accepted["events"])
        if not accepted["complete"]:
            return {"accepted": 1, "complete": False,
                    "received_ranks": [row["kv_rank"] for row in accepted["events"]],
                    "expected_ranks": tp_size}
        if accepted["duplicate"]:
            return {"accepted": 1, "complete": True, "duplicate": True}
        event = store_barrier.proxy_event
        deadline = time.monotonic() + 45
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while True:
                response = await client.get(base["http"] + "/status")
                response.raise_for_status()
                snapshot = response.json()
                sm = snapshot["storage_manager"]
                bm, l1, ctrl = pool(snapshot, cfg), sm["l1_manager"], sm["store_controller"]
                if (bm["objects"] > 0 and bm["store_bytes"] > 0
                        and bm["load_bytes"] == 0 and l1["total_object_count"] == 0
                        and l1["memory_used_bytes"] == 0 and ctrl["pending_keys_count"] == 0
                        and ctrl["in_flight_task_count"] == 0
                        and ctrl["deferred_l1_deletion_count"] == 0):
                    break
                if time.monotonic() >= deadline:
                    save("drain_timeout", snapshot)
                    raise HTTPException(504, "Native L2 commit / skip_l1 drain incomplete")
                await asyncio.sleep(0.05)
            save("before_decode", snapshot)
            gate.update(store_bytes=bm["store_bytes"], objects=bm["objects"],
                        p_request_id=event["request_ids_set"][0], l1_objects=0,
                        tensor_parallel_size=tp_size,
                        expected_tp_ranks=list(range(tp_size)),
                        p_store_events=accepted["events"])
            response = await client.post(base["telemetry"] + "/api/v1/telemetry", json=event)
            response.raise_for_status()
            assert response.json().get("notified") == 1, response.text
            return response.json()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p["gate"], log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    with httpx.Client(timeout=180, trust_env=False) as client:
        initial = status(client)
        save("initial", initial)
        assert (pool(initial, cfg)["objects"] == pool(initial, cfg)["store_bytes"]
                == pool(initial, cfg)["load_bytes"] == 0)
        before = {}
        for role in ("p", "d"):
            response = client.get(base[role] + "/metrics")
            response.raise_for_status()
            before[role] = response.text
            (out / (role + "_metrics_before.txt")).write_text(response.text, encoding="utf-8")
        body = {"model": "mf-l2-gate", "temperature": 0,
                "max_tokens": cfg.get("max_tokens", 32),
                "chat_template_kwargs": {"enable_thinking": False}, "messages": [
                    {"role": "system", "content": "Reply with exactly ORCHID."},
                    {"role": "user", "content": "\n".join(
                        f"Record {i:03d}: the access word is ORCHID. This line exists for cache validation."
                        for i in range(64)) + "\nGive the access word."}]}
        save("request", body)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.05)
            response = client.post(base["proxy"] + "/v1/chat/completions", json=body)
            save("http_response", {"status": response.status_code, "body": response.text})
            response.raise_for_status()
            result = response.json()
            save("response", result)
            assert result["choices"][0]["message"]["content"].strip() == "ORCHID", result
            assert gate, "No controlled L2 drain event"
            tokens = result["usage"]["prompt_tokens"]
            chunk_size = cfg.get("chunk_size", 128)
            expected_external = tokens // chunk_size * chunk_size
            expected_chunks = tokens // chunk_size
            gate["expected_chunks"] = expected_chunks
            # Every TP shard has a distinct ObjectKey. Shard byte sizes vary
            # with the model/TP layout, so validate total bytes and full shard
            # coverage instead of assuming a TP1 18 MiB object size.
            assert gate["objects"] == expected_chunks * tp_size
            assert gate["store_bytes"] > 0
            gate["expected_shard_objects"] = expected_chunks * tp_size
            gate["l2_byte_accounting"] = "aggregate_only_no_per_kv_rank_bytes"
            deadline = time.monotonic() + 15
            while True:
                after = {}
                for role in ("p", "d"):
                    response = client.get(base[role] + "/metrics")
                    response.raise_for_status()
                    after[role] = response.text
                prompt_delta = (metric(after["d"], "vllm:prompt_tokens_total")
                                - metric(before["d"], "vllm:prompt_tokens_total"))
                if prompt_delta in {tokens, tokens * tp_size}:
                    break
                assert time.monotonic() < deadline, "D metrics not ready"
                time.sleep(0.2)
            for role in ("p", "d"):
                (out / (role + "_metrics_after.txt")).write_text(after[role], encoding="utf-8")
            def delta(role, name, label=None):
                return metric(after[role], name, label) - metric(before[role], name, label)
            source = "vllm:prompt_tokens_by_source_total"
            external = delta("d", source, 'source="external_kv_transfer"')
            computed = delta("d", source, 'source="local_compute"')
            assert expected_external > 0
            metric_scale = _source_metric_scale(
                external, computed, tokens, expected_external, tp_size)
            prompt_metric_scale = prompt_delta // tokens
            assert delta("d", source, 'source="local_cache_hit"') == 0
            assert delta("p", source, 'source="local_compute"') == tokens * metric_scale
            assert delta("p", source, 'source="external_kv_transfer"') == 0
            snapshot = status(client)
            assert pool(snapshot, cfg)["load_bytes"] == gate["store_bytes"] > 0
            save("after_decode", snapshot)
            with service_log.open("rb") as stream:
                stream.seek(log_offset)
                log = stream.read().decode("utf-8", errors="replace")
            (out / "request_service_log.txt").write_text(log, encoding="utf-8")
            expected = gate["objects"]
            prefetch_review = _validate_prefetch_rows(
                log, result["id"], tp_size, expected_chunks, expected)
            save("d_prefetch_rows", prefetch_review)
            shard_review = _review_decode_shard_log(
                log, tp_size, expected_chunks, gate["store_bytes"],
                pool(snapshot, cfg)["load_bytes"])
            save("d_shard_log_review", shard_review)
            gate_status = ("FUNCTION_PASS_LIFECYCLE_REQUIRES_ROLE_WAITS"
                           if shard_review["status"] == "PASS"
                           else "FUNCTION_PENDING_SHARD_LOG_REVIEW")
            summary = {**gate, "d_request_id": result["id"], "prompt_tokens": tokens,
                       "max_tokens_requested": cfg.get("max_tokens", 32),
                       "d_external_tokens": external, "d_computed_tokens": computed,
                       "source_metric_rank_scale": metric_scale,
                       "prompt_metric_rank_scale": prompt_metric_scale,
                       "d_prefetch_rows": prefetch_review,
                       "d_shard_log_review": shard_review,
                       "l2_load_bytes": pool(snapshot, cfg)["load_bytes"], "answer": "ORCHID",
                       "prefetch_zero_l1": True, "manual_l1_clear": False,
                       "status": gate_status}
            save("summary", summary)
            marker = ("MEMFABRIC_PD_FUNCTION_PASS" if shard_review["status"] == "PASS"
                      else "MEMFABRIC_PD_FUNCTION_PENDING_SHARD_LOG_REVIEW")
            print(marker + " " + json.dumps(summary), flush=True)
            return 0 if shard_review["status"] == "PASS" else 2
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("plan", "service", "p", "d", "proxy", "gate"), default="plan")
    parser.add_argument("--max-tokens", type=int,
                        help="gate request max_tokens override; defaults to config or 32")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            parser.error("--max-tokens must be positive")
        cfg["max_tokens"] = args.max_tokens
    argv = commands(cfg, str(Path(args.config).resolve()))
    if args.role == "plan":
        print(json.dumps({"status": "PLAN_ONLY_NOT_EXECUTED", "commands": argv,
                          "devices": cfg["devices"], "native_hostpin_context": 0}, indent=2))
        return
    for key in ("LMCACHE_REQUEST_TELEMETRY_TYPE", "LMCACHE_REQUEST_TELEMETRY_ENDPOINT"):
        os.environ.pop(key, None)
    devices = cfg["devices"].get(args.role, [])
    if devices:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        os.environ["ASCEND_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    os.environ["PYTHONPATH"] = os.pathsep.join((cfg["core_path"], cfg["ascend_path"], os.environ.get("PYTHONPATH", "")))
    sys.path[:0] = [cfg["core_path"], cfg["ascend_path"]]
    if args.role == "p":
        os.environ["LMCACHE_REQUEST_TELEMETRY_TYPE"] = "fastapi"
        os.environ["LMCACHE_REQUEST_TELEMETRY_ENDPOINT"] = "http://127.0.0.1:%d/api/v1/telemetry" % cfg["ports"]["gate"]
    print("L2_MODEL_ROLE " + json.dumps({"role": args.role, "pid": os.getpid(), "devices": devices,
                                       "core_path": cfg["core_path"], "ascend_path": cfg["ascend_path"]}), flush=True)
    if args.role == "service":
        import ctypes
        import torch
        import torch_npu  # noqa: F401
        service_context_logical = cfg.get("service_context_logical_device", 0)
        if (type(service_context_logical) is not int
                or not 0 <= service_context_logical < len(devices)):
            raise ValueError("service_context_logical_device must index devices.service")
        torch.npu.set_device(service_context_logical)
        runtime = ctypes.CDLL("libascendcl.so")
        runtime.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        physical = ctypes.c_int(-1)
        code = runtime.aclrtGetPhyDevIdByLogicDevId(service_context_logical, ctypes.byref(physical))
        expected_physical = devices[service_context_logical]
        assert code == 0 and physical.value == expected_physical, (code, physical.value, expected_physical)
        import lmcache
        import lmcache_ascend
        print("SERVICE_NATIVE_CONTEXT " + json.dumps({"logical": service_context_logical, "physical": physical.value,
              "core_import": lmcache.__file__, "ascend_import": lmcache_ascend.__file__,
              "pin_memory_override": False}), flush=True)
        from lmcache.cli.main import main as cli_main
        cli_main(argv["service_cli"])
    elif args.role == "gate":
        return inference_gate(cfg)
    else:
        os.execvpe(argv[args.role][0], argv[args.role], os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
