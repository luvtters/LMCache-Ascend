#!/usr/bin/env python3
"""Single-request TP1 P/D L2 correctness gate, using unchanged official core.

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


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    ports = cfg["ports"]
    assert len(set(ports.values())) == len(ports)
    assert cfg["devices"]["p"] == [8] and cfg["devices"]["d"] == [10]
    assert cfg["devices"]["service"] == [8, 10, 12]
    assert cfg["l2"]["adapter_params"]["device_id"] == 2
    return cfg


def commands(cfg, config_path):
    py = cfg["python"]
    p = cfg["ports"]
    connector = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
                 "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                 "kv_connector_extra_config": {"lmcache.mp.host": "tcp://127.0.0.1",
                    "lmcache.mp.port": p["rpc"], "lmcache.mp.mp_transfer_mode": "lmcache_driven"}}
    engine = [py, "-m", "vllm.entrypoints.cli.main", "serve", cfg["model_path"],
              "--served-model-name", "mf-l2-gate", "--tensor-parallel-size", "1",
              "--dtype", "bfloat16", "--seed", "1024", "--max-model-len", "4096",
              "--max-num-batched-tokens", "1024", "--max-num-seqs", "1",
              "--gpu-memory-utilization", "0.7", "--block-size", "128",
              "--no-enable-prefix-caching", "--enforce-eager", "--host", "127.0.0.1",
              "--kv-transfer-config", json.dumps(connector)]
    return {
        "service": [py, "-u", str(Path(__file__).resolve()), "--config", config_path,
                    "--role", "service"],
        "service_cli": ["server", "--host", "127.0.0.1", "--port", str(p["rpc"]),
                        "--http-host", "127.0.0.1", "--http-port", str(p["http"]),
                        "--prometheus-port", str(p["metrics"]), "--chunk-size", "128",
                        "--l1-size-gb", "2", "--max-gpu-workers", "2",
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


def pool(status):
    adapters = status["storage_manager"]["l2_adapters"]
    assert len(adapters) == 1 and adapters[0]["protocol"] == "DEVICE_RDMA", adapters
    assert adapters[0]["is_healthy"], adapters[0]
    return adapters[0]


def metric(text, name, label=None):
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith(name + "{") and (label is None or label in line)]
    assert values, (name, label)
    return sum(values)


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
        if (event.get("event") != "request_store_finished" or event.get("world_size") != 1
                or event.get("kv_rank") != 0 or len(event.get("request_ids_set", [])) != 1):
            raise HTTPException(400, "Expected exactly one TP1 store event")
        if gate:
            raise HTTPException(409, "Only one request is accepted")
        save("p_store_event", event)
        deadline = time.monotonic() + 45
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while True:
                response = await client.get(base["http"] + "/status")
                response.raise_for_status()
                snapshot = response.json()
                sm = snapshot["storage_manager"]
                bm, l1, ctrl = pool(snapshot), sm["l1_manager"], sm["store_controller"]
                if (bm["objects"] > 0 and bm["store_bytes"] == bm["objects"] * (18 << 20)
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
                        p_request_id=event["request_ids_set"][0], l1_objects=0)
            response = await client.post(base["telemetry"] + "/api/v1/telemetry", json=event)
            response.raise_for_status()
            assert response.json().get("notified") == 1, response.text
            return response.json()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=p["gate"], log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    with httpx.Client(timeout=180, trust_env=False) as client:
        initial = status(client)
        save("initial", initial)
        assert pool(initial)["objects"] == pool(initial)["store_bytes"] == pool(initial)["load_bytes"] == 0
        before = {}
        for role in ("p", "d"):
            response = client.get(base[role] + "/metrics")
            response.raise_for_status()
            before[role] = response.text
            (out / (role + "_metrics_before.txt")).write_text(response.text, encoding="utf-8")
        body = {"model": "mf-l2-gate", "temperature": 0, "max_tokens": 32,
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
            expected_external = tokens // 128 * 128
            assert gate["objects"] == tokens // 128
            assert gate["store_bytes"] == expected_external * 147456
            deadline = time.monotonic() + 15
            while True:
                after = {}
                for role in ("p", "d"):
                    response = client.get(base[role] + "/metrics")
                    response.raise_for_status()
                    after[role] = response.text
                if metric(after["d"], "vllm:prompt_tokens_total") - metric(before["d"], "vllm:prompt_tokens_total") == tokens:
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
            assert external == expected_external > 0
            assert computed + external == tokens
            assert delta("d", source, 'source="local_cache_hit"') == 0
            assert delta("p", source, 'source="local_compute"') == tokens
            assert delta("p", source, 'source="external_kv_transfer"') == 0
            snapshot = status(client)
            assert pool(snapshot)["load_bytes"] == gate["store_bytes"] > 0
            save("after_decode", snapshot)
            with service_log.open("rb") as stream:
                stream.seek(log_offset)
                log = stream.read().decode("utf-8", errors="replace")
            (out / "request_service_log.txt").write_text(log, encoding="utf-8")
            rows = re.findall(r"Prefetch request completed \(L1\+L2\): (\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\).*?external_request_id=([^,]+),", log)
            expected = gate["objects"]
            assert len(rows) == 1, rows
            retained, requested, l1, l2, request_id = rows[0]
            assert (int(retained), int(requested), int(l1), int(l2)) == (expected, expected, 0, expected), rows
            public_id = result["id"].removeprefix("chatcmpl-")
            internal_id = request_id.removeprefix("chatcmpl-")
            assert internal_id == public_id or internal_id.startswith(public_id + "-"), rows
            summary = {**gate, "d_request_id": result["id"], "prompt_tokens": tokens,
                       "d_external_tokens": external, "d_computed_tokens": computed,
                       "l2_load_bytes": pool(snapshot)["load_bytes"], "answer": "ORCHID",
                       "prefetch_zero_l1": True, "manual_l1_clear": False,
                       "status": "FUNCTION_PASS_LIFECYCLE_REQUIRES_ROLE_WAITS"}
            save("summary", summary)
            print("MEMFABRIC_PD_FUNCTION_PASS " + json.dumps(summary), flush=True)
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("plan", "service", "p", "d", "proxy", "gate"), default="plan")
    args = parser.parse_args()
    cfg = read_config(args.config)
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
        torch.npu.set_device(0)
        runtime = ctypes.CDLL("libascendcl.so")
        runtime.aclrtGetPhyDevIdByLogicDevId.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        physical = ctypes.c_int(-1)
        code = runtime.aclrtGetPhyDevIdByLogicDevId(0, ctypes.byref(physical))
        assert code == 0 and physical.value == 8, (code, physical.value)
        import lmcache
        import lmcache_ascend
        print("SERVICE_NATIVE_CONTEXT " + json.dumps({"logical": 0, "physical": physical.value,
              "core_import": lmcache.__file__, "ascend_import": lmcache_ascend.__file__,
              "pin_memory_override": False}), flush=True)
        from lmcache.cli.main import main as cli_main
        cli_main(argv["service_cli"])
    elif args.role == "gate":
        inference_gate(cfg)
    else:
        os.execvpe(argv[args.role][0], argv[args.role], os.environ)


if __name__ == "__main__":
    main()
