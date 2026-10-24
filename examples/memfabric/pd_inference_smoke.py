# SPDX-License-Identifier: Apache-2.0
"""Verify one real proxy -> P -> MemFabric -> D request on a fresh service.

Use --l2-store-policy skip_l1. Point P's request telemetry at this program's
local relay (default :25769); the unchanged upstream proxy listens at :25768.
The relay forwards the store-finished event only after BM commits and L1 drains.
This is a single-request correctness check, not a production routing service.
"""

import argparse
import asyncio
import json
from pathlib import Path
import threading
import time

from fastapi import FastAPI, HTTPException, Request
import httpx
import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--prefiller", default="http://127.0.0.1:28001")
    parser.add_argument("--decoder", default="http://127.0.0.1:28002")
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--telemetry-port", type=int, default=25769)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--proxy-telemetry",
                        default="http://127.0.0.1:25768/api/v1/telemetry")
    parser.add_argument("--output", default="pd-memfabric-results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    gate = {}
    rank_events = {}
    forwarding = False
    app = FastAPI()

    def save(name, data):
        (output / (name + ".json")).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def pool(status):
        adapters = status["storage_manager"]["l2_adapters"]
        matches = [a for a in adapters if a.get("protocol") == "SDMA"]
        assert len(adapters) == len(matches) == 1, adapters
        assert matches[0]["is_healthy"], matches[0]
        assert matches[0]["storage_rank"] == 1, matches[0]
        return matches[0]

    def get(client, base, path):
        response = client.get(base + path)
        response.raise_for_status()
        return response

    def metric(text, name, label=None):
        values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                  if line.startswith(name + "{")
                  and (label is None or label in line)]
        assert values, (name, label)
        return sum(values)

    @app.post("/api/v1/telemetry")
    async def on_store_finished(request: Request):
        nonlocal forwarding
        payload = await request.json()
        if payload.get("event") != "request_store_finished":
            raise HTTPException(400, "Expected request_store_finished")
        if len(payload.get("request_ids_set", [])) != 1:
            raise HTTPException(400, "This check requires one request")
        rank = payload.get("kv_rank")
        if payload.get("world_size") != args.tp_size or rank not in range(args.tp_size):
            raise HTTPException(400, "Unexpected TP size / rank")
        if rank_events and payload["request_ids_set"] != next(
                iter(rank_events.values()))["request_ids_set"]:
            raise HTTPException(409, "This check accepts one request")
        rank_events[rank] = payload
        save("p_store_events", list(rank_events.values()))
        if len(rank_events) < args.tp_size or forwarding:
            return {"status": "accepted", "received_ranks": sorted(rank_events)}
        forwarding = True
        save("p_store_event", payload)
        deadline = time.monotonic() + 45
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while True:
                response = await client.get(args.service + "/status")
                response.raise_for_status()
                status = response.json()
                sm = status["storage_manager"]
                bm = pool(status)
                l1 = sm["l1_manager"]
                ctrl = sm["store_controller"]
                ready = (
                    bm["objects"] > 0 and bm["store_bytes"] > 0
                    and bm["load_bytes"] == 0
                    and l1["total_object_count"] == 0
                    and l1["memory_used_bytes"] == 0
                    and ctrl["pending_keys_count"] == 0
                    and ctrl["in_flight_task_count"] == 0
                    and ctrl["deferred_l1_deletion_count"] == 0)
                if ready:
                    break
                if time.monotonic() >= deadline:
                    save("gate_timeout", status)
                    raise HTTPException(504, "BM commit / automatic L1 drain timed out")
                await asyncio.sleep(0.05)
            # This snapshot is taken before the proxy is allowed to send to D.
            save("before_decode", status)
            gate.update({"p_request_id": payload["request_ids_set"][0],
                         "p_completed_ranks": sorted(rank_events),
                         "store_bytes": bm["store_bytes"],
                         "objects": bm["objects"], "load_bytes": bm["load_bytes"],
                         "l1_objects": l1["total_object_count"],
                         "l1_bytes": l1["memory_used_bytes"]})
            print("MEMFABRIC_PD_GATE_READY", json.dumps(gate), flush=True)
            response = await client.post(args.proxy_telemetry, json=payload)
            response.raise_for_status()
            notification = response.json()
            save("proxy_notification", notification)
            if notification.get("notified") != 1:
                raise HTTPException(502, "Proxy did not recognize the P request ID")
            return notification

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=args.telemetry_port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    with httpx.Client(timeout=120, trust_env=False) as client:
        for engine in (args.prefiller, args.decoder):
            get(client, engine, "/health")
        initial = get(client, args.service, "/status").json()
        save("initial", initial)
        assert pool(initial)["objects"] == pool(initial)["store_bytes"] == 0
        assert initial["storage_manager"]["l1_manager"]["total_object_count"] == 0
        before = {}
        for name, engine in [("p", args.prefiller), ("d", args.decoder)]:
            before[name] = get(client, engine, "/metrics").text
            (output / f"{name}_metrics_before.txt").write_text(before[name])
        body = {"model": "mf-test", "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": "Reply with exactly ORCHID."},
                    {"role": "user", "content": "\n".join(
                        f"Record {i:03d}: the access word is ORCHID. "
                        "This line exists for cache validation." for i in range(64))
                     + "\nGive the access word."}]}
        save("request", body)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.05)
            # The only inference call this client makes is to the upstream proxy.
            response = client.post(args.proxy + "/v1/chat/completions", json=body)
            response.raise_for_status()
            result = response.json()
            save("response", result)
            assert gate, "P telemetry did not pass through the BM gate"
            assert result["choices"][0]["message"]["content"].strip() == "ORCHID"
            status = get(client, args.service, "/status").json()
            save("after_decode", status)
            bm_read = pool(status)["load_bytes"] - gate["load_bytes"]
            assert bm_read == gate["store_bytes"] > 0
            prompt_tokens = result["usage"]["prompt_tokens"]
            assert gate["objects"] == (
                prompt_tokens // initial["chunk_size"] * args.tp_size)
            after = {}
            deadline = time.monotonic() + 15
            while True:
                for name, engine in [("p", args.prefiller), ("d", args.decoder)]:
                    after[name] = get(client, engine, "/metrics").text
                if metric(after["d"], "vllm:prompt_tokens_total") - metric(
                        before["d"], "vllm:prompt_tokens_total") == prompt_tokens:
                    break
                assert time.monotonic() < deadline, "D metrics not published"
                time.sleep(0.2)
            for name, text in after.items():
                (output / f"{name}_metrics_after.txt").write_text(text)

            def delta(engine, name, label=None):
                return metric(after[engine], name, label) - metric(
                    before[engine], name, label)

            source = "vllm:prompt_tokens_by_source_total"
            external = delta("d", source, 'source="external_kv_transfer"')
            local = delta("d", source, 'source="local_compute"')
            prefix = delta("d", source, 'source="local_cache_hit"')
            assert delta("p", source, 'source="local_compute"') == prompt_tokens
            assert delta("p", source, 'source="external_kv_transfer"') == 0
            assert delta("p", "vllm:generation_tokens_total") == 1
            assert external == (prompt_tokens // initial["chunk_size"]
                                * initial["chunk_size"]) > 0
            assert local + external == prompt_tokens and prefix == 0
            summary = {**gate, "d_request_id": result["id"],
                       "prompt_tokens": prompt_tokens,
                       "d_external_tokens": external, "d_computed_tokens": local,
                       "d_local_prefix_tokens": prefix, "bm_read_bytes": bm_read,
                       "answer": "ORCHID", "manual_l1_clear": False}
            save("summary", summary)
            print("MEMFABRIC_PD_INFERENCE_PASS", json.dumps(summary), flush=True)
        finally:
            server.should_exit = True
            thread.join(timeout=10)


if __name__ == "__main__":
    main()
