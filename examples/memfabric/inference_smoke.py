# SPDX-License-Identifier: Apache-2.0
"""Cold store -> clear only L1 -> second engine loads from MemFabric.

Run against a fresh, private experimental LMCache service and two engines.
This intentionally clears that service's L1. Artifacts retain the evidence.
"""

import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--engine-a", default="http://127.0.0.1:28001")
    parser.add_argument("--engine-b", default="http://127.0.0.1:28002")
    parser.add_argument("--output", default="memfabric-inference-results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def request(base, path, data=None):
        encoded = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(
            base + path, data=encoded,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.read().decode()

    def save_json(name, data):
        (output / (name + ".json")).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def status(name):
        result = json.loads(request(args.service, "/status"))
        save_json(name, result)
        return result["storage_manager"]

    def adapter(state):
        matches = [a for a in state["l2_adapters"] if a.get("protocol") == "SDMA"]
        assert len(matches) == 1, matches
        assert matches[0]["is_healthy"], matches[0]
        return matches[0]

    for engine in (args.engine_a, args.engine_b):
        request(engine, "/health")
    initial = status("initial")
    assert adapter(initial)["objects"] == 0, "Use a fresh MemFabric pool"
    prompt = "\n".join(
        f"Record {i:03d}: the access word is ORCHID. "
        "This line exists for cache validation." for i in range(64))
    body = {"model": "mf-test", "temperature": 0, "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "Reply with exactly ORCHID."},
                         {"role": "user", "content": prompt + "\nGive the access word."}]}
    save_json("request", body)

    def infer(engine, name):
        before = time.monotonic()
        result = json.loads(request(engine, "/v1/chat/completions", body))
        result["observed_wall_seconds"] = time.monotonic() - before
        save_json(name, result)
        print(name, result["choices"][0]["message"]["content"],
              result["usage"], flush=True)
        return result

    cold = infer(args.engine_a, "cold")
    assert cold["usage"]["prompt_tokens"] >= 1024, cold["usage"]
    deadline = time.monotonic() + 60
    while True:
        stored = status("after_store")
        ctrl = stored["store_controller"]
        if (adapter(stored)["store_bytes"] > 0
                and ctrl["pending_keys_count"] == 0
                and ctrl["in_flight_task_count"] == 0):
            break
        assert time.monotonic() < deadline, "L2 store did not finish"
        time.sleep(0.5)

    cleared = json.loads(request(args.service, "/cache/clear",
                                 {"tier": "l1", "force": False}))
    save_json("clear_l1", cleared)
    state = status("after_clear_l1")
    assert state["l1_manager"]["total_object_count"] == 0, state["l1_manager"]
    assert state["l1_manager"]["memory_used_bytes"] == 0, state["l1_manager"]
    assert adapter(state)["objects"] == adapter(stored)["objects"] > 0
    before_load = adapter(state)["load_bytes"]
    (output / "warm_metrics_before.txt").write_text(
        request(args.engine_b, "/metrics"), encoding="utf-8")
    warm = infer(args.engine_b, "warm")
    after = status("after_warm")
    (output / "warm_metrics_after.txt").write_text(
        request(args.engine_b, "/metrics"), encoding="utf-8")
    load_delta = adapter(after)["load_bytes"] - before_load
    assert load_delta > 0, "No MemFabric read occurred"
    cold_text = cold["choices"][0]["message"]["content"].strip()
    warm_text = warm["choices"][0]["message"]["content"].strip()
    assert cold_text == warm_text == "ORCHID", (cold_text, warm_text)
    summary = {"cold_text": cold_text, "warm_text": warm_text,
               "prompt_tokens": cold["usage"]["prompt_tokens"],
               "l1_empty_before_warm": True, "bm_load_bytes": load_delta,
               "bm_storage_rank": adapter(after)["storage_rank"]}
    save_json("summary", summary)
    print("MEMFABRIC_INFERENCE_PASS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
