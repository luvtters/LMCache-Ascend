# SPDX-License-Identifier: Apache-2.0
"""Stop our idle pool member and verify cache-miss fallback, without stale reads."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

import httpx

from pd_workload import body


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2))

    def metric(text, source):
        return sum(float(s.rsplit(" ", 1)[1]) for s in text.splitlines()
                   if s.startswith("vllm:prompt_tokens_by_source_total{")
                   and f'source="{source}"' in s)

    with httpx.Client(timeout=90, trust_env=False) as client:
        def status():
            response = client.get(args.service + "/status")
            response.raise_for_status()
            return response.json()

        def infer(name):
            response = client.post(args.engine + "/v1/chat/completions",
                                   json=body("SAFFRON"))
            response.raise_for_status()
            result = response.json()
            save(name, result)
            assert result["choices"][0]["message"]["content"].strip() == "SAFFRON"
            return result

        initial = status()["storage_manager"]["l2_adapters"][0]
        assert initial["is_healthy"]
        if not args.existing_pool:
            assert initial["objects"] == 0
        infer("seed_response")
        deadline = time.monotonic() + 20
        while True:
            before = status()
            sm = before["storage_manager"]
            pool = sm["l2_adapters"][0]
            ctrl = sm["store_controller"]
            if (pool["store_bytes"] > initial["store_bytes"] and pool["total_pins"] == 0
                    and ctrl["pending_keys_count"] == ctrl["in_flight_task_count"] == 0
                    and sm["l1_manager"]["total_object_count"] == 0):
                break
            assert time.monotonic() < deadline, "Seed did not settle"
            time.sleep(0.1)
        save("before_member_exit", before)
        metrics_before = client.get(args.engine + "/metrics").text
        (output / "metrics_before.txt").write_text(metrics_before)
        command = Path(f"/proc/{args.member_pid}/cmdline").read_bytes()
        assert b"pool_member.py" in command, "PID must be our pool_member process"
        os.kill(args.member_pid, getattr(signal, args.member_signal))
        deadline = time.monotonic() + 15
        while True:
            departed = status()
            if not departed["storage_manager"]["l2_adapters"][0]["is_healthy"]:
                break
            if time.monotonic() >= deadline:
                save("departure_not_detected", departed)
                raise AssertionError("Departed peer still reported healthy; no stale read attempted")
            time.sleep(0.1)
        save("after_member_exit", departed)
        response = infer("fallback_response")
        after = status()
        save("after_fallback", after)
        metrics_after = client.get(args.engine + "/metrics").text
        (output / "metrics_after.txt").write_text(metrics_after)
        new_pool = after["storage_manager"]["l2_adapters"][0]
        assert not new_pool["is_healthy"]
        assert new_pool["load_bytes"] == pool["load_bytes"]
        external = metric(metrics_after, "external_kv_transfer") - metric(
            metrics_before, "external_kv_transfer")
        computed = metric(metrics_after, "local_compute") - metric(
            metrics_before, "local_compute")
        assert external == 0 and computed == response["usage"]["prompt_tokens"]
        summary = {"answer": "SAFFRON", "adapter_healthy": False,
                   "new_bm_read_bytes": 0, "external_tokens": external,
                   "recomputed_tokens": computed, "member_signal": args.member_signal}
        save("summary", summary)
        print("PEER_LOSS_FALLBACK_PASS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-pid", required=True, type=int)
    parser.add_argument("--member-signal", choices=["SIGINT", "SIGKILL"], default="SIGINT")
    parser.add_argument("--existing-pool", action="store_true",
                        help="Allow prior unrelated objects; SAFFRON must be a new prefix")
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--engine", default="http://127.0.0.1:28002")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
