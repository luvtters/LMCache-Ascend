# SPDX-License-Identifier: Apache-2.0
"""Check bounded L1 usage after the experimental BM pool has already failed."""

import argparse
import json
from pathlib import Path
import time

import httpx

from pd_workload import body


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    words = ["ACORN", "BAMBOO", "CITRUS", "DAISY", "EMBER", "FALCON",
             "GARNET", "HAZEL", "INDIGO", "JASMINE", "KOALA", "LANTERN",
             "MARBLE", "NECTAR", "ONYX", "PANDA", "QUARTZ", "RUBY",
             "SILVER", "TOPAZ", "UMBER", "VIOLET", "WILLOW", "ZEBRA"]
    results = []
    with httpx.Client(timeout=90, trust_env=False) as client:
        def status():
            r = client.get(args.service + "/status")
            r.raise_for_status()
            return r.json()["storage_manager"]

        initial = status()
        pool_before = initial["l2_adapters"][0]
        assert not pool_before["is_healthy"], "Run peer_loss_smoke first"
        metrics_before = client.get(args.engine + "/metrics").text
        (output / "metrics_before.txt").write_text(metrics_before)
        for i, word in enumerate(words):
            r = client.post(args.engine + "/v1/chat/completions", json=body(word, 192))
            r.raise_for_status()
            response = r.json()
            (output / f"response_{i}.json").write_text(json.dumps(response, indent=2))
            answer = response["choices"][0]["message"]["content"].strip()
            assert answer == word, (word, answer)
            deadline = time.monotonic() + 15
            while True:
                sm = status()
                l1, ctrl, pool = sm["l1_manager"], sm["store_controller"], sm["l2_adapters"][0]
                if (ctrl["pending_keys_count"] == ctrl["in_flight_task_count"] == 0
                        and l1["read_locked_count"] == l1["write_locked_count"] == 0
                        and pool["total_pins"] == 0):
                    break
                assert time.monotonic() < deadline, "Failed stores retained locks or tasks"
                time.sleep(0.1)
            assert not pool["is_healthy"]
            assert pool["load_bytes"] == pool_before["load_bytes"]
            assert pool["store_bytes"] == pool_before["store_bytes"]
            assert l1["memory_used_bytes"] <= l1["memory_total_bytes"]
            results.append({"word": word, "prompt_tokens": response["usage"]["prompt_tokens"],
                            "l1_bytes": l1["memory_used_bytes"],
                            "l1_objects": l1["total_object_count"],
                            "pending_stores": 0, "pins": 0})
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(results[-1]), flush=True)
        after = client.get(args.engine + "/metrics").text
        (output / "metrics_after.txt").write_text(after)

        def metric(text, source):
            return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                       if line.startswith("vllm:prompt_tokens_by_source_total{")
                       and f'source="{source}"' in line)

        external = metric(after, "external_kv_transfer") - metric(metrics_before, "external_kv_transfer")
        computed = metric(after, "local_compute") - metric(metrics_before, "local_compute")
        assert external == 0 and computed == sum(r["prompt_tokens"] for r in results)
        # Every prefix differs in the first chunk. A decline demonstrates eviction.
        declines = sum(b["l1_objects"] < a["l1_objects"] for a, b in zip(results, results[1:]))
        assert declines > 0, "Workload did not demonstrate L1 eviction"
        summary = {"requests": len(results), "passed": len(results),
                   "external_tokens": external, "computed_tokens": computed,
                   "peak_l1_bytes": max(r["l1_bytes"] for r in results),
                   "l1_capacity_bytes": l1["memory_total_bytes"],
                   "l1_object_count_declines": declines, "new_bm_read_bytes": 0,
                   "new_bm_write_bytes": 0, "remaining_pins": 0, "pending_stores": 0}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("DEGRADED_WORKLOAD_PASS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="http://127.0.0.1:28002")
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
