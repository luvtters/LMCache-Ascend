# SPDX-License-Identifier: Apache-2.0
"""Client-concurrent warm-L2 restore and retained-L1 replay functional check.

Requires an existing healthy retain B service and ordinary P/D proxy. Client
concurrency is not engine concurrency: max-num-seqs=1 may serialize requests.
Cold concurrent restore may partially recompute. PASS requires correct cold
restore with actual external reads, then full-block external reuse on hot replay.
"""

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx

from retain_smoke import drained, pool, request_body, response_token_targets, source_counts, source_evidence


def cold_restore_evidence(report, full_block_target):
    for role, evidence in report.items():
        delta = evidence["delta"]
        assert delta["external_kv_transfer"] > 0, f"{role}: cold restore used no external KV"
        assert delta["local_cache_hit"] == 0, f"{role}: local cache masked cold external restore"
        evidence["full_block_external_target"] = full_block_target
        evidence["cold_full_block_reuse"] = delta["external_kv_transfer"] >= full_block_target
        block_tail = evidence["expected_prompt_tokens"] - full_block_target
        evidence["cold_extra_compute_beyond_block_tail"] = max(0, delta["local_compute"] - block_tail)
    return report


async def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    word = "MAPLE"
    body = request_body(word, output.name)
    save("request", body)
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        async def metrics(name, when, deadline=None):
            result = {}
            for role, url in (("p", args.prefiller), ("d", args.decoder)):
                remaining = deadline - time.monotonic() if deadline is not None else 10
                if remaining <= 0:
                    raise TimeoutError("Prompt source metrics did not align within 15 seconds")
                response = await client.get(url.rstrip("/") + "/metrics", timeout=min(10, remaining))
                response.raise_for_status()
                result[role] = response.text
                (output / f"{name}_{role}_metrics_{when}.txt").write_text(response.text, encoding="utf-8")
                source_counts(response.text)
            return result

        async def aligned_metrics(name, before, responses):
            expected, minimum = response_token_targets(responses)
            deadline = time.monotonic() + 15
            while True:
                after = await metrics(name, "after", deadline)
                report = {role: source_evidence(before[role], after[role], expected,
                                                minimum if name == "hot_replay" else None) for role in ("p", "d")}
                save(name + "_prompt_token_sources", report)
                if all(value["aligned"] for value in report.values()):
                    if name == "restore":
                        report = cold_restore_evidence(report, minimum)
                        save(name + "_prompt_token_sources", report)
                    return report
                if time.monotonic() >= deadline:
                    raise TimeoutError("Prompt source metrics did not align within 15 seconds")
                await asyncio.sleep(0.2)

        async def status():
            response = await client.get(args.service.rstrip("/") + "/status")
            response.raise_for_status()
            result = response.json()
            pool(result)
            return result

        async def settled():
            deadline = time.monotonic() + 30
            while True:
                result = await status()
                if drained(result):
                    return result
                if time.monotonic() >= deadline:
                    save("settle_timeout", result)
                    raise TimeoutError("Store, prefetch, locks or pool pins did not drain")
                await asyncio.sleep(0.2)

        async def infer():
            response = await client.post(args.proxy.rstrip("/") + "/v1/chat/completions", json=body)
            response.raise_for_status()
            return response.json()

        async def phase(name):
            before = await settled()
            save(name + "_before_status", before)
            metric_before = await metrics(name, "before")
            answers = await asyncio.gather(*(infer() for _ in range(args.concurrency)), return_exceptions=True)
            serializable = [{"error": str(answer)} if isinstance(answer, BaseException) else answer for answer in answers]
            save(name + "_responses", serializable)
            after = await settled()
            save(name + "_after_status", after)
            assert all(not isinstance(answer, BaseException) for answer in answers), f"{name}: request failure"
            assert all(answer["choices"][0]["message"]["content"].strip() == word for answer in answers)
            # One interval for the complete phase; no overlapping per-request sums.
            delta = pool(after)["load_bytes"] - pool(before)["load_bytes"]
            if name == "restore":
                assert delta > 0, "Concurrent restore did not read BM"
            else:
                assert delta == 0, "Hot replay still read BM"
            return after, delta, await aligned_metrics(name, metric_before, answers)

        initial = await settled()
        save("initial_status", initial)
        seed_metric_before = await metrics("seed", "before")
        seed = await infer()
        save("seed_response", seed)
        assert seed["choices"][0]["message"]["content"].strip() == word
        seeded = await settled()
        save("seed_status", seeded)
        seed_sources = await aligned_metrics("seed", seed_metric_before, [seed])
        assert pool(seeded)["objects"] > 0
        assert pool(seeded)["store_bytes"] > pool(initial)["store_bytes"]
        cleared_response = await client.post(args.service.rstrip("/") + "/cache/clear", json={"tier": "l1", "force": False})
        cleared_response.raise_for_status()
        save("clear_response", cleared_response.json())
        cleared = await settled()
        save("after_clear_status", cleared)
        l1 = cleared["storage_manager"]["l1_manager"]
        assert l1["total_object_count"] == l1["memory_used_bytes"] == 0
        assert pool(cleared)["objects"] > 0
        restored, restore_bytes, restore_sources = await phase("restore")
        assert restored["storage_manager"]["l1_manager"]["total_object_count"] > 0
        hot, hot_bytes, hot_sources = await phase("hot_replay")
        assert hot["storage_manager"]["l1_manager"]["total_object_count"] > 0 and drained(hot)
        summary = {"status": "CONCURRENT_RETAIN_SMOKE_PASS", "nonce": output.name,
                   "client_concurrency": args.concurrency, "correct_answers": 1 + args.concurrency * 2,
                   "engine_concurrency": "NOT_INFERRED; max-num-seqs=1 may serialize client requests",
                   "restore_load_bytes_delta": restore_bytes, "hot_replay_load_bytes_delta": hot_bytes,
                   "prompt_token_sources": {"seed": seed_sources, "restore": restore_sources, "hot_replay": hot_sources},
                   "cold_full_block_reuse": all(value["cold_full_block_reuse"] for value in restore_sources.values()),
                   "cold_full_block_reuse_by_endpoint": {role: value["cold_full_block_reuse"] for role, value in restore_sources.items()},
                   "cold_extra_compute_beyond_block_tail": {role: value["cold_extra_compute_beyond_block_tail"] for role, value in restore_sources.items()},
                   "l1_objects_after_hot_replay": hot["storage_manager"]["l1_manager"]["total_object_count"],
                   "read_locked_count": 0, "write_locked_count": 0, "pool_pins": 0,
                   "note": "PASS means correct cold restore with external KV and full-block hot external reuse; cold concurrent restore may partially recompute. Not an ordinary benchmark or hit-rate measurement."}
        save("summary", summary)
        print("CONCURRENT_RETAIN_SMOKE_PASS", json.dumps(summary), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--prefiller", default="http://127.0.0.1:28001")
    parser.add_argument("--decoder", default="http://127.0.0.1:28002")
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= 8:
        parser.error("concurrency must be between 1 and 8")
    return args


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
