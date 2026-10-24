# SPDX-License-Identifier: Apache-2.0
"""Cancel streaming P/D requests after decoding starts; check lock cleanup."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx

from pd_workload import body


async def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2))

    def aborted(text):
        return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                   if line.startswith("vllm:request_success_total{")
                   and 'finished_reason="abort"' in line)

    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        decoder_log = Path(args.decoder_log)
        log_offset = decoder_log.stat().st_size
        before = (await client.get(args.decoder + "/metrics")).text
        (output / "metrics_before.txt").write_text(before)
        observed = []
        for i in range(3):
            request = body("ORBIT")
            request["messages"][0]["content"] = (
                "Repeat BLUE separated by spaces. Continue until the token limit.")
            request["messages"][1]["content"] += f"\nStreaming round {i}."
            request.update(stream=True, max_tokens=256, ignore_eos=True)
            chunks = []
            async with client.stream("POST", args.proxy + "/v1/chat/completions",
                                     json=request) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    choices = data.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        chunks.append(data)
                        if len(chunks) == 4:
                            break
            assert len(chunks) == 4, "Stream completed before intentional cancellation"
            save(f"cancel_{i}_chunks", chunks)
            observed.append(chunks[0]["id"])
        deadline = time.monotonic() + 20
        while True:
            r = await client.get(args.service + "/status")
            r.raise_for_status()
            status = r.json()
            sm = status["storage_manager"]
            after = (await client.get(args.decoder + "/metrics")).text
            with decoder_log.open("rb") as handle:
                handle.seek(log_offset)
                log = handle.read().decode(errors="replace")
            abort_lines = [line for line in log.splitlines()
                           if "Aborted request" in line or "aborted." in line]
            # This vLLM version removes explicitly aborted requests before its
            # finished-request metric update. Verify the actual engine log.
            confirmed = [rid for rid in observed
                         if any(rid in line for line in abort_lines)]
            if (sm["l2_adapters"][0]["total_pins"] == 0
                    and sm["store_controller"]["pending_keys_count"] == 0
                    and sm["store_controller"]["in_flight_task_count"] == 0
                    and sm["prefetch_controller"]["in_flight_request_count"] == 0
                    and sm["l1_manager"]["total_object_count"] == 0
                    and len(confirmed) == 3):
                break
            if time.monotonic() >= deadline:
                save("cleanup_timeout", status)
                (output / "metrics_timeout.txt").write_text(after)
                raise AssertionError("Cancellation metrics or storage cleanup incomplete")
            await asyncio.sleep(0.2)
        save("after_cancel", status)
        (output / "decoder_abort_evidence.txt").write_text("\n".join(abort_lines))
        (output / "metrics_after.txt").write_text(after)
        r = await client.post(args.proxy + "/v1/chat/completions", json=body("ORBIT"))
        r.raise_for_status()
        response = r.json()
        save("followup_response", response)
        assert response["choices"][0]["message"]["content"].strip() == "ORBIT"
        summary = {"aborted_request_ids": confirmed,
                   "decoder_abort_metric_delta": aborted(after) - aborted(before),
                   "remaining_pins": 0, "followup_answer": "ORBIT"}
        save("summary", summary)
        print("CANCEL_CLEANUP_PASS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--decoder", default="http://127.0.0.1:28002")
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--decoder-log", required=True,
                        help="D log, started with --enable-log-requests")
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
