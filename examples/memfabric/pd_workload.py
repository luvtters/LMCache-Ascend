# SPDX-License-Identifier: Apache-2.0
"""Small correctness workloads for the ordinary, ungated MP P/D proxy."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx


def body(word, records=64, suffix=""):
    prompt = f"The password is {word}.\n" + "\n".join(
        f"Record {i:03d}: this document contains ordinary background material "
        "for a memory cache validation exercise." for i in range(records))
    return {"model": "mf-test", "temperature": 0, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content":
                 "Read the password from the document. Reply only with that password."},
                {"role": "user", "content": prompt + suffix + "\nWhat is the password?"}]}


async def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results = []

    def save(name, data):
        (out / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        async def snapshot(name):
            response = await client.get(args.service + "/status")
            response.raise_for_status()
            status = response.json()
            save(name, status)
            for label, port in [("p", args.p_port), ("d", args.d_port)]:
                r = await client.get(f"http://127.0.0.1:{port}/metrics")
                r.raise_for_status()
                (out / f"{name}_{label}_metrics.txt").write_text(r.text)
            return status

        async def settled(name):
            deadline = time.monotonic() + 15
            while True:
                r = await client.get(args.service + "/status")
                r.raise_for_status()
                sm = r.json()["storage_manager"]
                adapter = sm["l2_adapters"][0]
                ctrl = sm["store_controller"]
                if (ctrl["pending_keys_count"] == ctrl["in_flight_task_count"] == 0
                        and adapter["total_pins"] == 0
                        and sm["l1_manager"]["total_object_count"] == 0):
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError(f"Storage did not settle: {sm}")
                await asyncio.sleep(0.1)
            return await snapshot(name)

        async def infer(label, word, records=64, suffix=""):
            request = body(word, records, suffix)
            save(label + "_request", request)
            started = time.monotonic()
            result = {"label": label, "expected": word}
            try:
                r = await client.post(args.proxy + "/v1/chat/completions", json=request)
                r.raise_for_status()
                response = r.json()
                save(label + "_response", response)
                text = response["choices"][0]["message"]["content"].strip()
                result.update({"actual": text, "ok": text == word,
                               "id": response["id"], "usage": response["usage"]})
            except Exception as exc:
                result.update({"ok": False, "error": repr(exc)})
            result["elapsed_s"] = time.monotonic() - started
            results.append(result)
            print(json.dumps(result), flush=True)
            save("results", results)

        await snapshot("initial")
        if args.phase in ("quick", "serial"):
            cases = [("cold", "CEDAR", 64, ""),
                     ("repeat", "CEDAR", 64, ""),
                     ("partial", "CEDAR", 64, "\n" + "Additional context. " * 80),
                     ("short", "MAPLE", 1, "")]
            if args.phase == "serial":
                cases += [(f"evict_{i}", w, 64, "") for i, w in enumerate(
                    ["AMBER", "IVORY", "CORAL", "AZURE", "RAVEN", "TIGER"])]
                cases += [("old_prefix", "CEDAR", 64, "")]
            for case in cases:
                await infer(*case)
                await settled(case[0] + "_after")
        else:
            sem = asyncio.Semaphore(args.concurrency)

            async def task(i, word):
                async with sem:
                    await infer(f"parallel_{i}", word, args.records)

            words = (["BIRCH"] * 4 if args.phase == "shared" else
                     ["LOTUS", "PEARL", "FROST", "OLIVE", "ROBIN", "OTTER",
                      "MOSS", "FERN", "PLUM", "LILAC", "STONE", "FLAME"])
            await asyncio.gather(*(task(i, word) for i, word in enumerate(words)))
            await settled("parallel_after")
        final = await settled("final")
        adapter = final["storage_manager"]["l2_adapters"][0]
        summary = {"phase": args.phase, "requests": len(results),
                   "passed": sum(r["ok"] for r in results), "adapter": adapter}
        save("summary", summary)
        print("WORKLOAD_SUMMARY", json.dumps(summary), flush=True)
        assert all(r["ok"] for r in results), results
        assert adapter["is_healthy"] and adapter["total_pins"] == 0, adapter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["quick", "serial", "concurrent", "shared"],
                        default="quick")
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--p-port", type=int, default=28001)
    parser.add_argument("--d-port", type=int, default=28002)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
