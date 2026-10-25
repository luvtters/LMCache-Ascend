# SPDX-License-Identifier: Apache-2.0
"""Verify warm L2 restore is retained in L1 for a subsequent hot replay.

Requires an existing healthy B service with store=default/prefetch=retain and
the ordinary P/D proxy. This functional check is not a performance benchmark.
"""

import argparse
import json
import math
from pathlib import Path
import re
import time

import httpx


TOKEN_SOURCES = ("local_compute", "external_kv_transfer", "local_cache_hit")
SOURCE_LINE = re.compile(r'^vllm:prompt_tokens_by_source_total\{([^}]*)\}\s+(\S+)')
SOURCE_LABEL = re.compile(r'(?:^|,)\s*source="([^"]*)"(?:\s*,|\s*$)')


def source_counts(text):
    values = {source: None for source in TOKEN_SOURCES}
    for line in text.splitlines():
        match = SOURCE_LINE.match(line)
        label = SOURCE_LABEL.search(match.group(1)) if match else None
        if label is None or label.group(1) not in values:
            continue
        value = float(match.group(2))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid prompt source counter")
        source = label.group(1)
        values[source] = (values[source] or 0) + value
    if any(value is None for value in values.values()):
        raise ValueError("Missing prompt source counter; no reuse claim can be made")
    return values


def source_evidence(before_text, after_text, expected, minimum_external=None):
    before, after = source_counts(before_text), source_counts(after_text)
    delta = {source: after[source] - before[source] for source in TOKEN_SOURCES}
    if any(value < 0 for value in delta.values()):
        raise ValueError("Prompt source counter rolled back")
    observed = sum(delta.values())
    if observed > expected:
        raise ValueError("Prompt source window contains excess tokens/other traffic")
    aligned = observed == expected
    if aligned and minimum_external is not None:
        assert delta["external_kv_transfer"] >= minimum_external, "External KV restore was insufficient"
        assert delta["local_cache_hit"] == 0, "Local engine prefix cache masked the external path"
    return {"before": before, "after": after, "delta": delta,
            "expected_prompt_tokens": expected, "observed_prompt_tokens": observed,
            "minimum_external_tokens": minimum_external, "aligned": aligned}


def response_token_targets(responses):
    lengths = [response["usage"]["prompt_tokens"] for response in responses]
    assert all(isinstance(length, int) and length > 0 for length in lengths)
    return sum(lengths), sum(((length - 1) // 128) * 128 for length in lengths)


def pool(status):
    adapters = status["storage_manager"]["l2_adapters"]
    assert len(adapters) == 1 and adapters[0].get("protocol") == "SDMA"
    assert adapters[0].get("is_healthy"), "A healthy MemFabric B backend is required"
    return adapters[0]


def drained(status):
    sm = status["storage_manager"]
    ctrl, l1 = sm["store_controller"], sm["l1_manager"]
    return (ctrl.get("pending_keys_count") == ctrl.get("in_flight_task_count") == 0
            and sm["prefetch_controller"].get("in_flight_request_count") == 0
            and l1.get("read_locked_count") == l1.get("write_locked_count") == 0
            and pool(status).get("total_pins") == 0)


def request_body(word, nonce):
    text = f"The password is {word}.\nValidation nonce: {nonce}.\n"
    text += "\n".join(f"Record {i:03d}: ordinary background material for this key value cache validation."
                      for i in range(96))
    return {"model": "mf-test", "temperature": 0, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "Read the password in the document. Reply only with that password."},
                         {"role": "user", "content": text + "\nWhat is the password?"}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:28000")
    parser.add_argument("--service", default="http://127.0.0.1:28080")
    parser.add_argument("--prefiller", default="http://127.0.0.1:28001")
    parser.add_argument("--decoder", default="http://127.0.0.1:28002")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        (output / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    bodies = {word: request_body(word, output.name) for word in ("AURORA", "CORAL", "HARVEST")}
    save("requests", bodies)
    with httpx.Client(timeout=90, trust_env=False) as client:
        def metrics(name, when, deadline=None):
            result = {}
            for role, url in (("p", args.prefiller), ("d", args.decoder)):
                remaining = deadline - time.monotonic() if deadline is not None else 10
                if remaining <= 0:
                    raise TimeoutError("Prompt source metrics did not align within 15 seconds")
                response = client.get(url.rstrip("/") + "/metrics", timeout=min(10, remaining))
                response.raise_for_status()
                result[role] = response.text
                (output / f"{name}_{role}_metrics_{when}.txt").write_text(response.text, encoding="utf-8")
                source_counts(response.text)
            return result

        def aligned_metrics(name, before, responses):
            expected, minimum = response_token_targets(responses)
            deadline = time.monotonic() + 15
            while True:
                after = metrics(name, "after", deadline)
                report = {role: source_evidence(before[role], after[role], expected,
                                                minimum if name != "seed" else None) for role in ("p", "d")}
                save(name + "_prompt_token_sources", report)
                if all(value["aligned"] for value in report.values()):
                    return report
                if time.monotonic() >= deadline:
                    raise TimeoutError("Prompt source metrics did not align within 15 seconds")
                time.sleep(0.2)

        def status():
            response = client.get(args.service.rstrip("/") + "/status")
            response.raise_for_status()
            result = response.json()
            pool(result)
            return result

        def settled():
            deadline = time.monotonic() + 30
            while True:
                result = status()
                if drained(result):
                    return result
                if time.monotonic() >= deadline:
                    save("settle_timeout", result)
                    raise TimeoutError("Store, prefetch, locks or pool pins did not drain")
                time.sleep(0.2)

        def phase(name):
            deltas = {}
            metric_before = metrics(name, "before")
            responses = []
            for word, body in bodies.items():
                before = settled()
                response = client.post(args.proxy.rstrip("/") + "/v1/chat/completions", json=body)
                response.raise_for_status()
                answer = response.json()
                responses.append(answer)
                save(name + "_" + word + "_response", answer)
                assert answer["choices"][0]["message"]["content"].strip() == word
                after = settled()
                save(name + "_" + word + "_status", after)
                delta = pool(after)["load_bytes"] - pool(before)["load_bytes"]
                if name == "restore":
                    assert delta > 0, f"{word}: restore did not read BM"
                if name == "hot_replay":
                    assert delta == 0, f"{word}: hot replay still read BM"
                deltas[word] = delta
            save(name + "_status", after)
            return after, deltas, aligned_metrics(name, metric_before, responses)

        initial = settled()
        save("initial_status", initial)
        seeded, _, seed_sources = phase("seed")
        assert pool(seeded)["objects"] > 0
        assert pool(seeded)["store_bytes"] > pool(initial)["store_bytes"]
        response = client.post(args.service.rstrip("/") + "/cache/clear", json={"tier": "l1", "force": False})
        response.raise_for_status()
        save("clear_response", response.json())
        cleared = settled()
        save("after_clear_status", cleared)
        l1 = cleared["storage_manager"]["l1_manager"]
        assert l1["total_object_count"] == l1["memory_used_bytes"] == 0
        assert pool(cleared)["objects"] > 0, "L2 objects disappeared when clearing L1"
        restored, restore_bytes, restore_sources = phase("restore")
        assert restored["storage_manager"]["l1_manager"]["total_object_count"] > 0
        hot, hot_bytes, hot_sources = phase("hot_replay")
        assert hot["storage_manager"]["l1_manager"]["total_object_count"] > 0 and drained(hot)
        summary = {"status": "RETAIN_SMOKE_PASS", "nonce": output.name, "correct_answers": 9,
                   "seed_store_bytes_delta": pool(seeded)["store_bytes"] - pool(initial)["store_bytes"],
                   "restore_load_bytes_by_request": restore_bytes, "hot_replay_load_bytes_by_request": hot_bytes,
                   "prompt_token_sources": {"seed": seed_sources, "restore": restore_sources, "hot_replay": hot_sources},
                   "l1_objects_after_restore": restored["storage_manager"]["l1_manager"]["total_object_count"],
                   "l1_objects_after_hot_replay": hot["storage_manager"]["l1_manager"]["total_object_count"],
                   "read_locked_count": 0, "write_locked_count": 0, "pool_pins": 0}
        save("summary", summary)
        print("RETAIN_SMOKE_PASS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
