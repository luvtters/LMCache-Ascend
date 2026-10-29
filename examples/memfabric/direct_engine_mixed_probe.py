#!/usr/bin/env python3
"""Experiment-only warm/partial and warm/cold HTTP pairs with scheduler proof."""
import argparse
import json
from pathlib import Path
import re

import direct_followups as follow


def scheduler_id_matches_api_id(scheduler_id, api_id):
    """Installed vLLM assigns external ID + '-' + first eight lower hex chars.

    Accept the complete API ID itself when randomization is disabled. Never
    remove arbitrary suffixes or match client-ID substrings.
    """
    return scheduler_id == api_id or re.fullmatch(
        re.escape(api_id) + r"-[0-9a-f]{8}", scheduler_id
    ) is not None


def scheduler_evidence(before, after, responses):
    """Only a single actual D scheduling step containing both request IDs counts."""
    assert before["host_id"] == after["host_id"] and before["run_id"] == after["run_id"]
    assert len(responses) == 2
    assert len({row["response_request_id"] for row in responses}) == 2
    assert len({row["client_request_id"] for row in responses}) == 2
    api_ids = [row["response_request_id"] for row in responses]
    steps = [row for row in after["records"]
             if row["kind"] == "scheduler_step" and row["role"] == "D"
             and row["mono_ns"] >= before["mono_ns"]]
    ambiguous_steps = 0
    for row in steps:
        assert row["host_id"] == after["host_id"] and row["session_id"] == after["run_id"]
        ids = [key for key, value in row["num_scheduled_tokens"].items() if value > 0]
        owners = {key: [index for index, api_id in enumerate(api_ids)
                        if scheduler_id_matches_api_id(key, api_id)] for key in ids}
        matches = [{key for key, indices in owners.items() if index in indices}
                   for index in range(2)]
        if any(len(indices) > 1 for indices in owners.values()) or any(len(keys) > 1 for keys in matches):
            ambiguous_steps += 1
            continue
        if all(len(keys) == 1 for keys in matches) and matches[0].isdisjoint(matches[1]):
            return dict(proven=True, scheduler_pid=row["pid"], scheduler_step=row["scheduler_step"],
                        mono_ns=row["mono_ns"], num_scheduled_tokens=row["num_scheduled_tokens"],
                        response_ids=api_ids, scheduler_request_ids=[next(iter(keys)) for keys in matches],
                        id_contract="Exact API response ID, optionally followed by '-' and 8 lower hex characters",
                        boundary="Two successful HTTP requests scheduled in one D engine step")
    return dict(proven=False, observed_scheduler_steps=len(steps),
                ambiguous_scheduler_steps=ambiguous_steps,
                reason="No same-step positive-token pair linked to these HTTP identities")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        base = json.loads((args.gate_dir / "request.json").read_text(encoding="utf-8"))
        bodies = follow.variants(base)  # Original ORCHID gate, not the VELM42 variant.
        token_ids = {name: follow.tokenize(args, body, args.output / "tokenize" / name)
                     for name, body in bodies.items()}
        reference = token_ids["warm"]
        items = {name: dict(variant=name, body=body, tokens=token_ids[name],
                           chunks=follow.common_chunks(reference, token_ids[name], args.chunk_size))
                 for name, body in bodies.items()}
        assert 0 < items["partial"]["chunks"] < items["warm"]["chunks"]
        assert items["cold"]["chunks"] == 0
        follow.save(args.output / "token_geometry.derived.json", {
            name: dict(prompt_tokens=len(item["tokens"]), shared_chunks=item["chunks"])
            for name, item in items.items()})
        results = []
        for name, selected in (("warm_partial", ["warm", "partial"]),
                               ("warm_cold", ["warm", "cold"])):
            folder = args.output / name
            result = follow.run_case(args, name, [items[key] for key in selected], folder)
            before = json.loads((folder / "before/trace/response.body.bin").read_bytes())
            latest = max(folder.glob("after_*"), key=lambda path: int(path.name.split("_")[-1]))
            after = json.loads((latest / "trace/response.body.bin").read_bytes())
            proof = scheduler_evidence(before, after, result["requests"])
            result["scheduler_evidence"] = proof
            result["engine_concurrency"] = "SAME_STEP_TWO_REQUESTS_PROVEN" if proof["proven"] else "NOT_PROVEN"
            follow.save(folder / "mixed_summary.derived.json", result)
            results.append(result)
        proven = all(row["scheduler_evidence"]["proven"] for row in results)
        summary = dict(status="ENGINE_MIXED_SCHEDULING_FUNCTION_PASS" if proven
                       else "FUNCTION_PASS_SCHEDULING_NOT_PROVEN", cases=results,
                       evidence_boundary="Scheduler admission/steps; not asynchronous BM or performance",
                       lifecycle="Outer executor retains original JSONL, real waits and release")
        follow.save(args.output / "summary.derived.json", summary)
        return summary
    except BaseException as error:
        follow.save(args.output / "failure.derived.json", dict(status="FAIL", error=repr(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--service-http", required=True)
    parser.add_argument("--trace-url", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--expected-pool-memory-type", choices=("host", "device"), default="device")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--settle-timeout", type=float, default=10)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or args.timeout <= 0 or args.settle_timeout <= 0:
        parser.error("Chunk size and timeouts must be positive")
    if not args.run:
        print(json.dumps(dict(status="PLAN_ONLY", cases=["warm_partial", "warm_cold"],
            original_orchid_gate_required=True, engine_max_num_seqs_required=2,
            no_device_or_service_launch=True, scheduler_proof_required_for_mixed_claim=True)))
        return
    result = run(args)
    print(json.dumps(dict(status=result["status"])))


if __name__ == "__main__":
    main()
