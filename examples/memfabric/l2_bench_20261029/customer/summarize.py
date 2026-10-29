#!/usr/bin/env python3
"""Offline, partial-safe report of the four customer-cropped L2 cases."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

from generate import CASES
from run import drain_check


SOURCES = ("external_kv_transfer", "local_compute", "local_cache_hit")
STATUS = "PERFORMANCE_DIAGNOSTIC_BOTH_EXIT_GATES_FAILED"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_optional(path, errors):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append({"path": str(path), "error": str(exc)})
        return None


def source_counts(path):
    if not path.exists():
        return {key: None for key in SOURCES}
    values = {key: None for key in SOURCES}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^vllm:prompt_tokens_by_source_total\{([^}]*)\}\s+(\S+)', line)
        label = re.search(r'(?:^|,)\s*source="([^"]*)"(?:\s*,|\s*$)', match.group(1)) if match else None
        if label and label.group(1) in values:
            try:
                value = float(match.group(2))
            except ValueError:
                return {key: None for key in SOURCES}
            if not finite(value) or value < 0:
                return {key: None for key in SOURCES}
            source = label.group(1)
            values[source] = value if values[source] is None else values[source] + value
    return values


def delta(before, after):
    return after - before if finite(before) and finite(after) and after >= before else None


def status_evidence(before, after):
    a = (before or {}).get("storage_manager", {}).get("l2_adapters", [])
    b = (after or {}).get("storage_manager", {}).get("l2_adapters", [])
    adapters = []
    for index in range(max(len(a), len(b))):
        first, last = a[index] if index < len(a) else {}, b[index] if index < len(b) else {}
        keys = sorted(set(first) | set(last))
        # Keep actual byte, object, operation and protection fields; never add a
        # prompt-budget-derived payload to a missing native counter.
        selected = [key for key in keys if re.search(r"bytes|objects|count|pins|slots|operations|stores|loads|evict", key)]
        adapters.append({"index": index, "fields": {key: {"before": first.get(key), "after": last.get(key),
            "delta": (last[key] - first[key] if finite(first.get(key)) and finite(last.get(key))
                      and key in ("objects", "free_slots", "locked_objects", "total_pins")
                      else delta(first.get(key), last.get(key)))} for key in selected},
            "before_metadata": {key: value for key, value in first.items() if key not in selected},
            "after_metadata": {key: value for key, value in last.items() if key not in selected}})
    return {"adapters": adapters, "before_drain": drain_check(before or {}),
            "after_drain": drain_check(after or {}), "missing_counters_are_unknown": True}


def metric(native, name, statistic):
    value = native.get(statistic + "_" + name + "_ms")
    if finite(value):
        return value, "native_benchmark_ms"
    if name == "ttft":
        values = native.get("ttfts")
        if isinstance(values, list) and len(values) == 8 and all(finite(value) and value >= 0 for value in values):
            if statistic == "mean":
                return sum(values) / len(values) * 1000, "native_ttfts_seconds_mean"
            ordered = sorted(values)
            position = (len(values) - 1) * .95
            low, high = math.floor(position), math.ceil(position)
            return (ordered[low] + (ordered[high] - ordered[low]) * (position - low)) * 1000, "native_ttfts_seconds_linear_p95"
    return None, "unknown_no_reliable_native_field"


def read_record(folder, root, errors):
    manifest = read_optional(folder / "manifest.json", errors) or {}
    summary = read_optional(folder / "summary.json", errors) or {}
    native = read_optional(folder / "benchmark.json", errors) or {}
    validation = read_optional(folder / "customer_validation.json", errors)
    before = read_optional(folder / "before_status.json", errors)
    after = read_optional(folder / "after_status.json", errors)
    case = manifest.get("case_id", folder.parent.name)
    arm = summary.get("arm")
    if arm == "mooncake":
        arm = "store"
    if arm not in ("store", "memfabric"):
        arm = next((name for name in ("store", "memfabric")
                    if any(part.endswith("_" + name) for part in folder.parts)), None)
    specification = next((row for row in CASES if row[0] == case), None)
    if specification is None or arm is None:
        raise ValueError("Cannot identify one of the four case IDs and its backend")
    tag = next((part[:-(len(arm) + 1)] for part in reversed(folder.parts)
                if part.endswith("_" + arm)), None)
    if tag is None:
        raise ValueError("Cannot identify the backend-tag directory")
    _, inputs, prefix, outputs = specification
    expected = {"p": {"external_kv_transfer": 8 * prefix, "local_compute": 8 * (inputs - prefix), "local_cache_hit": 0},
                "d": {"external_kv_transfer": 8 * (inputs - 1), "local_compute": 8, "local_cache_hit": 0}}
    sources = {}
    for role in ("p", "d"):
        first, last = [source_counts(folder / (phase + "_" + role + "_metrics.txt")) for phase in ("before", "after")]
        sources[role] = {key: {"before": first[key], "after": last[key], "delta": delta(first[key], last[key])} for key in SOURCES}
    observed = {role: {key: value["delta"] for key, value in values.items()} for role, values in sources.items()}
    evidence = status_evidence(before, after)
    requests = []
    request_path = folder / "requests.jsonl"
    if request_path.exists():
        try:
            requests = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, ValueError) as exc:
            errors.append({"path": str(request_path), "error": str(exc)})
    required = ("manifest.json", "summary.json", "benchmark.json", "customer_validation.json", "before_status.json", "after_status.json", "requests.jsonl",
                "before_p_metrics.txt", "after_p_metrics.txt", "before_d_metrics.txt", "after_d_metrics.txt")
    missing = [name for name in required if not (folder / name).exists()]
    checks = {"completed_8_failed_0": native.get("completed") == 8 and native.get("failed") == 0,
              "input_total": native.get("total_input_tokens") == 8 * inputs,
              "output_total": native.get("total_output_tokens") == 8 * outputs,
              "input_lengths": native.get("input_lens") == [inputs] * 8,
              "output_lengths": native.get("output_lens") == [outputs] * 8,
              "error_array_empty": isinstance(native.get("errors"), list) and len(native["errors"]) == 8 and not any(native["errors"]),
              "eight_texts_recorded": isinstance(native.get("generated_texts"), list) and len(native["generated_texts"]) == 8 and all(isinstance(value, str) for value in native["generated_texts"]),
              "eight_inputs_recorded": len(requests) == 8 and all(isinstance(value.get("prompt"), str) and value["prompt"] for value in requests),
              "source_deltas": observed == expected,
              "before_l1_and_locks_zero": evidence["before_drain"]["all_observed_zero"],
              "after_l1_and_locks_zero": evidence["after_drain"]["all_observed_zero"],
              "customer_validation_passed": isinstance(validation, dict) and validation.get("passed") is True,
              "observe_native_checks": summary.get("valid_counts") is True,
              "duration_positive": finite(native.get("duration")) and native["duration"] > 0}
    metrics, provenance = {}, {}
    for name in ("ttft", "tpot"):
        for statistic in ("mean", "p95"):
            key = statistic + "_" + name + "_ms"
            metrics[key], provenance[key] = metric(native, name, statistic)
    metrics["duration_seconds"] = native.get("duration")
    metrics["output_token_s"] = native.get("total_output_tokens", 0) / native["duration"] if checks["duration_positive"] else None
    row = {"case_id": case, "dataset_id": manifest.get("dataset_id", summary.get("dataset_id")), "arm": arm, "tag": tag,
           "result_directory": folder.relative_to(root).as_posix(), "input_tokens": inputs, "prefix_tokens": prefix,
           "output_tokens": outputs, "concurrency": 1, "rounds": 1, "requested_requests": 8,
           "completed": native.get("completed"), "failed": native.get("failed"),
           "native_total_input_tokens": native.get("total_input_tokens"),
           "native_total_output_tokens": native.get("total_output_tokens"),
           "native_input_lens": native.get("input_lens"), "native_output_lens": native.get("output_lens"),
           "native_errors": native.get("errors"), "checks": checks, "missing_files": missing,
           "request_source_and_drain_checks_passed": not missing and all(checks.values()),
           "metrics": metrics, "metric_provenance": provenance,
           "expected_source_deltas": expected, "actual_source_counters": sources,
           "status_evidence": evidence, "customer_validation": validation,
           "before_customer_drain": read_optional(folder / "before_customer_drain.json", errors),
           "after_customer_drain": read_optional(folder / "after_customer_drain.json", errors),
           "performance_status": STATUS, "exit_gate": "FAILED_FOR_BOTH_BACKENDS_PER_CURRENT_EXPERIMENT",
           "l2_route_acceptance": {"state": "REQUIRES_NATIVE_ROUTE_LOG_AUDIT",
               "empty_snapshots_and_vllm_external_do_not_prove_all_l2": True,
               "external_audit_note": ("customer01 short-case native-log audit reported D 8 L1 / 24 L2 blocks per request for Store, versus 0 L1 / 32 L2 for MemFabric; attach the source logs separately"
                   if tag == "customer01" and case == "in4096_p50_out128" else None),
               "audit_note_provenance": "explicit task instruction; this parser does not derive native route classification from snapshots"},
           "sample_scope": "one round of eight requests; diagnostic only; no significance claim",
           "budget_note": "1680 chunks/52.5 GiB slots was prompt-only design budget; actual store/object/byte counters above remain authoritative, including any generated-token writes"}
    return row, {"native": native, "requests": requests}


def compare(store, memfabric, details):
    if store is None or memfabric is None:
        return {"status": "MISSING_ARM_NO_COMPARISON", "compared_requests": 0}
    if store["dataset_id"] != memfabric["dataset_id"]:
        return {"status": "DATASET_ID_MISMATCH_NO_COMPARISON", "compared_requests": 0}
    a, b = details[store["result_directory"]], details[memfabric["result_directory"]]
    if any(len(values["requests"]) != 8 or any(not isinstance(values["native"].get(key), list) or len(values["native"][key]) != 8
              for key in ("generated_texts", "input_lens", "output_lens", "errors")) for values in (a, b)):
        return {"status": "INCOMPLETE_REQUEST_ARRAYS_NO_COMPARISON", "compared_requests": 0}
    rows = []
    for index in range(8):
        left, right = a["native"], b["native"]
        same_input = a["requests"][index].get("prompt") == b["requests"][index].get("prompt")
        text_equal = left["generated_texts"][index] == right["generated_texts"][index]
        row = {"request_index_zero_based": index, "input_prompt_literal_equal": same_input,
               "input_lengths": {"store": left["input_lens"][index], "memfabric": right["input_lens"][index]},
               "output_lengths": {"store": left["output_lens"][index], "memfabric": right["output_lens"][index]},
               "errors": {"store": left["errors"][index], "memfabric": right["errors"][index]},
               "generated_text_literal_equal": text_equal}
        if not text_equal:
            row["store_generated_text"] = left["generated_texts"][index]
            row["memfabric_generated_text"] = right["generated_texts"][index]
        rows.append(row)
    inputs_match = all(row["input_prompt_literal_equal"] for row in rows)
    unequal = [row["request_index_zero_based"] for row in rows if not row["generated_text_literal_equal"]]
    gates = store["request_source_and_drain_checks_passed"] and memfabric["request_source_and_drain_checks_passed"]
    state = "COMPARED_LITERAL_TEXT" if gates else "COMPARED_LITERAL_TEXT_WITH_REQUEST_OR_SOURCE_GATE_FAILURE"
    return {"status": state if inputs_match else "INPUT_IDENTITY_MISMATCH_NOT_A_MATCHED_OUTPUT_COMPARISON",
            "compared_requests": 8 if inputs_match else 0,
            "both_request_source_and_drain_gates_passed": gates,
            "literal_input_prompts_all_equal": inputs_match,
            "generated_text_equal_count": 8 - len(unequal) if inputs_match else None,
            "generated_text_mismatch_indices_zero_based": unequal, "requests": rows,
            "method": "same case/dataset/C1/request index; exact input and output strings; no trimming or normalization",
            "token_id_equality": "NOT_TESTED_NATIVE_FILES_RECORD_TEXT_NOT_TOKEN_IDS"}


def display(value):
    return f"{value:.3f}" if finite(value) else "unknown"


def aggregate(case, arm, records, tags, details):
    if len(tags) != 1:
        raise ValueError("Each tag is a separate protocol; cross-tag aggregation is prohibited")
    rows = [row for row in records if row["tag"] == tags[0] and row["case_id"] == case and row["arm"] == arm]
    valid = [row for row in rows if row["request_source_and_drain_checks_passed"]
             and sum(other["tag"] == row["tag"] for other in rows) == 1]
    duration = sum(row["metrics"]["duration_seconds"] for row in valid)
    outputs = sum(row["native_total_output_tokens"] for row in valid)
    count = sum(row["completed"] for row in valid)
    rates = [row["metrics"]["output_token_s"] for row in valid]
    group = {"tag": tags[0], "case_id": case, "arm": arm, "observed_tag_ids": tags,
             "recorded_rounds": len(rows), "matched_rounds": len(valid),
             "matched_tag_ids": [row["tag"] for row in valid],
             "missing_or_failed_tag_ids": [tag for tag in tags if not any(row["tag"] == tag for row in valid)],
             "total_completed": count, "total_output_tokens": outputs, "total_duration_seconds": duration,
             "output_token_s": outputs / duration if duration else None,
             "per_round_output_token_s": rates, "round_output_token_s_min": min(rates) if rates else None,
             "round_output_token_s_max": max(rates) if rates else None,
             "round_output_token_s_population_std": statistics.pstdev(rates) if rates else None,
             "performance_status": STATUS, "rounds": rows,
             "aggregation": "same tag only; complete request/source/drain rounds; sum actual output / sum duration; request-weighted native means; native route acceptance is separate"}
    for name in ("ttft", "tpot"):
        key = "mean_" + name + "_ms"
        group[key] = sum(row["metrics"][key] * row["completed"] for row in valid) / count if count and all(finite(row["metrics"][key]) for row in valid) else None
        p95s = [row["metrics"]["p95_" + name + "_ms"] for row in valid]
        recorded = [value for value in p95s if finite(value)]
        arrays = [details[row["result_directory"]]["native"].get("ttfts" if name == "ttft" else "tpots") for row in valid]
        reliable = bool(arrays) and all(isinstance(values, list) and len(values) == 8 and all(finite(value) and value >= 0 for value in values) for values in arrays)
        pooled = sorted(value * 1000 for values in arrays for value in values) if reliable else []
        position = (len(pooled) - 1) * .95 if pooled else 0
        p95 = pooled[math.floor(position)] + (pooled[math.ceil(position)] - pooled[math.floor(position)]) * (position - math.floor(position)) if pooled else None
        group["p95_" + name + "_ms"] = p95
        group["p95_" + name + "_method"] = "complete_native_per_request_arrays_linear_percentile" if pooled else "per_round_p95_range_only"
        group["p95_" + name + "_pooled_samples"] = len(pooled)
        group["per_round_p95_" + name + "_ms"] = p95s
        group["round_p95_" + name + "_ms_min"] = min(recorded) if recorded else None
        group["round_p95_" + name + "_ms_max"] = max(recorded) if recorded else None
    return group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if not root.is_dir():
        p.error("root must be an existing local archive directory")
    records, details, errors = [], {}, []
    for folder in sorted(path for path in root.rglob("timed") if path.is_dir()):
        try:
            row, raw = read_record(folder, root, errors)
            records.append(row)
            details[row["result_directory"]] = raw
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append({"path": str(folder), "error": str(exc)})
    pairs = []
    tags = sorted({row["tag"] for row in records})
    for tag in tags or ["NO_RECORDED_TAG"]:
      for case, inputs, prefix, outputs in CASES:
        arms = {arm: [row for row in records if row["tag"] == tag and row["case_id"] == case and row["arm"] == arm] for arm in ("store", "memfabric")}
        unique = {arm: rows[0] if len(rows) == 1 else None for arm, rows in arms.items()}
        ambiguous = any(len(rows) > 1 for rows in arms.values())
        consistency = {"status": "DUPLICATE_RECORDS_NO_COMPARISON", "compared_requests": 0} if ambiguous else compare(unique["store"], unique["memfabric"], details)
        pairs.append({"tag": tag, "case_id": case, "input_tokens": inputs, "prefix_tokens": prefix, "output_tokens": outputs,
                      "arm_record_counts": {arm: len(rows) for arm, rows in arms.items()},
                      "arms": unique, "output_consistency": consistency,
                      "performance_status": STATUS})
    compared = sum(pair["output_consistency"]["compared_requests"] for pair in pairs)
    equal = sum(pair["output_consistency"].get("generated_text_equal_count") or 0 for pair in pairs)
    groups = [aggregate(case, arm, [row for row in records if row["tag"] == tag], [tag], details)
              for tag in tags or ["NO_RECORDED_TAG"] for case, _, _, _ in CASES for arm in ("store", "memfabric")]
    report = {"archive_root": str(root), "performance_status": STATUS, "observed_tag_ids": tags,
              "aggregation_protocol": "TAG_ISOLATION_NO_CROSS_TAG_POOLING_OR_AVERAGING",
              "route_evidence_rule": "L1-zero boundary snapshots and vLLM external counters cannot distinguish L1 from L2 retrieval; native route logs are required",
              "exit_gate_source": "explicit current task: both backend exit gates failed; client counts cannot override this",
              "partial": any(pair["output_consistency"]["compared_requests"] != 8 or
                             any(row is None or not row["request_source_and_drain_checks_passed"] for row in pair["arms"].values()) for pair in pairs),
              "expected_cases": 4, "expected_timed_requests_per_arm_per_case": 8,
              "expected_total_timed_requests_both_arms_per_tag": 64,
              "sample_scope": "eight-request round per tag/case/arm; no statistical significance claim; unstarted tags are not fabricated",
              "total_output_pairs_compared": compared, "total_literal_text_equal": equal,
              "total_literal_text_mismatches": compared - equal,
              "pairs": pairs, "case_arm_aggregates": groups, "all_records_including_duplicates": records, "read_errors": errors}
    output = a.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["tag", "case_id", "arm", "record_state", "completed", "failed", "output_token_s", "duration_seconds",
              "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "request_source_and_drain_checks_passed",
              "literal_text_equal_count", "performance_status"]
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            for arm in ("store", "memfabric"):
                row = pair["arms"][arm]
                values = {"tag": pair["tag"], "case_id": pair["case_id"], "arm": arm,
                    "record_state": "PRESENT" if row else "MISSING_OR_DUPLICATE", "performance_status": STATUS,
                    "literal_text_equal_count": pair["output_consistency"].get("generated_text_equal_count")}
                if row:
                    values.update({key: row[key] for key in ("completed", "failed", "request_source_and_drain_checks_passed")})
                    values.update(row["metrics"])
                writer.writerow(values)
    lines = ["# Customer-cropped compatible L2 diagnostics", "", "Both backend exit gates remain FAILED. Eight requests per tag/case/arm are diagnostic samples; no significance or formal performance acceptance is claimed.",
             "Each tag is a separate protocol: customer01 and customer02_strict are never pooled or averaged. Partial or duplicate pairs are explicitly excluded from matched output comparison. All original records, missing files, source counters, actual backend bytes/objects/evictions, L1/protection state and output mismatches are retained in JSON.",
             "Empty L1 snapshots and vLLM external counters do not prove all-L2 retrieval. The reported customer01 short-case native-log audit found Store D 8 L1 / 24 L2 blocks per request versus MemFabric 0 L1 / 32 L2; attach and review those external source logs. This parser does not reconstruct that route classification from snapshots.", "",
             "Observed tags: " + ", ".join(tags or ["none"]) + ".", "",
             "| tag / protocol | case | arm | matched/observed tag rounds | token/s | total duration s | TTFT mean/p95 ms | TPOT mean/p95 ms or round range |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for group in groups:
        def p95(name):
            value = group["p95_" + name + "_ms"]
            return display(value) + " pooled" if value is not None else display(group["round_p95_" + name + "_ms_min"]) + "-" + display(group["round_p95_" + name + "_ms_max"]) + " round range"
        lines.append(f"| {group['tag']} | {group['case_id']} | {group['arm']} | {group['matched_rounds']}/{len(group['observed_tag_ids'])} | {display(group['output_token_s'])} | {display(group['total_duration_seconds'])} | {display(group['mean_ttft_ms'])}/{p95('ttft')} | {display(group['mean_tpot_ms'])}/{p95('tpot')} |")
    lines += ["", "| tag | case | arm records store/MF | literal output equal | pair state |", "|---|---|---|---|---|"]
    for pair in pairs:
        c = pair["output_consistency"]
        texts = str(c.get("generated_text_equal_count")) + "/8" if c["compared_requests"] else "not compared"
        lines.append(f"| {pair['tag']} | {pair['case_id']} | {pair['arm_record_counts']['store']}/{pair['arm_record_counts']['memfabric']} | {texts} | {c['status']} |")
    lines += ["", f"Literal output equality: {equal}/{compared} compared request pairs. Partial report: {report['partial']}. Read errors: {len(errors)}.",
              "Throughput is sum(actual output)/sum(native duration) within the same tag only. Means are request-weighted within that tag. Percentiles pool complete native request arrays only within that tag when available; otherwise each round p95 is retained as a range and never averaged.",
              "The earlier 1680-chunk/52.5-GiB budget is a prompt-only design estimate, never a measured object/byte total. Any additional generated-token writes remain in the actual counter deltas. Missing Store byte counters remain unknown; no byte count is inferred from prompt tokens or successful chunks."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "partial": report["partial"], "compared_requests": compared,
                      "literal_text_equal": equal, "literal_text_mismatches": compared - equal,
                      "read_errors": len(errors), "performance_status": STATUS}, indent=2))


if __name__ == "__main__":
    main()
