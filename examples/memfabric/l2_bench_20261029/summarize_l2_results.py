#!/usr/bin/env python3
"""Summarize archived compatible-L2 rounds without contacting any service.

Example: python summarize_l2_results.py --root ../collected/results
    --output-prefix ../results_summary
Defaults: 32 requests, 32 output tokens and three rounds per dataset/arm/C.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


SOURCES = ("external_kv_transfer", "local_compute", "local_cache_hit")
STORE_STATUS = "PERFORMANCE_DIAGNOSTIC_EXIT_GATE_FAILED"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional_read(path):
    return read(path) if path.exists() else None


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def measured_metrics(native, summary):
    selected = dict(summary.get("metrics", {}))
    selected.update({key: value for key, value in native.items()
                     if key in ("duration", "output_throughput", "request_throughput")
                     or key.endswith(("_ttft_ms", "_tpot_ms", "_e2el_ms"))})
    return selected


def l1_snapshot(status):
    if not isinstance(status, dict):
        return None
    manager = status.get("storage_manager", {})
    values = manager.get("l1_managers")
    if not isinstance(values, dict):
        single = manager.get("l1_manager")
        values = {"_default": single} if isinstance(single, dict) else {}
    keys = ("total_object_count", "memory_used_bytes", "temporary_count", "staging_object_count",
            "staging_bytes", "read_locked_count", "write_locked_count", "is_healthy")
    return {name: {key: value.get(key) for key in keys} for name, value in values.items()
            if isinstance(value, dict)} or None


def l1_empty(values):
    if not values or any(not finite(row.get(key)) for row in values.values()
                         for key in ("total_object_count", "memory_used_bytes")):
        return None
    return all(row[key] == 0 for row in values.values()
               for key in ("total_object_count", "memory_used_bytes"))


def counter_evidence(summary):
    chunks, byte_counters = {}, {}
    for role, values in summary.get("observed_counter_deltas", {}).items():
        for name, value in (values or {}).items():
            key = role + ":" + name
            if re.search(r"num_chunks_loaded|(?:l2|L2).*(?:load|read|retrieve).*(?:chunk|success)", name):
                chunks[key] = value
            if re.search(r"(?:load|read|store|write).*bytes|bytes.*(?:load|read|store|write)", name, re.I):
                byte_counters[key] = value
    actual_read_bytes = {key: value for key, value in byte_counters.items()
                         if re.search(r"(?:load|read).*bytes|bytes.*(?:load|read)", key, re.I)}
    observed_positive = any(finite(value.get("delta")) and value["delta"] > 0
                            for value in actual_read_bytes.values())
    return {"successful_chunk_counters": chunks, "recorded_byte_counters": byte_counters,
            "actual_read_byte_counters": actual_read_bytes,
            "read_bytes_state": "OBSERVED_POSITIVE_COUNTER" if observed_positive else "NOT_ESTABLISHED",
            "bytes_are_observed_never_inferred": True,
            "chunk_counters_alone_do_not_prove_l2_transport": True}


def detailed_samples(native, name, count):
    """Return only complete per-request seconds arrays with a known meaning."""
    def array(key):
        values = native.get(key)
        return values if isinstance(values, list) and len(values) == count and all(
            finite(value) and value >= 0 for value in values) else None
    if name == "ttft":
        return array("ttfts"), "native_ttfts_seconds"
    if name == "e2el":
        return array("latencies"), "native_latencies_seconds"
    values = array("tpots")
    if values is not None:
        return values, "native_tpots_seconds"
    latencies, ttfts, lengths = array("latencies"), array("ttfts"), native.get("output_lens")
    if latencies is not None and ttfts is not None and isinstance(lengths, list) and len(lengths) == count:
        if all(finite(length) and length > 1 and latency >= ttft
               for latency, ttft, length in zip(latencies, ttfts, lengths)):
            return [(latency - ttft) / (length - 1)
                    for latency, ttft, length in zip(latencies, ttfts, lengths)], "derived_per_request_from_native_latency_ttft_output_length"
    return None, "no_complete_reliable_per_request_array"


def load_round(path, root, args):
    summary = read(path)
    folder = path.parent
    native = read(folder / "benchmark.json")
    count, output = args.expected_requests, args.expected_output_tokens
    input_tokens = summary.get("input_tokens_per_request")
    prefix = input_tokens - 127 if finite(input_tokens) and input_tokens - 127 in (1024, 3072) else None
    arm = summary.get("arm")
    arm = "store" if arm in ("store", "mooncake") else arm
    checks = {"expected_prefix": prefix is not None,
              "completed": native.get("completed") == count, "failed": native.get("failed") == 0,
              "total_input": prefix is not None and native.get("total_input_tokens") == count * input_tokens,
              "total_output": native.get("total_output_tokens") == count * output,
              "per_request_output": native.get("output_lens") == [output] * count,
              "duration": finite(native.get("duration")) and native["duration"] > 0,
              "observe_count_and_exit_checks": summary.get("valid_counts") is True}
    if "input_lens" in native:
        checks["per_request_input"] = native["input_lens"] == [input_tokens] * count
    alignment = summary.get("prompt_source_alignment") or {}
    actual_sources = {role: {source: alignment.get(role, {}).get("sources", {}).get(source, {}).get("delta")
                             for source in SOURCES} for role in ("p", "d")}
    expected_sources = {"external_kv_transfer": count * prefix if prefix else None,
                        "local_compute": count * 127, "local_cache_hit": 0}
    sources_match = prefix is not None and all(actual_sources[role] == expected_sources for role in ("p", "d"))
    states = {phase: l1_snapshot(optional_read(folder / (phase + "_status.json")))
              for phase in ("before", "after")}
    empty_values = [l1_empty(values) for values in states.values()]
    l1_state = "EMPTY_BOTH_SNAPSHOTS" if empty_values == [True, True] else (
        "NONEMPTY_SNAPSHOT" if False in empty_values else "UNKNOWN")
    protocol_formal = summary.get("requests") == count and summary.get("output_tokens_per_request") == output
    exclusions = []
    if not protocol_formal:
        exclusions.append("NONFORMAL_REQUEST_OR_OUTPUT_PROTOCOL")
    exclusions.extend("NATIVE_CHECK_FAILED:" + name for name, passed in checks.items() if not passed)
    if not sources_match:
        exclusions.append("PROMPT_SOURCE_EXPECTATION_MISMATCH")
    valid = protocol_formal and all(checks.values()) and sources_match
    status = STORE_STATUS if arm == "store" else (
        "COUNTS_AND_SOURCES_MATCH_REQUIRE_L2_AND_EXIT_REVIEW" if valid else "REQUEST_OR_SOURCE_GATE_FAILED")
    samples = {name: detailed_samples(native, name, count) for name in ("ttft", "tpot", "e2el")}
    row = {"result_directory": folder.relative_to(root).as_posix(), "dataset_id": summary.get("dataset_id"),
           "prefix_tokens": prefix, "input_tokens_per_request": input_tokens, "arm": arm,
           "concurrency": summary.get("concurrency"), "round": summary.get("round"),
           "native_completed": native.get("completed"), "native_failed": native.get("failed"),
           "native_errors": native.get("errors"), "observe_checks": summary.get("checks"),
           "native_total_input_tokens": native.get("total_input_tokens"),
           "native_total_output_tokens": native.get("total_output_tokens"),
           "native_output_lens": native.get("output_lens"), "native_input_lens": native.get("input_lens"),
           "expected_requests": count, "expected_output_tokens_per_request": output,
           "requested_requests": summary.get("requests"),
           "requested_output_tokens_per_request": summary.get("output_tokens_per_request"),
           "protocol_formal_round": protocol_formal, "aggregation_exclusion_reasons": exclusions,
           "native_checks": checks, "expected_source_deltas_per_role": expected_sources,
           "actual_source_deltas": actual_sources, "raw_source_alignment": alignment,
           "source_gate_passed": sources_match, "request_and_source_gate_passed": valid,
           "observe_accepted_for_comparison": summary.get("accepted_for_comparison"),
           "observe_acceptance_is_not_overall_acceptance": True, "metrics": measured_metrics(native, summary),
           "l1_state": l1_state, "l1_snapshots": states, "l2_read_evidence": counter_evidence(summary),
           "post_benchmark_drain": summary.get("post_benchmark_drain"),
           "performance_status": status,
           "runtime_exit_gate": "FAILED_NATIVE_SIGABRT_SERVICE_SIGSEGV_PD_SIGABRT" if arm == "store" else "NOT_REVIEWED_BY_THIS_SCRIPT",
           "native_detail_sample_provenance": {name: value[1] for name, value in samples.items()}}
    return row, samples


def aggregate(rows, details, expected_rounds):
    formal = [row for row in rows if row["protocol_formal_round"]]
    matched = [row for row in rows if row["request_and_source_gate_passed"]]
    duration = sum(row["metrics"]["duration"] for row in matched)
    output = sum(row["native_total_output_tokens"] for row in matched)
    rates = [row["native_total_output_tokens"] / row["metrics"]["duration"] for row in matched]
    group = {key: rows[0][key] for key in ("dataset_id", "prefix_tokens", "arm", "concurrency")}
    round_ids = [row["round"] for row in formal]
    group.update({"recorded_rounds": len(formal), "recorded_round_ids": round_ids,
                  "all_recorded_rounds": len(rows), "all_recorded_round_ids": [row["round"] for row in rows],
                  "nonformal_record_count": len(rows) - len(formal),
                  "expected_rounds": expected_rounds,
                  "round_coverage_complete": len(round_ids) == expected_rounds and set(round_ids) == set(range(1, expected_rounds + 1)),
                  "request_and_source_matched_rounds": len(matched),
                  "excluded_from_matched_aggregate": [row["result_directory"] for row in rows if row not in matched],
                  "total_output_tokens_matched": output, "total_duration_seconds_matched": duration,
                  "weighted_output_token_s": output / duration if duration else None,
                  "throughput_aggregation": "sum_actual_output_tokens / sum_native_duration_seconds; matched request/source rounds only",
                  "per_round_output_token_s": rates,
                  "round_throughput_min": min(rates) if rates else None,
                  "round_throughput_max": max(rates) if rates else None,
                  "round_throughput_population_std": statistics.pstdev(rates) if rates else None,
                  "all_round_l1_empty": all(row["l1_state"] == "EMPTY_BOTH_SNAPSHOTS" for row in rows),
                  "all_round_actual_read_bytes_observed": all(row["l2_read_evidence"]["read_bytes_state"] == "OBSERVED_POSITIVE_COUNTER" for row in rows),
                  "performance_status": STORE_STATUS if group["arm"] == "store" else "REQUIRES_PER_ROUND_L2_AND_EXIT_REVIEW",
                  "percentiles": {}, "rounds": rows})
    for name in ("ttft", "tpot", "e2el"):
        per_round = [row["metrics"].get("p95_" + name + "_ms") for row in matched]
        recorded_p95 = [value for value in per_round if finite(value)]
        arrays = [details[row["result_directory"]][name][0] for row in matched]
        can_pool = bool(arrays) and all(values is not None for values in arrays)
        pooled = [value * 1000 for values in arrays for value in values] if can_pool else []
        group["percentiles"][name] = {"pooled_p95_ms": percentile(pooled, .95) if pooled else None,
            "pooled_samples": len(pooled), "method": "linear_percentile_of_complete_per_request_seconds_arrays" if pooled else "per_round_p95_range_only",
            "per_round_p95_ms": per_round, "round_p95_min_ms": min(recorded_p95) if recorded_p95 else None,
            "round_p95_max_ms": max(recorded_p95) if recorded_p95 else None,
            "p95_is_never_averaged": True}
    return group


def number(value):
    return f"{value:.3f}" if finite(value) else "unknown"


def write_outputs(prefix, report):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["dataset_id", "prefix_tokens", "arm", "concurrency", "round", "performance_status",
              "protocol_formal_round", "aggregation_exclusion_reasons",
              "request_and_source_gate_passed", "native_completed", "native_failed", "native_total_input_tokens",
              "native_total_output_tokens", "duration_seconds", "output_token_s", "p95_ttft_ms", "p95_tpot_ms",
              *[role + "_" + source for role in ("p", "d") for source in SOURCES],
              "l1_state", "actual_read_byte_counters", "successful_chunk_counters", "result_directory"]
    with prefix.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in report["groups"]:
            for row in group["rounds"]:
                values = {key: row.get(key) for key in fields}
                values.update({"duration_seconds": row["metrics"].get("duration"),
                    "output_token_s": row["metrics"].get("output_throughput"),
                    "p95_ttft_ms": row["metrics"].get("p95_ttft_ms"), "p95_tpot_ms": row["metrics"].get("p95_tpot_ms")})
                for role in ("p", "d"):
                    values.update({role + "_" + source: row["actual_source_deltas"][role][source] for source in SOURCES})
                for key in ("actual_read_byte_counters", "successful_chunk_counters"):
                    values[key] = json.dumps(row["l2_read_evidence"][key], ensure_ascii=False)
                writer.writerow(values)
    lines = ["# 兼容 L2 归档汇总", "", "Store 性能统一标记 `" + STORE_STATUS + "`：本轮分别实测 native 关闭后 SIGABRT、service 退出 SIGSEGV、P/D 退出 SIGABRT；这些来自不同进程，退出 gate 尚未通过。",
             "现有 observe 的 accepted 不代表整体验收通过。Store 使用 relay 1.2 / 4:8 配置；原服务代码未改。",
             "吞吐只对请求与来源校验通过的轮次使用总输出 / 总原生计时。各轮失败和证据仍完整保留。实际读字节仅取归档计数，缺失保持未知。", "",
             "| dataset / prefix | arm | C | 正式记录/匹配/要求轮次 | 非正式记录 | 输出 token/s | 轮次吞吐范围 | TTFT p95 ms | TPOT p95 ms |", "|---|---|---:|---:|---:|---:|---|---|---|"]
    for group in report["groups"]:
        def p95(name):
            metric = group["percentiles"][name]
            if metric["pooled_p95_ms"] is not None:
                return number(metric["pooled_p95_ms"]) + " pooled(n=" + str(metric["pooled_samples"]) + ")"
            return number(metric["round_p95_min_ms"]) + "–" + number(metric["round_p95_max_ms"]) + " round range"
        lines.append(f"| {group['dataset_id']} / {group['prefix_tokens']} | {group['arm']} | {group['concurrency']} | {group['recorded_rounds']}/{group['request_and_source_matched_rounds']}/{group['expected_rounds']} | {group['nonformal_record_count']} | {number(group['weighted_output_token_s'])} | {number(group['round_throughput_min'])}–{number(group['round_throughput_max'])} | {p95('ttft')} | {p95('tpot')} |")
    lines += ["", "正式轮次按配置的请求数和每请求输出数协议认定；quick 等非正式记录不计入正式覆盖，仍完整保留逐轮记录与聚合排除原因。",
              "每组 JSON 保留逐轮 native 指标、成功失败、P/D 三来源、L1 前后状态、成功 chunks 和实际 L2 bytes 计数及来源。CSV 一行一轮。",
              "p95 只在该组所有匹配轮次均有完整可靠请求明细时池化；否则报告各轮 p95 范围，不平均 p95。", "",
              "解析错误：" + str(len(report["parse_errors"])) + "；详见 JSON。"]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Archive results root; scanned read-only")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Local JSON/CSV/Markdown output prefix")
    parser.add_argument("--expected-rounds", type=int, default=3)
    parser.add_argument("--expected-requests", type=int, default=32)
    parser.add_argument("--expected-output-tokens", type=int, default=32)
    args = parser.parse_args()
    if min(args.expected_rounds, args.expected_requests, args.expected_output_tokens) < 1:
        parser.error("expected rounds, requests and output tokens must be positive")
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("root must be an existing archive directory")
    grouped, details, errors = {}, {}, []
    for path in sorted(root.rglob("c*_r*_timed/summary.json")):
        try:
            row, samples = load_round(path, root, args)
            key = (str(row["dataset_id"]), row["prefix_tokens"], str(row["arm"]), row["concurrency"])
            grouped.setdefault(key, []).append(row)
            details[row["result_directory"]] = samples
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"summary_path": str(path), "error": str(exc)})
    report = {"archive_root": str(root), "scope": "OFFLINE_ARCHIVED_RESULTS_ONLY",
              "expected_requests_per_round": args.expected_requests,
              "expected_output_tokens_per_request": args.expected_output_tokens,
              "expected_rounds_per_condition": args.expected_rounds,
              "current_store_exit_gate": {"status": STORE_STATUS,
                  "reason": "separate observed processes: native close SIGABRT; service exit SIGSEGV; P/D exit SIGABRT",
                  "observed_exit_signals_by_process": {"native_close_probe": "SIGABRT", "service": "SIGSEGV", "prefiller": "SIGABRT", "decoder": "SIGABRT"},
                  "context": "real 18/36 MiB content and owner placement passed; 32 real KV objects owner-placed; L1 empty",
                  "configuration": "relay LocalCommRes 1.2 / 4:8", "service_code_modified": False,
                  "source": "explicit current experiment instruction; not inferred by this script"},
              "groups": [aggregate(rows, details, args.expected_rounds) for rows in grouped.values()],
              "parse_errors": errors}
    write_outputs(args.output_prefix, report)
    print(json.dumps({"groups": len(report["groups"]), "rounds": sum(len(rows) for rows in grouped.values()),
                      "parse_errors": len(errors), "outputs": [str(args.output_prefix.with_suffix(suffix)) for suffix in (".json", ".csv", ".md")]}, indent=2))
    if errors or not grouped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
