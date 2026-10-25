# SPDX-License-Identifier: Apache-2.0
"""Summarize basic_bench results without importing device libraries.

Usage: python summarize_results.py RESULTS_ROOT --output summary.json
Pairing checks the recorded benchmark command and dataset ID, not service
launch settings beyond TP/mode/prefetch policy or independently verified cache contents.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re


PATH_OPTIONS = {"--base-url", "--dataset-path", "--result-dir",
                "--result-filename", "--label"}
COUNTERS = ("store_bytes", "load_bytes", "evicted_objects")
MAIN_METRICS = ("mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms",
                "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms",
                "mean_e2el_ms", "median_e2el_ms", "p50_e2el_ms",
                "request_throughput", "output_throughput")
SOURCE_METRIC = "vllm:prompt_tokens_by_source_total"
TOKEN_SOURCES = ("local_compute", "external_kv_transfer", "local_cache_hit")
SERVICE_MATCH_FIELDS = ("tp_size", "mode", "prefetch_policy")
SOURCE_LINE = re.compile(r"^" + re.escape(SOURCE_METRIC) + r"\{([^}]*)\}\s+(\S+)")
SOURCE_LABEL = re.compile(r'(?:^|,)\s*source="([^"]*)"(?:\s*,|\s*$)')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path, errors):
    if not path.exists():
        errors.append(f"missing: {path.name}")
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        return result
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return {}


def command_signature(command):
    """Ignore declared input/output locations, retain all benchmark behavior."""
    if not isinstance(command, list) or len(command) < 3 or not all(isinstance(x, str) for x in command):
        return None
    prefix, options = [], []
    index = 0
    while index < len(command) and not command[index].startswith("--"):
        prefix.append(command[index])
        index += 1
    while index < len(command):
        flag, _, inline = command[index].partition("=")
        values = [inline] if "=" in command[index] else []
        index += 1
        while index < len(command) and not command[index].startswith("--"):
            values.append(command[index])
            index += 1
        if flag not in PATH_OPTIONS:
            options.append((flag, tuple(values)))
    return (tuple(prefix), tuple(sorted(options))) if options else None


def service_parameters(config):
    if config is None:
        return None
    return {key: config.get(key) for key in SERVICE_MATCH_FIELDS}


def memfabric_delta(before, after):
    def adapters(status):
        values = status.get("storage_manager", {}).get("l2_adapters")
        if not isinstance(values, list):
            return None
        return {index: value for index, value in enumerate(values)
                if isinstance(value, dict) and value.get("protocol") == "SDMA"}

    left, right = adapters(before), adapters(after)
    result = {"present_before": bool(left) if left is not None else None,
              "present_after": bool(right) if right is not None else None,
              "adapters": [], **{key + "_delta": None for key in COUNTERS}}
    if left is None or right is None:
        return result
    for index in sorted(left.keys() | right.keys()):
        first, last = left.get(index), right.get(index)
        same = (first is not None and last is not None
                and first.get("storage_rank") == last.get("storage_rank"))
        record = {"index": index, "storage_rank": (last or first).get("storage_rank"),
                  "healthy_before": first.get("is_healthy") if first else None,
                  "healthy_after": last.get("is_healthy") if last else None,
                  "counter_decreased": False}
        for key in COUNTERS:
            start, end = first.get(key) if first else None, last.get(key) if last else None
            valid = same and number(start) and number(end) and end >= start
            record[key + "_before"] = start if number(start) else None
            record[key + "_after"] = end if number(end) else None
            record[key + "_delta"] = end - start if valid else None
            if same and number(start) and number(end) and end < start:
                record["counter_decreased"] = True
        if record["counter_decreased"]:
            for key in COUNTERS:
                record[key + "_delta"] = None
        result["adapters"].append(record)
    for key in COUNTERS:
        values = [adapter[key + "_delta"] for adapter in result["adapters"]]
        if values and all(number(value) for value in values):
            result[key + "_delta"] = sum(values)
    return result


def token_source_deltas(before_text, after_text):
    """Sum series within one endpoint only; never combine P and D."""
    def parse(text):
        values = {source: None for source in TOKEN_SOURCES}
        invalid = set()
        for line in (text or "").splitlines():
            match = SOURCE_LINE.match(line)
            label = SOURCE_LABEL.search(match.group(1)) if match else None
            if label is None or label.group(1) not in values:
                continue
            source = label.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                invalid.add(source)
                continue
            if not number(value) or value < 0:
                invalid.add(source)
            else:
                values[source] = (values[source] or 0) + value
        for source in invalid:
            values[source] = None
        return values

    before, after = parse(before_text), parse(after_text)
    rollback = any(number(before[s]) and number(after[s]) and after[s] < before[s]
                   for s in TOKEN_SOURCES)
    sources = {}
    for source in TOKEN_SOURCES:
        start, end = before[source], after[source]
        notes = []
        if not number(start):
            notes.append("before file/source missing or invalid")
        if not number(end):
            notes.append("after file/source missing or invalid")
        if rollback:
            notes.append("endpoint counter rollback; all source deltas are not comparable")
        sources[source] = {"before": start, "after": end,
                           "delta": end - start if not notes else None,
                           "note": "; ".join(notes) if notes else None}
    return {"metric": SOURCE_METRIC, "sources": sources, "counter_decreased": rollback}


def prompt_token_sources(folder):
    result = {}
    for role in ("p", "d"):
        snapshots, errors = [], []
        for phase in ("before", "after"):
            path = folder / f"{phase}_{role}_metrics.txt"
            try:
                snapshots.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError) as exc:
                snapshots.append(None)
                errors.append(f"{path.name}: {exc}")
        result[role] = token_source_deltas(*snapshots)
        result[role]["read_errors"] = errors
    return result


def load_run(path, root):
    errors = []
    manifest = read_json(path, errors)
    metrics = read_json(path.parent / "benchmark.json", errors)
    before = read_json(path.parent / "before_status.json", errors)
    after = read_json(path.parent / "after_status.json", errors)
    selected = {key: metrics.get(key) if number(metrics.get(key)) else None for key in MAIN_METRICS}
    for key, value in metrics.items():
        if any(part in key for part in ("ttft", "tpot", "e2el", "itl", "throughput", "goodput", "duration")) and number(value):
            selected[key] = value
    complete = (not errors and manifest.get("status") == "FINISHED"
                and manifest.get("benchmark_exit_code") == 0
                and manifest.get("after_snapshot_exit_code") == 0
                and manifest.get("request_and_token_counts_match") is True
                and metrics.get("completed") == 8 and metrics.get("failed", 0) == 0
                and metrics.get("total_input_tokens") == 20480
                and metrics.get("total_output_tokens") == 1024
                and metrics.get("output_lens") == [128] * 8)
    return {"run": path.parent.relative_to(root).as_posix() or ".",
            "arm": manifest.get("arm"), "case": manifest.get("case"),
            "dataset_id": manifest.get("dataset_id"), "status": manifest.get("status"),
            "complete": complete, "cache_state_before_case": manifest.get("cache_state_before_case"),
            "command": manifest.get("command"),
            "service_config": manifest.get("service_config"),
            "completed": metrics.get("completed"), "failed": metrics.get("failed"),
            "total_input_tokens": metrics.get("total_input_tokens"),
            "total_output_tokens": metrics.get("total_output_tokens"),
            "metrics": selected, "memfabric": memfabric_delta(before, after),
            "prompt_token_sources": prompt_token_sources(path.parent),
            "runtime_cache_hit_fraction": None, "read_errors": errors}


def summarize(root):
    runs = [load_run(path, root) for path in sorted(root.rglob("run.json"))]
    groups, pairs, unpaired = defaultdict(lambda: {"A": [], "B": []}), [], []
    for run in runs:
        signature = command_signature(run["command"])
        eligible = (run["complete"] and run["arm"] in ("A", "B")
                    and run["dataset_id"] and run["case"] and signature is not None
                    and (run["service_config"] is None or isinstance(run["service_config"], dict)))
        if not eligible:
            unpaired.append({"run": run["run"], "reason": "incomplete run or missing pairing identity/command"})
            continue
        config_signature = json.dumps(service_parameters(run["service_config"]), sort_keys=True)
        key = (run["dataset_id"], run["case"], run["cache_state_before_case"], signature, config_signature)
        groups[key][run["arm"]].append(run)
    for group in groups.values():
        count = min(len(group["A"]), len(group["B"]))
        for a, b in zip(group["A"][:count], group["B"][:count]):
            changes = {}
            for metric in sorted(a["metrics"].keys() | b["metrics"].keys()):
                av, bv = a["metrics"].get(metric), b["metrics"].get(metric)
                valid = number(av) and number(bv)
                changes[metric] = {"A": av, "B": bv,
                                   "B_minus_A": bv - av if valid else None,
                                   "change_percent": (bv / av - 1) * 100 if valid and av else None}
            pairs.append({"dataset_id": a["dataset_id"], "case": a["case"],
                          "cache_state_before_case": a["cache_state_before_case"],
                          "service_parameters": service_parameters(a["service_config"]),
                          "A_run": a["run"], "B_run": b["run"], "metrics": changes})
        for run in group["A"][count:] + group["B"][count:]:
            unpaired.append({"run": run["run"], "reason": "no remaining opposite-arm run with matching identity, cache-state label, command and service parameters"})
    return {"results_root": str(root), "runs": runs, "pairs": pairs, "unpaired_runs": unpaired,
            "notes": ["Missing values remain null; runtime cache-hit fractions are not inferred.",
                      "MemFabric counters come from SDMA adapter snapshots and may lag asynchronous work.",
                      "An external_kv_transfer metric alone does not identify an L2 read.",
                      "Prompt-source counter deltas remain separate for P and D; missing/reset counters are null, not inferred hit rates.",
                      "Pairs match dataset ID, case, cache-state label and recorded benchmark parameters; locations/URLs may differ.",
                      "Recorded service configurations additionally match tp_size, mode and prefetch_policy; runs with and without a config never mix.",
                      "Within matching groups, sorted run paths are paired one-to-one; this is not statistical significance testing.",
                      "Service launch equivalence and dataset contents must be checked separately; dataset ID is a declared identity."]}


def display(value):
    return f"{value:.3f}" if number(value) else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path, help="Default: RESULTS_ROOT/summary.json")
    args = parser.parse_args()
    root = args.results_root.resolve()
    if not root.is_dir():
        parser.error("results_root must be an existing directory")
    report = summarize(root)
    output = args.output or root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("| Arm | Case | Complete | Requests | Mean TTFT ms | Mean TPOT ms | Output tok/s | BM read bytes |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in report["runs"]:
        metrics = run["metrics"]
        print(f"| {run['arm']} | {run['case']} | {run['complete']} | {run['completed']} "
              f"| {display(metrics.get('mean_ttft_ms'))} | {display(metrics.get('mean_tpot_ms'))} "
              f"| {display(metrics.get('output_throughput'))} | {display(run['memfabric']['load_bytes_delta'])} |")
    print(f"\nRuns: {len(report['runs'])}; matched A/B pairs: {len(report['pairs'])}; JSON: {output}")


if __name__ == "__main__":
    main()
