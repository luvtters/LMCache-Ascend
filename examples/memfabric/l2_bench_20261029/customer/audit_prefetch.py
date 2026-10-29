#!/usr/bin/env python3
"""Audit customer timed prefetch logs offline; return 1 for mixed or missing data.

Example:
  python audit_prefetch.py --archive-root ARCHIVE --tag customer02_strict \
    --store-run customer_store02 --memfabric-run customer_mf02 --output audit.json

Only standard-library code runs. This checks prefetch log sources and timings,
not device completion, pure copy bandwidth, output correctness, or clean exit.
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics


EXPECTED_REQUESTS = 8
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\]")
PREFETCH = re.compile(
    r"Prefetch request completed \(L1\+L2\): (\d+)/(\d+) retained keys "
    r"\((\d+) L1, (\d+) L2\) in ([\d.]+) ms "
    r"\(external_request_id=([^,]+), prefetch_request_id=(\d+)\)"
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_log(path, archive_root):
    """Keep the timestamp even when the prefetch message format is unexpected."""
    records = []
    source = path.relative_to(archive_root).as_posix()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = ANSI.sub("", raw)
        if "Prefetch request completed" not in text:
            continue
        timestamp = TIMESTAMP.search(text)
        if not timestamp:
            raise ValueError(f"{source}:{number}: prefetch timestamp missing")
        moment = dt.datetime.strptime(timestamp[1], "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=dt.timezone.utc
        )
        row = {
            "source": source,
            "line": number,
            "timestamp_utc": moment.isoformat(),
            "unix_time": moment.timestamp(),
            "log_text": text,
        }
        message = PREFETCH.search(text)
        if message:
            row.update(
                retained_keys=int(message[1]),
                requested_keys=int(message[2]),
                l1_keys=int(message[3]),
                l2_keys=int(message[4]),
                elapsed_ms=float(message[5]),
                external_request_id=message[6],
                prefetch_request_id=int(message[7]),
            )
        else:
            row["parse_error"] = "UNRECOGNIZED_PREFETCH_MESSAGE"
        records.append(row)
    return records


def percentile95(values):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def audit_segment(records, expected_keys, full_prompt_keys):
    failures = []
    if len(records) != EXPECTED_REQUESTS:
        failures.append(f"PREFETCH_COUNT_EXPECTED_8_OBSERVED_{len(records)}")
    invalid = [
        row for row in records
        if row["l1_keys"] != 0
        or row["l2_keys"] != expected_keys
        or row["retained_keys"] != expected_keys
        or row["requested_keys"] != full_prompt_keys
    ]
    if invalid:
        failures.append("PREFETCH_NOT_EXACT_ZERO_L1_EXPECTED_L2")
    values = [row["elapsed_ms"] for row in records]
    return {
        "status": "FAIL" if failures else "PASS",
        "failure_reasons": failures,
        "expected_requests": EXPECTED_REQUESTS,
        "expected_retained_and_l2_keys_per_request": expected_keys,
        "prefetch_count": len(records),
        "exact_zero_l1_expected_l2_count": len(records) - len(invalid),
        "l1_positive_record_count": sum(row["l1_keys"] > 0 for row in records),
        "matched_l1_chunks": sum(row["l1_keys"] for row in records),
        "matched_l2_chunks": sum(row["l2_keys"] for row in records),
        "mean_ms": statistics.mean(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "p95_ms_linear": percentile95(values),
        "records": records,
    }


def audit_arm(folder, records, archive_root):
    report = {"result_directory": folder.relative_to(archive_root).as_posix()}
    try:
        manifest = read_json(folder / "manifest.json")
        start = read_json(folder / "before_http.json")["unix_time"]
        end = read_json(folder / "after_http.json")["unix_time"]
        fields = ("input_tokens", "prefix_tokens", "chunk_size", "requests")
        if any(type(manifest.get(key)) is not int for key in fields):
            raise ValueError("manifest requires integer input/prefix/chunk_size/requests")
        total, prefix, chunk, requests = (manifest[key] for key in fields)
        if not (0 < prefix < total and chunk > 0 and total % chunk == prefix % chunk == 0):
            raise ValueError("requires distinct aligned P-prefix and full-prompt chunk counts")
        if requests != EXPECTED_REQUESTS:
            raise ValueError("manifest requests must equal 8")
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError("invalid before/after Unix window")
        p_keys, d_keys = prefix // chunk, total // chunk
        selected = [row for row in records if start <= row["unix_time"] <= end]
        parsed = [row for row in selected if "parse_error" not in row]
        unknown = [row for row in selected
                   if "parse_error" in row or row["retained_keys"] not in (p_keys, d_keys)]
        segments = {
            role: audit_segment([row for row in parsed if row["retained_keys"] == keys],
                                keys, d_keys)
            for role, keys in (("p", p_keys), ("d", d_keys))
        }
        failures = [f"{role.upper()}_SEGMENT_FAILED" for role, value in segments.items()
                    if value["status"] == "FAIL"]
        if unknown:
            failures.append("UNCLASSIFIED_PREFETCH_RECORDS_IN_TIMED_WINDOW")
        report.update(
            status="FAIL" if failures else "PASS",
            failure_reasons=failures,
            manifest={key: manifest.get(key) for key in
                      ("case_id", "input_tokens", "prefix_tokens", "output_tokens",
                       "chunk_size", "requests")},
            window={"start_unix_time": start, "end_unix_time": end,
                    "start_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
                    "end_utc": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat()},
            prefetch_count=len(selected),
            segments=segments,
            unclassified_records=unknown,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(status="FAIL", failure_reasons=[f"MISSING_OR_INVALID_EVIDENCE: {error}"])
    return report


def audit(archive_root, tag, runs):
    roots = {arm: archive_root / "customer_results" / f"{tag}_{arm}" for arm in runs}
    cases = sorted({path.name for root in roots.values() if root.is_dir()
                    for path in root.iterdir() if path.is_dir()})
    logs, log_errors = {}, {}
    for arm, run in runs.items():
        try:
            logs[arm] = parse_log(archive_root / "runs" / run / "service.log", archive_root)
        except (OSError, ValueError) as error:
            logs[arm] = []
            log_errors[arm] = str(error)
    results = []
    for case in cases:
        arms = {arm: audit_arm(root / case / "timed", logs[arm], archive_root)
                for arm, root in roots.items()}
        failures = [f"{arm.upper()}_FAILED" for arm, value in arms.items()
                    if value["status"] == "FAIL"]
        manifests = [value.get("manifest") for value in arms.values()]
        if all(manifests) and manifests[0] != manifests[1]:
            failures.append("ARM_MANIFESTS_DIFFER")
        results.append({"case_id": case, "status": "FAIL" if failures else "PASS",
                        "failure_reasons": failures, "arms": arms})
    passed = bool(results) and not log_errors and all(row["status"] == "PASS" for row in results)
    return {
        "status": "PASS" if passed else "FAIL",
        "scope": "OFFLINE_CUSTOMER_PREFETCH_L1_L2_AUDIT_ONLY",
        "archive_root": str(archive_root),
        "tag": tag,
        "runs": runs,
        "timezone": "UTC; before/after Unix bounds inclusive",
        "role_assignment": "manifest P-prefix chunks versus D full-prompt chunks; total retained keys",
        "p95_method": "linear interpolation at rank=(n-1)*0.95",
        "timing_boundary": "logged prefetch elapsed; not pure bandwidth or independent H2D completion proof",
        "expected_requests_per_role": EXPECTED_REQUESTS,
        "case_count": len(results),
        "log_errors": log_errors,
        "failure_reasons": [] if passed else (["NO_CUSTOMER_CASES"] if not results else ["CASE_OR_LOG_GATE_FAILED"]),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--store-run", required=True)
    parser.add_argument("--memfabric-run", required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()
    report = audit(args.archive_root.resolve(), args.tag,
                   {"store": args.store_run, "memfabric": args.memfabric_run})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "cases": {row["case_id"]: row["status"] for row in report["cases"]}}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
