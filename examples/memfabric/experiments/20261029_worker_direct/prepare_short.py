#!/usr/bin/env python3
"""Copy verified common inputs into distinct output-8 datasets; plan by default.

Pure standard library: no tokenizer/model imports, downloads, inference, SSH,
or service changes. Original datasets and their ordering remain unchanged.
"""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("MF_EXPERIMENT_ROOT", "/path/to/private/memfabric-worker-direct"))
CASES = (("bench1373_multi4", "short1373_multi4", 1373), ("bench3000", "short3000", 3000))


def prepare(root):
    prepared = []
    # Validate both inputs and both fresh destinations before writing either.
    for source_name, destination_name, expected_input in CASES:
        source = root / source_name / "data"
        destination = root / destination_name / "data"
        if destination.exists():
            raise FileExistsError("Refuse existing short dataset: " + str(destination))
        rows = [json.loads(line) for line in (source / "custom.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(rows) != 32 or any(not isinstance(row.get("prompt"), str) for row in rows):
            raise ValueError("Expected exactly 32 existing raw prompt rows: " + str(source))
        if any(row.get("output_tokens") != 128 for row in rows):
            raise ValueError("Expected the unchanged output-128 source group")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("input_tokens") != expected_input:
            raise ValueError("Source input manifest does not match its known case")
        short_rows = [dict(row, output_tokens=8) for row in rows]
        if [row["prompt"] for row in short_rows] != [row["prompt"] for row in rows]:
            raise AssertionError("Prompt content/order changed")
        short_manifest = dict(manifest, output_tokens=8, requests=32,
                              source_dataset=str(source / "custom.jsonl"),
                              source_manifest=str(source / "manifest.json"),
                              dataset=str(destination / "custom.jsonl"),
                              source_output_tokens=128, prompt_content_and_order_unchanged=True,
                              unique_prompts=len({row["prompt"] for row in rows}),
                              benchmark_variant="same_input_output_8",
                              separate_from_output_128_group=True,
                              results_must_not_be_mixed_with_output_128=True,
                              preparation="stdlib copy; no new tokenizer or runtime validation")
        if "output_tokens_per_request" in short_manifest:
            short_manifest["output_tokens_per_request"] = 8
        prepared.append((destination, short_rows, short_manifest))
    receipts = []
    for destination, rows, manifest in prepared:
        destination.mkdir(parents=True)
        with (destination / "custom.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        receipts.append(dict(dataset=str(destination / "custom.jsonl"), input_tokens=manifest["input_tokens"],
                             output_tokens=8, requests=32, unique_prompts=manifest["unique_prompts"],
                             prompt_content_and_order_unchanged=True))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = dict(mode="RUN" if args.run else "PLAN_ONLY", output_tokens=8, requests=32,
                model_or_device_imports=False,
                datasets=[dict(source=str(args.root / source / "data/custom.jsonl"),
                               destination=str(args.root / target / "data/custom.jsonl"), input_tokens=length)
                          for source, target, length in CASES])
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        print(json.dumps(dict(prepared=prepare(args.root)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
