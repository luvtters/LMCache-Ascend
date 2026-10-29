#!/usr/bin/env python3
"""CPU-only small prefix-pool data; reuse the inspected basic generator API."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator-file", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--prefix-tokens", type=int, default=1024)
    p.add_argument("--suffix-tokens", type=int, default=127)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", a.dataset_id):
        p.error("dataset-id must be 1-64 letters, digits, _ or -")
    if a.prefix_tokens < 128 or a.prefix_tokens % 128 or not 32 <= a.suffix_tokens < 128:
        p.error("prefix must align to 128 tokens; suffix must be 32..127 to avoid caching tail blocks")
    if a.requests < 4 or a.requests % 4 or a.rounds < 1 or any(c < 1 for c in a.concurrencies):
        p.error("requests must be a positive multiple of 4; rounds/concurrency must be positive")
    if a.output_tokens < 1:
        p.error("output-tokens must be positive")
    output, model = a.output.resolve(), a.tokenizer.resolve()
    if output.exists() or output == model or model in output.parents:
        p.error("output must be new and outside the model directory")
    spec = importlib.util.spec_from_file_location("basic_case_generator", a.generator_file)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    tokenizer, config = base.load_tokenizer(model)
    specs = [("seed", 4, 1)]
    for concurrency in a.concurrencies:
        for round_no in range(1, a.rounds + 1):
            stem = f"c{concurrency}_r{round_no}"
            specs.extend([(f"{stem}/warmup", 4, a.output_tokens),
                          (f"{stem}/custom", a.requests, a.output_tokens)])
    # The first differing character starts the tail. Each round has new tails.
    markers = {}
    for name, count, _ in specs:
        for i in range(count):
            group = i % 4
            markers[(name, i)] = f"{chr(65 + len(markers) % 26)} tail {name} request {i:04d}.\n"
    prefixes = {}
    for group in range(4):
        group_markers = [m for (name, i), m in markers.items() if i % 4 == group]
        header = f"{chr(65 + group)} independent corpus {a.dataset_id} group {group}. Read reference notes.\n"
        prefixes[group] = base.shared_prefix(tokenizer, header, group_markers, a.prefix_tokens)
    built, sequences = {}, {g: [] for g in range(4)}
    total = a.prefix_tokens + a.suffix_tokens
    for name, count, out_tokens in specs:
        rows = []
        for i in range(count):
            group = i % 4
            for ending in (base.ENDING, " " + base.ENDING, "\n" + base.ENDING):
                try:
                    text, ids = base.fit_exact(tokenizer, prefixes[group] + markers[(name, i)], total, ending)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError(f"{name}, request {i}, group {group}: cannot fit exact {total} tokens")
            sequences[group].append(ids)
            rows.append({"prompt": text, "output_tokens": out_tokens,
                         "metadata": {"dataset_id": a.dataset_id, "group": group,
                                      "phase": name, "request_index": i,
                                      "input_tokens": len(ids), "shared_prefix_tokens": a.prefix_tokens,
                                      "output_tokens": out_tokens, "runtime_l2_hit_tokens": None}})
        built[name] = rows
    for group, ids in sequences.items():
        if base.common_prefix(ids) != a.prefix_tokens:
            raise ValueError(f"group {group}: shared prefix is not exactly {a.prefix_tokens}")
        if len({tuple(s) for s in ids}) != len(ids):
            raise ValueError(f"group {group}: duplicated full inputs across phases")
    if len({tuple(ids[0][:128]) for ids in sequences.values()}) != 4:
        raise ValueError("Independent prefixes share a first full cache chunk")
    forbidden = [m for m in ("torch", "torch_npu", "vllm", "lmcache", "memfabric_hybrid") if m in sys.modules]
    if forbidden:
        raise RuntimeError(f"CPU-only boundary violated: {forbidden}")
    output.mkdir(parents=True)
    for name, rows in built.items():
        path = output / (name + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"dataset_id": a.dataset_id, "model_directory": str(model),
                "generator_file": str(a.generator_file.resolve()), "tokenizer_class": config.get("tokenizer_class"),
                "prefix_groups": 4, "prefix_tokens": a.prefix_tokens, "suffix_tokens": a.suffix_tokens,
                "input_tokens": total, "output_tokens": a.output_tokens, "requests_per_round": a.requests,
                "rounds": a.rounds, "concurrencies": a.concurrencies, "chunk_size": 128,
                "warmup_requests_per_round": 4, "seed_requests": 4,
                "full_prompt_unique_across_phases": True, "first_chunk_unique_groups": 4,
                "cached_full_chunks_per_prompt": a.prefix_tokens // 128,
                "encoding": "raw completions; full prompt re-encoded; no chat template",
                "cpu_only": True, "forbidden_modules_imported": forbidden,
                "runtime_l2_path_status": "NOT_VERIFIED"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
