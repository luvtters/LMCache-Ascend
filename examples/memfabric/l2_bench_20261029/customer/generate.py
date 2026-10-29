#!/usr/bin/env python3
"""Generate the four small customer-cropped raw-completion cases on CPU."""
import argparse
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys


CASES = [("in4096_p50_out128", 4096, 2048, 128),
         ("in16384_p50_out128", 16384, 8192, 128),
         ("in16384_p90_out128", 16384, 14720, 128),
         ("in16384_p90_out1000", 16384, 14720, 1000)]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator-file", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--case", choices=["all", *[row[0] for row in CASES]], default="all")
    a = p.parse_args()
    if a.requests < 2 or a.requests % 2:
        p.error("requests must be an even integer >= 2")
    model, output = a.tokenizer.resolve(), a.output.resolve()
    if output.exists() or output == model or model in output.parents:
        p.error("output must be a new directory outside the model")
    spec = importlib.util.spec_from_file_location("customer_base_generator", a.generator_file)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    tokenizer, config = base.load_tokenizer(model)
    selected = [row for row in CASES if a.case in ("all", row[0])]
    built, manifests, first_chunks = {}, [], set()
    cumulative_chunks = 0
    for case, inputs, prefix_tokens, outputs in selected:
        specs = [("seed", 2, 1), ("timed", a.requests, outputs)]
        markers = {(phase, i): f"{chr(65 + (i if phase == 'seed' else i + 2))} {case} {phase} request {i:04d}.\n"
                   for phase, count, _ in specs for i in range(count)}
        prefixes = {group: base.shared_prefix(tokenizer, f"{'AB'[group]} customer case {case} corpus {group}. Read notes.\n",
                     [marker for (phase, i), marker in markers.items() if i % 2 == group], prefix_tokens)
                    for group in range(2)}
        sequences = {0: [], 1: []}
        for phase, count, out in specs:
            rows = []
            for i in range(count):
                text, ids = base.fit_exact(tokenizer, prefixes[i % 2] + markers[(phase, i)], inputs, base.ENDING)
                if len(ids) != inputs or tokenizer.encode(text).ids != ids:
                    raise ValueError("Full raw prompt tokenization changed")
                sequences[i % 2].append(ids)
                rows.append({"prompt": text, "output_tokens": out,
                    "metadata": {"dataset_id": case + "_20261029", "phase": phase,
                        "request_index": i, "group": i % 2, "input_tokens": inputs,
                        "output_tokens": out, "shared_prefix_tokens": prefix_tokens}})
            built[(case, phase)] = rows
        for group, ids in sequences.items():
            if base.common_prefix(ids) != prefix_tokens:
                raise ValueError(f"{case} group {group}: shared prefix mismatch")
            if any(not prefix_tokens <= base.common_prefix([left, right]) < prefix_tokens + 128
                   for left, right in combinations(ids, 2)):
                raise ValueError("A pair shares an extra full tail chunk")
            if len({tuple(value) for value in ids}) != len(ids):
                raise ValueError("Duplicate full prompt across seed/timed")
            first = tuple(ids[0][:128])
            if first in first_chunks:
                raise ValueError("Cases/groups share a first full cache chunk")
            first_chunks.add(first)
        unique_chunks = 2 * (inputs // 128) + a.requests * ((inputs - prefix_tokens) // 128)
        cumulative_chunks += unique_chunks
        manifests.append({"case_id": case, "dataset_id": case + "_20261029", "input_tokens": inputs,
            "prefix_tokens": prefix_tokens, "prefix_fraction": prefix_tokens / inputs,
            "tail_tokens": inputs - prefix_tokens, "output_tokens": outputs, "requests": a.requests,
            "seed_requests": 2, "seed_output_tokens": 1, "prefix_groups": 2, "concurrency": 1,
            "warmup_requests": 0, "chunk_size": 128, "cpu_only": True,
            "model_directory": str(model), "generator_file": str(a.generator_file.resolve()),
            "tokenizer_class": config.get("tokenizer_class"),
            "encoding": "full raw completions re-encoded; no chat template; no wrapper special tokens",
            "strict_full_prompt_lengths": True, "pairwise_shared_prefix_has_no_extra_full_chunk": True,
            "unique_tail_across_seed_timed": True, "first_chunk_unique_across_cases_and_groups": True,
            "source_expectations": {
                "seed": {"p": {"external_kv_transfer": 0, "local_compute": 2 * inputs, "local_cache_hit": 0},
                         "d": {"external_kv_transfer": 2 * (inputs - 1), "local_compute": 2, "local_cache_hit": 0}},
                "timed": {"p": {"external_kv_transfer": a.requests * prefix_tokens,
                                  "local_compute": a.requests * (inputs - prefix_tokens), "local_cache_hit": 0},
                          "d": {"external_kv_transfer": a.requests * (inputs - 1),
                                  "local_compute": a.requests, "local_cache_hit": 0}}},
            "source_fence": "Qwen full-attention: P reuses shared prefix; P stores whole aligned prompt; D copies N tokens but accounts N-1 external plus 1 recomputed token",
            "unique_chunks_budget_no_eviction": unique_chunks,
            "cumulative_unique_chunks_budget_no_eviction": cumulative_chunks,
            "cumulative_payload_bytes_at_18_MiB": cumulative_chunks * 18 * 1024**2,
            "cumulative_slot_bytes_at_32_MiB": cumulative_chunks * 32 * 1024**2,
            "budget_is_design_not_measured_usage": True, "runtime_status": "NOT_VERIFIED"})
    forbidden = [name for name in ("torch", "torch_npu", "vllm", "lmcache", "memfabric_hybrid") if name in sys.modules]
    if forbidden:
        raise RuntimeError("CPU boundary violated: " + repr(forbidden))
    output.mkdir(parents=True)
    for manifest in manifests:
        folder = output / manifest["case_id"]
        folder.mkdir()
        for phase in ("seed", "timed"):
            (folder / (phase + ".jsonl")).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in built[(manifest["case_id"], phase)]), encoding="utf-8")
        manifest["forbidden_modules_imported"] = forbidden
        dump(folder / "manifest.json", manifest)
    root = {"cases": manifests, "total_unique_chunks_budget": cumulative_chunks,
            "total_slot_bytes_at_32_MiB": cumulative_chunks * 32 * 1024**2,
            "fits_64_GiB_slot_pool": cumulative_chunks * 32 * 1024**2 <= 64 * 1024**3,
            "fresh_backend_required_before_all_four_cases": True}
    dump(output / "manifest.json", root)
    print(json.dumps(root, indent=2))


if __name__ == "__main__":
    main()
