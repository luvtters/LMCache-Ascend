# SPDX-License-Identifier: Apache-2.0
"""Build three small Qwen3 datasets using CPU tokenization only.

The shared-prefix definition and full-prompt validation follow the project's
GLM candidate materials. Text here is synthetic, not customer-original data.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import tokenizers
from tokenizers import Tokenizer


INPUT_TOKENS = 2560
OUTPUT_TOKENS = 128
CHUNK_SIZE = 128
CASES = {"qwen_2560_unique": 0,
         "qwen_2560_prefix50": 1280,
         "qwen_2560_prefix90": 2304}
NOTES = " The cache stores reusable key value pairs. Each request reads its own notes."
ENDING = "\nPlease respond with READY.\nAnswer:"


def common_prefix(sequences):
    first = sequences[0]
    length = min(map(len, sequences))
    for other in sequences[1:]:
        length = next((i for i in range(length) if first[i] != other[i]), length)
    return length


def load_tokenizer(model):
    config = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    if config.get("tokenizer_class") not in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
        raise ValueError("This small generator supports the inspected Qwen3/Qwen2 tokenizer only")
    if config.get("auto_map") or config.get("add_bos_token") or config.get("add_eos_token"):
        raise ValueError("Custom code or wrapper-added special tokens require a separate review")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Tokenizer must not silently truncate or pad inputs")
    for text in ("Hello", "\n cache", "Read these notes.\nAnswer:"):
        if tokenizer.encode(text).ids != tokenizer.encode(text, add_special_tokens=False).ids:
            raise ValueError("Unexpected special-token overhead")
    return tokenizer, config


def fit_exact(tokenizer, base, target, ending=""):
    """Always re-encode the complete final text, including the ending."""
    raw = tokenizer.encode(base + NOTES * (target // 10 + 16),
                           add_special_tokens=False).ids
    cutoff = target - len(tokenizer.encode(ending).ids)
    tried = set()
    for _ in range(12):
        if cutoff in tried or not 0 < cutoff <= len(raw):
            break
        tried.add(cutoff)
        text = tokenizer.decode(raw[:cutoff], skip_special_tokens=False) + ending
        ids = tokenizer.encode(text).ids
        if len(ids) == target and text.startswith(base):
            return text, ids
        cutoff += target - len(ids)
    for cut in range(max(1, cutoff - 12), min(len(raw), cutoff + 12) + 1):
        text = tokenizer.decode(raw[:cut], skip_special_tokens=False) + ending
        if text.startswith(base) and len(tokenizer.encode(text).ids) == target:
            return text, tokenizer.encode(text).ids
    raise ValueError(f"Cannot construct exact length {target}; no completed dataset written")


def shared_prefix(tokenizer, header, markers, target):
    # The newline/marker boundary also participates in tokenization.
    for separator in ("\n", "\n\n", " "):
        length = target
        seen = set()
        for _ in range(8):
            if length in seen:
                break
            seen.add(length)
            prefix, _ = fit_exact(tokenizer, header, length)
            prefix += separator
            actual = common_prefix([tokenizer.encode(prefix + m).ids for m in markers])
            if actual == target:
                return prefix
            length += target - actual
    raise ValueError(f"Cannot construct exact shared prefix {target}")


def build_case(tokenizer, name, target, dataset_id):
    markers = [f"{letter} request {i:02d}.\n" for i, letter in enumerate("ABCDEFGH")]
    header = f"Dataset {dataset_id}; case {name}. Read the reference notes.\n"
    prefix = shared_prefix(tokenizer, header, markers, target) if target else header
    rows, sequences = [], []
    for i, marker in enumerate(markers):
        text, ids = fit_exact(tokenizer, prefix + marker, INPUT_TOKENS, ENDING)
        sequences.append(ids)
        rows.append({"prompt": text, "output_tokens": OUTPUT_TOKENS,
                     "metadata": {"origin": "synthetic_basic_case",
                                  "dataset_id": dataset_id, "case": name,
                                  "request_index": i, "input_tokens": len(ids),
                                  "design_shared_prefix_tokens": target,
                                  "runtime_cache_hit_fraction": None}})
    pairwise = [common_prefix([left, right]) for left, right in combinations(sequences, 2)]
    if len({r["prompt"] for r in rows}) != 8 or len({tuple(s) for s in sequences}) != 8:
        raise ValueError(f"{name}: duplicate inputs")
    if target and (min(pairwise) != target or max(pairwise) >= target + CHUNK_SIZE):
        raise ValueError(f"{name}: unexpected shared-prefix boundaries {min(pairwise)}, {max(pairwise)}")
    if not target and len({tuple(s[:CHUNK_SIZE]) for s in sequences}) != 8:
        raise ValueError(f"{name}: unique requests share a complete first cache chunk")
    report = {"case": name, "prompt_count": 8,
              "input_tokens_min": min(map(len, sequences)),
              "input_tokens_max": max(map(len, sequences)),
              "output_tokens_requested": OUTPUT_TOKENS,
              "design_shared_prefix_tokens": target,
              "design_shared_prefix_fraction": target / INPUT_TOKENS if target else None,
              "actual_group_lcp_tokens": common_prefix(sequences),
              "pairwise_lcp_tokens_min": min(pairwise),
              "pairwise_lcp_tokens_max": max(pairwise),
              "unique_prompt_count": 8, "unique_token_sequence_count": 8,
              "first_chunk_unique_count": len({tuple(s[:CHUNK_SIZE]) for s in sequences}),
              "runtime_cache_hit_fraction": None, "runtime_status": "NOT_MEASURED"}
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New private directory; existing outputs are never overwritten")
    parser.add_argument("--dataset-id", required=True,
                        help="Use the same ID/data for A and B; a new ID for a new cold pair")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.dataset_id):
        parser.error("dataset-id must contain 1-64 letters, digits, underscores or hyphens")
    model, output = args.tokenizer.resolve(), args.output.resolve()
    if output == model or model in output.parents:
        parser.error("output must not be inside the model directory")
    if output.exists():
        parser.error("output already exists; select a new directory")
    tokenizer, config = load_tokenizer(model)
    built = [build_case(tokenizer, name, target, args.dataset_id) for name, target in CASES.items()]
    first_chunks = []
    for rows, _ in built:
        chunks = {tuple(tokenizer.encode(r["prompt"]).ids[:CHUNK_SIZE]) for r in rows}
        if any(chunks & previous for previous in first_chunks):
            raise ValueError("Different cases unexpectedly share a first cache chunk")
        first_chunks.append(chunks)
    forbidden = [name for name in ("torch", "torch_npu", "vllm", "lmcache", "memfabric_hybrid")
                 if name in sys.modules]
    if forbidden:
        raise RuntimeError(f"CPU-only boundary violated: {forbidden}")
    output.mkdir(parents=True, exist_ok=False)
    for rows, report in built:
        folder = output / "dataset" / report["case"]
        folder.mkdir(parents=True)
        with (folder / "custom.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"dataset_id": args.dataset_id, "origin": "synthetic_basic_case",
              "method_reference": "GLM candidate fivecases 2026-10-19; not customer originals",
              "seed": 0, "construction": "deterministic; no random sampling",
              "model_directory": str(model), "tokenizers_version": tokenizers.__version__,
              "tokenizer_class": config.get("tokenizer_class"),
              "encoding": "full raw prompt, add_special_tokens=True, no chat template",
              "chunk_size": CHUNK_SIZE, "cpu_only": True,
              "forbidden_modules_imported": forbidden,
              "cases": [report for _, report in built]}
    with (output / "report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "CPU_VALIDATED", "output": str(output),
                      "cases": report["cases"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
