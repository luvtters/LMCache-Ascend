#!/usr/bin/env python3
"""Prepare four exact-length synthetic prefixes; plan by default, no inference."""
import argparse
import copy
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("MF_EXPERIMENT_ROOT", "/path/to/private/memfabric-worker-direct"))


def render_variants(tokenizer, request, expected_input):
    users = [index for index, item in enumerate(request["messages"]) if item.get("role") == "user"]
    if not users:
        raise ValueError("Expected the original user message")
    user_index = users[-1]
    original = request["messages"][user_index]["content"]
    if not isinstance(original, str) or "Record 000" not in original:
        raise ValueError("Expected original Record 000 marker")
    variants = []
    for record in ("000", "100", "200", "300"):
        variant = copy.deepcopy(request)
        variant["messages"][user_index]["content"] = original.replace("Record 000", "Record " + record, 1)
        prompt = tokenizer.apply_chat_template(variant["messages"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) != expected_input:
            raise ValueError(f"Record {record} has {len(tokens)} input tokens, expected {expected_input}")
        variant["max_tokens"] = 128
        variant.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        variants.append(dict(record=record, prompt=prompt, request=variant,
                             input_tokens=len(tokens), first_chunk_tokens=list(tokens[:128]),
                             with_special_tokens=len(tokenizer.encode(prompt, add_special_tokens=True))))
    if len({tuple(item["first_chunk_tokens"]) for item in variants}) != 4:
        raise ValueError("The four prefixes must already differ in the first complete 128-token chunk")
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, default=ROOT / "gate_request.json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-tokens", type=int, default=1373)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=ROOT / "bench1373_multi4")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.requests < 4 or args.requests % 4:
        parser.error("requests must be a positive multiple of four")
    data = args.output_root / "data"
    expected_external = (args.input_tokens - 1) // 128 * 128
    required_slots = 4 * (expected_external // 128)
    print(json.dumps(dict(mode="RUN" if args.run else "PLAN_ONLY", unique_prefixes=4,
                          input_tokens=args.input_tokens, output_tokens=128, requests=args.requests,
                          record_markers=["000", "100", "200", "300"], data_directory=str(data),
                          expected_cacheable_tokens=expected_external, required_slots=required_slots,
                          inference=False, downloads=False), indent=2))
    if not args.run:
        return 0
    if data.exists():
        raise FileExistsError("Refuse to overwrite the common dataset")
    if not Path(args.tokenizer).is_dir():
        raise ValueError("Tokenizer must be an existing local directory")
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    if not any(item.get("role") == "system" and "ORCHID" in item.get("content", "")
               for item in request["messages"]):
        raise ValueError("Expected original ORCHID system instruction")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    variants = render_variants(tokenizer, request, args.input_tokens)
    manifest = dict(unique_prefixes=4, input_tokens=args.input_tokens, output_tokens=128,
                    requests=args.requests, order="four unique prompts repeated cyclically",
                    first_complete_chunk_is_distinct=True, chunk_size=128,
                    expected_cacheable_tokens=expected_external, objects_per_prefix=expected_external // 128,
                    required_slots=required_slots, slot_size_mb=32, required_slot_bytes=required_slots * (32 << 20),
                    recommended_pool_gb=2, add_generation_prompt=True, enable_thinking=False,
                    material_type="synthetic_basic_four_prefix_case_not_customer_GLM_original",
                    tokenizer=args.tokenizer, request_json=str(args.request_json),
                    runtime_input_tokens_verified=False,
                    variants=[{key: value for key, value in item.items() if key not in ("prompt", "request")}
                              for item in variants])
    data.mkdir(parents=True)
    rows = [json.dumps(dict(prompt=variants[index % 4]["prompt"], output_tokens=128), ensure_ascii=False)
            for index in range(args.requests)]
    (data / "custom.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "requests.json").write_text(json.dumps([item["request"] for item in variants], indent=2,
                                                  ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
