#!/usr/bin/env python3
"""Prepare one exact-length synthetic chat-prefix dataset; plan by default.

Only --run loads a local tokenizer. Appends repeated ' context' to the final
user message and renders the complete chat, preserving generation markers.
No model file is changed and no network download or inference is performed.
"""

import argparse
import copy
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("MF_EXPERIMENT_ROOT", "/path/to/private/memfabric-worker-direct"))
PAD = " context"


def find_exact_prompt(tokenizer, original_messages, target):
    messages = copy.deepcopy(original_messages)
    users = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if not users or not isinstance(messages[users[-1]].get("content"), str):
        raise ValueError("Expected a final user message with text content")
    index, user = users[-1], messages[users[-1]]["content"]
    calls, seen = 0, {}

    def render(count):
        nonlocal calls
        if count not in seen:
            messages[index]["content"] = user + PAD * count
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            seen[count] = (prompt, length)
            calls += 1
        return seen[count]

    _, baseline = render(0)
    if baseline > target:
        raise ValueError(f"Existing chat has {baseline} tokens, above target {target}; no truncation allowed")
    low, high = 0, max(1, target - baseline)
    while render(high)[1] < target:
        high *= 2
        if high > target * 8:
            raise ValueError("Padding failed to reach target within the bounded search")
    selected = None
    while low <= high:
        middle = (low + high) // 2
        prompt, length = render(middle)
        if length == target:
            selected = middle
            break
        if length < target:
            low = middle + 1
        else:
            high = middle - 1
    if selected is None:
        # A short bounded check handles tokenizer merge effects near the edge;
        # an unreachable exact length is a failure, never a truncated prompt.
        for count in range(max(0, low - 32), low + 33):
            if render(count)[1] == target:
                selected = count
                break
    if selected is None:
        raise ValueError("Repeated context padding cannot produce the exact requested token length")
    messages[index]["content"] = user + PAD * selected
    prompt, length = render(selected)
    if messages[index]["content"][:len(user)] != user or length != target:
        raise AssertionError("Original user prefix or exact token count was not preserved")
    return dict(prompt=prompt, messages=messages, padding_count=selected,
                input_tokens=length, original_input_tokens=baseline, tokenizer_calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, default=ROOT / "gate_request.json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-tokens", type=int, default=3000)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=ROOT / "bench3000")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if min(args.input_tokens, args.requests, args.output_tokens) <= 0:
        parser.error("Token lengths and request count must be positive")
    data = args.output_root / "data"
    print(json.dumps(dict(mode="RUN" if args.run else "PLAN_ONLY", request_json=str(args.request_json),
                          tokenizer=args.tokenizer, target_input_tokens=args.input_tokens,
                          requests=args.requests, output_tokens=args.output_tokens,
                          data_directory=str(data), padding=PAD,
                          expected_cacheable_tokens=(args.input_tokens - 1) // 128 * 128,
                          downloads=False, inference=False), indent=2))
    if not args.run:
        return 0
    if data.exists():
        raise FileExistsError("Refuse to overwrite an existing dataset directory")
    if not Path(args.tokenizer).is_dir():
        raise ValueError("Tokenizer must be an existing local directory")
    request = json.loads(args.request_json.read_text(encoding="utf-8"))
    messages = request["messages"]
    if not any(message.get("role") == "system" and "ORCHID" in message.get("content", "")
               for message in messages):
        raise ValueError("Expected the original ORCHID system instruction")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    result = find_exact_prompt(tokenizer, messages, args.input_tokens)
    rendered_request = copy.deepcopy(request)
    rendered_request["messages"] = result["messages"]
    rendered_request["max_tokens"] = args.output_tokens
    rendered_request.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    manifest = dict(input_tokens=result["input_tokens"], output_tokens=args.output_tokens,
                    requests=args.requests, identical_prompt_rows=True,
                    original_input_tokens=result["original_input_tokens"], padding_word=PAD,
                    padding_repetitions=result["padding_count"], tokenizer_calls=result["tokenizer_calls"],
                    input_tokens_with_special_tokens=len(tokenizer.encode(result["prompt"], add_special_tokens=True)),
                    expected_cacheable_tokens=(args.input_tokens - 1) // 128 * 128,
                    chunk_size=128, add_generation_prompt=True, enable_thinking=False,
                    original_system_instruction_preserved=True, original_user_prefix_preserved=True,
                    prompt_truncated=False, request_json=str(args.request_json), tokenizer=args.tokenizer,
                    dataset=str(data / "custom.jsonl"),
                    material_type="synthetic_basic_functional_case_not_customer_GLM_original",
                    runtime_input_tokens_verified=False)
    data.mkdir(parents=True)
    row = json.dumps(dict(prompt=result["prompt"], output_tokens=args.output_tokens), ensure_ascii=False)
    (data / "custom.jsonl").write_text((row + "\n") * args.requests, encoding="utf-8")
    (data / "prompt.txt").write_text(result["prompt"], encoding="utf-8")
    for name, value in (("manifest.json", manifest), ("request.json", rendered_request)):
        (data / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
