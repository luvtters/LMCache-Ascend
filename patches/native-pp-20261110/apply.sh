#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repos=(/vllm-workspace/vllm /vllm-workspace/vllm-ascend)
patches=(0001-headless-executor.patch 0002-eplb-receive-buffers.patch)
states=()

# Check both target files before modifying either repository.
for i in 0 1; do
    if git -C "${repos[$i]}" apply --reverse --check "$patch_dir/${patches[$i]}" 2>/dev/null; then
        states[$i]=ALREADY_APPLIED
    else
        git -C "${repos[$i]}" apply --check "$patch_dir/${patches[$i]}"
        states[$i]=APPLIED
    fi
done

for i in 0 1; do
    if [[ "${states[$i]}" == APPLIED ]]; then
        git -C "${repos[$i]}" apply "$patch_dir/${patches[$i]}"
    fi
    printf '%s: %s\n' "${states[$i]}" "${patches[$i]}"
done

echo FRAMEWORK_PATCH_OK
