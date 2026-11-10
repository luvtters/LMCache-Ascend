#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
roots="$(python3 - <<'PY'
import importlib.util
from pathlib import Path
for name in ('vllm', 'vllm_ascend'):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f'PACKAGE_NOT_FOUND: {name}'
    print(Path(spec.origin).resolve().parent.parent)
PY
)"
mapfile -t package_roots <<< "$roots"
[[ ${#package_roots[@]} == 2 ]]
repos=("${package_roots[0]}" "${package_roots[1]}" "${package_roots[1]}")
patches=(0001-headless-executor.patch 0002-eplb-receive-buffers.patch 0003-ascend-official-headless-executor.patch)
directions=(reverse reverse forward)
states=()

# Check all three independent target files before changing any source.
for i in 0 1 2; do
    flags=()
    opposite=()
    if [[ "${directions[$i]}" == reverse ]]; then
        flags=(--reverse)
    else
        opposite=(--reverse)
    fi
    if git -C "${repos[$i]}" apply "${flags[@]}" --check "$patch_dir/${patches[$i]}" 2>/dev/null; then
        states[$i]=CHANGE
    elif git -C "${repos[$i]}" apply "${opposite[@]}" --check "$patch_dir/${patches[$i]}" 2>/dev/null; then
        states[$i]=ALREADY_DONE
    else
        printf 'SOURCE_MISMATCH: %s in %s\n' "${patches[$i]}" "${repos[$i]}" >&2
        exit 1
    fi
done

for i in 0 1 2; do
    if [[ "${states[$i]}" == CHANGE ]]; then
        flags=()
        if [[ "${directions[$i]}" == reverse ]]; then
            flags=(--reverse)
        fi
        git -C "${repos[$i]}" apply "${flags[@]}" "$patch_dir/${patches[$i]}"
    fi
    printf '%s %s: %s\n' "${states[$i]}" "${directions[$i]}" "${patches[$i]}"
done

echo OLD_TWO_PATCHES_REVERTED
echo OFFICIAL_HEADLESS_FIX_OK
