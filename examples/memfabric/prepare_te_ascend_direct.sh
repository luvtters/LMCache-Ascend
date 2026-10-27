#!/usr/bin/env bash
# Prepare a SEPARATE fixed-version classic Ascend Direct build. Never runs TE.
set -euo pipefail
mode="${1:---print-plan}"
[[ $# -eq 0 ]] || shift
case "$mode" in --print-plan|--configure-only|--build) ;; *) echo "Use --print-plan, --configure-only or --build" >&2; exit 2 ;; esac
: "${MOONCAKE_SOURCE:?Set the existing fixed Mooncake source root}"
: "${MOONCAKE_ASCEND_BUILD:?Set a new, separate Ascend Direct build directory}"
: "${ASCEND_HOME_PATH:?Set the existing CANN root, e.g. /usr/local/Ascend/cann-9.1.0}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
configure=("${CMAKE:-cmake}" -S "$MOONCAKE_SOURCE" -B "$MOONCAKE_ASCEND_BUILD"
    -DCMAKE_BUILD_TYPE=Release -DWITH_TE=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF
    -DWITH_CONDUCTOR=OFF -DWITH_P2P_STORE=OFF -DWITH_EP=OFF
    -DUSE_CUDA=OFF -DUSE_ASCEND=OFF -DUSE_ASCEND_DIRECT=ON
    -DUSE_ASCEND_HETEROGENEOUS=OFF -DUSE_MEMFABRIC=OFF -DUSE_TENT=OFF
    -DUSE_ETCD=OFF -DUSE_REDIS=OFF -DENABLE_MULTI_PROTOCOL=OFF
    -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF "$@")
build=("${CMAKE:-cmake}" --build "$MOONCAKE_ASCEND_BUILD" --target engine --parallel 2)
if [[ "$mode" == --print-plan ]]; then
    printf 'Fixed source required: f751cff2b67668ae97f1ff7e45fb2154fdbff64a\n'
    printf '%q ' "${configure[@]}"; printf '\n'
    printf '%q ' "${build[@]}"; printf '\nNo native module or device is initialized.\n'
    exit 0
fi
[[ -f "$MOONCAKE_SOURCE/CMakeLists.txt" ]] || { echo "Missing existing Mooncake source" >&2; exit 1; }
if [[ -e "$MOONCAKE_SOURCE/.git" ]]; then
    revision="$(git -C "$MOONCAKE_SOURCE" rev-parse HEAD)"
    [[ "$revision" == f751cff2b67668ae97f1ff7e45fb2154fdbff64a ]] || { echo "Unexpected source revision" >&2; exit 1; }
else
    echo "Archive checkout: verify fixed revision against its source acquisition receipt." >&2
fi
arch="$(uname -m)"
toolkit="$ASCEND_HOME_PATH/$arch-linux"
for file in include/acl/acl.h include/adxl/adxl_engine.h lib64/libllm_datadist.so lib64/libascendcl.so; do
    [[ -f "$toolkit/$file" ]] || { echo "Missing existing CANN dependency: $toolkit/$file" >&2; exit 1; }
done
cache="$MOONCAKE_ASCEND_BUILD/CMakeCache.txt"
if [[ -f "$cache" ]]; then
    echo "Use a new build directory; do not reconfigure the existing CPU or LMCache environment." >&2
    exit 1
fi
"${configure[@]}"
grep -Eq '^USE_ASCEND_DIRECT:BOOL=ON$' "$cache" || { echo "Ascend Direct not enabled" >&2; exit 1; }
for flag in USE_CUDA USE_ASCEND USE_ASCEND_HETEROGENEOUS USE_MEMFABRIC USE_TENT WITH_STORE WITH_EP ENABLE_MULTI_PROTOCOL; do
    if grep -Eq "^${flag}:[^=]*=(ON|TRUE|YES|1)$" "$cache"; then
        echo "Unexpected enabled feature: $flag" >&2; exit 1
    fi
done
[[ "$mode" != --build ]] || "${build[@]}"
echo "Build preparation complete; no TE import or device test was run."
