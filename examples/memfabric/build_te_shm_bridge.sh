#!/usr/bin/env bash
# Link the small ctypes bridge against an EXISTING fixed CPU-only TE build.
# No downloads, package installation, Python imports or device initialization.
set -euo pipefail

: "${MOONCAKE_SOURCE:?Set MOONCAKE_SOURCE to the fixed Mooncake source root}"
: "${MOONCAKE_BUILD:?Set MOONCAKE_BUILD to its existing CPU-only build root}"
output="${1:-${TE_SHM_BRIDGE_OUTPUT:-./libte_shm_bridge.so}}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd -- "$MOONCAKE_SOURCE" && pwd)"
build_root="$(cd -- "$MOONCAKE_BUILD" && pwd)"
cache="$build_root/CMakeCache.txt"
[[ -f "$cache" ]] || { echo "Missing CMakeCache.txt: $cache" >&2; exit 1; }

# These settings alter public headers or can pull accelerator runtime code
# into a shared library before the bridge has a chance to reject it.
for flag in USE_CUDA USE_ASCEND USE_ASCEND_DIRECT USE_ASCEND_HETEROGENEOUS \
    USE_MEMFABRIC USE_TENT USE_UBSHMEM USE_HIP USE_MUSA USE_MACA USE_MLU \
    USE_SUPA USE_NCCL_DEVICE USE_NCCL_HOST USE_MNNVL USE_INTRA_NVLINK \
    ENABLE_MULTI_PROTOCOL WITH_STORE WITH_EP; do
    if grep -Eq "^${flag}:[^=]*=(ON|TRUE|YES|1)$" "$cache"; then
        echo "CPU-only classic build required: $flag is enabled" >&2
        exit 1
    fi
done

te_lib="$build_root/mooncake-transfer-engine/src/libtransfer_engine.so"
[[ -f "$te_lib" ]] || { echo "Missing existing TE library: $te_lib" >&2; exit 1; }
[[ -f "$source_root/mooncake-transfer-engine/include/transfer_engine.h" ]] || exit 1

extra_cxx=()
extra_ld=()
[[ -z "${CXXFLAGS:-}" ]] || read -r -a extra_cxx <<< "$CXXFLAGS"
[[ -z "${LDFLAGS:-}" ]] || read -r -a extra_ld <<< "$LDFLAGS"
if [[ -n "${MOONCAKE_SYSROOT:-}" ]]; then
    multiarch="$("${CXX:-c++}" -print-multiarch)"
    extra_cxx+=("-I$MOONCAKE_SYSROOT/usr/include"
                "-I$MOONCAKE_SYSROOT/usr/include/jsoncpp")
    extra_ld+=("-L$MOONCAKE_SYSROOT/usr/lib/$multiarch")
fi

mkdir -p -- "$(dirname -- "$output")"
"${CXX:-c++}" -std=c++20 -O2 -fPIC -shared -pthread \
    -DGLOG_USE_GLOG_EXPORT \
    -I"$source_root/mooncake-transfer-engine/include" \
    -I"$source_root/mooncake-common/include" \
    "${extra_cxx[@]}" "$script_dir/te_shm_bridge.cpp" \
    -L"$(dirname -- "$te_lib")" -ltransfer_engine \
    "${extra_ld[@]}" -Wl,-z,defs \
    -Wl,-rpath,"$(dirname -- "$te_lib")" -o "$output"
echo "Built $output"
echo "Run with the TE build and private dependency library directories on LD_LIBRARY_PATH."
