# Source this only in the independent TE CPU terminal.
TE_CPU_ROOT="${TE_CPU_ROOT:-/workspace/d00896497/te_cpu_20261026}"
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MC_USE_TENT=0
unset PYTHONPATH LD_PRELOAD
export LD_LIBRARY_PATH="$TE_CPU_ROOT/sysroot/usr/lib/aarch64-linux-gnu:$TE_CPU_ROOT/sysroot/lib/aarch64-linux-gnu"
