#!/usr/bin/env bash
# Execute (do NOT source) the MegaMoE-only power probe:
#   bash benchmarks/prefill_ascend950/tools/run_mega_moe_power_probe.sh --iters 3000
# All environment changes stay inside this subshell.
set -euo pipefail

ROOT=/mnt/xds/sfs/y30048898/DeepInfer
cd "$ROOT"
# Save the caller's LD_PRELOAD, then clear it so setup commands do not repeat
# ld.so "cannot be preloaded" errors for host-absent entries; it is re-added
# (filtered by existence) before exec below.
CALLER_PRELOAD="${LD_PRELOAD:-}"
export LD_PRELOAD=
source build/prefill950-tools/prefill950-env.sh

GCC14_ROOT=/mnt/xds/sfs/l00972766/rt/gcc-toolset-14-root
export PREFILL950_GCC14=$GCC14_ROOT
export PATH="$GCC14_ROOT/usr/bin:$PATH"
export CC="$GCC14_ROOT/usr/bin/gcc"
export CXX="$GCC14_ROOT/usr/bin/g++"
export LD_LIBRARY_PATH="$GCC14_ROOT/usr/lib64:${LD_LIBRARY_PATH:-}"

# LD_PRELOAD: the msprof compat shim is required; keep existing entries only if
# they exist on this host (images differ: e.g. /usr/lib64/libjemalloc.so.2 is
# absent on some machines and only produces ld.so noise when preloaded).
COMPAT_PRELOAD="$ROOT/build/prefill950-tools/compat/libmsprof_compat.so"
[ -f "$COMPAT_PRELOAD" ] || { echo "missing $COMPAT_PRELOAD" >&2; exit 2; }
PRELOAD="$COMPAT_PRELOAD"
for entry in ${CALLER_PRELOAD//:/ }; do
    [ -z "$entry" ] && continue
    case "$entry" in
        /*) [ -f "$entry" ] && PRELOAD="$PRELOAD:$entry" ;;
        *) PRELOAD="$PRELOAD:$entry" ;;
    esac
done
export LD_PRELOAD="$PRELOAD"

# deep_jit resolves bisheng via ASCEND_HOME_PATH; use the hardened shim.
export PREFILL950_JIT_ASCEND_HOME=$ROOT/build/prefill950-tools/ascend-toolkit-shim
export ASCEND_HOME_PATH=$PREFILL950_JIT_ASCEND_HOME
export PREFILL950_ALLOW_LOCAL_SOURCE=1
export PYTHONPATH=$ROOT

export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-28100}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-24400-24500}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29517}

exec python benchmarks/prefill_ascend950/tools/mega_moe_power_probe.py "$@"
