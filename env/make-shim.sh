#!/bin/bash
# Recreate an ASCEND_HOME_PATH shim whose only difference from the real toolkit
# is bin/bisheng -> the hardened wrapper. Everything else is symlinked, so CANN
# runtime lookups keep resolving to the installed toolkit.
#
# Usage: make-shim.sh [REAL_CANN_HOME] [BISHENG_WRAPPER] [SHIM_DIR]
set -euo pipefail

REAL=${1:-/usr/local/Ascend/cann-9.2.0}
WRAPPER=${2:-/mnt/xds/sfs/y30048898/DeepInfer/build/prefill950-tools/bin/bisheng}
SHIM=${3:-/mnt/xds/sfs/y30048898/DeepInfer/build/prefill950-tools/ascend-toolkit-shim}

rm -rf "$SHIM"
mkdir -p "$SHIM/bin"

for entry in "$REAL"/*; do
    name=$(basename "$entry")
    if [ "$name" != bin ]; then
        ln -sfn "$entry" "$SHIM/$name"
    fi
done
for entry in "$REAL"/bin/*; do
    name=$(basename "$entry")
    if [ "$name" != bisheng ]; then
        ln -sfn "$entry" "$SHIM/bin/$name"
    fi
done
ln -sfn "$WRAPPER" "$SHIM/bin/bisheng"

"$SHIM/bin/bisheng" --version | head -1
test -d "$SHIM/aarch64-linux/asc/include/adv_api"
echo "shim ready: $SHIM"
