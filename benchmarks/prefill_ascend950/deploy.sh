#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p build/prefill950-deploy
exec > >(tee -a "build/prefill950-deploy/deploy-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
VENV=.venv-prefill950
PYTHON=python3.12
CANN_ENV=
MODEL_DIR=
WEIGHTS=dummy
WHEELHOUSE=
DEVICES=0,1,2,3,4,5,6,7
while (($#)); do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --cann-env) CANN_ENV="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux ]] || { echo "Linux/Ascend 950 is required" >&2; exit 2; }
[[ -n "$MODEL_DIR" && -n "$CANN_ENV" && -f "$CANN_ENV" ]] || {
  echo "Required: --cann-env <installed CANN set_env.sh> --model-dir <directory>" >&2; exit 2;
}
[[ "$WEIGHTS" == dummy || "$WEIGHTS" == real ]] || { echo "Invalid --weights" >&2; exit 2; }
set +u
# CANN scripts may reference unset vendor variables.
source "$CANN_ENV"
set -u
export DEEPINFER_ACCELERATOR=ascend
export MAX_JOBS="${MAX_JOBS:-16}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if [[ ! -d "$VENV" ]]; then "$PYTHON" -m venv "$VENV"; fi
source "$VENV/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "venv must use Python 3.12"'
OFFLINE=()
if [[ -n "$WHEELHOUSE" ]]; then
  export PIP_NO_INDEX=1
  export PIP_FIND_LINKS="$(realpath "$WHEELHOUSE")"
  OFFLINE=(--offline)
fi
python -m pip install setuptools==77.0.3 wheel==0.45.1
REQUIREMENTS=requirements-ascend.txt
if [[ -n "$WHEELHOUSE" ]]; then
  # Strip remote find-links from a flattened copy; never edit upstream requirements.
  python - <<'PY'
from pathlib import Path
def expand(path):
    for line in path.read_text().splitlines():
        if line.startswith('-r '):
            yield from expand(path.parent / line[3:].strip())
        elif not line.startswith('--'):
            yield line
Path('build/prefill950-deploy/requirements-offline.txt').write_text('\n'.join(expand(Path('requirements-ascend.txt'))) + '\n')
PY
  REQUIREMENTS=build/prefill950-deploy/requirements-offline.txt
fi
python -m pip install -r "$REQUIREMENTS"
# Ascend packages only; preserve upstream build target selectors and gitlink revisions.
export WHEEL_DIR="$ROOT/build/ascend-$(uname -m)/vendor-wheels"
mkdir -p "$WHEEL_DIR"
for PACKAGE in DeepSelect FlashMLA DeepEP-Ascend DeepGEMM-Ascend; do
  git submodule update --init --recursive -- "vendor/$PACKAGE"
  (
    cd "vendor/$PACKAGE"
    case "$PACKAGE" in
      DeepSelect) export DEEP_SELECT_BUILD_TARGET_PLATFORM=ASCEND ;;
      FlashMLA) export FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND ;;
    esac
    python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"
  )
done
python -m pip install --no-deps --no-index --find-links "$WHEEL_DIR" deep_select flash_mla deep_ep deep_gemm
bash scripts/build.sh build
python -m pip check
python -m benchmarks.prefill_ascend950.cli prepare-model --model-dir "$MODEL_DIR" --weights "$WEIGHTS" "${OFFLINE[@]}"
mkdir -p "$VENV/prefill950"
printf '%s\n' "$(realpath "$CANN_ENV")" > "$VENV/prefill950/cann-env.path"
python -m pip freeze > "$VENV/prefill950/requirements.resolved.txt"
python -m benchmarks.prefill_ascend950.cli deployment-manifest --out "$VENV/prefill950/deployment.json"
python -m benchmarks.prefill_ascend950.cli doctor --devices "$DEVICES" --model-dir "$MODEL_DIR" --weights "$WEIGHTS" --out "$VENV/prefill950/doctor"
echo "Deployment and preflight complete: $VENV/prefill950"
