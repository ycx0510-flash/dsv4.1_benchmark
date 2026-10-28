#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VENV=.venv-prefill950
CANN_ENV=
ARGS=()
while (($#)); do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --cann-env) CANN_ENV="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ "$(uname -s)" == Linux ]] || { echo "Linux/Ascend 950 is required" >&2; exit 2; }
if [[ -z "$CANN_ENV" && -f "$VENV/prefill950/cann-env.path" ]]; then
  IFS= read -r CANN_ENV < "$VENV/prefill950/cann-env.path"
fi
[[ -n "$CANN_ENV" && -f "$CANN_ENV" ]] || { echo "Specify --cann-env, or run deploy.sh first" >&2; exit 2; }
set +u
source "$CANN_ENV"
set -u
source "$VENV/bin/activate"
export DEEPINFER_ACCELERATOR=ascend
exec python -m benchmarks.prefill_ascend950.cli run "${ARGS[@]}"
