#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $(uname -m) == aarch64 ]] || { echo 'This runtime requires Linux aarch64/GB10.' >&2; exit 1; }
RUNTIME=${FLASHNEXT_RUNTIME:-$ROOT/.venv}
DATA=${FLASHNEXT_DATA:-$ROOT/data}
python3 -m venv "$RUNTIME"
"$RUNTIME/bin/python" -m pip install uv==0.12.17
"$RUNTIME/bin/uv" pip install --python "$RUNTIME/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/ \
  --extra-index-url https://flashinfer.ai/whl/cu130/ \
  -r "$ROOT/requirements-runtime.lock.txt"
"$RUNTIME/bin/uv" pip install --python "$RUNTIME/bin/python" -e "$ROOT"
mkdir -p "$DATA/models"
"$RUNTIME/bin/python" "$ROOT/scripts/download_model.py" --directory "$DATA/models/nvidia-flashnext"
"$RUNTIME/bin/uv" pip freeze --python "$RUNTIME/bin/python" > "$DATA/runtime-resolved.txt"
