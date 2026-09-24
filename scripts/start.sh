#!/usr/bin/env bash
# Start the 8 x 200k server with the recommended profile under the host-memory supervisor.
#   bash scripts/start.sh [--profile FILE] [extra vLLM arguments, e.g. --api-key KEY]
# Reads profiles/gb10-8x200k.env, then flashnext.local.env (written by setup_gb10.sh), then the
# caller's environment for anything already exported. Logs go to results/serve-<time>.log.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PROFILE=profiles/gb10-8x200k.env
if [[ ${1:-} == --profile ]]; then PROFILE=$2; shift 2; fi
[[ -f flashnext.local.env ]] || { echo "Run scripts/setup_gb10.sh first (flashnext.local.env is missing)." >&2; exit 1; }
# Exported variables win over both files.
saved=$(export -p | grep -E '^declare -x (FLASHNEXT_|PORT=|PYTORCH_)' || true)
set -a
. "$PROFILE"
. ./flashnext.local.env
set +a
eval "$saved"
if pgrep -f "[v]llm serve" >/dev/null; then
  echo "A vLLM server is already running; stop it first (two servers do not fit in GB10 memory)." >&2
  exit 1
fi
mkdir -p results
STAMP=$(date +%Y%m%d-%H%M%S)
echo "Profile $PROFILE; model $FLASHNEXT_MODEL; drafter ${FLASHNEXT_MTP_OVERRIDE:-checkpoint}"
echo "Serving on http://${FLASHNEXT_HOST:-127.0.0.1}:${PORT:-8000}/v1 as model \"flashnext\" once /health returns 200."
echo "Log: results/serve-$STAMP.log  memory receipt: results/serve-$STAMP.jsonl"
exec "$FLASHNEXT_RUNTIME/bin/python" scripts/supervise.py --receipt "results/serve-$STAMP.jsonl" -- "$@" \
  > >(tee "results/serve-$STAMP.log") 2>&1
