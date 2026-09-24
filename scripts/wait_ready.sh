#!/usr/bin/env bash
# Wait until the server started by scripts/start.sh answers /health (default timeout 60 minutes; the
# first start compiles kernels). Exits 1 with the log tail if the server exits first.
#   bash scripts/wait_ready.sh [TIMEOUT_SECONDS]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
set -a; . ./flashnext.local.env; set +a
URL=http://127.0.0.1:${PORT:-8000}/health
TIMEOUT=${1:-3600}
start=$(date +%s)
until curl -sf "$URL" >/dev/null; do
  if [[ -f results/server.pid ]] && ! kill -0 "$(cat results/server.pid)" 2>/dev/null; then
    echo "The server exited before becoming ready. Last log lines (results/serve-latest.log):" >&2
    tail -40 results/serve-latest.log >&2 || true
    exit 1
  fi
  if (( $(date +%s) - start > TIMEOUT )); then
    echo "Not ready after $TIMEOUT s; see results/serve-latest.log" >&2
    exit 1
  fi
  sleep 10
done
echo "Ready after $(( $(date +%s) - start )) s: http://127.0.0.1:${PORT:-8000}/v1 (model \"flashnext\")"
