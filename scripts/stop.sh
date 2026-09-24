#!/usr/bin/env bash
# Stop the server started from this checkout by scripts/start.sh (foreground or --background).
# Signals only this checkout's supervisor, which stops its own vLLM process group; other vLLM
# servers on the machine are not touched.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
pids=()
if [[ -f results/server.pid ]] && kill -0 "$(cat results/server.pid)" 2>/dev/null; then
  pids+=("$(cat results/server.pid)")
else
  for p in $(pgrep -f "[s]cripts/supervise.py --receipt" || true); do
    [[ $(readlink -f "/proc/$p/cwd" 2>/dev/null) == "$ROOT" ]] && pids+=("$p")
  done
fi
if ((${#pids[@]} == 0)); then
  rm -f results/server.pid
  echo "No server from this checkout is running."
  exit 0
fi
kill -TERM "${pids[@]}"
for _ in $(seq 60); do
  alive=0
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  ((alive)) || { rm -f results/server.pid; echo "Stopped."; exit 0; }
  sleep 1
done
echo "The supervisor did not exit within 60 s (PIDs ${pids[*]}); check results/serve-latest.log." >&2
exit 1
