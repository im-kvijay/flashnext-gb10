#!/usr/bin/env bash
# Start the 8 x 200k server with the recommended profile under the host-memory supervisor.
#   bash scripts/start.sh [--background] [--profile FILE] [extra vLLM arguments, e.g. --api-key KEY]
# Reads profiles/gb10-8x200k.env, then flashnext.local.env (written by setup_gb10.sh), then the
# caller's environment for anything already exported. Logs go to results/serve-<time>.log.
# --background returns immediately (server PID in results/server.pid); then use
# scripts/wait_ready.sh to wait for it and scripts/stop.sh to stop it.
#
# Memory fit: the profile peaked at about 111 GiB on the reference GB10, which had 118 GiB available
# before start (the supervisor stops the server below FLASHNEXT_MIN_AVAILABLE_GIB, 6). If this machine
# has less available now (desktop session, other services), the start shrinks the budget in this
# order until it fits, and prints what it changed: PLE row cache 3 -> 1 GiB (about 5% slower decode
# at 200k), 1,024-token prefill chunks (slower prefill), then one agent fewer at a time (3.06 GiB of
# KV cache each). Outputs are unaffected. FLASHNEXT_AUTO_FIT=0 disables it.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PROFILE=profiles/gb10-8x200k.env
BACKGROUND=0
while (($#)); do
  case $1 in
    --background) BACKGROUND=1; shift;;
    --profile) PROFILE=$2; shift 2;;
    *) break;;
  esac
done
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

if [[ ${FLASHNEXT_AUTO_FIT:-1} == 1 ]]; then
  # Settings the caller exported are left alone.
  pinned=$(grep -oE 'FLASHNEXT_[A-Z_]+=' <<<"$saved" | tr -d = | tr '\n' ' ' || true)
  fit=$(FLASHNEXT_PINNED=$pinned python3 - <<'PY'
import os
env = os.environ
pinned = set(env.get('FLASHNEXT_PINNED', '').split())
avail = float(env.get('FLASHNEXT_ASSUME_AVAILABLE_GIB') or 0) or next(
    int(l.split()[1]) / 2**20 for l in open('/proc/meminfo') if l.startswith('MemAvailable'))
floor = float(env.get('FLASHNEXT_MIN_AVAILABLE_GIB', '6'))
seqs0 = seqs = int(env.get('FLASHNEXT_SEQUENCES', '8'))
kv = int(env.get('FLASHNEXT_KV_BYTES') or 26306674688)
cache = float(env.get('FLASHNEXT_PLE_ROW_CACHE_GIB') or 0)
prefill = int(env.get('FLASHNEXT_PREFILL', '2048'))
# Peak use measured with the shipped profile (8 agents, 26,306,674,688 KV bytes, 3 GiB row cache,
# 2,048-token chunks): 111.2 GiB. Scale it for this configuration.
need = 111.2 - (26306674688 - kv) / 2**30 - (3 - cache) + floor + 0.5
deficit = need - avail
changes, out = [], {}
if deficit > 0 and cache > 1 and 'FLASHNEXT_PLE_ROW_CACHE_GIB' not in pinned:
    deficit -= cache - 1; out['FLASHNEXT_PLE_ROW_CACHE_GIB'] = '1'
    changes.append(f'PLE row cache {cache:g} -> 1 GiB')
if deficit > 0 and prefill > 1024 and 'FLASHNEXT_PREFILL' not in pinned:
    deficit -= 0.6; out['FLASHNEXT_PREFILL'] = '1024'
    changes.append(f'prefill chunks {prefill} -> 1024 tokens')
per_seq = kv / seqs
if not pinned & {'FLASHNEXT_SEQUENCES', 'FLASHNEXT_KV_BYTES'}:
    while deficit > 0 and seqs > 2:
        seqs -= 1; kv -= per_seq; deficit -= per_seq / 2**30
    if seqs != seqs0:
        out['FLASHNEXT_SEQUENCES'] = str(seqs); out['FLASHNEXT_KV_BYTES'] = str(int(kv))
        changes.append(f'{seqs} concurrent agents instead of {seqs0}')
for k, v in out.items():
    print(f'export {k}={v}')
print(f'echo "Memory: {avail:.1f} GiB available now; this configuration needs about {need:.1f} GiB."')
if changes:
    print('echo "Memory fit: ' + '; '.join(changes) + ' (FLASHNEXT_AUTO_FIT=0 disables this)."')
if deficit > 0:
    print('echo "Not enough memory for this configuration: stop other workloads (desktop session, '
          'other GPU jobs) or lower FLASHNEXT_SEQUENCES and FLASHNEXT_KV_BYTES." >&2; exit 1')
PY
)
  eval "$fit"
fi

mkdir -p results
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=results/serve-$STAMP.log
echo "Profile $PROFILE; model $FLASHNEXT_MODEL; drafter ${FLASHNEXT_MTP_OVERRIDE:-checkpoint}"
echo "${FLASHNEXT_SEQUENCES:-8} agents x ${FLASHNEXT_CONTEXT:-212992} tokens; serving on http://${FLASHNEXT_HOST:-127.0.0.1}:${PORT:-8000}/v1"
echo "as model \"flashnext\" once /health returns 200. Log: $LOG  memory receipt: results/serve-$STAMP.jsonl"
if ((BACKGROUND)); then
  setsid nohup "$FLASHNEXT_RUNTIME/bin/python" scripts/supervise.py --receipt "results/serve-$STAMP.jsonl" -- "$@" \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > results/server.pid
  ln -sfn "serve-$STAMP.log" results/serve-latest.log
  echo "Started in the background (PID $(cat results/server.pid)). Wait for it: bash scripts/wait_ready.sh"
  exit 0
fi
ln -sfn "serve-$STAMP.log" results/serve-latest.log
exec "$FLASHNEXT_RUNTIME/bin/python" scripts/supervise.py --receipt "results/serve-$STAMP.jsonl" -- "$@" \
  > >(tee "$LOG") 2>&1
