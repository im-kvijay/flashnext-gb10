#!/usr/bin/env bash
# Retrain the MTP drafter on this GB10 (self-distillation on the served target; see docs/drafters.md).
#   bash scripts/build_drafter.sh [--count N] [--epochs E] [--train-experts]
# Needs FLASHNEXT_RUNTIME, FLASHNEXT_DATA and FLASHNEXT_MODEL (from flashnext.local.env, or exported by
# setup_gb10.sh) and no other server running. Steps, each skipped when its output exists:
#   1. KodCode-Light-RL-10K prompts (disjoint from every benchmark here)
#   2. the target generates N coding answers with thinking (served with the recommended profile)
#   3. an eager server replays them and saves the drafter's inputs per prefill chunk (about 95 MB/sequence)
#   4. the checkpoint's MTP weights are copied out, and the non-expert drafter weights are trained
# Output: $FLASHNEXT_DATA/drafter/mtp_trained.pt, served with FLASHNEXT_MTP_OVERRIDE. The drafter only
# proposes tokens, so served outputs are unchanged; only the accepted length (throughput) moves.
# Default N=576 took about 4 hours on the reference host; N=96 (about 1.3 hours) gets most of the gain.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
COUNT=576
EPOCHS=1
EXPERTS=()
while (($#)); do
  case $1 in
    --count) COUNT=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --train-experts) EXPERTS=(--train-experts); shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [[ -z ${FLASHNEXT_RUNTIME:-} && -f flashnext.local.env ]]; then set -a; . ./flashnext.local.env; set +a; fi
: "${FLASHNEXT_RUNTIME:?}" "${FLASHNEXT_DATA:?}" "${FLASHNEXT_MODEL:?}"
PY=$FLASHNEXT_RUNTIME/bin/python
WORK=$FLASHNEXT_DATA/drafter-build
mkdir -p "$WORK" "$FLASHNEXT_DATA/drafter"
URL=http://127.0.0.1:8000
if pgrep -f "[v]llm serve" >/dev/null; then echo "Stop the running vLLM server first." >&2; exit 1; fi
log() { echo "$(date -u +%FT%TZ) $*"; }

SPID=""
start_server() {  # label, then KEY=VALUE overrides on top of the recommended profile
  local label=$1; shift
  log "starting server ($label; log $WORK/$label-server.log)"
  (set -a; . profiles/gb10-8x200k.env; FLASHNEXT_MTP_OVERRIDE=; set +a
   exec env FLASHNEXT_RUNTIME="$FLASHNEXT_RUNTIME" FLASHNEXT_DATA="$FLASHNEXT_DATA" FLASHNEXT_MODEL="$FLASHNEXT_MODEL" \
     FLASHNEXT_KV_BYTES=6442450944 "$@" "$PY" scripts/supervise.py --receipt "$WORK/$label-$(date +%s).jsonl") \
    > "$WORK/$label-server.log" 2>&1 &
  SPID=$!
  until curl -sf $URL/health >/dev/null; do
    kill -0 $SPID 2>/dev/null || { echo "server exited; see $WORK/$label-server.log" >&2; exit 1; }
    sleep 10
  done
  log "server ready"
}
stop_server() {
  [[ -n $SPID ]] || return 0
  kill -TERM $SPID 2>/dev/null || true
  wait $SPID 2>/dev/null || true
  SPID=""
  while pgrep -f "[v]llm serve" >/dev/null; do sleep 5; done
}
trap stop_server EXIT

# 1. Prompts.
PROMPTS=$WORK/kodcode_prompts.jsonl
if [[ ! -s $PROMPTS ]]; then
  log "downloading KodCode-Light-RL-10K prompts"
  "$PY" -c 'import pyarrow' 2>/dev/null || "$PY" -m pip install -q pyarrow
  "$PY" - "$PROMPTS" <<'PY'
import json, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
path = hf_hub_download('KodCode/KodCode-Light-RL-10K', 'data/train-00000-of-00001.parquet', repo_type='dataset')
rows = pq.read_table(path, columns=['question']).to_pylist()
with open(sys.argv[1], 'w') as f:
    f.write('\n'.join(json.dumps({'id': str(i), 'prompt': r['question']}) for i, r in enumerate(rows)))
print(len(rows), 'prompts')
PY
fi

# 2. Target generations (sampled with the checkpoint's defaults, thinking enabled).
GEN=$WORK/generations.jsonl
have=$([[ -f $GEN ]] && grep -c . "$GEN" || echo 0)
if ((have < COUNT)); then
  start_server generate FLASHNEXT_SEQUENCES=16 FLASHNEXT_CONTEXT=16384
  log "generating $COUNT answers ($have done)"
  "$PY" experiments/drafter/generate.py --model "$FLASHNEXT_MODEL" --url $URL --prompts "$PROMPTS" --output "$GEN" \
    --count "$COUNT" --max-tokens 4096 --concurrency 16
  stop_server
fi

# 3. Drafter inputs over prefill chunks of the replayed generations.
CAP=$WORK/captures
if [[ ! -f $WORK/captures.done ]]; then
  rm -rf "$CAP"; mkdir -p "$CAP"
  need=$(( $(grep -c . "$GEN") * 95 / 1000 + 10 ))
  free=$(df --output=avail -B1G "$WORK" | tail -1 | tr -d ' ')
  ((free >= need)) || { echo "captures need about ${need} GB, ${free} GB free at $WORK" >&2; exit 1; }
  start_server capture FLASHNEXT_EAGER=1 FLASHNEXT_CUDAGRAPH_MODE= FLASHNEXT_CAPTURE_MTP_DIR="$CAP" \
    FLASHNEXT_CAPTURE_MTP_OUTPUTS=0 FLASHNEXT_SEQUENCES=1 FLASHNEXT_CONTEXT=16384 FLASHNEXT_PREFILL=2048 \
    FLASHNEXT_PLE_EARLY=0 FLASHNEXT_PLE_PREFILL_AHEAD=
  log "capturing drafter inputs"
  "$PY" experiments/drafter/replay.py --url $URL --generations "$GEN" > "$WORK/replay.log"
  stop_server
  touch "$WORK/captures.done"
fi

# 4. Train.
WEIGHTS=$WORK/mtp-weights
[[ -f $WEIGHTS/target_mixer.safetensors ]] || "$PY" experiments/drafter/extract_mtp.py "$FLASHNEXT_MODEL" "$WEIGHTS"
log "training ($EPOCHS epoch(s)${EXPERTS:+, experts too})"
(cd experiments/drafter && "$PY" train_mtp.py --weights "$WEIGHTS" --captures "$CAP" \
   --draft-vocab "$ROOT/profiles/draft-vocab-vllm-a33b3ba.json" --epochs "$EPOCHS" "${EXPERTS[@]}" --output "$WORK/trained") \
  | tee "$WORK/train.log"
cp "$WORK/trained/mtp_trained.pt" "$FLASHNEXT_DATA/drafter/mtp_trained.pt"
log "drafter written to $FLASHNEXT_DATA/drafter/mtp_trained.pt; held-out acceptance in $WORK/trained/report.json"
log "the captures ($CAP) can be deleted now"
