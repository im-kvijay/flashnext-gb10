#!/usr/bin/env bash
# Quality check of a running server (start it with scripts/start.sh first); about 2 hours.
#   bash scripts/quality_check.sh [--reference DIR] [--quick]
# Runs, against http://127.0.0.1:$PORT:
#   bench/normal_tasks.py        everyday assistant tasks, graded (JSON, format, memory, code, tools, ...)
#   bench/tool_continuation.py   eight agents continuing multi-turn tool use at 4k
#   bench/retention.py           long-context retrieval
#   bench/suite.py               LiveCodeBench v6 (30), HumanEval (40), GSM8K (20), MMLU-Pro (28); skipped with --quick
# and compares with the base model's results in REFERENCE (default release/reference/base, from the bundle):
# NVIDIA's checkpoint as shipped, served by stock vLLM settings (default NVFP4 MoE kernel, BF16 dense and KV,
# FP32 GDN state, stock MTP). Results go to results/quality-<time>/.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
REF=release/reference/base
QUICK=0
while (($#)); do
  case $1 in
    --reference) REF=$2; shift 2;;
    --quick) QUICK=1; shift;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
set -a; . ./flashnext.local.env; set +a
PY=$FLASHNEXT_RUNTIME/bin/python
URL=http://127.0.0.1:${PORT:-8000}
OUT=results/quality-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
until curl -sf "$URL/health" >/dev/null; do sleep 10; done
EVAL=${FLASHNEXT_EVAL_DATA:-$FLASHNEXT_DATA/eval}
if ((QUICK == 0)) && [[ ! -s $EVAL/livecodebench_test6.jsonl ]]; then
  "$PY" -c 'import pyarrow' 2>/dev/null || "$PY" -m pip install -q pyarrow
  "$PY" scripts/prepare_eval_data.py --output "$EVAL"
fi

echo "== everyday tasks"
"$PY" bench/normal_tasks.py --url "$URL" --output "$OUT/normal-tasks.json" | tee "$OUT/normal-tasks.log"
echo "== multi-turn tool use, eight agents"
"$PY" bench/tool_continuation.py --url "$URL" --model "$FLASHNEXT_MODEL" --input-tokens 4096 \
  --output "$OUT/tools-c8-4k.json" > "$OUT/tools-c8-4k.log" 2>&1 || true
tail -3 "$OUT/tools-c8-4k.log"
echo "== long-context retrieval"
"$PY" bench/retention.py --url "$URL" --model "$FLASHNEXT_MODEL" --label candidate \
  $([[ -f $REF/retention.json ]] && echo --reference "$REF/retention.json") \
  --output "$OUT/retention.json" > "$OUT/retention.log" 2>&1 || true
tail -5 "$OUT/retention.log"
if ((QUICK == 0)); then
  echo "== task suite (about 90 minutes)"
  "$PY" bench/suite.py --url "$URL" --data "$EVAL" --output "$OUT/suite.json" > "$OUT/suite.log" 2>&1
  tail -8 "$OUT/suite.log"
  if [[ -f $REF/suite.json ]]; then
    echo "== paired comparison with the base model"
    "$PY" bench/suite.py --compare "$REF/suite.json" "$OUT/suite.json" | tee "$OUT/suite-vs-base.txt"
  fi
fi
if [[ -f $REF/normal-tasks.json ]]; then
  "$PY" - "$REF/normal-tasks.json" "$OUT/normal-tasks.json" <<'PY'
import json, sys
base, cand = (json.load(open(p)) for p in sys.argv[1:])
print(f"everyday tasks: base {base['passed']}/{base['total']}, this server {cand['passed']}/{cand['total']}")
PY
fi
echo "Results in $OUT"
