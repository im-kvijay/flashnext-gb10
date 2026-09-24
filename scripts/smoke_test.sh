#!/usr/bin/env bash
# End-to-end check of a running server (start it with scripts/start.sh first).
#   bash scripts/smoke_test.sh          chat + tool call + eight concurrent 4k agents (about 5 minutes)
#   bash scripts/smoke_test.sh --long   also eight distinct 200k-token contexts (about 30 minutes)
# Results are written to results/smoke-<time>/.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
set -a; . ./flashnext.local.env; set +a
PY=$FLASHNEXT_RUNTIME/bin/python
URL=http://127.0.0.1:${PORT:-8000}
OUT=results/smoke-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "Waiting for $URL/health ..."
until curl -sf "$URL/health" >/dev/null; do sleep 10; done

echo "== chat"
"$PY" - "$URL" <<'PY'
import json, sys, time, urllib.request
url = sys.argv[1]
def post(body):
    req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
t = time.time()
r = post({'model': 'flashnext', 'max_tokens': 2048,
          'messages': [{'role': 'user', 'content': 'What is 17 * 23? Reply with only the number.'}]})
msg = r['choices'][0]['message']
print(f"answer {msg['content']!r} ({r['usage']['completion_tokens']} tokens, {time.time() - t:.1f}s)")
assert '391' in (msg['content'] or ''), 'wrong or empty answer'
tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file from the repository',
          'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
r = post({'model': 'flashnext', 'max_tokens': 2048, 'tools': tools,
          'messages': [{'role': 'user', 'content': 'Open src/main.py and tell me what it does.'}]})
calls = r['choices'][0]['message'].get('tool_calls') or []
print('tool call', [(c['function']['name'], c['function']['arguments']) for c in calls])
assert calls and calls[0]['function']['name'] == 'read_file', 'no tool call'
PY

echo "== eight concurrent agents, 4k context, natural coding workload"
"$PY" bench/concurrency.py --url "$URL" --model "$FLASHNEXT_MODEL" --input-tokens 4096 --output-tokens 2048 \
  --mode workload --output "$OUT/workload-c8-4k.json" > "$OUT/workload-c8-4k.log"
"$PY" - "$OUT/workload-c8-4k.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get('engine_counter_deltas', {})
drafts = c.get('vllm:spec_decode_num_drafts_total', 0)
acc = 1 + c.get('vllm:spec_decode_num_accepted_tokens_total', 0) / drafts if drafts else float('nan')
print(f"aggregate {d['summary']['all_streams_overlap_output_tps']:.1f} output tok/s, accepted length {acc:.2f} "
      "(reference host: 120-128 tok/s, 2.45-2.5)")
PY

if [[ ${1:-} == --long ]]; then
  echo "== eight distinct 200k-token codebase contexts (primed, then 2048 new tokens each)"
  CORPUS=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  "$PY" bench/concurrency.py --url "$URL" --model "$FLASHNEXT_MODEL" --input-tokens 200000 --output-tokens 2048 \
    --mode codebase --corpus-root "$CORPUS" --warm-prefixes --timeout 7200 --output "$OUT/codebase-c8-200k.json" \
    > "$OUT/codebase-c8-200k.log"
  "$PY" - "$OUT/codebase-c8-200k.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get('engine_counter_deltas', {})
drafts = c.get('vllm:spec_decode_num_drafts_total', 0)
acc = 1 + c.get('vllm:spec_decode_num_accepted_tokens_total', 0) / drafts if drafts else float('nan')
print(f"8 x 200k: aggregate {d['summary']['all_streams_overlap_output_tps']:.1f} output tok/s, accepted length {acc:.2f}")
PY
fi
echo "OK. Details in $OUT"
