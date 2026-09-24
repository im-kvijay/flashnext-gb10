#!/usr/bin/env bash
# Check that this machine's server reproduces the reference numerics (about 15 minutes).
#   bash scripts/verify_fidelity.sh [REFERENCE_DIR]    (default: release/reference from the bundle)
# Teacher-forced scoring of 8 coding-agent continuations and 8 real source files (bench/fidelity.py),
# compared against records from the reference GB10:
#   fidelity-bf16-dense.json  unquantized dense layers, BF16 KV (the quality reference)
#   fidelity-profile.json     the recommended profile (what this machine should match)
# Pass: against the BF16-dense reference, workload top-1 >= 95% and approx KL <= 0.012, codebase top-1 >= 84%
# (reference host: 96.6% / 0.0072 and 86.7% / 0.20). Start the server with scripts/start.sh first.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
REF=${1:-release/reference}
set -a; . ./flashnext.local.env; set +a
PY=$FLASHNEXT_RUNTIME/bin/python
URL=http://127.0.0.1:${PORT:-8000}
for f in fidelity-source.json fidelity-bf16-dense.json fidelity-profile.json; do
  [[ -f $REF/$f ]] || { echo "missing $REF/$f (it ships in the release bundle)" >&2; exit 1; }
done
until curl -sf "$URL/health" >/dev/null; do sleep 10; done
OUT=results/fidelity-$(date +%Y%m%d-%H%M%S).json
mkdir -p results
CORPUS=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(cd bench && "$PY" fidelity.py --model "$FLASHNEXT_MODEL" --url "$URL" --source "$ROOT/$REF/fidelity-source.json" \
   --corpus-root "$CORPUS" --output "$ROOT/$OUT")
"$PY" - "$REF" "$OUT" <<'PY'
import sys
sys.path.insert(0, 'bench')
from fidelity import compare
ref, out = sys.argv[1:]
dense = compare(f'{ref}/fidelity-bf16-dense.json', out)
same = compare(f'{ref}/fidelity-profile.json', out)
for name, r in (('vs BF16-dense reference', dense), ('vs reference host, same profile', same)):
    print(name)
    for g in ('workload', 'codebase'):
        if g in r:
            print(f"  {g:9s} top-1 {r[g]['top1_agreement']:.1%}  approx KL {r[g]['approx_kl']:.4f}  "
                  f"NLL change {r[g]['mean_nll_increase']:+.4f}  ({r[g]['positions']} positions)")
ok = (dense.get('workload', {}).get('top1_agreement', 0) >= 0.95 and dense['workload']['approx_kl'] <= 0.012
      and dense.get('codebase', {}).get('top1_agreement', 0) >= 0.84)
print('PASS' if ok else 'FAIL: numerics differ from the reference more than expected')
sys.exit(0 if ok else 1)
PY
