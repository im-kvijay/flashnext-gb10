#!/usr/bin/env bash
# One-command setup of the 8 x 200k Flash Next server on a GB10 (DGX Spark class).
#
#   bash scripts/setup_gb10.sh [options]
#
#   --data DIR          where the checkpoint, caches and drafter live (default: ./data; use local NVMe)
#   --model DIR         reuse an already-downloaded checkpoint at the pinned revision (hashes are verified)
#   --drafter FILE      install a retrained MTP drafter (default: release/drafter/mtp_trained.pt when present)
#   --build-drafter     retrain the drafter on this machine instead (about 2-3 hours, see scripts/build_drafter.sh)
#   --runtime DIR       virtual environment for the pinned vLLM (default: ./.venv)
#   --reuse-vllm PY     use an existing Python environment whose vLLM is exactly the pinned build
#   --skip-model        do not download or verify the checkpoint (e.g. to re-run only later steps)
#
# The plugin patches vLLM internals, so it runs only on the vLLM commit in runtime.lock.json.
# An existing vLLM install of any other version is left untouched: a separate virtual
# environment with the pinned wheels is created next to it. Re-running is safe: finished
# steps are detected and skipped. Settings are written to flashnext.local.env, which
# scripts/start.sh reads.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
DATA=${FLASHNEXT_DATA:-$ROOT/data}
RUNTIME=${FLASHNEXT_RUNTIME:-$ROOT/.venv}
MODEL=""
DRAFTER=""
BUILD_DRAFTER=0
REUSE_PY=""
SKIP_MODEL=0
while (($#)); do
  case $1 in
    --data) DATA=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --drafter) DRAFTER=$2; shift 2;;
    --build-drafter) BUILD_DRAFTER=1; shift;;
    --runtime) RUNTIME=$2; shift 2;;
    --reuse-vllm) REUSE_PY=$2; shift 2;;
    --skip-model) SKIP_MODEL=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
mkdir -p "$DATA"
DATA=$(cd "$DATA" && pwd)
MODEL=${MODEL:-$DATA/models/nvidia-flashnext}
step() { echo; echo "==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
WANT_VLLM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["vllm_version"])' runtime.lock.json)

step "Preflight"
[[ $(uname -m) == aarch64 ]] || die "needs Linux aarch64 (GB10)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; install the NVIDIA driver"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU"
[[ $GPU == *GB10* ]] || echo "WARNING: expected a GB10; the profile's memory budget assumes 128 GB unified memory."
command -v python3.12 >/dev/null || die "python3.12 not found (sudo apt install python3.12 python3.12-venv python3.12-dev)"
command -v cc >/dev/null || die "a C compiler is needed for the direct PLE lookup (sudo apt install build-essential)"
echo 'int main(void){return 0;}' | cc -x c -fopenmp -o /dev/null - 2>/dev/null || die "cc lacks OpenMP (sudo apt install build-essential libgomp1)"
TOTAL_GIB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
AVAIL_GIB=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
echo "Memory: ${TOTAL_GIB} GiB total, ${AVAIL_GIB} GiB available now"
((TOTAL_GIB >= 115)) || die "the 8 x 200k profile needs a 128 GB GB10 (found ${TOTAL_GIB} GiB)"
if ((AVAIL_GIB < 112)); then
  echo "WARNING: only ${AVAIL_GIB} GiB available at idle. The 8 x 200k profile leaves about 7 GiB of headroom;"
  echo "         stop other GPU/desktop workloads (e.g. sudo systemctl isolate multi-user.target) or run"
  echo "         fewer agents (FLASHNEXT_SEQUENCES=6 FLASHNEXT_KV_BYTES=19730006016 in flashnext.local.env)."
fi
NEED_GB=$([[ -f $MODEL/verified-lfs.json ]] && echo 25 || echo 160)
FREE_GB=$(df --output=avail -B1G "$DATA" | tail -1 | tr -d ' ')
echo "Disk: ${FREE_GB} GB free at $DATA (need about ${NEED_GB} GB; the checkpoint is 124 GB)"
((FREE_GB >= NEED_GB)) || die "not enough disk at $DATA"
mountpoint_dev=$(df --output=source "$DATA" | tail -1)
[[ $mountpoint_dev == /dev/loop* || $mountpoint_dev == overlay ]] && \
  echo "NOTE: $DATA is on $mountpoint_dev; per-layer embedding reads are fastest from a plain NVMe filesystem."

step "Runtime (vLLM $WANT_VLLM)"
have_vllm() { "$1" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true; }
if [[ -n $REUSE_PY ]]; then
  GOT=$(have_vllm "$REUSE_PY")
  [[ $GOT == "$WANT_VLLM" ]] || die "$REUSE_PY has vLLM '${GOT:-none}', the plugin needs exactly $WANT_VLLM. Omit --reuse-vllm to create a separate environment."
  RUNTIME=$(cd "$(dirname "$REUSE_PY")/.." && pwd)
  echo "Reusing $RUNTIME"
elif [[ -x $RUNTIME/bin/python && $(have_vllm "$RUNTIME/bin/python") == "$WANT_VLLM" ]]; then
  echo "Pinned runtime already present at $RUNTIME"
else
  for py in $(command -v python3 python 2>/dev/null); do
    GOT=$(have_vllm "$py")
    [[ -n $GOT ]] && echo "Found vLLM $GOT at $py; $([[ $GOT == "$WANT_VLLM" ]] && echo 'pass --reuse-vllm to share it' || echo 'leaving it untouched')."
  done
  echo "Creating $RUNTIME with the pinned wheels (about 10 GB download)"
  python3.12 -m venv "$RUNTIME"
  "$RUNTIME/bin/python" -m pip install -q uv==0.12.17
  "$RUNTIME/bin/uv" pip install --python "$RUNTIME/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://flashinfer.ai/whl/ \
    --extra-index-url https://flashinfer.ai/whl/cu130/ \
    -r requirements-runtime.lock.txt
fi
PY=$RUNTIME/bin/python
if [[ -x $RUNTIME/bin/uv ]]; then
  "$RUNTIME/bin/uv" pip install -q --python "$PY" --no-deps -e "$ROOT"
else
  "$PY" -m pip install -q --no-deps -e "$ROOT"
fi
[[ $(have_vllm "$PY") == "$WANT_VLLM" ]] || die "vLLM in $RUNTIME is not $WANT_VLLM"
"$PY" -c 'import importlib.metadata as m; eps=[e.name for e in m.entry_points(group="vllm.general_plugins")]; assert "flashnext_nvme" in eps, eps; import flashnext_gb10.plugin' \
  || die "the flashnext vLLM plugin is not importable from $RUNTIME"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0))'

step "Checkpoint"
if ((SKIP_MODEL)); then
  echo "skipped (--skip-model)"
elif [[ -f $MODEL/verified-lfs.json ]] && "$PY" - "$MODEL" <<'PY'
import json, sys
from pathlib import Path
lock = json.load(open('runtime.lock.json'))
receipt = json.loads((Path(sys.argv[1]) / 'verified-lfs.json').read_text())
ok = receipt['model'] == lock['model'] and receipt['revision'] == lock['model_revision'] and all(
    (Path(sys.argv[1]) / f['path']).stat().st_size == f['size'] for f in receipt['verified'])
sys.exit(0 if ok else 1)
PY
then
  echo "Verified checkpoint at $MODEL"
else
  echo "Downloading/verifying $(python3 -c 'import json; l=json.load(open("runtime.lock.json")); print(l["model"], "@", l["model_revision"][:12])') into $MODEL"
  echo "(resumable; set HF_TOKEN if the Hugging Face repository asks for authentication)"
  mkdir -p "$MODEL"
  "$PY" scripts/download_model.py --directory "$MODEL"
fi

step "Drafter"
DRAFTER_DST=$DATA/drafter/mtp_trained.pt
if [[ -f release/SHA256SUMS ]]; then
  (cd release && sha256sum --quiet -c SHA256SUMS) || die "release bundle files do not match release/SHA256SUMS"
  echo "Release bundle files verified"
  if [[ -z $DRAFTER && ! -f $DRAFTER_DST && -f release/drafter/mtp_trained.pt ]]; then DRAFTER=release/drafter/mtp_trained.pt; fi
fi
if [[ -n $DRAFTER ]]; then
  mkdir -p "$DATA/drafter"
  cp "$DRAFTER" "$DRAFTER_DST.tmp" && mv "$DRAFTER_DST.tmp" "$DRAFTER_DST"
fi
if ((BUILD_DRAFTER)) && [[ ! -f $DRAFTER_DST ]]; then
  FLASHNEXT_RUNTIME=$RUNTIME FLASHNEXT_DATA=$DATA FLASHNEXT_MODEL=$MODEL bash scripts/build_drafter.sh
fi
if [[ -f $DRAFTER_DST ]]; then
  SUM=$(sha256sum "$DRAFTER_DST" | cut -d' ' -f1)
  if grep -q "$SUM" profiles/drafter.sha256 2>/dev/null; then
    echo "Drafter $DRAFTER_DST matches the released weights ($(grep "$SUM" profiles/drafter.sha256 | awk '{print $2}'))"
  else
    echo "Drafter $DRAFTER_DST (sha256 $SUM, not a released build)"
  fi
  "$PY" - "$DRAFTER_DST" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location='cpu', mmap=True)
assert state and all(k.startswith('mtp.') for k in state), 'not an MTP override file'
print(f'  {len(state)} tensors, {sum(v.numel() for v in state.values()) / 1e6:.1f}M parameters')
PY
else
  DRAFTER_DST=""
  echo "No retrained drafter: serving with the checkpoint's own MTP drafter (outputs are identical;"
  echo "fewer accepted draft tokens, so lower throughput). Add one later with --drafter FILE or --build-drafter."
fi

step "Writing flashnext.local.env"
{
  echo "# Machine-specific settings for scripts/start.sh (generated by setup_gb10.sh; not tracked by git)."
  echo "FLASHNEXT_RUNTIME=$RUNTIME"
  echo "FLASHNEXT_DATA=$DATA"
  echo "FLASHNEXT_MODEL=$MODEL"
  [[ -n $DRAFTER_DST ]] && echo "FLASHNEXT_MTP_OVERRIDE=$DRAFTER_DST"
  echo "# Serve beyond localhost: FLASHNEXT_HOST=0.0.0.0 (no authentication; use a firewall or --api-key)."
  echo "FLASHNEXT_HOST=127.0.0.1"
  echo "PORT=8000"
} > flashnext.local.env
cat flashnext.local.env

step "Done"
cat <<EOF
Start the server (first start compiles kernels and takes about 20-30 minutes; later starts about 15):
  bash scripts/start.sh
Then, from another shell, check it end to end:
  bash scripts/smoke_test.sh            # chat, tool call, 8 concurrent 4k agents
  bash scripts/smoke_test.sh --long     # + eight distinct 200k-token contexts (about 30 minutes)
OpenAI-compatible endpoint: http://127.0.0.1:8000/v1, model name "flashnext".
EOF
