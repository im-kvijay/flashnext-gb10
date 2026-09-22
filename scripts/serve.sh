#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME=${FLASHNEXT_RUNTIME:-$ROOT/.venv}
DATA=${FLASHNEXT_DATA:-$ROOT/data}
MODEL=${FLASHNEXT_MODEL:-$DATA/models/nvidia-flashnext}
export FLASHNEXT_PLE_NVME_DIR=${FLASHNEXT_PLE_NVME_DIR:-$DATA/ple}
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_USE_DEEP_GEMM=0
export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=4
export FLASHINFER_NVCC_THREADS=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VLLM_CACHE_ROOT=$DATA/cache/vllm
export TRITON_CACHE_DIR=$DATA/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$DATA/cache/inductor
export FLASHINFER_WORKSPACE_BASE=$DATA/cache/flashinfer
export PYTHONUNBUFFERED=1
mkdir -p "$FLASHNEXT_PLE_NVME_DIR" "$DATA/cache"
EXTRA=()
if [[ -n ${FLASHNEXT_KV_BYTES:-} ]]; then
  EXTRA+=(--kv-cache-memory-bytes "$FLASHNEXT_KV_BYTES")
fi
if [[ -n ${FLASHNEXT_KV_DTYPE:-} ]]; then
  EXTRA+=(--kv-cache-dtype "$FLASHNEXT_KV_DTYPE")
fi
if [[ -n ${FLASHNEXT_PREFILL_PER_REQUEST:-} ]]; then
  EXTRA+=(--long-prefill-token-threshold "$FLASHNEXT_PREFILL_PER_REQUEST")
fi
if [[ -n ${FLASHNEXT_PROFILE_DIR:-} ]]; then
  EXTRA+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${FLASHNEXT_PROFILE_DIR}\"}")
fi
if [[ ${FLASHNEXT_MTP:-0} != 0 ]]; then
  EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${FLASHNEXT_MTP}}")
fi
if [[ ${FLASHNEXT_EAGER:-0} == 1 ]]; then EXTRA+=(--enforce-eager); fi
exec "$RUNTIME/bin/vllm" serve "$MODEL" \
  --served-model-name flashnext --host 127.0.0.1 --port "${PORT:-8000}" \
  --max-model-len "${FLASHNEXT_CONTEXT:-212992}" \
  --max-num-seqs "${FLASHNEXT_SEQUENCES:-8}" \
  --max-num-batched-tokens "${FLASHNEXT_PREFILL:-2048}" \
  --enable-chunked-prefill \
  --gpu-memory-utilization "${FLASHNEXT_MEMORY_FRACTION:-0.88}" \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  "${EXTRA[@]}" "$@"
