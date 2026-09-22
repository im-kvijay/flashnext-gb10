#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME=${FLASHNEXT_RUNTIME:-$ROOT/.venv}
export PATH="$RUNTIME/bin:$PATH"
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
"$RUNTIME/bin/python" "$ROOT/scripts/reclaim_model_cache.py" "$MODEL" --also-directory "$RUNTIME"
EXTRA=()
if [[ -n ${FLASHNEXT_MOE_BACKEND:-} ]]; then
  EXTRA+=(--moe-backend "$FLASHNEXT_MOE_BACKEND")
fi
if [[ -n ${FLASHNEXT_KV_BYTES:-} ]]; then
  EXTRA+=(--kv-cache-memory-bytes "$FLASHNEXT_KV_BYTES")
fi
if [[ -n ${FLASHNEXT_KV_DTYPE:-} ]]; then
  EXTRA+=(--kv-cache-dtype "$FLASHNEXT_KV_DTYPE")
fi
if [[ -n ${FLASHNEXT_SSM_DTYPE:-} ]]; then
  # The checkpoint asks for float32 GDN state; bfloat16 halves state traffic
  # and memory but changes numerics, so it is opt-in and needs the quality gate.
  EXTRA+=(--mamba-ssm-cache-dtype "$FLASHNEXT_SSM_DTYPE")
fi
if [[ -n ${FLASHNEXT_PREFILL_PER_REQUEST:-} ]]; then
  EXTRA+=(--long-prefill-token-threshold "$FLASHNEXT_PREFILL_PER_REQUEST")
fi
if [[ -n ${FLASHNEXT_PROFILE_DIR:-} ]]; then
  EXTRA+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${FLASHNEXT_PROFILE_DIR}\",\"torch_profiler_with_stack\":false}")
fi
if [[ ${FLASHNEXT_TEXT_ONLY:-0} == 1 ]]; then EXTRA+=(--language-model-only); fi
if [[ -n ${FLASHNEXT_DFLASH_MODEL:-} ]]; then
  if [[ ${FLASHNEXT_MTP:-0} != 0 || -n ${FLASHNEXT_DRAFT_VOCAB:-} ]]; then
    echo 'DFlash requires MTP and its reduced draft vocabulary disabled' >&2
    exit 2
  fi
  if [[ ${FLASHNEXT_EAGER:-0} != 1 ]]; then
    echo 'The experimental DFlash attachment currently requires FLASHNEXT_EAGER=1' >&2
    exit 2
  fi
  DFLASH_CONFIG=$("$RUNTIME/bin/python" -c 'import json,sys; print(json.dumps(dict(method="dflash",model=sys.argv[1],num_speculative_tokens=int(sys.argv[2]),draft_tensor_parallel_size=1,quantization=None,kv_cache_dtype="auto")))' "$FLASHNEXT_DFLASH_MODEL" "${FLASHNEXT_DFLASH_K:-4}")
  EXTRA+=(--speculative-config "$DFLASH_CONFIG")
fi
if [[ ${FLASHNEXT_MTP:-0} != 0 ]]; then
  DRAFT_EXTRA=""
  if [[ -n ${FLASHNEXT_DRAFT_VOCAB:-} ]]; then
    export FLASHNEXT_DRAFT_VOCAB
    DRAFT_EXTRA=',"use_local_argmax_reduction":true'
  fi
  # Draft steps after the first reuse the first step's sparse-attention
  # indices. Affects only draft proposals; verification is unchanged.
  if [[ ${FLASHNEXT_MTP_INDEX_SHARE:-0} == 1 ]]; then
    DRAFT_EXTRA+=',"index_share_for_mtp_iteration":true'
  fi
  # The NVIDIA target uses NVFP4 experts but its MTP block uses FP8.
  # B12x's NVFP4 MoE backend cannot also serve the FP8 draft block.
  if [[ ${FLASHNEXT_MOE_BACKEND:-} == b12x || ${FLASHNEXT_MOE_BACKEND:-} == flashinfer_b12x ]]; then
    DRAFT_EXTRA+=',"moe_backend":"auto"'
  fi
  EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${FLASHNEXT_MTP}${DRAFT_EXTRA}}")
fi
if [[ ${FLASHNEXT_EAGER:-0} == 1 ]]; then EXTRA+=(--enforce-eager); fi
# FLASHNEXT_CUDAGRAPH_MODE=FULL_AND_PIECEWISE also captures uniform decode
# batches, including attention and GDN, as whole graphs.
# Capture sizes count tokens, not requests. MTP K verifies K+1 tokens per
# request, so eight MTP-3 agents need a 32-token target graph.
CAPTURE_SIZES=${FLASHNEXT_CAPTURE_SIZES:-$("$RUNTIME/bin/python" -c 'import json,sys; n,k=map(int,sys.argv[1:]); assert n>0 and k>=0; print(json.dumps(sorted(set(range(1,n+1)) | {i*(k+1) for i in range(1,n+1)})))' "${FLASHNEXT_SEQUENCES:-8}" "${FLASHNEXT_MTP:-0}")}
exec "$RUNTIME/bin/vllm" serve "$MODEL" \
  --served-model-name flashnext --host 127.0.0.1 --port "${PORT:-8000}" \
  --max-model-len "${FLASHNEXT_CONTEXT:-212992}" \
  --max-num-seqs "${FLASHNEXT_SEQUENCES:-8}" \
  --max-num-batched-tokens "${FLASHNEXT_PREFILL:-2048}" \
  --enable-chunked-prefill \
  --gpu-memory-utilization "${FLASHNEXT_MEMORY_FRACTION:-0.80}" \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --compilation-config "{\"cudagraph_mode\":\"${FLASHNEXT_CUDAGRAPH_MODE:-PIECEWISE}\",\"cudagraph_capture_sizes\":${CAPTURE_SIZES}}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  "${EXTRA[@]}" "$@"
