"""Launch parameters for the QSA indexer's decode scoring kernel on GB10.

vLLM launches `_qsa_mqa_paged_uniform_kernel` with 64-column tiles and one or
two warps, tuned on GB300. FLASHNEXT_QSA_DECODE_CONFIG=BLOCK_N,WARPS,STAGES,TILES
(from scripts/tune_qsa_indexer.py) replaces those for decode batches. Scores
and the top-k selection are computed exactly as before.
"""
import os

import torch


def parse(text):
    block_n, warps, stages, tiles = (int(v) for v in text.split(','))
    return block_n, warps, stages, tiles


def select_paged_decode(module, config, q, k_cache, page_table, visible_blocks, token_topk, compress_ratio,
                        decode_query_len, block_indices):
    from vllm.triton_utils import triton
    block_n, warps, stages, tiles = config
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert q.dtype == k_cache.dtype
    num_requests = q.shape[0] // decode_query_len
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    grid = (num_requests, triton.cdiv(columns, block_n * tiles))
    module._qsa_mqa_paged_uniform_kernel[grid](
        q, k_cache, page_table, visible_blocks, logits,
        *q.stride()[:-1], *k_cache.stride()[:2], *page_table.stride()[:-1], *logits.stride()[:-1],
        PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=page_table.shape[1], NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2], DECODE_QUERY_LEN=decode_query_len, BLOCK_N=block_n, TILES_PER_PROG=tiles,
        STAGES=stages, num_warps=warps)
    module._topk(logits, visible_blocks, token_topk, compress_ratio, block_indices,
                 torch.empty((module._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device))


def register_qsa_decode_config():
    from vllm.logger import init_logger
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    config = parse(os.environ['FLASHNEXT_QSA_DECODE_CONFIG'])
    if getattr(qsa_indexer, '_flashnext_decode_config', None) == config:
        return

    def qsa_select_paged_decode(*args, **kwargs):
        return select_paged_decode(qsa_indexer, config, *args, **kwargs)

    qsa_indexer.qsa_select_paged_decode = qsa_select_paged_decode
    qsa_indexer._flashnext_decode_config = config
    init_logger('vllm.flashnext.qsa').info('FlashNext QSA decode scoring config %s', config)
