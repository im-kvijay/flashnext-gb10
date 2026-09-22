"""File-backed PLE lookup, overlapping Grace CPU gather with decoder work.

Preserves upstream hashing, table layout, FP8 scale, dequantization, and request
state. CUDA graph breaks use upstream's supported eager-break mechanism. The
table is never pinned or exposed to a GPU kernel; only gathered rows are pinned.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, eager_break_during_capture

from .mapped_table import MappedTable

PREFORWARD = os.environ.get('FLASHNEXT_PLE_PREFORWARD') == '1'


def gather_bytes(weight, ids, start, end):
    """Gather storage bytes, including duplicates and zero-masked padding IDs."""
    flat = ids.reshape(-1).long()
    valid = (flat >= start) & (flat < end)
    local = (flat - start).clamp(0, weight.shape[0] - 1)
    # CPU index_select does not implement every FP8 dtype. A byte view is exact.
    raw = weight.view(torch.uint8)
    selected = torch.index_select(raw, 0, local)
    selected[~valid] = 0
    return selected.reshape(*ids.shape, raw.shape[-1])


def make_nvme_embedding(upstream, directory):
    class NVMeEmbedding(upstream.Qwen4ExpPLEPinnedHostEmbedding):
        _flashnext_nvme = True

        def __init__(self, num_embeddings, embedding_dim, *, params_dtype,
                     padding_size, prefix, embedding_method, num_ngram_heads=1,
                     max_total_tokens=0, data_parallel_rank=0):
            # Skip the UVA subclass initializer, which would pin the entire table.
            upstream.Qwen4ExpPLEEmbedding.__init__(
                self, num_embeddings, embedding_dim, params_dtype=params_dtype,
                padding_size=padding_size, prefix=prefix,
                embedding_method=embedding_method, num_ngram_heads=num_ngram_heads,
                max_total_tokens=max_total_tokens, data_parallel_rank=data_parallel_rank,
            )
            if self.tp_size != 1 or self.etp_data_parallel_size != 1:
                raise ValueError("NVMe backend currently requires a single GB10 (TP=DP=1)")
            self._device = torch.device("cuda", torch.cuda.current_device())
            self._prefetch_buffer = torch.empty(
                max_total_tokens, num_ngram_heads, self.embedding_dim,
                dtype=self.weight.dtype, device=self._device,
            )
            self._output_dim = num_ngram_heads * self.embedding_dim
            # cudaHostAlloc during a graph capture invalidates that capture,
            # including allocations made by a background CPU worker. Allocate
            # the bounded staging area once, before any capture can start.
            self._host_staging = torch.empty(
                max_total_tokens, num_ngram_heads, self.embedding_dim,
                dtype=self.weight.dtype, device="cpu", pin_memory=True,
            )
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-nvme")
            self._pending = None
            self._staging = None
            self._copy_done = torch.cuda.Event()
            self._copy_recorded = False
            self._async_ids = os.environ.get('FLASHNEXT_ASYNC_PLE') == '1'
            if self._async_ids or PREFORWARD:
                self._host_ids = torch.empty(
                    max_total_tokens, num_ngram_heads, dtype=torch.int64,
                    device='cpu', pin_memory=True,
                )
                self._ids_ready = torch.cuda.Event()
            upstream.logger.info("FlashNext NVMe PLE: %.2f GiB file-backed; weight pinned=%s",
                                 self.weight.numel() * self.weight.element_size() / 1024**3,
                                 self.weight.is_pinned())

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            if os.environ.get('FLASHNEXT_PLE_DIRECT') == '1':
                from .checkpoint_shards import CheckpointShards
                self._checkpoint_shards=CheckpointShards(embedding_dim,dtype,directory)
                # Only shape/dtype metadata is needed here. Lookup reads retained
                # checkpoint tensors, never this one-element expanded placeholder.
                return torch.empty(1,dtype=dtype,device='cpu').expand(num_embeddings,embedding_dim)
            size = num_embeddings * embedding_dim * torch.empty((), dtype=dtype, device="cpu").element_size()
            self._mapped_table = MappedTable(directory, size)
            return torch.frombuffer(self._mapped_table.mapping, dtype=dtype).reshape(num_embeddings, embedding_dim)

        def weight_loader(self, param, loaded_weight, checkpoint_start=None):
            if hasattr(self,'_checkpoint_shards') and param is self.weight:
                self._checkpoint_shards.add(checkpoint_start or 0,loaded_weight)
                return
            upstream.Qwen4ExpPLEEmbedding.weight_loader(self, param, loaded_weight, checkpoint_start)
            if param.device.type == "cpu" and param.data_ptr() == self.weight.data_ptr():
                row_bytes = self.embedding_dim * param.element_size()
                start = checkpoint_start or 0
                self._mapped_table.flush_rows(start * row_bytes, loaded_weight.shape[0] * row_bytes)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if hasattr(self,'_checkpoint_shards'):
                raise RuntimeError('Direct PLE cannot export placeholder weights; retain the original verified checkpoint')
            return super()._save_to_state_dict(destination,prefix,keep_vars)

        def _gather(self, ids):
            if hasattr(self,'_checkpoint_shards'):
                staging=self._host_staging[:ids.shape[0]].view(torch.uint8)
                self._checkpoint_shards.gather_into(ids,staging,self.shard_indices.org_vocab_end_index)
                return staging.view(self.weight.dtype)
            rows = gather_bytes(self.weight, ids,
                                self.shard_indices.org_vocab_start_index,
                                self.shard_indices.org_vocab_end_index)
            staging = self._host_staging[:ids.shape[0]].view(torch.uint8)
            staging.copy_(rows)
            return staging.view(self.weight.dtype)

        @eager_break_during_capture
        def start_prefetch(self, hidden_states, ngram_ids):
            if self._pending is not None:
                raise RuntimeError("PLE prefetch overwritten before consumption")
            if self._copy_recorded:
                self._copy_done.synchronize()
                self._staging = None
            if self._async_ids:
                if (ngram_ids.dtype != torch.int64 or ngram_ids.ndim != 2
                        or ngram_ids.shape[1] != self._host_ids.shape[1]
                        or ngram_ids.shape[0] > self._host_ids.shape[0]):
                    raise ValueError('Unexpected n-gram IDs for asynchronous PLE')
                ids = self._host_ids[:ngram_ids.shape[0]]
                ids.copy_(ngram_ids, non_blocking=True)
                self._ids_ready.record(torch.cuda.current_stream(self._device))
                if BreakableCUDAGraphCapture.is_active():
                    # A CUDA call from the worker while the next segment is being
                    # captured invalidates the capture (global capture mode). Wait
                    # here, inside the eager break, and give the worker CPU-only work.
                    self._ids_ready.synchronize()
                    self._pending = self._pool.submit(self._gather, ids)
                else:
                    self._pending = self._pool.submit(self._gather_after_ids, ids)
            else:
                ids = ngram_ids.to(device="cpu")
                self._pending = self._pool.submit(self._gather, ids)

        def _gather_after_ids(self, ids):
            # Wait off the model thread. The decoder can enqueue its first layer
            # while this stream-ordered D2H copy and the CPU gather complete.
            self._ids_ready.synchronize()
            return self._gather(ids)

        @eager_break_during_capture
        def _finalize_prefetch(self, prefetch_output, output):
            if self._pending is None:
                raise RuntimeError("PLE output requested without this step's prefetch")
            staging = self._pending.result()
            self._pending = None
            if staging.shape[0] != output.shape[0]:
                raise RuntimeError("PLE prefetch/output token count mismatch")
            self._staging = staging
            output.copy_(staging.flatten(-2), non_blocking=True)
            self._copy_done.record(torch.cuda.current_stream(self._device))
            self._copy_recorded = True

        def stage_before_forward(self, ngram_ids):
            """Pre-forward mode: gather this step's rows into the persistent device buffer.

            Runs outside the model forward and outside any graph, so the whole
            decode step can be one FULL CUDA graph that only reads the buffer.
            """
            count = ngram_ids.shape[0]
            if ngram_ids.shape[1] != self._host_ids.shape[1] or count > self._host_ids.shape[0]:
                raise ValueError('Unexpected n-gram IDs for pre-forward PLE')
            if self._copy_recorded:
                # The previous H2D copy must finish before its staging rows are reused.
                self._copy_done.synchronize()
            ids = self._host_ids[:count]
            ids.copy_(ngram_ids, non_blocking=True)
            stream = torch.cuda.current_stream(self._device)
            self._ids_ready.record(stream)
            self._ids_ready.synchronize()
            staging = self._gather(ids)
            self._prefetch_buffer[:count].copy_(staging, non_blocking=True)
            self._copy_done.record(stream)
            self._copy_recorded = True

    if PREFORWARD:
        # No eager breaks: the forward only copies from the staged device buffer.
        def start_prefetch(self, hidden_states, ngram_ids):
            return

        def _finalize_prefetch(self, prefetch_output, output):
            output.copy_(self._prefetch_buffer[:output.shape[0]].flatten(-2))

        NVMeEmbedding.start_prefetch = start_prefetch
        NVMeEmbedding._finalize_prefetch = _finalize_prefetch

    return NVMeEmbedding


def register_preforward(ngram_embedding, model_state):
    """Stage PLE rows while preparing inputs, before graph replay or eager forward."""
    NGram = ngram_embedding.Qwen4ExpNGramEmbedding
    original_start = NGram.start_prefetch

    def start_prefetch(self, hidden_states, input_ids, query_start_loc, ngram_context):
        if getattr(self.ngram_embedding, '_flashnext_nvme', False):
            return  # IDs were computed and rows staged before the forward.
        return original_start(self, hidden_states, input_ids, query_start_loc, ngram_context)

    NGram.start_prefetch = start_prefetch
    State = model_state.Qwen4ExpModelState
    original_prepare = State.prepare_inputs

    def prepare_inputs(self, input_batch, req_states):
        inputs = original_prepare(self, input_batch, req_states)
        if not self.uses_ngram_embedding:
            return inputs
        modules = getattr(self, '_flashnext_ple_modules', None)
        if modules is None:
            modules = [m for m in self.model.modules()
                       if isinstance(m, NGram) and getattr(m.ngram_embedding, '_flashnext_nvme', False)]
            self._flashnext_ple_modules = modules
        input_ids = input_batch.input_ids
        for module in modules:
            ids = module.compute_ngram_ids(input_ids, inputs['query_start_loc'], inputs['ngram_context'])
            module.ngram_embedding.stage_before_forward(ids)
        return inputs

    State.prepare_inputs = prepare_inputs
