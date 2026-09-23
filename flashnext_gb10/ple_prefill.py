"""Read prompt PLE rows ahead of chunked prefill (FLASHNEXT_PLE_PREFILL_AHEAD=<tokens>).

A prompt's tokens are all known when its request arrives, but each prefill
chunk's PLE rows are read only when that chunk is staged, so the GPU waits on
them (28% of 200k priming time on the rented host's disk). Here each step's
schedule says where every prefilling request will resume; a helper thread
computes the n-gram IDs of the next <tokens> prompt tokens on the CPU (vLLM's
reference path, same IDs) and reads their rows into the PLE row cache while
the GPU computes. Staging is unchanged and finds the rows cached, so outputs
are identical. The model thread only records (request, start, end) windows.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import types

import torch


def register_ple_prefill():
    from vllm.logger import init_logger
    from vllm.v1.worker.gpu import model_runner as runner_module

    logger = init_logger('vllm.flashnext.ple')
    Runner = runner_module.GPUModelRunner
    if getattr(Runner, '_flashnext_ple_prefill', False):
        return
    ahead = int(os.environ.get('FLASHNEXT_PLE_PREFILL_AHEAD', '8192'))
    original = Runner.execute_model

    class Ahead:
        def __init__(self, module):
            # CPU copies of the hash parameters: the helper thread must not touch CUDA.
            cls = type(module)
            self.ids = types.SimpleNamespace(
                layer_multipliers=module.layer_multipliers.cpu(),
                ngram_heads_vocab_sizes=module.ngram_heads_vocab_sizes.cpu(),
                ngram_heads_offsets=module.ngram_heads_offsets.cpu(),
                eos_token_id=module.eos_token_id, heads_per_ngram=module.heads_per_ngram,
                ngram_size=module.ngram_size, _shift_precompute=cls._shift_precompute,
                _shift_apply=cls._shift_apply)
            self.ids.compute = cls.compute_ngram_ids.__get__(self.ids)
            self.shards = module.ngram_embedding._checkpoint_shards
            self.eos = module.eos_token_id
            self.prompts = {}  # req_id -> [token list, prefetched up to]
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ple-prefill')
            self.lock = threading.Lock()
            self.windows = self.tokens = 0

        def schedule(self, out):
            for req_id in out.finished_req_ids:
                self.prompts.pop(req_id, None)
            starts = {}
            for new in out.scheduled_new_reqs:
                tokens = new.prefill_token_ids or new.prompt_token_ids
                if tokens:
                    self.prompts[new.req_id] = [tokens, 0]
                    starts[new.req_id] = new.num_computed_tokens
            cached = out.scheduled_cached_reqs
            for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
                starts[req_id] = computed
            for req_id, computed in starts.items():
                entry = self.prompts.get(req_id)
                if entry is None:
                    continue
                tokens, done = entry
                begin = max(done, computed + out.num_scheduled_tokens.get(req_id, 0))
                end = min(len(tokens), computed + out.num_scheduled_tokens.get(req_id, 0) + ahead)
                if begin >= len(tokens):
                    self.prompts.pop(req_id, None)  # prompt fully staged; decode rows use FLASHNEXT_PLE_EARLY
                    continue
                if end > begin:
                    entry[1] = end
                    self.pool.submit(self.read, tokens, begin, end)

        def read(self, tokens, begin, end):
            try:
                ids = torch.tensor(tokens[begin:end], dtype=torch.int64)
                context = torch.tensor([[tokens[i] if i >= 0 else self.eos for i in (begin - 2, begin - 1)]],
                                       dtype=torch.int64)
                loc = torch.tensor([0, end - begin], dtype=torch.int64)
                rows = self.ids.compute(ids, loc, context)
                self.shards.prefetch(rows.reshape(-1, rows.shape[-1]))
                with self.lock:
                    self.windows += 1
                    self.tokens += end - begin
                    if self.windows % 200 == 0:
                        logger.info('FlashNext prefill PLE read-ahead: %d windows, %d tokens', self.windows, self.tokens)
            except Exception:
                logger.exception('FlashNext prefill PLE read-ahead failed for one window')

    def execute_model(self, scheduler_output, *args, **kwargs):
        state = getattr(self, '_flashnext_prefill_ahead', None)
        if state is None:
            modules = getattr(getattr(self, 'model_state', None), '_flashnext_ple_modules', None)
            if modules:
                shards = getattr(modules[0].ngram_embedding, '_checkpoint_shards', None)
                state = self._flashnext_prefill_ahead = Ahead(modules[0]) if len(modules) == 1 and shards else False
                if state:
                    logger.info('FlashNext prefill PLE read-ahead enabled: %d tokens', ahead)
        if state and not kwargs.get('dummy_run'):
            try:
                state.schedule(scheduler_output)
            except Exception:
                logger.exception('FlashNext prefill PLE read-ahead disabled')
                self._flashnext_prefill_ahead = False
        return original(self, scheduler_output, *args, **kwargs)

    Runner.execute_model = execute_model
    Runner._flashnext_ple_prefill = True
