"""Start the next step's PLE row reads while the drafter runs (FLASHNEXT_PLE_EARLY=1).

After a step's tokens are sampled, each request's next input begins with its
bonus token, whose n-gram context (the two tokens before it) is already in the
request state. Its n-gram IDs are computed exactly as the next step will
compute them and copied to pinned memory; a helper thread reads those rows
into the PLE row cache while the MTP drafter proposes. The next step's lookup
is unchanged and simply finds the rows cached, so outputs are identical.

The helper makes no CUDA calls (a CUDA call from another thread can invalidate
a graph capture): the GPU copies the IDs and then a step counter to pinned
memory, and the helper polls the counter.
"""
from concurrent.futures import ThreadPoolExecutor
import time

import torch


def register_ple_early():
    from vllm.v1.worker.gpu import model_runner as runner_module
    from vllm.logger import init_logger

    logger = init_logger('vllm.flashnext.ple')
    Runner = runner_module.GPUModelRunner
    if getattr(Runner, '_flashnext_ple_early', False):
        return
    original = Runner.postprocess_sampled

    class Early:
        def __init__(self, runner, module, max_reqs):
            self.module = module
            self.shards = module.ngram_embedding._checkpoint_shards
            heads = module.ngram_embedding._host_ids.shape[1]
            device = runner.device
            self.ids = torch.empty(max_reqs, heads, dtype=torch.int64, pin_memory=True)
            self.valid = torch.empty(max_reqs, dtype=torch.bool, pin_memory=True)
            self.seq_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            self.seq_gpu = torch.zeros(1, dtype=torch.int64, device=device)
            self.positions = torch.arange(max_reqs + 1, dtype=torch.int32, device=device)
            self.offsets = torch.tensor([-2, -1], dtype=torch.int64, device=device)
            self.seq = 0
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ple-early')
            self.pending = None
            self.issued = self.skipped = 0

        def issue(self, runner, idx_mapping):
            if self.pending is not None and not self.pending.done():
                self.skipped += 1  # previous prefetch still reading; never block the model thread
                return
            num = idx_mapping.shape[0]
            if num == 0 or num > self.ids.shape[0]:
                return
            states = runner.req_states
            idx = idx_mapping.long()
            valid = idx >= 0
            idx = idx.clamp_min(0)
            n = states.num_computed_tokens.gpu[idx].long()
            positions = n.unsqueeze(1) + self.offsets
            context = states.all_token_ids.gpu[idx.unsqueeze(1), positions.clamp_min(0)]
            eos = self.module.eos_token_id
            context = torch.where(positions >= 0, context, context.new_full((), eos))
            tokens = states.last_sampled_tokens[idx, 0].to(torch.int32)  # input IDs are int32
            ids = self.module.compute_ngram_ids(tokens, self.positions[:num + 1], context)
            self.ids[:num].copy_(ids, non_blocking=True)
            self.valid[:num].copy_(valid, non_blocking=True)
            self.seq += 1
            self.seq_gpu.fill_(self.seq)
            self.seq_host.copy_(self.seq_gpu, non_blocking=True)
            self.pending = self.pool.submit(self.read, self.seq, num)
            self.issued += 1
            if self.issued % 2000 == 0:
                logger.info('FlashNext early PLE prefetch: %d issued, %d skipped', self.issued, self.skipped)

        def read(self, seq, num):
            deadline = time.monotonic() + 2.0
            while int(self.seq_host[0]) < seq:
                if time.monotonic() > deadline:
                    return
                time.sleep(0.0001)
            ids = self.ids[:num][self.valid[:num]]
            if ids.numel():
                self.shards.prefetch(ids)

    def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected,
                            query_start_loc=None):
        original(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)
        early = getattr(self, '_flashnext_early', None)
        if early is None:
            modules = getattr(self.model_state, '_flashnext_ple_modules', None)
            if not modules or torch.cuda.is_current_stream_capturing():
                return  # PLE modules are discovered on the first prepared step
            shards = getattr(modules[0].ngram_embedding, '_checkpoint_shards', None)
            if len(modules) != 1 or shards is None:
                self._flashnext_early = False
                return
            try:
                early = self._flashnext_early = Early(self, modules[0], self.max_num_reqs)
            except Exception:
                logger.exception('FlashNext early PLE prefetch unavailable')
                self._flashnext_early = False
                return
            logger.info('FlashNext early PLE prefetch enabled')
        if early and not torch.cuda.is_current_stream_capturing():
            try:
                early.issue(self, idx_mapping)
            except Exception:
                # A prefetch only warms the cache; never let it stop serving.
                logger.exception('FlashNext early PLE prefetch disabled')
                self._flashnext_early = False

    Runner.postprocess_sampled = postprocess_sampled
    Runner._flashnext_ple_early = True
