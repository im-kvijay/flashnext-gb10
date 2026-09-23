"""GDN speculative decode with one recurrent-state write per step (FLASHNEXT_GDN_REPLAY=1).

vLLM's fused MTP decode kernel writes the recurrent state after every
verified position (bonus + drafts) to its own slot, so the next step can
start from whichever position was accepted. With three drafts that is four
state writes per layer per request, most of them for rejected positions.

Here each request's first slot keeps the state at the start of the step, and
the per-token inputs of the update (normalized key, value, decay, beta) go to
a small replay record in the second slot's otherwise unused memory. The next
step, once acceptance is known, replays the accepted tokens on top of that
state, rounds to BF16 exactly where vLLM stores it, writes the result back to
the first slot once and continues with its new tokens. State traffic drops
from one read and four writes to one read and one write per step. The
arithmetic per token is the same as vLLM's kernel (FP32 in registers,
BF16 state between steps), so outputs match to FP32 rounding.

A replay record is used only when its fingerprint (two checksums of the BF16
state it was written against, plus the slot number) matches the first slot.
Any other writer of that slot (prefill, vLLM's non-fused paths) changes the
state and so invalidates the record; the kernel then starts from vLLM's
usual slot for the accepted position. Before any non-fused path runs, rows
with a live record are materialized: the current state is written both to
the first slot and to the accepted position's slot, which is where vLLM's
other kernels look.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MAGIC = 0x46_4E_52_50  # 'FNRP'
MAX_TOKENS = 8


@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, libdevice.log1p(tl.exp(x)))


@triton.jit
def _fingerprint(bits, offsets):
    """Two checksums of a [V, K] tile of BF16 bit patterns (as int32)."""
    h1 = tl.sum(tl.sum(bits, 1), 0)
    h2 = tl.sum(tl.sum(bits.to(tl.int64) * ((offsets % 8191) + 1).to(tl.int64), 1), 0)
    return h1, h2


@triton.jit
def _replay(h, rec, count, K: tl.constexpr, V: tl.constexpr, MAXT: tl.constexpr):
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    for t in range(count):
        k = tl.load(rec + t * K + offs_k)
        v = tl.load(rec + MAXT * K + t * V + offs_v)
        decay = tl.load(rec + MAXT * (K + V) + t)
        beta = tl.load(rec + MAXT * (K + V) + MAXT + t)
        h = h * decay
        hk = tl.sum(h * k[None, :], 1)
        delta = (v - hk) * beta
        h = h + delta[:, None] * k[None, :]
    return h


@triton.jit
def _gdn_replay_decode_kernel(
        qkv, qkv_row, a, a_row, b, b_row, a_log, dt_bias,
        indices, indices_row, cu_seqlens, accepted_ptr,
        state, slot_stride, gate, gate_row, norm_weight, out,
        H, HV, scale, eps,
        K: tl.constexpr, V: tl.constexpr, RATIO: tl.constexpr, MAXT: tl.constexpr,
        SIGMOID: tl.constexpr, MAGIC_: tl.constexpr):
    req = tl.program_id(0)
    vh = tl.program_id(1)
    bos = tl.load(cu_seqlens + req)
    n = tl.load(cu_seqlens + req + 1) - bos
    if n <= 0:
        return
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    slot0 = tl.load(indices + req * indices_row)
    slot1 = tl.load(indices + req * indices_row + 1)
    accepted = tl.load(accepted_ptr + req)
    head = vh * V * K
    tile = offs_v[:, None] * K + offs_k[None, :]
    base0 = state + slot0.to(tl.int64) * slot_stride + head
    rbase = state + slot1.to(tl.int64) * slot_stride + head
    rec = rbase.to(tl.pointer_type(tl.float32))
    meta = rbase.to(tl.pointer_type(tl.int32)) + MAXT * (K + V + 2)

    raw = tl.load(base0 + tile, mask=(slot0 > 0) & (tile >= 0), other=0.0)
    h1, h2 = _fingerprint(raw.to(tl.int16, bitcast=True).to(tl.int32), tile)
    ok = (slot0 > 0) & (slot1 > 0)
    magic = tl.load(meta, mask=ok, other=0)
    m1 = tl.load(meta + 1, mask=ok, other=0)
    m2 = tl.load(meta + 2, mask=ok, other=0).to(tl.int64) & 0xFFFFFFFF
    m2 = m2 | (tl.load(meta + 3, mask=ok, other=0).to(tl.int64) << 32)
    count = tl.load(meta + 4, mask=ok, other=0)
    mslot = tl.load(meta + 5, mask=ok, other=-1)
    valid = ok & (magic == MAGIC_) & (m1 == h1) & (m2 == h2) & (mslot == slot0) & (count >= 1)

    # Without a live record, start where vLLM's kernel would: the accepted position's slot.
    in_range = (accepted >= 1) & (accepted <= indices_row)
    src = tl.load(indices + req * indices_row + accepted - 1, mask=in_range, other=0)
    src = tl.where(valid, slot0, src)
    if (src <= 0) | (n > MAXT) | (slot1 <= 0):
        for t in range(n):
            tl.store(out + ((bos + t) * HV + vh) * V + offs_v, tl.zeros((V,), tl.float32).to(out.dtype.element_ty))
        return
    if src != slot0:
        raw = tl.load(state + src.to(tl.int64) * slot_stride + head + tile)
    h = raw.to(tl.float32)
    replayed = tl.where(valid, tl.minimum(accepted, count), 0)
    h = _replay(h, rec, replayed, K, V, MAXT)
    stored = h.to(tl.bfloat16)
    h = stored.to(tl.float32)
    if (replayed > 0) | (src != slot0):
        tl.store(base0 + tile, stored)
        h1, h2 = _fingerprint(stored.to(tl.int16, bitcast=True).to(tl.int32), tile)
    tl.debug_barrier()  # the record is read before it is overwritten below

    kh = vh // RATIO
    a_log_v = tl.load(a_log + vh).to(tl.float32)
    dt = tl.load(dt_bias + vh).to(tl.float32)
    weight = tl.load(norm_weight + offs_v).to(tl.float32)
    for t in range(n):
        token = (bos + t).to(tl.int64)
        row = qkv + token * qkv_row
        q = tl.load(row + kh * K + offs_k).to(tl.float32)
        k = tl.load(row + H * K + kh * K + offs_k).to(tl.float32)
        v = tl.load(row + 2 * H * K + vh * V + offs_v).to(tl.float32)
        q = q * (tl.math.rsqrt(tl.sum(q * q, 0) + 1e-6) * scale)
        k = k * tl.math.rsqrt(tl.sum(k * k, 0) + 1e-6)
        a_v = tl.load(a + token * a_row + vh).to(tl.float32)
        b_v = tl.load(b + token * b_row + vh).to(tl.float32)
        decay = tl.exp(-tl.exp(a_log_v) * _softplus(a_v + dt))
        beta = 1.0 / (1.0 + tl.exp(-b_v))
        h = h * decay
        hk = tl.sum(h * k[None, :], 1)
        delta = (v - hk) * beta
        h = h + delta[:, None] * k[None, :]
        o = tl.sum(h * q[None, :], 1).to(tl.bfloat16).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(o * o, 0) / V + eps)
        z = tl.load(gate + token * gate_row + vh * V + offs_v).to(tl.float32)
        g = 1.0 / (1.0 + tl.exp(-z))
        if not SIGMOID:
            g = z * g
        tl.store(out + (token * HV + vh) * V + offs_v, (o * rstd * weight * g).to(out.dtype.element_ty))
        tl.store(rec + t * K + offs_k, k)
        tl.store(rec + MAXT * K + t * V + offs_v, v)
        tl.store(rec + MAXT * (K + V) + t, decay)
        tl.store(rec + MAXT * (K + V) + MAXT + t, beta)
    tl.store(meta + 1, h1)
    tl.store(meta + 2, h2.to(tl.int32))  # low word
    tl.store(meta + 3, (h2 >> 32).to(tl.int32))
    tl.store(meta + 4, n)
    tl.store(meta + 5, slot0)
    tl.store(meta, MAGIC_)


@triton.jit
def _gdn_materialize_kernel(indices, indices_row, accepted_ptr, state, slot_stride,
                            K: tl.constexpr, V: tl.constexpr, MAXT: tl.constexpr, MAGIC_: tl.constexpr):
    req = tl.program_id(0)
    vh = tl.program_id(1)
    slot0 = tl.load(indices + req * indices_row)
    slot1 = tl.load(indices + req * indices_row + 1)
    if (slot0 <= 0) | (slot1 <= 0):
        return
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    head = vh * V * K
    tile = offs_v[:, None] * K + offs_k[None, :]
    rbase = state + slot1.to(tl.int64) * slot_stride + head
    meta = rbase.to(tl.pointer_type(tl.int32)) + MAXT * (K + V + 2)
    if tl.load(meta) != MAGIC_:
        return
    base0 = state + slot0.to(tl.int64) * slot_stride + head
    raw = tl.load(base0 + tile)
    h1, h2 = _fingerprint(raw.to(tl.int16, bitcast=True).to(tl.int32), tile)
    m2 = tl.load(meta + 2).to(tl.int64) & 0xFFFFFFFF
    m2 = m2 | (tl.load(meta + 3).to(tl.int64) << 32)
    count = tl.load(meta + 4)
    valid = (tl.load(meta + 1) == h1) & (m2 == h2) & (tl.load(meta + 5) == slot0) & (count >= 1)
    if not valid:
        return
    accepted = tl.load(accepted_ptr + req)
    accepted = tl.maximum(tl.minimum(accepted, count), 1)
    h = _replay(raw.to(tl.float32), rbase.to(tl.pointer_type(tl.float32)), accepted, K, V, MAXT)
    stored = h.to(tl.bfloat16)
    tl.debug_barrier()
    tl.store(base0 + tile, stored)
    dst = tl.load(indices + req * indices_row + accepted - 1, mask=accepted <= indices_row, other=0)
    if (dst > 0) & (dst != slot0):
        tl.store(state + dst.to(tl.int64) * slot_stride + head + tile, stored)
    if dst != slot1:
        tl.store(meta, 0)


def replay_decode(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
                  state, output_gate, norm_weight, out, num_k_heads, scale, norm_eps, sigmoid_gate):
    num_requests = state_indices.shape[0]
    HV, V, K = state.shape[1], state.shape[2], state.shape[3]
    _gdn_replay_decode_kernel[(num_requests, HV)](
        mixed_qkv, mixed_qkv.stride(0), a, a.stride(0), b, b.stride(0), A_log, dt_bias,
        state_indices, state_indices.stride(0), cu_seqlens, num_accepted_tokens,
        state, state.stride(0), output_gate, output_gate.stride(0), norm_weight, out,
        num_k_heads, HV, scale, norm_eps,
        K=K, V=V, RATIO=HV // num_k_heads, MAXT=MAX_TOKENS, SIGMOID=sigmoid_gate, MAGIC_=MAGIC,
        num_warps=8, num_stages=1)


def materialize(rows, accepted, state):
    if rows is None or rows.shape[0] == 0:
        return
    HV, V, K = state.shape[1], state.shape[2], state.shape[3]
    _gdn_materialize_kernel[(rows.shape[0], HV)](
        rows, rows.stride(0), accepted, state, state.stride(0),
        K=K, V=V, MAXT=MAX_TOKENS, MAGIC_=MAGIC, num_warps=8, num_stages=1)


def supported(state, state_indices, mixed_qkv):
    return (state.dtype == torch.bfloat16 and state.dim() == 4 and state.shape[2] == 128
            and state.shape[3] == 128 and state.stride(3) == 1 and state.stride(2) == 128
            and state.stride(1) == 128 * 128 and state_indices.shape[1] >= 2
            and state_indices.shape[1] <= MAX_TOKENS and mixed_qkv.stride(-1) == 1)


def register_gdn_replay():
    from vllm.logger import init_logger
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    logger = init_logger('vllm.flashnext.gdn_replay')
    Layer = gdn.QwenGatedDeltaNetAttention if hasattr(gdn, 'QwenGatedDeltaNetAttention') else None
    if Layer is None:
        Layer = next(v for v in vars(gdn).values() if isinstance(v, type)
                     and hasattr(v, '_forward_core_decode_spec_post_conv_fused_norm'))
    if getattr(Layer, '_flashnext_gdn_replay', False):
        return
    Builder = gdn_attn.GDNAttentionMetadataBuilder
    original_build = Builder.build

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        metadata = original_build(self, common_prefix_len, common_attn_metadata,
                                  num_accepted_tokens=num_accepted_tokens,
                                  num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                                  fast_build=fast_build)
        metadata.flashnext_rows = None
        metadata.flashnext_accepted = None
        if not self.use_spec_decode or num_accepted_tokens is None:
            return metadata
        m = common_attn_metadata
        width = self.num_spec + 1
        rows = getattr(self, '_flashnext_rows', None)
        if rows is None:
            rows = self._flashnext_rows = torch.zeros(self.vllm_config.scheduler_config.max_num_seqs * 2, width,
                                                      dtype=torch.int32, device=m.query_start_loc.device)
            self._flashnext_accepted = torch.ones(rows.shape[0], dtype=torch.int32, device=rows.device)
        n = min(m.num_reqs, num_accepted_tokens.shape[0], rows.shape[0])
        table = mamba_get_block_table_tensor(m.block_table_tensor, m.seq_lens, self.kv_cache_spec,
                                             self.vllm_config.cache_config.mamba_cache_mode)
        rows[:n].copy_(table[:n, :width], non_blocking=True)
        self._flashnext_accepted[:n].copy_(num_accepted_tokens[:n], non_blocking=True)
        metadata.flashnext_rows = rows[:n]
        metadata.flashnext_accepted = self._flashnext_accepted[:n]
        return metadata

    Builder.build = build

    original_core = Layer._forward_core

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        # Any path other than the replay kernel reads vLLM's slots: bring them up to date first.
        metadata = gdn.get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata.get(self.prefix)
        if metadata is not None and getattr(metadata, 'flashnext_rows', None) is not None:
            state = self.kv_cache[1]
            if state.dtype == torch.bfloat16:
                materialize(metadata.flashnext_rows, metadata.flashnext_accepted, state)
        return original_core(self, mixed_qkv, b, a, core_attn_out)

    original_post_conv = Layer._forward_core_decode_spec_post_conv_fused_norm

    def _forward_core_decode_spec_post_conv_fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out,
                                                       attn_metadata):
        state_indices = attn_metadata.spec_state_indices_tensor
        state = self.kv_cache[1]
        if not supported(state, state_indices, mixed_qkv):
            return original_post_conv(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata)
        num_requests = attn_metadata.num_spec_decodes
        replay_decode(mixed_qkv, a, b, self.A_log, self.dt_bias, state_indices[:num_requests],
                      attn_metadata.spec_query_start_loc[:num_requests + 1],
                      attn_metadata.num_accepted_tokens[:num_requests], state, output_gate,
                      self.norm.weight, core_attn_out, self.num_k_heads // self.tp_size,
                      self.head_k_dim ** -0.5, self.layer_norm_epsilon, self.norm.activation == 'sigmoid')

    Layer._forward_core = _forward_core
    Layer._forward_core_decode_spec_post_conv_fused_norm = _forward_core_decode_spec_post_conv_fused_norm
    Layer._flashnext_gdn_replay = True
    logger.info('FlashNext GDN replay decode enabled: one recurrent-state write per step')
