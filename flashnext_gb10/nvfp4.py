"""ModelOpt NVFP4 weight packing (E2M1 values, FP8 E4M3 per-16 block scales, FP32 global scale).

Convention (matches the NVIDIA checkpoint's routed experts):
  weight_scale_2 = global_amax / (6 * 448)
  weight_scale   = fp8(block_amax / 6 / weight_scale_2)
  value          = e2m1(w / (weight_scale * weight_scale_2)), two values per byte,
                   element 2i in the low nibble and 2i+1 in the high nibble.
"""
import torch

BLOCK = 16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8_MAX = 448.0


def global_scale(amax):
    return (amax.float() / (6.0 * FP8_MAX)).clamp(min=1e-12)


def _encode(blocks, scale, scale_2):
    effective = scale.float() * scale_2
    scaled = blocks / effective.clamp(min=1e-30).unsqueeze(-1)
    scaled = torch.where(effective.unsqueeze(-1) > 0, scaled, torch.zeros_like(scaled))
    codes = nearest_e2m1(scaled)
    grid = E2M1.to(blocks.device)
    decoded = grid[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0) * effective.unsqueeze(-1)
    return codes, ((decoded - blocks) ** 2).sum(dim=-1)


# Candidate block-scale multipliers for the error search; 1.0 is plain absmax scaling.
SEARCH = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 1.04)


def quantize(weight, scale_2=None, search=False, rows=2048):
    """Return (packed uint8 [out, in/2], weight_scale fp8 [out, in/16], weight_scale_2 fp32 scalar).

    With search=True each 16-element block takes the FP8 scale, among absmax scaling and
    a few clipped or widened variants, that minimizes its squared reconstruction error.
    """
    out, inp = weight.shape
    if inp % BLOCK:
        raise ValueError(f'input width {inp} is not a multiple of {BLOCK}')
    if scale_2 is None:
        scale_2 = global_scale(weight.float().abs().max())
    packed = torch.empty(out, inp // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(out, inp // BLOCK, dtype=torch.float8_e4m3fn, device=weight.device)
    for r in range(0, out, rows):
        blocks = weight[r:r + rows].float().view(-1, inp // BLOCK, BLOCK)
        base = blocks.abs().amax(dim=-1) / 6.0 / scale_2
        best_codes = best_scale = best_err = None
        for factor in (SEARCH if search else SEARCH[:1]):
            scale = (base * factor).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
            codes, err = _encode(blocks, scale, scale_2)
            if best_err is None:
                best_codes, best_scale, best_err = codes, scale, err
                continue
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale.view(torch.uint8), best_scale.view(torch.uint8)).view(torch.float8_e4m3fn)
            best_codes = torch.where(better.unsqueeze(-1), codes, best_codes)
        codes = best_codes.view(-1, inp)
        packed[r:r + rows] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        scales[r:r + rows] = best_scale
    return packed, scales, scale_2.reshape(()).float()


def nearest_e2m1(x):
    """Round to nearest E2M1 magnitude (ties to even code) and return 4-bit codes with sign bit 3."""
    grid = E2M1.to(x.device)
    mag = x.abs().clamp(max=6.0)
    # Distance to every grid point; ties resolve toward the lower (even-mantissa) entry first.
    idx = (mag.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    # argmin picks the first minimum; E2M1 ties at midpoints go to even codes, which are the
    # lower neighbours except 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0.
    mid_up = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    up_even = torch.tensor([False, True, False, True, False, True, False], device=x.device)
    for m, up in zip(mid_up, up_even):
        tie = mag == m
        if up:
            idx = torch.where(tie, idx + 1, idx)
    sign = (x < 0) & (idx > 0)
    return (idx | (sign.to(idx.dtype) << 3)).to(torch.int32)


def dequantize(packed, scale, scale_2):
    out = packed.shape[0]
    lo = (packed & 0xF).int()
    hi = (packed >> 4).int()
    codes = torch.stack([lo, hi], dim=-1).view(out, -1)
    grid = E2M1.to(packed.device)
    values = grid[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    blocks = values.view(out, -1, BLOCK) * (scale.float() * scale_2).unsqueeze(-1)
    return blocks.view(out, -1)
