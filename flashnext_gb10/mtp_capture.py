"""Capture MTP drafter inputs and outputs for drafter distillation.

FLASHNEXT_CAPTURE_MTP_DIR=<dir> (eager mode, MTP enabled) saves every
first-step drafter forward with at least FLASHNEXT_CAPTURE_MIN_TOKENS tokens,
which are prefill chunks: the drafter's input token IDs and positions, the
target's pre-final-mixer multi-stream hidden state [T, hc_count * hidden] it
consumes, and its outputs (the single stream for the LM head and the
multi stream for the next draft step). Replaying one request at a time makes
each chunk one contiguous span of one sequence. Outputs are unchanged.
FLASHNEXT_CAPTURE_MTP_OUTPUTS=0 keeps only the training inputs (IDs,
positions, target hidden), which halves the size of the capture.
"""
import itertools
import os
from pathlib import Path

import torch


def register_mtp_capture(directory):
    from vllm.models.qwen4_exp.nvidia import mtp as nvidia_mtp

    Predictor = nvidia_mtp.Qwen4ExpMultiTokenPredictor
    if getattr(Predictor, '_flashnext_mtp_capture', False):
        return
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    minimum = int(os.environ.get('FLASHNEXT_CAPTURE_MIN_TOKENS', '64'))
    keep_outputs = os.environ.get('FLASHNEXT_CAPTURE_MTP_OUTPUTS', '1') == '1'
    counter = itertools.count()
    original = Predictor.forward

    def forward(self, input_ids, positions, hidden_states=None, intermediate_tensors=None,
                inputs_embeds=None, spec_step_idx=0):
        result = original(self, input_ids, positions, hidden_states, intermediate_tensors,
                          inputs_embeds, spec_step_idx)
        if (spec_step_idx == 0 and input_ids is not None and hidden_states is not None
                and input_ids.shape[0] >= minimum and isinstance(result, tuple)
                and not torch.cuda.is_current_stream_capturing()):
            count = input_ids.shape[0]
            cpu = lambda t: t[:count].to('cpu')
            record = dict(ids=cpu(input_ids).int(), positions=cpu(positions.reshape(-1)).int(),
                          hidden=cpu(hidden_states).bfloat16())
            if keep_outputs:
                record.update(sample_hidden=cpu(result[0]).bfloat16(), multi_hidden=cpu(result[1]).bfloat16())
            torch.save(record, out / f'{next(counter):07d}.pt')
        return result

    Predictor.forward = forward
    Predictor._flashnext_mtp_capture = True
