"""Serve retrained MTP drafter weights in place of the checkpoint's.

FLASHNEXT_MTP_OVERRIDE=<file.pt> holds {checkpoint name: tensor} for the
drafter's non-expert weights (experiments/drafter/train_mtp.py). Each
tensor replaces the checkpoint tensor of the same name and shape while the
drafter loads; routed experts and every target weight are untouched. The
drafter only proposes tokens, so target outputs are unchanged.
"""
import logging

import torch

logger = logging.getLogger("vllm.flashnext.mtp_override")


def register_mtp_override(path):
    from vllm.models.qwen4_exp.nvidia import mtp as nvidia_mtp

    Model = nvidia_mtp.Qwen4ExpMTP
    if getattr(Model, "_flashnext_mtp_override", False):
        return
    original = Model.load_weights

    def load_weights(self, weights):
        override = torch.load(path, map_location="cpu")
        used = set()

        def substituted():
            for name, tensor in weights:
                replacement = override.get(name)
                if replacement is not None:
                    if replacement.shape != tensor.shape:
                        raise RuntimeError(f"{name}: override shape {tuple(replacement.shape)} "
                                           f"!= checkpoint {tuple(tensor.shape)}")
                    used.add(name)
                    tensor = replacement.to(tensor.dtype)
                yield name, tensor

        loaded = original(self, substituted())
        unused = sorted(set(override) - used)
        if unused:
            raise RuntimeError(f"MTP override tensors not in the checkpoint: {unused[:8]}")
        logger.info("FlashNext MTP override: replaced %d drafter tensors from %s", len(used), path)
        return loaded

    Model.load_weights = load_weights
    Model._flashnext_mtp_override = True
