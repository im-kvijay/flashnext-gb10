"""Opt-in vLLM extension; pinned to a source-reviewed upstream build."""
import os


def register():
    directory = os.environ.get("FLASHNEXT_PLE_NVME_DIR")
    if not directory:
        return
    import vllm
    if vllm.__version__ != "0.29.1rc1.dev518+ga33b3bac5":
        raise RuntimeError(f"Unvalidated vLLM build {vllm.__version__}; expected 0.29.1rc1.dev518+ga33b3bac5")
    from vllm.models.qwen4_exp.nvidia import ngram_embedding
    if getattr(ngram_embedding.Qwen4ExpPLEPinnedHostEmbedding, "_flashnext_nvme", False):
        return
    from .vllm_nvme import make_nvme_embedding
    ngram_embedding.Qwen4ExpPLEPinnedHostEmbedding = make_nvme_embedding(ngram_embedding, directory)
