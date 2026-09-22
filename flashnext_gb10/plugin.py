"""Opt-in vLLM extension; pinned to a source-reviewed upstream build."""
import os


def register():
    directory = os.environ.get("FLASHNEXT_PLE_NVME_DIR")
    if not directory:
        return
    import vllm
    if vllm.__version__ != "0.29.1rc1.dev518+ga33b3bac5":
        raise RuntimeError(f"Unvalidated vLLM build {vllm.__version__}; expected 0.29.1rc1.dev518+ga33b3bac5")
    if os.environ.get("FLASHNEXT_DFLASH_MODEL"):
        # Only the explicit experiment overlay provides this module.
        vllm.ModelRegistry.register_model(
            "DFlashQwen3DSparkModel", "dflash_epoch7:DFlashQwen3DSparkModel"
        )
    from vllm.models.qwen4_exp.nvidia import ngram_embedding
    if not getattr(ngram_embedding.Qwen4ExpPLEPinnedHostEmbedding, "_flashnext_nvme", False):
        from .vllm_nvme import make_nvme_embedding
        ngram_embedding.Qwen4ExpPLEPinnedHostEmbedding = make_nvme_embedding(ngram_embedding, directory)
    if os.environ.get("FLASHNEXT_DRAFT_VOCAB"):
        from .draft_vocab import register_draft_vocab
        register_draft_vocab(os.environ["FLASHNEXT_DRAFT_VOCAB"])
    if os.environ.get("FLASHNEXT_DIAGNOSTICS_DIR"):
        from .diagnostics import register_diagnostics
        register_diagnostics(os.environ["FLASHNEXT_DIAGNOSTICS_DIR"])
    if os.environ.get("FLASHNEXT_PACK_PLE_STATE") == "1":
        from .cache_packing import register_cache_packing
        register_cache_packing()
