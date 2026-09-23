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
        if os.environ.get("FLASHNEXT_PLE_PREFORWARD") == "1":
            from vllm.models.qwen4_exp.nvidia import model_state
            from .vllm_nvme import register_preforward
            register_preforward(ngram_embedding, model_state)
            ngram_embedding.logger.info("FlashNext PLE rows are staged before the forward")
    if os.environ.get("FLASHNEXT_DRAFT_VOCAB"):
        from .draft_vocab import register_draft_vocab
        register_draft_vocab(os.environ["FLASHNEXT_DRAFT_VOCAB"])
    if os.environ.get("FLASHNEXT_DIAGNOSTICS_DIR"):
        from .diagnostics import register_diagnostics
        register_diagnostics(os.environ["FLASHNEXT_DIAGNOSTICS_DIR"])
    if os.environ.get("FLASHNEXT_DENSE_FP8") == "1":
        from .dense_fp8 import register_dense_fp8
        register_dense_fp8()
    if os.environ.get("FLASHNEXT_DENSE_W8A16") == "1":
        if os.environ.get("FLASHNEXT_DENSE_FP8") == "1":
            raise RuntimeError("FLASHNEXT_DENSE_W8A16 and FLASHNEXT_DENSE_FP8 are alternatives")
        from .dense_w8a16 import register_dense_w8a16
        register_dense_w8a16()
    if os.environ.get("FLASHNEXT_COUNT_EXPERTS"):
        from .expert_count import register_expert_count
        register_expert_count(os.environ["FLASHNEXT_COUNT_EXPERTS"])
    if os.environ.get("FLASHNEXT_MTP_OVERRIDE"):
        from .mtp_override import register_mtp_override
        register_mtp_override(os.environ["FLASHNEXT_MTP_OVERRIDE"])
    if os.environ.get("FLASHNEXT_CAPTURE_MTP_DIR"):
        from .mtp_capture import register_mtp_capture
        register_mtp_capture(os.environ["FLASHNEXT_CAPTURE_MTP_DIR"])
    if os.environ.get("FLASHNEXT_CAPTURE_QSA_DIR"):
        from .qsa_capture import register_qsa_capture
        register_qsa_capture(os.environ["FLASHNEXT_CAPTURE_QSA_DIR"])
    if os.environ.get("FLASHNEXT_CAPTURE_MODULES_DIR"):
        from .module_capture import register_module_capture
        register_module_capture(os.environ["FLASHNEXT_CAPTURE_MODULES_DIR"])
    if os.environ.get("FLASHNEXT_PACK_PLE_STATE") == "1":
        from .cache_packing import register_cache_packing
        register_cache_packing()
