"""Load a frozen LLM reasoning backend L : D x Q -> A.

No multimodal fine-tuning is performed. The paper evaluates several frozen
backends: LLaMA-3.2-1B, Qwen2.5-1.5B, DeepSeek-R1-1.5B, Qwen2.5-7B (best on
POPE) and Qwen3-4B-Thinking / Qwen3-8B (best on VSR).
"""

import logging

logger = logging.getLogger(__name__)


def load_llm_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", max_length: int = 2048):
    """Load an instruction-tuned LLM and tokenizer in inference mode.

    Returns
    -------
    tuple
        ``(model, tokenizer)`` with the model in ``eval`` mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info("Loading LLM: %s", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer
    except Exception:
        logger.exception("Failed to load model %s", model_name)
        raise
