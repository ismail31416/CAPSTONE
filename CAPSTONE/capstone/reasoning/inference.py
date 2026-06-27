"""Batched zero-shot inference over symbolic descriptions.

Implements ``a_hat = L(P)`` for a whole dataset: prepares prompts, batches
them (with optional dynamic batch sizing), generates answers, and checkpoints
progress so long runs can resume.
"""

import gc
import logging
import os
import time
from typing import Dict, List

from tqdm import tqdm

from .prompts import create_prompt

logger = logging.getLogger(__name__)


def prepare_all_prompts(df, prompt_style: str = "chain_of_thought") -> List[Dict]:
    """Build a prompt for every row of ``df`` (needs ``formatted_results`` + ``question``)."""
    logger.info("Preparing prompts (style=%s)", prompt_style)
    formatted_results = df["formatted_results"].tolist()
    questions = df["question"].tolist()

    all_prompts = []
    for idx in tqdm(range(len(df)), total=len(df)):
        all_prompts.append({
            "index": idx,
            "prompt": create_prompt(formatted_results[idx], questions[idx], prompt_style),
        })
    logger.info("Prepared %d prompts", len(all_prompts))
    return all_prompts


def dynamic_batch_size(available_memory, model_size_per_example):
    """Pick a batch size from free GPU memory (min 1, max 32, 20% buffer)."""
    usable = available_memory * 0.8
    return max(1, min(32, int(usable // model_size_per_example)))


def process_batch(model, tokenizer, prompt_batch, max_new_tokens=1024,
                  temperature=0.7, top_p=0.9):
    """Generate answers for one batch of prompts."""
    import torch
    try:
        prompts = [item["prompt"] for item in prompt_batch]
        inputs = tokenizer(
            prompts, padding=True, return_tensors="pt", truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                use_cache=True,
            )

        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = []
        for prompt_item, full_text in zip(prompt_batch, generated):
            caption = full_text[len(prompt_item["prompt"]):].strip()
            if caption.startswith('"') and caption.endswith('"'):
                caption = caption[1:-1]
            results.append({"index": prompt_item["index"], "caption": caption})
        return results
    except Exception:
        logger.exception("Error processing batch")
        return [{"index": item["index"], "caption": ""} for item in prompt_batch]


def process_data_in_batches(df, model, tokenizer, batch_size=None,
                            output_dir="outputs", prompt_style="chain_of_thought",
                            max_new_tokens=1024):
    """Run inference over all rows with checkpointing and intermediate saves.

    Returns the dataframe with an added ``llm_caption`` column (the raw LLM
    output, which includes the reasoning chain and ``Answer: [YES/NO]``).
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    df_result = df.copy()
    if "llm_caption" not in df_result.columns:
        df_result["llm_caption"] = None

    rows_to_process = df_result["llm_caption"].isnull()
    if not rows_to_process.any():
        logger.info("All rows already processed, skipping")
        return df_result

    df_to_process = df_result[rows_to_process].reset_index(drop=True)
    all_prompts = prepare_all_prompts(df_to_process, prompt_style)

    if batch_size is None:
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(0).total_memory * 0.9
            mem_per_example = 500 * 1024 * 1024  # ~500 MB/example heuristic
            batch_size = dynamic_batch_size(free_mem, mem_per_example)
            logger.info("Dynamically determined batch size: %d", batch_size)
        else:
            batch_size = 4
            logger.info("No CUDA available, using default batch_size: %d", batch_size)

    checkpoint_file = os.path.join(output_dir, "processing_checkpoint.txt")
    processed_indices = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            processed_indices = {int(line.strip()) for line in f if line.strip()}
        logger.info("Found checkpoint with %d processed indices", len(processed_indices))

    prompts_to_process = [p for p in all_prompts if p["index"] not in processed_indices]
    logger.info("Processing %d / %d prompts", len(prompts_to_process), len(all_prompts))

    results = []
    start_time = time.time()
    for i in range(0, len(prompts_to_process), batch_size):
        batch_prompts = prompts_to_process[i:i + batch_size]
        batch_start = time.time()
        batch_results = process_batch(
            model, tokenizer, batch_prompts, max_new_tokens=max_new_tokens
        )
        results.extend(batch_results)

        with open(checkpoint_file, "a") as f:
            for item in batch_results:
                f.write(f"{item['index']}\n")

        batch_time = time.time() - batch_start
        eps = len(batch_prompts) / max(batch_time, 1e-9)
        processed = i + len(batch_prompts)
        logger.info(
            "Batch %d: %.2f ex/s | %d/%d (%.1f%%)",
            i // batch_size + 1, eps, processed, len(prompts_to_process),
            processed / len(prompts_to_process) * 100,
        )

        if i % (batch_size * 5) == 0 and i > 0:
            _write_back(df_result, df_to_process, results)
            inter = os.path.join(output_dir, f"intermediate_results_{i}.csv")
            df_result.to_csv(inter, index=False)
            logger.info("Saved intermediate results to %s", inter)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    _write_back(df_result, df_to_process, results)
    df_result["llm_caption"] = df_result["llm_caption"].fillna("")
    logger.info("Total inference time: %.2fs", time.time() - start_time)
    return df_result


def _write_back(df_result, df_to_process, results):
    """Map ``index -> caption`` results back onto the original dataframe."""
    result_map = {r["index"]: r["caption"] for r in results}
    original_indices = df_to_process.index.tolist()
    for idx, caption in result_map.items():
        df_result.at[original_indices[idx], "llm_caption"] = caption
