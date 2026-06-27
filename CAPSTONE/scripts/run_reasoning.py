#!/usr/bin/env python
"""Stage 2 — Frozen-LLM Zero-Shot Reasoning.

Reads the enhanced CSV produced by ``run_perception.py``, wraps each scene
description + question into a prompt, runs batched zero-shot inference with a
frozen LLM, and writes the model's answers (with reasoning chain) to
``final_results.csv``.

Example
-------
    python scripts/run_reasoning.py \
        --input vsr_detection_results_enhanced.csv \
        --model Qwen/Qwen2.5-7B-Instruct \
        --prompt-style chain_of_thought \
        --output-dir outputs
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capstone.reasoning import load_llm_model, process_data_in_batches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reasoning.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CAPSTONE reasoning stage")
    parser.add_argument("--input", default="detection_results_enhanced.csv",
                        help="Enhanced CSV from the perception stage")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Frozen LLM backend (HF id)")
    parser.add_argument("--prompt-style", default="chain_of_thought",
                        choices=["chain_of_thought", "cot", "direct", "pope"])
    parser.add_argument("--batch-size", type=int, default=None,
                        help="None => auto (from free GPU memory)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading data from %s", args.input)
    df = pd.read_csv(args.input)
    if args.max_samples:
        df = df.head(args.max_samples)
        logger.info("Limited to %d samples", args.max_samples)

    model, tokenizer = load_llm_model(args.model)

    df_result = process_data_in_batches(
        df, model, tokenizer,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        prompt_style=args.prompt_style,
        max_new_tokens=args.max_new_tokens,
    )

    final_output = os.path.join(args.output_dir, "final_results.csv")
    df_result.to_csv(final_output, index=False)
    logger.info("Saved final results to %s", final_output)

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
