# wordle_sft_grpo_local.py
"""
Local SFT -> GRPO pipeline for the Wordle example (TRL-based).
Usage:
    python wordle_sft_grpo_local.py \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --sft_epochs 1 \
        --grpo_steps 100

Notes:
- This script assumes reward_functions.py is in the same folder and exposes:
    output_format_check(prompt, completion, example)
    uses_previous_feedback(prompt, completion, example)
    guess_value(prompt, completion, example)
- For larger models use: `accelerate launch --multi_gpu ... python wordle_sft_grpo_local.py ...`
"""
import argparse
import logging
import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
import weave

import gc
torch.cuda.empty_cache()
gc.collect()

# import the reward functions you provided
from reward_functions import (
    output_format_check,
    uses_previous_feedback,
    guess_value,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---- Weave / W&B setup ----
wandb.init(project="wordle-sft-grpo", job_type="training")
weave.init(project="wordle-sft-grpo")


@weave.op()
def extract_text_from_completion(completion_item: Any) -> str:
    """
    GRPO/SFT can return completions in different shapes:
      - simple string
      - a list of strings
      - a list of message dicts: [{"role": "...", "content": "..."}]
    This helper normalizes that to a single string.
    """
    if completion_item is None:
        return ""
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, list):
        # list of dicts (conversational) or strings
        if len(completion_item) == 0:
            return ""
        first = completion_item[0]
        if isinstance(first, dict) and "content" in first:
            return "".join(m.get("content", "") for m in completion_item)
        else:
            # list of strings
            return "".join(map(str, completion_item))
    # fallback
    return str(completion_item)


def make_trl_wrapper(fn):
    """
    Wrap a single-sample reward function fn(prompt, completion, example)
    into a TRL-style reward function that accepts lists and returns lists.
    The trainer will pass additional dataset columns as keyword args;
    kwargs values are typically lists aligned with prompts.
    """
    @weave.op()
    def wrapper(prompts: List[str], completions: List[Any], **kwargs) -> List[float]:
        results = []
        n = len(prompts)
        # For each sample, build the example dict from kwargs
        for i in range(n):
            p = prompts[i]
            c = extract_text_from_completion(completions[i])
            # build example dict (each kwarg may be a list aligned to prompts)
            example: Dict[str, Any] = {}
            for name, val in kwargs.items():
                # if val is list-like and same length, take the i-th element
                try:
                    if hasattr(val, "__len__") and len(val) == n:
                        example[name] = val[i]
                    else:
                        example[name] = val
                except Exception:
                    example[name] = val
            try:
                v = float(fn(p, c, example) or 0.0)
            except Exception as exc:
                logger.exception("Exception inside reward function %s for sample %d: %s", fn.__name__, i, exc)
                v = 0.0
            results.append(float(v))
        return results
    wrapper.__name__ = f"trl_wrapped_{fn.__name__}"
    return wrapper


@weave.op()
def small_generate_and_score(model, tokenizer, dataset, reward_wrappers, num=5, device="cpu"):
    """
    Quick sanity check: generate completions for first `num` prompts and
    compute rewards via reward_wrappers (list).
    """
    model.eval()
    model.to(device)

    samples = dataset.select(range(min(num, len(dataset))))
    prompts = [row["prompt"] for row in samples]
    # tokenize + generate
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=32, do_sample=False)
    completions = [tokenizer.decode(x[len(enc["input_ids"][i]):], skip_special_tokens=True) for i, x in enumerate(out_ids)]
    # run the wrappers (they accept lists)
    total_rewards = [0.0] * len(prompts)
    for wrap in reward_wrappers:
        # convert dataset columns to kwargs for wrappers: gather lists for all columns present
        kwargs = {col: [row[col] for row in samples] for col in samples.column_names if col not in ("prompt",)}
        rewards = wrap(prompts=prompts, completions=completions, **kwargs)
        for i, r in enumerate(rewards):
            total_rewards[i] += float(r or 0.0)
    print("--- Sanity generation + reward check ---")
    for i, (p, c, r) in enumerate(zip(prompts, completions, total_rewards)):
        print(f"[{i}] PROMPT:\n{p}\nCOMPLETION:\n{c}\nREWARD_SUM: {r}\n{'-'*60}")


@weave.op()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model for SFT (pick a small one for local runs)")
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--sft_batch", type=int, default=2)
    parser.add_argument("--grpo_steps", type=int, default=100)
    parser.add_argument("--grpo_batch", type=int, default=2)
    parser.add_argument("--sft_output", type=str, default="./wordle-sft-checkpoint")
    parser.add_argument("--grpo_output", type=str, default="./wordle-grpo-checkpoint")
    parser.add_argument("--test_only", action="store_true", help="Only run the small generation/reward test, skip training")
    #args = parser.parse_args()
    args = parser.parse_known_args()[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1) Load datasets
    logger.info("Loading datasets...")
    sft_dataset = load_dataset("predibase/wordle-sft", split="train")
    grpo_dataset = load_dataset("predibase/wordle-grpo", split="train")

    logger.info("SFT dataset columns: %s", sft_dataset.column_names)
    logger.info("GRPO dataset columns: %s", grpo_dataset.column_names)

    # SFT: TRL can accept dataset with 'prompt' and 'completion' columns directly.
    # No need to concatenate into a single "text" column. SFTTrainer handles prompt+completion format. :contentReference[oaicite:1]{index=1}

    if args.test_only:
        # quick model load for generation test
        logger.info("Loading model & tokenizer for quick test: %s", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map="auto")
        # wrap reward fns
        wrappers = [make_trl_wrapper(output_format_check),
                    make_trl_wrapper(uses_previous_feedback),
                    make_trl_wrapper(guess_value)]
        small_generate_and_score(model, tokenizer, grpo_dataset, wrappers, num=4, device=device)
        return

    # ---------- SFT Stage ----------
    logger.info("Starting supervised fine-tuning (SFT)...")
    sft_config = SFTConfig(
        output_dir=args.sft_output,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.sft_batch,  # e.g. set to 1 or both SFT and GRPO.
        gradient_accumulation_steps=1,  # effective batch size = 4
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=100,
        max_length=1024,
        report_to="wandb",
        fp16=True,  # enables mixed precision training
    )

    # Instantiate SFTTrainer. Pass model name (trainer will load it).
    sft_trainer = SFTTrainer(
        model=args.model_name,
        args=sft_config,
        train_dataset=sft_dataset,
    )

    logger.info("Running SFT trainer.train() ...")
    sft_trainer.train()
    logger.info("SFT finished. Checkpoint saved to %s", args.sft_output)

    # ---------- Reward wrappers ----------
    logger.info("Wrapping reward functions for GRPOTrainer...")
    wrapped_output_format = make_trl_wrapper(output_format_check)
    wrapped_uses_prev = make_trl_wrapper(uses_previous_feedback)
    wrapped_guess_value = make_trl_wrapper(guess_value)
    reward_wrappers = [wrapped_output_format, wrapped_uses_prev, wrapped_guess_value]

    # Quick generation + reward sanity check BEFORE GRPO
    logger.info("Quick generation + reward test (before GRPO)")
    # Reuse the trained SFT model for generation test
    tokenizer = sft_trainer.tokenizer
    sft_model = sft_trainer.model
    small_generate_and_score(sft_model, tokenizer, grpo_dataset, reward_wrappers, num=4, device=device)

    # ---------- GRPO Stage ----------
    logger.info("Starting GRPO stage, continuing from SFT checkpoint: %s", args.sft_output)
    grpo_cfg = GRPOConfig(
        output_dir=args.grpo_output,
        max_steps=args.grpo_steps,
        per_device_train_batch_size=args.grpo_batch,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=100,
        num_generations=8,               # how many completions per prompt (small default)
        max_prompt_length=512,
        max_completion_length=16,
        report_to="wandb",
        # you can add vllm / generation kwargs here if you have vLLM available
    )

    # Initialize GRPO trainer continuing from SFT checkpoint
    grpo_trainer = GRPOTrainer(
        model=args.sft_output,                 # continue from SFT checkpoint
        args=grpo_cfg,
        train_dataset=grpo_dataset,
        reward_funcs=reward_wrappers,         # trl will sum them (or you can supply a single wrapper)
    )

    logger.info("Running GRPO trainer.train() ...")
    grpo_trainer.train()
    logger.info("GRPO finished. Checkpoint saved to %s", args.grpo_output)


if __name__ == "__main__":
    main()
