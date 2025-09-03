# sft_math_corrected.py
# Supervised fine-tuning on GSM8K with r1-style prompts.
# - Corrected experimental design: Each training run starts from a fresh model.
# - Efficient tokenization using label masking (ignore_index=-100).
# - Optimized attention implementation for faster training.

import os
import re
import json
import random
from typing import List, Dict, Tuple

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# -------- Make sure these helper modules are in your path --------
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.gsm8k_baseline import (
    make_r1_zero_prompt,
    extract_gold_answer,
    normalize_r1_zero_format,
)

# ============================================================
#               Tokenization / Loss Utilities
# (Refactored for efficiency and standard Hugging Face practice)
# ============================================================

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes prompts and outputs, combines them, and creates labels
    for supervised fine-tuning. The labels for prompt tokens are set
    to -100 to be ignored in the loss calculation.
    """
    # Tokenize the full text (prompt + output)
    full_texts = [p + o for p, o in zip(prompt_strs, output_strs)]
    
    # Use tokenizer's padding capabilities
    tokenized_full = tokenizer(
        full_texts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = tokenized_full.input_ids
    attention_mask = tokenized_full.attention_mask

    # Tokenize prompts separately to find their lengths
    tokenized_prompts = tokenizer(
        prompt_strs,
        padding=False,
        add_special_tokens=False,
    )

    # Create labels by cloning input_ids
    labels = input_ids.clone()

    # Mask out the prompt tokens in the labels
    for i in range(len(prompt_strs)):
        prompt_len = len(tokenized_prompts.input_ids[i])
        labels[i, :prompt_len] = -100 # -100 is the standard ignore_index in PyTorch

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def sft_microbatch_train_step(
    loss: torch.Tensor,
    gradient_accumulation_steps: int,
) -> Dict[str, torch.Tensor]:
    """
    Scales the loss for gradient accumulation and performs the backward pass.
    """
    # The loss from the model is already the mean over the batch tokens
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    meta = {
        "loss_value": loss.detach(),
    }
    return meta

# ============================================================
#                           Data
# ============================================================

TRAIN_PATH = "/home/zhangwj/assignment5/data/gsm8k/train.jsonl"
TEST_PATH  = "/home/zhangwj/assignment5/data/gsm8k/test.jsonl"

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH),  f"Missing {TEST_PATH}"

def load_gsm8k_local() -> Tuple[List[Dict], List[Dict]]:
    """
    Load JSONL GSM8K files and create SFT targets:
    <think> original GSM8K 'answer' </think> <answer> final_number </answer>
    """
    train_ds = load_dataset("json", data_files=TRAIN_PATH)["train"]
    test_ds  = load_dataset("json", data_files=TEST_PATH)["train"]

    def to_list(ds):
        out = []
        for ex in ds:
            q = ex["question"]
            a_full = ex["answer"]
            gold = extract_gold_answer(a_full)
            target = f"<think>\n{a_full.strip()}\n</think> <answer> {gold} </answer>"
            out.append({"question": q, "solution": a_full, "gold": gold, "target": target})
        return out

    return to_list(train_ds), to_list(test_ds)

# ============================================================
#                      Model / Tokenizer
# ============================================================

MODEL_ID = "/data/Qwen2.5-Math-1.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer():
    print(f"Loading model from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", # Use Flash Attention 2 for speed: flash_attention_2
    ).to(device) # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ============================================================
#                   Train / Evaluate Routines
# ============================================================

def build_batch(batch: List[Dict], tokenizer):
    prompts = [make_r1_zero_prompt(ex["question"]) for ex in batch]
    targets = [ex["target"] for ex in batch]
    pack = tokenize_prompt_and_output(prompts, targets, tokenizer)
    for k in pack:
        pack[k] = pack[k].to(device)
    return pack

def train_epoch(
    model,
    tokenizer,
    data: List[Dict],
    optimizer: torch.optim.Optimizer,
    batch_size: int = 4,
    grad_accum: int = 8,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    step = 0
    num_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        micro = data[i : i + batch_size]
        pack = build_batch(micro, tokenizer)
        
        # The model computes the masked cross-entropy loss directly
        outputs = model(**pack)
        loss = outputs.loss
        
        meta = sft_microbatch_train_step(
            loss=loss,
            gradient_accumulation_steps=grad_accum,
        )

        total_loss += meta["loss_value"].item()
        step += 1

        if step % grad_accum == 0 or i + batch_size >= len(data):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    
    avg_loss = total_loss / num_batches
    return {"avg_loss": avg_loss}

@torch.no_grad()
def eval_accuracy(model, tokenizer, samples: List[Dict], max_new_tokens: int = 128) -> float:
    model.eval()
    correct = 0
    for ex in samples:
        prompt = make_r1_zero_prompt(ex["question"])
        in_ids = tokenizer(prompt, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **in_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_txt = tokenizer.decode(
            gen_ids[0][in_ids["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        response = normalize_r1_zero_format(gen_txt)
        metrics = r1_zero_reward_fn(response=response, ground_truth=ex["gold"], fast=True)
        if metrics.get("answer_reward", 0.0) > 0.5:
            correct += 1
    return correct / max(1, len(samples))

def run_experiment(
    model,
    tokenizer,
    optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    label: str,
    batch_size: int = 4,
    grad_accum: int = 8,
    epochs: int = 1,
) -> Tuple[str, float]:
    """
    Run a single, complete training and evaluation experiment.
    """
    print(f"\n=== Training on configuration: {label} (N={len(train_data)}) ===")
    for ep in range(epochs):
        stats = train_epoch(model, tokenizer, train_data, optimizer, batch_size=batch_size, grad_accum=grad_accum)
        print(f"  epoch {ep+1}: avg_loss={stats['avg_loss']:.4f}")

    acc = eval_accuracy(model, tokenizer, val_data)
    print(f"  -> validation accuracy: {acc*100:.2f}%")
    return (label, acc)


# ============================================================
#                           Main
# ============================================================

if __name__ == "__main__":
    # Load data
    train_all, val_all = load_gsm8k_local()

    # Shuffle train once for reproducibility
    random.Random(7).shuffle(train_all)

    # --- Config ---
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    EPOCHS = 1
    LEARNING_RATE = 2e-5

    # ---------- Stage 1: accuracy vs dataset size ----------
    print("== Stage 1: GSM8K SFT — dataset size sweep ==")
    size_curve_results = []
    sizes_to_run = (128, 256, 512, 1024, 1536, 2048, None) # None means full dataset

    for sz in sizes_to_run:
        # **RELOAD** the model and optimizer for each independent run to ensure a fair comparison
        model, tokenizer = load_model_and_tokenizer()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        if sz is None:
            subset = train_all
            label = "full"
        else:
            subset = train_all[:min(sz, len(train_all))]
            label = str(sz)
        
        result = run_experiment(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_data=subset,
            val_data=val_all[:100], # INCREASED validation set size
            label=label,
            batch_size=BATCH_SIZE,
            grad_accum=GRAD_ACCUM,
            epochs=EPOCHS,
        )
        size_curve_results.append(result)

    with open("gsm8k_sft_size_curve.json", "w") as f:
        json.dump(size_curve_results, f, indent=2)

    print("\nDone. Results saved to:")
    print(" - gsm8k_sft_size_curve.json")

