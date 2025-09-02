import gc
import json
import os
import random
import shutil
import uuid
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams 

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.gsm8k import normalize_r1_zero_format

matplotlib.use("Agg")

# --- Configuration & File Paths ---
# Run dir & artifact paths
RUN_NAME = os.environ.get("RUN_NAME", "grpo_off_policy_noclip")
CLEAR_OLD = os.environ.get("CLEAR_OLD", "0") == "1"

run_dir = os.path.abspath(f"./{RUN_NAME}")

if os.path.exists(run_dir) and CLEAR_OLD:
    shutil.rmtree(run_dir)

os.makedirs(run_dir, exist_ok=True)

train_log_path = os.path.join(run_dir, "train_log.json")
eval_plot_path = os.path.join(run_dir, "eval_curve.png")
policy_outdir = os.path.join(run_dir, "policy_final")

# If NOT clearing, try to resume the JSON log
train_log = []
if os.path.exists(train_log_path) and not CLEAR_OLD:
    try:
        with open(train_log_path, "r", encoding="utf-8") as f:
            train_log = json.load(f)
    except Exception:
        train_log = []

# In-memory curves/logs
eval_steps, eval_accs, eval_format_rates = [], [], []

# Dataset paths
test_path = "/home/zhangwj/assignment5/data/gsm8k/test.jsonl"
in_path = "/home/zhangwj/assignment5/data/gsm8k/train.jsonl"

# --- Hyperparameters ---
n_grpo_steps: int = 100
learning_rate: float = 1e-5
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 8
sampling_temperature: float = 0.7
sampling_min_tokens: int = 4
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 4
train_batch_size: int = 256
gradient_accumulation_steps: int = 128
gpu_memory_utilization: float = 0.85
loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    "grpo_no_clip",
] = "grpo_no_clip"
use_std_normalization: bool = True
refresh_vllm_every: int = 1
eval_every: int = 1
cliprange: float = 0.2
model_path = "/data/Qwen2.5-Math-1.5B"

length_normalization_type: Literal["masked_mean", "masked_normalize"] = "masked_normalize"

# --- Assertions and Derived Configs ---
assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
)
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
)
n_prompts_per_rollout_batch = rollout_batch_size // group_size
assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
)
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes the unclipped GRPO loss."""
    # This is the ratio: pi_theta / pi_theta_old
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # This is the unclipped objective from the problem description
    loss = -ratio * advantages

    metadata = {
        "ratio": ratio.mean(),
    }
    return loss, metadata


# --- Helper Functions ---
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    reward = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        fn_out = reward_fn(response, ground_truth)
        reward.append(fn_out["reward"])

    reward_len = len(reward)
    n_prompts = reward_len // group_size

    reward_tensor = torch.tensor(reward)
    reward_reshape = reward_tensor.reshape(n_prompts, group_size)
    group_mean = torch.mean(reward_reshape, dim=1, keepdim=True)

    result = reward_reshape - group_mean
    if normalize_by_std:
        group_std = torch.std(reward_reshape, dim=1, keepdim=True)
        result = result / (group_std + advantage_eps)

    advantages = result.reshape(reward_len)
    raw_rewards = torch.tensor(reward).reshape(reward_len)
    meta = {"mean": group_mean}
    return advantages, raw_rewards, meta


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.tensor:  # pyright: ignore[reportGeneralTypeIssues]
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    new_old = torch.exp(policy_log_probs - old_log_probs)
    clip = torch.clamp(new_old, min=1 - cliprange, max=1 + cliprange)

    left = new_old * advantages
    right = clip * advantages
    result = -torch.min(left, right)

    clip_true = (new_old > 1 + cliprange) | (new_old < 1 - cliprange)
    clipped_fraction = clip_true.float().mean()

    metadata = {
        "clipped_fraction": clipped_fraction,
        "ratio": new_old.mean(),
    }
    return result, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs # type: ignore
        )  # pyright: ignore[reportArgumentType]
        metadata = {"raw_reward": raw_rewards}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            advantages, policy_log_probs # type: ignore
        )  # pyright: ignore[reportArgumentType]
        metadata = {"advantages": advantages}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange # type: ignore
        )  # pyright: ignore[reportArgumentType]
    elif loss_type == "grpo_no_clip":  # <-- ADD THIS BLOCK
        loss, metadata = compute_grpo_no_clip_loss(
            advantages, policy_log_probs, old_log_probs # type: ignore
        )  # pyright: ignore[reportArgumentType]

    return loss, metadata  # pyright: ignore[reportReturnType]


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    tensor_mask = tensor * mask
    if dim is not None:
        result = torch.sum(tensor_mask, dim=dim, keepdim=False) / torch.sum(
            mask, dim=dim, keepdim=False
        )
    else:
        result = torch.sum(tensor_mask) / torch.sum(mask)
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    constant_normalize: int | None = None,
):
    tensor_mask = tensor * mask
    if dim is not None:
        result = torch.sum(tensor_mask, dim=dim, keepdim=False) / constant_normalize # type: ignore
    else:
        result = torch.sum(tensor_mask) / constant_normalize # type: ignore
    return result    


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization_type: Literal["masked_mean", "masked_normalize"] = "masked_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Executes a forward and backward pass on a microbatch."""
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    if length_normalization_type == "masked_mean":
        masked_loss = masked_mean(loss, response_mask)
    elif length_normalization_type == "masked_normalize":
        max_seq_len = response_mask.size(1)
        masked_loss = masked_normalize(loss, response_mask, dim=1, constant_normalize=max_seq_len)
    else:
        raise ValueError(f"Unknown normalization type: {length_normalization_type}")
    micro_batch_loss = masked_loss.mean()
    scaled_loss = micro_batch_loss / gradient_accumulation_steps
    scaled_loss.backward()
    return micro_batch_loss, metadata


def rebuild_vllm_from_policy(
    policy, tokenizer, gen_device_id, gpu_memory_utilization
):
    tmp_dir = f"/tmp/_tmp_vllm_eval_ckpt_{uuid.uuid4().hex}"
    policy.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gen_device_id)
    new_vllm = LLM(
        model=tmp_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return new_vllm, tmp_dir


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@torch.no_grad()
def evaluate_with_vllm(
    vllm_gen,
    tokenizer,
    test_path,
    system_prompt,
    sampling_params_eval,
    eval_batch_size: int = 64,
) -> tuple[float, float]:
    """
    Returns (answer_accuracy, format_accuracy) using vLLM.
    - answer_accuracy := mean of 'reward' (1 only when formatted AND correct)
    - format_accuracy := mean of 'format_reward' (1 when formatted; 0 otherwise)
    """
    QAs_test = []
    with open(test_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            q_text = ex.get("question")
            ans_raw = ex.get("answer")
            a_text = ans_raw.split("####")[-1].strip()
            QAs_test.append({"Q": q_text, "A": a_text})

    if len(QAs_test) == 0:
        return 0.0, 0.0

    prompts = [ex["Q"] for ex in QAs_test]
    gts = [ex["A"] for ex in QAs_test]
    prompts_chat = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    sum_reward = 0.0
    sum_format = 0.0
    N = len(prompts_chat)

    for pc_chunk, gt_chunk in zip(
        chunked(prompts_chat, eval_batch_size), chunked(gts, eval_batch_size)
    ):
        outs = vllm_gen.generate(pc_chunk, sampling_params_eval)
        for out, gt in zip(outs, gt_chunk):
            text = out.outputs[0].text
            norm_text = normalize_r1_zero_format(text)
            res = r1_zero_reward_fn(norm_text, gt)
            sum_reward += float(res.get("reward", 0.0))
            sum_format += float(res.get("format_reward", 0.0))

    acc = sum_reward / N
    fmt = sum_format / N
    return acc, fmt


# --- Main Training Loop ---
if __name__ == "__main__":
    train_device_id = 0
    gen_device_id = 1
    torch.cuda.set_device(train_device_id)
    device = torch.device(f"cuda:{train_device_id}")
    print(f"🔥 Training process started on GPU {train_device_id}")

    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa"
    ).to(device) # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    vllm_gen, _tmp_dir = rebuild_vllm_from_policy(
        policy, tokenizer, gen_device_id, gpu_memory_utilization
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_device_id)

    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
    )
    sampling_params_eval = SamplingParams(
        n=1, temperature=0.0, max_tokens=sampling_max_tokens)

    QAs = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            q_text = ex.get("question")
            ans_raw = ex.get("answer")
            a_text = ans_raw.split("####")[-1].strip()
            QAs.append({"Q": q_text, "A": a_text})

    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    try:
        for step in range(1, n_grpo_steps + 1):
            print(f"\n--- GRPO Step {step}/{n_grpo_steps} ---")

            print("Starting data generation for this step...")
            prompts_data = random.sample(QAs, n_prompts_per_rollout_batch)
            prompts = [x["Q"] for x in prompts_data]
            ground_truths = [x["A"] for x in prompts_data]

            prompts_chat_format = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]

            vllm_outputs = vllm_gen.generate(prompts_chat_format, sampling_params)

            rollout_responses = []
            old_token_logprobs = []
            for output in vllm_outputs:
                for sample in output.outputs:
                    rollout_responses.append(normalize_r1_zero_format(sample.text))


            repeated_ground_truths = [
                gt for gt in ground_truths for _ in range(group_size)
            ]
            advantages, raw_rewards, _ = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_responses,
                repeated_ground_truths,
                group_size,
                advantage_eps,
                use_std_normalization,
            )
            repeated_prompts_chat_format = [
                p for p in prompts_chat_format for _ in range(group_size)
            ]

            combined_tokens_old = tokenizer(
                [p + r for p, r in zip(repeated_prompts_chat_format, rollout_responses)],
                padding="longest",
                truncation=True,
                max_length=sampling_max_tokens + 512,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            prompt_tokens_old = tokenizer(
                repeated_prompts_chat_format, padding=False, truncation=False, add_special_tokens=False
            )
            prompt_lens_old = torch.tensor(
                [len(x) for x in prompt_tokens_old["input_ids"]], device=device
            )

            with torch.no_grad():
                policy.eval()  # Set model to evaluation mode for this forward pass
                input_ids_full = combined_tokens_old["input_ids"].to(device)
                attention_mask_full = combined_tokens_old["attention_mask"].to(device)
                
                # Process in smaller chunks to avoid OOM
                log_probs_chunks = []
                # Use the same micro_train_batch_size for chunking
                for i in range(0, input_ids_full.size(0), micro_train_batch_size):
                    chunk_end = i + micro_train_batch_size
                    
                    # Get chunks of the input data
                    input_ids_chunk = input_ids_full[i:chunk_end]
                    attention_mask_chunk = attention_mask_full[i:chunk_end]
                    
                    # Perform forward pass on the smaller chunk
                    outputs_chunk = policy(input_ids_chunk, attention_mask=attention_mask_chunk)
                    logits_chunk = outputs_chunk.logits[:, :-1, :]
                    labels_chunk = input_ids_chunk[:, 1:]
                    
                    log_probs_full_chunk = torch.nn.functional.log_softmax(logits_chunk, dim=-1)
                    
                    gathered_log_probs_chunk = torch.gather(
                        log_probs_full_chunk, 2, labels_chunk.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    log_probs_chunks.append(gathered_log_probs_chunk)

                # Combine the results from all chunks into a single tensor
                old_log_probs_full_batch = torch.cat(log_probs_chunks, dim=0)

            print("✅ Data generation complete. Starting training.")
            policy.train()
            step_losses = []
            total_grad_norm = 0.0

            for epoch in range(epochs_per_rollout_batch):
                optimizer.zero_grad()
                indices = torch.randperm(rollout_batch_size)
                accum = 0
                for i in range(0, rollout_batch_size, micro_train_batch_size):
                    micro_batch_indices = indices[i : i + micro_train_batch_size]

                    input_ids = combined_tokens_old["input_ids"][micro_batch_indices].to(device)
                    attention_mask = combined_tokens_old["attention_mask"][micro_batch_indices].to(device)
                    prompt_lens = prompt_lens_old[micro_batch_indices].to(device)
                    valid_mask_micro = attention_mask[:, 1:] == 1
                    positions_micro = torch.arange(input_ids.size(1) - 1, device=device).unsqueeze(0)
                    start_idx_micro = (prompt_lens - 1).clamp_min(0).unsqueeze(1)
                    response_mask = ((positions_micro >= start_idx_micro) & valid_mask_micro).float()

                    outputs = policy(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, :-1, :]
                    labels = input_ids[:, 1:]
                    log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
                    gathered_log_probs = torch.gather(
                        log_probs_full, 2, labels.unsqueeze(-1)
                    ).squeeze(-1)

                    policy_log_probs = gathered_log_probs
                
                    old_log_probs_micro = old_log_probs_full_batch[micro_batch_indices]
                    
                    adv_dtype = gathered_log_probs.dtype
                    advantages_micro = advantages[micro_batch_indices].to(device).to(adv_dtype).unsqueeze(1)
                    raw_rewards_micro = raw_rewards[micro_batch_indices].to(device).to(adv_dtype).unsqueeze(1)
                    num_ones = int((raw_rewards == 1).sum().item())
                    num_zeros = int((raw_rewards == 0).sum().item())
                    total = raw_rewards.numel()

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=raw_rewards_micro,
                        advantages=advantages_micro,
                        old_log_probs=old_log_probs_micro,
                        cliprange=cliprange,
                        length_normalization_type=length_normalization_type,
                    )
                    step_losses.append(float(loss.item()))
                    accum += 1
                    print(
                        f"Epoch {epoch+1}, Step {step}: Micro-batch loss: {loss.item():.4f}"
                    )
                    if (accum % gradient_accumulation_steps == 0) or (i + micro_train_batch_size >= rollout_batch_size):
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item()
                        optimizer.step()
                        optimizer.zero_grad()
                        print("✅ Optimizer step performed. Model weights updated.")
            if step % refresh_vllm_every == 0:
                try:
                    del vllm_gen
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

                prev_tmp_dir = _tmp_dir
                vllm_gen, _tmp_dir = rebuild_vllm_from_policy(
                    policy, tokenizer, gen_device_id, gpu_memory_utilization
                )
                shutil.rmtree(prev_tmp_dir, ignore_errors=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(train_device_id)
                print("🔄 vLLM reloaded with the latest policy weights.")

            train_reward_mean = float(raw_rewards.float().mean().item())
            train_reward_frac_ones = float((raw_rewards == 1).float().mean().item())
            num_reward_ones = int((raw_rewards == 1).sum().item())
            num_reward_zeros = int((raw_rewards == 0).sum().item())
            mean_loss = float(sum(step_losses) / max(1, len(step_losses)))

            if (step % eval_every == 0) or (step == n_grpo_steps):
                acc, fmt = evaluate_with_vllm(
                    vllm_gen,
                    tokenizer,
                    test_path,
                    system_prompt,
                    sampling_params_eval,
                    eval_batch_size=64,
                )
                eval_steps.append(step)
                eval_accs.append(acc)
                eval_format_rates.append(fmt)

                plt.figure()
                plt.plot(eval_steps, eval_accs, label="accuracy")
                plt.plot(eval_steps, eval_format_rates, label="format_ok")
                plt.ylim(0, 1)
                plt.xlabel("step")
                plt.ylabel("rate")
                plt.title("Evaluation on test.jsonl")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(eval_plot_path)
                plt.close()

                train_log.append(
                    {
                        "step": step,
                        "mean_loss": mean_loss,
                        "train_reward_mean": train_reward_mean,
                        "train_reward_frac_ones": train_reward_frac_ones,
                        "num_reward_ones": num_reward_ones,
                        "num_reward_zeros": num_reward_zeros,
                        "eval_accuracy": acc,
                        "eval_format_ok_rate": fmt,
                        "grad_norm": total_grad_norm,  # type: ignore
                    }
                )
                print(f"[EVAL] step={step} acc={acc:.3f} fmt={fmt:.3f} → {eval_plot_path}")
            else:
                train_log.append(
                    {
                        "step": step,
                        "mean_loss": mean_loss,
                        "train_reward_mean": train_reward_mean,
                        "train_reward_frac_ones": train_reward_frac_ones,
                        "num_reward_ones": num_reward_ones,
                        "num_reward_zeros": num_reward_zeros,
                        "eval_accuracy": None,
                        "eval_format_ok_rate": None,
                        "grad_norm": total_grad_norm,  # type: ignore
                    }
                )

            with open(train_log_path, "w", encoding="utf-8") as f:
                json.dump(train_log, f, ensure_ascii=False, indent=2)

            print(
                f"[LOG] step={step} loss={mean_loss:.4f} "
                f"train_r_mean={train_reward_mean:.3f} frac1={train_reward_frac_ones:.3f} "
                f"grad_norm={total_grad_norm:.4f} (log → {train_log_path})" # type: ignore
            )

        print("\nTraining finished.")
        os.makedirs(policy_outdir, exist_ok=True)
        policy.save_pretrained(policy_outdir)
        tokenizer.save_pretrained(policy_outdir)
        print(f"💾 Saved final policy to: {policy_outdir}")
        print(f"🧾 Training log JSON: {train_log_path}")
        print(f"📈 Eval curve PNG: {eval_plot_path}")
    finally:    
        print("\n🚨 Cleaning up VLLM instance and temporary files...")
        if 'vllm_gen' in locals():
            del vllm_gen # type: ignore
        
        if '_tmp_dir' in locals() and os.path.exists(_tmp_dir):
            shutil.rmtree(_tmp_dir, ignore_errors=True)
            print(f"🧹 Removed temporary directory: {_tmp_dir}")
        
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Cleanup complete.")

    print(f"🧾 Training log JSON: {train_log_path}")
    print(f"📈 Eval curve PNG: {eval_plot_path}")