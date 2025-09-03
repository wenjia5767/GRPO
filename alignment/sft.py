import csv
import os
import re
import json
import random
import argparse
import pandas as pd
from typing import List, Dict, Tuple
from unittest.mock import patch
import wandb
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams # pyright: ignore[reportMissingImports]
from vllm.model_executor import set_random_seed as vllm_set_random_seed # pyright: ignore[reportMissingImports]
import datetime

from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.gsm8k_baseline import (
    make_r1_zero_prompt,
    extract_gold_answer,
    normalize_r1_zero_format,
)


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes prompt and output separately, concatenates, pads to the same length,
    and returns:
      - input_ids: (B, max_len-1)
      - labels:    (B, max_len-1) = shifted input_ids
      - response_mask: (B, max_len-1) mask over RESPONSE tokens in labels (1=response, 0=prompt/pad)
    """
    # 1) Tokenize without special tokens
    prompt_tokens_list = tokenizer(prompt_strs, add_special_tokens=False).input_ids
    output_tokens_list = tokenizer(output_strs, add_special_tokens=False).input_ids

    # 2) Concatenate per example
    combined_ids_list = [p + o for p, o in zip(prompt_tokens_list, output_tokens_list)]

    # 3) Pad to max length
    max_len = max(len(ids) for ids in combined_ids_list)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_combined_ids = [ids + [pad_token_id] * (max_len - len(ids)) for ids in combined_ids_list]
    combined_ids = torch.tensor(padded_combined_ids, dtype=torch.long)

    # 4) Build response mask over the FULL sequence (1 on output tokens)
    response_mask_full = torch.zeros_like(combined_ids, dtype=torch.long)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_tokens_list, output_tokens_list)):
        p_len, o_len = len(p_ids), len(o_ids)
        response_mask_full[i, p_len:p_len + o_len] = 1

    # 5) Slice to produce input_ids, labels, and mask aligned with LABELS
    input_ids = combined_ids[:, :-1]
    labels = combined_ids[:, 1:]
    response_mask = response_mask_full[:, 1:]   # <-- align mask with labels

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,   # (batch, seq_len)
    labels: torch.Tensor,      # (batch, seq_len)
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Return per-token conditional log-probabilities log p(x_t | x_<t) and,
    optionally, the per-token entropy of the model's next-token distribution.
    Shapes in/out all follow the spec: (batch, seq_len).
    """
    model.eval()
    with torch.no_grad():
        # (batch, seq_len, vocab)
        logits = model(input_ids).logits

        # log softmax over vocab to get log-probabilities
        log_probs_all = torch.log_softmax(logits, dim=-1)

        # pick the log-prob assigned to the true next token at each position
        # gather expects the index tensor to have an extra last dim
        log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        out: Dict[str, torch.Tensor] = {"log_probs": log_probs}

        if return_token_entropy:
            out["token_entropy"] = compute_entropy(logits)  # (batch, seq_len)

    return out


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those
    elements where mask == 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included.
        normalize_constant: float, the constant to divide by for normalization.
        dim: int | None, dimension to sum along. If None, sum over all dimensions.

    Returns:
        torch.Tensor: the normalized sum, ignoring elements where mask == 0.
    """
    # Apply mask
    masked_tensor = tensor * mask

    # Sum over the specified dimension
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    # Normalize by constant
    normalized = summed / normalize_constant

    return normalized


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,   # (B, T)
    response_mask: torch.Tensor,      # (B, T) bool or 0/1
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # 1. Calculate the negative log-likelihood (NLL) loss for each token.
    nll_loss_per_token = -policy_log_probs

    # 2. Use the provided helper function to sum the loss and divide by normalize_constant.
    # This correctly applies the normalization from the function's arguments.
    normalized_sum = masked_normalize(
        tensor=nll_loss_per_token,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )

    # 3. Get the batch size from the input tensor shape.
    batch_size = policy_log_probs.shape[0]

    # 4. Apply the remaining scaling factors for batch size and gradient accumulation.
    loss = normalized_sum / (batch_size * gradient_accumulation_steps)

    # 5. Perform the backward pass on the final loss.
    loss.backward()

    # 6. Prepare the values to return.
    metadata = {}

    return loss.detach(), metadata


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    ground_truths: List[str],
    max_new_tokens: int = 60,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):  # type: ignore
    """
    Generates responses from a model for given prompts and logs detailed information,
    using the provided advanced math grader for reward calculation.
    """
    model.eval()
    model.to(device) # pyright: ignore[reportArgumentType]
    log_data = []

    for prompt_text, ground_truth_text in zip(prompts, ground_truths):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            ) # pyright: ignore[reportCallIssue]

        generated_ids = outputs.sequences[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 4. Reward information using the new math grader
        reward_info = r1_zero_reward_fn(generated_text, ground_truth_text)

        # 5. Average token entropy of the response
        response_logits = torch.stack(outputs.scores, dim=1).squeeze(0)
        avg_entropy = compute_entropy(response_logits)

        # 6. Response length information
        response_length = len(generated_ids)

        log_data.append({
            "input_prompt": prompt_text,
            "generated_response": generated_text,
            "ground_truth": ground_truth_text,
            "format_reward": reward_info["format_reward"],
            "answer_reward": reward_info["answer_reward"],
            "total_reward": reward_info["reward"],
            "avg_token_entropy": avg_entropy,
            "response_length": response_length,
        })

    log_df = pd.DataFrame(log_data)

    # Calculate aggregate statistics
    avg_response_length = log_df['response_length'].mean()
    correct_mask = log_df['answer_reward'] == 1.0
    avg_len_correct = log_df[correct_mask]['response_length'].mean()
    avg_len_incorrect = log_df[~correct_mask]['response_length'].mean()

    print("--- Generation Log Summary ---")
    print(f"Average response length (all): {avg_response_length:.2f} tokens")
    print(f"Average response length (correct): {avg_len_correct:.2f} tokens")
    print(f"Average response length (incorrect): {avg_len_incorrect:.2f} tokens")
    print(f"Overall Accuracy: {log_df['answer_reward'].mean():.2%}")
    print("------------------------------")

    return log_df


def init_vllm(
    model_id: str,
    seed: int,
    gpu_memory_utilization: float = 0.30,
    vllm_cuda_visible_devices: str = "1",
    tokenizer_id: str | None = None
):
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_cuda_visible_devices)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    try:
        with world_size_patch, profiling_patch:
            llm = LLM(
                model=model_id,
                tokenizer=(tokenizer_id or model_id),
                dtype=torch.bfloat16,
                enable_prefix_caching=True,
                gpu_memory_utilization=gpu_memory_utilization,
                seed=seed,
                tensor_parallel_size=1,
            )
    finally:
        if prev_cvd is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
    return llm


def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    llm: LLM | None,
    vllm_cuda_visible_devices: str = "1",
    tokenizer_id: str | None = None,
    gpu_mem_util: float = 0.30,
):
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="_tmp_vllm_eval_ckpt_")
    policy.save_pretrained(tmp_dir)

    # hard-stop the previous engine to free GPU:1
    try:
        if llm is not None and hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception as e:
        print(f"Warning: vLLM shutdown raised: {e}")

    # re-create on GPU:1 with smaller memory request
    new_llm = init_vllm(
        model_id=tmp_dir,
        seed=42,
        vllm_cuda_visible_devices=vllm_cuda_visible_devices,
        tokenizer_id=(tokenizer_id or policy.name_or_path),
        gpu_memory_utilization=gpu_mem_util,
    )
    return new_llm


class SFTDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def load_data_from_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def format_gsm8k_for_sft(dataset):
    return [{"prompt": ex["question"], "response": ex["answer"]} for ex in dataset]


def collate_fn(batch, tokenizer):
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    return tokenize_prompt_and_output(prompts, responses, tokenizer)


def evaluate(policy_model, vllm_instance, tokenizer, validation_data, eval_batch_size):
    """
    Evaluate with vLLM on a separate GPU. If vLLM can't start due to low free VRAM
    or any other error, fall back to HF generate on the policy_model's device.
    """

    # ---- local HF fallback (no vLLM) ----
    def evaluate_hf(policy_model, tokenizer, validation_data, device=None, max_new_tokens=1024):
        if device is None:
            device = str(policy_model.device)
        policy_model.eval()
        acc = 0
        with torch.no_grad():
            for ex in validation_data:
                prompt = make_r1_zero_prompt(ex['prompt'])
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_len = inputs.input_ids.shape[1]
                out_ids = policy_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0][input_len:]
                text = tokenizer.decode(out_ids, skip_special_tokens=True)
                reward = r1_zero_reward_fn(
                    normalize_r1_zero_format(text),
                    extract_gold_answer(ex['response'])
                )
                acc += int(reward.get("answer_reward", 0.0) > 0.5)
        return acc / len(validation_data) if validation_data else 0.0

    # ---- try vLLM path ----
    VLLM_GPU = "1"          # pin eval to GPU:1
    VLLM_MEM = 0.30         # request a smaller fraction to avoid startup failure

    was_training = policy_model.training
    policy_model.eval()

    try:
        # Always (re)create a fresh vLLM instance with current weights.
        # This function should save policy weights to a temp dir, shutdown the old engine,
        # and init a new engine on GPU:1 with tokenizer from the base model path.
        llm = load_policy_into_vllm_instance(
            policy=policy_model,
            llm=vllm_instance,                      # can be None; function will handle it
            vllm_cuda_visible_devices=VLLM_GPU,
            tokenizer_id=tokenizer.name_or_path,    # tokenizer lives at your base model path
            gpu_mem_util=VLLM_MEM,
        )

        # Build eval prompts/labels
        prompts = [make_r1_zero_prompt(ex['prompt']) for ex in validation_data]
        gts = [extract_gold_answer(ex['response']) for ex in validation_data]

        params = SamplingParams(
            temperature=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

        try:
            outputs = llm.generate(prompts, params)
            correct = 0
            for out, gold in zip(outputs, gts):
                text = normalize_r1_zero_format(out.outputs[0].text)
                reward = r1_zero_reward_fn(text, gold)
                correct += int(reward.get("answer_reward", 0.0) > 0.5)
            acc = correct / len(prompts) if prompts else 0.0
        finally:
            # IMPORTANT: free GPU:1 immediately after eval
            try:
                llm.shutdown()
            except Exception as _e:
                print(f"Warning: vLLM shutdown raised: {_e}")
            llm = None

        return acc

    except Exception as e:
        print(f"vLLM eval failed ({e}); falling back to HF generate on {policy_model.device}.")
        return evaluate_hf(policy_model, tokenizer, validation_data, device=str(policy_model.device))

    finally:
        if was_training:
            policy_model.train()


def run_sft_experiment(args):
    """Main function to run a single SFT experiment."""

    # --- CHANGE 1: Setup for CSV Logging instead of wandb ---
    run_name = f"sft_size_{args.num_examples}_lr_{args.lr}_bs_{args.batch_size}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_file_path = f"./{run_name}_logs.csv"
    log_fieldnames = ['train_step', 'train_loss', 'eval_step', 'eval_accuracy']

    # Write the header of the CSV file
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fieldnames)
        writer.writeheader()

    print(f"Logging results to {log_file_path}")

    # 1. LOAD MODEL AND TOKENIZER
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device) # pyright: ignore[reportArgumentType]

    # 2. INITIALIZE VLLM FOR EVALUATION
    # eval_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    # vllm_instance = init_vllm(args.model_id, seed=42,
    #                       vllm_cuda_visible_devices="1",
    #                       tokenizer_id=args.model_id)
    vllm_instance = None

    # 3. LOAD AND PREPARE DATA
    train_data_raw = load_data_from_jsonl(args.train_file_path)
    validation_data_raw = load_data_from_jsonl(args.test_file_path)
    full_train_data = format_gsm8k_for_sft(train_data_raw)
    validation_data = format_gsm8k_for_sft(validation_data_raw)

    random.seed(42)
    random.shuffle(full_train_data)
    num_train_examples = len(full_train_data) if args.num_examples == -1 else args.num_examples
    train_subset = full_train_data[:num_train_examples]

    train_dataset = SFTDataset(train_subset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer), num_workers=4
    )

    # 4. SETUP OPTIMIZER
    optimizer = AdamW(policy_model.parameters(), lr=args.lr)

    # 5. TRAINING LOOP
    global_step = 0
    best_accuracy = -1.0
    for epoch in range(args.num_epochs):
        policy_model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = policy_model(input_ids=batch['input_ids']).logits

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
            per_token_loss = per_token_loss.view(logits.size(0), -1)
            masked_loss = per_token_loss * batch['response_mask']
            loss = masked_loss.sum() / batch['response_mask'].sum()

            scaled_loss = loss / args.grad_accumulation_steps
            scaled_loss.backward()

            if (step + 1) % args.grad_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # --- CHANGE 2: Log training loss to CSV ---
                with open(log_file_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=log_fieldnames)
                    writer.writerow({'train_step': global_step, 'train_loss': loss.item()})

        # 6. EVALUATION at the end of each epoch
        print(f"--- Finished Epoch {epoch+1}, starting evaluation... ---")
        accuracy = evaluate(policy_model, vllm_instance, tokenizer, validation_data, args.eval_batch_size)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # --- CHANGE 3: Log evaluation accuracy to CSV ---
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fieldnames)
            writer.writerow({'eval_step': epoch + 1, 'eval_accuracy': accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            output_dir = f"./sft_model_best_{run_name}"
            print(f"New best accuracy! Saving model to {output_dir}")
            policy_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT experiment on GSM8K with CSV logging.")
    parser.add_argument("--model_id", type=str, required=True, help="Local path to the base model.")
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to your train.jsonl file.")
    parser.add_argument("--test_file_path", type=str, required=True, help="Path to your test.jsonl file.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=128)

    args = parser.parse_args()

    dataset_sizes = [128, 256, 512, 1024, -1]
    for size in dataset_sizes:
        print(f"\n{'='*20} RUNNING SFT FOR {size if size != -1 else 'ALL'} EXAMPLES {'='*20}\n")
        args.num_examples = size
        args.num_epochs = 3 if size in [128, 256] else 1
        run_sft_experiment(args)

    print(f"\n{'='*20} RUNNING SFT FOR FILTERED (FULL) DATASET {'='*20}\n")
    args.num_examples = -1
    args.num_epochs = 1
    run_sft_experiment(args)

   # CUDA_VISIBLE_DEVICES=0,1 python -m cs336_alignment.sft     --model_id "/home/zhangwj/Qwen2.5-Math-1.5B"     --train_file_path "/home/zhangwj/assignment5/data/gsm8k/train.jsonl"     --test_file_path "/home/zhangwj/assignment5/data/gsm8k/test.jsonl"
