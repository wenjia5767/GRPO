import gc
import math
import os
import json
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import shutil
import uuid
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from tqdm import tqdm
from vllm import LLM, SamplingParams
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.gsm8k_baseline import (
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
    prompt_tokens_list = tokenizer(prompt_strs, add_special_tokens=False).input_ids
    output_tokens_list = tokenizer(output_strs, add_special_tokens=False).input_ids

    combined_ids_list = [p + o for p, o in zip(prompt_tokens_list, output_tokens_list)]

    max_len = max(len(ids) for ids in combined_ids_list)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_combined_ids = [ids + [pad_token_id] * (max_len - len(ids)) for ids in combined_ids_list]
    combined_ids = torch.tensor(padded_combined_ids, dtype=torch.long)

    response_mask_full = torch.zeros_like(combined_ids, dtype=torch.long)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_tokens_list, output_tokens_list)):
        p_len, o_len = len(p_ids), len(o_ids)
        response_mask_full[i, p_len:p_len + o_len] = 1

    input_ids = combined_ids[:, :-1]
    labels = combined_ids[:, 1:]
    response_mask = response_mask_full[:, 1:] 

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
    input_ids: torch.Tensor, 
    labels: torch.Tensor,    
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    """
    Return per-token conditional log-probabilities log p(x_t | x_<t) and,
    optionally, the per-token entropy of the model's next-token distribution.
    Shapes in/out all follow the spec: (batch, seq_len).
    """
    
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs_all = torch.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        out: Dict[str, torch.Tensor] = {"log_probs": log_probs}
        if return_token_entropy:
            out["token_entropy"] = compute_entropy(logits)  # (batch, seq_len)
    if was_training:
        model.train()
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
    masked_tensor = tensor * mask
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)
    normalized = summed / normalize_constant
    return normalized


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor, 
    response_mask: torch.Tensor,   
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    nll_loss_per_token = -policy_log_probs

    normalized_sum = masked_normalize(
        tensor=nll_loss_per_token,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )
    batch_size = policy_log_probs.shape[0]
    loss = normalized_sum / (batch_size * gradient_accumulation_steps)
    loss.backward()
    metadata = {}
    return loss.detach(), metadata


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
    test_path,
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
            parts = ans_raw.split("####")
            reasoning_text = parts[0].strip()
            a_text = parts[-1].strip()
            QAs_test.append({"Q": q_text, "R": reasoning_text, "A": a_text})
        
    prompts_o, output_o = make_r1_zero_format(QAs_test)

    if len(QAs_test) == 0:
        return 0.0, 0.0

    prompts = prompts_o
    gts = [ex["A"] for ex in QAs_test]

    sum_reward = 0.0
    sum_format = 0.0
    N = len(prompts)

    for pc_chunk, gt_chunk in zip(
        chunked(prompts, eval_batch_size), chunked(gts, eval_batch_size)
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

def make_r1_zero_format(QAs):
    prompts = []
    output = []
    for QA in QAs:
        prompts.append("A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        f"User: {QA["Q"]}\n"
        "Assistant: <think>")
        output.append(f"{QA["R"]}</think> <answer>{QA["A"]}</answer>.\n")
    return prompts, output


# --- Configuration & File Paths ---
# Run dir & artifact paths
RUN_NAME = os.environ.get("RUN_NAME", "sft_gsm8k")
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT experiments with hyperparameter tuning.")
    # --- 参数定义 (保持不变) ---
    parser.add_argument("--model_path", type=str, default="/data/Qwen2.5-Math-1.5B", help="Local path to the base model.")
    parser.add_argument("--train_path", type=str, default="/home/zhangwj/GRPO/data/gsm8k/train.jsonl", help="Path to train.jsonl.")
    parser.add_argument("--test_path", type=str, default="/home/zhangwj/GRPO/data/gsm8k/test.jsonl", help="Path to test.jsonl.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for AdamW optimizer.")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs for each run.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--train_device_id", type=int, default=0, help="GPU device ID for training.")
    parser.add_argument("--gen_device_id", type=int, default=1, help="GPU device ID for vLLM evaluation.")
    
    args = parser.parse_args()

    # --- 实验设置 ---
    dataset_sizes = [128, 256, 512, 1024, -1]
    all_results = {} 

    # --- 数据加载 (只加载一次) ---
    print("Pre-loading and formatting datasets...")
    QAs_train_full = []
    with open(args.train_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            q_text, ans_raw = ex.get("question"), ex.get("answer")
            parts = ans_raw.split("####")
            QAs_train_full.append({"Q": q_text, "R": parts[0].strip(), "A": parts[-1].strip()})
    
    prompts_o_full, output_o_full = make_r1_zero_format(QAs_train_full)
    prompt_output_pairs_full = [{"P": a, "O": b} for a, b in zip(prompts_o_full, output_o_full)]

    # --- 自动化实验循环 ---
    for size in dataset_sizes:
        run_name = f"sft_size-{size if size != -1 else 'full'}_lr-{args.learning_rate}_bs-{args.batch_size}"
        print(f"\n{'='*25}\n🚀 Starting Experiment: {run_name}\n{'='*25}")

        torch.cuda.set_device(args.train_device_id)
        device = torch.device(f"cuda:{args.train_device_id}")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        ).to(device) # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        # --- 为当前实验准备数据 ---
        num_examples = len(prompt_output_pairs_full) if size == -1 else size
        random.shuffle(prompt_output_pairs_full)
        train_subset = prompt_output_pairs_full[:num_examples]

        train_dataset = [{"prompt": ex["P"], "output": ex["O"]} for ex in train_subset]
        train_dataloader = DataLoader(
            train_dataset, # type: ignore
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: tokenize_prompt_and_output([item['prompt'] for item in b], [item['output'] for item in b], tokenizer)
        )
        
        # --- 动态计算评估间隔 ---
        # 目标是在每个 epoch 中评估大约5次
        num_evals_per_epoch = 5
        total_batches_per_epoch = len(train_dataloader)
        total_global_steps_per_epoch = total_batches_per_epoch // args.gradient_accumulation_steps
        dynamic_eval_steps = max(1, total_global_steps_per_epoch // num_evals_per_epoch)
        print(f"Dynamic evaluation every {dynamic_eval_steps} global steps.")

        # --- 训练循环 ---
        global_step = 0
        best_accuracy = -1.0
        run_eval_steps = []
        run_eval_accs = []
        vllm_gen, _tmp_dir = None, None
        
        try:
            # --- 在训练前进行一次评估 (第0步) ---
            print(f"\n🔄 Running initial evaluation at step 0 (before training)...")
            vllm_gen, _tmp_dir = rebuild_vllm_from_policy(model, tokenizer, args.gen_device_id, 0.85)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_device_id)
            sampling_params_eval = SamplingParams(n=1, temperature=0.0, max_tokens=2048)
            acc, fmt = evaluate_with_vllm(vllm_gen, args.test_path, sampling_params_eval, args.eval_batch_size)
            print(f"[EVAL] Step=0 | Accuracy={acc:.4f} | Format OK={fmt:.4f}")
            run_eval_steps.append(0)
            run_eval_accs.append(acc)

            for epoch in range(args.num_epochs):
                print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
                model.train()
                optimizer.zero_grad()
                
                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    logits = model(input_ids=batch['input_ids']).logits
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    policy_log_probs = torch.gather(log_probs_all, dim=-1, index=batch['labels'].unsqueeze(-1)).squeeze(-1)
                    
                    loss, _ = sft_microbatch_train_step(
                        policy_log_probs,
                        batch["response_mask"],
                        args.gradient_accumulation_steps,
                        normalize_constant=batch["response_mask"].sum()
                    )

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        if global_step > 0 and global_step % dynamic_eval_steps == 0:
                            print(f"\n🔄 Running evaluation at global step {global_step}...")
                            if vllm_gen is not None: del vllm_gen
                            if _tmp_dir and os.path.exists(_tmp_dir): shutil.rmtree(_tmp_dir, ignore_errors=True)
                            gc.collect()
                            torch.cuda.empty_cache()

                            vllm_gen, _tmp_dir = rebuild_vllm_from_policy(model, tokenizer, args.gen_device_id, 0.85)
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_device_id)

                            acc, fmt = evaluate_with_vllm(vllm_gen, args.test_path, sampling_params_eval, args.eval_batch_size)
                            print(f"[EVAL] Step={global_step} | Accuracy={acc:.4f} | Format OK={fmt:.4f}")
                            run_eval_steps.append(global_step)
                            run_eval_accs.append(acc)
                            
                            if acc > best_accuracy:
                                best_accuracy = acc
                                print(f"🎉 New best accuracy for this run: {best_accuracy:.4f}")

            # --- 在每个 epoch 结束后强制进行一次评估 ---
            print(f"\n🔄 Running final evaluation for Epoch {epoch+1} at global step {global_step}...") # type: ignore
            if vllm_gen is not None: del vllm_gen
            if _tmp_dir and os.path.exists(_tmp_dir): shutil.rmtree(_tmp_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()

            vllm_gen, _tmp_dir = rebuild_vllm_from_policy(model, tokenizer, args.gen_device_id, 0.85)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_device_id)

            acc, fmt = evaluate_with_vllm(vllm_gen, args.test_path, sampling_params_eval, args.eval_batch_size)
            print(f"[EVAL] Step={global_step} | Accuracy={acc:.4f} | Format OK={fmt:.4f}")
            
            # 避免重复添加最后一个点
            if not run_eval_steps or run_eval_steps[-1] != global_step:
                run_eval_steps.append(global_step)
                run_eval_accs.append(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                print(f"🎉 New best accuracy for this run: {best_accuracy:.4f}")

        finally:
            # 清理资源
            if vllm_gen is not None: del vllm_gen # type: ignore
            if _tmp_dir and os.path.exists(_tmp_dir): shutil.rmtree(_tmp_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()

        all_results[f"n={num_examples}"] = {"steps": run_eval_steps, "accuracy": run_eval_accs}

    # --- 最终绘图 (这部分代码保持不变) ---
    print(f"\n\n{'='*25}\n📈 All experiments finished. Generating plots...\n{'='*25}")
    # ... (省略与上一版相同的绘图代码)
    # 1. 为每一次实验生成单独的图表
    print("\nGenerating individual plots for each run...")
    for run_label, data in all_results.items():
        if not data["steps"] or not data["accuracy"]:
            print(f"Skipping plot for {run_label} due to no evaluation data.")
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(data["steps"], data["accuracy"], marker='o', linestyle='-', label=run_label)
        plt.title(f"Validation Accuracy vs. Training Steps for {run_label}")
        plt.xlabel("Global Training Steps")
        plt.ylabel("Validation Accuracy")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.ylim(0, max(data['accuracy']) * 1.2 if data['accuracy'] else 0.2)
        individual_plot_path = f"./sft_experiment_{run_label.replace('=', '_')}.png"
        plt.savefig(individual_plot_path)
        plt.close()
        print(f"✅ Individual plot saved to: {individual_plot_path}")
    # 2. 生成包含所有曲线的对比图表
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(10, 6))
    for run_label, data in all_results.items():
        if data["steps"] and data["accuracy"]:
            plt.plot(data["steps"], data["accuracy"], marker='o', linestyle='-', label=run_label)
    plt.title("Validation Accuracy vs. Training Steps for Different Dataset Sizes")
    plt.xlabel("Global Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.ylim(0, max(max(d['accuracy']) for d in all_results.values() if d['accuracy']) * 1.2 if any(d['accuracy'] for d in all_results.values()) else 0.2)
    comparison_plot_path = "./sft_experiments_comparison.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"\n✅ Comparison plot saved to: {comparison_plot_path}")
