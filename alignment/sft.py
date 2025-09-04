import gc
import math
import os
import json
import random
import argparse
import time
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

# ==============================================================================
def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes prompt and output separately, concatenates, pads to the same length,
    and returns:
      - input_ids: (B, max_len-1)
      - labels:     (B, max_len-1) = shifted input_ids
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
            out["token_entropy"] = compute_entropy(logits) 
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
    """
    QAs_test = []
    with open(test_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            q_text, ans_raw = ex.get("question"), ex.get("answer")
            parts = ans_raw.split("####")
            # Store the original question as the prompt for evaluation
            QAs_test.append({"Q": q_text, "R": parts[0].strip(), "A": parts[-1].strip()})
    
    if len(QAs_test) == 0:
        return 0.0, 0.0

    # Use the original questions as prompts
    prompts = [ex["Q"] for ex in QAs_test]
    # Use the final answer part for ground truth
    gts = [ex["A"] for ex in QAs_test]
    
    sum_reward, sum_format = 0.0, 0.0
    N = len(prompts)

    for pc_chunk, gt_chunk in zip(chunked(prompts, eval_batch_size), chunked(gts, eval_batch_size)):
        outs = vllm_gen.generate(pc_chunk, sampling_params_eval)
        for out, gt in zip(outs, gt_chunk):
            text = out.outputs[0].text
            # The format check might be part of the problem.
            # `normalize_r1_zero_format` might be too strict.
            norm_text = normalize_r1_zero_format(text)
            res = r1_zero_reward_fn(norm_text, gt)
            sum_reward += float(res.get("reward", 0.0))
            sum_format += float(res.get("format_reward", 0.0))

    return sum_reward / N, sum_format / N

def make_r1_zero_format(QAs):
    prompts, output = [], []
    for QA in QAs:
        prompts.append("A conversation between User and Assistant. ... <think>") # Prompts content remains the same
        output.append(f"{QA['R']}</think> <answer>{QA['A']}</answer>.\n")
    return prompts, output


def update_and_save_plots(results_data: Dict, output_dir: str):
    if not results_data:
        print("No data to plot yet.")
        return

    # 1. 绘制并保存总的对比图
    plt.figure(figsize=(12, 7))
    for run_label, data in results_data.items():
        if data.get("steps") and data.get("accuracy"):
            plt.plot(data["steps"], data["accuracy"], marker='o', linestyle='-', label=run_label)
    
    plt.title("Comparison: Validation Accuracy vs. Training Steps")
    plt.xlabel("Global Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    # 动态调整Y轴范围，使其更美观
    if any(d.get('accuracy') for d in results_data.values()):
        max_acc = max(max(d['accuracy']) for d in results_data.values() if d.get('accuracy'))
        plt.ylim(0, max(0.2, max_acc * 1.2))

    comparison_plot_path = os.path.join(output_dir, "sft_experiments_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"\n📊 Comparison plot updated and saved to: {comparison_plot_path}")

    # 2. 绘制并保存最新一次实验的单独图表
    latest_run_label = list(results_data.keys())[-1]
    latest_data = results_data[latest_run_label]
    if latest_data.get("steps") and latest_data.get("accuracy"):
        plt.figure(figsize=(10, 6))
        plt.plot(latest_data["steps"], latest_data["accuracy"], marker='o', linestyle='-')
        plt.title(f"Run Details: {latest_run_label}")
        plt.xlabel("Global Training Steps")
        plt.ylabel("Validation Accuracy")
        plt.grid(True)
        if latest_data['accuracy']:
            plt.ylim(0, max(0.2, max(latest_data['accuracy']) * 1.2))
        
        individual_plot_path = os.path.join(output_dir, f"sft_experiment_{latest_run_label.replace('=', '_')}.png")
        plt.savefig(individual_plot_path)
        plt.close()
        print(f"📈 Individual plot for '{latest_run_label}' saved to: {individual_plot_path}")

def append_jsonl(path: str, record: Dict):
    """将一条记录以 JSONL 形式追加写入"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_training_loss_plot(steps: List[int], losses: List[float], output_path: str):
    """根据内存中的训练 loss 轨迹，实时出图到文件"""
    if not steps:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linestyle='-')
    plt.title("Training Loss (live)")
    plt.xlabel("Global Training Steps")
    plt.ylabel("Avg Loss (per optimizer step)")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT experiments with hyperparameter tuning.")
    # --- 参数定义 ---
    parser.add_argument("--model_path", type=str, default="/data/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_path", type=str, default="/home/zhangwj/GRPO/data/gsm8k/train.jsonl")
    parser.add_argument("--test_path", type=str, default="/home/zhangwj/GRPO/data/gsm8k/test.jsonl")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--train_device_id", type=int, default=0)
    parser.add_argument("--gen_device_id", type=int, default=1)
    
    args = parser.parse_args()

    # --- 目录和日志文件设置 ---
    RUN_NAME = f"sft_gsm8k_lr{args.learning_rate}"
    run_dir = os.path.abspath(f"./{RUN_NAME}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 使用持久化的JSON文件来记录所有实验结果 ✨
    results_log_path = os.path.join(run_dir, "all_results.json")
    all_results = {}
    if os.path.exists(results_log_path):
        try:
            with open(results_log_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"✅ Resumed results from: {results_log_path}")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not parse {results_log_path}. Starting fresh.")
            all_results = {}
    
    # --- 实验设置 ---
    dataset_sizes = [128, 256, 512, 1024, -1] # -1代表全量数据

    # --- 数据预加载 ---
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
        num_examples = len(prompt_output_pairs_full) if size == -1 else size
        run_label = f"n={num_examples}" 
                # === 每个实验单独的训练日志(JSONL)与实时loss缓存 ===
        train_log_path = os.path.join(run_dir, f"{run_label}_train_log.jsonl")
        # 若希望每次重跑都清空该 run 的日志，可取消下一行的注释
        # if os.path.exists(train_log_path): os.remove(train_log_path)
        live_loss_steps: List[int] = []
        live_loss_values: List[float] = []

        # ✨ 新增：断点续跑功能，如果某个实验已完成，则跳过 ✨
        if run_label in all_results:
            print(f"\n{'='*25}\n⏭️ Skipping Experiment: {run_label} (already completed)\n{'='*25}")
            continue
        
        print(f"\n{'='*25}\n🚀 Starting Experiment: {run_label}\n{'='*25}")

        torch.cuda.set_device(args.train_device_id)
        device = torch.device(f"cuda:{args.train_device_id}")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        ).to(device) # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        # --- 准备当前实验数据 ---
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
        num_evals_per_epoch = 5
        total_batches_per_epoch = len(train_dataloader)
        total_global_steps_per_epoch = total_batches_per_epoch // args.gradient_accumulation_steps
        dynamic_eval_steps = max(1, total_global_steps_per_epoch // num_evals_per_epoch)
        print(f"Dynamic evaluation every {dynamic_eval_steps} global steps.")

        # --- 训练循环 ---
        global_step = 0
        best_accuracy = -1.0
        run_eval_steps, run_eval_accs, train_losses = [], [], []
        vllm_gen, _tmp_dir = None, None
        
        try:
            # --- 训练前评估 (Step 0) ---
            print(f"\n🔄 Running initial evaluation at step 0...")
            vllm_gen, _tmp_dir = rebuild_vllm_from_policy(model, tokenizer, args.gen_device_id, 0.85)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_device_id)
            sampling_params_eval = SamplingParams(n=1, temperature=0.0, max_tokens=2048)
            acc, fmt = evaluate_with_vllm(vllm_gen, args.test_path, sampling_params_eval, args.eval_batch_size)
            print(f"  [EVAL] Step=0 | Accuracy={acc:.4f} | Format OK={fmt:.4f}")
            run_eval_steps.append(0)
            run_eval_accs.append(acc)
            append_jsonl(train_log_path, {
                "type": "eval",
                "run": run_label,
                "epoch": 0,
                "global_step": 0,
                "accuracy": acc,
                "format_accuracy": fmt,
                "time": time.time()
            })
            _tmp_results = {run_label: {"steps": run_eval_steps, "accuracy": run_eval_accs}}
            update_and_save_plots(_tmp_results, run_dir)

            for epoch in range(args.num_epochs):
                print(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
                model.train()
                optimizer.zero_grad()
                
                # ✨ 修改：使用 TQDM 显示更多信息 ✨
                pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                for step, batch in enumerate(pbar):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    logits = model(input_ids=batch['input_ids']).logits
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    policy_log_probs = torch.gather(log_probs_all, dim=-1, index=batch['labels'].unsqueeze(-1)).squeeze(-1)
                    
                    loss_detached, _ = sft_microbatch_train_step(
                        policy_log_probs,
                        batch["response_mask"],
                        args.gradient_accumulation_steps,
                        normalize_constant=batch["response_mask"].sum()
                    )
                    # 记录每个micro-batch的loss，用于后续平均
                    train_losses.append(loss_detached.item() * args.gradient_accumulation_steps)

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # ✨ 修改：计算并显示平均loss ✨
                        avg_loss = sum(train_losses) / len(train_losses) if train_losses else 0
                        append_jsonl(train_log_path, {
                            "type": "train",
                            "run": run_label,
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "avg_loss": avg_loss,
                            "time": time.time()
                        })

                        # 更新内存中的loss轨迹并立即出图（覆盖保存同名文件以达到“实时”效果）
                        live_loss_steps.append(global_step)
                        live_loss_values.append(avg_loss)
                        loss_plot_path = os.path.join(run_dir, f"live_loss_{run_label.replace('=', '_')}.png")
                        update_training_loss_plot(live_loss_steps, live_loss_values, loss_plot_path)
                        pbar.set_description(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}")
                        train_losses = [] # 重置loss列表

                        if global_step > 0 and global_step % dynamic_eval_steps == 0:
                            print(f"\n🔄 Running evaluation at global step {global_step}...")

                            if vllm_gen is not None: del vllm_gen
                            if _tmp_dir and os.path.exists(_tmp_dir): shutil.rmtree(_tmp_dir, ignore_errors=True)
                            torch.cuda.empty_cache()

                            vllm_gen, _tmp_dir = rebuild_vllm_from_policy(model, tokenizer, args.gen_device_id, 0.85)
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_device_id)

                            acc, fmt = evaluate_with_vllm(vllm_gen, args.test_path, sampling_params_eval, args.eval_batch_size)
                            print(f"  [EVAL] Step={global_step} | Accuracy={acc:.4f} | Format OK={fmt:.4f}")
                            run_eval_steps.append(global_step)
                            run_eval_accs.append(acc)
                            append_jsonl(train_log_path, {
                                "type": "eval",
                                "run": run_label,
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "accuracy": acc,
                                "format_accuracy": fmt,
                                "time": time.time()
                            })

                            # 只用当前run的数据，实时更新验证曲线图
                            _tmp_results = {run_label: {"steps": run_eval_steps, "accuracy": run_eval_accs}}
                            update_and_save_plots(_tmp_results, run_dir)
                            
                            if acc > best_accuracy:
                                best_accuracy = acc
                                print(f"  🎉 New best accuracy for this run: {best_accuracy:.4f}")
        
        finally:
            # 清理评估用的vLLM模型资源
            if vllm_gen is not None: del vllm_gen
            if _tmp_dir and os.path.exists(_tmp_dir): shutil.rmtree(_tmp_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()

        # ✨ 修改：每次实验结束后，立即记录结果并更新绘图 ✨
        all_results[run_label] = {"steps": run_eval_steps, "accuracy": run_eval_accs}
        
        print(f"\n💾 Saving results for '{run_label}' to {results_log_path}...")
        with open(results_log_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        
        print("🎨 Updating plots with the latest data...")
        update_and_save_plots(all_results, run_dir)

    print(f"\n\n{'='*25}\n✅ All experiments finished! Final results are in '{run_dir}'.\n{'='*25}")