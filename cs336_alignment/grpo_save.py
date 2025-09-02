import os
import json
import random
import gc
import typer
from typing import List, Dict, Tuple, Literal
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

# 确保这些辅助模块在你的Python路径中
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.gsm8k import (
    make_r1_zero_prompt,
    extract_gold_answer,
    normalize_r1_zero_format,
)




class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class RolloutDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def load_data(path: str) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def safe_shutdown_vllm(llm):
    if llm is None: return
    try:
        if hasattr(llm, "shutdown") and callable(llm.shutdown):
            llm.shutdown()
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
    except Exception as e:
        print(f"Warning: could not cleanly shutdown vLLM: {e}")
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

@torch.no_grad()
def get_log_probs(model, tokenizer, prompts, responses, device):
    """计算模型对给定响应的每个序列的对数概率。"""
    log_probs_list = []
    masks_list = []
    
    for prompt, response in zip(prompts, responses):
        full_text = prompt + response
        tokenized = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized.input_ids.to(device)
        
        # 创建 labels
        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokenized.input_ids)
        
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100 # Mask prompt part
        
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        # 获取 response 部分的 log_probs
        response_logits = logits[:, prompt_len-1:-1, :]
        response_ids = input_ids[:, prompt_len:]
        
        log_probs_all = torch.log_softmax(response_logits, dim=-1)
        log_probs_gathered = torch.gather(log_probs_all, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
        
        # 创建 response mask
        response_mask = (response_ids != tokenizer.pad_token_id).float()
        
        log_probs_list.append(log_probs_gathered.squeeze(0)) # Remove batch dim
        masks_list.append(response_mask.squeeze(0))
    
    # Pad to the same length for batching
    padded_log_probs = torch.nn.utils.rnn.pad_sequence(log_probs_list, batch_first=True, padding_value=0.0)
    padded_masks = torch.nn.utils.rnn.pad_sequence(masks_list, batch_first=True, padding_value=0.0)

    return padded_log_probs, padded_masks

@torch.no_grad()
def evaluate_on_validation(policy_model, tokenizer, val_data, group_size, device):
    """在验证集上进行评估，返回平均奖励。"""
    policy_model.eval()
    prompts = [make_r1_zero_prompt(ex['question']) for ex in val_data]
    ground_truths = [extract_gold_answer(ex['answer']) for ex in val_data]

    # 使用确定性解码进行评估
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    outputs = policy_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False, # 确定性解码
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 计算奖励
    _, raw_rewards, _ = compute_group_normalized_rewards(
        reward_fn=lambda r, gt: r1_zero_reward_fn(normalize_r1_zero_format(r), gt, fast=True),
        rollout_responses=responses,
        repeated_ground_truths=ground_truths,
        group_size=len(val_data), # Treat the whole val batch as one group
        advantage_eps=1e-6,
        normalize_by_std=False,
    )
    
    return raw_rewards.mean().item()


# ============================================================
#                      主训练循环
# ============================================================

def grpo_train_loop(
    # --- 超参数 ---
    model_id: str = "/home/zhangwj/Qwen2.5-Math-1.5B",
    train_data_path: str = "/home/zhangwj/assignment5/data/gsm8k/train.jsonl",
    val_data_path: str = "/home/zhangwj/assignment5/data/gsm8k/test.jsonl",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 1,
    loss_type: str = "grpo_clip",
    cliprange: float = 0.2,
    advantage_eps: float = 1e-6,
    use_std_normalization: bool = True,
    eval_interval: int = 10,
):
    """完整的GRPO训练、日志和评估循环。"""
    
    # 【修改】初始化一个列表来收集所有日志
    all_logs = []
    
    # 1. 初始化
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device) # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = AdamW(policy_model.parameters(), lr=learning_rate)
    
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    prompt_dataset = PromptDataset(train_data)
    prompt_dataloader = DataLoader(prompt_dataset, batch_size=rollout_batch_size, shuffle=True)

    assert train_batch_size % gradient_accumulation_steps == 0, \
    "train_batch_size must be divisible by gradient_accumulation_steps"
    assert rollout_batch_size % group_size == 0, \
        "rollout_batch_size must be divisible by group_size"
    
    vllm_instance = LLM(model=model_id, dtype='bfloat16', tensor_parallel_size=1)
    
    validation_rewards = []
    steps = []

    prompt_iter = iter(prompt_dataloader)

    # 2. 主训练循环
    for step in range(n_grpo_steps):
        print(f"\n{'='*20} GRPO Step {step + 1}/{n_grpo_steps} {'='*20}")
        
        # --- 阶段一: 采样/生成 (Rollout) ---
        print("--- Phase 1: Rolling out responses ---")
        policy_model.eval()
        
        try:
            batch = next(prompt_iter)
        except StopIteration:
            prompt_iter = iter(prompt_dataloader)
            batch = next(prompt_iter)

        # DataLoader 默认 collate 后得到 dict of lists
        if isinstance(batch, dict):
            questions = batch["question"]
            answers   = batch["answer"]
        else:
            # 若以后改了 collate_fn，仍可兼容样本列表
            questions = [ex["question"] for ex in batch]
            answers   = [ex["answer"]   for ex in batch]

        prompts = [make_r1_zero_prompt(q) for q in questions]
        ground_truths = [extract_gold_answer(a) for a in answers]
        
        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]
        
        sampling_params = SamplingParams(temperature=1.0, max_tokens=256, stop=["</answer>"])
        outputs = vllm_instance.generate(repeated_prompts, sampling_params)
        rollout_responses = [out.outputs[0].text for out in outputs]
        
        # --- 阶段二: 评估/奖励 (Reward) ---
        print("--- Phase 2: Computing rewards ---")
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=lambda r, gt: r1_zero_reward_fn(normalize_r1_zero_format(r), gt, fast=True),
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        mean_raw_reward = raw_rewards.mean().item()
        
        # --- 阶段三: 学习/更新 (Update) ---
        print("--- Phase 3: Updating policy ---")
        policy_model.train()
        
        # 计算旧的对数概率
        with torch.no_grad():
            old_log_probs, _ = get_log_probs(policy_model, tokenizer, repeated_prompts, rollout_responses, device)

        # 构建用于训练的数据集
        rollout_data = []
        for i in range(len(repeated_prompts)):
            rollout_data.append({
                "prompt": repeated_prompts[i],
                "response": rollout_responses[i],
                "advantage": advantages[i],
                "raw_reward": raw_rewards[i],
                "old_log_prob": old_log_probs[i],
            })
        
        rollout_dataset = RolloutDataset(rollout_data)
        update_dataloader = DataLoader(rollout_dataset, batch_size=train_batch_size, shuffle=True)
        
        for epoch in range(epochs_per_rollout_batch):
            for i, update_batch in enumerate(update_dataloader):
                
                # 计算新的对数概率
                new_log_probs, response_masks = get_log_probs(
                    policy_model, tokenizer, update_batch['prompt'], update_batch['response'], device
                )

                # 调用你的辅助函数进行训练步骤
                loss_value, metadata = grpo_microbatch_train_step(
                    policy_log_probs=new_log_probs.to(device),
                    response_mask=response_masks.to(device),
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type, # type: ignore
                    raw_rewards=update_batch['raw_reward'].to(device),
                    advantages=update_batch['advantage'].to(device),
                    old_log_probs=update_batch['old_log_prob'].to(device),
                    cliprange=cliprange,
                )

                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # >>> 新增：把张量转换为可序列化的标量
                    def _to_float(x):
                        return x.item() if torch.is_tensor(x) else float(x)

                    log_dict = {
                        "type": "train",
                        "step": step,
                        "update_step": i + 1,
                        "loss": float(loss_value.item()),
                        "mean_raw_reward": float(mean_raw_reward),
                        "clipped_fraction": _to_float(metadata.get("clipped_fraction", 0.0)),
                        "ratio": _to_float(metadata.get("ratio", 1.0)),
                    }
                    all_logs.append(log_dict)
                    print(
                        f"  Update Step {i+1}, "
                        f"Loss: {log_dict['loss']:.4f}, "
                        f"Mean Reward: {log_dict['mean_raw_reward']:.4f}, "
                        f"ClipFrac: {log_dict['clipped_fraction']:.3f}, "
                        f"Ratio: {log_dict['ratio']:.3f}"
                    )

        # --- 定期验证 ---
        if (step + 1) % eval_interval == 0:
            print(f"--- Running validation at step {step + 1} ---")
            val_reward = evaluate_on_validation(policy_model, tokenizer, val_data, group_size, device)
            validation_rewards.append(val_reward)
            steps.append(step + 1)
            print(f"Validation Reward: {val_reward:.4f}")
            
            # --- 【修改】记录验证日志 ---
            all_logs.append({
                "type": "validation",
                "step": step + 1,
                "validation_reward": val_reward,
            })

    # --- 训练结束，清理和保存 ---
    print("\nTraining finished. Shutting down VLLM and saving results.")
    safe_shutdown_vllm(vllm_instance)
    
    # 保存最终模型
    policy_model.save_pretrained(f"./grpo_qwen_final_model")
    tokenizer.save_pretrained(f"./grpo_qwen_final_model")
    
    # 【修改】将所有日志写入JSON文件
    logs_file = "grpo_training_logs.json"
    with open(logs_file, "w") as f:
        json.dump(all_logs, f, indent=4)
    print(f"Full training logs saved to {logs_file}")

    # 绘制并保存奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, validation_rewards, marker='o')
    plt.title("Validation Reward vs. GRPO Steps")
    plt.xlabel("GRPO Step")
    plt.ylabel("Average Validation Reward")
    plt.grid(True)
    plt.savefig("grpo_validation_rewards.png")
    print("Validation rewards plot saved to grpo_validation_rewards.png")

if __name__ == "__main__":
    typer.run(grpo_train_loop)