import os
import json
import random
import gc  # add this import at top with the others
import argparse
from typing import List, Dict, Tuple
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from vllm import LLM, SamplingParams

# 确保这些辅助模块在你的Python路径中
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.gsm8k import (
    make_r1_zero_prompt,
    extract_gold_answer,
    normalize_r1_zero_format,
)

# ============================================================
#               Tokenization & Data Handling (已更新)
# ============================================================

def safe_shutdown_vllm(llm):
    """Shut down a vLLM engine across versions, without crashing."""
    if llm is None:
        return
    try:
        # Newer vLLM exposes LLM.shutdown()
        if hasattr(llm, "shutdown") and callable(llm.shutdown):
            llm.shutdown()
        # Older vLLM exposes LLM.llm_engine.shutdown()
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
    except Exception as e:
        print(f"Warning: could not cleanly shutdown vLLM: {e}")
    finally:
        try:
            # Best-effort GPU memory cleanup
            del llm
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

def tokenize_and_mask_for_sft(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: AutoTokenizer,
) -> Dict[str, torch.Tensor]:
    """
    使用 Hugging Face 标准的 ignore_index (-100) 来进行高效的 Tokenization 和 Masking。
    """
    # 将 prompt 和 output 拼接
    full_texts = [p + o for p, o in zip(prompt_strs, output_strs)]
    
    # 对完整文本进行分词和填充
    tokenized_full = tokenizer(
        full_texts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = tokenized_full.input_ids
    attention_mask = tokenized_full.attention_mask

    # 单独对 prompt 分词以获取其长度
    tokenized_prompts = tokenizer(
        prompt_strs,
        padding=False,
        add_special_tokens=False,
    )

    # 创建 labels，并将 prompt 部分的 token 设置为 -100
    labels = input_ids.clone()
    for i in range(len(prompt_strs)):
        prompt_len = len(tokenized_prompts.input_ids[i])
        labels[i, :prompt_len] = -100
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_and_prepare_data(file_path: str) -> List[Dict]:
    """
    加载JSONL文件，并创建结构化的学习目标。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            question = ex["question"]
            full_answer = ex["answer"]
            gold_answer = extract_gold_answer(full_answer)
            
            # 创建结构化的学习目标 (第3点要求)
            structured_target = f"<think>\n{full_answer.strip()}\n</think> <answer> {gold_answer} </answer>"
            
            data.append({
                "prompt": question,
                "full_answer": full_answer, # 保留原始答案以供评估
                "gold_answer": gold_answer,
                "target": structured_target,
            })
    return data

# ============================================================
#               vLLM & Evaluation Utilities (已更新)
# ============================================================

def init_vllm(model_id: str, tokenizer_id: str, seed: int = 42, vllm_gpu: str = "1"):
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = vllm_gpu
    # vLLM 在单GPU上运行时，有时需要 patch world_size
    with patch("torch.distributed.get_world_size", return_value=1):
        llm = LLM(
            model=model_id,
            tokenizer=tokenizer_id,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.9,
            seed=seed,
            tensor_parallel_size=1,
        )
    if prev_cvd is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
    return llm

@torch.no_grad()
def evaluate(
    policy_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    validation_data: List[Dict],
    vllm_instance: LLM | None,
) -> Tuple[float, LLM | None]:
    """
    使用vLLM进行高效、确定性的评估。
    """
    was_training = policy_model.training
    policy_model.eval()
    
    # 动态保存当前模型权重，以便vLLM加载
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy_model.save_pretrained(tmp_dir)
        
        # 如果有旧的vLLM实例，先关闭
        if vllm_instance is not None:
            safe_shutdown_vllm(vllm_instance)

        # 在GPU 1上初始化新的vLLM实例
        llm = init_vllm(model_id=tmp_dir, tokenizer_id=tokenizer.name_or_path, vllm_gpu="1")

        prompts = [make_r1_zero_prompt(ex['prompt']) for ex in validation_data]
        ground_truths = [ex['gold_answer'] for ex in validation_data]
        
        # 使用确定性解码 (第4点要求)
        params = SamplingParams(
            temperature=0,  # 0代表贪心搜索 (Greedy Search)
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        
        outputs = llm.generate(prompts, params)
        
        correct = 0
        for out, gold in zip(outputs, ground_truths):
            text = normalize_r1_zero_format(out.outputs[0].text)
            reward_info = r1_zero_reward_fn(text, gold, fast=True)
            correct += int(reward_info.get("answer_reward", 0.0) > 0.5)
        
        acc = correct / len(prompts) if prompts else 0.0

    if was_training:
        policy_model.train()
        
    # 返回准确率和新的vLLM实例（尽管我们每次都新建，但保持接口一致）
    # 在主循环中处理关闭，确保资源释放
    return acc, llm

# ============================================================
#                       Main Experiment Loop
# ============================================================

def run_sft_experiment(
    args,
    train_data: List[Dict],
    validation_data: List[Dict]
):
    """为单个数据集规模运行一次完整的SFT实验。"""

    # 1. 加载全新的预训练模型和分词器
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 2. 准备数据
    prompts = [make_r1_zero_prompt(ex['prompt']) for ex in train_data]
    targets = [ex['target'] for ex in train_data]
    
    tokenized_data = tokenize_and_mask_for_sft(prompts, targets, tokenizer)
    
    # 使用TensorDataset和DataLoader来处理批次
    dataset = torch.utils.data.TensorDataset(
        tokenized_data['input_ids'],
        tokenized_data['attention_mask'],
        tokenized_data['labels']
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. 训练循环
    model.train()
    for epoch in range(args.num_epochs):
        print(f"--- Starting Epoch {epoch + 1}/{args.num_epochs} ---")
        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # 梯度累积
            scaled_loss = loss / args.grad_accumulation_steps
            scaled_loss.backward()
            
            if (i + 1) % args.grad_accumulation_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                print(f"  Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    # 4. 评估
    print("--- Finished training, starting evaluation... ---")
    vllm_instance = None # 每次评估都新建
    try:
        accuracy, vllm_instance = evaluate(model, tokenizer, validation_data, vllm_instance)
        print(f"Validation Accuracy: {accuracy:.4f}")
    finally:
        if vllm_instance is not None:
            safe_shutdown_vllm(vllm_instance)

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT experiment on GSM8K.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID or local path.")
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to train.jsonl.")
    parser.add_argument("--test_file_path", type=str, required=True, help="Path to test.jsonl.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    args = parser.parse_args()

    # 1. 一次性加载和准备所有数据 (第3点要求)
    full_train_data = load_and_prepare_data(args.train_file_path)
    validation_data = load_and_prepare_data(args.test_file_path)
    random.seed(42)
    random.shuffle(full_train_data)

    # 2. 定义实验配置
    dataset_sizes = [128, 256, 512, 1024, 2048, 4096, len(full_train_data)]
    experiment_results = []

    # 3. 运行系列实验 (第1点要求)
    for size in dataset_sizes:
        label = str(size) if size != len(full_train_data) else "full"
        print(f"\n{'='*20} RUNNING SFT FOR {label} EXAMPLES {'='*20}\n")
        
        train_subset = full_train_data[:size]
        
        # 动态设置epoch数量
        num_epochs = 3 if size <= 256 else 1
        args.num_epochs = num_epochs
        
        accuracy = run_sft_experiment(args, train_subset, validation_data)
        
        experiment_results.append({
            "dataset_size": size,
            "num_epochs": num_epochs,
            "learning_rate": args.lr,
            "accuracy": accuracy,
        })

    # 4. 统一记录所有结果 (第2点要求)
    results_file = "sft_experiment_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=4)

    print(f"\n{'='*20} EXPERIMENT COMPLETE {'='*20}")
    print(f"All results saved to {results_file}")
    print("Results summary:")
    for res in experiment_results:
        print(f"  - Size: {res['dataset_size']:<5}, Accuracy: {res['accuracy']:.4f}")
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass



# CUDA_VISIBLE_DEVICES=0,1 python -m cs336_alignment.sft_new     --model_id "/home/zhangwj/Qwen2.5-Math-1.5B"     --train_file_path "/home/zhangwj/assignment5/data/gsm8k/train.jsonl"     --test_file_path "/home/zhangwj/assignment5/data/gsm8k/test.jsonl"