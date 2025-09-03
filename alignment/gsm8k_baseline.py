import re
import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from alignment.drgrpo_grader import r1_zero_reward_fn

# ---------------------------
# Load GSM8K (local parquet if available, else HF Hub)
# ---------------------------
def load_gsm8k(
    split: str = "test",
    prefer_local_root: str = "/home/zhangwj/gsm8k",  # <-- change if your local path differs
    config: str = "main",
) -> Dataset:
    root = Path(prefer_local_root)
    local_parquet = root / config / f"{split}-00000-of-00001.parquet"
    if local_parquet.exists():
        ds = load_dataset("parquet", data_files={split: str(local_parquet)})[split]
        return ds # type: ignore
    return load_dataset("openai/gsm8k", config, split=split) # type: ignore


# ---------------------------
# r1_zero prompt (NO gold answer)
# ---------------------------
def make_r1_zero_prompt(question: str) -> str:
    return (
        "A conversation between User and Assistant. The User asks a question, "
        "and the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in the mind and then provides the User with thdenglue answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed "
        "within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n"
        f"User: {question.strip()}\n"
        "Assistant:"
    )


# ---------------------------
# Extract gold answer from GSM8K "answer"
# ---------------------------
FINAL_ANS_RE = re.compile(r"####\s*(.*)")

def extract_gold_answer(answer_field: str) -> str:
    m = FINAL_ANS_RE.search(answer_field.strip())
    return m.group(1).strip() if m else answer_field.strip()


# ---------------------------
# tiny normalization so the format check passes
# (reward_fn requires the exact substring '</think> <answer>')
# ---------------------------
def normalize_r1_zero_format(generated: str) -> str:
    # Turn '</think>\n<answer>' or multiple spaces into '</think> <answer>'
    gen = generated.replace("</think>\n<answer>", "</think> <answer>")
    gen = re.sub(r"</think>\s+<answer>", "</think> <answer>", gen)
    return gen


# ---------------------------
# vLLM evaluation loop using r1_zero_reward_fn
# ---------------------------
def evaluate_vllm_gsm8k(
    llm: LLM,
    prompts: List[str],
    gold_answers: List[str],
    sampling_params: SamplingParams,
    save_jsonl: str = "gsm8k_eval_results.jsonl",
    summary_json: str = "gsm8k_eval_summary.json",
) -> Dict[str, float]:
    outputs = llm.generate(prompts, sampling_params)

    n = len(prompts)
    fmt_ok = 0
    correct = 0
    rewards_sum = 0.0

    with open(save_jsonl, "w", encoding="utf-8") as f:
        for i, (prompt, out, gold) in enumerate(zip(prompts, outputs, gold_answers)):
            gen_text = out.outputs[0].text
            response = normalize_r1_zero_format(gen_text)
            metrics = r1_zero_reward_fn(response=response, ground_truth=gold, fast=True)
            fmt_ok += int(metrics.get("format_reward", 0.0) > 0.5)
            correct += int(metrics.get("answer_reward", 0.0) > 0.5)
            rewards_sum += float(metrics.get("reward", 0.0))
            row = {
                "id": i,
                "prompt": prompt,
                "generation": gen_text,     # raw generation from the model
                "response_for_grader": response,  # after tiny normalization
                "gold_answer": gold,
                "metrics": metrics,         # format_reward / answer_reward / reward
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_examples": n,
        "format_rate": fmt_ok / max(n, 1),
        "accuracy": correct / max(n, 1),
        "avg_reward": rewards_sum / max(n, 1),
        "results_path": save_jsonl,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    # 1) Load GSM8K (use "train" if you prefer)
    split = "test"
    ds = load_gsm8k(split=split, prefer_local_root="/home/zhangwj/gsm8k", config="main")

    questions = [ex["question"] for ex in ds] # type: ignore
    golds = [extract_gold_answer(ex["answer"]) for ex in ds] # type: ignore
    prompts = [make_r1_zero_prompt(q) for q in questions]

    # 2) vLLM model + generation settings
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],                
        include_stop_str_in_output=True,   
    )

    llm = LLM(model="/data/Qwen2.5-Math-1.5B")  

    summary = evaluate_vllm_gsm8k(
        llm,
        prompts,
        gold_answers=golds,
        sampling_params=sampling_params,
        save_jsonl="gsm8k_eval_results.jsonl",
        summary_json="gsm8k_eval_summary.json",
    )

    print("Summary:", summary)
    print("Saved per-example to gsm8k_eval_results.jsonl and summary to gsm8k_eval_summary.json")
