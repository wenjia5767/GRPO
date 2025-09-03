import os
from typing import Any
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams # type: ignore

import pandas as pd


def parse_mmlu_response(
    model_output
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    pattern = r"\b[ABCD]\b"
    results_list = re.findall(pattern, model_output)
    if len(results_list) == 1:
        return results_list[0]
    else:
        return None
    
# write a script to evaluate Llama 3.1 8B zero-shot performance on MMLU



def load_mmlu_data(data_dir):
    """
    Loads all MMLU test CSVs from a directory into a single DataFrame.
    """
    all_data = []
    test_dir = os.path.join(data_dir, "test")

    for file_name in os.listdir(test_dir):
        if file_name.endswith('_test.csv'):
            subject = file_name.replace('_test.csv', '')
            file_path = os.path.join(test_dir, file_name)
            df = pd.read_csv(file_path, header=None)
            df['subject'] = subject
            df.columns = ['question', 'A', 'B', 'C', 'D', 'answer', 'subject']
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} examples from {len(all_data)} subjects.")
    return combined_df

def format_prompt(row):
    """
    Formats a single row into the specific MMLU prompt format requested."""
    subject_formatted = row['subject'].replace('_', " ")
    question = row['question']
    choices = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"

    prompt = (
        f"Answer the following multiple choice question about {subject_formatted}. "
        f"Respond with a single sentence of the form \"The correct answer is _\", "
        f"filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n"
        f"Question: {question}\n\n{choices}\n\nAnswer:"
    )
    return prompt

def evaluate_with_vllm(model_path, data_df):
    """
    Evaluates the model using vLLM with the specified prompt and decoding strategy.
    """
    print(f"Loading model using vLLM with the specified prompt and decoding strategy")

    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization = 0.90)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = []

    for index, row in data_df.iterrows():
        prompts.append(format_prompt(row))

    sampling_params = SamplingParams(max_tokens=15, temperature=0.0, top_p=1.0)
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")): # type: ignore
        row = data_df.iloc[i]
        generated_text = output.outputs[0].text
        predicted_answer = parse_mmlu_response(model_output=generated_text)
        ground_truth = row['answer']
        is_correct = (predicted_answer == ground_truth)

        results.append({
            'subject': row['subject'],
            'question': row['question'],
            'ground_truth': ground_truth,
            'model_generation': generated_text.strip(),
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })
    return pd.DataFrame(results)


def main():
    data_dir = "/home/zhangwj/assignment5/data/mmlu"
    model_path = "/data/Llama-3.1-8B"
    output_dir = "/home/zhangwj/assignment5/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    mmlu_df = load_mmlu_data(data_dir)
    results_df = evaluate_with_vllm(model_path, mmlu_df)
    
    overall_accuracy = results_df['is_correct'].mean()
    per_subject_accuracy = results_df.groupby('subject')['is_correct'].mean().sort_values(ascending=False)
    
    print("\n--- Evaluation Summary ---")
    print(f"📊 Overall Accuracy: {overall_accuracy:.4f}")
    print("\nAccuracy per Subject:")
    print(per_subject_accuracy)
    
    model_name = os.path.basename(model_path.rstrip('/'))
    detailed_results_path = os.path.join(output_dir, f'mmlu_detailed_results_{model_name}.csv')
    summary_path = os.path.join(output_dir, f'mmlu_summary_scores_{model_name}.txt')
    
    results_df.to_csv(detailed_results_path, index=False)
    print(f"\n💾 Detailed results saved to: {detailed_results_path}")
    
    with open(summary_path, 'w') as f:
        f.write("--- MMLU Evaluation Summary ---\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
        f.write("--- Accuracy per Subject ---\n")
        f.write(per_subject_accuracy.to_string())
        
    print(f"💾 Summary scores saved to: {summary_path}")

if __name__ == "__main__":
    main()












