import gzip
import json
import os
import random
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def alpaca_format(instruction: str, response: str):
    system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    # CHANGE: add a blank line after system_prompt
    results = system_prompt + f"\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return results


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:  # type: ignore
    was_training, was_training_ref = lm.training, lm_ref.training
    lm.eval(); lm_ref.eval()

    al_format_chosen = alpaca_format(prompt, response_chosen)
    al_format_rejected = alpaca_format(prompt, response_rejected)

    # devices for each model
    device_lm  = next(lm.parameters()).device
    device_ref = next(lm_ref.parameters()).device

    # tokenize on CPU
    chosen_cpu = tokenizer(al_format_chosen, return_tensors="pt", add_special_tokens=False)
    reject_cpu = tokenizer(al_format_rejected, return_tensors="pt", add_special_tokens=False)

    # append EOS on CPU
    eos = tokenizer.eos_token_id
    if eos is not None:
        for ids in (chosen_cpu, reject_cpu):
            ids["input_ids"] = torch.cat([ids["input_ids"], torch.tensor([[eos]])], dim=1) # type: ignore
            ids["attention_mask"] = torch.cat([ids["attention_mask"], torch.tensor([[1]])], dim=1) # type: ignore

    # prompt length (CPU ok)
    prompt_formatted = alpaca_format(prompt, "")
    prompt_len = tokenizer(prompt_formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]  # type: ignore

    # copies on correct devices
    chosen_for_lm  = {k: v.to(device_lm)  for k, v in chosen_cpu.items()}
    reject_for_lm  = {k: v.to(device_lm)  for k, v in reject_cpu.items()}
    chosen_for_ref = {k: v.to(device_ref) for k, v in chosen_cpu.items()}
    reject_for_ref = {k: v.to(device_ref) for k, v in reject_cpu.items()}

    # forwards
    model_chosen_logits = lm(**chosen_for_lm).logits
    model_reject_logits = lm(**reject_for_lm).logits
    with torch.no_grad():  # ← new: ref model is frozen
        ref_chosen_logits = lm_ref(**chosen_for_ref).logits
        ref_reject_logits = lm_ref(**reject_for_ref).logits

    # log-probs
    model_chosen_log = torch.log_softmax(model_chosen_logits, dim=-1)
    model_reject_log = torch.log_softmax(model_reject_logits, dim=-1)
    ref_chosen_log   = torch.log_softmax(ref_chosen_logits,   dim=-1)
    ref_reject_log   = torch.log_softmax(ref_reject_logits,   dim=-1)

    # sums on their own devices
    pi_logp_yw  = model_chosen_log[:, prompt_len-1:-1, :].gather(
        -1, chosen_for_lm["input_ids"][:, prompt_len:].unsqueeze(-1)
    ).sum()
    pi_logp_yl  = model_reject_log[:, prompt_len-1:-1, :].gather(
        -1, reject_for_lm["input_ids"][:, prompt_len:].unsqueeze(-1)
    ).sum()
    ref_logp_yw = ref_chosen_log[:,   prompt_len-1:-1, :].gather(
        -1, chosen_for_ref["input_ids"][:, prompt_len:].unsqueeze(-1)
    ).sum()
    ref_logp_yl = ref_reject_log[:,   prompt_len-1:-1, :].gather(
        -1, reject_for_ref["input_ids"][:, prompt_len:].unsqueeze(-1)
    ).sum()

    # ← new: bring reference scalars onto the policy device before arithmetic
    ref_logp_yw = ref_logp_yw.to(device_lm)
    ref_logp_yl = ref_logp_yl.to(device_lm)

    left  = beta * (pi_logp_yw - ref_logp_yw)
    right = beta * (pi_logp_yl - ref_logp_yl)
    loss = -torch.nn.functional.logsigmoid(left - right)

    if was_training: lm.train()
    if was_training_ref: lm_ref.train()
    return loss




def load_hh_rlhf_for_dpo(
        base_path: str,
        subsets: Optional[List[str]] = None
) -> List[Dict[str, str]]: # type: ignore
    """
    Loads and processes the Anthropic HH-RLHF dataset for DPO training.
    
    This function reads all "train.jsonl.gz" files from the specified subsets,
    filters for single-turn conversations, and extracts the instruction,
    chosen response, and rejected response.
    
    Args:
        base_path: The root directory of the "hh-rlhf" dataset.
        subsets: A list of sub-dataset names"""
    if subsets is None:
        subsets = [
            "harmless-base",
            "helpful-base",
            "helpful-online",
            "helpful-rejection-sampled"
        ]

    dataset = []
    print(f"Loading data from the following subsets: {subsets}")

    for subset in subsets:
        file_path = os.path.join(base_path, subset, 'train.jsonl.gz')

        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}. Skipping.")
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)

                # We only want single-turn conversations.
                # A single turn means one "\n\nHuman:" prompt
                if record['chosen'].count("\n\nHuman:") != 1:
                    continue

                # The conversation always starts with "\n\nHuman: ".
                # We split the conversation by the assistant's turn.
                parts = record['chosen'].split("\n\nAssistant:")
                instruction = parts[0].replace("\n\nHuman:", "").strip()
                chosen_response = parts[1].strip()

                rejected_response = record['rejected'].split("\n\nAssistant:")[1].strip()

                dataset.append({
                    "instruction": instruction,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                })
        return dataset
    
def get_log_probs(
        lm: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        response: str
):
    """Helper function to compute the log-probability of a response given a prompt."""
    al_format = alpaca_format(prompt, response)
    device = next(lm.parameters()).device

    inputs = tokenizer(al_format, return_tensors="pt", add_special_tokens=False).to(device)

    eos = tokenizer.eos_token_id
    if eos is not None:
        inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[eos]], device=device)], dim=1) # type: ignore
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1) # type: ignore

    prompt_formatted = alpaca_format(prompt, "")
    prompt_len = tokenizer(prompt_formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1] # type: ignore
    
    with torch.no_grad():
        logits = lm(**inputs).logits
    
    log_probs = torch.log_softmax(logits, dim=-1)

    response_log_prob = log_probs[:, prompt_len-1:-1, :].gather(-1, inputs["input_ids"][:, prompt_len:].unsqueeze(-1)).sum() # type: ignore

    return response_log_prob

def run_validation(policy_model, tokenizer, validation_set):
    """
    Computes the 'classification accuracy' on the validation set.
    This is the percentage of examples where the policy model assigns a 
    higher log-probability to the chosen response than the rejected one.
    """
    policy_model.eval()
    correct_predictions = 0

    print("Running validation...")
    for item in tqdm(validation_set, desc="Validation"):
        prompt = item["instruction"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        log_prob_chosen = get_log_probs(policy_model, tokenizer, prompt, chosen)
        log_prob_rejected = get_log_probs(policy_model, tokenizer, prompt, rejected)

        if log_prob_chosen > log_prob_rejected:
            correct_predictions += 1

    accuracy = correct_predictions / len(validation_set)
    policy_model.train()
    return accuracy

def main():
    # --- 1.Hyperparameters ---\
    # These are set according to the instruction
    beta = 0.1
    learning_rate = 1e-6
    batch_size = 64
    validation_set_size = 200
    num_epochs = 1
    save_path = "dpo_llama_model.pth"
    data_dir = "/home/zhangwj/hh-rlhf"
    model_path = "/home/zhangwj/Llama-3.1-8B"
    # --- 2. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load two copies of the model as instructed
    policy_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    reference_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    

    # --- 3.Setup Models on GPUs ---
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise ConnectionAbortedError("This script requires at least 2 GPUs.")
    
    policy_model.to("cuda:0") # type: ignore
    reference_model.to("cuda:1")  # type: ignore

    for param in reference_model.parameters():
        param.requires_grad = False

    # 4. --- Load and Prepare Dataset...
    dataset = load_hh_rlhf_for_dpo(data_dir)
    random.shuffle(dataset)

    # Seperate out a validation set
    validation_set = dataset[:validation_set_size]
    train_set = dataset[validation_set_size:]

    # --- 5. Optimizer Setup ---
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=learning_rate)

    # --- 6.Training Loop ---
    step = 0
    best_val_accuracy = -1.0
    val_accuracies = []
    steps_for_val = []
    val_interval = 5

    print("Starting DPO traning...")
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        policy_model.train()

        optimizer.zero_grad()

        for i, item in enumerate(tqdm(train_set, desc=f"Epoch {epoch + 1}")):
            prompt = item["instruction"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # Compute loss for a single instance
            loss = compute_per_instance_dpo_loss(policy_model, reference_model, tokenizer, beta, prompt, chosen, rejected)

            # Normalize the loss for gradient accumulation
            loss = loss / batch_size
            loss.backward()

            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                print(f"Step {step}, Batch {i+1}, Loss: {loss.item() * batch_size}")

                # Validate periodically within the epoch
                if step % val_interval == 0:
                    val_accuracy = run_validation(policy_model, tokenizer, validation_set)
                    val_accuracies.append(val_accuracy)
                    steps_for_val.append(step)
                    print(f"Step {step} Validation Accuracy: {val_accuracy:.4f}")

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        print(f"New best validation accuracy! Saving model to {save_path}")
                        torch.save(policy_model.state_dict(), save_path)
    
    print("Training finished.")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")

    # Plot validation accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(steps_for_val, val_accuracies, marker='o')
    plt.title("DPO Validation Accuracy Curve")
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.savefig("validation_accuracy_curve.png")
    print("Validation accuracy curve saved to validation_accuracy_curve.png")    

if __name__ == "__main__":
    main()
    
    









