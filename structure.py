import torch
@torch.no_grad()
def get_log_probs(model, tokenizer, prompts, responses, device):
    """计算模型对给定响应的每个序列的对数概率。"""
    log_probs_list = []
    masks_list = []

    for prompt, response in zip(prompts, responses):
        full_text = prompt + response
        tokenized = tokenizer(full_text, return_tensors = "pt", add_special_tokens=False)
        input_ids = tokenized.input_ids.to(device)

        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokenized.input_ids)

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100 # mask prompt part

        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        response_logits = logits[:, prompt_len-1:-1, :]
        response_ids = input_ids[:, prompt_len:]

        log_probs_all = torch.log_softmax(response_logits, dim=-1)
        log_probs_gathered = torch.gather(log_probs_all, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

        response_mask = (response_ids != tokenizer.pad_token_id).float()

        log_probs_list.append(log_probs_gathered.squeeze(0))
        masks_list.append(response_mask.squeeze(0))
    
    padded_log_probs = torch.nn.utils.rnn.pad_sequence(log_probs_list, batch_first=True, padding_value=0.0)
    padded_masks = torch.nn.utils.rnn.pad_sequence(masks_list, batch_first=True, padding_value=0.0)

    return padded_log_probs, padded_masks


def alpaca_format(instruction: str, response: str):
    system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    # CHANGE: add a blank line after system_prompt
    results = system_prompt + f"\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return results



def compute_per_instance_dpo_loss(
        lm,
        lm_ref,
        tokenizer,
        beta,
        prompt,
        response_chosen,
        response_rejected
):
    was_training, was_training_ref = lm.training, lm_ref.training
    lm.eval(); lm_ref.eval()

    al_format_chosen = alpaca_format(prompt, response_chosen)
    al_format_rejected = alpaca_format(prompt, response_rejected)

    device = next(lm.parameters()).device
    chosen_id = tokenizer(al_format_chosen, return_tensors="pt", add_special_tokens=False).to(device)
    reject_id = tokenizer(al_format_rejected, return_tensors="pt", add_special_tokens=False).to(deivce)

    eos = tokenizer.eos_token_id
    if eos is not None:
        for ids in (chosen_id, reject_id):
            ids["input_ids"] = torch.cat([ids["input_ids"], torch.tensor([[eos]], device=device)], dim=1)
            ids["attention_mask"] = torch.cat([ids["attention_mask"], torch.tensor([[1]], device=device, dim=1)])

    prompt_formatted = alpaca_format(prompt, "")
    prompt_len = tokenizer(prompt_formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

     




