def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    p_token = tokenizer(prompt_strs, add_special_toje)