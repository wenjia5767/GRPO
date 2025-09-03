from vllm import LLM, SamplingParams

def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object, stopping generation on new line.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"], logprobs=1
    )

    # Create an LLM
    llm = LLM(model="/home/zhangwj/Qwen2.5-3B-Instruct")
    # llm = LLM(model="./Qwen2.5-Math-1.5B")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    outputs = llm.generate(prompts, sampling_params)


    # Print the outputs.
    for output in outputs:
        for sample in output.outputs: 
            # generated_text = output.outputs[0].text
            zz = [list(x.values())[0].logprob for x in sample.logprobs]
            print(f"sample {zz}")


if __name__ == "__main__":
    main()


