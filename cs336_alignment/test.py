import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, torch.Tensor]:
    pass

prompts_strs = ["Let's rewrite the sum as an expection", "how are you?"]
outputs_strs = ["using the following indentity", "I'm fine"]
model = AutoModelForCausalLM.from_pretrained("../model/OLMo-2-0425-1B")
tokenizer = AutoTokenizer.from_pretrained("../model/OLMo-2-0425-1B")

input_ids = tokenizer(prompts_strs, return_tensors="pt", padding=True)["input_ids"]

breakpoint()


