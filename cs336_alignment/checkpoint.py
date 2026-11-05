import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer


model, tokenizer = get_model_and_tokenizer(model_id_or_dir="../model/OLMo-2-0425-1B",
                        device="npu")

inputs_text = "how are you?"
inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs))


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    
    pass