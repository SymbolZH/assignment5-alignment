import random
from collections import defaultdict
from pathlib import Path
import json
from datetime import datetime
import jsonlines
import vllm_utils
from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

# question和gt两个key
with jsonlines.open('../data/gsm8k/train.jsonl') as reader:
    data_list = [{"question": data["question"],
                  "gt": data["answer"].split("#### ")[-1]} for data in list(reader)]

prompt_types = ["question_only", "r1_zero_three_shot_gsm8k", "r1_zero"]
templates = {}
for prompt_type in prompt_types:
    with open(f"./prompts/{prompt_type}.prompt", "r") as fin:
        templates[prompt_type] = fin.read()

llm = vllm_utils.VLLMServer(model_id="../model/OLMo-2-0425-1B",
                            device_type="npu")

llm.start()


# 采样参数
temperature = 1
top_p = 1
max_tokens = 512
n = 1
seed = 0
stop = None
sampling_params = {
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
    "n": n,
    "seed": seed,
    "stop": stop
}


def run_baseline(data_list, prompt_type):
    choosen_idx = list(range(10))
    data_list = [data_list[idx] for idx in choosen_idx]
    
    outputs_list = []
    prompts_list = [templates[prompt_type].format(question=data["question"]) for data in data_list]
    gts_list = [data["gt"] for data in data_list]

    # answers_list 
    responses = llm.generate_completions(prompts=prompts_list, sampling_params=sampling_params)

    llm_outputs_list = [response.text for response in responses ]
    reward_fn = question_only_reward_fn if prompt_type == "quesiton_only" else r1_zero_reward_fn

    judge_result = [reward_fn(output, gt) for output, gt in zip(llm_outputs_list, gts_list)]

    for idx, data in enumerate(data_list):
        final_output_dict = {}
        final_output_dict["question"] = data_list[idx]["question"]
        final_output_dict["gt"] = data_list[idx]["gt"]
        final_output_dict["output"] = llm_outputs_list[idx]
        final_output_dict["judge_result"] =judge_result[idx]
        outputs_list.append(final_output_dict)
    return outputs_list

prompt_type = "question_only"
run_baseline(data_list, prompt_type)

timestap = datetime.now().strftime("%Y%m%d_%H%M%S")


dir_saved_name = (  
                    f"{timestap}_"
                    f"temperature{temperature}_top_p{top_p}"
                    f"_max_tokens{max_tokens}_n{n}_seed{seed}"
)
save_dir = Path(__file__).resolve().parents[1] / "results" / dir_saved_name
save_dir.mkdir(parents=True, exist_ok=True)

saved_file = {"prediction": "predition.json"}

for key, file_name in saved_file.items():
    with open(save_dir / file_name, "w", encoding="utf8") as fin:
        json.dump(data_list, fin)