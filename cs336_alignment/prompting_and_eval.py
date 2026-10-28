import random
from collections import defaultdict
from pathlib import Path
import json
import jsonlines
import vllm_utils
from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from datetime import datetime, timedelta, timezone

# 缓存的prompt模板
prompt_types = ["question_only", "r1_zero_three_shot_gsm8k", "r1_zero"]
templates = {}
for prompt_type in prompt_types:
    with open(f"./prompts/{prompt_type}.prompt", "r") as fin:
        templates[prompt_type] = fin.read()

# 加载数据，从answer中提取gt
def load_data():
    # question和gt两个key
    with jsonlines.open('../data/gsm8k/test.jsonl') as reader:
        data_list = [{"question": data["question"],
                    "gt": data["answer"].split("#### ")[-1]} for data in list(reader)]
    # choosen_idx = list(range(10))
    # data_list = [data_list[idx] for idx in choosen_idx]
    return data_list

def get_sampling_params(prompt_type):
    # 采样参数
    temperature = 1
    top_p = 1
    max_tokens = 512
    n = 1
    seed = 0
    stop = ["</answer>"] if prompt_type != "question_only" else None
    include_stop_str_in_output = True if prompt_type != "question_only" else False
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
        "stop": stop,
        "include_stop_str_in_output": include_stop_str_in_output
    }
    return sampling_params

def run_baseline(data_list, prompt_type, sampling_params):
    outputs_list = []
    prompts_list = [templates[prompt_type].format(question=data["question"]) for data in data_list]
    gts_list = [data["gt"] for data in data_list]

    # answers_list 
    responses = llm.generate_completions(prompts=prompts_list, sampling_params=sampling_params)

    llm_outputs_list = [response.text for response in responses ]
    reward_fn = question_only_reward_fn if prompt_type == "question_only" else r1_zero_reward_fn

    judge_result = [reward_fn(output, gt) for output, gt in zip(llm_outputs_list, gts_list)]

    for idx, data in enumerate(data_list):
        final_output_dict = {}
        final_output_dict["question"] = data_list[idx]["question"]
        final_output_dict["gt"] = data_list[idx]["gt"]
        final_output_dict["output"] = llm_outputs_list[idx]
        final_output_dict["judge_result"] =judge_result[idx]
        outputs_list.append(final_output_dict)
    return outputs_list

def save_output(outputs_list, prompt_type, sampling_params):
    beijing_tz = timezone(timedelta(hours=8))
    timestap = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    dir_saved_name = (  
                        f"{prompt_type}_{timestap}_"
                        f"temperature{sampling_params["temperature"]}_top_p{sampling_params["top_p"]}"
                        f"_max_tokens{sampling_params["max_tokens"]}_n{sampling_params["n"]}_seed{sampling_params["seed"]}"
    )
    save_dir = Path(__file__).resolve().parents[1] / "results" / dir_saved_name
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_file = {"prediction": "predition.json",
                  "status": "status.json"}
    
    for key, file_name in saved_file.items():
        with open(save_dir / file_name, "w", encoding="utf8") as fin:
            if key == "prediction":
                json.dump(outputs_list, fin)
            else:
                format1_answer1 = sum([output["judge_result"]["format_reward"] and output["judge_result"]["answer_reward"] for output in outputs_list])
                format1_answer0 = sum([output["judge_result"]["format_reward"] and output["judge_result"]["answer_reward"] == 0.0 for output in outputs_list])
                format0_answer0 = sum([output["judge_result"]["format_reward"] == 0.0 and output["judge_result"]["answer_reward"] == 0.0 for output in outputs_list])
                status_dic = {
                    "format1_answer1": format1_answer1,
                    "format1_answer0": format1_answer0,
                    "format0_answer0": format0_answer0,

                }
                json.dump(status_dic, fin)
"""
（1）格式和正确性奖励均为 1；（2）格式奖励为 1、正确性奖励为 0；（3）两项奖励均为 0。
"""

llm = vllm_utils.VLLMServer(model_id="../model/OLMo-2-0425-1B",
                                device_type="npu")
llm.start()

for prompt_type in prompt_types:
    data_list = load_data()
    sampling_params = get_sampling_params(prompt_type)


    outputs_list = run_baseline(data_list, prompt_type, sampling_params)

    save_output(outputs_list, prompt_type, sampling_params)