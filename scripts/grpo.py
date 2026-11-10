import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import jsonlines
import argparse
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from tests.adapters import run_grpo_train_step



def parse_args():
    parser = argparse.ArgumentParser(
        description="Training configuration for rollout-based RL/GRPO training."
    )

    # ---------- Model ----------
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/OLMo-2-0425-1B",
        help="Path of training model",
    )

    # ---------- Data ----------
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="data/gsm8k/train.jsonl",
        help="Path of training dataset",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path of test dataset",
    )
    parser.add_argument(
        "--n_train_examples",
        type=int,
        default=6400,
        help="Number of training examples.",
    )
    parser.add_argument(
        "--n_val_examples",
        type=int,
        default=1024,
        help="Number of validation examples.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="question_only",   
        choices=["question_only", "r1_zero_three_shot_gsm8k", "r1_zero"],  # 三选一
        help="prompting model answer format",
    )

    # ---------- Rollout ----------
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        default=200,
        help="Number of rollout steps.",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=256,
        help="Rollout batch size.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Group size (number of samples per prompt for group-based methods).",
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature during rollout.",
    )
    parser.add_argument(
        "--sampling_max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to sample during rollout.",
    )
    parser.add_argument(
        "--sampling_top_p",
        type=float,
        default=1.0,
        help="Maximum number of tokens to sample during rollout.",
    )

    # ---------- Optimization ----------
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )   
    parser.add_argument(
        "--seed",
        type=int,
        default=0,   
        help="pytorch seed setting",
    )

    return parser.parse_args()

def get_sampling_params(args):
    # 采样参数
    sampling_temperature = args.sampling_temperature
    top_p = args.sampling_top_p
    max_tokens = args.sampling_max_tokens
    prompt_type = args.prompt_type
    seed = args.seed

    stop = ["</answer>"] if prompt_type != "question_only" else None
    include_stop_str_in_output = True if prompt_type != "question_only" else False
    sampling_params = {
        "temperature": sampling_temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": None,
        "stop": stop,
        "include_stop_str_in_output": include_stop_str_in_output
    }
    return sampling_params

# 缓存的prompt模板
prompt_types = ["question_only", "r1_zero_three_shot_gsm8k", "r1_zero"]
templates = {}
for prompt_type in prompt_types:
    with open(f"./cs336_alignment/prompts/{prompt_type}.prompt", "r") as fin:
        templates[prompt_type] = fin.read()


# 加载训练数据，从answer中提取gt，并且切分训练集和验证集
def load_train_data(args):
    train_dataset = args.train_dataset
    n_train_example = args.n_train_examples
    n_val_example = args.n_val_examples
    # question和gt两个key
    with jsonlines.open(train_dataset) as reader:
        train_dataset_list = [{"question": data["question"],
                    "gt": data["answer"].split("#### ")[-1]} for data in list(reader)]
    random.shuffle(train_dataset_list)
    train_set = train_dataset_list[:n_train_example]
    val_set = train_dataset_list[n_train_example: n_train_example + n_val_example]
    return train_set, val_set

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = args.model_path
    llm = VLLMServer(model_id=model_path, 
                     device_type="npu",
                     gpu=1,
                     seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("npu:0")
    model.train()
    model.config.use_cache = False
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    train_set, val_set = load_train_data(args)
    sampling_params = get_sampling_params(args)
    prompt_type = args.prompt_type
    llm.start()
    llm.init_weight_sync("npu:0")

    # 取出若干个进行rollout，然后计算梯度进行优化
    rollout_query_nums = args.rollout_batch_size // args.group_size
    for start_idx in range(0, len(train_set), rollout_query_nums):
        end_idx = min(start_idx + rollout_query_nums, len(train_set))
        rollout_batch = train_set[start_idx: end_idx]

        rollout_batch = [item for item in rollout_batch for _ in range(args.group_size)]
        llm.sync_policy_weights(model)

        # 放采样的内容
        outputs_list_rollout_batch = []
        prompts_list_rollout_batch = [templates[prompt_type].format(question=data["question"]) for data in rollout_batch]
        gts_list_rollout_batch = [data["gt"] for data in rollout_batch]
        responses_rollout_batch = llm.generate_completions(prompts=prompts_list_rollout_batch, 
                                                           sampling_params=sampling_params)
        llm_outputs_list_rollout_batch = [response.text for response in responses_rollout_batch ]
        reward_fn = question_only_reward_fn if prompt_type == "question_only" else r1_zero_reward_fn
        judge_result_rollout_batch = [reward_fn(output, gt) for output, gt in zip(llm_outputs_list_rollout_batch, gts_list_rollout_batch)]
        for idx, data in enumerate(rollout_batch):
            final_output_dict = {}
            final_output_dict["question"] = rollout_batch[idx]["question"]
            final_output_dict["gt"] = rollout_batch[idx]["gt"]

            final_output_dict["prompt"] = prompts_list_rollout_batch[idx]
            final_output_dict["output"] = llm_outputs_list_rollout_batch[idx]
            final_output_dict["judge_result"] =judge_result_rollout_batch[idx]
            outputs_list_rollout_batch.append(final_output_dict)
        # 占用卡的位置应该如何解决？目前暂时没有涉及卡
        # 然后优化后的模型权重没有进行存储和加载
        run_grpo_train_step(model=model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                            max_grad_norm=args.max_grad_norm,
                            reward_fn=reward_fn,
                            repeated_prompts=[item["prompt"] for item in outputs_list_rollout_batch],
                            rollout_responses=[item["output"] for item in outputs_list_rollout_batch],
                            repeated_ground_truths=[item["gt"] for item in outputs_list_rollout_batch],
                            group_size=args.group_size)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(get_sampling_params(args))
    main(args)
