import torch
import random
import argparse
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime, timedelta, timezone
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.prompting_and_eval import run_baseline, get_sampling_params, save_output
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


# 仅仅是这里没有区分而已，但其实RL的train和val往往是不同的
def get_val_sampling_params(args):
    # 采样参数
    sampling_temperature = args.sampling_temperature
    top_p = args.sampling_top_p
    max_tokens = args.sampling_max_tokens
    n = 1
    seed = None
    prompt_type = args.prompt_type

    return get_sampling_params(prompt_type,
                               temperature=sampling_temperature,
                               top_p=top_p,
                               max_tokens=max_tokens,
                               seed=seed,
                               n=n)


def get_train_sampling_params(args):
    # 采样参数
    sampling_temperature = args.sampling_temperature
    top_p = args.sampling_top_p
    max_tokens = args.sampling_max_tokens
    n = 1
    seed = None
    prompt_type = args.prompt_type

    return get_sampling_params(prompt_type,
                               temperature=sampling_temperature,
                               top_p=top_p,
                               max_tokens=max_tokens,
                               seed=seed,
                               n=n)




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

def log_reward_metrics(writer, outputs_list, step, split):
    """Record mean reward, answer accuracy, and format rate for a rollout."""
    if not outputs_list:
        return
    for metric, reward_key in (
        ("reward", "reward"),
        ("accuracy", "answer_reward"),
        ("format_rate", "format_reward"),
    ):
        value = sum(output["judge_result"][reward_key] for output in outputs_list) / len(outputs_list)
        writer.add_scalar(f"{split}/{metric}", value, step)


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
    prompt_type = args.prompt_type

    beijing_tz = timezone(timedelta(hours=8))
    timestap = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S_%f")
    based_dir = Path(__file__).resolve().parents[1] / "grpo_training" / timestap
    based_dir.mkdir(parents=True, exist_ok=False)

    with SummaryWriter(log_dir=str(based_dir / "tensorboard"), flush_secs=10) as writer:
        try:
            llm.start()
            llm.init_weight_sync("npu:0")
            llm.sync_policy_weights(model)

            val_output_list = run_baseline(llm, val_set, prompt_type, get_val_sampling_params(args))
            save_output(val_output_list, based_dir / "val-0-steps")
            log_reward_metrics(writer, val_output_list, step=0, split="val")
            last_eval_step = 0

            # 取出若干个进行rollout，然后计算梯度进行优化
            rollout_query_nums = args.rollout_batch_size // args.group_size
            step_nums = 0
            for start_idx in range(0, len(train_set), rollout_query_nums):
                step_nums = start_idx // rollout_query_nums + 1
                end_idx = min(start_idx + rollout_query_nums, len(train_set))
                rollout_batch = train_set[start_idx: end_idx]
                rollout_batch = [item for item in rollout_batch for _ in range(args.group_size)]

                outputs_list_rollout_batch = run_baseline(llm, rollout_batch, prompt_type, get_train_sampling_params(args))
                save_output(outputs_list_rollout_batch, based_dir / f"train-{step_nums}-steps")
                reward_fn = question_only_reward_fn if prompt_type == "question_only" else r1_zero_reward_fn
                loss, metadata = run_grpo_train_step(model=model,
                                                    tokenizer=tokenizer,
                                                    optimizer=optimizer,
                                                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                    max_grad_norm=args.max_grad_norm,
                                                    reward_fn=reward_fn,
                                                    repeated_prompts=[item["prompt"] for item in outputs_list_rollout_batch],
                                                    rollout_responses=[item["output"] for item in outputs_list_rollout_batch],
                                                    repeated_ground_truths=[item["gt"] for item in outputs_list_rollout_batch],
                                                    group_size=args.group_size)
                writer.add_scalar("train/loss", loss.detach().item(), step_nums)
                log_reward_metrics(writer, outputs_list_rollout_batch, step=step_nums, split="train")
                llm.sync_policy_weights(model)

                # 每20步评测并保存模型
                if step_nums % 20 == 0:
                    val_output_list = run_baseline(llm, val_set, prompt_type, get_val_sampling_params(args))
                    save_output(val_output_list, based_dir / f"val-{step_nums}-steps")
                    log_reward_metrics(writer, val_output_list, step=step_nums, split="val")
                    last_eval_step = step_nums

                    checkpoint_dir = based_dir / f"checkpoint/step-{step_nums}"
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

            # 最后一步若已评估，复用结果，避免同一步记录两组验证指标
            if step_nums != last_eval_step:
                val_output_list = run_baseline(llm, val_set, prompt_type, get_val_sampling_params(args))
                log_reward_metrics(writer, val_output_list, step=step_nums, split="val")
            save_output(val_output_list, based_dir / "final-steps-val")

            checkpoint_dir = based_dir / "checkpoint/step-final"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        finally:
            llm.stop()

if __name__ == "__main__":
    args = parse_args()
    main(args)
