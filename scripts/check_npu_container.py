"""Verify HTTP inference and reversible HCCL weight updates on two NPUs.

Run inside assignment5-ascend-dev. The test zeros the inference model's
output embedding, verifies that output changes, then restores its weights.
The temporary server is stopped on exit.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from openai import OpenAI
from safetensors import safe_open
import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM
from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

from cs336_alignment.vllm_utils import (
    generate_completions,
    init_weight_sync,
    stop_server,
    sync_policy_weights,
    wait_for_server,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / ".cache/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--server-device", type=int, default=0)
    parser.add_argument("--trainer-device", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / ".cache/assignment5-validation")
    args = parser.parse_args()
    if args.server_device == args.trainer_device:
        raise ValueError("The HCCL test requires two different NPUs")
    root = Path(__file__).resolve().parents[1]
    cache = args.output_dir
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "ASCEND_RT_VISIBLE_DEVICES": str(args.server_device),
        "PORT": str(args.port),
        "MAX_MODEL_LEN": "1024",
        "MAX_NUM_SEQS": "4",
        "NPU_MEMORY_UTILIZATION": "0.2",
    })
    base_url = f"http://127.0.0.1:{args.port}"
    with (cache / "server.log").open("w") as log:
        process = subprocess.Popen(
            ["bash", str(root / "scripts/serve_npu_container.sh"), args.model,
             "--served-model-name", "qwen-npu"],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            wait_for_server(base_url, process, timeout=args.startup_timeout)
            client = OpenAI(base_url=base_url + "/v1", api_key="local", timeout=120)

            def generate() -> str:
                response = client.chat.completions.create(
                    model="qwen-npu",
                    messages=[{"role": "user", "content": "What is 1+1?"}],
                    temperature=0, max_tokens=16,
                )
                return response.choices[0].message.content or ""

            before = generate()
            assert before, "Inference returned no text"
            print("Initial inference:", repr(before), flush=True)
            completions = generate_completions(
                base_url, "qwen-npu", ["Hello", "The answer is"],
                {"temperature": 0, "max_tokens": 8, "n": 1, "seed": 0},
            )
            assert len(completions) == 2 and all(c.token_ids for c in completions)
            with safe_open(str(Path(args.model) / "model.safetensors"), framework="pt") as weights:
                parameter_name = (
                    "lm_head.weight" if "lm_head.weight" in weights.keys()
                    else "model.embed_tokens.weight"
                )
                original = weights.get_tensor(parameter_name)
            policy = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=original.dtype, attn_implementation="eager"
            ).to(f"npu:{args.trainer_device}")
            policy_weight = policy.get_parameter(parameter_name)
            with torch.no_grad():
                policy_weight.zero_()
            group = init_weight_sync(base_url, f"npu:{args.trainer_device}")
            sync_policy_weights(policy, base_url, group)
            changed = generate()
            assert changed != before, (before, changed)
            print("After HCCL zero-weight update:", repr(changed), flush=True)
            with torch.no_grad():
                policy_weight.copy_(original)
            sync_policy_weights(policy, base_url, group)
            restored = generate()
            assert restored == before, (before, restored)
            print("After HCCL restore:", repr(restored), flush=True)
            stream = client.chat.completions.create(
                model="qwen-npu", messages=[{"role": "user", "content": "Say hello."}],
                temperature=0, max_tokens=16, stream=True,
            )
            streamed = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
            assert streamed
            result = {
                "transfer_engine": HCCLWeightTransferEngine.__module__,
                "packed": True,
                "versions": {p: importlib.metadata.version(p) for p in
                             ["vllm", "vllm-ascend", "torch", "torch-npu", "transformers"]},
                "server_device": args.server_device,
                "trainer_device": args.trainer_device,
                "updated_parameter": parameter_name,
                "completion_token_ids": [c.token_ids for c in completions],
                "before": before, "changed": changed, "restored": restored,
                "streamed": streamed,
            }
            (cache / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print("PASS: HTTP inference, streaming, HCCL update and restoration", flush=True)
        finally:
            stop_server(process, timeout=60)


if __name__ == "__main__":
    main()
