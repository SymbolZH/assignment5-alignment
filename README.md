# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Conda on Ascend NPU (CANN 8.3.RC1)

Use Python 3.11 for the available ARM64 `torch-npu` 2.8.0 wheels:

```sh
conda create -n llm python=3.11 pip -y
conda activate llm
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -e '.[npu]' 'torch==2.8.0' 'torch-npu==2.8.0'
python -c 'import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())'
```

Load the CANN environment in each new shell before using the NPU. The local
`llm` environment also has Conda activation hooks to do this automatically.
The `npu` extra installs the PyTorch NPU backend; training code that uses CUDA,
NCCL, or vLLM still needs separate Ascend adaptation.

CANN 8.3.RC1's `op-compile-tool` metadata lists `getopt`, `inspect`, and
`multiprocessing` as package dependencies. They are Python standard-library
modules, so `pip check` may report them as missing when CANN is on `PYTHONPATH`.

### uv

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

