# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Ascend development container (vLLM 0.23.0, native HCCL)

GPU and NPU use the same vLLM 0.23.0 weight-update API. The NPU container is
based on the official ARM64 `vllm-ascend:0.23.0` release, pinned by digest in
`docker/ascend/Dockerfile`. Its HCCL engine, worker methods, packed transport,
and weight-reload hooks are supplied by the official release.

The NPU stack uses Python 3.12, Torch 2.10.0, TorchNPU 2.10.0.post4,
Transformers 5.5.4, and CANN/NNAL 9.1.0. The GPU extra uses vLLM 0.23.0,
Transformers 5.5.4, and its matching Torch 2.11.0 / CUDA 13 binaries. The
standalone FlashAttention wheel targets NVIDIA/Linux x86_64 with Python 3.12.
The Modal image uses CUDA 13.0.2 to match these GPU binaries.
The two framework builds have the same serving API and different device dependencies.

The `npu` and `npu-container` extras describe Linux ARM dependencies. Install the
container environment through the Dockerfile rather than `uv sync`: the official
image supplies device-specific framework builds. `uv.lock` covers the existing
macOS ARM and Linux x86_64 development targets.

From the host project directory:

```sh
bash scripts/ascend_container.sh build
bash scripts/ascend_container.sh start
bash scripts/ascend_container.sh shell
```

For an existing container built from an older image, use `recreate` after
building. This stops and replaces that container; the project bind mount and
named cache volume persist. The `start` command rejects a mismatched image.

The container is named `assignment5-ascend-dev`. It mounts this repository at
`/workspace/assignment5-alignment` and uses the host user's UID/GID for files.
The project is installed in editable mode automatically. Model and compiler
caches persist in the named Docker volume `assignment5-ascend-cache`.
Use `bash scripts/ascend_container.sh exec COMMAND...` for a noninteractive
command, and `status` or `stop` to inspect or stop the container.

On this machine, exposing the same NPU to another container while the old
container still owns it can produce device-busy errors. Stop the previous
assignment container before testing or starting its replacement.

Inside the container, start a local server on NPU 0:

```sh
ASCEND_RT_VISIBLE_DEVICES=0 NPU_MEMORY_UTILIZATION=0.2 \
    bash scripts/serve_npu_container.sh \
    "$HOME/.cache/models/Qwen2.5-0.5B-Instruct" --served-model-name qwen-npu
```

The server listens on `127.0.0.1:8000`. Substitute another supported local
model or model ID as needed. The launcher uses `--weight-transfer-config
'{"backend":"hccl"}'`, enables the official development endpoints, and disables
NZ weight layouts as required by Ascend checkpoint updates. Keep these endpoints
on a trusted interface.

For RL, use `VLLMServer(device_type="cuda")` or
`VLLMServer(device_type="npu")`. Both use the same public helpers:

```python
server.init_weight_sync(policy_device)  # e.g. "cuda:0" or "npu:0"
server.sync_policy_weights(policy)
```

Only device selection and the official engine imports differ:
`NCCLWeightTransferEngine` from vLLM on GPU, and `HCCLWeightTransferEngine`
from vllm-ascend on NPU. Both call `trainer_init` / `trainer_send_weights` and
use packed transfers. The common update sequence is:

```text
pause -> start_weight_update -> update_weights + trainer_send_weights
      -> finish_weight_update -> reset_prefix_cache -> resume
```

A transfer failure leaves the server paused, preventing inference with partially
updated weights. The inference server and trainer should use different physical
accelerators. The previous project-defined worker extension has been removed.

Validate the native implementation on two NPUs:

```sh
python -m pytest tests/test_vllm_weight_sync.py -q
python scripts/check_npu_container.py
```

The integration check uses a temporary server on port 18081 and a full
Hugging Face Qwen2.5-0.5B policy on a second NPU. It transfers modified weights,
checks that generation changes, restores all policy weights, and checks that
the original output returns. It also checks completion token IDs and streaming.
Logs, native engine provenance, and results are stored in
`$HOME/.cache/assignment5-validation`; the temporary server is stopped on exit.
The NPU launchers leave graph execution enabled by default. On this machine,
Qwen2.5-0.5B passed the same update/restore check with `enforce_eager=False`,
`FULL_AND_PIECEWISE` graph mode, and confirmed ACL graph replay. The tested
configuration used a 1024-token context and at most four concurrent sequences.
Use `--enforce-eager` explicitly if a model requires eager execution. For a
separate validation log directory, pass `--output-dir PATH` to the check script;
`--startup-timeout` controls how long it waits for initial compilation.

The pinned upstream image has one pre-existing dependency declaration conflict:
OpenCV 5 declares NumPy >=2, while its Triton Ascend compiler requires NumPy
1.26.4. This environment preserves the official compiler/runtime packages and
records that declaration conflict in `/opt/assignment5/pip-check.txt`.
Other dependency conflicts fail the build. On ARM, pytest 8.3.2 matches the
compiler's required test-tool version. Upstream framework files are unmodified.
The Dockerfile also grants the development user read/execute access to the
official image's packaged operators, which otherwise are accessible only to root.

### Legacy host Conda (fixed-weight inference, CANN 8.3.RC1)

Use Python 3.11 and the [validated version combination](https://docs.vllm.ai/projects/ascend/en/v0.13.0/community/versioning_policy.html)
for CANN 8.3.RC1: PyTorch/torch-npu 2.7.1, vLLM 0.11.0, and
vllm-ascend 0.11.0rc1. Newer Ascend releases require a newer CANN installation.

```sh
conda create -n llm python=3.11 pip -y
conda activate llm
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -e '.[npu]'
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
python -m pip install 'setuptools>=77.0.3,<80' setuptools-scm wheel ninja cmake

# Build the Python-only vLLM core; vllm-ascend provides the NPU backend.
# The ordinary PyPI vLLM wheel targets a different Torch/device combination.
mkdir -p "$HOME/soft/vllm-npu/src"
git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git \
    "$HOME/soft/vllm-npu/src/vllm-0.11.0"
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation \
    -e "$HOME/soft/vllm-npu/src/vllm-0.11.0"
python -c 'import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())'
```

Load CANN and ATB in each new shell before using the NPU. The local `llm`
environment has Conda activation/deactivation hooks to load and restore both.
Reactivate an already-open environment after changing the hooks.

Start an OpenAI-compatible inference server on NPU 0:

```sh
conda activate llm
ASCEND_RT_VISIBLE_DEVICES=0 NPU_MEMORY_UTILIZATION=0.2 \
    bash scripts/serve_npu.sh \
    "$HOME/soft/vllm-npu/models/Qwen2.5-0.5B-Instruct" \
    --served-model-name qwen-npu
```

The local Qwen2.5-0.5B-Instruct directory is downloaded during this machine's
installation verification. You can substitute another supported local model
or Hugging Face model ID. The launcher defaults to `127.0.0.1:8000`, eager
execution, a 4096-token context, and tensor parallel size 1. Configure `HOST`,
`PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `NPU_MEMORY_UTILIZATION`, or
`TENSOR_PARALLEL_SIZE` as environment variables; additional arguments are passed
to `vllm serve`. For two NPUs, use `ASCEND_RT_VISIBLE_DEVICES=0,1` and
`TENSOR_PARALLEL_SIZE=2` with a model that supports that parallel size.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen-npu","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

The `npu` extra and launcher configure inference. Assignment training code that
uses CUDA, NCCL, or CUDA-specific vLLM internals still needs Ascend adaptation.

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

