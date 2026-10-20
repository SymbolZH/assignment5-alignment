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

Inside the container, run the assignment's experiment code directly. Use
`cs336_alignment.vllm_utils.VLLMServer` to manage inference from that code:

```python
from cs336_alignment.vllm_utils import VLLMServer

server = VLLMServer(model_id="/path/to/model", device_type="npu", gpu=1)
try:
    server.start()
    completions = server.generate_completions(
        prompts=["Hello"],
        sampling_params={"temperature": 1.0, "max_tokens": 512, "n": 1, "seed": 0},
    )
    print(completions[0].text)
finally:
    server.stop()
```

`start()` returns when the server is ready, allowing evaluation or training
to continue. `startup_timeout` limits the wait for readiness. The service
stays available while the experiment runs and is stopped during cleanup.

The wrapper selects the visible accelerator and HCCL backend, enables the
official development endpoints, and disables NZ weight layouts as required
by Ascend checkpoint updates. It defaults to `127.0.0.1:8000`; keep these
endpoints on a trusted interface.

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

Run the weight-sync protocol regression tests inside the container:

```sh
python -m pytest tests/test_vllm_weight_sync.py -q
```

These tests mock device communication and check CUDA/NPU protocol ordering
and failure handling; they do not require accelerators. Complete the assignment's
implementations and run its required tests and experiments as described in
the handout.

`VLLMServer` leaves graph execution enabled by default. Do not force eager
execution solely because the device is an NPU.

The pinned upstream image has one pre-existing dependency declaration conflict:
OpenCV 5 declares NumPy >=2, while its Triton Ascend compiler requires NumPy
1.26.4. This environment preserves the official compiler/runtime packages and
records that declaration conflict in `/opt/assignment5/pip-check.txt`.
Other dependency conflicts fail the build. On ARM, pytest 8.3.2 matches the
compiler's required test-tool version. Upstream framework files are unmodified.
The Dockerfile also grants the development user read/execute access to the
official image's packaged operators, which otherwise are accessible only to root.

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

