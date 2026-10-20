# Code Agent Instructions

<!-- init_agent:maintenance:start -->
## Shared Agent Configuration

This file is the canonical project instruction source.

- Claude Code uses `CLAUDE.md -> CODEAGENT.md`.
- Codex uses `AGENTS.md -> CODEAGENT.md`.
- Shared skills live in `skills/`.
- `.claude/skills` and `.agents/skills` link to `skills/`.

Maintain shared instructions here and reusable workflows in
`skills/<skill-name>/SKILL.md`.
<!-- init_agent:maintenance:end -->

## Project Overview

CS336 Assignment 5：Alignment / RL。以作业 PDF 和 Markdown 中的实现、实验及交付要求为目标，当前本机 NPU 开发与调试统一使用下面的 Docker 环境。

NPU 适配集中在依赖、设备选择、推理服务及权重同步等必要差异上。实验从作业代码入口执行；一次性的环境探测、硬件诊断和验证脚本放在仓库外，不作为作业流程的前置步骤。保留用于验证必要适配行为的回归测试。

## Repository Structure

- `cs336_alignment/vllm_utils.py`：推理服务启动、请求和权重同步的项目封装。
- `docker/ascend/`：开发镜像 Dockerfile、依赖约束和入口脚本。
- `scripts/ascend_container.sh`：容器构建、启动、进入和重建。
- `tests/`：作业测试及必要的兼容性回归测试。

## Development Commands

### 固定的 Ascend 开发环境

截至 2026-10-19，使用以下环境：

| 项目 | 配置 |
| --- | --- |
| 容器 | `assignment5-ascend-dev` |
| 开发镜像 | `assignment5-ascend:vllm0.23.0` |
| 基础镜像 | 官方 ARM64 vLLM Ascend 0.23.0，Dockerfile 使用 SHA256 digest 固定 |
| NPU 框架 | vLLM / vllm-ascend 0.23.0、Torch 2.10.0、torch-npu 2.10.0.post4 |
| 其他核心版本 | Python 3.12、Transformers 5.5.4、CANN/NNAL 9.1.0 |
| 项目挂载 | 宿主机 `/home/zhanghao/my_code/assignment5-alignment` → 容器 `/workspace/assignment5-alignment` |
| 缓存 | Docker volume `assignment5-ascend-cache` → 容器 `/home/zhanghao/.cache` |

开发镜像在官方镜像上添加项目工具、普通用户及算子文件读取权限；使用官方 HCCL 实现，未自行实现传输引擎。基础镜像 digest 和核心依赖约束分别见 `docker/ascend/Dockerfile`、`docker/ascend/constraints.txt`。

### 从宿主机进入和执行命令

以下命令在宿主机项目根目录执行：

```bash
bash scripts/ascend_container.sh status
bash scripts/ascend_container.sh start
bash scripts/ascend_container.sh shell

# 非交互执行
bash scripts/ascend_container.sh exec python -m pytest tests/test_vllm_weight_sync.py -q

# 也可以直接进入容器
docker exec -it -w /workspace/assignment5-alignment assignment5-ascend-dev bash
```

容器主进程是 `sleep infinity`，`docker attach` 不会打开命令行；使用 `docker exec`。
项目以 editable 模式安装，修改挂载目录中的源码即可生效，无需重建镜像；已运行的 Python 进程或服务通常需要重启才能加载修改。

### 作业中的推理服务

在作业实验入口中使用 `VLLMServer(device_type="npu", ...)`，由封装处理服务启动与 HCCL 后端选择。`start()` 等服务就绪后返回，随后继续生成、评估或训练；`startup_timeout` 只限制启动等待时间。用 `try/finally` 在实验结束或异常时调用 `stop()`，完整调用示例见 `README.md`。

`tests/test_vllm_weight_sync.py` 使用 mock 检查 CUDA/NPU 的权重同步顺序和失败处理，无需占用加速卡；它不替代作业要求的测试和实验。

2026-10-17 已在 Qwen2.5-0.5B、1024 上下文、最大并发 4 下验证 `enforce_eager=False`、`FULL_AND_PIECEWISE` 和实际 ACL 图重放；权重更新与恢复通过，测试输出与 eager 模式一致。
宿主机记录见 `/home/zhanghao/soft/assignment5-native/graph-validation/validation.json`，该结论仅覆盖已测配置。GPU 侧已更新依赖和公共接口测试，尚未在 NVIDIA GPU 上实机验证。

## Development Rules

- 本机 NPU 调试优先复用现有开发容器。宿主机 Conda `llm` 是旧 CANN 8.3 / vLLM 0.11 推理环境，与当前容器环境不同。
- 不要在 NPU 容器中用 `uv sync --extra gpu` 或通用 PyPI 安装流程替换官方 device-specific 框架；`uv.lock` 当前覆盖 macOS ARM / Linux x86_64，NPU 依赖由 Docker 构建配置管理。
- GPU/NPU 的 vLLM 版本均为 0.23.0，但官方配套 Torch 分别为 GPU 2.11.0 / NPU 2.10.0，不能强求所有底层包版本相同。
- “原生 HCCL”表示官方实现权重传输：HTTP 接口及 `trainer_init` / `trainer_send_weights` 方法一致，但 Python 导入路径和类名仍不同（`NCCLWeightTransferEngine` / `HCCLWeightTransferEngine`）。项目封装负责选择后端，不要将其描述为上游 Python 接口完全统一。
- `ASCEND_RT_VISIBLE_DEVICES` 选择可见设备；`VLLM_ASCEND_ENABLE_NZ=0` 禁用 NZ 权重布局，是当前官方权重更新路径的检查要求。两者含义不同，保留后者。
- `VLLMServer` 默认不强制 `--enforce-eager`。图模式已在上述配置验证；不要仅因使用 NPU 就自动加回 eager。
- 镜像存在已知上游依赖声明冲突：OpenCV 5 要求 NumPy >=2，而 Triton Ascend 固定 NumPy 1.26.4；诊断见容器 `/opt/assignment5/pip-check.txt`。保留官方运行栈，不要为消除这一提示直接升级 NumPy。
- 同一 NPU 被不同容器占用时可能发生 device-busy；测试前检查占用，替换本项目旧容器时先停止旧实例，避免干扰其他项目容器。
- 容器内临时安装的包仅保存在该容器的可写层；需要长期保留时更新 `docker/ascend/requirements.txt` / `constraints.txt` 等构建配置。
- 仅变更镜像环境时才需在宿主机执行 `bash scripts/ascend_container.sh build`，再按需执行 `bash scripts/ascend_container.sh recreate`。`recreate` 会停止并删除旧容器，保留项目 bind mount 和缓存 volume，但不会保留其他容器可写层改动，重建前应先保存这些改动。
- 复用现有镜像可保持其已安装环境；重新构建时，未精确锁定的附加工具依赖仍可能变化。宿主机 NPU 驱动不包含在镜像中，不要将镜像固定理解为宿主机也被冻结。
