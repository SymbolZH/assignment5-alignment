#!/usr/bin/env bash
# OpenAI-compatible vLLM server for the llm Conda environment on Ascend.
set -eo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 MODEL_PATH_OR_ID [additional vllm serve options]" >&2
    exit 2
fi

model=$1
shift
conda_root=${LLM_CONDA_ROOT:-"$HOME/soft/miniforge3"}
source "$conda_root/etc/profile.d/conda.sh"
conda activate llm

# Also load explicitly so the launcher works with environments without hooks.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec vllm serve "$model" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-8000}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --gpu-memory-utilization "${NPU_MEMORY_UTILIZATION:-0.8}" \
    --enforce-eager \
    "$@"
