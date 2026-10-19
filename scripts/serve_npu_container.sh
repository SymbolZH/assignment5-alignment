#!/usr/bin/env bash
# Run inside the official Ascend 0.23.0 development container.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 MODEL_PATH_OR_ID [additional vllm serve options]" >&2
    exit 2
fi
model=$1
shift
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_SERVER_DEV_MODE=1
export VLLM_ASCEND_ENABLE_NZ=0
exec vllm serve "$model" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --gpu-memory-utilization "${NPU_MEMORY_UTILIZATION:-0.8}" \
    --weight-transfer-config '{"backend":"hccl"}' \
    "$@"
