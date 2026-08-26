#!/usr/bin/env bash
# Launch FSDP2 training via torchrun on a single node (RunPod friendly).
#
# Usage:
#   NPROC=4 CONFIG=config/train_config.json bash scripts/run_train.sh
#
set -euo pipefail

NPROC="${NPROC:-4}"
CONFIG="${CONFIG:-config/train_config.json}"

if ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun not found. Install a CUDA-enabled torch (>=2.5) first." >&2
    exit 1
fi

echo "🚀 Launching FSDP2 training: ${NPROC} processes, config=${CONFIG}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    src/train/fsdp_train.py --config "${CONFIG}"