#!/usr/bin/env bash
# Download HF data and train in parallel.
#
# Starts the downloader in the background (writing JSONL incrementally), then
# launches training immediately. Training consumes rows as they land and stops
# tailing when the downloader writes the `.done` marker.
#
# Usage:
#   NPROC=4 CONFIG=config/train_config.json \
#   DATASET=HuggingFaceFW/fineweb-edu CONFIG_NAME=sample-10BT NUM_ROWS=5000 \
#   bash scripts/run_parallel.sh
#
set -euo pipefail

NPROC="${NPROC:-4}"
CONFIG="${CONFIG:-config/train_config.json}"
DATA_DIR="${DATA_DIR:-./training_data/fineweb}"

DATASET="${DATASET:-HuggingFaceFW/fineweb}"
CONFIG_NAME="${CONFIG_NAME:-}"
SPLIT="${SPLIT:-train}"
CHUNK_MB="${CHUNK_MB:-50}"
MAX_BUF_GB="${MAX_BUF_GB:-3}"
RESUME_BUF_GB="${RESUME_BUF_GB:-2}"

if ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun not found. Install a CUDA-enabled torch (>=2.5) first." >&2
    exit 1
fi

# Start the sliding-window downloader if we don't already have live chunk files.
if find "${DATA_DIR}" -name 'chunk_*.jsonl' 2>/dev/null | grep -q .; then
    echo "==> Chunks already present in ${DATA_DIR}, skipping download."
else
    CONFIG_NAME_ARG=()
    if [ -n "${CONFIG_NAME}" ]; then
        CONFIG_NAME_ARG=(--config-name "${CONFIG_NAME}")
    fi
    echo "==> Starting HF sliding-window download in background..."
    python src/datautils/download_data.py \
        --dataset "${DATASET}" \
        "${CONFIG_NAME_ARG[@]+"${CONFIG_NAME_ARG[@]}"}" \
        --split "${SPLIT}" \
        --data-dir "${DATA_DIR}" \
        --chunk-size-mb "${CHUNK_MB}" \
        --max-buffer-gb "${MAX_BUF_GB}" \
        --resume-buffer-gb "${RESUME_BUF_GB}" &
    DOWNLOAD_PID=$!
    echo "    downloader PID: ${DOWNLOAD_PID}"
fi

echo "==> Launching training (NPROC=${NPROC})..."
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    src/train/fsdp_train.py --config "${CONFIG}"
TRAIN_STATUS=$?

# Wait for the downloader to finish so the script doesn't orphan it.
if [ -n "${DOWNLOAD_PID:-}" ] && kill -0 "${DOWNLOAD_PID}" 2>/dev/null; then
    echo "==> Waiting for background downloader (${DOWNLOAD_PID}) to finish..."
    wait "${DOWNLOAD_PID}"
fi

echo "==> Parallel run finished (train exit: ${TRAIN_STATUS})"
exit "${TRAIN_STATUS}"