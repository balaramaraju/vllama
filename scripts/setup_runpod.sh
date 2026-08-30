#!/usr/bin/env bash
# Prepares a RunPod workspace for a vllama FSDP2 run.
#
# Ensures:
#   1. A local tokenizer exists at ./llama_tokenizer_local
#      (skipped if already present; else downloaded from $TOKENIZER_REPO).
#   2. Training data exists at ./fineweb_sample_5k.jsonl
#      (skipped if present; else downloaded from $DATA_URL; else a tiny
#       synthetic sample is generated so the run can be validated end-to-end).
#
# Usage:
#   TOKENIZER_REPO=meta-llama/Llama-3.2-3B-Instruct \
#   DATA_URL=https://example.com/fineweb_sample_5k.jsonl \
#   bash scripts/setup_runpod.sh
#
set -euo pipefail

TOKENIZER_REPO="${TOKENIZER_REPO:-HuggingFaceTB/SmolLM2-135M}"
DATA_URL="${DATA_URL:-}"
DATA="${DATA:-./fineweb_sample_5k.jsonl}"

echo "==> Step 1/2: tokenizer"
python - "${TOKENIZER_REPO}" <<'PY'
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

target = Path("./llama_tokenizer_local")
if target.exists() and (target / "tokenizer.json").exists():
    print("    llama_tokenizer_local already present — skipping download.")
    sys.exit(0)

repo = sys.argv[1]
print(f"    Downloading tokenizer from {repo} -> {target}")
AutoTokenizer.from_pretrained(repo).save_pretrained(target)
print("    Tokenizer saved to", target)
PY

echo "==> Step 2/2: training data"
if [ -f "${DATA}" ]; then
    echo "    Training data found at ${DATA}"
elif [ -n "${DATA_URL}" ]; then
    echo "    Downloading training data from ${DATA_URL}"
    curl -L "${DATA_URL}" -o "${DATA}"
    echo "    Training data downloaded to ${DATA}"
else
    echo "    No data found and DATA_URL unset — generating a tiny synthetic JSONL sample."
    python - "${DATA}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for i in range(500):
        text = (
            f"Sample document {i}. "
            + "The quick brown fox jumps over the lazy dog while a llama trains "
            + "in a distributed cluster. ".repeat(20)
        )
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
print(f"    Wrote synthetic sample data to {out}")
PY
fi

echo "==> OK. Next: bash scripts/run.sh"