# vllama

A small educational Llama implementation in PyTorch, trained with FSDP2
(fully-sharded data parallel) on a single multi-GPU node (RunPod friendly).

Highlights:

- **FSDP2 (`fully_shard`)** bottom-up sharding, incl. weight-tied embedding/output
- **Eager-mode** training (no `torch.compile`), **bf16** params + AdamW state on CUDA
- **Sliding-window dataset**: downloader streams Hugging Face FineWeb into ~50 MB
  chunk files; the loader consumes them **in order and deletes each shard once all
  GPUs are done with it** — the data folder self-balances at 2–3 GB.
- **DCP sharded checkpoints** (model + optimizer), auto-resume of the latest `step_*`,
  pruning to the newest N checkpoints.

---

## Quick start (RunPod)

### 1. Provision a RunPod pod
Pick a template with a CUDA-enabled PyTorch image (e.g. **PyTorch 2.5+ CUDA**),
with **at least 1 GPU** and enough VRAM:
- ~24 GB VRAM/GPU × 1 for a small test
- **4 × 24 GB (e.g. 4090)** or **2 × 48 GB (A6000)** for the full 3B config
- Add plenty of disk (the sliding window only needs a few GB, but checkpoints × N
  add up; prune keeps it small).

### 2. Get the code onto the pod
```bash
cd ~
git clone https://github.com/balaramaraju/vllama.git
cd vllama

# Install deps (torch is usually preinstalled in the image; ensure >= 2.5)
pip install -r requirements.txt
```

### 3. Prepare tokenizer + assets
`scripts/setup_runpod.sh` downloads a small tokenizer into `./llama_tokenizer_local`
and (optionally) seeds the initial data file.

```bash
# Uses HuggingFaceTB/SmolLM2-135M by default (gated-free, tiny)
bash scripts/setup_runpod.sh
```

> For the real 3B run your config points `tokenizer_path: ./llama_tokenizer_local`.
> If you already have a tokenizer directory, copy it there and skip the download.

### 4. Launch training (data downloads in parallel)

**Single entry point — `scripts/run.sh` handles both training-only and train+download:**

```bash
# Train with existing data (no downloader):
NPROC=4 bash scripts/run.sh

# Train + download in parallel (full FineWeb):
NPROC=4 DATASET=HuggingFaceFW/fineweb bash scripts/run.sh

# A subset instead of the full 1TB FineWeb:
DATASET=HuggingFaceFW/fineweb CONFIG_NAME=sample-10BT NPROC=4 bash scripts/run.sh

# Long-running: full dataset, custom buffer caps:
MAX_BUF_GB=3 RESUME_BUF_GB=2 CHUNK_MB=50 NPROC=4 bash scripts/run.sh

# Performance flags:
COMPILE=1 ACCUMULATE=2 NO_ACTIVATION_CKPT=1 NPROC=4 bash scripts/run.sh
```

What `run.sh` does:
1. If chunks (`chunk_*.jsonl`) already exist under `DATA_DIR`, **skips download** and launches training directly.
2. Otherwise, starts the **HF downloader in the background**: streams `HuggingFaceFW/fineweb`
   (full dataset) into `./training_data/fineweb/chunk_*.jsonl` (~50 MB each),
   pausing when the folder exceeds **3 GB** and resuming once training cleans it
   back down to **2 GB**.
3. Launches `torchrun` training immediately (4 processes). Training begins as soon
   as the first chunk lands, consumes chunks **in order**, and **deletes each shard
   once all GPUs are done with it**.
4. When the downloader finishes (or is Ctrl-C'd), the script waits for it and exits.

### 5. Monitor + resume
- Progress prints each `Step NNNN | Loss: x.xxxx | Time: NNNms | Peak VRAM: NNNNMB` (log_every) on rank 0.
- Checkpoints are saved under `./checkpoints/llama_3b/step_%07d/` every `save_every`
  steps; only the newest `keep_last` (default **2**) are retained.
- **To resume** (e.g. after a pod restart), just re-run the same launch command:
  the trainer auto-loads the **latest** `step_*` checkpoint and continues.
- To force-start fresh: set `"resume": false` (or `"load_from": null` + delete the
  checkpoint dir).

---

## Configuration

All settings live in `config/train_config.json`:

```jsonc
{
  "training_data": {
    "hf_path": "HuggingFaceFW/fineweb",
    "config": null,                // dataset config; null = full dataset
    "split": "train",
    "block_size": 4096,            // tokens per sequence
    "batch_size": 16,               // sequences per step
    "is_local_file": true,         // true = read local chunk files
    "tokenizer_path": "./llama_tokenizer_local",
    "wait_for_data": true,         // poll for file/chunks instead of failing
    "poll_interval": 1.0,
    "source_mode": "shard_dir",    // "file" | "shard_dir" | "hf"
    "data_dir": "./training_data/fineweb",
    "delete_consumed": true        // delete shards once all GPUs consume them
  },
  "config": {
    "model": { "vocab_size": 128256, "n_embd": 3072, "n_blocks": 28, "n_heads": 24,
               "n_kv_heads": 8, "max_seq_len": 2048 },
    "optimizer": { "lr": 0.0003 },
    "training": { "max_steps": 10000, "log_every": 10 },
    "checkpoint": {
      "checkpoint_dir": "./checkpoints/llama_3b",
      "resume": true,
      "load_from": null,           // explicit step dir; overrides resume
      "save_every": 1000,
      "save_final": true,
      "keep_last": 2               // prune to newest 2
    },
    "profile": { /* torch.profiler settings */ }
  }
}
```

### `source_mode` options
| mode | use |
|---|---|
| `shard_dir` | **Sliding-window chunk files** (downloader + delete-on-consume). Best for long runs / full dataset. |
| `file` | A single growing JSONL (`hf_path`), optionally tailed when `wait_for_data`. |
| `hf` | Stream directly from Hugging Face (`is_local_file: false`, `hf_path` = dataset). |

---

## Checkpointing & data-lifetime semantics

| Item | Behavior |
|---|---|
| What's saved | FSDP2-sharded **model weights + AdamW optimizer state** (DCP). |
| Where | `checkpoint_dir/step_%07d/` (`.metadata` + per-rank `__*.distcp`). |
| Cadence | `save_every` steps + a final save when training stops (`save_final`). |
| Retention | Only the newest `keep_last` dirs remain on disk (rank 0 prunes). |
| Resume | Auto-load latest `step_*` unless `load_from` given or `resume` false. |
| Data position | NOT persisted — a resumed run starts from the earliest **remaining** chunk (deleted chunks are gone). Weights/optimizer resume correctly. |
| Disk cap | Downloader pauses at 3 GB, resumes when trainer deletes back to 2 GB. |

> **Resume recommendation:** keep the same `data_dir` (with any leftover chunks) so a
> restarted run picks up where the previous one left off — the loader always restarts
> from `chunk_00000` if it's still present, so delete stale consumed chunks before
> resuming a long run if you want to avoid re-walking old data.

---

## Local CPU smoke test (no GPU, no network)

```bash
# Builds a tiny offline tokenizer + synthetic data + local_test_config.json
python scripts/prepare_local_test.py

# Run twice to validate fresh + auto-resume
python src/train/fsdp_train.py --config config/local_test_config.json   # fresh
python src/train/fsdp_train.py --config config/local_test_config.json   # resumes
```

---

## Project layout

```
config/train_config.json         # main 3B training config
config/local_test_config.json    # tiny CPU test config (generated)
scripts/
  run.sh                           # single entry: train-only + train+download
  setup_runpod.sh                  # tokenizer + seed data
  prepare_local_test.py          # offline smoke-test assets
src/
  train/fsdp_trainer.py          # FSDP2 trainer
  train/fsdp_train.py            # plain-train entry
  train/fsdp_profile.py          # profiler entry
  train/config.py                # JSON -> dataclasses
  train/dist_dataset_loader.py   # streaming/shard-dir dataset, cross-rank delete
  datautils/download_data.py     # sliding-window HF streamer
  datautils/distributed_checkpoint_manager.py  # DCP save/load
  model/llama.py                 # Llama model (GQA, RoPE, SwiGLU, RMSNorm)
```

## Requirements

- Python ≥ 3.10
- `torch>=2.5.0` (CUDA build on the pod)
- `datasets`, `safetensors`, `transformers`
- (Dev) `pyright`/`ruff` for static checks
