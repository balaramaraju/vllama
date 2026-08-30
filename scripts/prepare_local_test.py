#!/usr/bin/env python3
"""Build offline assets for a local CPU smoke test of the FSDP2 trainer.

Creates (with no network access):
  - a minimal BPE tokenizer saved to ./llama_tokenizer_local
  - synthetic JSONL training data at ./fineweb_sample_5k.jsonl
  - config/local_test_config.json (tiny model, few steps, quick save cadence)

Run the trainer twice to validate the auto-resume path:

  python scripts/prepare_local_test.py
  python src/train/fsdp_train.py --config config/local_test_config.json   # fresh
  python src/train/fsdp_train.py --config config/local_test_config.json   # resumes
"""
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "llama_tokenizer_local"
DATA_PATH = ROOT / "fineweb_sample_5k.jsonl"
CONFIG_PATH = ROOT / "config" / "local_test_config.json"


def build_tokenizer() -> None:
    if TOK_DIR.joinpath("tokenizer.json").exists():
        print("==> tokenizer already present, skipping")
        return
    print("==> building minimal offline BPE tokenizer")
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<|eot|>"],
        min_frequency=1,
    )
    sample = ["hello world the quick brown fox jumps over the lazy llama test"]
    bpe.train_from_iterator(sample * 200, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        unk_token="<unk>",
        eos_token="<|eot|>",
        bos_token="<|eot|>",
        pad_token="<|eot|>",
    )
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(TOK_DIR))
    print(f"==> tokenizer saved to {TOK_DIR}")


def build_data() -> None:
    if DATA_PATH.exists():
        print("==> data already present, skipping")
        return
    print("==> writing synthetic JSONL training data")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        for i in range(500):
            text = (
                f"Sample document {i}. "
                + "the quick brown fox jumps over the lazy dog while a tiny llama trains. "
                * 20
            )
            f.write(json.dumps({"text": text}) + "\n")
    print(f"==> data saved to {DATA_PATH}")


def build_config() -> None:
    cfg = {
        "training_data": {
            "hf_path": "./fineweb_sample_5k.jsonl",
            "hf_name": None,
            "split": "train",
            "block_size": 64,
            "batch_size": 2,
            "is_local_file": True,
            "tokenizer_path": "./llama_tokenizer_local",
            "wait_for_data": True,
            "poll_interval": 0.1,
        },
        "config": {
            "model": {
                "vocab_size": 128256,
                "n_embd": 128,
                "n_blocks": 2,
                "n_heads": 4,
                "n_kv_heads": 1,
                "max_seq_len": 256,
            },
            "optimizer": {"lr": 0.001},
            "training": {"max_steps": 5, "log_every": 1},
            "checkpoint": {
                "checkpoint_dir": "./checkpoints/llama_local_test",
                "resume": True,
                "load_from": None,
                "save_every": 2,
                "save_final": True,
                "keep_last": 2,
            },
            "profile": {
                "skip_first": 1,
                "wait": 0,
                "warmup": 0,
                "active": 1,
                "repeat": 0,
                "max_steps": 3,
                "trace_dir": "./log/local_test",
            },
        },
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"==> config saved to {CONFIG_PATH}")


if __name__ == "__main__":
    build_tokenizer()
    build_data()
    build_config()
    print("==> done. Run the trainer twice to exercise fresh + resume:")
    print("    python src/train/fsdp_train.py --config config/local_test_config.json")