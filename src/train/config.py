import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.model.config import LlamaConfig

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "train_config.json")


@dataclass
class TrainingDataConfig:
    """Dataset / dataloader parameters from the ``training_data`` JSON section."""

    hf_path: str = "./fineweb_sample_5k.jsonl"
    hf_name: Optional[str] = None
    split: str = "train"
    block_size: int = 128
    batch_size: int = 2
    is_local_file: bool = True
    tokenizer_path: str = "./llama_tokenizer_local"
    wait_for_data: bool = True            # poll for the JSONL file if it doesn't exist yet
    poll_interval: float = 1.0            # seconds between file-growth polls in tail mode
    source_mode: str = "file"             # "file" | "shard_dir" | "hf"
    data_dir: Optional[str] = "./training_data/fineweb"
    delete_consumed: bool = True          # delete dataset shards once all ranks finish them


@dataclass
class OptimizerConfig:
    """Optimizer parameters from the ``config.optimizer`` JSON section."""

    lr: float = 1e-3


@dataclass
class TrainingConfig:
    """Training-loop control parameters from the ``config.training`` JSON section."""

    max_steps: int = -1  # -1 = run until the dataset is exhausted
    log_every: int = 1


@dataclass
class CheckpointConfig:
    """Distributed checkpoint (DCP) save / resume parameters."""

    checkpoint_dir: str = "./checkpoints/llama_3b"
    resume: bool = True             # auto-resume the latest step_* under checkpoint_dir
    load_from: Optional[str] = None  # explicit step dir; overrides `resume`
    save_every: int = 50
    save_final: bool = True
    keep_last: int = 2              # retain only the most recent N step_* checkpoints
    first_save_at: int = 500          # first checkpoint step (subsequent cadence = save_every)


@dataclass
class ProfileConfig:
    """Profiler parameters from the ``config.profile`` JSON section."""

    skip_first: int = 10
    wait: int = 5
    warmup: int = 2
    active: int = 5
    repeat: int = 1
    max_steps: int = 25
    trace_dir: str = "./log/fsdp_profile"


@dataclass
class TrainConfig:
    """Top-level training configuration loaded from the JSON file."""

    training_data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    model: LlamaConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


def load_train_config(path: str = DEFAULT_CONFIG_PATH) -> TrainConfig:
    """Load and parse the training configuration JSON into a :class:`TrainConfig`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training config file missing at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    if "training_data" not in data or "config" not in data:
        raise ValueError(
            f"Training config at {path} must contain both 'training_data' and 'config' sections."
        )

    td = data["training_data"]
    training_data = TrainingDataConfig(
        hf_path=td.get("hf_path", TrainingDataConfig.hf_path),
        hf_name=td.get("hf_name", TrainingDataConfig.hf_name),
        split=td.get("split", TrainingDataConfig.split),
        block_size=td.get("block_size", TrainingDataConfig.block_size),
        batch_size=td.get("batch_size", TrainingDataConfig.batch_size),
        is_local_file=td.get("is_local_file", TrainingDataConfig.is_local_file),
        tokenizer_path=td.get("tokenizer_path", TrainingDataConfig.tokenizer_path),
        wait_for_data=td.get("wait_for_data", TrainingDataConfig.wait_for_data),
        poll_interval=td.get("poll_interval", TrainingDataConfig.poll_interval),
        source_mode=td.get("source_mode", TrainingDataConfig.source_mode),
        data_dir=td.get("data_dir", TrainingDataConfig.data_dir),
        delete_consumed=td.get("delete_consumed", TrainingDataConfig.delete_consumed),
    )

    cfg = data["config"]
    model_section = cfg.get("model", {})
    model = LlamaConfig(**model_section)

    opt = cfg.get("optimizer", {})
    optimizer = OptimizerConfig(lr=opt.get("lr", OptimizerConfig.lr))

    train = cfg.get("training", {})
    training = TrainingConfig(
        max_steps=train.get("max_steps", TrainingConfig.max_steps),
        log_every=train.get("log_every", TrainingConfig.log_every),
    )

    ckpt = cfg.get("checkpoint", {})
    checkpoint = CheckpointConfig(
        checkpoint_dir=ckpt.get("checkpoint_dir", CheckpointConfig.checkpoint_dir),
        resume=ckpt.get("resume", CheckpointConfig.resume),
        load_from=ckpt.get("load_from", CheckpointConfig.load_from),
        save_every=ckpt.get("save_every", CheckpointConfig.save_every),
        save_final=ckpt.get("save_final", CheckpointConfig.save_final),
        keep_last=ckpt.get("keep_last", CheckpointConfig.keep_last),
        first_save_at=ckpt.get("first_save_at", CheckpointConfig.first_save_at),
    )

    prof = cfg.get("profile", {})
    profile = ProfileConfig(
        skip_first=prof.get("skip_first", ProfileConfig.skip_first),
        wait=prof.get("wait", ProfileConfig.wait),
        warmup=prof.get("warmup", ProfileConfig.warmup),
        active=prof.get("active", ProfileConfig.active),
        repeat=prof.get("repeat", ProfileConfig.repeat),
        max_steps=prof.get("max_steps", ProfileConfig.max_steps),
        trace_dir=prof.get("trace_dir", ProfileConfig.trace_dir),
    )

    return TrainConfig(
        training_data=training_data,
        model=model,
        optimizer=optimizer,
        training=training,
        checkpoint=checkpoint,
        profile=profile,
    )
