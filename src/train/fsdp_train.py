import argparse
import os
import sys

# Make the `src` package importable when running this file directly or via torchrun.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.fsdp_trainer import FSDP2Trainer


class FSDP2Train(FSDP2Trainer):
    """Plain FSDP2 multi-GPU training run (no profiling)."""

    def __init__(self, config_path: str | None = None):
        super().__init__(profile=False, config_path=config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 multi-GPU training (no profiling).")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a train_config.json. Defaults to config/train_config.json.",
    )
    args = parser.parse_args()
    trainer = FSDP2Train(config_path=args.config)
    trainer.training_loop()
