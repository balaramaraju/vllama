import argparse
import os
import sys

# Make the `src` package importable when running this file directly or via torchrun.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.train.fsdp_trainer import FSDP2Trainer


class FSDP2Profiler(FSDP2Trainer):
    """FSDP2 training run wrapped in torch.profiler for performance analysis."""

    def __init__(self, config_path: str | None = None):
        super().__init__(profile=True, config_path=config_path)

    def run_profiler(self):
        self.training_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 training with torch.profiler tracing.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a train_config.json. Defaults to config/train_config.json.",
    )
    args = parser.parse_args()
    profiler = FSDP2Profiler(config_path=args.config)
    profiler.run_profiler()
