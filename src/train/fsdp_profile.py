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

    def __init__(self, config_path: str | None = None, compile_model: bool = False, accumulation_steps: int = 1, activation_checkpoint: bool = True):
        super().__init__(profile=True, config_path=config_path, compile_model=compile_model, accumulation_steps=accumulation_steps, activation_checkpoint=activation_checkpoint)

    def run_profiler(self):
        self.training_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 training with torch.profiler tracing.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a train_config.json. Defaults to config/train_config.json.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (mode='reduce-overhead') on CUDA. Expect first-step warmup latency.",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="Gradient accumulation steps (simulate larger batch with same VRAM).",
    )
    parser.add_argument(
        "--no-activation-checkpoint",
        action="store_true",
        help="Disable activation checkpointing (faster, higher VRAM). Default: checkpoint enabled.",
    )
    args = parser.parse_args()
    profiler = FSDP2Profiler(config_path=args.config, compile_model=args.compile, accumulation_steps=args.accumulate, activation_checkpoint=not args.no_activation_checkpoint)
    profiler.run_profiler()
