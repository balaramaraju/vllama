from src.train.fsdp_trainer import FSDP2Trainer


class FSDP2Train(FSDP2Trainer):
    """Plain FSDP2 multi-GPU training run (no profiling)."""

    def __init__(self, config_path: str = None):
        super().__init__(profile=False, config_path=config_path)


if __name__ == "__main__":
    trainer = FSDP2Train()
    trainer.training_loop()
