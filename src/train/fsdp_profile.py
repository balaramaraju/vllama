from src.train.fsdp_trainer import FSDP2Trainer


class FSDP2Profiler(FSDP2Trainer):
    """FSDP2 training run wrapped in torch.profiler for performance analysis."""

    def __init__(self, config_path: str = None):
        super().__init__(profile=True, config_path=config_path)

    def run_profiler(self):
        self.training_loop()


if __name__ == "__main__":
    profiler = FSDP2Profiler()
    profiler.run_profiler()
