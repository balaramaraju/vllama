import os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_saver import save
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter

class DistributedCheckpointManager:
    def __init__(self, model, optimizer):
        """Checkpoint Manager using PyTorch Distributed Checkpoint (DCP)."""
        self.model = model
        self.optimizer = optimizer

    def save_checkpoint(self, checkpoint_root: str, step: int):
        """Dump sharded model + optimizer weights to disk from all GPUs in parallel."""
        step_dir = os.path.join(checkpoint_root, f"step_{step:07d}")
        if dist.get_rank() == 0:
            os.makedirs(step_dir, exist_ok=True)

        # All ranks wait until the directory layer is created before writing.
        dist.barrier()

        # Sharded (DTensor) state dicts; reshardable to a different mesh on load.
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        state_dict = {
            "model": model_state,
            "optimizer": optim_state,
        }

        save(state_dict=state_dict, storage_writer=FileSystemWriter(step_dir))

        if dist.get_rank() == 0:
            print(f"💾 Distributed parallel DCP state successfully written to: {step_dir}")

    def load_checkpoint(self, step_dir: str):
        """Load sharded weights back into the current device mesh."""
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"Target distributed checkpoint directory missing at: {step_dir}")

        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        state_dict = {
            "model": model_state,
            "optimizer": optim_state,
        }

        load(state_dict=state_dict, storage_reader=FileSystemReader(step_dir))

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        if dist.get_rank() == 0:
            print(f"✅ State successfully rehydrated from distributed checkpoint: {step_dir}")
