import os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_saver import save
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter

class DistributedCheckpointManager:
    def __init__(self, model, optimizer, local_rank: int = 0):
        """Checkpoint Manager using PyTorch Distributed Checkpoint (DCP).

        Tracks the local device rank so cross-GPU synchronization barriers can be
        issued against the correct device, preventing distributed freezes.
        """
        self.model = model
        self.optimizer = optimizer
        self.local_rank = local_rank

    def _sync(self):
        """Barrier across ranks. `device_ids` is NCCL-only, so it is skipped on gloo/CPU."""
        if torch.cuda.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])
        elif dist.is_initialized():
            dist.barrier()

    def save_checkpoint(self, checkpoint_root: str, step: int):
        """Dump sharded state weights and optimizer tracks to disk in parallel from all GPUs."""
        step_dir = os.path.join(checkpoint_root, f"step_{step:07d}")

        if dist.get_rank() == 0:
            os.makedirs(step_dir, exist_ok=True)

        # All ranks wait until the directory metadata layer exists before writing.
        self._sync()

        # FSDP2 get_state_dict already yields SHARDED (DTensor) state dicts, so no
        # duplicate full-size weight dumps occur across ranks.
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        state_dict = {
            "model": model_state,
            "optimizer": optim_state,
        }

        save(state_dict=state_dict, storage_writer=FileSystemWriter(step_dir))

        # Clear network channels after the heavy parallel write.
        self._sync()

        if dist.get_rank() == 0:
            print(f"💾 Distributed parallel DCP state successfully written to: {step_dir}")

    def load_checkpoint(self, step_dir: str):
        """Load sharded parameter shards back into the active distributed device mesh layers."""
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
