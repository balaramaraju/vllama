import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp import FSDPModule

class DistributedCheckpointManager:
    def __init__(self, model, optimizer):
        """
        Staff-Level Checkpoint Manager using native PyTorch Distributed Checkpoint (DCP).
        Handles parallel sharded I/O directly from DTensors to storage paths.
        """
        self.model = model
        self.optimizer = optimizer

    def save_checkpoint(self, checkpoint_root: str, step: int):
        """
        Dumps sharded state weights directly to disk from all GPUs concurrently.
        Zero communication bottleneck; handles multi-billion parameter sizes cleanly.
        """
        # Create a unique directory for this specific checkpoint step
        step_dir = os.path.join(checkpoint_root, f"step_{step:07d}")
        if dist.get_rank() == 0:
            os.makedirs(step_dir, exist_ok=True)
            
        # Ensure all ranks wait until the directory metadata layer is created
        dist.barrier()
        
        # Configure FSDP to export sharded tensor references instead of full copies
        with FSDPModule.state_dict_type(self.model, "SHARDED_STATE_DICT"):
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": FSDPModule.optim_state_dict(self.model, self.optimizer)
            }
            
            # Parallel I/O: Every GPU writes its own chunk to storage at the same time
            dcp.save(
                state_dict=state_dict, 
                storage_writer=dcp.FileSystemWriter(step_dir)
            )
            
        if dist.get_rank() == 0:
            print(f"💾 Distributed parallel DCP state successfully written to: {step_dir}")

    def load_checkpoint(self, step_dir: str):
        """
        Loads sharded weights back into your current multi-GPU cluster mesh setup.
        """
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"Target distributed checkpoint directory missing at: {step_dir}")

        with FSDPModule.state_dict_type(self.model, "SHARDED_STATE_DICT"):
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": FSDPModule.optim_state_dict(self.model, self.optimizer)
            }
            
            # Parallel Read: Every GPU pulls its respective shard directly from storage
            dcp.load(
                state_dict=state_dict, 
                storage_reader=dcp.FileSystemReader(step_dir)
            )
            
            # Rehydrate the model parameters and optimizer states live in memory
            self.model.load_state_dict(state_dict["model"])
            FSDPModule.set_optim_state_dict(self.model, self.optimizer, state_dict["optimizer"])
            
        if dist.get_rank() == 0:
            print(f"✅ State successfully rehydrated from distributed checkpoint: {step_dir}")
