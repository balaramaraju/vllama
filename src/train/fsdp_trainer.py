from __future__ import annotations
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, FSDPModule  # Added FSDPModule wrapper
import torch.distributed.checkpoint as dcp                          # Day 1 Parallel IO engine
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoTokenizer

from src.model.llama import Llama
from src.train.config import load_train_config
from src.train.dist_dataset_loader import (
    GenericStreamingDataset,
    build_distributed_dataloader,
)

class FSDP2Trainer:
    def __init__(self, profile: bool = False, config_path: str | None = None):
        self.profile = profile
        self.cfg = load_train_config() if config_path is None else load_train_config(config_path)
        self.config = self.cfg.model
        self._trace_dir = self.cfg.profile.trace_dir
        
        # New variable tracking paths for Day 1
        self.checkpoint_root = "./checkpoints/llama_3b"
        
        self.isCuda = torch.cuda.is_available()
        if self.isCuda:
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            self.backend = 'nccl'
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cpu")
            self.backend = 'gloo'

    def _setup(self):
        dist.init_process_group(self.backend)
        print(f"Worker process {self.local_rank}/{self.world_size} linked to distributed backbone.")
        
        device_type = "cuda" if self.isCuda else "cpu"
        mesh = init_device_mesh(device_type, (self.world_size,))
        
        model = Llama(config=self.config).to(self.device)
        
        # FSDP2 Bottom-Up Nesting Sharding Loops
        if hasattr(model, "tok_embeddings"):
            fully_shard(model.tok_embeddings, mesh=mesh)
        for llamaBlock in model.blocks:
            fully_shard(llamaBlock, mesh=mesh)
        if hasattr(model, "output"):
            fully_shard(model.output, mesh=mesh)
        fully_shard(model, mesh=mesh)
        
        lr = self.cfg.optimizer.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        if self.isCuda:
            model = torch.compile(model, mode="reduce-overhead")
            if self.rank == 0:
                print("🚀 Graph compiled safely AFTER distributed sharding topologies.")
                
        tokenizer = AutoTokenizer.from_pretrained("./llama_tokenizer_local")
        td = self.cfg.training_data
        
        dataset = GenericStreamingDataset(
            hf_path=td.hf_path,
            split=td.split,
            block_size=td.block_size,
            tokenizer=tokenizer,
            is_local_file=td.is_local_file,
        )
        dataloader = build_distributed_dataloader(dataset=dataset, batch_size=td.batch_size)
        
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataloader = dataloader

    def _save_distributed_checkpoint(self, step: int):
        """Dumps sharded parameter weights and optimizer tracks across all GPUs in parallel."""
        step_dir = os.path.join(self.checkpoint_root, f"step_{step:07d}")
        if self.rank == 0:
            os.makedirs(step_dir, exist_ok=True)
            
        dist.barrier()  # Synchronize before parallel write execution
        
        # Configure FSDP to export sharded tensor references instead of full copies
        with FSDPModule.state_dict_type(self.model, "SHARDED_STATE_DICT"):
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": FSDPModule.optim_state_dict(self.model, self.optimizer)
            }
            dcp.save(state_dict=state_dict, storage_writer=dcp.FileSystemWriter(step_dir))
            
        if self.rank == 0:
            print(f"💾 Distributed parallel DCP state successfully written to: {step_dir}")

    def _forward(self, x, y):
        if self.device.type == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = self.model(x, targets=y)
        else:
            _, loss = self.model(x, targets=y)
        return loss

    def _backward(self, loss):
        loss.backward()

    def _optimizer_step(self):
        self.optimizer.step()

    def _train_step(self, x, y):
        self.optimizer.zero_grad(set_to_none=True) # set_to_none=True saves VRAM cycles
        loss = self._forward(x, y)
        self._backward(loss)
        self._optimizer_step()
        return loss

    def training_loop(self):
        self._setup()
        prof = None
        
        if self.profile:
            os.makedirs(self._trace_dir, exist_ok=True)
            prof_schedule = schedule(
                skip_first=self.cfg.profile.skip_first,
                wait=self.cfg.profile.wait,
                warmup=self.cfg.profile.warmup,
                active=self.cfg.profile.active,
                repeat=self.cfg.profile.repeat,
            )
            if self.rank == 0:
                print("📊 Performance trace collection engine activated...")
                
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.isCuda else [ProfilerActivity.CPU],
                schedule=prof_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self._trace_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False
            )
            prof.__enter__()
            
        try:
            for batch_idx, batch_raw in enumerate(self.dataloader):
                x = batch_raw[:, :-1].to(self.device)
                y = batch_raw[:, 1:].to(self.device).contiguous()
                
                if self.profile:
                    self.optimizer.zero_grad(set_to_none=True)
                    with record_function("forward_pass"):
                        loss = self._forward(x, y)
                    with record_function("backward_pass"):
                        self._backward(loss)
                    with record_function("optimizer_step"):
                        self._optimizer_step()
                    if prof is not None:
                        prof.step()
                else:
                    loss = self._train_step(x, y)
                    
                if self.rank == 0:
                    prefix = "Profile Step" if self.profile else "Step"
                    print(f"{prefix} {batch_idx:04d} | Step Loss: {loss.item():.4f}")
                    
                # Day 1 Save cadence trigger boundary: save sharded states every 50 steps
                if batch_idx > 0 and batch_idx % 50 == 0 and not self.profile:
                    self._save_distributed_checkpoint(step=batch_idx)
                    
                if self.profile and batch_idx >= self.cfg.profile.max_steps:
                    break
                    
        finally:
            if prof is not None:
                prof.__exit__(None, None, None)
                
            # Final fallback checkpoint save at training termination bounds
            if not self.profile:
                self._save_distributed_checkpoint(step=batch_idx)
                
            dist.destroy_process_group()
