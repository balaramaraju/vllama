from __future__ import annotations
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoTokenizer

from src.datautils.distributed_checkpoint_manager import DistributedCheckpointManager
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
        self.checkpoint_root = self.cfg.checkpoint.checkpoint_dir

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
        
        # bf16 model params on CUDA so AdamW tiers are allocated in bf16, shrinking
        # both VRAM and the on-disk optimizer state.
        if self.isCuda:
            model = Llama(config=self.config).to(self.device).bfloat16()
        else:
            model = Llama(config=self.config).to(self.device)

        embedding_attr = "embedding" if hasattr(model, "embedding") else "tok_embeddings"
        embed_module = getattr(model, embedding_attr, None)
        if (
            embed_module is not None
            and hasattr(model, "output")
            and model.output.weight is embed_module.weight
        ):
            # Weight-tied output/embedding: same FSDP group so the shared Parameter is sharded once.
            fully_shard([embed_module, model.output], mesh=mesh)
        else:
            if embed_module is not None:
                fully_shard(embed_module, mesh=mesh)
            if hasattr(model, "output"):
                fully_shard(model.output, mesh=mesh)
        for llamaBlock in model.blocks:
            fully_shard(llamaBlock, mesh=mesh)
        fully_shard(model, mesh=mesh)

        lr = self.cfg.optimizer.lr
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            capturable=True if self.isCuda else False,
        )

        self.raw_model = model
        self.optimizer = optimizer
        self.checkpoint_manager = DistributedCheckpointManager(
            self.raw_model, self.optimizer, local_rank=self.local_rank
        )

        self._load_checkpoint_if_needed()

        td = self.cfg.training_data
        tokenizer = AutoTokenizer.from_pretrained(td.tokenizer_path)

        dataset = GenericStreamingDataset(
            hf_path=td.hf_path,
            hf_name=td.hf_name,
            split=td.split,
            block_size=td.block_size,
            tokenizer=tokenizer,
            is_local_file=td.is_local_file,
            wait_for_data=td.wait_for_data,
            poll_interval=td.poll_interval,
            source_mode=td.source_mode,
            data_dir=td.data_dir,
            delete_consumed=td.delete_consumed,
        )
        dataloader = build_distributed_dataloader(dataset=dataset, batch_size=td.batch_size)

        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataloader = dataloader

    def _latest_checkpoint_dir(self) -> str | None:
        """Return the highest-numbered ``step_*`` directory under the checkpoint root, if any."""
        if not os.path.isdir(self.checkpoint_root):
            return None
        step_numbers = []
        for entry in os.listdir(self.checkpoint_root):
            if entry.startswith("step_"):
                suffix = entry[len("step_"):]
                if suffix.isdigit() and os.path.isdir(os.path.join(self.checkpoint_root, entry)):
                    step_numbers.append(int(suffix))
        if not step_numbers:
            return None
        return os.path.join(self.checkpoint_root, f"step_{max(step_numbers):07d}")

    def _load_checkpoint_if_needed(self):
        """Resume from an explicit or latest checkpoint; otherwise start fresh."""
        cfg = self.cfg.checkpoint
        target: str | None = None
        if cfg.load_from:
            if not os.path.isdir(cfg.load_from):
                raise FileNotFoundError(
                    f"checkpoint.load_from is set but directory not found: {cfg.load_from}"
                )
            target = cfg.load_from
        elif cfg.resume:
            target = self._latest_checkpoint_dir()

        if target is None:
            if self.rank == 0:
                print("🌱 No checkpoint found — starting fresh training.")
            return

        if self.rank == 0:
            print(f"📂 Resuming from checkpoint: {target}")
        self.checkpoint_manager.load_checkpoint(target)

    def _save_distributed_checkpoint(self, step: int):
        """Dump sharded weights + optimizer state across all GPUs in parallel."""
        self.checkpoint_manager.save_checkpoint(self.checkpoint_root, step=step)
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        """Keep only the newest ``keep_last`` step_* checkpoints (rank 0 only).

        Deletes only directories matching ``step_<digits>`` under the checkpoint
        root; never touches the root itself. A barrier keeps all ranks aligned
        after the deletion.
        """
        keep = self.cfg.checkpoint.keep_last
        if keep < 0 or not os.path.isdir(self.checkpoint_root) or self.rank != 0:
            return

        step_dirs: list[tuple[int, str]] = []
        for entry in os.listdir(self.checkpoint_root):
            if entry.startswith("step_"):
                suffix = entry[len("step_"):]
                full = os.path.join(self.checkpoint_root, entry)
                if suffix.isdigit() and os.path.isdir(full):
                    step_dirs.append((int(suffix), full))
        step_dirs.sort(reverse=True)

        for _, old_dir in step_dirs[keep:]:
            print(f"🧹 Pruning old checkpoint: {old_dir}")
            dist.barrier()
            import shutil
            shutil.rmtree(old_dir, ignore_errors=True)
            dist.barrier()

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
            
        last_step = -1
        max_steps = self.cfg.training.max_steps
        log_every = max(1, self.cfg.training.log_every)
        try:
            for batch_idx, batch_raw in enumerate(self.dataloader):
                last_step = batch_idx
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

                if self.rank == 0 and batch_idx % log_every == 0:
                    prefix = "Profile Step" if self.profile else "Step"
                    print(f"{prefix} {batch_idx:04d} | Step Loss: {loss.item():.4f}")

                # Periodic checkpoint save (skipped while profiling)
                save_every = self.cfg.checkpoint.save_every
                if (
                    save_every > 0
                    and batch_idx > 0
                    and batch_idx % save_every == 0
                    and not self.profile
                ):
                    self._save_distributed_checkpoint(step=batch_idx)

                # Stop at training.max_steps (or profile.max_steps when profiling).
                limit = self.cfg.profile.max_steps if self.profile else max_steps
                if limit > 0 and batch_idx + 1 >= limit:
                    break

        finally:
            if prof is not None:
                prof.__exit__(None, None, None)

            # Final save unless the last step already hit the save cadence
            if not self.profile and self.cfg.checkpoint.save_final and last_step >= 0:
                save_every = self.cfg.checkpoint.save_every
                already_saved = save_every > 0 and last_step > 0 and last_step % save_every == 0
                if not already_saved:
                    self._save_distributed_checkpoint(step=last_step)

            dist.destroy_process_group()
