from __future__ import annotations
import os
import time
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
    def __init__(self, profile: bool = False, config_path: str | None = None, compile_model: bool = False, accumulation_steps: int = 1, activation_checkpoint: bool = True):
        self.profile = profile
        self.compile_model = compile_model
        self.accumulation_steps = max(1, accumulation_steps)
        self.activation_checkpoint = activation_checkpoint
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
            model = Llama(config=self.config, use_activation_checkpoint=self.activation_checkpoint).to(self.device).bfloat16()
        else:
            model = Llama(config=self.config, use_activation_checkpoint=self.activation_checkpoint).to(self.device)

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
            fully_shard(llamaBlock, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh)

        # Optional torch.compile (opt-in via --compile flag) — wraps the
        # FSDP-sharded model so the compiler sees the full distributed graph.
        if self.compile_model and self.isCuda:
            if self.rank == 0:
                print("⚡ Enabling torch.compile (mode='reduce-overhead') — expect first-step warmup latency.")
            model = torch.compile(model, mode="reduce-overhead")

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

    def _print_training_summary(self):
        """Print a one-shot training configuration banner (rank 0 only)."""
        if self.rank != 0:
            return
        m = self.cfg.model
        td = self.cfg.training_data
        ckpt = self.cfg.checkpoint
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        def _yn(b: bool) -> str:
            return "ON" if b else "OFF"

        device_info = (
            f"CUDA ({torch.cuda.get_device_name(self.device)})"
            if self.isCuda
            else "CPU"
        )

        lines = [
            "",
            "=" * 72,
            "                    TRAINING RUN SUMMARY",
            "=" * 72,
            "",
            "-- Model --",
            f"  Architecture           LLaMA (custom)",
            f"  Parameters             {total_params:,}  ({trainable_params:,} trainable)",
            f"  Vocab / Embed dim      {m.vocab_size:,} / {m.n_embd}",
            f"  Blocks / Heads / KV    {m.n_blocks} / {m.n_heads} / {m.n_kv_heads}",
            f"  Max sequence length    {m.max_seq_len}",
            "",
            "-- Training --",
            f"  Block size (tokens)    {td.block_size}",
            f"  Batch size / GPU       {td.batch_size}",
            f"  Effective global batch {td.batch_size * self.world_size * self.accumulation_steps}",
            f"  Max steps              {self.cfg.training.max_steps}",
            f"  Learning rate          {self.cfg.optimizer.lr}",
            f"  Log every N steps      {self.cfg.training.log_every}",
            "",
            "-- System --",
            f"  Device / Backend       {device_info} / {self.backend}",
            f"  World size (GPUs)      {self.world_size}",
            f"  Precision               bfloat16 (autocast)",
            "",
            "-- Features --",
            f"  Activation checkpoint  {_yn(self.activation_checkpoint)}",
            f"  torch.compile          {_yn(self.compile_model)}",
            f"  Gradient accumulation  {self.accumulation_steps}x",
            f"  FSDP reshard blocks    {_yn(True)}",
            "",
            "-- Data --",
            f"  Source mode            {td.source_mode}",
            f"  Data directory         {td.data_dir}",
            f"  Delete consumed        {_yn(td.delete_consumed)}",
            "",
            "-- Checkpoint --",
            f"  Directory              {ckpt.checkpoint_dir}",
            f"  First save at step     {ckpt.first_save_at}",
            f"  Save every N steps     {ckpt.save_every}",
            f"  Keep last N            {ckpt.keep_last}",
            f"  Resume from ckpt       {_yn(ckpt.resume)}",
            "",
            "=" * 72,
            "",
        ]
        for line in lines:
            print(line, flush=True)

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

    def _cap_trace_dir(self, max_size_gb: float = 2.0):
        """Delete oldest ``*.pt.trace.json`` files in ``_trace_dir`` if total exceeds ``max_size_gb``.

        Keeps the most recent profiles so long-running training doesn't silently fill disk.
        Only rank 0 performs the cleanup; a barrier keeps all ranks aligned afterwards.
        """
        if not os.path.isdir(self._trace_dir) or self.rank != 0:
            return
        trace_files: list[tuple[float, str]] = []
        total_bytes = 0
        for entry in os.listdir(self._trace_dir):
            if not entry.endswith(".pt.trace.json"):
                continue
            full = os.path.join(self._trace_dir, entry)
            try:
                st = os.stat(full)
                trace_files.append((st.st_mtime, full))
                total_bytes += st.st_size
            except OSError:
                continue
        cap_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        if total_bytes <= cap_bytes:
            return
        trace_files.sort()  # oldest first
        print(f"[TRACE CAP] Trace dir at {total_bytes / (1024**3):.1f}GB exceeds {max_size_gb}GB cap -- pruning oldest...")
        for _mtime, path in trace_files:
            if total_bytes <= cap_bytes:
                break
            try:
                sz = os.path.getsize(path)
                os.remove(path)
                total_bytes -= sz
                print(f"   Deleted {os.path.basename(path)} ({sz / (1024**2):.0f}MB)")
            except OSError:
                continue
        if dist.is_initialized():
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
        loss = self._forward(x, y)
        loss = loss / self.accumulation_steps  # normalize for gradient accumulation
        self._backward(loss)
        return loss

    def training_loop(self):
        self._setup()
        self._print_training_summary()
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

                micro_step = batch_idx  # every batch is a micro-batch in accumulation
                step_start = time.perf_counter()

                if self.profile:
                    if micro_step % self.accumulation_steps == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    with record_function("forward_pass"):
                        loss = self._forward(x, y)
                        loss = loss / self.accumulation_steps
                    with record_function("backward_pass"):
                        self._backward(loss)
                    with record_function("optimizer_step"):
                        if (micro_step + 1) % self.accumulation_steps == 0:
                            self._optimizer_step()
                    if prof is not None:
                        prof.step()
                else:
                    if micro_step % self.accumulation_steps == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    loss = self._train_step(x, y)
                    if (micro_step + 1) % self.accumulation_steps == 0:
                        self._optimizer_step()

                if self.rank == 0 and batch_idx % log_every == 0:
                    step_ms = (time.perf_counter() - step_start) * 1000
                    prefix = "Profile Step" if self.profile else "Step"
                    if self.isCuda:
                        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                        print(f"{prefix} {batch_idx:04d} | Loss: {loss.item():.4f} | Time: {step_ms:.1f}ms | Peak VRAM: {peak_mb:.0f}MB")
                        torch.cuda.reset_peak_memory_stats(self.device)
                    else:
                        print(f"{prefix} {batch_idx:04d} | Loss: {loss.item():.4f} | Time: {step_ms:.1f}ms")

                # Periodic checkpoint save (skipped while profiling)
                save_every = self.cfg.checkpoint.save_every
                first_save_at = self.cfg.checkpoint.first_save_at
                do_save = (
                    save_every > 0
                    and batch_idx > 0
                    and not self.profile
                    and (
                        batch_idx == first_save_at
                        or batch_idx % save_every == 0
                    )
                )
                if do_save:
                    self._save_distributed_checkpoint(step=batch_idx)

                # Stop at training.max_steps (or profile.max_steps when profiling).
                limit = self.cfg.profile.max_steps if self.profile else max_steps
                if limit > 0 and batch_idx + 1 >= limit:
                    break

        finally:
            if prof is not None:
                prof.__exit__(None, None, None)
                self._cap_trace_dir()

            # Final save unless the last step already hit the save cadence
            if not self.profile and self.cfg.checkpoint.save_final and last_step >= 0:
                save_every = self.cfg.checkpoint.save_every
                first_save_at = self.cfg.checkpoint.first_save_at
                already_saved = save_every > 0 and last_step > 0 and (
                    last_step == first_save_at or last_step % save_every == 0
                )
                if not already_saved:
                    self._save_distributed_checkpoint(step=last_step)

            dist.destroy_process_group()
