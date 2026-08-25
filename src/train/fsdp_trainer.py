import os
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from src.model.llama import Llama
from src.train.config import load_train_config
from src.train.dist_dataset_loader import GenericStreamingDataset, build_distributed_dataloader
from transformers import AutoTokenizer


class FSDP2Trainer:
    """Shared FSDP2 training logic for both plain training and profiling runs.

    The core training loop lives here exactly once. Set ``profile=True`` to wrap
    each step in torch.profiler ``record_function`` spans and emit TensorBoard
    traces; leave it ``False`` for a plain training run.

    Configuration (dataset, model, optimizer, profiler) is loaded from a JSON
    file with two top-level sections: ``training_data`` and ``config``.
    """

    def __init__(self, profile: bool = False, config_path: str = None):
        self.profile = profile
        self.cfg = load_train_config() if config_path is None else load_train_config(config_path)
        # Typed config object: self.cfg.training_data, self.cfg.model,
        # self.cfg.optimizer, self.cfg.profile.
        self.config = self.cfg.model
        self._trace_dir = self.cfg.profile.trace_dir

        self.isCuda = torch.cuda.is_available()
        if self.isCuda:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
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
        """Build the distributed process group, mesh, sharded model, and
        optimizer / tokenizer / dataset / dataloader. All artifacts are stored
        on ``self`` for reuse by the training loop."""
        dist.init_process_group(self.backend)
        print(f"Worker process {self.local_rank}/{self.world_size} linked to distributed backbone.")

        device_type = "cuda" if self.isCuda else "cpu"
        mesh = init_device_mesh(device_type, (self.world_size,))

        model = Llama(config=self.config).to(self.device)

        if hasattr(model, "tok_embeddings"):
            fully_shard(model.tok_embeddings, mesh=mesh)
        for llamaBlock in model.blocks:
            fully_shard(llamaBlock, mesh=mesh)
        if hasattr(model, "output"):
            fully_shard(model.output, mesh=mesh)
        fully_shard(model, mesh=mesh)

        if self.isCuda:
            model = torch.compile(model, mode="reduce-overhead")
            print("🚀 Graph compiled safely AFTER distributed sharding topologies.")

        lr = self.cfg.optimizer.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        tokenizer = AutoTokenizer.from_pretrained("./llama_tokenizer_local")

        td = self.cfg.training_data
        # We are testing RunPod, following changes are temporary for testing fsdp training.
        dataset = GenericStreamingDataset(
            hf_path=td.hf_path,
            hf_name=td.hf_name,  # Cleaned up for local files
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

    def _forward(self, x, y):
        """Run the forward pass (autocast on CUDA, plain on CPU) and return loss."""
        if self.device.type == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = self.model(x, targets=y)
        else:
            logits, loss = self.model(x, targets=y)
        return loss

    def _backward(self, loss):
        loss.backward()

    def _optimizer_step(self):
        self.optimizer.step()

    def _train_step(self, x, y):
        """Single un-profiled training step: zero-grad -> forward -> backward -> step."""
        self.optimizer.zero_grad()
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
                profile_memory=True,  # Tracks hardware allocations
                with_stack=False
            )
            prof.__enter__()

        try:
            for batch_idx, batch_raw in enumerate(self.dataloader):
                x = batch_raw[:, :-1].to(self.device)
                y = batch_raw[:, 1:].to(self.device).contiguous()

                if self.profile:
                    self.optimizer.zero_grad()
                    with record_function("forward_pass"):
                        loss = self._forward(x, y)
                    with record_function("backward_pass"):
                        self._backward(loss)
                    with record_function("optimizer_step"):
                        self._optimizer_step()
                    prof.step()
                else:
                    loss = self._train_step(x, y)

                if self.rank == 0:
                    prefix = "Profile Step" if self.profile else "Step"
                    print(f"{prefix} {batch_idx:04d} | Step Loss: {loss.item():.4f}")

                # Stop early once we capture the target trace metrics window.
                if self.profile and batch_idx >= self.cfg.profile.max_steps:
                    break
        finally:
            if prof is not None:
                prof.__exit__(None, None, None)

        dist.destroy_process_group()
