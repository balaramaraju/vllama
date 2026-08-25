import os
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from src.model.llama import Llama
from src.model.config import LlamaConfig, LocalLlamaConfig
from src.train.dist_dataset_loader import GenericStreamingDataset, build_distributed_dataloader
from transformers import AutoTokenizer

class FSDP2Profiler:
    def __init__(self):
        self.isCuda = torch.cuda.is_available()
        if self.isCuda:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            self.backend = 'nccl'
            self.config = LlamaConfig()
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cpu")
            self.backend = 'gloo'
            self.config = LocalLlamaConfig()

    def run_profiler(self):
        dist.init_process_group(self.backend)
        print(f"Profiler worker process {self.local_rank}/{self.world_size} initialized.")

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

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokenizer = AutoTokenizer.from_pretrained("./llama_tokenizer_local")

        dataset = GenericStreamingDataset(
            hf_path="./fineweb_sample_5k.jsonl",
            hf_name=None,
            split="train",
            block_size=128,
            tokenizer=tokenizer,
            is_local_file=True
        )
        dataloader = build_distributed_dataloader(dataset=dataset, batch_size=2)

        os.makedirs("./log/fsdp_profile", exist_ok=True)
        
        prof_schedule = schedule(
            skip_first=10,  
            wait=5,
            warmup=2,
            active=5,       
            repeat=1
        )

        if self.rank == 0:
            print("📊 Performance trace collection engine activated...")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.isCuda else [ProfilerActivity.CPU],
            schedule=prof_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fsdp_profile'),
            record_shapes=True,
            profile_memory=True, # Tracks hardware allocations
            with_stack=False
        ) as prof:

            for batch_idx, batch_raw in enumerate(dataloader):
                x = batch_raw[:, :-1].to(self.device)
                y = batch_raw[:, 1:].to(self.device).contiguous()

                optimizer.zero_grad()
                
                with record_function("forward_pass"):
                    if self.device.type == "cuda":
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits, loss = model(x, targets=y)
                    else:
                        logits, loss = model(x, targets=y)
                        
                with record_function("backward_pass"):
                    loss.backward()
                    
                with record_function("optimizer_step"):
                    optimizer.step()

                prof.step()

                if self.rank == 0:
                    print(f"Profile Step {batch_idx:03d} | Loss: {loss.item():.4f}")

                # Stop early once we capture the target trace metrics window
                if batch_idx >= 25:
                    break

        dist.destroy_process_group()

if __name__ == "__main__":
    profiler = FSDP2Profiler()
    profiler.run_profiler()
