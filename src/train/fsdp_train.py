import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from src.model.llama import Llama
from src.model.config import LlamaConfig, LocalLlamaConfig
from src.train.dist_dataset_loader import GenericStreamingDataset, build_distributed_dataloader
from transformers import AutoTokenizer

class FSDP2Train:
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

    def training_loop(self):
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

        dataset = GenericStreamingDataset(
            hf_path="HuggingFaceFW/fineweb-edu",
            hf_name="sample-10BT",
            split="train",
            block_size=128,
            tokenizer=tokenizer
        )
        dataloader = build_distributed_dataloader(dataset=dataset, batch_size=2)

        # Training Loop Execution
        for batch_raw in dataloader:
            x = batch_raw[:, :-1].to(self.device)
            y = batch_raw[:, 1:].to(self.device).contiguous()
            
            optimizer.zero_grad()
            
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, targets=y)
            else:
                logits, loss = model(x, targets=y)
                
            loss.backward()
            optimizer.step()
            
            if self.rank == 0:
                print(f"Step Loss: {loss.item():.4f}")

        dist.destroy_process_group()

# This structure is compatible with standard terminal command calls:
# torchrun --nproc_per_node=NUM_GPUS your_script.py
if __name__ == "__main__":
    trainer = FSDP2Train()
    trainer.training_loop()
