import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

class GenericStreamingDataset(IterableDataset):
    def __init__(self, hf_path: str, hf_name: str, split: str, block_size: int, tokenizer):
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.split = split
        # block_size + 1 ensures we have enough tokens to split into X and Y pairs later
        self.block_size = block_size + 1
        self.tokenizer = tokenizer

    def __iter__(self):
        # 1. Open the stream from Hugging Face (Zero download wait, zero high RAM use)
        print(f"Opening live stream for {self.hf_path} on process rank {dist.get_rank() if dist.is_initialized() else 0}...")
        raw_stream = load_dataset(self.hf_path, name=self.hf_name, split=self.split, streaming=True)
        
        # 2. Split the stream safely between your distributed workers (No duplicates!)
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            raw_stream = split_dataset_by_node(raw_stream, rank=rank, world_size=world_size)
            
        # 3. Stream, tokenize, and pack tokens on the fly
        column_names = list(raw_stream.features.keys()) if hasattr(raw_stream, 'features') and raw_stream.features else ["text"]
        text_column = "text" if "text" in column_names else column_names[0]
        eos_token_id = self.tokenizer.eos_token_id
        
        buffer = []
        for example in raw_stream:
            text_content = example[text_column]
            tokens = self.tokenizer.encode(text_content, add_special_tokens=False)
            tokens.append(eos_token_id)
            buffer.extend(tokens)
            
            # While we have enough tokens for a full block, yield it
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[:self.block_size], dtype=torch.long)
                buffer = buffer[self.block_size:]


def build_distributed_dataloader(dataset, batch_size: int):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,    # Set to 0 for simple local CPU testing to prevent deadlocks
        pin_memory=False,  # Set to False for local CPU simulation
        drop_last=True     # Drops uneven trailing batches across processes
    )
    return dataloader

#----- Local test ---- TODO : Delete
def run_worker_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    print(f"Worker process {rank}/{world_size} initialized successfully.")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    dataset = GenericStreamingDataset(
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="sample-10BT",
        split="train",         # We don't slice strings here; we stream directly
        block_size=128,
        tokenizer=tokenizer
    )
    
    dataloader = build_distributed_dataloader(dataset=dataset, batch_size=2)
    
    data_iter = iter(dataloader)
    
    for step in range(2):
        batch = next(data_iter)
        print(f"[Step {step} | Worker Rank {rank}] Fetched batch shape: {list(batch.shape)} | First Token ID: {batch[0, 0].item()}")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 3
    print(f"🚀 Starting a local CPU test mimicking {WORLD_SIZE} distributed streaming workers...")
    mp.spawn(
        run_worker_test,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True
    )
    print("✅ Local streaming distributed simulation completed successfully!")
