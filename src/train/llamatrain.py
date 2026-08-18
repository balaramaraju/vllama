import os
import time
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from src.model.llama import Llama
from src.model.config import LlamaConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FineWebEduDataset(IterableDataset):
    def __init__(self, hf_path: str, hf_name: str, split: str, block_size: int, tokenizer):
        from datasets import load_dataset
        self.dataset = load_dataset(hf_path, name=hf_name, split=split, streaming=True)
        self.block_size = block_size + 1
        self.tokenizer = tokenizer

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[:self.block_size], dtype=torch.long)
                buffer = buffer[self.block_size:]

def build_dataloader(hf_path, hf_name, split, block_size, batch_size, tokenizer):
    dataset = FineWebEduDataset(hf_path, hf_name, split, block_size, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

def train():
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print("LLaMA Tokenizer loaded successfully.")

    config = LlamaConfig()
    
    model = Llama(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Llama model initialized successfully with {n_params} parameters.")
    
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled with mode='reduce-overhead'")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    dataloader = build_dataloader(
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="sample-10BT",
        split="train",
        block_size=config.max_seq_len,
        batch_size=8,
        tokenizer=tokenizer
    )
    
    print("\nStarting actual training loop with PyTorch Profiler...")
    
    prof_schedule = schedule(
        skip_first=10,
        wait=5,
        warmup=1,
        active=5,
        repeat=1
    )
    
    os.makedirs("./log/llama_profile", exist_ok=True)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/llama_profile'),
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        start_total = time.perf_counter()
        start_fetch = time.perf_counter()
        
        for batch_idx, batch_raw in enumerate(dataloader):
            fetch_time = time.perf_counter() - start_fetch
            print(f"\n--- Batch {batch_idx} (Profiler Step) ---")
            print(f"Data Fetch Time: {fetch_time:.4f} seconds")
            
            start_process = time.perf_counter()
            x = batch_raw[:, :-1].to(device)
            y = batch_raw[:, 1:].to(device)
            
            optimizer.zero_grad()
            
            with record_function("forward_pass"):
                if device.type == "cuda":
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(x, targets=y)
                else:
                    logits, loss = model(x, targets=y)
                    
            with record_function("backward_pass"):
                loss.backward()
                
            with record_function("optimizer_step"):
                optimizer.step()
                
            process_time = time.perf_counter() - start_process
            print(f"Loss: {loss.item():.4f} | Training Step Time: {process_time:.4f} seconds")
            
            prof.step()
            
            if batch_idx >= 22:
                break
                
            start_fetch = time.perf_counter()
            
    print("\n" + "=" * 60)
    print("📊 PyTorch Profiler Summary (Top 15 Operators by CPU/CUDA Time)")
    print("=" * 60)
    sort_metric = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_metric, row_limit=15))
    print("=" * 60)
    
    total_time = time.perf_counter() - start_total
    print(f"\nTotal Profiled Run Execution Time: {total_time:.4f} seconds")

if __name__ == "__main__":
    train()
