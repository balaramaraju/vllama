"""Autoregressive text generation for the repo's custom ``Llama`` model.

Milestone 1: naive generation — each step re-runs the full sequence through the
model (no KV cache yet). This is intentionally simple; KV caching arrives in a
later milestone and will slot into :func:`generate`.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.inference.smol_config import make_smol2_config
from src.model.llama import Llama
from src.kvcache.kv_cache import BlockSpaceManager, KVCacheMemory

def load_model(model_dir: str, device: torch.device, dtype: torch.dtype) -> Llama:
    """Build the repo model and load the converted SmolLM2 weights."""
    model = Llama(config=make_smol2_config(), use_activation_checkpoint=False).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    sd = dict(load_file(os.path.join(model_dir, "model.safetensors")))
    sd["output.weight"] = sd["embedding.weight"]  # re-tie the output head
    model.load_state_dict(sd)
    model.eval()
    return model


def load_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def compute_kv_cache_blocks(model, config, gpu_utilization=0.90, block_size=16) -> int:
    """Derive the number of KV-cache blocks from available GPU memory.

    Surplus VRAM (the fraction ``gpu_utilization`` of total memory after the
    model weights and a framework-activation overhead allowance) is divided by
    the byte footprint of a single KV block. Falls back to a fixed count on
    non-CUDA devices where VRAM isn't measurable.
    """
    # 1. Measure total physical GPU memory boundaries.
    device = next(model.parameters()).device
    if device.type != "cuda":
        return 512  # Fallback limit if running non-accelerated CPU pipelines

    total_vram = torch.cuda.get_device_properties(device).total_memory

    # 2. Track model weight space footprint natively.
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters())

    # 3. Reserve framework activation overhead boundaries (e.g., ~1-2 GB).
    framework_overhead = 1.5 * 1024 * 1024 * 1024

    # Determine cache headroom allocations.
    cache_headroom = (total_vram * gpu_utilization) - model_mem - framework_overhead

    # 4. Compute single block byte weight footprint sizes.
    bytes_per_element = 2  # Assuming BF16/FP16 precision types.
    head_dim = config.n_embd // config.n_heads
    bytes_per_block = (
        2 * config.n_blocks * config.n_kv_heads * head_dim * block_size * bytes_per_element
    )

    num_blocks = int(cache_headroom // bytes_per_block)
    return max(1, num_blocks)


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    """Sample one next-token id from the final-position logits [batch, vocab]."""
    logits = logits[..., -1, :].float()

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift so we always keep at least the single most-likely token.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_stream(
    model: Llama,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time.

    The first yield is produced by the compute-bound prefill forward over the
    full prompt; every subsequent yield is one memory-bound decode step. This
    split is what :class:`ServingPerformanceProfiler` times (TTFT vs ITL).
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if max_new_tokens <= 0:
        return

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    config = model.config
    
    batch_size, prompt_len = input_ids.shape
    seq_ids = list(range(int(batch_size)))
    # Track the current context length of each sequence in the batch
    current_lens = torch.tensor([int(prompt_len)] * int(batch_size), dtype=torch.long, device=device)
    
    # KV page size is an explicit cache config value (config.kv_block_size) —
    # distinct from the dataset "block_size" the training configs use, which is
    # the per-sample sequence length. The number of physical cache blocks is
    # derived from the model + available GPU memory rather than hardcoded.
    block_size = int(config.kv_block_size)
    head_dim = int(config.n_embd // config.n_heads)
    num_total_blocks = compute_kv_cache_blocks(model, config, block_size=block_size)

    block_manager = BlockSpaceManager(num_blocks=num_total_blocks, block_size=block_size, device=device)
    kv_memory = KVCacheMemory(
        num_blocks=num_total_blocks,
        block_size=block_size,
        num_layers=int(config.n_blocks),
        n_kv_heads=int(config.n_kv_heads),
        n_head_dims=head_dim,
        device=device,
        dtype=dtype
    )

    # Pre-allocate blocks for each sequence to fit the prompt + max generated tokens
    for seq_id in seq_ids:
        block_manager.allocate(seq_id, int(prompt_len) + max_new_tokens)

    # Initialize the telemetry hook the profiler reads after generation completes.
    model.profiler_metrics = {"kv_cache_efficiency": 0.0, "peak_memory_mb": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    try:
        # ---- PREFILL STAGE ----
        ids = input_ids.to(device)
        
        # Build logical token positions and structural mappings for the prompt
        flat_seq_ids, flat_positions = [], []
        for b_idx, seq_id in enumerate(seq_ids):
            for pos in range(int(prompt_len)):
                flat_seq_ids.append(seq_id)
                flat_positions.append(pos)
                
        slot_mapping = block_manager.build_slot_mapping(flat_seq_ids, flat_positions)
        block_tables = block_manager.get_block_table_tensor(seq_ids)
        
        # Continuous sequence position matrix: [0, 1, ..., prompt_len - 1]
        position_ids = torch.arange(0, int(prompt_len), dtype=torch.long, device=device).unsqueeze(0).repeat(int(batch_size), 1)

        with torch.no_grad():
            logits, _ = model(
                tokens=ids,
                position_ids=position_ids,
                kv_memory=kv_memory,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=current_lens
            )
            
        next_id = sample_token(logits, temperature, top_k, top_p)
        yield int(next_id.item())
        
        if eos_token_id is not None and next_id.item() == eos_token_id:
            return

        # ---- DECODE STAGE ----
        for _ in range(max_new_tokens - 1):
            # Input changes from the full cumulative sequence array down to just the single newest token ID
            ids = next_id.unsqueeze(-1)
            
            # The token's position matches its current position in the sequence timeline
            position_ids = current_lens.unsqueeze(-1)
            
            # Advance the timeline tracker to account for the new token
            current_lens += 1
            
            # Map the precise allocation slot for the single incoming element
            slot_mapping = block_manager.build_slot_mapping(seq_ids, position_ids.squeeze(-1).tolist())
            block_tables = block_manager.get_block_table_tensor(seq_ids)
            
            with torch.no_grad():
                logits, _ = model(
                    tokens=ids,
                    position_ids=position_ids,
                    kv_memory=kv_memory,
                    slot_mapping=slot_mapping,
                    block_tables=block_tables,
                    context_lens=current_lens
                )
                
            next_id = sample_token(logits, temperature, top_k, top_p)
            yield int(next_id.item())
            
            if eos_token_id is not None and next_id.item() == eos_token_id:
                return
    finally:
        # ---- TELEMETRY CAPTURE ----
        # Runs here (before deallocation) so it also covers early EOS stops.
        model.profiler_metrics["kv_cache_efficiency"] = block_manager.get_cache_efficiency(seq_ids, current_lens)
        model.profiler_metrics["peak_memory_mb"] = (
            torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )

        # ---- SAFE DEALLOCATION HOOKS ----
        # Ensures that block arrays are returned to the free tracking stack 
        # even if generation errors out mid-sentence loop execution.
        for seq_id in seq_ids:
            block_manager.free_sequence(seq_id)

def generate(
    model: Llama,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Greedy/sampled autoregressive decode (naive: full forward each step).

    Thin wrapper over :func:`generate_stream` that returns the full id tensor.
    """
    device = next(model.parameters()).device
    base = input_ids.to(device)
    new_tokens = list(
        generate_stream(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
    )
    if new_tokens:
        ext = torch.tensor([new_tokens], dtype=torch.long, device=device)
        return torch.cat([base, ext], dim=-1)
    return base
