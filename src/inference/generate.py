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

    ids = input_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        logits, _ = model(ids)
    next_id = sample_token(logits, temperature, top_k, top_p)
    ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
    yield next_id.item()
    if eos_token_id is not None and next_id.item() == eos_token_id:
        return

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits, _ = model(ids)
        next_id = sample_token(logits, temperature, top_k, top_p)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
        yield next_id.item()
        if eos_token_id is not None and next_id.item() == eos_token_id:
            return


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
