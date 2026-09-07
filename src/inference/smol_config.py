"""SmolLM2-360M(-Instruct) configuration for the repo's custom ``Llama`` model.

The values mirror ``HuggingFaceTB/SmolLM2-360M(-Instruct)`` ``config.json`` so the
custom model can load the pretrained weights 1:1. The base and Instruct checkpoints
share this exact architecture; only the tokenizer (special tokens / chat template)
differs between them.
"""

from src.model.config import LlamaConfig

# Architecture constants taken from HuggingFaceTB/SmolLM2-360M config.json.
SMOL2_360M = {
    "vocab_size": 49152,
    "n_embd": 960,
    "n_blocks": 32,
    "n_heads": 15,
    "n_kv_heads": 5,
    "max_seq_len": 8192,
    "rope_theta": 100000.0,
    "rms_norm_eps": 1e-5,
}

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def make_smol2_config() -> LlamaConfig:
    """Build a :class:`LlamaConfig` matching the SmolLM2-360M architecture."""
    return LlamaConfig(**SMOL2_360M)
