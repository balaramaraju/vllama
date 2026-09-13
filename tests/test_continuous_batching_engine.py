"""Regression test for the continuous-batching engine loop.

Guards the end-to-end glue between the Scheduler, the paged KV cache, and the
model's per-sequence forward layout: each returned stream must start with its
prompt and have generated between 1 and ``max_tokens`` new tokens. A bug where
sequences were flattened into one batch row (or slot mappings misaligned)
surfaces here as a length/prefix mismatch or an exception.
"""

from src.inference.continuous_batching.engine import run_continuous_batching_loop
from src.model.config import LlamaConfig
from src.model.llama import Llama


class _FakeTokenizer:
    eos_token_id = 30210


def _tiny_config():
    return LlamaConfig(
        vocab_size=32000,
        n_embd=64,
        n_blocks=1,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        kv_block_size=4,
    )


def test_engine_generates_across_mixed_batch():
    model = Llama(_tiny_config(), use_activation_checkpoint=False).eval()

    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
    max_tokens = 5
    outputs = run_continuous_batching_loop(
        model,
        _FakeTokenizer(),
        prompts,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=None,
        top_k=50,
    )

    for prompt, out in zip(prompts, outputs):
        assert out[: len(prompt)] == prompt
        generated = len(out) - len(prompt)
        assert 1 <= generated <= max_tokens
