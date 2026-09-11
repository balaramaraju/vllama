"""Model-level regression tests for the custom Llama.

These exercise the real :class:`Llama` forward pass (as opposed to the
storage/block-manager layer covered in ``test_kv_cache.py``) and guard against
regressions in the attention mask and the optional ``position_ids`` API.
"""

import pytest
import torch

from src.model.llama import Llama
from src.kvcache import BlockSpaceManager, KVCacheMemory


class _MiniConfig:
    vocab_size = 32000
    n_embd = 64
    n_blocks = 2
    n_heads = 4
    n_kv_heads = 2
    max_seq_len = 2048
    rope_theta = 10000.0
    rms_norm_eps = 1e-6


@pytest.fixture
def mini_config():
    return _MiniConfig()


def _instrument_mask(model):
    """Wrap the first block's attention to capture the mask Llama.forward built."""
    captured = {}

    def _make_spy(orig):
        def spy(*args, **kwargs):
            if kwargs.get("mask") is not None:
                captured["mask"] = kwargs["mask"].clone()
            return orig(*args, **kwargs)

        return spy

    model.blocks[0].attention.forward = _make_spy(model.blocks[0].attention.forward)
    return captured


class TestMaskPadding:
    def test_mixed_length_decode_masks_padding_columns(self, mini_config):
        """Shorter sequences must get -inf on padding columns, not unmasked zeros."""
        model = Llama(mini_config, use_activation_checkpoint=False)
        captured = _instrument_mask(model)

        batch, block_size, num_blocks = 2, 4, 64
        manager = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size)
        kv = KVCacheMemory(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=mini_config.n_blocks,
            n_kv_heads=mini_config.n_kv_heads,
            n_head_dims=mini_config.n_embd // mini_config.n_heads,
            device="cpu",
        )
        seq_ids = [0, 1]
        for seq_id in seq_ids:
            manager.allocate(seq_id=seq_id, num_tokens=32)

        # Mixed-length decode: seq 0 has 11 history slots, seq 1 has 21.
        ids = torch.tensor([[111], [222]])
        position_ids = torch.tensor([[10], [20]])
        slot_mapping = manager.build_slot_mapping(seq_ids, [10, 20])
        block_tables = manager.get_block_table_tensor(seq_ids)
        context_lens = torch.tensor([11, 21])

        with torch.no_grad():
            logits, _ = model(
                ids,
                position_ids=position_ids,
                kv_memory=kv,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        mask = captured["mask"]
        assert mask.shape == (batch, 1, 1, 21)
        # seq 0 valid history stays causally unmasked (0)
        assert torch.all(mask[0, 0, 0, :11] == 0)
        # seq 0 padding columns must be fully masked (-inf)
        assert torch.all(mask[0, 0, 0, 11:] == float("-inf"))
        # seq 1 has no padding: all valid, all unmasked
        assert torch.all(mask[1, 0, 0, :21] == 0)
        assert torch.isfinite(logits).all()

    def test_homogeneous_prefill_produces_causal_triangle(self, mini_config):
        """With equal-length prefill, the mask is the standard upper-tri causal shape."""
        model = Llama(mini_config, use_activation_checkpoint=False)
        captured = _instrument_mask(model)

        batch, prompt_len, block_size, num_blocks = 2, 5, 4, 64
        manager = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size)
        kv = KVCacheMemory(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=mini_config.n_blocks,
            n_kv_heads=mini_config.n_kv_heads,
            n_head_dims=mini_config.n_embd // mini_config.n_heads,
            device="cpu",
        )
        seq_ids = [0, 1]
        flat_seq_ids, flat_positions = [], []
        for seq_id in seq_ids:
            for pos in range(prompt_len):
                flat_seq_ids.append(seq_id)
                flat_positions.append(pos)
            manager.allocate(seq_id=seq_id, num_tokens=prompt_len)

        ids = torch.randint(0, mini_config.vocab_size, (batch, prompt_len))
        position_ids = torch.arange(prompt_len).unsqueeze(0).repeat(batch, 1)
        slot_mapping = manager.build_slot_mapping(flat_seq_ids, flat_positions)
        block_tables = manager.get_block_table_tensor(seq_ids)
        context_lens = torch.full((batch,), prompt_len, dtype=torch.long)

        with torch.no_grad():
            _, _ = model(
                ids,
                position_ids=position_ids,
                kv_memory=kv,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        mask = captured["mask"][0, 0]  # (prompt_len, prompt_len)
        assert mask.shape == (prompt_len, prompt_len)
        # Strictly-upper-triangle masked (-inf); below-and-on the diagonal unmasked.
        upper = torch.triu(torch.ones_like(mask), diagonal=1).bool()
        assert torch.isneginf(mask[upper]).all()
        assert torch.all(mask.tril() == 0)


class TestPositionIdsOpt:
    """When position_ids is omitted, training/smoke-test call sites must work."""

    def test_omitted_position_ids_training_step(self, mini_config):
        model = Llama(mini_config, use_activation_checkpoint=True)
        x = torch.randint(0, mini_config.vocab_size, (4, 16))
        y = torch.randint(0, mini_config.vocab_size, (4, 16))
        logits, loss = model(x, targets=y)
        assert logits.shape == (4, 16, mini_config.vocab_size)
        assert loss is not None and torch.isfinite(loss)

    def test_omitted_position_ids_eager(self, mini_config):
        model = Llama(mini_config, use_activation_checkpoint=False)
        x = torch.randint(0, mini_config.vocab_size, (4, 16))
        y = torch.randint(0, mini_config.vocab_size, (4, 16))
        logits, loss = model(x, targets=y)
        assert logits.shape == (4, 16, mini_config.vocab_size)
        assert loss is not None and torch.isfinite(loss)
