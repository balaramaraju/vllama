"""Unit tests for the paged KV-cache storage and block manager."""

import pytest
import torch

from src.kvcache import BlockSpaceManager, KVCacheMemory


@pytest.fixture
def block_manager():
    return BlockSpaceManager(num_blocks=4, block_size=4)


class TestBlockSpaceManager:
    def test_allocate_rounds_up_blocks(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=1)
        assert len(block_manager.block_tables[0]) == 1

        block_manager.allocate(seq_id=1, num_tokens=5)
        assert len(block_manager.block_tables[1]) == 2

    def test_allocate_grows_existing_sequence(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=4)
        assert len(block_manager.block_tables[0]) == 1

        block_manager.allocate(seq_id=0, num_tokens=8)
        assert len(block_manager.block_tables[0]) == 2

    def test_allocate_raises_when_out_of_memory(self, block_manager):
        with pytest.raises(RuntimeError):
            # 4 blocks * 4 slots = 16 slots; 17 tokens needs 5 blocks.
            block_manager.allocate(seq_id=0, num_tokens=17)

    def test_free_sequence_returns_blocks(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=6)
        assert len(block_manager.free_blocks) == 2  # used 2 of 4

        block_manager.free_sequence(seq_id=0)
        assert len(block_manager.free_blocks) == 4
        assert 0 not in block_manager.block_tables

    def test_free_missing_sequence_is_noop(self, block_manager):
        block_manager.free_sequence(seq_id=99)  # should not raise

    def test_build_slot_mapping(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=6)
        # seq 0 has 2 blocks (indices 0 and 1 from the free list, popped LIFO:
        # 0..3 in order, so first pops 3, then 2).
        slots = block_manager.build_slot_mapping([0, 0, 0], [0, 4, 5])
        assert slots.dtype == torch.long
        assert slots.tolist() == [3 * 4 + 0, 2 * 4 + 0, 2 * 4 + 1]

    def test_build_slot_mapping_missing_sequence(self, block_manager):
        with pytest.raises(KeyError):
            block_manager.build_slot_mapping([0], [0])

    def test_build_slot_mapping_out_of_capacity(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=4)
        with pytest.raises(IndexError):
            block_manager.build_slot_mapping([0], [4])

    def test_get_block_table_tensor_pads(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=4)
        block_manager.allocate(seq_id=1, num_tokens=6)
        table = block_manager.get_block_table_tensor([0, 1])
        assert table.dtype == torch.int32
        assert table.shape == (2, 2)
        # seq 0 has 1 block, padded to -1.
        assert table[0, 1].item() == -1

    def test_get_block_table_tensor_empty(self, block_manager):
        table = block_manager.get_block_table_tensor([])
        assert table.shape == (0, 0)

    def test_get_cache_efficiency(self, block_manager):
        block_manager.allocate(seq_id=0, num_tokens=6)  # 2 blocks * 4 = 8 slots
        eff = block_manager.get_cache_efficiency([0], torch.tensor([4]))
        assert eff == pytest.approx(4 / 8 * 100.0)

    def test_get_cache_efficiency_multiple_sequences(self):
        manager = BlockSpaceManager(num_blocks=4, block_size=4)
        manager.allocate(seq_id=0, num_tokens=4)  # 1 block * 4 = 4 slots
        manager.allocate(seq_id=1, num_tokens=8)  # 2 blocks * 4 = 8 slots
        # active = 2 + 6 = 8, allocated = 4 + 8 = 12
        eff = manager.get_cache_efficiency([0, 1], torch.tensor([2, 6]))
        assert eff == pytest.approx(8 / 12 * 100.0)

    def test_get_cache_efficiency_missing_sequence(self, block_manager):
        with pytest.raises(KeyError):
            block_manager.get_cache_efficiency([0], torch.tensor([1]))


class TestKVCacheMemory:
    def _make_cache(self, **kwargs):
        defaults = dict(
            num_blocks=4,
            block_size=4,
            num_layers=2,
            n_kv_heads=3,
            n_head_dims=8,
            device="cpu",
        )
        defaults.update(kwargs)
        return KVCacheMemory(**defaults)

    def test_shapes(self):
        cache = self._make_cache()
        assert cache.key_cache.shape == (2, 4, 4, 3, 8)
        assert cache.value_cache.shape == (2, 4, 4, 3, 8)
        assert cache.total_slots == 16

    def test_write_and_read_roundtrip(self):
        cache = self._make_cache()
        key = torch.randn(2, 3, 8)
        value = torch.randn(2, 3, 8)
        slot_mapping = torch.tensor([3, 9], dtype=torch.long)

        cache.write_kv(key, value, slot_mapping, layer_idx=1)

        k_layer, v_layer = cache.get_key_value(1)
        flat_k = k_layer.view(cache.total_slots, 3, 8)
        assert torch.allclose(flat_k[3], key[0])
        assert torch.allclose(flat_k[9], key[1])
        flat_v = v_layer.view(cache.total_slots, 3, 8)
        assert torch.allclose(flat_v[9], value[1])

    def test_dtype_mismatch_raises(self):
        cache = self._make_cache(dtype=torch.float32)
        key = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        value = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        with pytest.raises(TypeError):
            cache.write_kv(key, value, torch.tensor([0]), layer_idx=0)

    def test_head_dim_mismatch_raises(self):
        cache = self._make_cache(n_kv_heads=3, n_head_dims=8)
        key = torch.randn(1, 5, 8)  # wrong n_kv_heads
        value = torch.randn(1, 5, 8)
        with pytest.raises(ValueError):
            cache.write_kv(key, value, torch.tensor([0]), layer_idx=0)

    def test_layer_index_bounds(self):
        cache = self._make_cache()
        key = torch.randn(1, 3, 8)
        with pytest.raises(IndexError):
            cache.write_kv(key, key, torch.tensor([0]), layer_idx=2)
        with pytest.raises(IndexError):
            cache.get_key_value(layer_idx=2)

    def test_block_manager_to_cache_roundtrip(self):
        # End-to-end: allocate blocks, build a slot mapping, and scatter K/V
        # into the cache — the same flow the inference engine will use.
        manager = BlockSpaceManager(num_blocks=4, block_size=4, device="cpu")
        cache = self._make_cache(num_blocks=4, block_size=4)

        manager.allocate(seq_id=0, num_tokens=6)
        slot_mapping = manager.build_slot_mapping([0, 0, 0], [0, 4, 5])

        key = torch.randn(3, 3, 8)
        value = torch.randn(3, 3, 8)
        cache.write_kv(key, value, slot_mapping, layer_idx=0)

        flat_k = cache.key_cache[0].view(cache.total_slots, 3, 8)
        for i, slot in enumerate(slot_mapping.tolist()):
            assert torch.allclose(flat_k[slot], key[i])
