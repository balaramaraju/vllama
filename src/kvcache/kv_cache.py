"""Paged KV-cache storage and block allocation for the vllama engine.

This module implements the memory-management of "paged attention": KV
tensors live in fixed-size blocks (like OS virtual-memory pages) so that
variable-length sequences don't waste GPU memory on padding. A block manager
hands out physical blocks to sequences and maps logical token positions to
physical slots.

Scope note: this file owns the *storage* and *allocation* layers only. The
actual paged-attention kernel (gathering a sequence's blocks from its block
table and computing attention) is not implemented here and belongs in a
separate module; :meth:`KVCacheMemory.get_key_value` returns the raw layer
cache that such a kernel would index via a per-sequence block table.
"""

from __future__ import annotations

import torch


class Block:
    """A single physical KV-cache block."""

    __slots__ = ("block_idx", "ref_count")

    def __init__(self, block_idx: int):
        self.block_idx = block_idx
        # Number of sequences referencing this block. Reserved for future
        # copy-on-write / prefix sharing (vLLM's ``fork``). Today blocks are
        # owned by exactly one sequence, so this is 0 or 1.
        self.ref_count = 0


class KVCacheMemory:
    """Flat tensor storage for the KV cache, laid out as physical blocks.

    Per-layer shape: ``[num_blocks, block_size, n_kv_heads, n_head_dims]``.
    The cache uses ``n_kv_heads`` (not ``n_heads``) because the model projects
    K/V with grouped-query attention (GQA).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        n_kv_heads: int,
        n_head_dims: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.n_head_dims = n_head_dims
        self.device = torch.device(device)
        self.dtype = dtype

        # Shape: [num_layers, num_blocks, block_size, n_kv_heads, n_head_dims]
        self.key_cache = torch.empty(
            (num_layers, num_blocks, block_size, n_kv_heads, n_head_dims),
            dtype=dtype,
            device=self.device,
        )
        self.value_cache = torch.empty(
            (num_layers, num_blocks, block_size, n_kv_heads, n_head_dims),
            dtype=dtype,
            device=self.device,
        )

    @property
    def total_slots(self) -> int:
        return self.num_blocks * self.block_size

    def write_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Scatter K/V activations into their physical slots.

        Args:
            key/value: ``[num_tokens, n_kv_heads, n_head_dims]``.
            slot_mapping: ``[num_tokens]`` of physical slot indices.
            layer_idx: which transformer layer's cache to write.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        num_tokens, n_kv_heads, head_dim = key.shape
        if n_kv_heads != self.n_kv_heads or head_dim != self.n_head_dims:
            raise ValueError(
                f"Expected key shape [*, {self.n_kv_heads}, {self.n_head_dims}], "
                f"got [*, {n_kv_heads}, {head_dim}]"
            )
        if value.shape != key.shape:
            raise ValueError(f"key/value shape mismatch: {key.shape} vs {value.shape}")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise TypeError(
                f"KV cache dtype {self.dtype} does not match key/value dtype "
                f"{key.dtype}/{value.dtype}"
            )

        slot_mapping = slot_mapping.to(self.device)

        key_target = self.key_cache[layer_idx].view(self.total_slots, n_kv_heads, head_dim)
        key_target[slot_mapping] = key

        value_target = self.value_cache[layer_idx].view(self.total_slots, n_kv_heads, head_dim)
        value_target[slot_mapping] = value

    def get_key_value(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full layer cache as flat physical slots.

        This returns the raw storage a paged-attention kernel would index via a
        per-sequence block table; it does not itself compute attention.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class BlockSpaceManager:
    """Allocates physical KV-cache blocks to sequences (vLLM-style).

    Maps each sequence (``seq_id``) to a *block table*: an ordered list of
    physical blocks whose slots hold that sequence's tokens. Logical token
    position ``p`` lives in logical block ``p // block_size`` at offset
    ``p % block_size``, which maps to physical slot
    ``block.block_idx * block_size + offset``.

    Limitations (documented for clarity):
      * No copy-on-write / prefix sharing yet: each block is owned by exactly
        one sequence, so ``Block.ref_count`` never exceeds 1.
      * Blocks are only ever appended; there is no shrink/defragmentation path.
    """

    def __init__(self, num_blocks: int, block_size: int = 16, device: str | torch.device = "cpu"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.free_blocks: list[Block] = [Block(block_idx=i) for i in range(num_blocks)]
        self.block_tables: dict[int, list[Block]] = {}

    @staticmethod
    def _num_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
        return (num_tokens + block_size - 1) // block_size

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """Ensure ``seq_id`` has enough blocks to hold ``num_tokens`` tokens."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")

        table = self.block_tables.setdefault(seq_id, [])
        total_blocks_needed = self._num_blocks_for_tokens(num_tokens, self.block_size)
        new_blocks_needed = total_blocks_needed - len(table)
        if new_blocks_needed <= 0:
            return

        if len(self.free_blocks) < new_blocks_needed:
            raise RuntimeError(
                f"Out of KV-cache memory: need {new_blocks_needed} block(s), "
                f"only {len(self.free_blocks)} free"
            )

        for _ in range(new_blocks_needed):
            block = self.free_blocks.pop()
            block.ref_count = 1
            table.append(block)

    def free_sequence(self, seq_id: int) -> None:
        """Release all blocks owned by ``seq_id`` back to the free list."""
        table = self.block_tables.pop(seq_id, None)
        if table is None:
            return
        for block in table:
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block)

    def build_slot_mapping(
        self, seq_ids: list[int], token_positions: list[int]
    ) -> torch.Tensor:
        """Map (seq_id, token_pos) pairs to physical slot indices.

        Raises ``KeyError`` if a sequence has no block table and ``IndexError``
        if a token position exceeds the sequence's allocated capacity.
        """
        if len(seq_ids) != len(token_positions):
            raise ValueError("seq_ids and token_positions must have the same length")

        slots = []
        for seq_id, token_pos in zip(seq_ids, token_positions):
            table = self.block_tables.get(seq_id)
            if table is None:
                raise KeyError(f"seq_id {seq_id} has no allocated blocks")

            logical_block_idx = token_pos // self.block_size
            block_offset = token_pos % self.block_size

            if logical_block_idx >= len(table):
                raise IndexError(
                    f"token position {token_pos} for seq_id {seq_id} exceeds "
                    f"allocated capacity ({len(table)} blocks / "
                    f"{len(table) * self.block_size} tokens)"
                )

            physical_block = table[logical_block_idx]
            slots.append(physical_block.block_idx * self.block_size + block_offset)

        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def get_block_table_tensor(self, seq_ids: list[int]) -> torch.Tensor:
        """Return padded ``[batch, max_blocks]`` block-index tensor.

        Rows are padded with ``-1`` so a paged-attention kernel can ignore
        non-allocated blocks. Returns an empty ``[0, 0]`` tensor when
        ``seq_ids`` is empty.
        """
        if not seq_ids:
            return torch.empty((0, 0), dtype=torch.int32, device=self.device)

        tables = []
        for seq_id in seq_ids:
            table = self.block_tables.get(seq_id)
            if table is None:
                raise KeyError(f"seq_id {seq_id} has no allocated blocks")
            tables.append(table)

        max_blocks = max(len(t) for t in tables)

        batch_block_tables = []
        for table in tables:
            block_indices = [b.block_idx for b in table]
            # Pad with -1 so the kernel ignores non-allocated blocks.
            padding = [-1] * (max_blocks - len(block_indices))
            batch_block_tables.append(block_indices + padding)

        return torch.tensor(batch_block_tables, dtype=torch.int32, device=self.device)

    def get_cache_efficiency(self, seq_ids: list[int], current_lens: torch.Tensor) -> float:
        """Return KV-cache memory efficiency as active/allocated tokens (percent).

        ``active_tokens`` is the sum of the sequences' current context lengths;
        ``allocated_tokens`` is the total physical capacity reserved by their
        block tables (``len(table) * block_size`` per sequence). A sequence
        that stopped early (e.g. hit EOS) uses fewer slots than it reserved,
        so the result is ``active / allocated * 100`` and lives in ``[0, 100]``.
        """
        if isinstance(current_lens, torch.Tensor):
            active_tokens = int(current_lens.sum().item())
        else:
            active_tokens = int(sum(current_lens))

        allocated_tokens = 0
        for seq_id in seq_ids:
            table = self.block_tables.get(seq_id)
            if table is None:
                raise KeyError(f"seq_id {seq_id} has no allocated blocks")
            allocated_tokens += len(table) * self.block_size

        if allocated_tokens <= 0:
            return 0.0
        return active_tokens / allocated_tokens * 100.0
