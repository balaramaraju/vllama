"""Continuous-batching inference engine for the paged KV cache.

This is the integration glue between :class:`Scheduler`, :class:`Sequence`,
and the paged-cache aware ``Llama.forward``. Each ``schedule()`` iteration
hands us a set of prefill sequences (full prompts) and decode sequences
(single newest token each), which we consume in two homogeneous matrix calls:

  * prefill sequences are processed one forward at a time — vLLM batches these
    by padding to a common length, but this model cannot mask out the padded
    tokens in its mask/KV-reconstruction paths, so individual forwards are the
    correct (if slightly slower) choice;
  * decode sequences are batched together as a rectangular ``[n_decode, 1]``
    tensor, exactly the layout ``generate_stream`` already uses.

Both forwards keep the proven ``[batch, len]`` "one row per sequence" layout:
``block_tables[i]`` and ``context_lens[i]`` must index the same row as sequence
``i``. Flattening all sequences into one row would corrupt the per-sequence KV
reconstruction in the attention block.
"""

from __future__ import annotations

import torch

from src.inference.continuous_batching.scheduler import Scheduler
from src.inference.continuous_batching.sequence import Sequence
from src.inference.generate import compute_kv_cache_blocks, sample_token
from src.kvcache.kv_cache import BlockSpaceManager, KVCacheMemory
from src.model.llama import Llama


@torch.inference_mode()
def run_continuous_batching_loop(
    model: Llama,
    tokenizer,
    prompts: list[list[int]],
    max_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = 0.9,
) -> list[list[int]]:
    """Run a continuous iteration-level scheduling loop over ``prompts``.

    Simulates vLLM-style incoming request consumption over the paged cache:
    each iteration schedules prefill and decode work, executes the model, and
    samples a token for every active sequence. Returns the full token stream
    (prompt + generated) for each input prompt.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    config = model.config

    # 1. Initialize paged memory from the model + config (not hardcoded).
    # ``kv_block_size`` is the paged-attention page size; ``block_size`` on the
    # config is the *dataset* sequence length and must not be used here.
    block_size = int(config.kv_block_size)
    num_total_blocks = compute_kv_cache_blocks(model, config, block_size=block_size)
    head_dim = int(config.n_embd // config.n_heads)

    block_manager = BlockSpaceManager(
        num_blocks=num_total_blocks, block_size=block_size, device=device
    )
    kv_memory = KVCacheMemory(
        num_blocks=num_total_blocks,
        block_size=block_size,
        num_layers=int(config.n_blocks),
        n_kv_heads=int(config.n_kv_heads),
        n_head_dims=head_dim,
        device=device,
        dtype=dtype,
    )

    scheduler = Scheduler(block_manager=block_manager)

    # 2. Queue all incoming requests.
    sequences: list[Sequence] = []
    for prompt in prompts:
        seq = Sequence(
            prompt_token_ids=prompt,
            max_tokens=max_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        scheduler.add_sequence(seq)
        sequences.append(seq)

    # 3. Iterate until no more schedulable work exists.
    #    Break when the scheduler returns nothing: either everything completed,
    #    or remaining waiting sequences cannot be allocated (OOM).
    while True:
        prefill_seqs, decode_seqs = scheduler.schedule()
        if not prefill_seqs and not decode_seqs:
            break

        # Prefill pass: full prompts, one forward per sequence.
        for seq in prefill_seqs:
            _run_prefill(
                model,
                seq,
                kv_memory,
                block_manager,
                temperature,
                top_k,
                top_p,
                device,
            )

        # Decode pass: one batched forward over all decode sequences.
        if decode_seqs:
            _run_decode_batch(
                model,
                decode_seqs,
                kv_memory,
                block_manager,
                temperature,
                top_k,
                top_p,
                device,
            )

    _warn_about_unstarted(scheduler, sequences)

    return [seq.get_token_ids() for seq in sequences]


def _run_prefill(
    model: Llama,
    seq: Sequence,
    kv_memory: KVCacheMemory,
    block_manager: BlockSpaceManager,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
) -> None:
    """Run the full-prompt forward for one prefill sequence and sample its next token."""
    token_ids = seq.get_token_ids()
    seq_len = len(token_ids)

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)  # [1, seq_len]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    context_lens = torch.tensor([seq_len], dtype=torch.long, device=device)
    slot_mapping = block_manager.build_slot_mapping(
        [seq.seq_id] * seq_len, list(range(seq_len))
    )
    block_tables = block_manager.get_block_table_tensor([seq.seq_id])

    logits, _ = model(
        tokens=ids,
        position_ids=position_ids,
        kv_memory=kv_memory,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
    )

    next_id = sample_token(logits, temperature, top_k, top_p)
    seq.add_token(int(next_id.item()))


def _run_decode_batch(
    model: Llama,
    decode_seqs: list[Sequence],
    kv_memory: KVCacheMemory,
    block_manager: BlockSpaceManager,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
) -> None:
    """Run one batched single-token forward over all decode sequences."""
    if not decode_seqs:
        return

    seq_ids = [seq.seq_id for seq in decode_seqs]
    positions = [seq.get_positional_id()[0] for seq in decode_seqs]

    ids = torch.tensor(
        [[seq.get_token_ids()[-1] for seq in decode_seqs]],
        dtype=torch.long,
        device=device,
    ).t()  # -> [n_decode, 1]
    position_ids = torch.tensor(positions, dtype=torch.long, device=device).unsqueeze(-1)
    context_lens = torch.tensor(
        [seq.get_len() for seq in decode_seqs], dtype=torch.long, device=device
    )
    slot_mapping = block_manager.build_slot_mapping(seq_ids, positions)
    block_tables = block_manager.get_block_table_tensor(seq_ids)

    logits, _ = model(
        tokens=ids,
        position_ids=position_ids,
        kv_memory=kv_memory,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
    )

    next_ids = sample_token(logits, temperature, top_k, top_p)
    for seq, next_id in zip(decode_seqs, next_ids):
        seq.add_token(int(next_id.item()))


def _warn_about_unstarted(scheduler: Scheduler, sequences: list[Sequence]) -> None:
    """Emit a warning if any sequence never ran due to OOM."""
    started = {s.seq_id for s in scheduler.completed_queue} | {
        s.seq_id for s in scheduler.running_queue
    }
    dropped = [s for s in sequences if s.seq_id not in started]
    if dropped:
        import warnings

        warnings.warn(
            f"Continuous-batching loop ended with {len(dropped)} sequence(s) "
            f"never started (block capacity exhausted); their outputs are "
            f"prompt-only.",
            RuntimeWarning,
        )

