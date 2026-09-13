from src.inference.continuous_batching.sequence import (
    Sequence,
    SequenceState,
    SequenceStatus,
)
from src.kvcache.kv_cache import BlockSpaceManager


class Scheduler:
    """Manages active, pending, and completed execution queues across virtual memory blocks."""

    def __init__(self, block_manager: BlockSpaceManager):
        self.block_manager = block_manager

        self.waiting_queue: list[Sequence] = []
        self.running_queue: list[Sequence] = []
        self.completed_queue: list[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        """Pushes an incoming request directly onto the waiting stack."""
        seq.state = SequenceState.PREFILL
        seq.status = SequenceStatus.WAITING
        self.waiting_queue.append(seq)

    def _can_allocate(self, seq: Sequence) -> bool:
        """Checks if the physical block pool has enough remaining space for the request.

        The capacity is computed against the sequence's *worst-case* footprint
        (``num_prompt_tokens + max_tokens``), matching the reservation made in
        :meth:`schedule`. ``max_tokens`` is the new-token budget, so the total
        token count a sequence can ever reach is prompt length plus that budget.
        """
        max_total_tokens = seq.get_len() + (seq.max_tokens - len(seq.output_token_ids))
        blocks_needed = (
            max_total_tokens + self.block_manager.block_size - 1
        ) // self.block_manager.block_size

        # Deduct blocks already assigned to see what new allocations are requested
        allocated_blocks = len(self.block_manager.block_tables.get(seq.seq_id, []))
        new_blocks_needed = blocks_needed - allocated_blocks

        return len(self.block_manager.free_blocks) >= new_blocks_needed

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """
        Pulls items out from waiting or running stacks based on physical resource constraints.
        Returns:
            (prefill_seqs, decode_seqs): Two lists dividing the active execution batch.
        """
        prefill_seqs: list[Sequence] = []
        decode_seqs: list[Sequence] = []

        still_running: list[Sequence] = []

        # 1. Check running requests first to prevent mid-generation starvation dropouts
        for seq in self.running_queue:
            if seq.status == SequenceStatus.COMPLETED:
                self.block_manager.free_sequence(seq.seq_id)
                self.completed_queue.append(seq)
            elif self._can_allocate(seq):
                seq.status = SequenceStatus.RUNNING
                decode_seqs.append(seq)
                still_running.append(seq)
            else:
                # Preemption / Eviction path (if space completely fills up, fallback safely)
                seq.status = SequenceStatus.WAITING
                seq.state = SequenceState.PREFILL
                self.waiting_queue.insert(0, seq)
                self.block_manager.free_sequence(seq.seq_id)

        self.running_queue = still_running

        while self.waiting_queue:
            next_seq = self.waiting_queue[0]

            if self._can_allocate(next_seq):
                next_seq = self.waiting_queue.pop(0)
                next_seq.state = SequenceState.PREFILL
                next_seq.status = SequenceStatus.RUNNING

                # Reserve the worst-case footprint up front so running decode
                # sequences never need growth blocks (which would otherwise
                # trigger a build_slot_mapping IndexError at the prefill size).
                reserved_tokens = next_seq.num_prompt_tokens + next_seq.max_tokens
                self.block_manager.allocate(next_seq.seq_id, reserved_tokens)

                if next_seq.state == SequenceState.PREFILL:
                    prefill_seqs.append(next_seq)
                else:
                    decode_seqs.append(next_seq)

                # Keep the sequence in the running set for the next schedule()
                # so its decode continues instead of vanishing after one token.
                still_running.append(next_seq)
            else:
                break

        return prefill_seqs, decode_seqs
