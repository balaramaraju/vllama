"""Unit tests for the continuous-batching scheduler and its sequence model.

These validate the two scheduler bug fixes:
  * decode sequences survive across ``schedule()`` calls (the "vanishes after
    the first token" bug); and
  * prefill reserves the worst-case block footprint so running decode sequences
    never exceed their allocated capacity.
"""

from src.inference.continuous_batching.scheduler import Scheduler
from src.inference.continuous_batching.sequence import (
    Sequence,
    SequenceState,
    SequenceStatus,
)
from src.kvcache import BlockSpaceManager


class TestSequence:
    def test_initial_state(self):
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=4, eos_token_id=7)
        assert seq.status == SequenceStatus.WAITING
        assert seq.state == SequenceState.PREFILL
        assert seq.num_prompt_tokens == 3
        assert seq.seq_id >= 0  # auto-assigned unique id

    def test_state_transition_to_decode(self):
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=4, eos_token_id=7)
        seq.add_token(9)  # ordinary token, not EOS
        assert seq.state == SequenceState.DECODE
        assert seq.status == SequenceStatus.WAITING  # not complete yet

    def test_completes_on_eos(self):
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=10, eos_token_id=7)
        seq.add_token(4)
        seq.add_token(7)  # EOS
        assert seq.status == SequenceStatus.COMPLETED

    def test_completes_on_max_tokens(self):
        seq = Sequence(prompt_token_ids=[1, 2], max_tokens=2, eos_token_id=7)
        seq.add_token(9)  # 1/2 new tokens
        assert seq.status == SequenceStatus.WAITING
        seq.add_token(10)  # 2/2 new tokens -> complete
        assert seq.status == SequenceStatus.COMPLETED

    def test_eos_token_id_none_never_completes_on_token(self):
        # None (the common default) must not break the EOS comparison.
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=100, eos_token_id=None)
        seq.add_token(1)
        seq.add_token(2)
        assert seq.status == SequenceStatus.WAITING

    def test_get_positional_id_prefill_vs_decode(self):
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=4, eos_token_id=None)
        # PREFILL covers every prompt position.
        assert seq.get_positional_id() == [0, 1, 2]
        seq.add_token(9)
        # DECODE returns just the newly generated position.
        assert seq.get_positional_id() == [3]

    def test_get_len_and_get_token_ids(self):
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=4, eos_token_id=None)
        assert seq.get_len() == 3
        assert seq.get_token_ids() == [1, 2, 3]

        seq.add_token(9)
        seq.add_token(10)
        assert seq.get_len() == 5
        assert seq.get_token_ids() == [1, 2, 3, 9, 10]


class TestScheduler:
    def _make_scheduler(self, num_blocks=8, block_size=4):
        block_manager = BlockSpaceManager(num_blocks=num_blocks, block_size=block_size)
        return Scheduler(block_manager), block_manager

    def test_started_sequence_keeps_appearing_in_decode(self):
        # Regression for the "sequence vanishes after first token" bug.
        scheduler, _ = self._make_scheduler()
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=4, eos_token_id=None)
        scheduler.add_sequence(seq)

        prefill, decode = scheduler.schedule()
        assert seq in prefill
        assert decode == []
        assert seq in scheduler.running_queue

        # A subsequent schedule must keep serving the sequence as a decode.
        prefill, decode = scheduler.schedule()
        assert prefill == []
        assert seq in decode
        assert seq in scheduler.running_queue

    def test_completed_sequence_is_freed_and_completed(self):
        scheduler, block_manager = self._make_scheduler()
        seq = Sequence(prompt_token_ids=[1, 2], max_tokens=5, eos_token_id=7)
        scheduler.add_sequence(seq)

        scheduler.schedule()
        # 7 worst-case tokens -> 2 blocks used out of 8.
        assert len(block_manager.free_blocks) == 8 - 2

        seq.add_token(7)  # EOS -> completed
        prefill, decode = scheduler.schedule()
        assert prefill == []
        assert decode == []
        assert seq in scheduler.completed_queue
        assert seq not in scheduler.running_queue
        # Blocks are returned to the free pool.
        assert len(block_manager.free_blocks) == 8

    def test_worst_case_reservation_no_growth_needed(self):
        # Worst-case reservation means a running decode never needs extra blocks,
        # so build_slot_mapping can never IndexError mid-generation.
        scheduler, block_manager = self._make_scheduler()
        seq = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=4, eos_token_id=None)
        scheduler.add_sequence(seq)

        scheduler.schedule()
        assert len(block_manager.block_tables[seq.seq_id]) == 2  # ceil(7 / 4)

        # Consume the full budget; capacity was reserved up front.
        for _ in range(seq.max_tokens):
            seq.add_token(9)
        assert seq.status == SequenceStatus.COMPLETED

    def test_oom_second_sequence_stays_waiting(self):
        # Two sequences whose combined worst-case reservations exceed capacity:
        # the first starts, the second remains WAITING.
        scheduler, _ = self._make_scheduler(num_blocks=4, block_size=4)  # 16 slots
        seq1 = Sequence(prompt_token_ids=[1, 2, 3], max_tokens=8, eos_token_id=None)
        seq2 = Sequence(prompt_token_ids=[1, 2, 3, 4], max_tokens=8, eos_token_id=None)
        scheduler.add_sequence(seq1)
        scheduler.add_sequence(seq2)

        prefill, decode = scheduler.schedule()
        assert seq1 in prefill
        assert seq1.status == SequenceStatus.RUNNING
        assert seq1 in scheduler.running_queue
        assert decode == []

        assert seq2.status == SequenceStatus.WAITING
        assert seq2 in scheduler.waiting_queue
