from __future__ import annotations

from enum import Enum


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"


class SequenceState(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Sequence:
    """A single inference request tracked across prefill and decode phases.

    ``seq_id`` is auto-assigned from a class-level counter so every sequence is
    guaranteed a unique key for the block manager — callers never have to mint
    one themselves.
    """

    _next_id: int = 0

    def __init__(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        eos_token_id: int | None = None,
    ):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1

        self.prompt_token_ids = prompt_token_ids
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        self.status = SequenceStatus.WAITING
        self.state = SequenceState.PREFILL

        # It will contain both prompt tokens + generated tokens
        self.output_token_ids: list[int] = []

        self.num_prompt_tokens = len(self.prompt_token_ids)

    def get_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_len(self) -> int:
        return len(self.get_token_ids())

    def get_positional_id(self) -> list[int]:
        """
        Determines the structural position indices for RoPE.
        If PREFILL: returns [0, 1, 2, ..., prompt_len-1]
        If DECODE: returns [total_len - 1] (single token dimension offset)
        """
        if self.state == SequenceState.PREFILL:
            return list(range(self.get_len()))
        else:
            return [self.get_len() - 1]

    def add_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

        if self.state == SequenceState.PREFILL:
            self.state = SequenceState.DECODE

        # ``max_tokens`` is the *new-token* budget: the sequence completes once
        # it has generated ``max_tokens`` outputs, regardless of EOS.
        hit_eos = self.eos_token_id is not None and token_id == self.eos_token_id
        if hit_eos or len(self.output_token_ids) >= self.max_tokens:
            self.status = SequenceStatus.COMPLETED
