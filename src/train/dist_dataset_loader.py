import os
import json
import time
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import torch.distributed as dist

from transformers import AutoTokenizer


class GenericStreamingDataset(IterableDataset):
    def __init__(
        self,
        hf_path: str,
        hf_name: Optional[str],
        split: str,
        block_size: int,
        tokenizer,
        is_local_file: bool = False,
        wait_for_data: bool = False,
        poll_interval: float = 1.0,
        source_mode: str = "file",        # "file" | "shard_dir" | "hf"
        data_dir: Optional[str] = None,   # used when source_mode == "shard_dir"
        delete_consumed: bool = True,     # delete a shard once all ranks are done with it
    ):
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.split = split
        self.is_local_file = is_local_file
        # block_size + 1 ensures we have enough tokens to split into X and Y pairs later.
        self.block_size = block_size + 1
        self.tokenizer = tokenizer
        self.wait_for_data = wait_for_data
        self.poll_interval = poll_interval
        self.source_mode = source_mode
        self.data_dir = data_dir
        self.delete_consumed = delete_consumed

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.source_mode == "shard_dir":
            assert self.data_dir is not None, "source_mode='shard_dir' requires data_dir"
            yield from self._iter_shard_dir()
        elif self.is_local_file:
            yield from self._iter_local()
        else:
            yield from self._iter_hf()

    def _wait_until_exists(self, path: str):
        """Poll until a file/dir exists (or raise immediately when not waiting)."""
        if not os.path.exists(path):
            if not self.wait_for_data:
                raise FileNotFoundError(f"Local data target missing at {path}")
            while not os.path.exists(path):
                time.sleep(self.poll_interval)

    def _iter_shard_dir(self) -> Iterator[torch.Tensor]:
        """Consume numbered chunk files in sorted order.

        Each chunk is tokenized into blocks. When all ranks finish a chunk, a
        collective barrier fires and rank 0 deletes the file (if delete_consumed).
        Stops when the next chunk is missing AND the ``FINISHED`` marker exists.
        """
        if not self.data_dir:
            raise ValueError("source_mode='shard_dir' requires data_dir")
        dir_: str = self.data_dir
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        finished_marker = os.path.join(dir_, "FINISHED")

        chunk_idx = 0
        while True:
            path = os.path.join(dir_, f"chunk_{chunk_idx:05d}.jsonl")
            # Wait for the next chunk. If not present and FINISHED is written, we're done.
            if not os.path.exists(path):
                if not self.wait_for_data:
                    if os.path.exists(finished_marker):
                        break
                    # All chunks consumed but no FINISHED yet: treat stream as complete
                    # when holding nothing else (covers training finish / smoke tests).
                    if not any(
                        f.startswith("chunk_") and f.endswith(".jsonl")
                        for f in os.listdir(dir_)
                    ):
                        break
                    raise FileNotFoundError(f"Chunk file missing at {path}")
                # wait_for_data path: poll for the chunk or FINISHED.
                while not os.path.exists(path):
                    if os.path.exists(finished_marker):
                        return
                    time.sleep(self.poll_interval)

            yield from self._tokenize_and_shard(self._rows_from_file(path))

            # All ranks are done with this chunk once they've exhausted the iterator.
            if world_size > 1:
                dist.barrier()
            if self.delete_consumed and rank == 0 and os.path.exists(path):
                print(f"[cleanup] Consumed + deleted shard: {path}")
                os.remove(path)
            if world_size > 1:
                dist.barrier()

            chunk_idx += 1

        # We only reach here if there were no chunks and wait_for_data is False.
        print("[shard-dir] exhausted (no chunks, FINISHED present).")
        return

    def _rows_from_file(self, path: str) -> Iterator[str]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield self._extract_text(json.loads(line.strip()))

    def _extract_text(self, example) -> str:
        """Return the text column field from a dict/example."""
        if isinstance(example, dict):
            if "text" in example:
                return example["text"]
            if example:
                return next(iter(example.values()))
        return str(example)

    def _iter_local(self) -> Iterator[torch.Tensor]:
        self._wait_until_exists(self.hf_path)
        yield from self._tokenize_and_shard(self._tail_texts(self.hf_path))

    def _tail_texts(self, path: str) -> Iterator[str]:
        """Yield JSONL text rows, optionally tailing a growing file.

        When ``wait_for_data`` is True, this consumes appended lines until the
        ``<path>.done`` marker (written by the downloader) is present.
        """
        if not self.wait_for_data:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield self._extract_text(json.loads(line.strip()))
            return

        marker = path + ".done"
        pending = ""
        with open(path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(1 << 16)
                if chunk:
                    pending += chunk
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        if line.strip():
                            yield self._extract_text(json.loads(line.strip()))
                else:
                    if os.path.exists(marker):
                        if pending.strip():
                            yield self._extract_text(json.loads(pending.strip()))
                        return
                    time.sleep(self.poll_interval)

    def _iter_hf(self) -> Iterator[torch.Tensor]:
        print(
            f"Opening live stream for {self.hf_path} on process rank "
            f"{dist.get_rank() if dist.is_initialized() else 0}..."
        )
        raw_stream = load_dataset(
            self.hf_path, name=self.hf_name, split=self.split, streaming=True
        )
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            raw_stream = split_dataset_by_node(raw_stream, rank=rank, world_size=world_size)

        yield from self._tokenize_and_shard((self._extract_text(e) for e in raw_stream))

    def _tokenize_and_shard(self, texts: Iterator[str]) -> Iterator[torch.Tensor]:
        """Tokenize a shared text stream, emitting equal block shares per rank.

        Block-level round-robin keeps every rank's batch count identical even when
        doc lengths differ, so collective operations stay aligned across ranks.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        eos_token_id = self.tokenizer.eos_token_id

        buffer = []
        block_index = 0
        for text in texts:
            buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            buffer.append(eos_token_id)
            while len(buffer) >= self.block_size:
                if block_index % world_size == rank:
                    yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
                block_index += 1
                buffer = buffer[self.block_size :]


def build_distributed_dataloader(dataset, batch_size: int):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,     # no worker processes (avoids distributed rank deadlocks)
        pin_memory=False,
        drop_last=True     # drops uneven trailing batches across processes
    )
    return dataloader

