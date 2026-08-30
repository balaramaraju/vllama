#!/usr/bin/env python3
"""Sliding-window Hugging Face dataset downloader.

Streams documents from a Hugging Face dataset and writes them to numbered
~50 MB JSONL chunk files under a data directory, so a parallel training process
can begin consuming data before the download finishes.

Disk pacing (hysteresis):
  - The folder grows as chunks are completed.
  - When total folder size exceeds ``max_buffer_gb`` (default 3 GB), the
    downloader pauses.
  - It resumes when the trainer has consumed/deleted enough that the folder
    drops back to ``resume_buffer_gb`` (default 2 GB).

When the stream is exhausted, a ``FINISHED`` marker file is written so readers
know the download completed and can stop waiting for the next chunk.

Usage:
    python src/datautils/download_data.py \\
        --dataset HuggingFaceFW/fineweb \\
        --config-name sample-10BT \\
        --data-dir ./training_data/fineweb \\
        --chunk-size-mb 50 \\
        --max-buffer-gb 3 \\
        --resume-buffer-gb 2
"""
import argparse
import json
import pathlib
import sys
import time

from datasets import load_dataset

CHUNK_PREFIX = "chunk_"
READY_SUFFIX = ".jsonl"
PART_SUFFIX = ".jsonl.part"


def _folder_size_gb(directory: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in directory.iterdir() if f.is_file()) / (1024 ** 3)


def _next_chunk_index(directory: pathlib.Path) -> int:
    n = 0
    for f in directory.glob(f"{CHUNK_PREFIX}*{READY_SUFFIX}"):
        try:
            idx = int(f.name[len(CHUNK_PREFIX):-len(READY_SUFFIX)])
            n = max(n, idx + 1)
        except ValueError:
            continue
    return n


def _wait_until_under(resume_gb: float, directory: pathlib.Path, poll: float = 2.0):
    """Sleep until the data folder drops back to <= resume_gb."""
    while _folder_size_gb(directory) > resume_gb:
        print(
            f"💤 data dir > {resume_gb:.1f} GB ({_folder_size_gb(directory):.2f} GB) — "
            f"pausing download until trainer consumes chunks..."
        )
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a HF dataset into a sliding window of chunk files."
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config-name", default=None, help="Dataset config/subset. None = full dataset.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-dir", default="./training_data/fineweb")
    parser.add_argument("--chunk-size-mb", type=float, default=50.0)
    parser.add_argument("--max-buffer-gb", type=float, default=3.0, help="Pause above this folder size.")
    parser.add_argument("--resume-buffer-gb", type=float, default=2.0, help="Resume once folder <= this size.")
    parser.add_argument("--num-rows", type=int, default=None, help="Optional cap on total rows (None = stream until end/interrupt).")
    parser.add_argument("--text-column", default=None, help="Column holding the text. Auto-detected if omitted.")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = int(args.chunk_size_mb * (1024 ** 2))

    print(
        f"Streaming {args.dataset} ({args.config_name or '<full dataset>'}) "
        f"-> {data_dir}/ (chunk={args.chunk_size_mb:.0f}MB, pause>{args.max_buffer_gb:.1f}GB, resume<={args.resume_buffer_gb:.1f}GB)"
    )
    stream = load_dataset(args.dataset, name=args.config_name, split=args.split, streaming=True)

    text_column: str | None = args.text_column
    manifest_path = data_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        chunk_idx = _next_chunk_index(data_dir)
        part_path = data_dir / f"{CHUNK_PREFIX}{chunk_idx:05d}{PART_SUFFIX}"
        out = part_path.open("w", encoding="utf-8")
        written_in_chunk = 0
        total_written = 0

        for idx, example in enumerate(stream):
            if text_column is None:
                text_column = "text" if "text" in example else next(iter(example))
            row = {text_column: example[text_column]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_in_chunk += 1
            total_written += 1

            # Close + rename this chunk when it reaches the size target.
            if out.tell() >= chunk_bytes:
                out.close()
                ready = data_dir / f"{CHUNK_PREFIX}{chunk_idx:05d}{READY_SUFFIX}"
                part_path.rename(ready)
                manifest.write(
                    json.dumps(
                        {
                            "chunk": f"{chunk_idx:05d}",
                            "rows": written_in_chunk,
                            "bytes": ready.stat().st_size,
                        }
                    )
                    + "\n"
                )
                written_in_chunk = 0
                chunk_idx += 1

                # Pacing: pause above max_buf, resume once training cleans up to resume_buf.
                _wait_until_under(args.resume_buffer_gb, data_dir)

                part_path = data_dir / f"{CHUNK_PREFIX}{chunk_idx:05d}{PART_SUFFIX}"
                out = part_path.open("w", encoding="utf-8")

            if args.num_rows is not None and total_written >= args.num_rows:
                break

        # Flush any partial last chunk.
        if written_in_chunk:
            out.close()
            ready = data_dir / f"{CHUNK_PREFIX}{chunk_idx:05d}{READY_SUFFIX}"
            part_path.rename(ready)
            manifest.write(
                json.dumps({"chunk": f"{chunk_idx:05d}", "rows": written_in_chunk, "bytes": ready.stat().st_size}) + "\n"
            )
        else:
            try:
                out.close()
            except Exception:
                pass

    # Marker that the stream is complete (readers use this to stop waiting).
    (data_dir / "FINISHED").write_text("done\n", encoding="utf-8")
    print(f"✅ Done. {total_written} rows -> {data_dir} (FINISHED marker written)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user; leaving downloaded chunks in place.")
        sys.exit(130)
