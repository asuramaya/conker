#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARAMETER_GOLF_MAGIC = 20240520
PARAMETER_GOLF_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize


def _load_golf_shard(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == PARAMETER_GOLF_MAGIC
            and int(header[1]) == PARAMETER_GOLF_VERSION
        ):
            token_count = int(header[2])
            payload = np.frombuffer(blob[HEADER_BYTES:], dtype=np.uint16, count=token_count)
            if payload.size == token_count:
                return payload.astype(np.int32, copy=False)
    return np.frombuffer(blob, dtype=np.uint16).astype(np.int32, copy=False)


def _count_shard_tokens(path: Path) -> int:
    blob = path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == PARAMETER_GOLF_MAGIC
            and int(header[1]) == PARAMETER_GOLF_VERSION
        ):
            return int(header[2])
    return len(blob) // np.dtype(np.uint16).itemsize


def build_shard_index(paths: list[Path]) -> tuple[list[int], int]:
    counts = [_count_shard_tokens(path) for path in paths]
    offsets = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += count
    return offsets, total


def locate_position(offsets: list[int], counts: list[int], pos: int) -> tuple[int, int]:
    lo = 0
    hi = len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start = offsets[mid]
        end = start + counts[mid]
        if start <= pos < end:
            return mid, pos - start
        if pos < start:
            hi = mid - 1
        else:
            lo = mid + 1
    raise IndexError(pos)


def sample_windows(
    paths: list[Path],
    counts: list[int],
    offsets: list[int],
    total: int,
    length: int,
    samples: int,
    rng: random.Random,
) -> list[np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    out: list[np.ndarray] = []
    max_start = total - length
    for _ in range(samples):
        pos = rng.randrange(0, max_start)
        shard_idx, local = locate_position(offsets, counts, pos)
        shard = cache.get(shard_idx)
        if shard is None:
            shard = _load_golf_shard(paths[shard_idx])
            cache[shard_idx] = shard
        if local + length <= shard.shape[0]:
            out.append(np.ascontiguousarray(shard[local : local + length]))
            continue
        remain = length
        cursor = pos
        pieces: list[np.ndarray] = []
        while remain > 0:
            cur_idx, cur_local = locate_position(offsets, counts, cursor)
            cur = cache.get(cur_idx)
            if cur is None:
                cur = _load_golf_shard(paths[cur_idx])
                cache[cur_idx] = cur
            step = min(remain, cur.shape[0] - cur_local)
            pieces.append(cur[cur_local : cur_local + step])
            cursor += step
            remain -= step
        out.append(np.ascontiguousarray(np.concatenate(pieces, axis=0)))
    return out


def hash_window(window: np.ndarray) -> str:
    return hashlib.blake2b(window.astype(np.uint16, copy=False).tobytes(), digest_size=16).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampled train/val exact-window overlap audit for golf token shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--length", type=int, action="append", default=[])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    lengths = args.length or [32, 64, 128, 256]
    root = Path(args.data_root)
    train_paths = sorted(root.glob("fineweb_train_*.bin"))
    val_paths = sorted(root.glob("fineweb_val_*.bin"))
    train_counts = [_count_shard_tokens(path) for path in train_paths]
    val_counts = [_count_shard_tokens(path) for path in val_paths]
    train_offsets, train_total = build_shard_index(train_paths)
    val_offsets, val_total = build_shard_index(val_paths)

    rng = random.Random(args.seed)
    rows = []
    for length in lengths:
        train_windows = sample_windows(train_paths, train_counts, train_offsets, train_total, length, args.train_samples, rng)
        val_windows = sample_windows(val_paths, val_counts, val_offsets, val_total, length, args.val_samples, rng)
        train_hashes = {hash_window(window) for window in train_windows}
        val_hashes = [hash_window(window) for window in val_windows]
        overlap = sum(1 for h in val_hashes if h in train_hashes)
        rows.append(
            {
                "length": length,
                "train_samples": args.train_samples,
                "val_samples": args.val_samples,
                "overlap_count": overlap,
                "overlap_rate": overlap / max(len(val_hashes), 1),
            }
        )

    out = {
        "data_root": str(root),
        "train_total_tokens": train_total,
        "val_total_tokens": val_total,
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
