#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover
    spm = None


PARAMETER_GOLF_MAGIC = 20240520
PARAMETER_GOLF_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize


def load_golf_shard(path: Path) -> np.ndarray:
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


def load_split_tokens(data_root: Path, split: str, token_limit: int | None) -> np.ndarray:
    pattern = "fineweb_train_*.bin" if split == "train" else "fineweb_val_*.bin"
    files = sorted(data_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shards matched {pattern} under {data_root}")
    chunks: list[np.ndarray] = []
    total = 0
    for path in files:
        tokens = load_golf_shard(path)
        if token_limit is None:
            chunks.append(tokens)
            continue
        left = token_limit - total
        if left <= 0:
            break
        take = min(left, int(tokens.size))
        chunks.append(tokens[:take])
        total += take
        if total >= token_limit:
            break
    if not chunks:
        raise ValueError(f"No tokens loaded for split={split} token_limit={token_limit}")
    return np.ascontiguousarray(np.concatenate(chunks, axis=0))


def build_sentencepiece_luts(
    sp: "spm.SentencePieceProcessor", vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def tokens_per_byte(tokens: np.ndarray, tokenizer_path: Path, vocab_size: int) -> float:
    if spm is None:
        raise RuntimeError("sentencepiece is required to compute tokens_per_byte")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, vocab_size)
    prev_ids = tokens[:-1]
    tgt_ids = tokens[1:]
    bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
    bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16, copy=False)
    total_tokens = float(tgt_ids.size)
    total_bytes = float(bytes_np.astype(np.float64).sum())
    return total_tokens / total_bytes


def encode_context(tokens: np.ndarray, end: int, order: int, vocab_size: int) -> int:
    key = 0
    start = end - order
    for idx in range(start, end):
        key = key * vocab_size + int(tokens[idx])
    return key


@dataclass
class EvalResult:
    order: int
    mode: str
    discount: float
    tokens_scored: int
    bits_per_token: float
    bpb: float | None
    elapsed_sec: float
    fallback_histogram: dict[str, int]
    distinct_contexts: dict[str, int]


def eval_online_ngram(
    tokens: np.ndarray,
    vocab_size: int,
    max_order: int,
    mode: str,
    discount: float,
    tok_per_byte: float | None,
) -> EvalResult:
    tables: list[dict[int, dict[int, int]]] = [defaultdict(dict) for _ in range(max_order + 1)]
    totals: list[dict[int, int]] = [defaultdict(int) for _ in range(max_order + 1)]
    losses = 0.0
    scored = 0
    fallback_hist = {str(i): 0 for i in range(max_order + 1)}
    start = time.time()

    def prob_for_target(target: int, order: int, pos: int) -> float:
        if order <= 0 or pos - order < 0:
            fallback_hist["0"] += 1
            return 1.0 / vocab_size
        key = encode_context(tokens, pos, order, vocab_size)
        row = tables[order].get(key)
        total = totals[order].get(key, 0)
        if not row or total <= 0:
            return prob_for_target(target, order - 1, pos)
        if mode == "hard":
            count = row.get(target, 0)
            if count > 0:
                fallback_hist[str(order)] += 1
                return count / total
            return prob_for_target(target, order - 1, pos)
        lower = prob_for_target(target, order - 1, pos)
        count = row.get(target, 0)
        if mode == "witten_bell":
            distinct = len(row)
            lam = total / (total + distinct) if (total + distinct) > 0 else 0.0
            mle = count / total
            fallback_hist[str(order)] += 1
            return lam * mle + (1.0 - lam) * lower
        if mode == "absolute_discount":
            distinct = len(row)
            fallback_hist[str(order)] += 1
            return max(count - discount, 0.0) / total + (discount * distinct / total) * lower
        raise ValueError(f"Unknown mode: {mode}")

    for pos in range(1, int(tokens.size)):
        target = int(tokens[pos])
        p = max(prob_for_target(target, max_order, pos), 1e-12)
        losses += -math.log2(p)
        scored += 1
        max_ctx = min(max_order, pos)
        for order in range(1, max_ctx + 1):
            key = encode_context(tokens, pos, order, vocab_size)
            row = tables[order][key]
            row[target] = row.get(target, 0) + 1
            totals[order][key] += 1

    elapsed = time.time() - start
    bpt = losses / max(scored, 1)
    return EvalResult(
        order=max_order,
        mode=mode,
        discount=discount,
        tokens_scored=scored,
        bits_per_token=bpt,
        bpb=None if tok_per_byte is None else bpt * tok_per_byte,
        elapsed_sec=elapsed,
        fallback_histogram=fallback_hist,
        distinct_contexts={str(i): len(tables[i]) for i in range(1, max_order + 1)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-6 legal online n-gram cache probe.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--token-limit", type=int, default=204800)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", default=["hard", "witten_bell", "absolute_discount"])
    parser.add_argument("--discount", type=float, default=0.75)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    tokens = load_split_tokens(data_root, args.split, args.token_limit)
    tokenizer_path = data_root.parents[1] / "tokenizers" / "fineweb_1024_bpe.model"
    tok_per_byte = None
    if args.split == "test" and tokenizer_path.exists():
        tok_per_byte = tokens_per_byte(tokens, tokenizer_path, args.vocab_size)

    results = []
    for order in args.orders:
        for mode in args.modes:
            result = eval_online_ngram(
                tokens=tokens,
                vocab_size=args.vocab_size,
                max_order=order,
                mode=mode,
                discount=args.discount,
                tok_per_byte=tok_per_byte,
            )
            results.append(
                {
                    "order": result.order,
                    "mode": result.mode,
                    "discount": result.discount,
                    "tokens_scored": result.tokens_scored,
                    "bits_per_token": result.bits_per_token,
                    "bpb": result.bpb,
                    "elapsed_sec": result.elapsed_sec,
                    "fallback_histogram": result.fallback_histogram,
                    "distinct_contexts": result.distinct_contexts,
                }
            )
            print(
                f"order={order} mode={mode} "
                f"bpt={result.bits_per_token:.4f} "
                f"bpb={(result.bpb if result.bpb is not None else float('nan')):.4f} "
                f"time={result.elapsed_sec:.1f}s"
            )

    out = {
        "title": "conker-6 legal online ngram cache probe",
        "data_root": str(data_root),
        "split": args.split,
        "token_limit": int(tokens.size),
        "vocab_size": args.vocab_size,
        "tokens_per_byte": tok_per_byte,
        "discount": args.discount,
        "results": results,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
