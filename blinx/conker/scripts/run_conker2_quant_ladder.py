#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.conker2 import ConkerTwoConfig, ConkerTwoModel
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import (
    bits_per_token_from_loss,
    estimate_trainable_payload_bytes,
    quantize_trainable_params,
)


def build_runtime(args: argparse.Namespace):
    runtime = RuntimeConfig(profile=args.profile)
    return replace(
        runtime,
        train=replace(
            train_config_for_profile(args.profile),
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=(args.seed,),
        ),
    )


def build_model(vocab_size: int, seq_len: int, linear_modes: int) -> ConkerTwoModel:
    config = ConkerTwoConfig(
        max_seq_len=seq_len,
        linear_modes=linear_modes,
        share_embedding=False,
    )
    return ConkerTwoModel(vocab_size=vocab_size, config=config)


def evaluate_scheme(
    model: nn.Module,
    full_state: dict[str, mx.array],
    trainable_names: set[str],
    dataset,
    runtime,
    bits: int | None,
) -> dict[str, float]:
    if bits is None:
        model.update(nn.utils.tree_unflatten(list(full_state.items())))
        test_eval = evaluate(model, dataset, runtime.train, "test")
        test_bpt = bits_per_token_from_loss(test_eval)
        return {
            "scheme": "fp16",
            "bits": 16.0,
            "test_eval_loss": test_eval,
            "test_bits_per_token": test_bpt,
            "test_bpb": test_bpt * dataset.test_tokens_per_byte,
            "payload_bytes_est": estimate_trainable_payload_bytes(full_state, trainable_names),
            "payload_mb_est": estimate_trainable_payload_bytes(full_state, trainable_names) / (1024.0 * 1024.0),
        }

    quantized_state, stats = quantize_trainable_params(full_state, trainable_names, bits)
    model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
    test_eval = evaluate(model, dataset, runtime.train, "test")
    test_bpt = bits_per_token_from_loss(test_eval)
    result = {
        "scheme": f"uniform_int{bits}",
        "bits": float(bits),
        "test_eval_loss": test_eval,
        "test_bits_per_token": test_bpt,
        "test_bpb": test_bpt * dataset.test_tokens_per_byte,
    }
    result.update(stats)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization ladder for Conker-2 untied_base on official golf shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    model = build_model(dataset.vocab_size, runtime.train.seq_len, args.linear_modes)

    print("\n  conker-2 quant ladder\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size}"
    )
    print(
        f"  train_shards={len(dataset.train_files)} val_shards={len(dataset.test_files)} "
        f"train_tokens={dataset.train_token_count:,} val_tokens={dataset.test_token_count:,}"
    )
    print(f"  variant=untied_base params={count_trainable_params(model):,}")

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker2_quant_ladder")
    train_eval = evaluate(model, dataset, runtime.train, "train")

    flat_full = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    schemes = [None, 8, 6, 4, 3]
    quant_results = []
    for bits in schemes:
        result = evaluate_scheme(model, flat_full, trainable_names, dataset, runtime, bits)
        quant_results.append(result)
        print(
            f"  {result['scheme']}: "
            f"bpb:{result['test_bpb']:.4f} "
            f"bpt:{result['test_bits_per_token']:.4f} "
            f"payload_mb_est:{result['payload_mb_est']:.3f}"
        )

    model.update(nn.utils.tree_unflatten(list(flat_full.items())))

    result = {
        "title": "conker-2 quant ladder cell",
        "config": asdict(runtime),
        "dataset": {
            "source_path": dataset.source_path,
            "tokenizer": dataset.tokenizer,
            "tokenizer_path": dataset.tokenizer_path,
            "train_token_count": int(dataset.train_token_count),
            "test_token_count": int(dataset.test_token_count),
            "test_tokens_per_byte": dataset.test_tokens_per_byte,
            "test_bytes_per_token": dataset.test_bytes_per_token,
        },
        "model": {
            "preset": "conker2",
            "variant": "untied_base",
            "params": metrics.params,
            "seed": args.seed,
            "linear_modes": args.linear_modes,
            "share_embedding": False,
            "linear_impl": "kernel",
            "train_eval_loss": train_eval,
            "train_bits_per_token": bits_per_token_from_loss(train_eval),
            "overfit_pct": metrics.overfit_pct,
            "train_time_sec": metrics.train_time_sec,
        },
        "quantization": quant_results,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
