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
import numpy as np

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import (
    _dequantize_float_array,
    _quantize_float_array,
    bits_per_token_from_loss,
)


def build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    runtime = RuntimeConfig(profile=args.profile)
    base_train = train_config_for_profile(args.profile)
    return replace(
        runtime,
        train=replace(
            base_train,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_batches=args.eval_batches,
            learning_rate=base_train.learning_rate if args.learning_rate is None else args.learning_rate,
            seeds=(args.seed,),
        ),
    )


def build_model(vocab_size: int, args: argparse.Namespace) -> ConkerThreeModel:
    config = ConkerThreeConfig(max_seq_len=args.seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
    if args.variant == "window4":
        config = replace(config, local_window=4)
    elif args.variant == "gated":
        config = replace(config, mix_mode="gated")
    elif args.variant == "base":
        pass
    else:
        raise ValueError(f"Unsupported audit variant: {args.variant}")
    config = scale_config(config, args.scale)
    return ConkerThreeModel(vocab_size=vocab_size, config=config)


def _matrix_stats(arr: np.ndarray) -> dict[str, float | list[int]]:
    flat = arr.reshape(-1)
    abs_arr = np.abs(arr)
    fro = float(np.linalg.norm(flat))
    max_abs = float(abs_arr.max()) if abs_arr.size else 0.0
    p99 = float(np.quantile(abs_arr, 0.99)) if abs_arr.size else 0.0
    row_norms = np.linalg.norm(arr, axis=1) if arr.ndim >= 2 else np.abs(arr)
    col_norms = np.linalg.norm(arr, axis=0) if arr.ndim >= 2 else np.abs(arr)
    singular_values = np.linalg.svd(arr.astype(np.float32, copy=False), compute_uv=False) if arr.ndim == 2 else np.array([], dtype=np.float32)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    stable_rank = float((fro * fro) / (sigma_max * sigma_max)) if sigma_max > 0.0 else 0.0
    return {
        "shape": list(arr.shape),
        "numel": int(arr.size),
        "fro_norm": fro,
        "max_abs": max_abs,
        "p99_abs": p99,
        "max_over_p99": (max_abs / p99) if p99 > 0.0 else 0.0,
        "row_norm_mean": float(row_norms.mean()) if row_norms.size else 0.0,
        "row_norm_std": float(row_norms.std()) if row_norms.size else 0.0,
        "col_norm_mean": float(col_norms.mean()) if col_norms.size else 0.0,
        "col_norm_std": float(col_norms.std()) if col_norms.size else 0.0,
        "sigma_max": sigma_max,
        "stable_rank": stable_rank,
    }


def _evaluate_bpb(model: nn.Module, dataset, train_config) -> tuple[float, float]:
    dataset.test_stream.reset()
    test_loss = evaluate(model, dataset, train_config, "test")
    test_bpt = bits_per_token_from_loss(test_loss)
    test_bpb = test_bpt * dataset.test_tokens_per_byte
    return test_loss, test_bpb


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-3 weight-geometry audit for low-bit packing.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--variant", choices=["base", "gated", "window4"], default="window4")
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 6])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    model = build_model(dataset.vocab_size, args)

    print("\n  conker-3 geometry audit\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} variant={args.variant} scale={args.scale:.3f} "
        f"steps={runtime.train.steps} eval_batches={runtime.train.eval_batches}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker3_geometry_audit")
    baseline_loss, baseline_bpb = _evaluate_bpb(model, dataset, runtime.train)

    full_state = dict(nn.utils.tree_flatten(model.parameters()))
    trainable = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    trainable_names = [name for name, arr in trainable.items() if mx.issubdtype(arr.dtype, mx.floating)]
    trainable_names.sort(key=lambda name: int(trainable[name].size), reverse=True)
    target_names = trainable_names[: args.top_k]

    report = {
        "title": "conker-3 geometry audit",
        "config": asdict(runtime),
        "model": {
            "variant": args.variant,
            "scale": args.scale,
            "params": count_trainable_params(model),
            "seed": args.seed,
            "test_eval_loss": baseline_loss,
            "test_bpb": baseline_bpb,
            "train_time_sec": metrics.train_time_sec,
        },
        "targets": [],
    }

    for name in target_names:
        arr = trainable[name]
        np_arr = np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)
        entry = {
            "name": name,
            **_matrix_stats(np_arr),
            "quant": [],
        }
        for bits in sorted(set(args.bits)):
            q, scale = _quantize_float_array(arr, bits)
            updated = dict(full_state)
            updated[name] = _dequantize_float_array(q, scale, arr.dtype)
            model.update(nn.utils.tree_unflatten(list(updated.items())))
            q_loss, q_bpb = _evaluate_bpb(model, dataset, runtime.train)
            entry["quant"].append(
                {
                    "bits": bits,
                    "test_eval_loss": q_loss,
                    "test_bpb": q_bpb,
                    "delta_bpb": q_bpb - baseline_bpb,
                    "scale_bytes": int(scale.nbytes),
                }
            )
            model.update(nn.utils.tree_unflatten(list(full_state.items())))
        report["targets"].append(entry)

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  baseline_bpb:{baseline_bpb:.4f} train_time:{metrics.train_time_sec:.1f}s top_k:{len(target_names)}")
    for entry in report["targets"]:
        deltas = " ".join(
            f"int{row['bits']}:{row['delta_bpb']:+.4f}"
            for row in entry["quant"]
        )
        print(f"  {entry['name']}: {deltas}")
    print(f"\n  Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
