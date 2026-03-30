#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, loss_fn, train_model
from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker6 import ConkerSixConfig, ConkerSixModel, scale_config
from conker.src.golf_data import _load_golf_shard, build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss


def base_config_for_variant(args: argparse.Namespace, seq_len: int) -> ConkerThreeConfig:
    config = ConkerThreeConfig(max_seq_len=seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
    if args.variant == "window4":
        variant_cfg = replace(config, local_window=4)
    elif args.variant == "window16":
        variant_cfg = replace(config, local_window=16)
    elif args.variant == "gated":
        variant_cfg = replace(config, mix_mode="gated")
    elif args.variant == "linear_only":
        variant_cfg = replace(config, enable_local=False)
    elif args.variant == "base":
        variant_cfg = config
    else:
        raise ValueError(f"Unknown Conker-6 base variant: {args.variant}")
    return replace(
        variant_cfg,
        linear_half_life_max=args.linear_half_life_max,
        oscillatory_frac=args.oscillatory_frac,
        oscillatory_period_min=args.oscillatory_period_min,
        oscillatory_period_max=args.oscillatory_period_max,
        static_bank_gate=args.static_bank_gate,
        bank_gate_span=args.bank_gate_span,
        input_proj_scheme=args.input_proj_scheme,
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
            learning_rate=base_train.learning_rate if args.learning_rate is None else args.learning_rate,
            seeds=(args.seed,),
        ),
    )


def evaluate_full_split(model, dataset, runtime: RuntimeConfig, split: str) -> tuple[float, int]:
    files = dataset.train_files if split == "train" else dataset.test_files
    tokens = np.ascontiguousarray(np.concatenate([_load_golf_shard(path) for path in files], axis=0))
    batch_size = runtime.train.batch_size
    seq_len = runtime.train.seq_len
    usable = ((tokens.size - 1) // (batch_size * seq_len)) * (batch_size * seq_len)
    if usable <= 0:
        raise ValueError(f"{split} split is too short for batch_size={batch_size}, seq_len={seq_len}")

    total = 0.0
    num_batches = usable // (batch_size * seq_len)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size * seq_len
        chunk = tokens[start : start + batch_size * seq_len + 1]
        x = mx.array(chunk[:-1].reshape(batch_size, seq_len), dtype=mx.int32)
        y = mx.array(chunk[1:].reshape(batch_size, seq_len), dtype=mx.int32)
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        total += float(loss.item())
    return total / num_batches, int(num_batches * batch_size * seq_len)


def strict_lower_mask(size: int) -> np.ndarray:
    return np.tril(np.ones((size, size), dtype=np.float32), k=-1)


def magnitude_prune(mask: np.ndarray, support: np.ndarray, sparsity: float) -> np.ndarray:
    keep_frac = 1.0 - sparsity
    if keep_frac <= 0.0:
        return np.zeros_like(mask)
    active_vals = np.abs(mask[support > 0])
    if active_vals.size == 0:
        return mask.copy()
    threshold = np.quantile(active_vals, 1.0 - keep_frac)
    kept = np.where((support > 0) & (np.abs(mask) >= threshold), mask, 0.0)
    return kept.astype(np.float32, copy=False)


def row_topk_prune(mask: np.ndarray, support: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    if topk <= 0:
        return out
    for row_idx in range(mask.shape[0]):
        row_support = support[row_idx] > 0
        if not np.any(row_support):
            continue
        cols = np.flatnonzero(row_support)
        row_vals = np.abs(mask[row_idx, cols])
        keep = min(topk, cols.size)
        if keep <= 0:
            continue
        top_cols = cols[np.argpartition(-row_vals, keep - 1)[:keep]]
        out[row_idx, top_cols] = mask[row_idx, top_cols]
    return out


def mask_stats(mask: np.ndarray, support: np.ndarray) -> dict[str, float]:
    active = mask[support > 0]
    nz = np.count_nonzero(active)
    total = active.size
    return {
        "nonzero": int(nz),
        "total": int(total),
        "density": float(nz / total if total else 0.0),
        "l1_mean": float(np.mean(np.abs(active)) if total else 0.0),
        "max_abs": float(np.max(np.abs(active)) if total else 0.0),
    }


def mask_deviation(mask: np.ndarray, baseline: np.ndarray, support: np.ndarray) -> dict[str, float]:
    active_mask = mask[support > 0]
    active_base = baseline[support > 0]
    diff = active_mask - active_base
    l1 = float(np.mean(np.abs(diff)) if diff.size else 0.0)
    l2 = float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0)
    max_abs = float(np.max(np.abs(diff)) if diff.size else 0.0)
    denom = float(np.linalg.norm(active_mask) * np.linalg.norm(active_base))
    if diff.size == 0 or denom == 0.0:
        corr = None
    else:
        corr = float(np.dot(active_mask, active_base) / denom)
    return {
        "mask_l1_deviation": l1,
        "mask_l2_deviation": l2,
        "mask_max_abs_deviation": max_abs,
        "mask_cosine_similarity": corr,
    }


def evaluate_variant(
    model: ConkerSixModel,
    dataset,
    runtime: RuntimeConfig,
    base_state: dict[str, mx.array],
    causal_mask_np: np.ndarray,
    support: np.ndarray,
    variant_name: str,
    transform: str,
) -> dict[str, float]:
    if transform == "baseline":
        new_mask = causal_mask_np
    elif transform == "nonnegative":
        new_mask = np.maximum(causal_mask_np, 0.0).astype(np.float32, copy=False)
    elif transform.startswith("mag"):
        sparsity = float(transform.split("_", 1)[1])
        new_mask = magnitude_prune(causal_mask_np, support, sparsity)
    elif transform.startswith("rowtopk"):
        topk = int(transform.split("_", 1)[1])
        new_mask = row_topk_prune(causal_mask_np, support, topk)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    patched = dict(base_state)
    patched["causal_mask"] = mx.array(new_mask, dtype=mx.float32)
    model.update(nn.utils.tree_unflatten(list(patched.items())))
    full_loss, full_tokens = evaluate_full_split(model, dataset, runtime, "test")
    full_bpt = bits_per_token_from_loss(full_loss)
    full_bpb = full_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    return {
        "variant": variant_name,
        "full_test_eval_loss": float(full_loss),
        "full_test_bits_per_token": float(full_bpt),
        "full_test_bpb": None if full_bpb is None else float(full_bpb),
        "full_test_tokens": int(full_tokens),
        **mask_stats(new_mask, support),
        **mask_deviation(new_mask, causal_mask_np, support),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-6 learned-causal-mask ablation.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--input-proj-scheme", choices=["random", "orthogonal_rows", "kernel_energy", "split_banks"], default="random")
    parser.set_defaults(static_bank_gate=True)
    parser.add_argument("--static-bank-gate", action="store_true", dest="static_bank_gate")
    parser.add_argument("--no-static-bank-gate", action="store_false", dest="static_bank_gate")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--variant", choices=["base", "window4", "window16", "gated", "linear_only"], default="window4")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    base_cfg = base_config_for_variant(args, runtime.train.seq_len)
    config = scale_config(
        ConkerSixConfig(
            base_config=base_cfg,
            freeze_base=True,
            enable_exact3=True,
            exact_context_span=0,
            learnable_vocab_axis=False,
            learnable_causal_mask=True,
            blend_mode="cache_only",
        ),
        args.scale,
    )
    model = ConkerSixModel(vocab_size=dataset.vocab_size, config=config)

    print("\n  conker-6 mask ablation\n")
    print(
        f"  seed={args.seed} steps={runtime.train.steps} seq_len={runtime.train.seq_len} "
        f"batch_size={runtime.train.batch_size} lr={runtime.train.learning_rate:g} "
        f"params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker6_mask_ablation")
    flat_state = dict(nn.utils.tree_flatten(model.parameters()))
    causal_mask_np = np.array(flat_state["causal_mask"], dtype=np.float32, copy=True)
    support = strict_lower_mask(causal_mask_np.shape[0])

    variants = [
        ("baseline", "baseline"),
        ("nonnegative", "nonnegative"),
        ("mag_0.90", "mag_0.90"),
        ("mag_0.95", "mag_0.95"),
        ("mag_0.98", "mag_0.98"),
        ("rowtopk_16", "rowtopk_16"),
        ("rowtopk_8", "rowtopk_8"),
    ]

    rows = []
    for name, transform in variants:
        row = evaluate_variant(model, dataset, runtime, flat_state, causal_mask_np, support, name, transform)
        rows.append(row)
    baseline_bpb = rows[0]["full_test_bpb"]
    for row in rows:
        row["delta_bpb"] = None if baseline_bpb is None else float(row["full_test_bpb"] - baseline_bpb)
        print(
            f"  {row['variant']}: bpb:{row['full_test_bpb']:.4f} "
            f"delta:{row['delta_bpb']:+.4f} "
            f"density:{row['density']:.5f} "
            f"mask_l2:{row['mask_l2_deviation']:.4f}"
        )

    result = {
        "title": "conker-6 mask ablation",
        "config": asdict(runtime),
        "model": {
            "preset": "conker6_mask_ablation",
            "seed": args.seed,
            "params": metrics.params,
            "train_time_sec": float(metrics.train_time_sec),
            "blend_mode": "cache_only",
            "freeze_base": True,
            "learnable_vocab_axis": False,
            "learnable_causal_mask": True,
        },
        "variants": rows,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
