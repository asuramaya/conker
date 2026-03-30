#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

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
    cosine = None if diff.size == 0 or denom == 0.0 else float(np.dot(active_mask, active_base) / denom)
    return {
        "mask_l1_deviation": l1,
        "mask_l2_deviation": l2,
        "mask_max_abs_deviation": max_abs,
        "mask_cosine_similarity": cosine,
    }


def evaluate_variant(
    model: ConkerSixModel,
    dataset,
    runtime: RuntimeConfig,
    base_state: dict[str, mx.array],
    causal_mask_np: np.ndarray,
    support: np.ndarray,
    variant_name: str,
    new_mask: np.ndarray,
) -> dict[str, float]:
    patched = dict(base_state)
    patched["causal_mask"] = mx.array(new_mask.astype(np.float32, copy=False), dtype=mx.float32)
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


def toeplitz_mean(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        out += np.diag(np.full(size - lag, float(np.mean(vals)), dtype=np.float32), k=-lag)
    return out * support


def toeplitz_band(mask: np.ndarray, support: np.ndarray, max_lag: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    size = mask.shape[0]
    max_lag = min(max_lag, size - 1)
    for lag in range(1, max_lag + 1):
        vals = np.diag(mask, k=-lag)
        out += np.diag(np.full(size - lag, float(np.mean(vals)), dtype=np.float32), k=-lag)
    return out * support


def lowrank_masked(mask: np.ndarray, support: np.ndarray, rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(mask, full_matrices=False)
    recon = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    return (recon.astype(np.float32, copy=False) * support).astype(np.float32, copy=False)


def row_normalized(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = mask.astype(np.float32, copy=True)
    row_sums = np.sum(np.abs(out), axis=1, keepdims=True)
    target = float(np.mean(row_sums[row_sums > 0])) if np.any(row_sums > 0) else 1.0
    out = out / np.maximum(row_sums, 1e-8) * target
    return out * support


def save_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(matrix.tolist())


def save_lag_profile_csv(path: Path, mask: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    size = mask.shape[0]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lag", "count", "mean", "std", "min", "max"])
        for lag in range(1, size):
            vals = np.diag(mask, k=-lag)
            row = {
                "lag": lag,
                "count": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            rows.append(row)
            writer.writerow([row["lag"], row["count"], row["mean"], row["std"], row["min"], row["max"]])
    return rows


def save_vector_csv(path: Path, values: np.ndarray, header: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header])
        for value in values.tolist():
            writer.writerow([float(value)])


def _to_grayscale_image(matrix: np.ndarray, path: Path, scale: int = 4) -> None:
    clipped = matrix.astype(np.float32, copy=False)
    clipped = clipped - np.min(clipped)
    denom = float(np.max(clipped))
    norm = clipped / denom if denom > 0 else clipped
    pixels = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
    image.save(path)


def _abs_diff_image(a: np.ndarray, b: np.ndarray, path: Path, scale: int = 4) -> None:
    diff = np.abs(a - b).astype(np.float32, copy=False)
    _to_grayscale_image(diff, path, scale=scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-6 mask dump, visualization, and structure attacks.")
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
    parser.add_argument("--out-prefix", required=True)
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

    print("\n  conker-6 mask geometry\n")
    print(
        f"  seed={args.seed} steps={runtime.train.steps} seq_len={runtime.train.seq_len} "
        f"batch_size={runtime.train.batch_size} lr={runtime.train.learning_rate:g} "
        f"params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker6_mask_geometry")
    flat_state = dict(nn.utils.tree_flatten(model.parameters()))
    causal_mask_np = np.array(flat_state["causal_mask"], dtype=np.float32, copy=True)
    support = strict_lower_mask(causal_mask_np.shape[0])

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_prefix.with_suffix(".mask.npy"), causal_mask_np)
    save_matrix_csv(out_prefix.with_suffix(".mask.csv"), causal_mask_np)
    lag_rows = save_lag_profile_csv(out_prefix.with_suffix(".lag_profile.csv"), causal_mask_np)
    singular_values = np.linalg.svd(causal_mask_np, compute_uv=False)
    save_vector_csv(out_prefix.with_suffix(".singular_values.csv"), singular_values, "singular_value")

    lag_mask = toeplitz_mean(causal_mask_np, support)
    _to_grayscale_image(causal_mask_np, out_prefix.with_suffix(".mask.png"))
    _to_grayscale_image(lag_mask, out_prefix.with_suffix(".lag_mean_mask.png"))
    _abs_diff_image(causal_mask_np, lag_mask, out_prefix.with_suffix(".lag_mean_diff.png"))

    base_state = dict(flat_state)
    variants = [
        ("baseline", causal_mask_np),
        ("toeplitz_mean", lag_mask),
        ("toeplitz_band_32", toeplitz_band(causal_mask_np, support, 32)),
        ("toeplitz_band_64", toeplitz_band(causal_mask_np, support, 64)),
        ("lowrank_8_masked", lowrank_masked(causal_mask_np, support, 8)),
        ("lowrank_16_masked", lowrank_masked(causal_mask_np, support, 16)),
        ("row_normalized", row_normalized(causal_mask_np, support)),
    ]

    results = []
    baseline_bpb = None
    for name, variant_mask in variants:
        result = evaluate_variant(model, dataset, runtime, base_state, causal_mask_np, support, name, variant_mask)
        if baseline_bpb is None:
            baseline_bpb = result["full_test_bpb"]
            result["delta_bpb"] = 0.0
        else:
            result["delta_bpb"] = float(result["full_test_bpb"] - baseline_bpb)
        results.append(result)
        print(
            f"  {name}: bpb:{result['full_test_bpb']:.4f} "
            f"delta:{result['delta_bpb']:+.4f} "
            f"cos:{result['mask_cosine_similarity'] if result['mask_cosine_similarity'] is not None else 'nan'}"
        )

    summary = {
        "title": "conker-6 mask geometry",
        "config": asdict(runtime),
        "model": {
            "preset": "conker6_mask_geometry",
            "seed": args.seed,
            "params": count_trainable_params(model),
            "train_time_sec": float(metrics.train_time_sec),
            "blend_mode": "cache_only",
            "freeze_base": True,
            "learnable_vocab_axis": False,
            "learnable_causal_mask": True,
        },
        "matrix_shape": list(causal_mask_np.shape),
        "matrix_files": {
            "mask_npy": str(out_prefix.with_suffix(".mask.npy")),
            "mask_csv": str(out_prefix.with_suffix(".mask.csv")),
            "lag_profile_csv": str(out_prefix.with_suffix(".lag_profile.csv")),
            "singular_values_csv": str(out_prefix.with_suffix(".singular_values.csv")),
            "mask_png": str(out_prefix.with_suffix(".mask.png")),
            "lag_mean_mask_png": str(out_prefix.with_suffix(".lag_mean_mask.png")),
            "lag_mean_diff_png": str(out_prefix.with_suffix(".lag_mean_diff.png")),
        },
        "lag_profile_summary": {
            "lag_mean_min": float(min(row["mean"] for row in lag_rows)),
            "lag_mean_max": float(max(row["mean"] for row in lag_rows)),
            "lag_std_mean": float(np.mean([row["std"] for row in lag_rows])),
            "lag_std_max": float(max(row["std"] for row in lag_rows)),
        },
        "singular_value_summary": {
            "top_8_energy_fraction": float(np.sum(singular_values[:8]) / np.sum(singular_values)),
            "top_16_energy_fraction": float(np.sum(singular_values[:16]) / np.sum(singular_values)),
            "top_32_energy_fraction": float(np.sum(singular_values[:32]) / np.sum(singular_values)),
        },
        "variants": results,
    }

    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote geometry summary to {json_path}")


if __name__ == "__main__":
    main()
