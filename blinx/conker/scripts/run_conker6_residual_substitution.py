#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
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
from carving_machine.training import loss_fn
from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker6 import ConkerSixConfig, ConkerSixModel, scale_config
from conker.src.golf_data import _load_golf_shard, build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss


def build_runtime(profile: str, seq_len: int, batch_size: int) -> RuntimeConfig:
    runtime = RuntimeConfig(profile=profile)
    base_train = train_config_for_profile(profile)
    return replace(base_train, seq_len=seq_len, batch_size=batch_size)


def evaluate_full_split(model, dataset, train_cfg, split: str) -> tuple[float, int]:
    files = dataset.train_files if split == "train" else dataset.test_files
    tokens = np.ascontiguousarray(np.concatenate([_load_golf_shard(path) for path in files], axis=0))
    batch_size = train_cfg.batch_size
    seq_len = train_cfg.seq_len
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


def toeplitz_mean(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        out += np.diag(np.full(size - lag, float(np.mean(vals)), dtype=np.float32), k=-lag)
    return out * support


def uniform_noise_like(residual: np.ndarray, support: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active = residual[support > 0]
    std = float(np.std(active))
    radius = np.sqrt(3.0) * std
    out = np.zeros_like(residual, dtype=np.float32)
    out[support > 0] = rng.uniform(-radius, radius, size=active.size).astype(np.float32)
    return out


def lag_matched_uniform_noise(residual: np.ndarray, support: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(residual, dtype=np.float32)
    size = residual.shape[0]
    for lag in range(1, size):
        vals = np.diag(residual, k=-lag)
        std = float(np.std(vals))
        radius = np.sqrt(3.0) * std
        noise = rng.uniform(-radius, radius, size=vals.size).astype(np.float32)
        out += np.diag(noise, k=-lag)
    return out * support


def shuffled_residual(residual: np.ndarray, support: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active = residual[support > 0].copy()
    rng.shuffle(active)
    out = np.zeros_like(residual, dtype=np.float32)
    out[support > 0] = active.astype(np.float32)
    return out


def sign_randomized_residual(residual: np.ndarray, support: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active = np.abs(residual[support > 0])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=active.size)
    out = np.zeros_like(residual, dtype=np.float32)
    out[support > 0] = (active * signs).astype(np.float32)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return None
    return float(np.dot(a, b) / denom)


def evaluate_variant(model, dataset, train_cfg, base_state, mask: np.ndarray, baseline_mask: np.ndarray, support: np.ndarray, name: str) -> dict:
    patched = dict(base_state)
    patched["causal_mask"] = mx.array(mask.astype(np.float32, copy=False), dtype=mx.float32)
    model.update(nn.utils.tree_unflatten(list(patched.items())))
    loss, tokens = evaluate_full_split(model, dataset, train_cfg, "test")
    bpt = bits_per_token_from_loss(loss)
    bpb = float(bpt * dataset.test_tokens_per_byte)
    active = mask[support > 0]
    base_active = baseline_mask[support > 0]
    diff = active - base_active
    return {
        "variant": name,
        "full_test_eval_loss": float(loss),
        "full_test_bits_per_token": float(bpt),
        "full_test_bpb": float(bpb),
        "full_test_tokens": int(tokens),
        "active_mean": float(np.mean(active)),
        "active_std": float(np.std(active)),
        "mask_cosine_to_baseline": cosine(active, base_active),
        "mask_l2_deviation": float(np.sqrt(np.mean(diff * diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-6 residual substitution attack.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed-dump-prefix", required=True, help="Prefix like conker/out/conker6_seed_residual_compare_2026-03-28_seed")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--swap-seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--scale", type=float, default=10.0)
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

    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    train_cfg = build_runtime(args.profile, args.seq_len, args.batch_size)
    base_cfg = ConkerThreeConfig(max_seq_len=args.seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
    if args.variant == "window4":
        base_cfg = replace(base_cfg, local_window=4)
    elif args.variant == "window16":
        base_cfg = replace(base_cfg, local_window=16)
    elif args.variant == "gated":
        base_cfg = replace(base_cfg, mix_mode="gated")
    elif args.variant == "linear_only":
        base_cfg = replace(base_cfg, enable_local=False)
    base_cfg = replace(
        base_cfg,
        linear_half_life_max=args.linear_half_life_max,
        oscillatory_frac=args.oscillatory_frac,
        oscillatory_period_min=args.oscillatory_period_min,
        oscillatory_period_max=args.oscillatory_period_max,
        static_bank_gate=args.static_bank_gate,
        bank_gate_span=args.bank_gate_span,
        input_proj_scheme=args.input_proj_scheme,
    )
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
    base_state = dict(nn.utils.tree_flatten(model.parameters()))

    prefix = Path(args.seed_dump_prefix)
    base_dir = Path(f"{prefix}{args.base_seed}")
    base_mask = np.load(base_dir / "mask.npy")
    base_toe = np.load(base_dir / "toeplitz_mean.npy")
    base_res = np.load(base_dir / "residual.npy")
    support = strict_lower_mask(base_mask.shape[0])
    upperdiag = base_mask * (1.0 - support)

    variants: list[tuple[str, np.ndarray]] = [
        ("baseline_seed42_full", base_mask),
        ("baseline_seed42_strictlower", base_mask * support),
        ("toeplitz_lower_plus_upperdiag_seed42", upperdiag + base_toe),
        ("toeplitz_plus_uniform_noise_0", upperdiag + base_toe + uniform_noise_like(base_res, support, 0)),
        ("toeplitz_plus_uniform_noise_1", upperdiag + base_toe + uniform_noise_like(base_res, support, 1)),
        ("toeplitz_plus_lagmatched_noise_0", upperdiag + base_toe + lag_matched_uniform_noise(base_res, support, 0)),
        ("toeplitz_plus_shuffled_residual_0", upperdiag + base_toe + shuffled_residual(base_res, support, 0)),
        ("toeplitz_plus_signrand_residual_0", upperdiag + base_toe + sign_randomized_residual(base_res, support, 0)),
    ]
    for swap_seed in args.swap_seeds:
        swap_dir = Path(f"{prefix}{swap_seed}")
        swap_res = np.load(swap_dir / "residual.npy")
        variants.append((f"toeplitz42_plus_residual{swap_seed}", upperdiag + base_toe + swap_res))

    print("\n  conker-6 residual substitution\n")
    print(f"  base_seed={args.base_seed} seq_len={args.seq_len} batch_size={args.batch_size} variants={len(variants)}")

    results = []
    baseline_bpb = None
    for name, mask in variants:
        mask = mask.astype(np.float32, copy=False)
        row = evaluate_variant(model, dataset, train_cfg, base_state, mask, base_mask, support, name)
        if baseline_bpb is None:
            baseline_bpb = row["full_test_bpb"]
            row["delta_bpb"] = 0.0
        else:
            row["delta_bpb"] = float(row["full_test_bpb"] - baseline_bpb)
        results.append(row)
        print(
            f"  {name}: bpb:{row['full_test_bpb']:.4f} "
            f"delta:{row['delta_bpb']:+.4f} "
            f"cos:{row['mask_cosine_to_baseline']}"
        )

    summary = {
        "title": "conker-6 residual substitution",
        "config": {
            "profile": args.profile,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "base_seed": args.base_seed,
            "swap_seeds": args.swap_seeds,
        },
        "baseline_paths": {
            "mask": str(base_dir / "mask.npy"),
            "toeplitz": str(base_dir / "toeplitz_mean.npy"),
            "residual": str(base_dir / "residual.npy"),
        },
        "variants": results,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote residual substitution summary to {out_path}")


if __name__ == "__main__":
    main()
