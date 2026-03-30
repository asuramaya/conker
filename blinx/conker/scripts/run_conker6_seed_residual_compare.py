#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
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


def build_runtime(args: argparse.Namespace, seed: int) -> RuntimeConfig:
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
            seeds=(seed,),
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


def toeplitz_mean(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    size = mask.shape[0]
    for lag in range(1, size):
        vals = np.diag(mask, k=-lag)
        out += np.diag(np.full(size - lag, float(np.mean(vals)), dtype=np.float32), k=-lag)
    return out * support


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return None
    return float(np.dot(a, b) / denom)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size == 0 or b.size == 0:
        return None
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return None
    return float(np.dot(a, b) / denom)


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sign(a.reshape(-1)) == np.sign(b.reshape(-1))))


def _to_grayscale_image(matrix: np.ndarray, path: Path, scale: int = 4) -> None:
    clipped = matrix.astype(np.float32, copy=False)
    clipped = clipped - np.min(clipped)
    denom = float(np.max(clipped))
    norm = clipped / denom if denom > 0 else clipped
    pixels = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-6 cross-seed residual-to-Toeplitz comparison.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
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

    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    base_cfg = base_config_for_variant(args, args.seq_len)
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

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    support = strict_lower_mask(args.seq_len)

    for seed in args.seeds:
        runtime = build_runtime(args, seed)
        model = ConkerSixModel(vocab_size=dataset.vocab_size, config=config)
        print(
            f"\n  conker-6 seed residual compare: seed={seed} "
            f"steps={runtime.train.steps} seq_len={runtime.train.seq_len} batch={runtime.train.batch_size}"
        )
        metrics = train_model(model, dataset, runtime.train, seed, "conker6_seed_residual_compare")
        flat_state = dict(nn.utils.tree_flatten(model.parameters()))
        mask = np.array(flat_state["causal_mask"], dtype=np.float32, copy=True)
        lag = toeplitz_mean(mask, support)
        residual = (mask - lag).astype(np.float32, copy=False) * support
        full_loss, full_tokens = evaluate_full_split(model, dataset, runtime, "test")
        bpt = bits_per_token_from_loss(full_loss)
        bpb = float(bpt * dataset.test_tokens_per_byte)

        seed_dir = out_prefix.parent / f"{out_prefix.name}_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "mask.npy", mask)
        np.save(seed_dir / "toeplitz_mean.npy", lag)
        np.save(seed_dir / "residual.npy", residual)
        _to_grayscale_image(mask, seed_dir / "mask.png")
        _to_grayscale_image(lag, seed_dir / "toeplitz_mean.png")
        _to_grayscale_image(np.abs(residual), seed_dir / "residual_abs.png")

        active = mask[support > 0]
        res_active = residual[support > 0]
        seed_rows.append(
            {
                "seed": seed,
                "full_test_eval_loss": float(full_loss),
                "full_test_bits_per_token": float(bpt),
                "full_test_bpb": float(bpb),
                "full_test_tokens": int(full_tokens),
                "params": count_trainable_params(model),
                "train_time_sec": float(metrics.train_time_sec),
                "mask_path": str(seed_dir / "mask.npy"),
                "toeplitz_path": str(seed_dir / "toeplitz_mean.npy"),
                "residual_path": str(seed_dir / "residual.npy"),
                "active_mean": float(np.mean(active)),
                "active_std": float(np.std(active)),
                "residual_mean": float(np.mean(res_active)),
                "residual_std": float(np.std(res_active)),
                "residual_norm_fraction": float(np.linalg.norm(res_active) / np.linalg.norm(active)),
            }
        )

    pairwise = []
    by_seed = {row["seed"]: row for row in seed_rows}
    for seed_a, seed_b in itertools.combinations(args.seeds, 2):
        dir_a = out_prefix.parent / f"{out_prefix.name}_seed{seed_a}"
        dir_b = out_prefix.parent / f"{out_prefix.name}_seed{seed_b}"
        mask_a = np.load(dir_a / "mask.npy")
        mask_b = np.load(dir_b / "mask.npy")
        toe_a = np.load(dir_a / "toeplitz_mean.npy")
        toe_b = np.load(dir_b / "toeplitz_mean.npy")
        res_a = np.load(dir_a / "residual.npy")
        res_b = np.load(dir_b / "residual.npy")
        active_mask_a = mask_a[support > 0]
        active_mask_b = mask_b[support > 0]
        active_toe_a = toe_a[support > 0]
        active_toe_b = toe_b[support > 0]
        active_res_a = res_a[support > 0]
        active_res_b = res_b[support > 0]
        pairwise.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "raw_cosine": cosine(active_mask_a, active_mask_b),
                "raw_pearson": pearson(active_mask_a, active_mask_b),
                "toeplitz_cosine": cosine(active_toe_a, active_toe_b),
                "toeplitz_pearson": pearson(active_toe_a, active_toe_b),
                "residual_cosine": cosine(active_res_a, active_res_b),
                "residual_pearson": pearson(active_res_a, active_res_b),
                "residual_sign_agreement": sign_agreement(active_res_a, active_res_b),
                "residual_l2_diff": float(np.sqrt(np.mean((active_res_a - active_res_b) ** 2))),
                "bpb_gap": float(abs(by_seed[seed_a]["full_test_bpb"] - by_seed[seed_b]["full_test_bpb"])),
            }
        )

    summary = {
        "title": "conker-6 cross-seed residual compare",
        "config": {
            "profile": args.profile,
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "scale": args.scale,
        },
        "seed_rows": seed_rows,
        "pairwise": pairwise,
    }
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote seed comparison summary to {json_path}")


if __name__ == "__main__":
    main()
