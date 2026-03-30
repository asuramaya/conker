#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker6 import ConkerSixConfig, ConkerSixModel, scale_config
from conker.src.golf_data import build_parameter_golf_dataset


def base_config_for_variant(args: argparse.Namespace) -> ConkerThreeConfig:
    config = ConkerThreeConfig(max_seq_len=args.seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
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
        raise ValueError(f"Unknown variant: {args.variant}")
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


def build_model(args: argparse.Namespace, vocab_size: int) -> ConkerSixModel:
    base_cfg = base_config_for_variant(args)
    config = scale_config(
        ConkerSixConfig(
            base_config=base_cfg,
            freeze_base=args.freeze_base,
            enable_exact3=not args.disable_exact3,
            exact_context_span=args.exact_context_span,
            causal_projection=args.causal_projection,
            blend_mode=args.blend_mode,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_temperature=args.gate_temperature,
            fixed_base_weight=args.fixed_base_weight,
            fixed_exact1_weight=args.fixed_exact1_weight,
            fixed_exact2_weight=args.fixed_exact2_weight,
            fixed_exact3_weight=args.fixed_exact3_weight,
        ),
        args.scale,
    )
    return ConkerSixModel(vocab_size=vocab_size, config=config)


def future_invariance_error(model: ConkerSixModel, x: mx.array, prefix_len: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base_logits = model(x)
    x_np = np.array(x, copy=True)
    if prefix_len < x_np.shape[1]:
        x_np[:, prefix_len:] = rng.integers(0, model.vocab_size, size=x_np[:, prefix_len:].shape, dtype=np.int32)
    x_alt = mx.array(x_np, dtype=mx.int32)
    alt_logits = model(x_alt)
    diff = np.abs(np.array(base_logits[:, :prefix_len, :], copy=False) - np.array(alt_logits[:, :prefix_len, :], copy=False))
    return {
        "prefix_len": int(prefix_len),
        "max_abs_logit_diff": float(diff.max(initial=0.0)),
        "mean_abs_logit_diff": float(diff.mean() if diff.size else 0.0),
    }


def flat_future_invariance_error(model: ConkerSixModel, x: mx.array, prefix_len: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x_np = np.array(x, copy=True)
    batch, seq_len = x_np.shape
    flat_len = batch * seq_len
    prefix_len = min(prefix_len, flat_len)
    base_logits = np.array(model(x), copy=False).reshape(flat_len, model.vocab_size)
    flat_tokens = x_np.reshape(flat_len)
    if prefix_len < flat_len:
        flat_tokens[prefix_len:] = rng.integers(0, model.vocab_size, size=(flat_len - prefix_len,), dtype=np.int32)
    alt_logits = np.array(model(mx.array(flat_tokens.reshape(batch, seq_len), dtype=mx.int32)), copy=False).reshape(flat_len, model.vocab_size)
    diff = np.abs(base_logits[:prefix_len] - alt_logits[:prefix_len])
    return {
        "prefix_len": int(prefix_len),
        "max_abs_logit_diff": float(diff.max(initial=0.0)),
        "mean_abs_logit_diff": float(diff.mean() if diff.size else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attack Conker-6 normalization and legality.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--variant", choices=["base", "window4", "window16", "gated", "linear_only"], default="window4")
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--input-proj-scheme", choices=["random", "orthogonal_rows", "kernel_energy", "split_banks"], default="random")
    parser.set_defaults(static_bank_gate=True)
    parser.add_argument("--static-bank-gate", action="store_true", dest="static_bank_gate")
    parser.add_argument("--no-static-bank-gate", action="store_false", dest="static_bank_gate")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--blend-mode", choices=["cache_only", "fixed_blend", "learned_gate"], default="cache_only")
    parser.set_defaults(freeze_base=True)
    parser.add_argument("--freeze-base", action="store_true", dest="freeze_base")
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")
    parser.add_argument("--disable-exact3", action="store_true")
    parser.add_argument("--exact-context-span", type=int, default=0)
    parser.add_argument("--causal-projection", choices=["none", "strict_lower", "strict_lower_nonnegative"], default="none")
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--fixed-base-weight", type=float, default=0.15)
    parser.add_argument("--fixed-exact1-weight", type=float, default=0.10)
    parser.add_argument("--fixed-exact2-weight", type=float, default=0.25)
    parser.add_argument("--fixed-exact3-weight", type=float, default=0.50)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    model = build_model(args, dataset.vocab_size)

    x, _ = dataset.batch(args.split, args.batch_size, args.seq_len)
    logits = model(x)
    probs = np.exp(np.array(logits, copy=False))
    prob_sums = probs.sum(axis=-1)

    normalization = {
        "sum_min": float(prob_sums.min()),
        "sum_max": float(prob_sums.max()),
        "sum_mean": float(prob_sums.mean()),
        "max_abs_sum_error": float(np.max(np.abs(prob_sums - 1.0))),
        "mean_abs_sum_error": float(np.mean(np.abs(prob_sums - 1.0))),
    }

    prefix_len = min(args.prefix_len, args.seq_len)
    invariance = future_invariance_error(model, x, prefix_len=prefix_len, seed=args.seed + 1)

    row_boundary = max(1, args.seq_len // 2)
    row_boundary_audit = future_invariance_error(model, x, prefix_len=row_boundary, seed=args.seed + 2)
    flat_prefix = max(1, (args.batch_size * args.seq_len) // 2)
    flat_audit = flat_future_invariance_error(model, x, prefix_len=flat_prefix, seed=args.seed + 3)

    result = {
        "config": {
            "split": args.split,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "blend_mode": args.blend_mode,
            "freeze_base": args.freeze_base,
            "exact_context_span": args.exact_context_span,
            "causal_projection": args.causal_projection,
            "enable_exact3": not args.disable_exact3,
        },
        "normalization": normalization,
        "future_invariance": invariance,
        "row_boundary_invariance": row_boundary_audit,
        "flat_stream_invariance": flat_audit,
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
