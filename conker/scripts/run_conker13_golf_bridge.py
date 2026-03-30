#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.nn as nn

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.conker13 import ConkerThirteenConfig, ConkerThirteenModel, scale_config
from conker.src.conker3 import ConkerThreeConfig
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, estimate_trainable_payload_bytes, quantize_trainable_params


def base_config_for_variant(args: argparse.Namespace, seq_len: int) -> ConkerThreeConfig:
    config = ConkerThreeConfig(max_seq_len=seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
    if args.variant == "window4":
        variant_cfg = replace(config, local_window=4)
    elif args.variant == "window16":
        variant_cfg = replace(config, local_window=16)
    elif args.variant == "gated":
        variant_cfg = replace(config, mix_mode="gated")
    elif args.variant == "base":
        variant_cfg = config
    else:
        raise ValueError(f"Unknown Conker-13 base variant: {args.variant}")
    if args.linear_half_life_max is not None:
        variant_cfg = replace(variant_cfg, linear_half_life_max=args.linear_half_life_max)
    if args.oscillatory_frac is not None:
        variant_cfg = replace(
            variant_cfg,
            oscillatory_frac=args.oscillatory_frac,
            oscillatory_period_min=args.oscillatory_period_min,
            oscillatory_period_max=args.oscillatory_period_max,
            input_proj_scheme=args.input_proj_scheme,
        )
    else:
        variant_cfg = replace(variant_cfg, input_proj_scheme=args.input_proj_scheme)
    if args.static_bank_gate:
        variant_cfg = replace(variant_cfg, static_bank_gate=True, bank_gate_span=args.bank_gate_span)
    if not variant_cfg.enable_linear or not variant_cfg.enable_local:
        raise ValueError("Conker-13 requires both the linear and local Conker-3 paths.")
    return variant_cfg


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


def parse_lag_lookbacks(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Conker-13 requires at least one lag lookback.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conker-13 bridge: five-axis controller directly over the Conker-3 multiscale substrate and local residual coder."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--input-proj-scheme", choices=["random", "orthogonal_rows", "kernel_energy", "split_banks"], default="random")
    parser.set_defaults(static_bank_gate=True)
    parser.add_argument("--static-bank-gate", action="store_true", dest="static_bank_gate")
    parser.add_argument("--no-static-bank-gate", action="store_false", dest="static_bank_gate")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--variant", choices=["base", "window4", "window16", "gated"], default="window4")
    parser.add_argument("--lag-lookbacks", default="2,4,8,16,32,64,128,0")
    parser.add_argument("--lag-temperature", type=float, default=1.0)
    parser.add_argument("--mode-groups", type=int, default=8)
    parser.add_argument("--program-slots", type=int, default=4)
    parser.add_argument("--program-temperature", type=float, default=1.0)
    parser.add_argument("--linear-gate-span", type=float, default=1.0)
    parser.add_argument("--local-gate-span", type=float, default=1.0)
    parser.add_argument("--local-scale-span", type=float, default=0.5)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    base_cfg = base_config_for_variant(args, runtime.train.seq_len)
    config = scale_config(
        ConkerThirteenConfig(
            base_config=base_cfg,
            lag_lookbacks=parse_lag_lookbacks(args.lag_lookbacks),
            lag_temperature=args.lag_temperature,
            mode_groups=args.mode_groups,
            program_slots=args.program_slots,
            program_temperature=args.program_temperature,
            linear_gate_span=args.linear_gate_span,
            local_gate_span=args.local_gate_span,
            local_scale_span=args.local_scale_span,
        ),
        args.scale,
    )
    model = ConkerThirteenModel(vocab_size=dataset.vocab_size, config=config)

    print("\n  conker-13 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} lag_lookbacks={config.lag_lookbacks} "
        f"mode_groups={config.mode_groups} program_slots={config.program_slots} "
        f"lag_temp={config.lag_temperature:.2f} program_temp={config.program_temperature:.2f} "
        f"linear_gate_span={config.linear_gate_span:.2f} local_gate_span={config.local_gate_span:.2f} "
        f"local_scale_span={config.local_scale_span:.2f} params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker13_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    probe_tokens, _ = dataset.batch("test", runtime.train.batch_size, runtime.train.seq_len)
    controller_summary = model.controller_snapshot(probe_tokens)

    result = {
        "title": "conker-13 golf bridge cell",
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
            "preset": "conker13",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "lag_lookbacks": list(config.lag_lookbacks),
            "lag_temperature": config.lag_temperature,
            "mode_groups": config.mode_groups,
            "program_slots": config.program_slots,
            "program_temperature": config.program_temperature,
            "linear_gate_span": config.linear_gate_span,
            "local_gate_span": config.local_gate_span,
            "local_scale_span": config.local_scale_span,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "train_bpb": None,
            "test_bpb": test_bpb,
            "overfit_pct": metrics.overfit_pct,
            "train_time_sec": metrics.train_time_sec,
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": None,
            "payload_mb_est": None,
            "embedding_dim": config.base_config.embedding_dim,
            "linear_modes": config.base_config.linear_modes,
            "linear_hidden": list(config.base_config.linear_hidden),
            "local_window": config.base_config.local_window,
            "local_hidden": list(config.base_config.local_hidden),
            "local_scale": config.base_config.local_scale,
            "mix_mode": config.base_config.mix_mode,
        },
        "controller_summary": controller_summary,
    }

    flat_full = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    result["model"]["payload_bytes_est"] = estimate_trainable_payload_bytes(flat_full, trainable_names)
    result["model"]["payload_mb_est"] = result["model"]["payload_bytes_est"] / (1024.0 * 1024.0)

    quant_rows = []
    for bits in sorted(set(args.quant_bits)):
        quantized_state, stats = quantize_trainable_params(flat_full, trainable_names, bits)
        model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
        q_test_eval = evaluate(model, dataset, runtime.train, "test")
        q_test_bpt = bits_per_token_from_loss(q_test_eval)
        quant_rows.append(
            {
                "scheme": f"uniform_int{bits}",
                "bits": float(bits),
                "test_eval_loss": q_test_eval,
                "test_bits_per_token": q_test_bpt,
                "test_bpb": q_test_bpt * dataset.test_tokens_per_byte,
                **stats,
            }
        )
    if quant_rows:
        result["quantization"] = quant_rows
        model.update(nn.utils.tree_unflatten(list(flat_full.items())))

    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{result['model']['test_bits_per_token']:.4f} "
        f"bpb:{result['model']['test_bpb']:.4f} "
        f"Of:{metrics.overfit_pct:+.1f}% "
        f"T:{metrics.train_time_sec:.0f}s"
    )
    for row in quant_rows:
        print(
            f"  {row['scheme']}: "
            f"bpb:{row['test_bpb']:.4f} "
            f"bpt:{row['test_bits_per_token']:.4f} "
            f"payload_mb_est:{row['payload_mb_est']:.3f}"
        )

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
