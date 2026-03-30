#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import sys

import mlx.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker5 import ConkerFiveConfig, ConkerFiveModel, scale_config
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
    elif args.variant == "linear_only":
        variant_cfg = replace(config, enable_local=False)
    elif args.variant == "base":
        variant_cfg = config
    else:
        raise ValueError(f"Unknown Conker-5 base variant: {args.variant}")
    variant_cfg = replace(
        variant_cfg,
        linear_half_life_max=args.linear_half_life_max,
        oscillatory_frac=args.oscillatory_frac,
        oscillatory_period_min=args.oscillatory_period_min,
        oscillatory_period_max=args.oscillatory_period_max,
        static_bank_gate=args.static_bank_gate,
        bank_gate_span=args.bank_gate_span,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-5 bridge on official parameter-golf token shards.")
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
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.set_defaults(static_bank_gate=True)
    parser.add_argument("--static-bank-gate", action="store_true", dest="static_bank_gate")
    parser.add_argument("--no-static-bank-gate", action="store_false", dest="static_bank_gate")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--state-proj-dim", type=int, default=32)
    parser.add_argument("--shared-hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-rank", type=int, default=8)
    parser.add_argument("--residual-cap", type=float, default=2.0)
    parser.add_argument(
        "--variant",
        choices=["base", "window4", "window16", "gated", "linear_only"],
        default="window4",
    )
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    base_cfg = base_config_for_variant(args, runtime.train.seq_len)
    config = scale_config(
        ConkerFiveConfig(
            base_config=base_cfg,
            freeze_base=True,
            state_proj_dim=args.state_proj_dim,
            shared_hidden_dim=args.shared_hidden_dim,
            num_heads=args.num_heads,
            head_rank=args.head_rank,
            residual_cap=args.residual_cap,
        ),
        args.scale,
    )
    model = ConkerFiveModel(vocab_size=dataset.vocab_size, config=config)

    print("\n  conker-5 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} "
        f"lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} linear_modes={config.base_config.linear_modes} "
        f"local_window={config.base_config.local_window} half_life_max={config.base_config.linear_half_life_max:.1f} "
        f"osc_frac={config.base_config.oscillatory_frac:.2f} static_bank_gate={config.base_config.static_bank_gate} "
        f"num_heads={config.num_heads} head_rank={config.head_rank} params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker5_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    result = {
        "title": "conker-5 golf bridge cell",
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
            "preset": "conker5",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "linear_modes": config.base_config.linear_modes,
            "local_window": config.base_config.local_window,
            "linear_half_life_max": config.base_config.linear_half_life_max,
            "oscillatory_frac": config.base_config.oscillatory_frac,
            "oscillatory_period_min": config.base_config.oscillatory_period_min,
            "oscillatory_period_max": config.base_config.oscillatory_period_max,
            "static_bank_gate": config.base_config.static_bank_gate,
            "state_proj_dim": config.state_proj_dim,
            "shared_hidden_dim": config.shared_hidden_dim,
            "num_heads": config.num_heads,
            "head_rank": config.head_rank,
            "residual_cap": config.residual_cap,
            "train_eval_loss": float(train_eval),
            "test_eval_loss": float(test_eval),
            "train_bits_per_token": float(train_bpt),
            "test_bits_per_token": float(test_bpt),
            "test_bpb": None if test_bpb is None else float(test_bpb),
            "train_time_sec": float(metrics.train_time_sec),
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": None,
            "payload_mb_est": None,
        },
        "quantization": [],
    }

    flat_full = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    result["model"]["payload_bytes_est"] = estimate_trainable_payload_bytes(flat_full, trainable_names)
    result["model"]["payload_mb_est"] = result["model"]["payload_bytes_est"] / (1024.0 * 1024.0)

    for bits in sorted(set(args.quant_bits)):
        quantized_state, quant_stats = quantize_trainable_params(flat_full, trainable_names, bits)
        model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
        quant_test = evaluate(model, dataset, runtime.train, "test")
        quant_bpt = bits_per_token_from_loss(quant_test)
        quant_bpb = quant_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
        result["quantization"].append(
            {
                "scheme": f"uniform_int{bits}",
                "bits": float(bits),
                "test_eval_loss": float(quant_test),
                "test_bits_per_token": float(quant_bpt),
                "test_bpb": None if quant_bpb is None else float(quant_bpb),
                "payload_bytes_est": float(quant_stats["payload_bytes_est"]),
                "payload_mb_est": float(quant_stats["payload_mb_est"]),
                "quantized_params": int(quant_stats["quantized_params"]),
                "kept_params": int(quant_stats["kept_params"]),
            }
        )
    if result["quantization"]:
        model.update(nn.utils.tree_unflatten(list(flat_full.items())))

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
