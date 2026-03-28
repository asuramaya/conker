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

import mlx.nn as nn

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, estimate_trainable_payload_bytes, quantize_trainable_params


def config_for_variant(args: argparse.Namespace, seq_len: int) -> ConkerThreeConfig:
    config = ConkerThreeConfig(max_seq_len=seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
    if args.variant == "base":
        variant_cfg = config
    elif args.variant == "linear_only":
        variant_cfg = replace(config, enable_local=False)
    elif args.variant == "local_only":
        variant_cfg = replace(config, enable_linear=False)
    elif args.variant == "gated":
        variant_cfg = replace(config, mix_mode="gated")
    elif args.variant == "window4":
        variant_cfg = replace(config, local_window=4)
    elif args.variant == "window16":
        variant_cfg = replace(config, local_window=16)
    elif args.variant == "shared_embedding":
        variant_cfg = replace(config, share_embedding=True)
    else:
        raise ValueError(f"Unknown Conker-3 variant: {args.variant}")
    if args.decay_bank == "narrow":
        variant_cfg = replace(variant_cfg, linear_half_life_max=32.0)
    if args.linear_half_life_max is not None:
        variant_cfg = replace(variant_cfg, linear_half_life_max=args.linear_half_life_max)
    if args.oscillatory_frac is not None:
        variant_cfg = replace(
            variant_cfg,
            oscillatory_frac=args.oscillatory_frac,
            oscillatory_period_min=args.oscillatory_period_min,
            oscillatory_period_max=args.oscillatory_period_max,
        )
    if args.static_bank_gate:
        variant_cfg = replace(variant_cfg, static_bank_gate=True, bank_gate_span=args.bank_gate_span)
    variant_cfg = scale_config(variant_cfg, args.scale)
    if args.local_hidden_mult is not None:
        variant_cfg = replace(
            variant_cfg,
            local_hidden=tuple(max(int(round(width * args.local_hidden_mult)), 1) for width in variant_cfg.local_hidden),
        )
    if args.local_scale_override is not None:
        variant_cfg = replace(variant_cfg, local_scale=args.local_scale_override)
    return variant_cfg


def build_runtime(args: argparse.Namespace):
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
    parser = argparse.ArgumentParser(description="Conker-3 bridge on official parameter-golf token shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument("--decay-bank", choices=["logspace", "narrow", "custom"], default="logspace")
    parser.add_argument("--decays-json", default=None)
    parser.add_argument("--linear-half-life-max", type=float, default=None)
    parser.add_argument("--oscillatory-frac", type=float, default=None)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--static-bank-gate", action="store_true")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--local-hidden-mult", type=float, default=None)
    parser.add_argument("--local-scale-override", type=float, default=None)
    parser.add_argument(
        "--variant",
        choices=["base", "linear_only", "local_only", "gated", "window4", "window16", "shared_embedding"],
        default="base",
    )
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    config = config_for_variant(args, runtime.train.seq_len)
    model = ConkerThreeModel(vocab_size=dataset.vocab_size, config=config)
    if args.decay_bank == "custom":
        if not args.decays_json:
            raise ValueError("--decay-bank custom requires --decays-json")
        decay_payload = json.loads(Path(args.decays_json).read_text(encoding="utf-8"))
        decays = decay_payload.get("decays")
        if decays is None:
            raise ValueError(f"Decay bank JSON is missing 'decays': {args.decays_json}")
        model.set_linear_decays(decays)

    print("\n  conker-3 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} "
        f"lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  train_shards={len(dataset.train_files)} val_shards={len(dataset.test_files)} "
        f"train_tokens={dataset.train_token_count:,} val_tokens={dataset.test_token_count:,}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} linear_modes={config.linear_modes} "
        f"local_window={config.local_window} decay_bank={args.decay_bank} "
        f"half_life_max={config.linear_half_life_max:.1f} "
        f"osc_frac={config.oscillatory_frac:.2f} "
        f"static_bank_gate={config.static_bank_gate} "
        f"local_hidden={list(config.local_hidden)} local_scale={config.local_scale:.3f} "
        f"params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker3_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    result = {
        "title": "conker-3 golf bridge cell",
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
            "preset": "conker3",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "linear_modes": config.linear_modes,
            "local_window": config.local_window,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "decay_bank": args.decay_bank,
            "decays_json": args.decays_json,
            "linear_half_life_min": config.linear_half_life_min,
            "linear_half_life_max": config.linear_half_life_max,
            "oscillatory_frac": config.oscillatory_frac,
            "oscillatory_period_min": config.oscillatory_period_min,
            "oscillatory_period_max": config.oscillatory_period_max,
            "static_bank_gate": config.static_bank_gate,
            "bank_gate_span": config.bank_gate_span,
            "local_hidden_mult": args.local_hidden_mult,
            "local_scale_override": args.local_scale_override,
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
            "embedding_dim": config.embedding_dim,
            "linear_hidden": list(config.linear_hidden),
            "local_hidden": list(config.local_hidden),
            "local_scale": config.local_scale,
            "mix_mode": config.mix_mode,
        },
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
