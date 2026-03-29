#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import RunMetrics, count_trainable_params, evaluate, train_model
from conker.src.conker10 import ConkerTenConfig, ConkerTenModel, build_packed_tables, scale_config
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
    elif args.variant == "linear_only":
        variant_cfg = replace(config, enable_local=False)
    elif args.variant == "base":
        variant_cfg = config
    else:
        raise ValueError(f"Unknown Conker-10 base variant: {args.variant}")
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


def save_state_npz(path: Path, state: dict[str, mx.array]) -> None:
    arrays = {name: np.array(value) for name, value in state.items()}
    np.savez(path, **arrays)


def _fmt_float(value: float | None, precision: int = 4) -> str:
    return f"{value:.{precision}f}" if value is not None else "n/a"


def parse_fixed_weights(raw: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("Conker-10 fixed weights must have exactly 4 comma-separated values.")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-10 bridge: packed training memory plus a causal controller.")
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
    parser.add_argument("--variant", choices=["base", "window4", "window16", "gated", "linear_only"], default="window4")
    parser.set_defaults(freeze_base=False)
    parser.add_argument("--freeze-base", action="store_true", dest="freeze_base")
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")
    parser.add_argument("--packed-tokens", type=int, default=4_000_000)
    parser.add_argument("--trigram-buckets", type=int, default=16_384)
    parser.add_argument("--blend-mode", choices=["learned_mixer", "fixed_interp", "memory_only"], default="learned_mixer")
    parser.add_argument("--structure-proxy", action="store_true", help="Add causal memory-confidence proxy features to the controller.")
    parser.add_argument("--structure-proxy-entropy", action="store_true")
    parser.add_argument("--structure-proxy-peak", action="store_true")
    parser.add_argument("--structure-proxy-candidate4", action="store_true")
    parser.add_argument("--structure-proxy-agreement", action="store_true")
    parser.add_argument("--fixed-weights", default="0.25,0.10,0.25,0.40")
    parser.add_argument("--alpha-bigram", type=float, default=4.0)
    parser.add_argument("--alpha-trigram", type=float, default=2.0)
    parser.add_argument("--controller-hidden", type=int, default=16)
    parser.add_argument("--controller-temperature", type=float, default=1.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    tables = build_packed_tables(dataset, token_budget=args.packed_tokens, trigram_buckets=args.trigram_buckets)
    base_cfg = base_config_for_variant(args, runtime.train.seq_len)
    if args.structure_proxy:
        args.structure_proxy_entropy = True
        args.structure_proxy_peak = True
        args.structure_proxy_candidate4 = True
        args.structure_proxy_agreement = True
    config = scale_config(
        ConkerTenConfig(
            base_config=base_cfg,
            freeze_base=args.freeze_base,
            blend_mode=args.blend_mode,
            structure_proxy_entropy=args.structure_proxy_entropy,
            structure_proxy_peak=args.structure_proxy_peak,
            structure_proxy_candidate4=args.structure_proxy_candidate4,
            structure_proxy_agreement=args.structure_proxy_agreement,
            fixed_component_weights=parse_fixed_weights(args.fixed_weights),
            alpha_bigram=args.alpha_bigram,
            alpha_trigram=args.alpha_trigram,
            controller_hidden=args.controller_hidden,
            controller_temperature=args.controller_temperature,
        ),
        args.scale,
    )
    model = ConkerTenModel(vocab_size=dataset.vocab_size, tables=tables, config=config)

    print("\n  conker-10 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} freeze_base={config.freeze_base} "
        f"blend_mode={config.blend_mode} structure_proxy="
        f"{int(config.structure_proxy_entropy) + int(config.structure_proxy_peak) + int(config.structure_proxy_candidate4) + int(config.structure_proxy_agreement)} "
        f"packed_tokens={tables.token_budget:,} trigram_buckets={tables.trigram_buckets:,} "
        f"alpha_bigram={config.alpha_bigram:.2f} alpha_trigram={config.alpha_trigram:.2f} "
        f"params={count_trainable_params(model):,} packed_bytes={tables.bytes_total:,}"
    )

    if args.skip_train:
        start = time.time()
        train_eval = evaluate(model, dataset, runtime.train, "train")
        test_eval = evaluate(model, dataset, runtime.train, "test")
        metrics = RunMetrics(
            seed=args.seed,
            params=count_trainable_params(model),
            train_loss=float(train_eval),
            test_loss=float(test_eval),
            overfit_pct=(float(test_eval) / float(train_eval) - 1.0) * 100.0,
            train_time_sec=time.time() - start,
        )
    else:
        metrics = train_model(model, dataset, runtime.train, args.seed, "conker10_golf_bridge")
        train_eval = evaluate(model, dataset, runtime.train, "train")
        test_eval = evaluate(model, dataset, runtime.train, "test")
    if args.skip_train:
        train_eval = metrics.train_loss
        test_eval = metrics.test_loss
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    train_bpb = train_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    state = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    payload_bytes_est = estimate_trainable_payload_bytes(state, trainable_names)

    result = {
        "title": "conker-10 golf bridge cell",
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
            "preset": "conker10",
            "variant": args.variant,
            "scale": args.scale,
            "freeze_base": config.freeze_base,
            "params": metrics.params,
            "seed": args.seed,
            "packed_tokens": tables.token_budget,
            "trigram_buckets": tables.trigram_buckets,
            "packed_bytes": tables.bytes_total,
            "blend_mode": config.blend_mode,
            "include_structure_proxy": bool(
                config.structure_proxy_entropy
                or config.structure_proxy_peak
                or config.structure_proxy_candidate4
                or config.structure_proxy_agreement
            ),
            "structure_proxy_entropy": config.structure_proxy_entropy,
            "structure_proxy_peak": config.structure_proxy_peak,
            "structure_proxy_candidate4": config.structure_proxy_candidate4,
            "structure_proxy_agreement": config.structure_proxy_agreement,
            "fixed_component_weights": list(config.fixed_component_weights),
            "alpha_bigram": config.alpha_bigram,
            "alpha_trigram": config.alpha_trigram,
            "controller_hidden": config.controller_hidden,
            "controller_temperature": config.controller_temperature,
        },
        "train": {
            "steps": runtime.train.steps,
            "train_time_sec": float(metrics.train_time_sec),
            "train_eval_loss": float(train_eval),
            "test_eval_loss": float(test_eval),
            "overfit_pct": float(metrics.overfit_pct),
            "nats_per_token": float(train_eval),
            "bits_per_token": float(train_bpt),
            "bits_per_byte": None if train_bpb is None else float(train_bpb),
        },
        "test": {
            "nats_per_token": float(test_eval),
            "bits_per_token": float(test_bpt),
            "bits_per_byte": None if test_bpb is None else float(test_bpb),
        },
        "artifact": {
            "trainable_payload_bytes_est": float(payload_bytes_est),
            "trainable_payload_mb_est": float(payload_bytes_est) / (1024.0 * 1024.0),
            "packed_memory_bytes": int(tables.bytes_total),
            "total_bytes_est": float(payload_bytes_est) + float(tables.bytes_total),
            "total_mb_est": (float(payload_bytes_est) + float(tables.bytes_total)) / (1024.0 * 1024.0),
        },
        "quantized": [],
    }

    for bits in sorted(set(args.quant_bits)):
        quantized_state, stats = quantize_trainable_params(state, trainable_names, bits)
        model.load_weights(list(quantized_state.items()), strict=False)
        q_train_eval = evaluate(model, dataset, runtime.train, "train")
        q_test_eval = evaluate(model, dataset, runtime.train, "test")
        q_test_bpt = bits_per_token_from_loss(q_test_eval)
        q_test_bpb = q_test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
        stats["total_bytes_est"] = float(stats["payload_bytes_est"]) + float(tables.bytes_total)
        stats["total_mb_est"] = stats["total_bytes_est"] / (1024.0 * 1024.0)
        result["quantized"].append(
            {
                "bits": bits,
                "train_nats_per_token": float(q_train_eval),
                "test_nats_per_token": float(q_test_eval),
                "test_bits_per_token": float(q_test_bpt),
                "test_bits_per_byte": None if q_test_bpb is None else float(q_test_bpb),
                "stats": {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in stats.items()},
            }
        )
        model.load_weights(list(state.items()), strict=False)

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if args.save_state:
        save_state_npz(Path(args.save_state), state)

    print(
        f"  Te:{_fmt_float(test_eval)} "
        f"bpt:{_fmt_float(test_bpt)} "
        f"bpb:{_fmt_float(test_bpb)} "
        f"Of:{metrics.overfit_pct:+.1f}% "
        f"T:{metrics.train_time_sec:.0f}s"
    )
    for item in result["quantized"]:
        print(
            f"  quant bits={item['bits']} "
            f"test_bpb={item['test_bits_per_byte'] if item['test_bits_per_byte'] is not None else 'n/a'} "
            f"total_mb_est={item['stats']['total_mb_est']:.3f}"
        )
    print(f"\n  Wrote JSON summary to {args.json}")
    if args.save_state:
        print(f"  Wrote NPZ state to {args.save_state}")


if __name__ == "__main__":
    main()
