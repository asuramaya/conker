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
from conker.src.quantize import KEEP_FLOAT_MAX_NUMEL, _dequantize_float_array, _quantize_float_array, bits_per_token_from_loss


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
    config = ConkerThreeConfig(max_seq_len=args.seq_len, local_window=4)
    if args.linear_half_life_max is not None:
        config = replace(config, linear_half_life_max=args.linear_half_life_max)
    config = replace(
        config,
        oscillatory_frac=args.oscillatory_frac,
        oscillatory_period_min=args.oscillatory_period_min,
        oscillatory_period_max=args.oscillatory_period_max,
    )
    if args.static_bank_gate:
        config = replace(config, static_bank_gate=True, bank_gate_span=args.bank_gate_span)
    config = scale_config(config, args.scale)
    return ConkerThreeModel(vocab_size=vocab_size, config=config)


def evaluate_bpb(model: nn.Module, dataset, train_config) -> tuple[float, float]:
    dataset.test_stream.reset()
    loss = evaluate(model, dataset, train_config, "test")
    bpt = bits_per_token_from_loss(loss)
    return loss, bpt * dataset.test_tokens_per_byte


def quantize_named(
    full_state: dict[str, mx.array],
    trainable_names: set[str],
    *,
    linear_bits: int,
    local_bits: int | None,
    keep_fp16_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, mx.array], dict[str, float]]:
    updated = dict(full_state)
    payload_bytes = 0.0
    quantized_params = 0
    kept_params = 0

    for name in trainable_names:
        arr = full_state[name]
        if not mx.issubdtype(arr.dtype, mx.floating):
            payload_bytes += float(arr.nbytes)
            continue
        if any(name.startswith(prefix) for prefix in keep_fp16_prefixes) or int(arr.size) <= KEEP_FLOAT_MAX_NUMEL:
            kept = np.array(arr.astype(mx.float16), dtype=np.float16, copy=False)
            updated[name] = mx.array(kept, dtype=arr.dtype)
            payload_bytes += float(kept.nbytes)
            kept_params += int(arr.size)
            continue
        if name.startswith("local_"):
            bits = local_bits if local_bits is not None else linear_bits
        else:
            bits = linear_bits
        q, scale = _quantize_float_array(arr, bits)
        updated[name] = _dequantize_float_array(q, scale, arr.dtype)
        payload_bytes += float((arr.size * bits) / 8.0 + scale.nbytes)
        quantized_params += int(arr.size)

    return updated, {
        "payload_bytes_est": payload_bytes,
        "payload_mb_est": payload_bytes / (1024.0 * 1024.0),
        "quantized_params": quantized_params,
        "kept_params": kept_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-3 subsystem quantization probe.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--static-bank-gate", action="store_true")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    model = build_model(dataset.vocab_size, args)

    print("\n  conker-3 subsystem quant probe\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} scale={args.scale:.3f} "
        f"steps={runtime.train.steps} half_life_max={args.linear_half_life_max:.1f} "
        f"osc_frac={args.oscillatory_frac:.3f}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker3_subsystem_quant_probe")
    baseline_loss, baseline_bpb = evaluate_bpb(model, dataset, runtime.train)

    full_state = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    schemes = [
        ("uniform_int6", 6, 6, ()),
        ("uniform_int4", 4, 4, ()),
        ("linear_int6_local_fp16", 6, None, ("local_",)),
        ("linear_int4_local_fp16", 4, None, ("local_",)),
        ("linear_int6_local_int8", 6, 8, ()),
        ("linear_int4_local_int8", 4, 8, ()),
    ]

    rows = []
    print(f"  baseline_bpb:{baseline_bpb:.4f} train_time:{metrics.train_time_sec:.1f}s")
    for label, linear_bits, local_bits, keep_fp16_prefixes in schemes:
        q_state, stats = quantize_named(
            full_state,
            trainable_names,
            linear_bits=linear_bits,
            local_bits=local_bits,
            keep_fp16_prefixes=keep_fp16_prefixes,
        )
        model.update(nn.utils.tree_unflatten(list(q_state.items())))
        q_loss, q_bpb = evaluate_bpb(model, dataset, runtime.train)
        rows.append(
            {
                "scheme": label,
                "linear_bits": linear_bits,
                "local_bits": local_bits,
                "keep_fp16_prefixes": list(keep_fp16_prefixes),
                "test_eval_loss": q_loss,
                "test_bpb": q_bpb,
                "delta_bpb": q_bpb - baseline_bpb,
                **stats,
            }
        )
        print(f"  {label}: bpb:{q_bpb:.4f} delta:{q_bpb - baseline_bpb:+.4f} payload_mb_est:{stats['payload_mb_est']:.3f}")
        model.update(nn.utils.tree_unflatten(list(full_state.items())))

    result = {
        "title": "conker-3 subsystem quant probe",
        "config": asdict(runtime),
        "model": {
            "variant": "window4",
            "scale": args.scale,
            "params": count_trainable_params(model),
            "seed": args.seed,
            "linear_half_life_max": args.linear_half_life_max,
            "oscillatory_frac": args.oscillatory_frac,
            "oscillatory_period_min": args.oscillatory_period_min,
            "oscillatory_period_max": args.oscillatory_period_max,
            "static_bank_gate": args.static_bank_gate,
            "bank_gate_span": args.bank_gate_span,
            "test_eval_loss": baseline_loss,
            "test_bpb": baseline_bpb,
            "train_time_sec": metrics.train_time_sec,
        },
        "schemes": rows,
    }

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
