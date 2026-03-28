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
from conker.src.quantize import (
    KEEP_FLOAT_MAX_NUMEL,
    _dequantize_float_array,
    _quantize_float_array,
    bits_per_token_from_loss,
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
            eval_batches=args.eval_batches,
            learning_rate=base_train.learning_rate if args.learning_rate is None else args.learning_rate,
            seeds=(args.seed,),
        ),
    )


def build_model(vocab_size: int, args: argparse.Namespace) -> ConkerThreeModel:
    config = ConkerThreeConfig(max_seq_len=args.seq_len, linear_modes=args.linear_modes, local_window=4)
    if args.decay_bank == "narrow":
        config = replace(config, linear_half_life_max=32.0)
    if args.linear_half_life_max is not None:
        config = replace(config, linear_half_life_max=args.linear_half_life_max)
    config = scale_config(config, args.scale)
    model = ConkerThreeModel(vocab_size=vocab_size, config=config)
    if args.decay_bank == "custom":
        if not args.decays_json:
            raise ValueError("--decay-bank custom requires --decays-json")
        decay_payload = json.loads(Path(args.decays_json).read_text(encoding="utf-8"))
        decays = decay_payload.get("decays")
        if decays is None:
            raise ValueError(f"Decay bank JSON is missing 'decays': {args.decays_json}")
        model.set_linear_decays(decays)
    return model


def evaluate_bpb(model: nn.Module, dataset, train_config) -> tuple[float, float]:
    dataset.test_stream.reset()
    loss = evaluate(model, dataset, train_config, "test")
    bpt = bits_per_token_from_loss(loss)
    return loss, bpt * dataset.test_tokens_per_byte


def mixed_quantize(
    full_state: dict[str, mx.array],
    trainable_names: set[str],
    default_bits: int,
    override_bits: dict[str, int] | None,
    keep_fp16_names: set[str],
) -> tuple[dict[str, mx.array], dict[str, float]]:
    updated = dict(full_state)
    payload_bytes = 0.0
    quantized_params = 0
    kept_params = 0
    override_bits = override_bits or {}

    for name in trainable_names:
        arr = full_state[name]
        if not mx.issubdtype(arr.dtype, mx.floating):
            payload_bytes += float(arr.nbytes)
            continue
        if name in keep_fp16_names or int(arr.size) <= KEEP_FLOAT_MAX_NUMEL:
            kept = np.array(arr.astype(mx.float16), dtype=np.float16, copy=False)
            updated[name] = mx.array(kept, dtype=arr.dtype)
            payload_bytes += float(kept.nbytes)
            kept_params += int(arr.size)
            continue
        bits = override_bits.get(name, default_bits)
        q, scale = _quantize_float_array(arr, bits)
        updated[name] = _dequantize_float_array(q, scale, arr.dtype)
        payload_bytes += float((arr.size * bits) / 8.0 + scale.nbytes)
        quantized_params += int(arr.size)

    return updated, {
        "bits": float(default_bits),
        "payload_bytes_est": payload_bytes,
        "payload_mb_est": payload_bytes / (1024.0 * 1024.0),
        "quantized_params": quantized_params,
        "kept_params": kept_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-3 mixed low-bit audit.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--decay-bank", choices=["logspace", "narrow", "custom"], default="logspace")
    parser.add_argument("--decays-json", default=None)
    parser.add_argument("--linear-half-life-max", type=float, default=None)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    model = build_model(dataset.vocab_size, args)

    print("\n  conker-3 mixed quant audit\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} scale={args.scale:.3f} "
        f"decay_bank={args.decay_bank} half_life_max={args.linear_half_life_max} "
        f"steps={runtime.train.steps} eval_batches={runtime.train.eval_batches}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker3_mixed_quant_audit")
    baseline_loss, baseline_bpb = evaluate_bpb(model, dataset, runtime.train)

    full_state = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}

    schemes = [
        ("uniform_int3", 3, {}, set()),
        ("uniform_int4", 4, {}, set()),
        ("int4_keep_local_out", 4, {}, {"local_readout.out.weight"}),
        ("int4_keep_local_out_embed", 4, {}, {"local_readout.out.weight", "local_embedding.weight"}),
        (
            "int4_keep_local_path",
            4,
            {},
            {
                name
                for name in trainable_names
                if name.startswith("local_readout.") or name.startswith("local_embedding.")
            },
        ),
        ("int3_keep_local_out", 3, {}, {"local_readout.out.weight"}),
        ("int3_keep_local_out_embed", 3, {}, {"local_readout.out.weight", "local_embedding.weight"}),
        (
            "int3_keep_local_path",
            3,
            {},
            {
                name
                for name in trainable_names
                if name.startswith("local_readout.") or name.startswith("local_embedding.")
            },
        ),
        (
            "int3_rest_local_int4",
            3,
            {
                name: 4
                for name in trainable_names
                if name.startswith("local_readout.") or name.startswith("local_embedding.")
            },
            set(),
        ),
    ]

    result = {
        "title": "conker-3 mixed quant audit",
        "config": asdict(runtime),
        "model": {
            "variant": "window4",
            "scale": args.scale,
            "decay_bank": args.decay_bank,
            "decays_json": args.decays_json,
            "linear_half_life_max": args.linear_half_life_max,
            "params": count_trainable_params(model),
            "seed": args.seed,
            "test_eval_loss": baseline_loss,
            "test_bpb": baseline_bpb,
            "train_time_sec": metrics.train_time_sec,
        },
        "schemes": [],
    }

    print(f"  baseline_bpb:{baseline_bpb:.4f} train_time:{metrics.train_time_sec:.1f}s")
    for name, bits, override_bits, keep_fp16 in schemes:
        quant_state, stats = mixed_quantize(full_state, trainable_names, bits, override_bits, keep_fp16)
        model.update(nn.utils.tree_unflatten(list(quant_state.items())))
        q_loss, q_bpb = evaluate_bpb(model, dataset, runtime.train)
        row = {
            "scheme": name,
            "bits": bits,
            "override_bits": dict(sorted(override_bits.items())),
            "keep_fp16": sorted(keep_fp16),
            "test_eval_loss": q_loss,
            "test_bpb": q_bpb,
            "delta_bpb": q_bpb - baseline_bpb,
            **stats,
        }
        result["schemes"].append(row)
        print(
            f"  {name}: bpb:{q_bpb:.4f} delta:{q_bpb - baseline_bpb:+.4f} "
            f"payload_mb_est:{stats['payload_mb_est']:.3f}"
        )
        model.update(nn.utils.tree_unflatten(list(full_state.items())))

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
