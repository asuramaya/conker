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
from conker.src.conker2 import ConkerTwoConfig, ConkerTwoModel
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, estimate_trainable_payload_bytes, quantize_trainable_params


def scaled_count(base: int, scale: float) -> int:
    return max(int(round(base * scale)), 1)


def scaled_sample_count(base: int, size: int, scale: float) -> int:
    return min(size, max(int(round(base * scale)), 1))


def scale_config(config: ConkerTwoConfig, scale: float) -> ConkerTwoConfig:
    if scale == 1.0:
        return config
    corr = config.correction
    fast_size = scaled_count(corr.fast_size, scale)
    mid_size = scaled_count(corr.mid_size, scale)
    slow_size = scaled_count(corr.slow_size, scale)
    return replace(
        config,
        embedding_dim=scaled_count(config.embedding_dim, scale),
        linear_modes=scaled_count(config.linear_modes, scale),
        linear_hidden=tuple(scaled_count(width, scale) for width in config.linear_hidden),
        mixer_hidden=tuple(scaled_count(width, scale) for width in config.mixer_hidden),
        correction=replace(
            corr,
            fast_size=fast_size,
            mid_size=mid_size,
            slow_size=slow_size,
            controller_width=scaled_count(corr.controller_width, scale),
            fast_sample_size=scaled_sample_count(corr.fast_sample_size, fast_size, scale),
            mid_sample_size=scaled_sample_count(corr.mid_sample_size, mid_size, scale),
            slow_sample_size=scaled_sample_count(corr.slow_sample_size, slow_size, scale),
            readout_hidden=tuple(scaled_count(width, scale) for width in corr.readout_hidden),
        ),
    )


def config_for_variant(args: argparse.Namespace, seq_len: int) -> ConkerTwoConfig:
    config = ConkerTwoConfig(max_seq_len=seq_len, linear_modes=args.linear_modes)
    if args.variant == "base":
        variant_cfg = config
    elif args.variant == "untied_base":
        variant_cfg = replace(config, share_embedding=False)
    elif args.variant == "untied_base_fft":
        variant_cfg = replace(config, share_embedding=False, linear_impl="fft")
    elif args.variant == "linear_only":
        variant_cfg = replace(config, enable_correction=False, use_bias=False)
    elif args.variant == "linear_only_fft":
        variant_cfg = replace(config, enable_correction=False, use_bias=False, linear_impl="fft")
    elif args.variant == "correction_only":
        variant_cfg = replace(config, enable_linear=False, use_bias=False)
    elif args.variant == "equal_logit":
        variant_cfg = replace(config, mix_mode="equal")
    elif args.variant == "untied_equal_logit":
        variant_cfg = replace(config, mix_mode="equal", share_embedding=False)
    elif args.variant == "no_bias":
        variant_cfg = replace(config, use_bias=False)
    elif args.variant == "probability_mix":
        variant_cfg = replace(config, mix_space="probability", use_bias=False)
    else:
        raise ValueError(f"Unknown Conker-2 variant: {args.variant}")
    return scale_config(variant_cfg, args.scale)


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
    parser = argparse.ArgumentParser(description="Conker-2 bridge on official parameter-golf token shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument(
        "--variant",
        choices=[
            "base",
            "untied_base",
            "untied_base_fft",
            "linear_only",
            "linear_only_fft",
            "correction_only",
            "equal_logit",
            "untied_equal_logit",
            "no_bias",
            "probability_mix",
        ],
        default="base",
    )
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    config = config_for_variant(args, runtime.train.seq_len)
    model = ConkerTwoModel(vocab_size=dataset.vocab_size, config=config)

    print("\n  conker-2 golf bridge\n")
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
        f"params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker2_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    train_bpb = None
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    result = {
        "title": "conker-2 golf bridge cell",
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
            "preset": "conker2",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "linear_modes": config.linear_modes,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "train_bpb": train_bpb,
            "test_bpb": test_bpb,
            "overfit_pct": metrics.overfit_pct,
            "train_time_sec": metrics.train_time_sec,
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": None,
            "payload_mb_est": None,
            "embedding_dim": config.embedding_dim,
            "linear_hidden": list(config.linear_hidden),
            "mixer_hidden": list(config.mixer_hidden),
            "fast_size": config.correction.fast_size,
            "mid_size": config.correction.mid_size,
            "slow_size": config.correction.slow_size,
            "controller_width": config.correction.controller_width,
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

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
