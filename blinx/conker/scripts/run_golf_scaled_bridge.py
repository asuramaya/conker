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

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.experiments import MODEL_PRESETS
from carving_machine.models import HierarchicalCarverModel, MixedMemoryHierarchicalModel
from carving_machine.training import count_trainable_params, evaluate, seed_everything, train_model
from conker.src.golf_data import build_parameter_golf_dataset


def bits_per_token_from_loss(token_loss_nats: float) -> float:
    import math

    return token_loss_nats / math.log(2.0)


def scaled_count(base: int, scale: float) -> int:
    return max(int(round(base * scale)), 1)


def scaled_sample_count(base: int, size: int, scale: float) -> int:
    return min(size, max(int(round(base * scale)), 1))


def build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    runtime = RuntimeConfig(profile=args.profile)
    return replace(
        runtime,
        train=replace(
            train_config_for_profile(args.profile),
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            seeds=(args.seed,),
        ),
    )


def build_scaled_model(preset_name: str, scale: float, runtime: RuntimeConfig, vocab_size: int):
    preset = MODEL_PRESETS[preset_name]
    cfg = preset.hierarchical
    if cfg is None:
        raise ValueError(f"Preset {preset_name} does not have a hierarchical config.")

    fast_size = scaled_count(cfg.fast_size, scale)
    mid_size = scaled_count(cfg.mid_size, scale)
    slow_size = scaled_count(cfg.slow_size, scale)
    scaled_cfg = replace(
        cfg,
        fast_size=fast_size,
        mid_size=mid_size,
        slow_size=slow_size,
        fast_sample_size=scaled_sample_count(cfg.fast_sample_size, fast_size, scale),
        mid_sample_size=scaled_sample_count(cfg.mid_sample_size, mid_size, scale),
        slow_sample_size=scaled_sample_count(cfg.slow_sample_size, slow_size, scale),
    )

    if preset.kind == "hierarchical":
        model = HierarchicalCarverModel(
            vocab_size=vocab_size,
            embedding_dim=runtime.reservoir.embedding_dim,
            config=scaled_cfg,
        )
    elif preset.kind == "mixed_memory":
        model = MixedMemoryHierarchicalModel(
            vocab_size=vocab_size,
            embedding_dim=runtime.reservoir.embedding_dim,
            config=scaled_cfg,
        )
    else:
        raise ValueError(f"Unsupported preset kind for golf scaled bridge: {preset.kind}")

    model.freeze_static()
    return model, scaled_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaled single-expert bridge on official parameter-golf token shards.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    seed_everything(args.seed)
    model, scaled_cfg = build_scaled_model(args.preset, args.scale, runtime, dataset.vocab_size)

    print("\n  conker golf scaled bridge\n")
    print(
        f"  preset={args.preset} scale={args.scale:.3f} seed={args.seed} "
        f"steps={runtime.train.steps} seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size}"
    )
    print(
        f"  train_shards={len(dataset.train_files)} val_shards={len(dataset.test_files)} "
        f"train_tokens={dataset.train_token_count:,} val_tokens={dataset.test_token_count:,}"
    )
    print(
        f"  sizes=({scaled_cfg.fast_size},{scaled_cfg.mid_size},{scaled_cfg.slow_size}) "
        f"samples=({scaled_cfg.fast_sample_size},{scaled_cfg.mid_sample_size},{scaled_cfg.slow_sample_size})"
    )
    print(f"  params={count_trainable_params(model):,}")

    metrics = train_model(model, dataset, runtime.train, args.seed, f"{args.preset}_scale_{args.scale:g}_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    result = {
        "title": "conker golf scaled bridge cell",
        "config": asdict(runtime),
        "scale": args.scale,
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
            "preset": args.preset,
            "params": metrics.params,
            "seed": args.seed,
            "scale": args.scale,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "train_bpb": None,
            "test_bpb": test_bpb,
            "overfit_pct": metrics.overfit_pct,
            "train_time_sec": metrics.train_time_sec,
            "fast_size": scaled_cfg.fast_size,
            "mid_size": scaled_cfg.mid_size,
            "slow_size": scaled_cfg.slow_size,
            "fast_sample_size": scaled_cfg.fast_sample_size,
            "mid_sample_size": scaled_cfg.mid_sample_size,
            "slow_sample_size": scaled_cfg.slow_sample_size,
        },
    }

    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{test_bpt:.4f} "
        f"bpb:{test_bpb:.4f} "
        f"Of:{metrics.overfit_pct:+.1f}% "
        f"T:{metrics.train_time_sec:.0f}s"
    )

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
