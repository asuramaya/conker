#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from math import log
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.data import load_text8
from carving_machine.experiments import MODEL_PRESETS
from carving_machine.models import HierarchicalCarverModel, MixedMemoryHierarchicalModel
from carving_machine.training import count_trainable_params, evaluate, seed_everything, train_model


def bits_per_byte_from_token_loss(token_loss_nats: float, tokens_per_char: float) -> float:
    return (token_loss_nats / log(2.0)) * tokens_per_char


def scaled_count(base: int, scale: float) -> int:
    return max(int(round(base * scale)), 1)


def scaled_sample_count(base: int, size: int, scale: float) -> int:
    return min(size, max(int(round(base * scale)), 1))


def build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    runtime = RuntimeConfig(profile=args.profile)
    runtime = replace(
        runtime,
        data=replace(
            runtime.data,
            path=args.data,
            tokenizer="bpe_1024",
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_cache_path=args.bpe_cache,
        ),
        train=replace(
            train_config_for_profile(args.profile),
            steps=args.steps,
            seeds=(args.seed,),
        ),
    )
    return runtime


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
        raise ValueError(f"Unsupported preset kind for budget bridge: {preset.kind}")

    model.freeze_static()
    return model, scaled_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="BPE held-out budget bridge for conker.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--data", default=str(Path("data") / "text8"))
    parser.add_argument("--bpe-vocab-size", type=int, default=1024)
    parser.add_argument("--bpe-cache", default=str(Path("data") / "bpe_1024.json"))
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = load_text8(runtime.data)
    seed_everything(args.seed)
    model, scaled_cfg = build_scaled_model(args.preset, args.scale, runtime, dataset.vocab_size)

    print(f"\n  conker bpe budget bridge\n")
    print(f"  preset={args.preset} scale={args.scale:.3f} seed={args.seed} steps={runtime.train.steps}")
    print(f"  source={dataset.source_path} tokenizer={dataset.tokenizer}")
    print(
        f"  sizes=({scaled_cfg.fast_size},{scaled_cfg.mid_size},{scaled_cfg.slow_size}) "
        f"samples=({scaled_cfg.fast_sample_size},{scaled_cfg.mid_sample_size},{scaled_cfg.slow_sample_size})"
    )
    print(f"  params={count_trainable_params(model):,}")

    metrics = train_model(model, dataset, runtime.train, args.seed, f"conker_{args.preset}_scale_{args.scale:g}_bpe_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")

    result = {
        "title": "conker bpe budget bridge cell",
        "config": asdict(runtime),
        "scale": args.scale,
        "dataset": {
            "source_path": dataset.source_path,
            "tokenizer": dataset.tokenizer,
            "train_token_count": int(len(dataset.train_tokens)),
            "test_token_count": int(len(dataset.test_tokens)),
            "train_char_count": int(dataset.train_char_count),
            "test_char_count": int(dataset.test_char_count),
            "train_tokens_per_char": float(dataset.train_tokens_per_char),
            "test_tokens_per_char": float(dataset.test_tokens_per_char),
        },
        "model": {
            "preset": args.preset,
            "params": metrics.params,
            "seed": args.seed,
            "train_loss": metrics.train_loss,
            "test_loss": metrics.test_loss,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bpb": bits_per_byte_from_token_loss(train_eval, dataset.train_tokens_per_char),
            "test_bpb": bits_per_byte_from_token_loss(test_eval, dataset.test_tokens_per_char),
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
        f"bpb:{result['model']['test_bpb']:.4f} "
        f"Of:{metrics.overfit_pct:+.1f}% "
        f"T:{metrics.train_time_sec:.0f}s"
    )

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
