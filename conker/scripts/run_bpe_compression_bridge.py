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
from carving_machine.experiments import build_model
from carving_machine.training import count_trainable_params, evaluate, train_model


def bits_per_byte_from_token_loss(token_loss_nats: float, tokens_per_char: float) -> float:
    return (token_loss_nats / log(2.0)) * tokens_per_char


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


def main() -> None:
    parser = argparse.ArgumentParser(description="BPE held-out compression bridge for conker.")
    parser.add_argument("--preset", required=True)
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
    model = build_model(args.preset, runtime, dataset, reservoir=None)

    print(f"\n  conker bpe compression bridge\n")
    print(f"  preset={args.preset} seed={args.seed} steps={runtime.train.steps}")
    print(f"  source={dataset.source_path} tokenizer={dataset.tokenizer}")
    print(
        f"  train_tokens={len(dataset.train_tokens):,} test_tokens={len(dataset.test_tokens):,} "
        f"| train chars/token={dataset.train_char_count / max(len(dataset.train_tokens), 1):.2f} "
        f"test chars/token={dataset.test_char_count / max(len(dataset.test_tokens), 1):.2f}"
    )
    print(f"  params={count_trainable_params(model):,}")

    metrics = train_model(model, dataset, runtime.train, args.seed, f"conker_{args.preset}_bpe_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")

    result = {
        "title": "conker bpe compression bridge cell",
        "config": asdict(runtime),
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
