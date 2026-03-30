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
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.conker1 import ConkerOneModel
from conker.src.golf_data import build_parameter_golf_dataset


def bits_per_token_from_loss(token_loss_nats: float) -> float:
    import math

    return token_loss_nats / math.log(2.0)


def build_runtime(args: argparse.Namespace):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-1 bridge on official parameter-golf token shards.")
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
    model = ConkerOneModel(vocab_size=dataset.vocab_size)

    print("\n  conker-1 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size}"
    )
    print(
        f"  train_shards={len(dataset.train_files)} val_shards={len(dataset.test_files)} "
        f"train_tokens={dataset.train_token_count:,} val_tokens={dataset.test_token_count:,}"
    )
    print(f"  params={count_trainable_params(model):,}")

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker1_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")

    result = {
        "title": "conker-1 golf bridge cell",
        "config": asdict(runtime),
        "dataset": {
            "source_path": dataset.source_path,
            "tokenizer": dataset.tokenizer,
            "train_token_count": int(dataset.train_token_count),
            "test_token_count": int(dataset.test_token_count),
        },
        "model": {
            "preset": "conker1",
            "params": metrics.params,
            "seed": args.seed,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": bits_per_token_from_loss(train_eval),
            "test_bits_per_token": bits_per_token_from_loss(test_eval),
            "overfit_pct": metrics.overfit_pct,
            "train_time_sec": metrics.train_time_sec,
        },
    }

    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{result['model']['test_bits_per_token']:.4f} "
        f"Of:{metrics.overfit_pct:+.1f}% "
        f"T:{metrics.train_time_sec:.0f}s"
    )

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
