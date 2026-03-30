#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, seed_everything, train_loss_fn
from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, quantize_trainable_params


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
            eval_batches=args.eval_batches,
        ),
    )


def build_model(vocab_size: int, args: argparse.Namespace) -> ConkerThreeModel:
    config = ConkerThreeConfig(max_seq_len=args.seq_len, local_window=4)
    if args.linear_half_life_max is not None:
        config = replace(config, linear_half_life_max=args.linear_half_life_max)
    if args.oscillatory_frac is not None:
        config = replace(
            config,
            oscillatory_frac=args.oscillatory_frac,
            oscillatory_period_min=args.oscillatory_period_min,
            oscillatory_period_max=args.oscillatory_period_max,
        )
    if args.static_bank_gate:
        config = replace(config, static_bank_gate=True, bank_gate_span=args.bank_gate_span)
    config = scale_config(config, args.scale)
    if args.local_hidden_mult is not None:
        config = replace(
            config,
            local_hidden=tuple(max(int(round(width * args.local_hidden_mult)), 1) for width in config.local_hidden),
        )
    if args.local_scale_override is not None:
        config = replace(config, local_scale=args.local_scale_override)
    return ConkerThreeModel(vocab_size=vocab_size, config=config)


def train_with_quantization(model: nn.Module, dataset, train_config, seed: int, train_quant_bits: int):
    params = count_trainable_params(model)
    value_and_grad = nn.value_and_grad(model, train_loss_fn)
    optimizer = optim.AdamW(learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay)
    seed_everything(seed + 1000)
    losses: list[float] = []
    best = float("inf")
    start = time.time()

    for step in range(1, train_config.steps + 1):
        x, y = dataset.batch("train", train_config.batch_size, train_config.seq_len)
        loss, grads = value_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=train_config.grad_clip)
        optimizer.update(model, grads)
        flat_full = dict(nn.utils.tree_flatten(model.parameters()))
        trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
        quantized_state, _ = quantize_trainable_params(flat_full, trainable_names, train_quant_bits)
        model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
        mx.eval(model.parameters(), optimizer.state)

        current = loss.item()
        losses.append(current)
        if current < best:
            best = current

        if step % train_config.log_every == 0:
            recent = float(np.mean(losses[-train_config.log_every :]))
            speed = (step * train_config.batch_size * train_config.seq_len) / max(time.time() - start, 1e-9)
            print(f"      {step:5d} | loss {recent:.4f} | best {best:.4f} | {speed:.0f} ch/s")

    elapsed = time.time() - start
    train_eval = evaluate(model, dataset, train_config, "train")
    test_eval = evaluate(model, dataset, train_config, "test")
    return {
        "params": params,
        "train_eval_loss": train_eval,
        "test_eval_loss": test_eval,
        "overfit_pct": (test_eval / train_eval - 1.0) * 100.0,
        "train_time_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-3 pack-trained bridge.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1800)
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
    parser.add_argument("--train-quant-bits", type=int, default=6)
    parser.add_argument("--quant-bits", type=int, action="append", default=[4, 6])
    parser.add_argument("--local-hidden-mult", type=float, default=None)
    parser.add_argument("--local-scale-override", type=float, default=None)
    parser.add_argument("--static-bank-gate", action="store_true")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    model = build_model(dataset.vocab_size, args)

    print("\n  conker-3 pack-trained bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} scale={args.scale:.3f} "
        f"steps={runtime.train.steps} lr={runtime.train.learning_rate:g} "
        f"train_quant_bits={args.train_quant_bits}"
    )
    print(
        f"  half_life_max={model.config.linear_half_life_max:.1f} osc_frac={model.config.oscillatory_frac:.3f} "
        f"periods={model.config.oscillatory_period_min:.1f}..{model.config.oscillatory_period_max:.1f} "
        f"static_bank_gate={model.config.static_bank_gate} params={count_trainable_params(model):,}"
    )

    metrics = train_with_quantization(model, dataset, runtime.train, args.seed, args.train_quant_bits)
    train_bpt = bits_per_token_from_loss(metrics["train_eval_loss"])
    test_bpt = bits_per_token_from_loss(metrics["test_eval_loss"])
    test_bpb = test_bpt * dataset.test_tokens_per_byte

    result = {
        "title": "conker-3 pack-trained bridge",
        "config": asdict(runtime),
        "model": {
            "variant": "window4",
            "scale": args.scale,
            "params": metrics["params"],
            "seed": args.seed,
            "linear_half_life_max": model.config.linear_half_life_max,
            "oscillatory_frac": model.config.oscillatory_frac,
            "oscillatory_period_min": model.config.oscillatory_period_min,
            "oscillatory_period_max": model.config.oscillatory_period_max,
            "static_bank_gate": model.config.static_bank_gate,
            "bank_gate_span": model.config.bank_gate_span,
            "local_hidden": list(model.config.local_hidden),
            "local_scale": model.config.local_scale,
            "train_quant_bits": args.train_quant_bits,
            "train_eval_loss": metrics["train_eval_loss"],
            "test_eval_loss": metrics["test_eval_loss"],
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "test_bpb": test_bpb,
            "overfit_pct": metrics["overfit_pct"],
            "train_time_sec": metrics["train_time_sec"],
        },
    }

    flat_full = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
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
    result["quantization"] = quant_rows

    print(
        f"  Te:{metrics['test_eval_loss']:.4f} "
        f"bpt:{test_bpt:.4f} bpb:{test_bpb:.4f} "
        f"Of:{metrics['overfit_pct']:+.1f}% T:{metrics['train_time_sec']:.0f}s"
    )
    for row in quant_rows:
        print(
            f"  {row['scheme']}: bpb:{row['test_bpb']:.4f} "
            f"payload_mb_est:{row['payload_mb_est']:.3f}"
        )

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
