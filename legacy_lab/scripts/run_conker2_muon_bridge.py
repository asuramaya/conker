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
from conker.src.conker2 import ConkerTwoConfig, ConkerTwoModel
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.muon import SplitMuonAdam
from conker.src.quantize import bits_per_token_from_loss, estimate_trainable_payload_bytes, quantize_trainable_params


def config_for_variant(args: argparse.Namespace, seq_len: int) -> ConkerTwoConfig:
    config = ConkerTwoConfig(max_seq_len=seq_len, linear_modes=args.linear_modes)
    if args.variant != "untied_base":
        raise ValueError("Muon pilot currently supports only the untied_base Conker-2 branch.")
    variant_cfg = replace(config, share_embedding=False)
    return scale_config(variant_cfg, args.scale)


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


def train_with_muon(model: nn.Module, dataset, train_config, seed: int, optimizer: SplitMuonAdam):
    params = count_trainable_params(model)
    value_and_grad = nn.value_and_grad(model, train_loss_fn)
    seed_everything(seed + 1000)
    losses: list[float] = []
    best = float("inf")
    start = time.time()

    for step in range(1, train_config.steps + 1):
        x, y = dataset.batch("train", train_config.batch_size, train_config.seq_len)
        loss, grads = value_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=train_config.grad_clip)
        optimizer.step(model, grads, step=step)

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
    parser = argparse.ArgumentParser(description="Conker-2 Muon bridge on official parameter-golf token shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--scale", type=float, default=11.5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument("--variant", choices=["untied_base"], default="untied_base")
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-backend-steps", type=int, default=5)
    parser.add_argument("--muon-momentum-warmup-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=500)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    config = config_for_variant(args, runtime.train.seq_len)
    model = ConkerTwoModel(vocab_size=dataset.vocab_size, config=config)
    optimizer = SplitMuonAdam(
        model,
        learning_rate=runtime.train.learning_rate,
        weight_decay=runtime.train.weight_decay,
        muon_momentum=args.muon_momentum,
        muon_backend_steps=args.muon_backend_steps,
        muon_momentum_warmup_start=args.muon_momentum_warmup_start,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
    )

    print("\n  conker-2 muon bridge\n")
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
    print(
        f"  optimizer=muon+adam muon_momentum={args.muon_momentum} "
        f"muon_steps={args.muon_backend_steps} warmup_start={args.muon_momentum_warmup_start} "
        f"warmup_steps={args.muon_momentum_warmup_steps}"
    )

    metrics = train_with_muon(model, dataset, runtime.train, args.seed, optimizer)
    train_bpt = bits_per_token_from_loss(metrics["train_eval_loss"])
    test_bpt = bits_per_token_from_loss(metrics["test_eval_loss"])
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    result = {
        "title": "conker-2 muon bridge cell",
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
            "params": metrics["params"],
            "seed": args.seed,
            "linear_modes": config.linear_modes,
            "share_embedding": config.share_embedding,
            "linear_impl": config.linear_impl,
            "train_eval_loss": metrics["train_eval_loss"],
            "test_eval_loss": metrics["test_eval_loss"],
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "train_bpb": None,
            "test_bpb": test_bpb,
            "overfit_pct": metrics["overfit_pct"],
            "train_time_sec": metrics["train_time_sec"],
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
            "optimizer": "muon+adam",
            "muon_momentum": args.muon_momentum,
            "muon_backend_steps": args.muon_backend_steps,
            "muon_momentum_warmup_start": args.muon_momentum_warmup_start,
            "muon_momentum_warmup_steps": args.muon_momentum_warmup_steps,
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
        f"  Te:{metrics['test_eval_loss']:.4f} "
        f"bpt:{result['model']['test_bits_per_token']:.4f} "
        f"bpb:{result['model']['test_bpb']:.4f} "
        f"Of:{metrics['overfit_pct']:+.1f}% "
        f"T:{metrics['train_time_sec']:.0f}s"
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
