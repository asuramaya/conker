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
import mlx.optimizers as optim
import numpy as np

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, train_model
from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, quantize_trainable_params


class ChunkModeGate(nn.Module):
    def __init__(self, batch_size: int, group_count: int):
        super().__init__()
        self.logits = mx.zeros((batch_size, group_count), dtype=mx.float32)


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
    config = replace(config, linear_half_life_max=args.linear_half_life_max)
    config = scale_config(config, args.scale)
    return ConkerThreeModel(vocab_size=vocab_size, config=config)


def expand_group_gate(logits: mx.array, mode_count: int, gate_span: float) -> mx.array:
    group_count = int(logits.shape[-1])
    boundaries = np.linspace(0, mode_count, group_count + 1, dtype=np.int32)
    values = 1.0 + gate_span * mx.tanh(logits)
    pieces = []
    for idx in range(group_count):
        width = int(boundaries[idx + 1] - boundaries[idx])
        if width <= 0:
            continue
        pieces.append(mx.broadcast_to(values[:, idx : idx + 1], (values.shape[0], width)))
    return mx.concatenate(pieces, axis=-1)


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    batch_size, timesteps, vocab_size = logits.shape
    return mx.mean(
        nn.losses.cross_entropy(
            logits.reshape(batch_size * timesteps, vocab_size),
            targets.reshape(batch_size * timesteps),
        )
    )


def evaluate_online_gate_ttt(
    model: ConkerThreeModel,
    dataset,
    runtime: RuntimeConfig,
    *,
    chunk_len: int,
    eval_chunks: int,
    group_count: int,
    gate_span: float,
    adapt_steps: int,
    adapt_lr: float,
    gate_l2: float,
) -> dict[str, float]:
    dataset.test_stream.reset()
    gate = ChunkModeGate(runtime.train.batch_size, group_count)
    optimizer = optim.Adam(learning_rate=adapt_lr) if adapt_steps > 0 else None

    def gate_loss_fn(gate_module: ChunkModeGate, x: mx.array, y: mx.array) -> mx.array:
        mode_gate = expand_group_gate(gate_module.logits, model.config.linear_modes, gate_span)
        logits = model.forward_with_mode_gate(x, mode_gate)
        loss = cross_entropy_loss(logits, y)
        if gate_l2 > 0:
            loss = loss + gate_l2 * mx.mean(gate_module.logits * gate_module.logits)
        return loss

    value_and_grad = nn.value_and_grad(gate, gate_loss_fn) if adapt_steps > 0 else None

    total_loss = 0.0
    total_gate_rms = 0.0
    total_tokens = 0

    for _ in range(eval_chunks):
        x, y = dataset.batch("test", runtime.train.batch_size, chunk_len)
        mode_gate = expand_group_gate(gate.logits, model.config.linear_modes, gate_span)
        logits = model.forward_with_mode_gate(x, mode_gate)
        loss = cross_entropy_loss(logits, y)
        gate_rms = mx.sqrt(mx.mean(gate.logits * gate.logits))
        mx.eval(loss, gate_rms)
        total_loss += float(loss.item()) * x.shape[0] * x.shape[1]
        total_gate_rms += float(gate_rms.item())
        total_tokens += int(x.shape[0]) * int(x.shape[1])

        if adapt_steps > 0 and value_and_grad is not None and optimizer is not None:
            for _ in range(adapt_steps):
                adapt_loss, grads = value_and_grad(gate, x, y)
                optimizer.update(gate, grads)
                mx.eval(gate.parameters(), optimizer.state, adapt_loss)

    mean_loss = total_loss / max(total_tokens, 1)
    mean_bpt = bits_per_token_from_loss(mean_loss)
    return {
        "test_eval_loss": mean_loss,
        "test_bits_per_token": mean_bpt,
        "test_bpb": mean_bpt * dataset.test_tokens_per_byte,
        "mean_gate_rms": total_gate_rms / max(eval_chunks, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-3 online TTT gate probe.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--chunk-len", type=int, default=64)
    parser.add_argument("--eval-chunks", type=int, default=32)
    parser.add_argument("--group-count", type=int, default=16)
    parser.add_argument("--gate-span", type=float, default=0.5)
    parser.add_argument("--gate-l2", type=float, default=1e-3)
    parser.add_argument("--quant-bits", type=int, action="append", default=[6])
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    model = build_model(dataset.vocab_size, args)

    print("\n  conker-3 ttt probe\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} scale={args.scale:.3f} "
        f"half_life_max={args.linear_half_life_max:.1f} steps={runtime.train.steps} "
        f"chunk_len={args.chunk_len} eval_chunks={args.eval_chunks}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker3_ttt_probe")

    eval_rows = []
    schedules = [
        ("baseline", 0, 0.0),
        ("gate_ttt_s1_lr0p10", 1, 0.10),
        ("gate_ttt_s3_lr0p03", 3, 0.03),
    ]

    for label, adapt_steps, adapt_lr in schedules:
        row = {
            "label": label,
            "adapt_steps": adapt_steps,
            "adapt_lr": adapt_lr,
            "weight_state": "fp16",
        }
        row.update(
            evaluate_online_gate_ttt(
                model,
                dataset,
                runtime,
                chunk_len=args.chunk_len,
                eval_chunks=args.eval_chunks,
                group_count=args.group_count,
                gate_span=args.gate_span,
                adapt_steps=adapt_steps,
                adapt_lr=adapt_lr,
                gate_l2=args.gate_l2,
            )
        )
        eval_rows.append(row)
        print(f"  {label}: fp16 bpb:{row['test_bpb']:.4f} gate_rms:{row['mean_gate_rms']:.4f}")

    full_state = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    for bits in sorted(set(args.quant_bits)):
        quantized_state, stats = quantize_trainable_params(full_state, trainable_names, bits)
        model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
        for label, adapt_steps, adapt_lr in schedules:
            row = {
                "label": label,
                "adapt_steps": adapt_steps,
                "adapt_lr": adapt_lr,
                "weight_state": f"uniform_int{bits}",
                **stats,
            }
            row.update(
                evaluate_online_gate_ttt(
                    model,
                    dataset,
                    runtime,
                    chunk_len=args.chunk_len,
                    eval_chunks=args.eval_chunks,
                    group_count=args.group_count,
                    gate_span=args.gate_span,
                    adapt_steps=adapt_steps,
                    adapt_lr=adapt_lr,
                    gate_l2=args.gate_l2,
                )
            )
            eval_rows.append(row)
            print(f"  {label}: int{bits} bpb:{row['test_bpb']:.4f} gate_rms:{row['mean_gate_rms']:.4f}")
        model.update(nn.utils.tree_unflatten(list(full_state.items())))

    result = {
        "title": "conker-3 ttt gate probe",
        "config": asdict(runtime),
        "model": {
            "preset": "conker3",
            "variant": "window4",
            "scale": args.scale,
            "seed": args.seed,
            "half_life_max": args.linear_half_life_max,
            "params": count_trainable_params(model),
            "chunk_len": args.chunk_len,
            "eval_chunks": args.eval_chunks,
            "group_count": args.group_count,
            "gate_span": args.gate_span,
            "gate_l2": args.gate_l2,
            "train_time_sec": metrics.train_time_sec,
        },
        "rows": eval_rows,
    }

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
