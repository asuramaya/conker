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
from carving_machine.training import count_trainable_params, evaluate, loss_fn, train_model
from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker6 import ConkerSixConfig, ConkerSixModel, scale_config
from conker.src.golf_data import _load_golf_shard, build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, estimate_trainable_payload_bytes, quantize_trainable_params


def base_config_for_variant(args: argparse.Namespace, seq_len: int) -> ConkerThreeConfig:
    config = ConkerThreeConfig(max_seq_len=seq_len, linear_modes=args.linear_modes, local_window=args.local_window)
    if args.variant == "window4":
        variant_cfg = replace(config, local_window=4)
    elif args.variant == "window16":
        variant_cfg = replace(config, local_window=16)
    elif args.variant == "gated":
        variant_cfg = replace(config, mix_mode="gated")
    elif args.variant == "linear_only":
        variant_cfg = replace(config, enable_local=False)
    elif args.variant == "base":
        variant_cfg = config
    else:
        raise ValueError(f"Unknown Conker-6 base variant: {args.variant}")
    return replace(
        variant_cfg,
        linear_half_life_max=args.linear_half_life_max,
        oscillatory_frac=args.oscillatory_frac,
        oscillatory_period_min=args.oscillatory_period_min,
        oscillatory_period_max=args.oscillatory_period_max,
        static_bank_gate=args.static_bank_gate,
        bank_gate_span=args.bank_gate_span,
        input_proj_scheme=args.input_proj_scheme,
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
            learning_rate=base_train.learning_rate if args.learning_rate is None else args.learning_rate,
            seeds=(args.seed,),
        ),
    )


def evaluate_full_split(model, dataset, runtime: RuntimeConfig, split: str) -> tuple[float, int]:
    files = dataset.train_files if split == "train" else dataset.test_files
    tokens = np.ascontiguousarray(np.concatenate([_load_golf_shard(path) for path in files], axis=0))
    batch_size = runtime.train.batch_size
    seq_len = runtime.train.seq_len
    usable = ((tokens.size - 1) // (batch_size * seq_len)) * (batch_size * seq_len)
    if usable <= 0:
        raise ValueError(f"{split} split is too short for batch_size={batch_size}, seq_len={seq_len}")

    total = 0.0
    num_batches = usable // (batch_size * seq_len)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size * seq_len
        chunk = tokens[start : start + batch_size * seq_len + 1]
        x = mx.array(chunk[:-1].reshape(batch_size, seq_len), dtype=mx.int32)
        y = mx.array(chunk[1:].reshape(batch_size, seq_len), dtype=mx.int32)
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        total += float(loss.item())
    return total / num_batches, int(num_batches * batch_size * seq_len)


def trained_legality_audit(model, dataset, runtime: RuntimeConfig, seed: int, split: str = "test") -> dict[str, object]:
    x, _ = dataset.batch(split, runtime.train.batch_size, runtime.train.seq_len)
    logits = model(x)
    probs = np.exp(np.array(logits, copy=False))
    prob_sums = probs.sum(axis=-1)

    rng = np.random.default_rng(seed + 1)
    prefix_len = runtime.train.seq_len // 2
    x_np = np.array(x, copy=True)
    if prefix_len < x_np.shape[1]:
        x_np[:, prefix_len:] = rng.integers(0, model.vocab_size, size=x_np[:, prefix_len:].shape, dtype=np.int32)
    alt_logits = model(mx.array(x_np, dtype=mx.int32))
    row_diff = np.abs(np.array(logits[:, :prefix_len, :], copy=False) - np.array(alt_logits[:, :prefix_len, :], copy=False))

    flat_rng = np.random.default_rng(seed + 2)
    x_np = np.array(x, copy=True)
    batch, seq_len = x_np.shape
    flat_len = batch * seq_len
    flat_prefix = flat_len // 2
    flat_tokens = x_np.reshape(flat_len)
    if flat_prefix < flat_len:
        flat_tokens[flat_prefix:] = flat_rng.integers(0, model.vocab_size, size=(flat_len - flat_prefix,), dtype=np.int32)
    flat_logits = np.array(model(mx.array(flat_tokens.reshape(batch, seq_len), dtype=mx.int32)), copy=False).reshape(flat_len, model.vocab_size)
    base_flat = np.array(logits, copy=False).reshape(flat_len, model.vocab_size)
    flat_diff = np.abs(base_flat[:flat_prefix] - flat_logits[:flat_prefix])

    return {
        "split": split,
        "normalization": {
            "sum_min": float(prob_sums.min()),
            "sum_max": float(prob_sums.max()),
            "sum_mean": float(prob_sums.mean()),
            "max_abs_sum_error": float(np.max(np.abs(prob_sums - 1.0))),
            "mean_abs_sum_error": float(np.mean(np.abs(prob_sums - 1.0))),
        },
        "row_future_invariance": {
            "prefix_len": int(prefix_len),
            "max_abs_logit_diff": float(row_diff.max(initial=0.0)),
            "mean_abs_logit_diff": float(row_diff.mean() if row_diff.size else 0.0),
        },
        "flat_stream_invariance": {
            "prefix_len": int(flat_prefix),
            "max_abs_logit_diff": float(flat_diff.max(initial=0.0)),
            "mean_abs_logit_diff": float(flat_diff.mean() if flat_diff.size else 0.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-6 normalized cache bridge on official parameter-golf token shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--linear-modes", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--quant-bits", type=int, action="append", default=[])
    parser.add_argument("--linear-half-life-max", type=float, default=16.0)
    parser.add_argument("--oscillatory-frac", type=float, default=0.875)
    parser.add_argument("--oscillatory-period-min", type=float, default=4.0)
    parser.add_argument("--oscillatory-period-max", type=float, default=64.0)
    parser.add_argument("--input-proj-scheme", choices=["random", "orthogonal_rows", "kernel_energy", "split_banks"], default="random")
    parser.set_defaults(static_bank_gate=True)
    parser.add_argument("--static-bank-gate", action="store_true", dest="static_bank_gate")
    parser.add_argument("--no-static-bank-gate", action="store_false", dest="static_bank_gate")
    parser.add_argument("--bank-gate-span", type=float, default=0.5)
    parser.add_argument("--blend-mode", choices=["cache_only", "fixed_blend", "learned_gate", "witten_bell", "absolute_discount"], default="learned_gate")
    parser.set_defaults(freeze_base=False)
    parser.add_argument("--freeze-base", action="store_true", dest="freeze_base")
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")
    parser.add_argument("--disable-exact3", action="store_true")
    parser.add_argument("--exact-context-span", type=int, default=0)
    parser.add_argument("--causal-projection", choices=["none", "strict_lower", "strict_lower_nonnegative"], default="none")
    parser.set_defaults(learnable_vocab_axis=True)
    parser.add_argument("--learnable-vocab-axis", action="store_true", dest="learnable_vocab_axis")
    parser.add_argument("--fixed-vocab-axis", action="store_false", dest="learnable_vocab_axis")
    parser.set_defaults(learnable_causal_mask=True)
    parser.add_argument("--learnable-causal-mask", action="store_true", dest="learnable_causal_mask")
    parser.add_argument("--fixed-causal-mask", action="store_false", dest="learnable_causal_mask")
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--unigram-discount", type=float, default=0.0)
    parser.add_argument("--exact1-discount", type=float, default=0.75)
    parser.add_argument("--exact2-discount", type=float, default=0.75)
    parser.add_argument("--exact3-discount", type=float, default=0.75)
    parser.add_argument("--fixed-base-weight", type=float, default=0.15)
    parser.add_argument("--fixed-exact1-weight", type=float, default=0.10)
    parser.add_argument("--fixed-exact2-weight", type=float, default=0.25)
    parser.add_argument("--fixed-exact3-weight", type=float, default=0.50)
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--audit-trained-legality", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["base", "window4", "window16", "gated", "linear_only"],
        default="window4",
    )
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    base_cfg = base_config_for_variant(args, runtime.train.seq_len)
    config = scale_config(
        ConkerSixConfig(
            base_config=base_cfg,
            freeze_base=args.freeze_base,
            enable_exact3=not args.disable_exact3,
            exact_context_span=args.exact_context_span,
            causal_projection=args.causal_projection,
            learnable_vocab_axis=args.learnable_vocab_axis,
            learnable_causal_mask=args.learnable_causal_mask,
            blend_mode=args.blend_mode,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_temperature=args.gate_temperature,
            unigram_discount=args.unigram_discount,
            exact1_discount=args.exact1_discount,
            exact2_discount=args.exact2_discount,
            exact3_discount=args.exact3_discount,
            fixed_base_weight=args.fixed_base_weight,
            fixed_exact1_weight=args.fixed_exact1_weight,
            fixed_exact2_weight=args.fixed_exact2_weight,
            fixed_exact3_weight=args.fixed_exact3_weight,
        ),
        args.scale,
    )
    model = ConkerSixModel(vocab_size=dataset.vocab_size, config=config)

    print("\n  conker-6 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} "
        f"lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} linear_modes={config.base_config.linear_modes} "
        f"local_window={config.base_config.local_window} half_life_max={config.base_config.linear_half_life_max:.1f} "
        f"osc_frac={config.base_config.oscillatory_frac:.2f} input_proj={config.base_config.input_proj_scheme} "
        f"static_bank_gate={config.base_config.static_bank_gate} "
        f"blend_mode={config.blend_mode} freeze_base={config.freeze_base} exact3={config.enable_exact3} "
        f"exact_context_span={config.exact_context_span} params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker6_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    full_test_eval = None
    full_test_bpt = None
    full_test_bpb = None
    full_test_tokens = None
    if args.full_eval:
        full_test_eval, full_test_tokens = evaluate_full_split(model, dataset, runtime, "test")
        full_test_bpt = bits_per_token_from_loss(full_test_eval)
        full_test_bpb = full_test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
    legality = trained_legality_audit(model, dataset, runtime, args.seed, split="test") if args.audit_trained_legality else None

    result = {
        "title": "conker-6 golf bridge cell",
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
            "preset": "conker6",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "blend_mode": config.blend_mode,
            "freeze_base": config.freeze_base,
            "linear_modes": config.base_config.linear_modes,
            "local_window": config.base_config.local_window,
            "learnable_vocab_axis": config.learnable_vocab_axis,
            "learnable_causal_mask": config.learnable_causal_mask,
            "linear_half_life_max": config.base_config.linear_half_life_max,
            "oscillatory_frac": config.base_config.oscillatory_frac,
            "oscillatory_period_min": config.base_config.oscillatory_period_min,
            "oscillatory_period_max": config.base_config.oscillatory_period_max,
            "input_proj_scheme": config.base_config.input_proj_scheme,
            "static_bank_gate": config.base_config.static_bank_gate,
            "enable_exact3": config.enable_exact3,
            "exact_context_span": config.exact_context_span,
            "causal_projection": config.causal_projection,
            "gate_hidden_dim": config.gate_hidden_dim,
            "gate_temperature": config.gate_temperature,
            "unigram_discount": config.unigram_discount,
            "exact1_discount": config.exact1_discount,
            "exact2_discount": config.exact2_discount,
            "exact3_discount": config.exact3_discount,
            "fixed_base_weight": config.fixed_base_weight,
            "fixed_exact1_weight": config.fixed_exact1_weight,
            "fixed_exact2_weight": config.fixed_exact2_weight,
            "fixed_exact3_weight": config.fixed_exact3_weight,
            "train_eval_loss": float(train_eval),
            "test_eval_loss": float(test_eval),
            "train_bits_per_token": float(train_bpt),
            "test_bits_per_token": float(test_bpt),
            "test_bpb": None if test_bpb is None else float(test_bpb),
            "full_test_eval_loss": None if full_test_eval is None else float(full_test_eval),
            "full_test_bits_per_token": None if full_test_bpt is None else float(full_test_bpt),
            "full_test_bpb": None if full_test_bpb is None else float(full_test_bpb),
            "full_test_tokens": full_test_tokens,
            "train_time_sec": float(metrics.train_time_sec),
            "learning_rate": runtime.train.learning_rate,
            "payload_bytes_est": None,
            "payload_mb_est": None,
            "trained_legality": legality,
        },
        "quantization": [],
    }

    flat_full = dict(nn.utils.tree_flatten(model.parameters()))
    trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
    result["model"]["payload_bytes_est"] = estimate_trainable_payload_bytes(flat_full, trainable_names)
    result["model"]["payload_mb_est"] = result["model"]["payload_bytes_est"] / (1024.0 * 1024.0)

    for bits in sorted(set(args.quant_bits)):
        quantized_state, quant_stats = quantize_trainable_params(flat_full, trainable_names, bits)
        model.update(nn.utils.tree_unflatten(list(quantized_state.items())))
        quant_test = evaluate(model, dataset, runtime.train, "test")
        quant_bpt = bits_per_token_from_loss(quant_test)
        quant_bpb = quant_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
        result["quantization"].append(
            {
                "scheme": f"uniform_int{bits}",
                "bits": float(bits),
                "test_eval_loss": float(quant_test),
                "test_bits_per_token": float(quant_bpt),
                "test_bpb": None if quant_bpb is None else float(quant_bpb),
                "payload_bytes_est": float(quant_stats["payload_bytes_est"]),
                "payload_mb_est": float(quant_stats["payload_mb_est"]),
                "quantized_params": int(quant_stats["quantized_params"]),
                "kept_params": int(quant_stats["kept_params"]),
            }
        )
    if result["quantization"]:
        model.update(nn.utils.tree_unflatten(list(flat_full.items())))

    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{result['model']['test_bits_per_token']:.4f} "
        f"bpb:{result['model']['test_bpb']:.4f} "
        f"T:{metrics.train_time_sec:.0f}s"
    )
    if args.full_eval:
        print(
            f"  full_test: "
            f"loss:{result['model']['full_test_eval_loss']:.4f} "
            f"bpt:{result['model']['full_test_bits_per_token']:.4f} "
            f"bpb:{result['model']['full_test_bpb']:.4f} "
            f"tokens:{result['model']['full_test_tokens']}"
        )
    for row in result["quantization"]:
        print(
            f"  {row['scheme']}: "
            f"bpb:{row['test_bpb']:.4f} "
            f"bpt:{row['test_bits_per_token']:.4f} "
            f"payload_mb_est:{row['payload_mb_est']:.3f}"
        )

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
