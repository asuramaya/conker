#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import count_trainable_params, evaluate, train_model
from conker.src.asura import AsuraConfig, AsuraModel, scale_config
from conker.src.conker3 import ConkerThreeConfig
from conker.src.golf_data import build_parameter_golf_dataset
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
        raise ValueError(f"Unknown Asura base variant: {args.variant}")
    if args.linear_half_life_max is not None:
        variant_cfg = replace(variant_cfg, linear_half_life_max=args.linear_half_life_max)
    if args.oscillatory_frac is not None:
        variant_cfg = replace(
            variant_cfg,
            oscillatory_frac=args.oscillatory_frac,
            oscillatory_period_min=args.oscillatory_period_min,
            oscillatory_period_max=args.oscillatory_period_max,
            input_proj_scheme=args.input_proj_scheme,
        )
    else:
        variant_cfg = replace(variant_cfg, input_proj_scheme=args.input_proj_scheme)
    if args.static_bank_gate:
        variant_cfg = replace(variant_cfg, static_bank_gate=True, bank_gate_span=args.bank_gate_span)
    return variant_cfg


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


def save_state_npz(path: Path, state: dict[str, mx.array]) -> None:
    arrays = {name: np.array(value) for name, value in state.items()}
    np.savez(path, **arrays)


def parse_lag_lookbacks(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Asura requires at least one lag lookback.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Asura bridge: recursive causal routing over legal lag buckets.")
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
    parser.add_argument("--learning-rate", type=float, default=None)
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
    parser.add_argument("--recency-half-life", type=float, default=8.0)
    parser.add_argument("--residual-cap", type=float, default=4.0)
    parser.add_argument("--base-feature-scale", type=float, default=1.0)
    parser.set_defaults(dynamic_support_gates=False)
    parser.add_argument("--dynamic-support-gates", action="store_true", dest="dynamic_support_gates")
    parser.add_argument("--no-dynamic-support-gates", action="store_false", dest="dynamic_support_gates")
    parser.add_argument("--dynamic-gate-span", type=float, default=0.5)
    parser.set_defaults(gate_only_mode=True)
    parser.add_argument("--gate-only-mode", action="store_true", dest="gate_only_mode")
    parser.add_argument("--no-gate-only-mode", action="store_false", dest="gate_only_mode")
    parser.add_argument("--support-gate-mode", choices=["independent", "softmax"], default="independent")
    parser.add_argument("--support-gate-topk", type=int, default=0)
    parser.add_argument("--support-gate-temperature", type=float, default=1.0)
    parser.add_argument("--support-overlap-penalty", type=float, default=0.0)
    parser.add_argument("--disable-exact1", action="store_true")
    parser.add_argument("--disable-exact2", action="store_true")
    parser.add_argument("--enable-exact3", action="store_true")
    parser.add_argument("--enable-special2", action="store_true")
    parser.add_argument("--enable-number2", action="store_true")
    parser.add_argument("--enable-urlpath2", action="store_true")
    parser.add_argument("--enable-markup2", action="store_true")
    parser.add_argument("--enable-attr2", action="store_true")
    parser.add_argument("--enable-entity2", action="store_true")
    parser.add_argument("--enable-stack2", action="store_true")
    parser.add_argument("--enable-wordclass2", action="store_true")
    parser.add_argument("--enable-delim2", action="store_true")
    parser.add_argument("--enable-delimsub2", action="store_true")
    parser.add_argument("--disable-recency", action="store_true")
    parser.set_defaults(exact1_opens_mask=False)
    parser.add_argument("--exact1-opens-mask", action="store_true", dest="exact1_opens_mask")
    parser.add_argument("--no-exact1-opens-mask", action="store_false", dest="exact1_opens_mask")
    parser.set_defaults(delim2_opens_mask=False)
    parser.add_argument("--delim2-opens-mask", action="store_true", dest="delim2_opens_mask")
    parser.add_argument("--no-delim2-opens-mask", action="store_false", dest="delim2_opens_mask")
    parser.add_argument("--variant", choices=["base", "window4", "window16", "gated", "linear_only"], default="window4")
    parser.set_defaults(freeze_base=False)
    parser.add_argument("--freeze-base", action="store_true", dest="freeze_base")
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")
    parser.add_argument("--lag-lookbacks", default="2,4,8,16,32,64,128,0")
    parser.add_argument("--lag-controller-temperature", type=float, default=1.0)
    parser.add_argument("--source-controller-temperature", type=float, default=1.0)
    parser.add_argument("--opener-controller-temperature", type=float, default=1.0)
    parser.add_argument("--residual-controller-temperature", type=float, default=1.0)
    parser.add_argument("--source-topk", type=int, default=0)
    parser.add_argument("--candidate-floor", type=float, default=0.0)
    parser.add_argument("--global-lag-cap", type=float, default=0.5)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    runtime = build_runtime(args)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    tokenizer_vocab_path = None
    if dataset.tokenizer_path is not None:
        vocab_candidate = Path(dataset.tokenizer_path).with_suffix(".vocab")
        if vocab_candidate.exists():
            tokenizer_vocab_path = str(vocab_candidate)

    base_cfg = base_config_for_variant(args, runtime.train.seq_len)
    config = scale_config(
        AsuraConfig(
            base_config=base_cfg,
            freeze_base=args.freeze_base,
            enable_exact1=not args.disable_exact1,
            enable_exact2=not args.disable_exact2,
            enable_exact3=args.enable_exact3,
            enable_special2=args.enable_special2,
            enable_number2=args.enable_number2,
            enable_urlpath2=args.enable_urlpath2,
            enable_markup2=args.enable_markup2,
            enable_attr2=args.enable_attr2,
            enable_entity2=args.enable_entity2,
            enable_stack2=args.enable_stack2,
            enable_wordclass2=args.enable_wordclass2,
            enable_delim2=args.enable_delim2,
            enable_delimsub2=args.enable_delimsub2,
            enable_recency=not args.disable_recency,
            tokenizer_vocab_path=tokenizer_vocab_path,
            recency_half_life=args.recency_half_life,
            exact_context_span=0,
            residual_cap=args.residual_cap,
            base_feature_scale=args.base_feature_scale,
            dynamic_support_gates=args.dynamic_support_gates,
            dynamic_gate_span=args.dynamic_gate_span,
            gate_only_mode=args.gate_only_mode,
            support_gate_mode=args.support_gate_mode,
            support_gate_topk=args.support_gate_topk,
            support_gate_temperature=args.support_gate_temperature,
            support_overlap_penalty=args.support_overlap_penalty,
            exact1_opens_mask=args.exact1_opens_mask,
            delim2_opens_mask=args.delim2_opens_mask,
            lag_lookbacks=parse_lag_lookbacks(args.lag_lookbacks),
            lag_controller_temperature=args.lag_controller_temperature,
            source_controller_temperature=args.source_controller_temperature,
            opener_controller_temperature=args.opener_controller_temperature,
            residual_controller_temperature=args.residual_controller_temperature,
            source_topk=args.source_topk,
            candidate_floor=args.candidate_floor,
            global_lag_cap=args.global_lag_cap,
        ),
        args.scale,
    )
    model = AsuraModel(vocab_size=dataset.vocab_size, config=config)

    print("\n  asura golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} freeze_base={config.freeze_base} "
        f"lag_lookbacks={config.lag_lookbacks} lag_temp={config.lag_controller_temperature:.2f} "
        f"global_lag_cap={config.global_lag_cap:.2f} "
        f"source_temp={config.source_controller_temperature:.2f} opener_temp={config.opener_controller_temperature:.2f} "
        f"residual_temp={config.residual_controller_temperature:.2f} candidate_floor={config.candidate_floor:.2f} "
        f"params={count_trainable_params(model):,}"
    )

    metrics = train_model(model, dataset, runtime.train, args.seed, "asura_golf_bridge")
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    probe_tokens, _ = dataset.batch("test", runtime.train.batch_size, runtime.train.seq_len)
    controller_summary = model.controller_snapshot(probe_tokens)

    result = {
        "title": "asura golf bridge cell",
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
            "preset": "asura",
            "variant": args.variant,
            "scale": args.scale,
            "freeze_base": config.freeze_base,
            "params": metrics.params,
            "seed": args.seed,
            "lag_lookbacks": list(config.lag_lookbacks),
            "lag_controller_temperature": config.lag_controller_temperature,
            "source_controller_temperature": config.source_controller_temperature,
            "opener_controller_temperature": config.opener_controller_temperature,
            "residual_controller_temperature": config.residual_controller_temperature,
            "source_topk": config.source_topk,
            "candidate_floor": config.candidate_floor,
            "global_lag_cap": config.global_lag_cap,
            "enable_exact1": config.enable_exact1,
            "enable_exact2": config.enable_exact2,
            "enable_exact3": config.enable_exact3,
            "enable_special2": config.enable_special2,
            "enable_number2": config.enable_number2,
            "enable_urlpath2": config.enable_urlpath2,
            "enable_markup2": config.enable_markup2,
            "enable_attr2": config.enable_attr2,
            "enable_entity2": config.enable_entity2,
            "enable_stack2": config.enable_stack2,
            "enable_wordclass2": config.enable_wordclass2,
            "enable_delim2": config.enable_delim2,
            "enable_delimsub2": config.enable_delimsub2,
            "enable_recency": config.enable_recency,
            "train_eval_loss": train_eval,
            "test_eval_loss": test_eval,
            "train_bits_per_token": train_bpt,
            "test_bits_per_token": test_bpt,
            "test_bits_per_byte": test_bpb,
            "controller_summary": controller_summary,
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

    if args.save_state:
        save_path = Path(args.save_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_state_npz(save_path, flat_full)
        result["model"]["saved_state_path"] = str(save_path)

    print(
        f"  Te:{test_eval:.4f} "
        f"bpt:{test_bpt:.4f} "
        f"bpb:{test_bpb:.4f} "
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

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote JSON summary to {args.json}")
    if args.save_state:
        print(f"  Wrote NPZ state to {args.save_state}")


if __name__ == "__main__":
    main()
