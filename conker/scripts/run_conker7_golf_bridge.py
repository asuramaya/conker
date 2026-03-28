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
from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker4b import ConkerFourBConfig
from conker.src.conker7 import ConkerSevenConfig, ConkerSevenModel, scale_config
from conker.src.golf_data import build_parameter_golf_dataset
from conker.src.quantize import bits_per_token_from_loss, estimate_trainable_payload_bytes, quantize_trainable_params


def save_state_npz(path: Path, state: dict[str, mx.array]) -> None:
    arrays = {name: np.array(value) for name, value in state.items()}
    np.savez(path, **arrays)


def load_state_npz(path: Path) -> dict[str, mx.array]:
    with np.load(path, allow_pickle=False) as data:
        return {name: mx.array(data[name]) for name in data.files}


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
        raise ValueError(f"Unknown Conker-7 base variant: {args.variant}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker-7 bridge: legal Conker-4b student with future-aware teacher.")
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
    parser.add_argument("--exact-context-span", type=int, default=0)
    parser.add_argument("--residual-cap", type=float, default=4.0)
    parser.add_argument("--base-feature-scale", type=float, default=1.0)
    parser.set_defaults(dynamic_support_gates=True)
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
    parser.add_argument(
        "--variant",
        choices=["base", "window4", "window16", "gated", "linear_only"],
        default="window4",
    )
    parser.set_defaults(freeze_base=False)
    parser.add_argument("--freeze-base", action="store_true", dest="freeze_base")
    parser.add_argument("--no-freeze-base", action="store_false", dest="freeze_base")
    parser.add_argument("--teacher-mask-mode", choices=["future", "bidirectional"], default="future")
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--teacher-start-step", type=int, default=0)
    parser.set_defaults(teacher_enable_exact2=True)
    parser.add_argument("--teacher-enable-exact2", action="store_true", dest="teacher_enable_exact2")
    parser.add_argument("--teacher-disable-exact2", action="store_false", dest="teacher_enable_exact2")
    parser.set_defaults(teacher_enable_exact3=True)
    parser.add_argument("--teacher-enable-exact3", action="store_true", dest="teacher_enable_exact3")
    parser.add_argument("--teacher-disable-exact3", action="store_false", dest="teacher_enable_exact3")
    parser.set_defaults(teacher_enable_special2=False)
    parser.add_argument("--teacher-enable-special2", action="store_true", dest="teacher_enable_special2")
    parser.add_argument("--teacher-disable-special2", action="store_false", dest="teacher_enable_special2")
    parser.set_defaults(teacher_enable_number2=False)
    parser.add_argument("--teacher-enable-number2", action="store_true", dest="teacher_enable_number2")
    parser.add_argument("--teacher-disable-number2", action="store_false", dest="teacher_enable_number2")
    parser.set_defaults(teacher_enable_markup2=False)
    parser.add_argument("--teacher-enable-markup2", action="store_true", dest="teacher_enable_markup2")
    parser.add_argument("--teacher-disable-markup2", action="store_false", dest="teacher_enable_markup2")
    parser.set_defaults(teacher_enable_attr2=False)
    parser.add_argument("--teacher-enable-attr2", action="store_true", dest="teacher_enable_attr2")
    parser.add_argument("--teacher-disable-attr2", action="store_false", dest="teacher_enable_attr2")
    parser.set_defaults(teacher_enable_delim2=False)
    parser.add_argument("--teacher-enable-delim2", action="store_true", dest="teacher_enable_delim2")
    parser.add_argument("--teacher-disable-delim2", action="store_false", dest="teacher_enable_delim2")
    parser.add_argument("--teacher-exact2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-exact3-weight", type=float, default=2.0)
    parser.add_argument("--teacher-special2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-number2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-markup2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-attr2-weight", type=float, default=1.0)
    parser.add_argument("--teacher-delim2-weight", type=float, default=0.5)
    parser.add_argument("--load-state", default=None)
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
    student_cfg = ConkerFourBConfig(
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
        exact_context_span=args.exact_context_span,
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
    )
    config = scale_config(
        ConkerSevenConfig(
            student_config=student_cfg,
            teacher_mask_mode=args.teacher_mask_mode,
            teacher_weight=args.teacher_weight,
            teacher_start_step=args.teacher_start_step,
            teacher_enable_exact2=args.teacher_enable_exact2,
            teacher_enable_exact3=args.teacher_enable_exact3,
            teacher_enable_special2=args.teacher_enable_special2,
            teacher_enable_number2=args.teacher_enable_number2,
            teacher_enable_markup2=args.teacher_enable_markup2,
            teacher_enable_attr2=args.teacher_enable_attr2,
            teacher_enable_delim2=args.teacher_enable_delim2,
            teacher_exact2_weight=args.teacher_exact2_weight,
            teacher_exact3_weight=args.teacher_exact3_weight,
            teacher_special2_weight=args.teacher_special2_weight,
            teacher_number2_weight=args.teacher_number2_weight,
            teacher_markup2_weight=args.teacher_markup2_weight,
            teacher_attr2_weight=args.teacher_attr2_weight,
            teacher_delim2_weight=args.teacher_delim2_weight,
        ),
        args.scale,
    )
    model = ConkerSevenModel(vocab_size=dataset.vocab_size, config=config)
    if args.load_state:
        loaded = load_state_npz(Path(args.load_state))
        remapped: dict[str, mx.array] = {}
        for name, value in loaded.items():
            remapped[f"student.{name}"] = value if not name.startswith("student.") else value
        model.update(nn.utils.tree_unflatten(list(remapped.items())))

    print("\n  conker-7 golf bridge\n")
    print(
        f"  data_root={args.data_root} seed={args.seed} steps={runtime.train.steps} "
        f"seq_len={runtime.train.seq_len} batch_size={runtime.train.batch_size} "
        f"lr={runtime.train.learning_rate:g}"
    )
    print(
        f"  variant={args.variant} scale={args.scale:.3f} linear_modes={config.student_config.base_config.linear_modes} "
        f"local_window={config.student_config.base_config.local_window} half_life_max={config.student_config.base_config.linear_half_life_max:.1f} "
        f"osc_frac={config.student_config.base_config.oscillatory_frac:.2f} input_proj={config.student_config.base_config.input_proj_scheme} "
        f"static_bank_gate={config.student_config.base_config.static_bank_gate} teacher={config.teacher_mask_mode} "
        f"teacher_weight={config.teacher_weight:.3f} teacher_start={config.teacher_start_step} "
        f"warmstart={args.load_state is not None} params={count_trainable_params(model):,}"
    )

    def _on_step(step: int, current_model: nn.Module, losses: list[float]) -> None:
        if hasattr(current_model, "set_train_step"):
            current_model.set_train_step(step)

    metrics = train_model(model, dataset, runtime.train, args.seed, "conker7_golf_bridge", on_step=_on_step)
    train_eval = evaluate(model, dataset, runtime.train, "train")
    test_eval = evaluate(model, dataset, runtime.train, "test")
    train_bpt = bits_per_token_from_loss(train_eval)
    test_bpt = bits_per_token_from_loss(test_eval)
    test_bpb = test_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None

    result = {
        "title": "conker-7 golf bridge cell",
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
            "preset": "conker7",
            "variant": args.variant,
            "scale": args.scale,
            "params": metrics.params,
            "seed": args.seed,
            "train_eval_loss": float(train_eval),
            "test_eval_loss": float(test_eval),
            "train_bits_per_token": float(train_bpt),
            "test_bits_per_token": float(test_bpt),
            "test_bpb": None if test_bpb is None else float(test_bpb),
            "train_time_sec": float(metrics.train_time_sec),
            "learning_rate": runtime.train.learning_rate,
            "teacher_mask_mode": config.teacher_mask_mode,
            "teacher_weight": config.teacher_weight,
            "teacher_start_step": config.teacher_start_step,
            "teacher_enable_exact2": config.teacher_enable_exact2,
            "teacher_enable_exact3": config.teacher_enable_exact3,
            "teacher_enable_special2": config.teacher_enable_special2,
            "teacher_enable_number2": config.teacher_enable_number2,
            "teacher_enable_markup2": config.teacher_enable_markup2,
            "teacher_enable_attr2": config.teacher_enable_attr2,
            "teacher_enable_delim2": config.teacher_enable_delim2,
            "loaded_state_path": args.load_state,
            "payload_bytes_est": None,
            "payload_mb_est": None,
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
        quant_eval = evaluate(model, dataset, runtime.train, "test")
        quant_bpt = bits_per_token_from_loss(quant_eval)
        quant_bpb = quant_bpt * dataset.test_tokens_per_byte if dataset.test_tokens_per_byte is not None else None
        result["quantization"].append(
            {
                "bits": bits,
                "test_eval_loss": float(quant_eval),
                "test_bits_per_token": float(quant_bpt),
                "test_bpb": None if quant_bpb is None else float(quant_bpb),
                **quant_stats,
            }
        )
    if args.quant_bits:
        model.update(nn.utils.tree_unflatten(list(flat_full.items())))

    if args.save_state:
        save_path = Path(args.save_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_state_npz(save_path, flat_full)
        result["model"]["saved_state_path"] = str(save_path)

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out_path}\n")
    if args.save_state:
        print(f"  wrote {args.save_state}")


if __name__ == "__main__":
    main()
