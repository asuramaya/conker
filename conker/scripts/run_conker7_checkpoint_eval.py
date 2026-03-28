#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carving_machine.config import RuntimeConfig
from carving_machine.training import loss_fn
from conker.scripts.run_conker7_golf_bridge import (
    base_config_for_variant,
    build_runtime,
    load_state_npz,
)
from conker.src.conker4b import ConkerFourBConfig
from conker.src.conker7 import ConkerSevenConfig, ConkerSevenModel, scale_config
from conker.src.golf_data import (
    HEADER_BYTES,
    HEADER_INTS,
    PARAMETER_GOLF_MAGIC,
    PARAMETER_GOLF_VERSION,
    _build_sentencepiece_luts,
    build_parameter_golf_dataset,
    spm,
)
from conker.src.quantize import bits_per_token_from_loss
from conker.src.quantize import dequantize_packed_params, pack_trainable_params, serialize_packed_params_zlib


def load_golf_shard(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == PARAMETER_GOLF_MAGIC
            and int(header[1]) == PARAMETER_GOLF_VERSION
        ):
            token_count = int(header[2])
            payload = np.frombuffer(blob[HEADER_BYTES:], dtype=np.uint16, count=token_count)
            if payload.size == token_count:
                return payload.astype(np.int32, copy=False)
    return np.frombuffer(blob, dtype=np.uint16).astype(np.int32, copy=False)


def transform_batch(x: mx.array, y: mx.array, mode: str, rng: np.random.Generator) -> tuple[mx.array, mx.array]:
    if mode == "none":
        return x, y
    x_np = np.array(x, copy=False)
    y_np = np.array(y, copy=False)
    joined = np.concatenate([x_np, y_np[:, -1:]], axis=1)
    if mode == "reverse":
        joined = joined[:, ::-1]
    elif mode == "shuffle":
        shuffled = np.empty_like(joined)
        for row_idx in range(joined.shape[0]):
            perm = rng.permutation(joined.shape[1])
            shuffled[row_idx] = joined[row_idx, perm]
        joined = shuffled
    else:
        raise ValueError(f"Unknown transform: {mode}")
    return (
        mx.array(joined[:, :-1], dtype=mx.int32),
        mx.array(joined[:, 1:], dtype=mx.int32),
    )


def evaluate_with_transform(
    model: ConkerSevenModel,
    compiled_loss,
    dataset,
    runtime: RuntimeConfig,
    split: str,
    transform: str,
    seed: int,
) -> tuple[float, int]:
    total = 0.0
    total_tokens = 0
    rng = np.random.default_rng(seed)
    for _ in range(runtime.train.eval_batches):
        x, y = dataset.batch(split, runtime.train.batch_size, runtime.train.seq_len)
        x, y = transform_batch(x, y, transform, rng)
        loss = compiled_loss(x, y)
        mx.eval(loss)
        total += float(loss.item())
        total_tokens += int(runtime.train.batch_size * runtime.train.seq_len)
    return total / runtime.train.eval_batches, total_tokens


def load_full_split_tokens(dataset, split: str, seq_len: int) -> np.ndarray:
    files = dataset.train_files if split == "train" else dataset.test_files
    tokens = np.ascontiguousarray(np.concatenate([load_golf_shard(path) for path in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"{split} split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def evaluate_full_split_with_transform(
    model: ConkerSevenModel,
    compiled_loss,
    dataset,
    runtime: RuntimeConfig,
    split: str,
    transform: str,
    seed: int,
) -> tuple[float, float | None, int, int]:
    tokens = load_full_split_tokens(dataset, split, runtime.train.seq_len)
    rng = np.random.default_rng(seed)
    total_loss_sum = 0.0
    total_tokens = 0
    total_bytes = 0.0
    batch_size = runtime.train.batch_size
    seq_len = runtime.train.seq_len
    total_seqs = (tokens.size - 1) // seq_len
    base_bytes_lut = None
    has_leading_space_lut = None
    is_boundary_token_lut = None
    if split == "test" and dataset.tokenizer_path is not None and spm is not None:
        sp = spm.SentencePieceProcessor(model_file=str(dataset.tokenizer_path))
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = _build_sentencepiece_luts(
            sp,
            dataset.vocab_size,
        )
    for seq_start in range(0, total_seqs, batch_size):
        seq_end = min(seq_start + batch_size, total_seqs)
        raw_start = seq_start * seq_len
        raw_end = seq_end * seq_len + 1
        chunk = tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        x, y = transform_batch(x, y, transform, rng)
        loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(loss)
        token_count = int(x.shape[0] * x.shape[1])
        total_loss_sum += float(loss.item()) * token_count
        total_tokens += token_count
        if split == "test" and base_bytes_lut is not None:
            y_eval = np.array(y, copy=False)
            row_bytes = base_bytes_lut[y_eval]
            if has_leading_space_lut is not None:
                row_bytes = row_bytes + has_leading_space_lut[y_eval].astype(np.float64)
            if is_boundary_token_lut is not None:
                row_bytes = row_bytes * (~is_boundary_token_lut[y_eval]).astype(np.float64)
            total_bytes += float(row_bytes.sum())
    eval_loss = total_loss_sum / max(total_tokens, 1)
    eval_bpb = None
    if split == "test" and total_bytes > 0.0:
        eval_bpb = bits_per_token_from_loss(eval_loss) * (float(total_tokens) / float(total_bytes))
    return eval_loss, eval_bpb, total_tokens, int(tokens.size - 1)


def build_model_from_args(args: argparse.Namespace, dataset) -> tuple[ConkerSevenModel, RuntimeConfig]:
    runtime = build_runtime(args)
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
    return ConkerSevenModel(vocab_size=dataset.vocab_size, config=config), runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-process evaluation of a saved Conker-7 NPZ checkpoint.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--state-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--artifact-out", default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--transform", choices=["none", "reverse", "shuffle"], default="none")
    parser.add_argument("--transform-seed", type=int, default=42)
    parser.add_argument("--full-split", action="store_true")
    parser.add_argument("--quant-bits", type=int, default=0)
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
    parser.add_argument("--variant", choices=["base", "window4", "window16", "gated", "linear_only"], default="window4")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=args.vocab_size)
    model, runtime = build_model_from_args(args, dataset)
    state = load_state_npz(Path(args.state_npz))

    artifact_bytes_zlib = None
    artifact_bytes_raw = None
    packed_stats = None
    if args.quant_bits > 0:
        trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
        packed_state, packed_stats = pack_trainable_params(state, trainable_names, args.quant_bits)
        blob, raw_bytes = serialize_packed_params_zlib(packed_state)
        artifact_bytes_zlib = len(blob)
        artifact_bytes_raw = raw_bytes
        if args.artifact_out:
            out_path = Path(args.artifact_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            artifact_bytes_zlib = out_path.stat().st_size
        state = dequantize_packed_params(packed_state)

    model.update(nn.utils.tree_unflatten(list(state.items())))
    compiled_loss = mx.compile(lambda x, y: loss_fn(model, x, y), inputs=model.state, outputs=model.state)
    warm_x, warm_y = dataset.batch(args.split, runtime.train.batch_size, runtime.train.seq_len)
    warm_x, warm_y = transform_batch(warm_x, warm_y, args.transform, np.random.default_rng(args.transform_seed))
    warm_loss = compiled_loss(warm_x, warm_y)
    mx.eval(warm_loss)

    if args.full_split:
        eval_loss, eval_bpb, eval_tokens, source_tokens = evaluate_full_split_with_transform(
            model,
            compiled_loss,
            dataset,
            runtime,
            args.split,
            args.transform,
            args.transform_seed,
        )
    else:
        eval_loss, eval_tokens = evaluate_with_transform(
            model,
            compiled_loss,
            dataset,
            runtime,
            args.split,
            args.transform,
            args.transform_seed,
        )
        eval_bpb = None
        source_tokens = eval_tokens
        tokens_per_byte = dataset.test_tokens_per_byte if args.split == "test" else None
        if tokens_per_byte is not None:
            eval_bpb = bits_per_token_from_loss(eval_loss) * tokens_per_byte
    out = {
        "state_npz": str(Path(args.state_npz)),
        "split": args.split,
        "transform": args.transform,
        "transform_seed": args.transform_seed,
        "full_split": args.full_split,
        "quant_bits": args.quant_bits,
        "eval_loss": eval_loss,
        "eval_bits_per_token": bits_per_token_from_loss(eval_loss),
        "eval_bpb": eval_bpb,
        "eval_tokens": eval_tokens,
        "source_tokens": source_tokens,
    }
    if packed_stats is not None:
        out["packed_stats"] = packed_stats
        out["artifact_bytes_zlib"] = artifact_bytes_zlib
        out["artifact_bytes_raw"] = artifact_bytes_raw
        if args.artifact_out:
            out["artifact_out"] = str(Path(args.artifact_out))
    Path(args.output_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
