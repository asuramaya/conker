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

from carving_machine.config import RuntimeConfig, train_config_for_profile
from carving_machine.training import loss_fn
from conker.src.conker3 import ConkerThreeConfig
from conker.src.conker4b import ConkerFourBConfig, ConkerFourBModel, scale_config
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


def load_state_npz(path: Path) -> dict[str, mx.array]:
    with np.load(path, allow_pickle=False) as data:
        return {name: mx.array(data[name]) for name in data.files}


def runtime_from_summary(summary: dict) -> RuntimeConfig:
    profile = summary["config"]["profile"]
    runtime = RuntimeConfig(profile=profile)
    base_train = train_config_for_profile(profile)
    train_cfg = summary["config"]["train"]
    return RuntimeConfig(
        profile=profile,
        train=base_train.__class__(
            batch_size=train_cfg["batch_size"],
            seq_len=train_cfg["seq_len"],
            steps=train_cfg["steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            grad_clip=train_cfg["grad_clip"],
            log_every=train_cfg["log_every"],
            eval_batches=train_cfg["eval_batches"],
            seeds=tuple(train_cfg["seeds"]),
        ),
    )


def model_from_summary(summary: dict, state: dict[str, mx.array]) -> ConkerFourBModel:
    model_cfg = summary["model"]
    embedding_dim = int(state["base.linear_embedding.weight"].shape[1])
    linear_modes = int(state["base.linear_decays"].shape[0])
    linear_hidden = (int(state["base.linear_readout.layers.0.bias"].shape[0]),)
    local_hidden = (int(state["base.local_readout.layers.0.bias"].shape[0]),)
    local_window = int(state["base.local_readout.layers.0.weight"].shape[1] // max(embedding_dim, 1))
    base_cfg = ConkerThreeConfig(
        embedding_dim=embedding_dim,
        linear_modes=linear_modes,
        max_seq_len=summary["config"]["train"]["seq_len"],
        linear_half_life_min=1.5,
        linear_half_life_max=model_cfg["linear_half_life_max"],
        linear_hidden=linear_hidden,
        local_window=local_window,
        local_hidden=local_hidden,
        local_scale=0.25,
        mix_mode="additive" if model_cfg["variant"] != "gated" else "gated",
        share_embedding=False,
        linear_impl="kernel",
        enable_linear=model_cfg["variant"] != "linear_only",
        enable_local=model_cfg["variant"] != "linear_only",
        oscillatory_frac=model_cfg["oscillatory_frac"],
        oscillatory_period_min=model_cfg["oscillatory_period_min"],
        oscillatory_period_max=model_cfg["oscillatory_period_max"],
        static_bank_gate=model_cfg["static_bank_gate"],
        bank_gate_span=0.5,
        input_proj_scheme=model_cfg["input_proj_scheme"],
    )
    cfg = ConkerFourBConfig(
        base_config=base_cfg,
        freeze_base=model_cfg["freeze_base"],
        enable_exact1=model_cfg["enable_exact1"],
        enable_exact2=model_cfg["enable_exact2"],
        enable_exact3=model_cfg["enable_exact3"],
        enable_special2=model_cfg["enable_special2"],
        enable_number2=model_cfg["enable_number2"],
        enable_urlpath2=model_cfg["enable_urlpath2"],
        enable_markup2=model_cfg["enable_markup2"],
        enable_attr2=model_cfg["enable_attr2"],
        enable_entity2=model_cfg["enable_entity2"],
        enable_stack2=model_cfg["enable_stack2"],
        enable_wordclass2=model_cfg["enable_wordclass2"],
        enable_delim2=model_cfg["enable_delim2"],
        enable_delimsub2=model_cfg["enable_delimsub2"],
        enable_recency=model_cfg["enable_recency"],
        tokenizer_vocab_path=model_cfg["tokenizer_vocab_path"],
        recency_half_life=model_cfg["recency_half_life"],
        exact_context_span=model_cfg.get("exact_context_span", 0),
        residual_cap=model_cfg["residual_cap"],
        base_feature_scale=model_cfg["base_feature_scale"],
        dynamic_support_gates=model_cfg["dynamic_support_gates"],
        dynamic_gate_span=model_cfg["dynamic_gate_span"],
        gate_only_mode=model_cfg["gate_only_mode"],
        support_gate_mode=model_cfg["support_gate_mode"],
        support_gate_topk=model_cfg["support_gate_topk"],
        support_gate_temperature=model_cfg["support_gate_temperature"],
        support_overlap_penalty=model_cfg["support_overlap_penalty"],
        exact1_opens_mask=model_cfg["exact1_opens_mask"],
        special2_opens_mask=model_cfg["special2_opens_mask"],
        number2_opens_mask=model_cfg["number2_opens_mask"],
        urlpath2_opens_mask=model_cfg["urlpath2_opens_mask"],
        markup2_opens_mask=model_cfg["markup2_opens_mask"],
        attr2_opens_mask=model_cfg["attr2_opens_mask"],
        entity2_opens_mask=model_cfg["entity2_opens_mask"],
        stack2_opens_mask=model_cfg["stack2_opens_mask"],
        wordclass2_opens_mask=model_cfg["wordclass2_opens_mask"],
        delim2_opens_mask=model_cfg["delim2_opens_mask"],
        delimsub2_opens_mask=model_cfg["delimsub2_opens_mask"],
    )
    model = ConkerFourBModel(vocab_size=summary["dataset"]["train_token_count"] * 0 + 1024, config=cfg)
    return model


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
    model: ConkerFourBModel,
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
    model: ConkerFourBModel,
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
        x_eval = np.array(x, copy=False)
        y_eval = np.array(y, copy=False)
        token_count = int(y_eval.size)
        total_loss_sum += float(loss.item()) * token_count
        total_tokens += token_count
        if split == "test" and base_bytes_lut is not None:
            prev_ids = x_eval.reshape(-1)
            tgt_ids = y_eval.reshape(-1)
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            total_bytes += float(bytes_np.astype(np.float64).sum())
    eval_loss = total_loss_sum / max(total_tokens, 1)
    eval_bpb = None
    if split == "test" and total_bytes > 0.0:
        eval_bpb = (eval_loss / np.log(2.0)) * (total_tokens / total_bytes)
    return eval_loss, eval_bpb, total_tokens, int(tokens.size - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh-process evaluation of a saved Conker-4b NPZ checkpoint.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--state-npz", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--transform", choices=["none", "reverse", "shuffle"], default="none")
    parser.add_argument("--transform-seed", type=int, default=42)
    parser.add_argument("--full-split", action="store_true")
    parser.add_argument("--quant-bits", type=int, default=0)
    parser.add_argument("--artifact-out", default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    runtime = runtime_from_summary(summary)
    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    state = load_state_npz(Path(args.state_npz))
    model = model_from_summary(summary, state)
    artifact_bytes_zlib = None
    artifact_bytes_raw = None
    packed_stats = None
    if args.quant_bits > 0:
        trainable_names = {name for name, _ in nn.utils.tree_flatten(model.trainable_parameters())}
        packed_state, packed_stats = pack_trainable_params(state, trainable_names, args.quant_bits)
        if args.artifact_out:
            blob, raw_bytes = serialize_packed_params_zlib(packed_state)
            out_path = Path(args.artifact_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            artifact_bytes_zlib = out_path.stat().st_size
            artifact_bytes_raw = raw_bytes
        else:
            blob, raw_bytes = serialize_packed_params_zlib(packed_state)
            artifact_bytes_zlib = len(blob)
            artifact_bytes_raw = raw_bytes
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
    eval_bpt = bits_per_token_from_loss(eval_loss)

    out = {
        "summary_json": str(Path(args.summary_json)),
        "state_npz": str(Path(args.state_npz)),
        "split": args.split,
        "transform": args.transform,
        "transform_seed": args.transform_seed,
        "full_split": args.full_split,
        "quant_bits": args.quant_bits,
        "eval_loss": eval_loss,
        "eval_bits_per_token": eval_bpt,
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
