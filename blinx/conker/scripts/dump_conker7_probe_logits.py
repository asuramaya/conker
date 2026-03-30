#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from conker.scripts.run_conker7_checkpoint_eval import build_model_from_args, build_parser, load_state_npz
from conker.src.golf_data import build_parameter_golf_dataset


def parse_tokens(raw: str) -> list[int]:
    tokens = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    if not tokens:
        raise ValueError("token list was empty")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump full Conker-7 logits for a fixed token sequence.")
    parser.add_argument("--state-npz", required=True)
    parser.add_argument("--tokens", required=True, help="Comma-separated token ids")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--data-root", default="conker/data/datasets/fineweb10B_sp1024")
    args = parser.parse_args()

    tokens = parse_tokens(args.tokens)

    ckpt_parser = build_parser()
    ckpt_args = ckpt_parser.parse_args(
        [
            "--data-root",
            args.data_root,
            "--state-npz",
            args.state_npz,
            "--output-json",
            "/tmp/ignore_conker7_probe.json",
            "--split",
            "test",
            "--transform",
            "none",
            "--seed",
            "42",
            "--steps",
            "1000",
            "--seq-len",
            "256",
            "--batch-size",
            "16",
            "--profile",
            "pilot",
            "--vocab-size",
            "1024",
            "--variant",
            "window4",
            "--scale",
            "10.0",
            "--learning-rate",
            "5e-4",
            "--linear-half-life-max",
            "16",
            "--oscillatory-frac",
            "0.875",
            "--oscillatory-period-min",
            "4",
            "--oscillatory-period-max",
            "64",
            "--static-bank-gate",
            "--dynamic-support-gates",
            "--gate-only-mode",
            "--enable-exact3",
            "--enable-delim2",
            "--enable-special2",
            "--enable-number2",
            "--enable-markup2",
            "--enable-attr2",
            "--disable-recency",
            "--no-exact1-opens-mask",
            "--no-delim2-opens-mask",
            "--teacher-mask-mode",
            "bidirectional",
            "--teacher-weight",
            "0.10",
            "--teacher-enable-exact2",
            "--teacher-enable-exact3",
            "--teacher-disable-special2",
            "--teacher-disable-number2",
            "--teacher-disable-markup2",
            "--teacher-disable-attr2",
            "--teacher-disable-delim2",
        ]
    )

    dataset = build_parameter_golf_dataset(ckpt_args.data_root, vocab_size=ckpt_args.vocab_size)
    model, _runtime = build_model_from_args(ckpt_args, dataset)
    state = load_state_npz(Path(args.state_npz))
    model.update(nn.utils.tree_unflatten(list(state.items())))

    x = mx.array(np.array(tokens, dtype=np.int32)[None, :], dtype=mx.int32)
    base_logits = model.student.base(x)
    logits, support_activations = model.student._forward_impl(x, return_support_activations=True)
    (
        exact1,
        exact2,
        exact3,
        special2,
        number2,
        _urlpath2,
        markup2,
        attr2,
        _entity2,
        _stack2,
        _wordclass2,
        delim2,
        _delimsub2,
        recency,
    ) = model.student._count_features(x)
    mx.eval(
        base_logits,
        logits,
        exact1,
        exact2,
        exact3,
        special2,
        number2,
        markup2,
        attr2,
        delim2,
        recency,
        *support_activations.values(),
    )
    base_logits_np = np.array(base_logits, copy=False)[0]
    logits_np = np.array(logits, copy=False)[0]

    out = {
        "state_npz": args.state_npz,
        "tokens": tokens,
        "base_logits": base_logits_np.tolist(),
        "logits": logits_np.tolist(),
        "support_activations": {
            name: np.array(value, copy=False)[0].tolist()
            for name, value in support_activations.items()
        },
        "sources": {
            "exact1": None if exact1 is None else np.array(exact1, copy=False)[0].tolist(),
            "exact2": None if exact2 is None else np.array(exact2, copy=False)[0].tolist(),
            "exact3": None if exact3 is None else np.array(exact3, copy=False)[0].tolist(),
            "special2": None if special2 is None else np.array(special2, copy=False)[0].tolist(),
            "number2": None if number2 is None else np.array(number2, copy=False)[0].tolist(),
            "markup2": None if markup2 is None else np.array(markup2, copy=False)[0].tolist(),
            "attr2": None if attr2 is None else np.array(attr2, copy=False)[0].tolist(),
            "delim2": None if delim2 is None else np.array(delim2, copy=False)[0].tolist(),
            "recency": None if recency is None else np.array(recency, copy=False)[0].tolist(),
        },
    }
    Path(args.output_json).write_text(json.dumps(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
