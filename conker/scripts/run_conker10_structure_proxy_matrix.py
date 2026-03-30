#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


CASES = (
    ("baseline", ()),
    ("entropy", ("--structure-proxy-entropy",)),
    ("peak", ("--structure-proxy-peak",)),
    ("candidate4", ("--structure-proxy-candidate4",)),
    ("agreement", ("--structure-proxy-agreement",)),
    ("agreement_mass", ("--structure-proxy-agreement-mass",)),
    ("entropy_peak", ("--structure-proxy-entropy", "--structure-proxy-peak")),
    ("peak_candidate4", ("--structure-proxy-peak", "--structure-proxy-candidate4")),
    ("peak_agreement", ("--structure-proxy-peak", "--structure-proxy-agreement")),
    ("peak_agreement_mass", ("--structure-proxy-peak", "--structure-proxy-agreement-mass")),
    ("candidate4_agreement_mass", ("--structure-proxy-candidate4", "--structure-proxy-agreement-mass")),
    ("all", ("--structure-proxy",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Conker-10 structure-proxy ablation matrix.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--packed-tokens", type=int, default=8000)
    parser.add_argument("--trigram-buckets", type=int, default=256)
    return parser.parse_args()


def _resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _run_case(args: argparse.Namespace, name: str, feature_flags: tuple[str, ...]) -> dict:
    json_base = Path(args.json_out)
    out_json = json_base.with_name(f"{json_base.stem}_{name}.case.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pieces = [
        "python3",
        str(REPO_ROOT / "conker" / "scripts" / "run_conker10_golf_bridge.py"),
        "--data-root",
        str(args.data_root),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--profile",
        "pilot",
        "--variant",
        "window4",
        "--scale",
        str(args.scale),
        "--learning-rate",
        str(args.learning_rate),
        "--packed-tokens",
        str(args.packed_tokens),
        "--trigram-buckets",
        str(args.trigram_buckets),
        "--alpha-bigram",
        "4.0",
        "--alpha-trigram",
        "2.0",
        "--json",
        str(out_json),
        *feature_flags,
    ]
    subprocess.run(
        pieces,
        cwd=REPO_ROOT,
        env=dict(os.environ, MLX_DISABLE_METAL="1"),
        check=True,
    )
    data = json.loads(out_json.read_text())
    return {
        "case": name,
        "flags": list(feature_flags),
        "test_bits_per_token": data["test"]["bits_per_token"],
        "train_bits_per_token": data["train"]["bits_per_token"],
        "json": str(out_json),
        "model": data["model"],
    }


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd()
    args.data_root = str(_resolve_path(args.data_root, launch_cwd))
    args.json_out = str(_resolve_path(args.json_out, launch_cwd))
    args.md_out = str(_resolve_path(args.md_out, launch_cwd))
    rows = [_run_case(args, name, flags) for name, flags in CASES]
    rows.sort(key=lambda row: row["test_bits_per_token"])
    result = {
        "branch": "conker10-structure-proxy-matrix",
        "data_root": args.data_root,
        "seed": args.seed,
        "rows": rows,
    }
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Conker-10 Structure Proxy Matrix",
        "",
        f"- data_root: `{args.data_root}`",
        f"- seed: `{args.seed}`",
        "",
        "| case | test_bpt | train_bpt | flags |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['test_bits_per_token']:.4f} | {row['train_bits_per_token']:.4f} | "
            f"`{' '.join(row['flags'])}` |"
        )
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
