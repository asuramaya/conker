#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64, copy=False)
    if path.suffix == ".csv":
        with path.open() as f:
            rows = [list(map(float, row)) for row in csv.reader(f) if row]
        return np.array(rows, dtype=np.float64)
    raise ValueError(f"Unsupported matrix format: {path.suffix}")


def spectral_stats(matrix: np.ndarray, topk: int) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular * singular
    total_energy = float(np.sum(energy))
    top_energy = float(np.sum(energy[: min(topk, singular.size)]))
    sigma1 = float(singular[0])
    sigmak = float(singular[min(topk - 1, singular.size - 1)])
    sigmalast = float(singular[-1])
    return {
        "sigma1": sigma1,
        f"sigma{topk}": sigmak,
        "sigma_last": sigmalast,
        f"top{topk}_energy_frac": float(top_energy / total_energy) if total_energy > 0 else 0.0,
        f"decay_1_to_{topk}": float(sigma1 / max(sigmak, 1e-12)),
        "decay_1_to_last": float(sigma1 / max(sigmalast, 1e-12)),
    }


def region_stats(matrix: np.ndarray) -> dict[str, float]:
    upper = np.triu(matrix, 1)
    diag = np.diag(np.diag(matrix))
    lower = np.tril(matrix, -1)
    total = float(np.linalg.norm(matrix))
    upper_l2 = float(np.linalg.norm(upper))
    diag_l2 = float(np.linalg.norm(diag))
    lower_l2 = float(np.linalg.norm(lower))
    return {
        "upper_l2": upper_l2,
        "diag_l2": diag_l2,
        "lower_l2": lower_l2,
        "upper_frac": float(upper_l2 / total) if total > 0 else 0.0,
        "diag_frac": float(diag_l2 / total) if total > 0 else 0.0,
        "upper_plus_diag_frac": float(np.linalg.norm(upper + diag) / total) if total > 0 else 0.0,
    }


def compare_stats(matrix: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    if matrix.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {matrix.shape} vs {reference.shape}")
    flat = matrix.reshape(-1)
    ref = reference.reshape(-1)
    denom = float(np.linalg.norm(flat) * np.linalg.norm(ref))
    cosine = None if denom == 0.0 else float(np.dot(flat, ref) / denom)
    diff = flat - ref
    return {
        "cosine_to_reference": cosine,
        "l2_deviation": float(np.sqrt(np.mean(diff * diff))),
        "l1_deviation": float(np.mean(np.abs(diff))),
        "max_abs_deviation": float(np.max(np.abs(diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conker side-channel detector for saved matrices.")
    parser.add_argument("matrix", help="Path to .npy or .csv matrix")
    parser.add_argument("--reference", help="Optional clean/reference matrix for comparison")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--json", help="Optional output JSON path")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = load_matrix(matrix_path)
    result = {
        "matrix": str(matrix_path),
        "shape": list(matrix.shape),
        "spectral": spectral_stats(matrix, args.topk),
        "regions": region_stats(matrix),
    }
    if args.reference:
        ref_path = Path(args.reference)
        reference = load_matrix(ref_path)
        result["reference"] = str(ref_path)
        result["compare"] = compare_stats(matrix, reference)

    out = json.dumps(result, indent=2)
    print(out)
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
