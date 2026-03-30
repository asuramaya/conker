#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from conker.src.golf_data import build_parameter_golf_dataset


def estimate_autocorr(tokens: np.ndarray, vocab_size: int, projection_dim: int, max_lag: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(vocab_size, projection_dim))
    signal = proj[tokens]
    signal -= signal.mean(axis=0, keepdims=True)
    n = signal.shape[0]
    fft = np.fft.rfft(signal, axis=0)
    power = (fft * np.conjugate(fft)).real.mean(axis=1)
    autocorr = np.fft.irfft(power, n=n).real / max(n, 1)
    autocorr = autocorr[: max_lag + 1]
    if autocorr[0] != 0:
        autocorr = autocorr / float(autocorr[0])
    return autocorr.astype(np.float32, copy=False)


def matched_half_lives(
    autocorr: np.ndarray,
    modes: int,
    min_half_life: float,
    max_half_life: float,
) -> np.ndarray:
    lags = np.arange(1, autocorr.shape[0], dtype=np.float32)
    weights = np.abs(autocorr[1:]).astype(np.float64, copy=False)
    weights = np.maximum(weights, 1e-12)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    quantiles = (np.arange(modes, dtype=np.float64) + 0.5) / float(modes)
    half_lives = np.interp(quantiles, cdf, lags).astype(np.float32, copy=False)
    half_lives = np.clip(half_lives, min_half_life, max_half_life)
    return np.sort(half_lives.astype(np.float32, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an autocorrelation-matched decay bank for Conker-3.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-tokens", type=int, default=524288)
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--max-lag", type=int, default=512)
    parser.add_argument("--modes", type=int, required=True)
    parser.add_argument("--min-half-life", type=float, default=1.5)
    parser.add_argument("--max-half-life", type=float, default=512.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    dataset = build_parameter_golf_dataset(args.data_root, vocab_size=1024)
    dataset.train_stream.reset()
    tokens = dataset.train_stream.take(args.sample_tokens).astype(np.int32, copy=False)
    autocorr = estimate_autocorr(tokens, dataset.vocab_size, args.projection_dim, args.max_lag, args.seed)
    half_lives = matched_half_lives(
        autocorr,
        modes=args.modes,
        min_half_life=args.min_half_life,
        max_half_life=args.max_half_life,
    )
    decays = np.exp(np.log(0.5, dtype=np.float32) / half_lives)

    payload = {
        "title": "conker-3 autocorrelation-matched decay bank",
        "data_root": str(args.data_root),
        "sample_tokens": int(tokens.size),
        "projection_dim": int(args.projection_dim),
        "max_lag": int(args.max_lag),
        "modes": int(args.modes),
        "seed": int(args.seed),
        "half_lives": half_lives.tolist(),
        "decays": decays.astype(np.float32).tolist(),
        "autocorr": autocorr.tolist(),
    }

    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"built matched bank modes={args.modes} sample_tokens={tokens.size} "
        f"half_life_range=({float(half_lives[0]):.3f}, {float(half_lives[-1]):.3f})"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
