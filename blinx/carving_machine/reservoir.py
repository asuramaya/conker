from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .config import ReservoirConfig, ReservoirTopology


@dataclass(frozen=True)
class ReservoirBundle:
    Wr: mx.array
    Wi: mx.array


def _scale_spectral_radius(W: np.ndarray, spectral_radius_target: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if not np.any(W):
        return W
    v = rng.standard_normal(W.shape[0], dtype=np.float32)
    for _ in range(200):
        Wv = W @ v
        norm = np.linalg.norm(Wv)
        if norm > 0:
            v = Wv / norm

    spectral_radius = np.linalg.norm(W @ v)
    if spectral_radius > 0:
        W = W * (spectral_radius_target / spectral_radius)
    return W


def _small_world_degree(size: int, connectivity: float) -> int:
    if size <= 2:
        return max(size - 1, 0)
    degree = int(round(connectivity * (size - 1)))
    degree = max(2, min(degree, size - 1))
    if degree % 2 == 1:
        degree = degree - 1 if degree > 2 else degree + 1
    if degree >= size:
        degree = size - 1 if (size - 1) % 2 == 0 else size - 2
    return max(degree, 2)


def _small_world_mask(size: int, connectivity: float, rewire_prob: float, seed: int) -> np.ndarray:
    degree = _small_world_degree(size, connectivity)
    if degree <= 0:
        return np.zeros((size, size), dtype=np.float32)

    rng = np.random.default_rng(seed)
    adjacency = np.zeros((size, size), dtype=np.bool_)
    half_degree = degree // 2

    for node in range(size):
        for offset in range(1, half_degree + 1):
            neighbor = (node + offset) % size
            adjacency[node, neighbor] = True
            adjacency[neighbor, node] = True

    for node in range(size):
        for offset in range(1, half_degree + 1):
            neighbor = (node + offset) % size
            if not adjacency[node, neighbor] or rng.random() >= rewire_prob:
                continue

            adjacency[node, neighbor] = False
            adjacency[neighbor, node] = False

            candidates = np.flatnonzero(~adjacency[node])
            candidates = candidates[candidates != node]
            if candidates.size == 0:
                adjacency[node, neighbor] = True
                adjacency[neighbor, node] = True
                continue

            new_neighbor = int(rng.choice(candidates))
            adjacency[node, new_neighbor] = True
            adjacency[new_neighbor, node] = True

    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float32)


def _dense_matrix(
    size: int,
    connectivity: float,
    spectral_radius_target: float,
    seed: int,
    topology: ReservoirTopology = "erdos_renyi",
    rewire_prob: float = 0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((size, size), dtype=np.float32)
    if topology == "erdos_renyi":
        mask = (rng.random((size, size), dtype=np.float32) < connectivity).astype(np.float32)
    elif topology == "small_world":
        mask = _small_world_mask(size, connectivity, rewire_prob, seed)
    else:
        raise ValueError(f"Unknown reservoir topology: {topology}")
    W = W * mask
    return _scale_spectral_radius(W, spectral_radius_target, seed + 97)


def build_dense_matrix(
    size: int,
    connectivity: float,
    spectral_radius_target: float,
    seed: int,
    topology: ReservoirTopology = "erdos_renyi",
    rewire_prob: float = 0.1,
) -> mx.array:
    return mx.array(
        _dense_matrix(
            size,
            connectivity,
            spectral_radius_target,
            seed,
            topology=topology,
            rewire_prob=rewire_prob,
        )
    )


def build_dense_reservoir(config: ReservoirConfig) -> ReservoirBundle:
    rng = np.random.default_rng(config.seed)
    W = _dense_matrix(
        size=config.size,
        connectivity=config.connectivity,
        spectral_radius_target=config.spectral_radius,
        seed=config.seed,
        topology=config.topology,
        rewire_prob=config.rewire_prob,
    )

    Wi = rng.standard_normal((config.size, config.embedding_dim), dtype=np.float32) * 0.05
    Wr = mx.array(W)
    W_in = mx.array(Wi)
    mx.eval(Wr, W_in)
    return ReservoirBundle(Wr=Wr, Wi=W_in)
