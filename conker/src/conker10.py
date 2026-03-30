from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config as scale_conker3_config
from conker.src.giddy_up.features import structure_proxy_feature_arrays
from conker.src.golf_data import GolfTokenShardDataset


TRIGRAM_HASH_MUL_A = np.uint64(1_315_423_911)
TRIGRAM_HASH_MUL_B = np.uint64(2_654_435_761)


@dataclass(frozen=True)
class ConkerTenPackedTables:
    unigram_probs: np.ndarray
    bigram_counts: np.ndarray
    bigram_totals: np.ndarray
    trigram_counts: np.ndarray
    trigram_totals: np.ndarray
    trigram_buckets: int
    token_budget: int
    bytes_total: int


@dataclass(frozen=True)
class ConkerTenConfig:
    base_config: ConkerThreeConfig = ConkerThreeConfig(max_seq_len=256, linear_modes=256, local_window=4)
    freeze_base: bool = False
    blend_mode: str = "learned_mixer"
    structure_proxy_entropy: bool = False
    structure_proxy_peak: bool = False
    structure_proxy_candidate4: bool = False
    structure_proxy_agreement: bool = False
    structure_proxy_agreement_mass: bool = False
    fixed_component_weights: tuple[float, float, float, float] = (0.25, 0.10, 0.25, 0.40)
    alpha_bigram: float = 4.0
    alpha_trigram: float = 2.0
    controller_hidden: int = 16
    controller_temperature: float = 1.0


def _hash_trigram_context(prev2: np.ndarray, prev1: np.ndarray, buckets: int) -> np.ndarray:
    hashed = (prev2.astype(np.uint64) * TRIGRAM_HASH_MUL_A + prev1.astype(np.uint64) * TRIGRAM_HASH_MUL_B) % np.uint64(buckets)
    return hashed.astype(np.int64, copy=False)


def build_packed_tables(
    dataset: GolfTokenShardDataset,
    token_budget: int,
    trigram_buckets: int,
) -> ConkerTenPackedTables:
    if trigram_buckets <= 0:
        raise ValueError("Conker-10 requires trigram_buckets > 0.")

    take_n = min(int(token_budget), max(int(dataset.train_token_count) - 1, 2))
    dataset.train_stream.reset()
    tokens = dataset.train_stream.take(take_n)
    dataset.train_stream.reset()
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.size < 4:
        raise ValueError("Conker-10 needs at least 4 training tokens to build packed tables.")

    next_tok = tokens[1:]
    prev1 = tokens[:-1]
    unigram = np.bincount(next_tok, minlength=dataset.vocab_size).astype(np.float32, copy=False)
    unigram_sum = float(np.maximum(unigram.sum(), 1.0))
    unigram_probs = unigram / unigram_sum

    bigram_flat = prev1 * dataset.vocab_size + next_tok
    bigram_counts = np.bincount(
        bigram_flat,
        minlength=dataset.vocab_size * dataset.vocab_size,
    ).astype(np.float32, copy=False).reshape(dataset.vocab_size, dataset.vocab_size)
    bigram_totals = bigram_counts.sum(axis=1, dtype=np.float32)

    prev2 = tokens[:-2]
    prev1_tri = tokens[1:-1]
    next_tri = tokens[2:]
    trigram_bucket_ids = _hash_trigram_context(prev2, prev1_tri, trigram_buckets)
    trigram_flat = trigram_bucket_ids * dataset.vocab_size + next_tri
    trigram_counts = np.bincount(
        trigram_flat,
        minlength=trigram_buckets * dataset.vocab_size,
    ).astype(np.float32, copy=False).reshape(trigram_buckets, dataset.vocab_size)
    trigram_totals = trigram_counts.sum(axis=1, dtype=np.float32)

    bytes_total = int(
        unigram_probs.nbytes
        + bigram_counts.nbytes
        + bigram_totals.nbytes
        + trigram_counts.nbytes
        + trigram_totals.nbytes
    )
    return ConkerTenPackedTables(
        unigram_probs=np.ascontiguousarray(unigram_probs.astype(np.float32, copy=False)),
        bigram_counts=np.ascontiguousarray(bigram_counts.astype(np.float32, copy=False)),
        bigram_totals=np.ascontiguousarray(bigram_totals.astype(np.float32, copy=False)),
        trigram_counts=np.ascontiguousarray(trigram_counts.astype(np.float32, copy=False)),
        trigram_totals=np.ascontiguousarray(trigram_totals.astype(np.float32, copy=False)),
        trigram_buckets=int(trigram_buckets),
        token_budget=int(take_n),
        bytes_total=bytes_total,
    )


class ConkerTenModel(nn.Module):
    """Packed training memory plus a small causal controller and base fallback."""

    def __init__(
        self,
        vocab_size: int,
        tables: ConkerTenPackedTables,
        config: ConkerTenConfig = ConkerTenConfig(),
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        self.tables = tables

        self.base = ConkerThreeModel(vocab_size=vocab_size, config=config.base_config)
        self.packed_unigram_probs = mx.array(tables.unigram_probs, dtype=mx.float32)
        self.packed_bigram_counts = mx.array(tables.bigram_counts, dtype=mx.float32)
        self.packed_bigram_totals = mx.array(tables.bigram_totals, dtype=mx.float32)
        self.packed_trigram_counts = mx.array(tables.trigram_counts, dtype=mx.float32)
        self.packed_trigram_totals = mx.array(tables.trigram_totals, dtype=mx.float32)
        self.controller_hidden = None
        self.controller_out = None
        if config.blend_mode == "learned_mixer":
            controller_features = (
                7
                + int(config.structure_proxy_entropy)
                + int(config.structure_proxy_peak)
                + int(config.structure_proxy_candidate4)
                + int(config.structure_proxy_agreement)
                + int(config.structure_proxy_agreement_mass)
            )
            self.controller_hidden = nn.Linear(controller_features, config.controller_hidden)
            self.controller_out = nn.Linear(config.controller_hidden, 4)
        elif config.blend_mode not in {"fixed_interp", "memory_only"}:
            raise ValueError(f"Unknown Conker-10 blend_mode: {config.blend_mode}")
        self.freeze(
            keys=[
                "packed_unigram_probs",
                "packed_bigram_counts",
                "packed_bigram_totals",
                "packed_trigram_counts",
                "packed_trigram_totals",
            ],
            strict=False,
        )

    def _shift_right(self, chars: mx.array, amount: int) -> mx.array:
        if amount <= 0:
            return chars
        batch, timesteps = chars.shape
        if amount >= timesteps:
            return mx.zeros((batch, timesteps), dtype=mx.int32)
        pad = mx.zeros((batch, amount), dtype=mx.int32)
        return mx.concatenate([pad, chars[:, :-amount]], axis=1)

    def _memory_probs(self, chars: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        batch, timesteps = chars.shape
        prev1 = self._shift_right(chars, 1)
        prev2 = self._shift_right(chars, 2)
        valid1 = (mx.arange(timesteps)[None, :] >= 1).astype(mx.float32)
        valid2 = (mx.arange(timesteps)[None, :] >= 2).astype(mx.float32)

        p_uni = mx.broadcast_to(self.packed_unigram_probs[None, None, :], (batch, timesteps, self.vocab_size))

        bigram_counts = self.packed_bigram_counts[prev1]
        bigram_totals = self.packed_bigram_totals[prev1][..., None]
        alpha1 = mx.array(self.config.alpha_bigram, dtype=mx.float32)
        p_bigram = (bigram_counts + alpha1 * p_uni) / mx.maximum(bigram_totals + alpha1, mx.array(1e-8, dtype=mx.float32))
        p_bigram = valid1[:, :, None] * p_bigram + (1.0 - valid1[:, :, None]) * p_uni

        trigram_hash = (
            prev2.astype(mx.uint32).astype(mx.uint64) * mx.array(int(TRIGRAM_HASH_MUL_A), dtype=mx.uint64)
            + prev1.astype(mx.uint32).astype(mx.uint64) * mx.array(int(TRIGRAM_HASH_MUL_B), dtype=mx.uint64)
        ) % mx.array(self.tables.trigram_buckets, dtype=mx.uint64)
        trigram_hash = trigram_hash.astype(mx.int32)
        trigram_counts = self.packed_trigram_counts[trigram_hash]
        trigram_totals = self.packed_trigram_totals[trigram_hash][..., None]
        alpha2 = mx.array(self.config.alpha_trigram, dtype=mx.float32)
        p_trigram = (trigram_counts + alpha2 * p_bigram) / mx.maximum(
            trigram_totals + alpha2, mx.array(1e-8, dtype=mx.float32)
        )
        p_trigram = valid2[:, :, None] * p_trigram + (1.0 - valid2[:, :, None]) * p_bigram
        return p_uni, p_bigram, p_trigram, bigram_totals[..., 0], trigram_totals[..., 0]

    @staticmethod
    def _base_features(base_logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        log_probs = base_logits - mx.logsumexp(base_logits, axis=-1, keepdims=True)
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        max_logit = mx.max(base_logits, axis=-1)
        centered = base_logits - mx.mean(base_logits, axis=-1, keepdims=True)
        variance = mx.mean(centered * centered, axis=-1)
        return entropy, max_logit, variance

    def _controller_weights(
        self,
        base_logits: mx.array,
        base_probs: mx.array,
        p_bigram: mx.array,
        p_trigram: mx.array,
        bigram_totals: mx.array,
        trigram_totals: mx.array,
    ) -> mx.array:
        batch, timesteps, _ = base_logits.shape
        entropy, max_logit, variance = self._base_features(base_logits)
        valid1 = mx.broadcast_to((mx.arange(timesteps)[None, :] >= 1).astype(mx.float32), (batch, timesteps))
        valid2 = mx.broadcast_to((mx.arange(timesteps)[None, :] >= 2).astype(mx.float32), (batch, timesteps))
        features = mx.stack(
            [
                entropy,
                max_logit,
                variance,
                mx.log1p(bigram_totals),
                mx.log1p(trigram_totals),
                valid1,
                valid2,
            ],
            axis=-1,
        )
        proxy_columns: list[mx.array] = []
        if (
            self.config.structure_proxy_entropy
            or self.config.structure_proxy_peak
            or self.config.structure_proxy_candidate4
            or self.config.structure_proxy_agreement
            or self.config.structure_proxy_agreement_mass
        ):
            proxy = structure_proxy_feature_arrays(base_probs, p_trigram)
            if self.config.structure_proxy_entropy:
                proxy_columns.append(proxy["entropy"][..., None])
            if self.config.structure_proxy_peak:
                proxy_columns.append(proxy["peak"][..., None])
            if self.config.structure_proxy_candidate4:
                proxy_columns.append(proxy["candidate4"][..., None])
            if self.config.structure_proxy_agreement:
                proxy_columns.append(proxy["agreement"][..., None])
            if self.config.structure_proxy_agreement_mass:
                proxy_columns.append(proxy["agreement_mass"][..., None])
        if proxy_columns:
            features = mx.concatenate([features, *proxy_columns], axis=-1)
        hidden = mx.tanh(self.controller_hidden(features))
        logits = self.controller_out(hidden)
        temperature = max(self.config.controller_temperature, 1e-4)
        return mx.softmax(logits / temperature, axis=-1)

    def __call__(self, chars: mx.array) -> mx.array:
        base_logits = self.base(chars)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)
        base_probs = mx.softmax(base_logits, axis=-1)
        p_uni, p_bigram, p_trigram, bigram_totals, trigram_totals = self._memory_probs(chars)
        if self.config.blend_mode == "memory_only":
            mixed = p_trigram
        else:
            if self.config.blend_mode == "learned_mixer":
                weights = self._controller_weights(
                    base_logits,
                    base_probs,
                    p_bigram,
                    p_trigram,
                    bigram_totals,
                    trigram_totals,
                )
            else:
                fixed = np.asarray(self.config.fixed_component_weights, dtype=np.float32)
                fixed = fixed / np.maximum(fixed.sum(), 1e-8)
                weights = mx.broadcast_to(mx.array(fixed, dtype=mx.float32)[None, None, :], (chars.shape[0], chars.shape[1], 4))
            mixed = (
                weights[:, :, 0:1] * base_probs
                + weights[:, :, 1:2] * p_uni
                + weights[:, :, 2:3] * p_bigram
                + weights[:, :, 3:4] * p_trigram
            )
        mixed = mixed / mx.maximum(mx.sum(mixed, axis=-1, keepdims=True), mx.array(1e-8, dtype=mixed.dtype))
        return mx.log(mx.maximum(mixed, mx.array(1e-8, dtype=mixed.dtype)))


def scale_config(config: ConkerTenConfig, scale: float) -> ConkerTenConfig:
    if scale == 1.0:
        return config
    return replace(config, base_config=scale_conker3_config(config.base_config, scale))
