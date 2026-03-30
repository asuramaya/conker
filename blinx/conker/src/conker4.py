from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from carving_machine.models import MLP
from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config as scale_conker3_config


def _strict_causal_mask(max_seq_len: int) -> np.ndarray:
    return np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.float32), k=-1)


def _recency_kernel(max_seq_len: int, half_life: float) -> np.ndarray:
    if half_life <= 0.0:
        raise ValueError("Conker-4 recency_half_life must be > 0.")
    decay = float(np.exp(np.log(0.5) / half_life))
    time_idx = np.arange(max_seq_len, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    mask = delta > 0
    safe_delta = np.where(mask, delta - 1, 0).astype(np.float32)
    kernel = np.power(decay, safe_delta, dtype=np.float32)
    return np.where(mask, kernel, 0.0).astype(np.float32)


@dataclass(frozen=True)
class ConkerFourConfig:
    neural_config: ConkerThreeConfig = field(default_factory=ConkerThreeConfig)
    enable_neural: bool = True
    freeze_neural: bool = True
    enable_exact1: bool = True
    enable_exact2: bool = True
    enable_recency: bool = True
    recency_half_life: float = 8.0
    count_alpha: float = 0.05
    mixer_mode: str = "support"
    support_scale: float = 0.5
    neural_bias_init: float = 2.0
    aux_bias_init: float = -1.0
    mixer_hidden: tuple[int, ...] = (32,)


class ConkerFourModel(nn.Module):
    """Conker-4: Conker-3 plus exact-context experts with a tiny probability mixer."""

    def __init__(self, vocab_size: int, config: ConkerFourConfig = ConkerFourConfig()):
        super().__init__()
        if not any((config.enable_neural, config.enable_exact1, config.enable_exact2, config.enable_recency)):
            raise ValueError("Conker-4 must enable at least one expert.")
        self.vocab_size = vocab_size
        self.config = config
        self.vocab_axis = mx.arange(vocab_size, dtype=mx.int32)
        self.causal_mask = mx.array(_strict_causal_mask(config.neural_config.max_seq_len))
        self.recency_kernel = mx.array(_recency_kernel(config.neural_config.max_seq_len, config.recency_half_life))

        self.neural = None
        if config.enable_neural:
            self.neural = ConkerThreeModel(vocab_size=vocab_size, config=config.neural_config)
            if config.freeze_neural:
                self.neural.freeze(recurse=True)

        self.expert_names: list[str] = []
        if config.enable_neural:
            self.expert_names.append("neural")
        if config.enable_exact1:
            self.expert_names.append("exact1")
        if config.enable_exact2:
            self.expert_names.append("exact2")
        if config.enable_recency:
            self.expert_names.append("recency")

        self.mixer = None
        self.expert_bias_logits = None
        if len(self.expert_names) > 1:
            if config.mixer_mode == "mlp":
                self.mixer = MLP(3 * len(self.expert_names), config.mixer_hidden, len(self.expert_names))
            elif config.mixer_mode == "support":
                bias_init = np.array(
                    [
                        config.neural_bias_init if name == "neural" else config.aux_bias_init
                        for name in self.expert_names
                    ],
                    dtype=np.float32,
                )
                self.expert_bias_logits = mx.array(bias_init)
            else:
                raise ValueError(f"Unknown Conker-4 mixer_mode: {config.mixer_mode}")

        self.freeze(keys=["vocab_axis", "causal_mask", "recency_kernel"], strict=False)

    def _one_hot(self, chars: mx.array) -> mx.array:
        return mx.where(chars[..., None] == self.vocab_axis[None, None, :], 1.0, 0.0)

    @staticmethod
    def _log_probs_from_logits(logits: mx.array) -> mx.array:
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _log_probs_from_counts(self, counts: mx.array) -> tuple[mx.array, mx.array]:
        alpha = self.config.count_alpha
        totals = mx.sum(counts, axis=-1, keepdims=True)
        probs = (counts + (alpha / self.vocab_size)) / (totals + alpha)
        support = mx.log1p(totals[..., 0])
        return mx.log(mx.maximum(probs, 1e-9)), support

    @staticmethod
    def _expert_features(log_probs: mx.array, support: mx.array) -> mx.array:
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        max_prob = mx.max(probs, axis=-1)
        return mx.stack([entropy, max_prob, support], axis=-1)

    def _count_experts(self, chars: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        batch, timesteps = chars.shape
        if timesteps > self.config.neural_config.max_seq_len:
            raise ValueError(
                f"Conker-4 max_seq_len={self.config.neural_config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        one_hot = self._one_hot(chars)
        zeros = mx.zeros((batch, 1, self.vocab_size), dtype=one_hot.dtype)
        next_one_hot = mx.concatenate([one_hot[:, 1:, :], zeros], axis=1)
        prev_one_hot = mx.concatenate([zeros, one_hot[:, :-1, :]], axis=1)
        mask = self.causal_mask[:timesteps, :timesteps]

        cur_match = mx.matmul(one_hot, mx.transpose(one_hot, (0, 2, 1))) * mask[None, :, :]
        exact1 = mx.matmul(cur_match, next_one_hot)

        prev_match = mx.matmul(prev_one_hot, mx.transpose(prev_one_hot, (0, 2, 1)))
        exact2 = mx.matmul(cur_match * prev_match, next_one_hot)

        recency_kernel = self.recency_kernel[:timesteps, :timesteps]
        recency = mx.matmul(mx.broadcast_to(recency_kernel[None, :, :], (batch, timesteps, timesteps)), one_hot)
        return exact1, exact2, recency

    def __call__(self, chars: mx.array) -> mx.array:
        expert_log_probs: list[mx.array] = []
        feature_rows: list[mx.array] = []
        support_rows: list[mx.array] = []

        exact1_counts = exact2_counts = recency_counts = None
        need_counts = any((self.config.enable_exact1, self.config.enable_exact2, self.config.enable_recency))
        if need_counts:
            exact1_counts, exact2_counts, recency_counts = self._count_experts(chars)

        if self.config.enable_neural:
            if self.neural is None:
                raise RuntimeError("Conker-4 neural expert is missing.")
            neural_logits = self.neural(chars)
            neural_log_probs = self._log_probs_from_logits(neural_logits)
            if self.config.freeze_neural:
                neural_log_probs = mx.stop_gradient(neural_log_probs)
            support = mx.ones(chars.shape, dtype=neural_log_probs.dtype)
            expert_log_probs.append(neural_log_probs)
            feature_rows.append(self._expert_features(neural_log_probs, support))
            support_rows.append(support)

        if self.config.enable_exact1:
            exact1_log_probs, exact1_support = self._log_probs_from_counts(exact1_counts)
            expert_log_probs.append(exact1_log_probs)
            feature_rows.append(self._expert_features(exact1_log_probs, exact1_support))
            support_rows.append(exact1_support)

        if self.config.enable_exact2:
            exact2_log_probs, exact2_support = self._log_probs_from_counts(exact2_counts)
            expert_log_probs.append(exact2_log_probs)
            feature_rows.append(self._expert_features(exact2_log_probs, exact2_support))
            support_rows.append(exact2_support)

        if self.config.enable_recency:
            recency_log_probs, recency_support = self._log_probs_from_counts(recency_counts)
            expert_log_probs.append(recency_log_probs)
            feature_rows.append(self._expert_features(recency_log_probs, recency_support))
            support_rows.append(recency_support)

        if len(expert_log_probs) == 1:
            return expert_log_probs[0]

        stacked_log_probs = mx.stack(expert_log_probs, axis=2)
        expert_probs = mx.exp(stacked_log_probs)
        if self.mixer is not None:
            mixer_features = mx.concatenate(feature_rows, axis=-1)
            weights = mx.softmax(self.mixer(mixer_features), axis=-1)
        else:
            support_stack = mx.stack(support_rows, axis=-1)
            weights = mx.softmax(
                self.expert_bias_logits[None, None, :] + self.config.support_scale * support_stack,
                axis=-1,
            )
        mixed_probs = mx.sum(weights[..., None] * expert_probs, axis=2)
        return mx.log(mx.maximum(mixed_probs, 1e-9))


def scale_config(config: ConkerFourConfig, scale: float) -> ConkerFourConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        neural_config=scale_conker3_config(config.neural_config, scale),
    )
