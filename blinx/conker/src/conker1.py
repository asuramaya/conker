from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from carving_machine.config import HierarchicalCarverConfig
from carving_machine.models import HierarchicalCarverModel, MLP, MixedMemoryHierarchicalModel


@dataclass(frozen=True)
class ConkerOneConfig:
    embedding_dim: int = 32
    mixer_hidden: tuple[int, ...] = (32,)
    mixer_bias_scale: float = 0.25
    fast_mid_delay: HierarchicalCarverConfig = field(
        default_factory=lambda: HierarchicalCarverConfig(
            fast_memory_mode="delay",
            mid_memory_mode="delay",
            slow_memory_mode="recurrent",
        )
    )
    v6_silenced: HierarchicalCarverConfig = field(
        default_factory=lambda: HierarchicalCarverConfig(aux_source="zeros")
    )


class ConkerOneModel(nn.Module):
    """Two-expert compressor: fast+mid-delay + v6_silenced with a tiny causal mixer."""

    def __init__(self, vocab_size: int, config: ConkerOneConfig = ConkerOneConfig()):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        self.fast_mid_delay = MixedMemoryHierarchicalModel(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            config=config.fast_mid_delay,
        )
        self.v6_silenced = HierarchicalCarverModel(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            config=config.v6_silenced,
        )
        self.fast_mid_delay.freeze_static()
        self.v6_silenced.freeze_static()
        self.mixer = MLP(8, config.mixer_hidden, 2)
        self.bias_proj = nn.Linear(8, vocab_size)

    @staticmethod
    def _logit_features(logits: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        max_logit = mx.max(logits, axis=-1)
        mean_logit = mx.mean(logits, axis=-1)
        centered = logits - mean_logit[..., None]
        variance = mx.mean(centered * centered, axis=-1)
        return entropy, max_logit, mean_logit, variance

    def __call__(self, chars: mx.array) -> mx.array:
        logits_a = self.fast_mid_delay(chars)
        logits_b = self.v6_silenced(chars)

        ent_a, max_a, mean_a, var_a = self._logit_features(logits_a)
        ent_b, max_b, mean_b, var_b = self._logit_features(logits_b)
        features = mx.stack(
            [
                ent_a,
                ent_b,
                max_a,
                max_b,
                var_a,
                var_b,
                ent_a - ent_b,
                max_a - max_b,
            ],
            axis=-1,
        )
        mix_logits = self.mixer(features)
        mix = mx.softmax(mix_logits, axis=-1)
        residual_bias = self.config.mixer_bias_scale * self.bias_proj(features)
        return (
            mix[..., 0:1] * logits_a
            + mix[..., 1:2] * logits_b
            + residual_bias
        )
