from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn

from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config as scale_conker3_config


@dataclass(frozen=True)
class ConkerFiveConfig:
    base_config: ConkerThreeConfig = ConkerThreeConfig()
    freeze_base: bool = True
    state_proj_dim: int = 32
    stat_feature_dim: int = 9
    shared_hidden_dim: int = 64
    num_heads: int = 8
    head_rank: int = 8
    residual_cap: float = 2.0


class ConkerFiveModel(nn.Module):
    """Frozen Conker-3 substrate plus learned residual discriminators and gates."""

    def __init__(self, vocab_size: int, config: ConkerFiveConfig = ConkerFiveConfig()):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        self.base = ConkerThreeModel(vocab_size=vocab_size, config=config.base_config)
        if config.freeze_base:
            self.base.freeze(recurse=True)

        if config.base_config.enable_linear:
            self.state_proj = nn.Linear(config.base_config.linear_modes, config.state_proj_dim)
        else:
            self.state_proj = None
        self.shared_proj = nn.Linear(config.state_proj_dim + config.stat_feature_dim, config.shared_hidden_dim)
        self.gate_proj = nn.Linear(config.shared_hidden_dim, config.num_heads)
        self.value_proj = nn.Linear(config.shared_hidden_dim, config.num_heads * config.head_rank)
        self.out_basis = mx.zeros((config.num_heads, config.head_rank, vocab_size), dtype=mx.float32)
        self.bias = mx.zeros((vocab_size,), dtype=mx.float32)

    def _combine_base_logits(
        self,
        logits_linear: mx.array | None,
        logits_local: mx.array | None,
    ) -> mx.array:
        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear
        if self.base.gate_proj is None:
            gate = self.base.config.local_scale
        else:
            ent_l, max_l, var_l = self.base._logit_features(logits_linear)
            ent_r, max_r, var_r = self.base._logit_features(logits_local)
            features = mx.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], axis=-1)
            gate = mx.sigmoid(self.base.gate_proj(features)) * self.base.config.local_scale
        return logits_linear + gate * logits_local

    def _substrate_outputs(
        self,
        chars: mx.array,
    ) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
        linear_states = None
        logits_linear = self.base._linear_logits(chars) if self.base.config.enable_linear else None
        if self.base.config.enable_linear:
            linear_states, _ = self.base._linear_states(chars)
        logits_local = self.base._local_logits(chars) if self.base.config.enable_local else None
        base_logits = self._combine_base_logits(logits_linear, logits_local)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)
            if linear_states is not None:
                linear_states = mx.stop_gradient(linear_states)
            if logits_linear is not None:
                logits_linear = mx.stop_gradient(logits_linear)
            if logits_local is not None:
                logits_local = mx.stop_gradient(logits_local)
        return base_logits, linear_states, logits_linear, logits_local

    def _stat_features(
        self,
        base_logits: mx.array,
        logits_linear: mx.array | None,
        logits_local: mx.array | None,
    ) -> mx.array:
        ent_b, max_b, var_b = self.base._logit_features(base_logits)
        zeros = mx.zeros_like(ent_b)
        if logits_linear is not None:
            ent_l, max_l, var_l = self.base._logit_features(logits_linear)
        else:
            ent_l, max_l, var_l = zeros, zeros, zeros
        if logits_local is not None:
            ent_r, max_r, var_r = self.base._logit_features(logits_local)
        else:
            ent_r, max_r, var_r = zeros, zeros, zeros
        return mx.stack([ent_b, max_b, var_b, ent_l, max_l, var_l, ent_r, max_r, var_r], axis=-1)

    def __call__(self, chars: mx.array) -> mx.array:
        base_logits, linear_states, logits_linear, logits_local = self._substrate_outputs(chars)
        stat_features = self._stat_features(base_logits, logits_linear, logits_local)
        if self.state_proj is not None and linear_states is not None:
            state_features = mx.tanh(self.state_proj(linear_states))
        else:
            batch, timesteps, _ = stat_features.shape
            state_features = mx.zeros((batch, timesteps, self.config.state_proj_dim), dtype=stat_features.dtype)
        shared_in = mx.concatenate([state_features, stat_features], axis=-1)
        shared = mx.tanh(self.shared_proj(shared_in))
        gates = mx.sigmoid(self.gate_proj(shared))
        values = mx.tanh(self.value_proj(shared))
        values = mx.reshape(
            values,
            (values.shape[0], values.shape[1], self.config.num_heads, self.config.head_rank),
        )
        delta = mx.broadcast_to(self.bias, base_logits.shape)
        for head_idx in range(self.config.num_heads):
            head_delta = mx.matmul(values[:, :, head_idx, :], self.out_basis[head_idx])
            delta = delta + gates[:, :, head_idx : head_idx + 1] * head_delta
        residual = self.config.residual_cap * mx.tanh(delta / self.config.residual_cap)
        return base_logits + residual


def scale_config(config: ConkerFiveConfig, scale: float) -> ConkerFiveConfig:
    if scale == 1.0:
        return config
    return replace(config, base_config=scale_conker3_config(config.base_config, scale))
