from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config as scale_conker3_config


CONKER13_FEATURE_DIM = 8
CONKER13_EFFECT_DIM = 3


def _mode_group_projection(linear_modes: int, group_count: int) -> np.ndarray:
    if linear_modes <= 0:
        raise ValueError("Conker-13 requires linear_modes > 0.")
    groups = max(min(int(group_count), int(linear_modes)), 1)
    mode_ids = np.floor(np.arange(linear_modes, dtype=np.float32) * groups / linear_modes).astype(np.int32)
    proj = np.zeros((groups, linear_modes), dtype=np.float32)
    proj[mode_ids, np.arange(linear_modes)] = 1.0
    return proj


@dataclass(frozen=True)
class ConkerThirteenConfig:
    base_config: ConkerThreeConfig = ConkerThreeConfig(max_seq_len=256, linear_modes=256, local_window=4)
    lag_lookbacks: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 0)
    lag_temperature: float = 1.0
    mode_groups: int = 8
    program_slots: int = 4
    program_temperature: float = 1.0
    linear_gate_span: float = 1.0
    local_gate_span: float = 1.0
    local_scale_span: float = 0.5


class ConkerThirteenModel(nn.Module):
    """Five-axis controller directly over Conker-3 linear groups and local window offsets."""

    def __init__(self, vocab_size: int, config: ConkerThirteenConfig = ConkerThirteenConfig()):
        super().__init__()
        if not config.base_config.enable_linear or not config.base_config.enable_local:
            raise ValueError("Conker-13 requires both the Conker-3 linear and local paths.")
        if len(config.lag_lookbacks) == 0:
            raise ValueError("Conker-13 requires at least one lag lookback.")

        self.vocab_size = vocab_size
        self.config = config
        self.base = ConkerThreeModel(vocab_size=vocab_size, config=config.base_config)
        self.mode_group_count = max(min(int(config.mode_groups), int(config.base_config.linear_modes)), 1)
        self.mode_group_projection = mx.array(
            _mode_group_projection(config.base_config.linear_modes, self.mode_group_count),
            dtype=mx.float32,
        )

        lag_count = len(config.lag_lookbacks)
        window = int(config.base_config.local_window)
        slot_count = max(int(config.program_slots), 1)

        self.conker13_lag_feature_weights = mx.zeros((lag_count, CONKER13_FEATURE_DIM), dtype=mx.float32)
        self.conker13_lag_bias = mx.zeros((lag_count,), dtype=mx.float32)
        self.conker13_program_feature_weights = mx.zeros((slot_count, CONKER13_FEATURE_DIM), dtype=mx.float32)
        self.conker13_program_bias = mx.zeros((slot_count,), dtype=mx.float32)
        self.conker13_program_tensor = mx.zeros(
            (lag_count, self.mode_group_count, window, slot_count, CONKER13_EFFECT_DIM),
            dtype=mx.float32,
        )

    @staticmethod
    def _controller_features(logits_linear: mx.array, logits_local: mx.array) -> mx.array:
        ent_l, max_l, var_l = ConkerThreeModel._logit_features(logits_linear)
        ent_r, max_r, var_r = ConkerThreeModel._logit_features(logits_local)
        centered_linear = logits_linear - mx.mean(logits_linear, axis=-1, keepdims=True)
        centered_local = logits_local - mx.mean(logits_local, axis=-1, keepdims=True)
        mean_abs_diff = mx.mean(mx.abs(centered_local - centered_linear), axis=-1)
        max_gap = max_r - max_l
        return mx.stack(
            [ent_l, ent_r, max_l, max_r, var_l, var_r, mean_abs_diff, max_gap],
            axis=-1,
        )

    def _lag_controller(self, features: mx.array) -> mx.array:
        logits = (
            mx.sum(features[..., None, :] * self.conker13_lag_feature_weights[None, None, :, :], axis=-1)
            + self.conker13_lag_bias[None, None, :]
        )
        temperature = max(float(self.config.lag_temperature), 1e-4)
        return mx.softmax(logits / temperature, axis=-1)

    def _program_mix(self, features: mx.array) -> mx.array:
        logits = (
            mx.sum(features[..., None, :] * self.conker13_program_feature_weights[None, None, :, :], axis=-1)
            + self.conker13_program_bias[None, None, :]
        )
        temperature = max(float(self.config.program_temperature), 1e-4)
        return mx.softmax(logits / temperature, axis=-1)

    def _program_effects(
        self,
        lag_weights: mx.array,
        program_mix: mx.array,
        dtype: mx.Dtype,
    ) -> mx.array:
        joint = (
            lag_weights[:, :, :, None, None, None, None]
            * program_mix[:, :, None, None, None, :, None]
            * self.conker13_program_tensor[None, None, :, :, :, :, :].astype(dtype)
        )
        return mx.sum(joint, axis=(2, 5))

    def _local_window_views(self, chars: mx.array) -> tuple[mx.array, mx.array]:
        x = self.base._embed_local(chars)
        stacked = self.base._local_window_stack(x)
        batch, timesteps, _ = stacked.shape
        window = self.base.config.local_window
        dim = self.base.config.embedding_dim
        return x, mx.reshape(stacked, (batch, timesteps, window, dim))

    def _controller_state(self, chars: mx.array) -> dict[str, mx.array]:
        linear_states, linear_embed = self.base._linear_states(chars)
        _local_embed, local_stack = self._local_window_views(chars)

        raw_linear_logits = self.base.linear_readout(mx.concatenate([linear_states, linear_embed], axis=-1))
        raw_local_logits = self.base.local_readout(mx.reshape(local_stack, (*local_stack.shape[:2], -1)))

        features = self._controller_features(raw_linear_logits, raw_local_logits)
        lag_weights = self._lag_controller(features)
        program_mix = self._program_mix(features)
        effects = self._program_effects(lag_weights, program_mix, raw_linear_logits.dtype)

        linear_group_scores = mx.mean(effects[..., 0], axis=-1)
        local_offset_scores = mx.mean(effects[..., 1], axis=-2)
        local_scale_scores = mx.mean(effects[..., 2], axis=(-2, -1))

        linear_group_gates = mx.array(1.0, dtype=raw_linear_logits.dtype) + self.config.linear_gate_span * mx.tanh(linear_group_scores)
        linear_mode_gates = mx.matmul(linear_group_gates, self.mode_group_projection.astype(raw_linear_logits.dtype))

        local_offset_gates = mx.array(1.0, dtype=raw_linear_logits.dtype) + self.config.local_gate_span * mx.tanh(local_offset_scores)
        local_scale = mx.maximum(
            mx.array(0.0, dtype=raw_linear_logits.dtype),
            mx.array(1.0, dtype=raw_linear_logits.dtype) + self.config.local_scale_span * mx.tanh(local_scale_scores),
        )[..., None]

        gated_linear_states = linear_states * linear_mode_gates
        linear_logits = self.base.linear_readout(mx.concatenate([gated_linear_states, linear_embed], axis=-1))

        gated_local_stack = local_stack * local_offset_gates[..., None]
        local_logits = self.base.local_readout(mx.reshape(gated_local_stack, (*gated_local_stack.shape[:2], -1)))

        return {
            "features": features,
            "lag_weights": lag_weights,
            "program_mix": program_mix,
            "effects": effects,
            "raw_linear_logits": raw_linear_logits,
            "raw_local_logits": raw_local_logits,
            "linear_logits": linear_logits,
            "local_logits": local_logits,
            "linear_group_gates": linear_group_gates,
            "linear_mode_gates": linear_mode_gates,
            "local_offset_gates": local_offset_gates,
            "local_scale": local_scale,
        }

    def controller_snapshot(self, chars: mx.array) -> dict[str, object]:
        state = self._controller_state(chars)
        lag_weight_means = np.array(mx.mean(state["lag_weights"], axis=(0, 1)), copy=False)
        program_slot_means = np.array(mx.mean(state["program_mix"], axis=(0, 1)), copy=False)
        linear_group_gate_means = np.array(mx.mean(state["linear_group_gates"], axis=(0, 1)), copy=False)
        local_offset_gate_means = np.array(mx.mean(state["local_offset_gates"], axis=(0, 1)), copy=False)
        mode_gate_means = np.array(mx.mean(state["linear_mode_gates"], axis=(0, 1)), copy=False)

        return {
            "lag_weight_means": {
                str(lookback): float(value)
                for lookback, value in zip(self.config.lag_lookbacks, lag_weight_means.tolist())
            },
            "program_slot_means": {
                str(slot_idx): float(value)
                for slot_idx, value in enumerate(program_slot_means.tolist())
            },
            "linear_group_gate_means": {
                str(group_idx): float(value)
                for group_idx, value in enumerate(linear_group_gate_means.tolist())
            },
            "local_offset_gate_means": {
                str(offset_idx): float(value)
                for offset_idx, value in enumerate(local_offset_gate_means.tolist())
            },
            "linear_mode_gate_mean": float(np.mean(mode_gate_means)),
            "local_scale_mean": float(np.array(mx.mean(state["local_scale"]), copy=False)),
            "effect_abs_mean": float(np.array(mx.mean(mx.abs(state["effects"])), copy=False)),
            "raw_linear_logit_abs_mean": float(np.array(mx.mean(mx.abs(state["raw_linear_logits"])), copy=False)),
            "raw_local_logit_abs_mean": float(np.array(mx.mean(mx.abs(state["raw_local_logits"])), copy=False)),
        }

    def __call__(self, chars: mx.array) -> mx.array:
        state = self._controller_state(chars)
        logits_linear = state["linear_logits"]
        logits_local = state["local_logits"]

        if self.base.gate_proj is None:
            mix_gate = mx.array(self.base.config.local_scale, dtype=logits_linear.dtype) * state["local_scale"]
        else:
            ent_l, max_l, var_l = ConkerThreeModel._logit_features(logits_linear)
            ent_r, max_r, var_r = ConkerThreeModel._logit_features(logits_local)
            gate_features = mx.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], axis=-1)
            mix_gate = mx.sigmoid(self.base.gate_proj(gate_features)) * self.base.config.local_scale * state["local_scale"]

        return logits_linear + mix_gate * logits_local


def scale_config(config: ConkerThirteenConfig, scale: float) -> ConkerThirteenConfig:
    if scale == 1.0:
        return config
    return replace(config, base_config=scale_conker3_config(config.base_config, scale))
