from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from conker.src.asura import ASURA_FEATURE_DIM, AsuraConfig, AsuraModel, ROUTED_SOURCE_NAMES
from conker.src.conker3 import scale_config as scale_conker3_config


PROGRAM_EFFECT_DIM = 2


@dataclass(frozen=True)
class AsuraTetraConfig(AsuraConfig):
    program_slots: int = 4
    program_temperature: float = 1.0
    program_route_span: float = 1.0
    program_opener_span: float = 1.0


class AsuraTetraModel(AsuraModel):
    """Asura with a compact four-axis program tensor over lag, source, slot, and effect."""

    def __init__(self, vocab_size: int, config: AsuraTetraConfig = AsuraTetraConfig()):
        super().__init__(vocab_size=vocab_size, config=config)
        slot_count = max(int(config.program_slots), 1)
        lag_count = len(config.lag_lookbacks)
        source_count = len(ROUTED_SOURCE_NAMES)
        self.asura_program_feature_weights = mx.zeros((slot_count, ASURA_FEATURE_DIM), dtype=mx.float32)
        self.asura_program_bias = mx.zeros((slot_count,), dtype=mx.float32)
        self.asura_program_tensor = mx.zeros((lag_count, source_count, slot_count, PROGRAM_EFFECT_DIM), dtype=mx.float32)

    @property
    def tetra_config(self) -> AsuraTetraConfig:
        return self.config

    def _program_mix(self, features: mx.array) -> mx.array:
        logits = (
            mx.sum(features[..., None, :] * self.asura_program_feature_weights[None, None, :, :], axis=-1)
            + self.asura_program_bias[None, None, :]
        )
        temperature = max(self.tetra_config.program_temperature, 1e-4)
        return mx.softmax(logits / temperature, axis=-1)

    def _program_effects(
        self,
        lag_weights: mx.array,
        program_mix: mx.array,
        dtype: mx.Dtype,
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        joint = (
            lag_weights[:, :, :, None, None, None]
            * program_mix[:, :, None, None, :, None]
            * self.asura_program_tensor[None, None, :, :, :, :].astype(dtype)
        )
        effects = mx.sum(joint, axis=(2, 4))
        route_deltas = self.tetra_config.program_route_span * mx.tanh(effects[..., 0])
        opener_deltas = self.tetra_config.program_opener_span * mx.tanh(effects[..., 1])
        route_map = {
            name: route_deltas[..., idx]
            for idx, name in enumerate(ROUTED_SOURCE_NAMES)
        }
        opener_map = {
            name: opener_deltas[..., idx]
            for idx, name in enumerate(ROUTED_SOURCE_NAMES)
        }
        return route_map, opener_map

    def _source_router_programmed(
        self,
        features: mx.array,
        sources: dict[str, mx.array | None],
        route_adjustments: dict[str, mx.array],
        dtype: mx.Dtype,
    ) -> dict[str, mx.array]:
        active_items = [(name, source) for name, source in sources.items() if source is not None]
        if not active_items:
            return {}
        logits = []
        names = []
        for name, source in active_items:
            source_idx = ROUTED_SOURCE_NAMES.index(name)
            base_logit = self._source_controller_logit(source_idx, features, source, dtype)
            logits.append(base_logit + route_adjustments.get(name, mx.zeros_like(base_logit)))
            names.append(name)
        abstain_logit = (
            mx.sum(features * self.asura_source_abstain_feature_weights[None, None, :], axis=-1)
            + self.asura_source_abstain_bias
        ).astype(dtype)
        temperature = max(self.routing_config.source_controller_temperature, 1e-4)
        gate_logits = mx.stack(logits + [abstain_logit], axis=-1)
        gate_probs = mx.softmax(gate_logits / temperature, axis=-1)
        support_probs = gate_probs[..., :-1]
        support_mass = 1.0 - gate_probs[..., -1:]
        if self.routing_config.source_topk > 0:
            k = min(self.routing_config.source_topk, support_probs.shape[-1])
            thresh = mx.sort(support_probs, axis=-1)[..., -k:]
            cutoff = thresh[..., :1]
            top_mask = (support_probs >= cutoff).astype(dtype)
            support_probs = support_probs * top_mask
            denom = mx.maximum(mx.sum(support_probs, axis=-1, keepdims=True), mx.array(1e-6, dtype=dtype))
            support_probs = support_mass * (support_probs / denom)
        else:
            support_probs = support_mass * support_probs
        return {name: support_probs[..., idx : idx + 1] for idx, name in enumerate(names)}

    def _opener_router_programmed(
        self,
        features: mx.array,
        sources: dict[str, mx.array | None],
        opener_adjustments: dict[str, mx.array],
        dtype: mx.Dtype,
    ) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        for name, source in sources.items():
            if source is None:
                continue
            source_idx = ROUTED_SOURCE_NAMES.index(name)
            _, hit = self._source_mass_and_hit(source, dtype)
            temperature = max(self.routing_config.opener_controller_temperature, 1e-4)
            base_logit = self._opener_controller_logit(source_idx, features, source, dtype)
            adjusted = base_logit + opener_adjustments.get(name, mx.zeros_like(base_logit))
            activation = mx.sigmoid(adjusted / temperature) * hit
            out[name] = activation[..., None]
        return out

    def _routing_state_full(self, chars: mx.array) -> dict[str, object]:
        base_logits = self.base(chars)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)
        base_features = self._support_gate_features(chars, base_logits)
        lag_weights = self._lag_controller(base_features)
        sources = self._source_dict_from_tuple(self._count_features_with_controller(chars, lag_weights))
        route_features = self._asura_features(base_features, lag_weights)
        program_mix = self._program_mix(route_features)
        route_adjustments, opener_adjustments = self._program_effects(lag_weights, program_mix, base_logits.dtype)
        source_probs = self._source_router_programmed(route_features, sources, route_adjustments, base_logits.dtype)
        opener_probs = self._opener_router_programmed(route_features, sources, opener_adjustments, base_logits.dtype)
        residual_scale = self._residual_scale(route_features)
        return {
            "base_logits": base_logits,
            "sources": sources,
            "lag_weights": lag_weights,
            "source_probs": source_probs,
            "opener_probs": opener_probs,
            "residual_scale": residual_scale,
            "program_mix": program_mix,
            "route_adjustments": route_adjustments,
            "opener_adjustments": opener_adjustments,
        }

    def _routing_state(
        self,
        chars: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array | None], mx.array, dict[str, mx.array], dict[str, mx.array], mx.array]:
        state = self._routing_state_full(chars)
        return (
            state["base_logits"],
            state["sources"],
            state["lag_weights"],
            state["source_probs"],
            state["opener_probs"],
            state["residual_scale"],
        )

    def controller_snapshot(self, chars: mx.array) -> dict[str, object]:
        state = self._routing_state_full(chars)
        lag_weights = state["lag_weights"]
        base_logits = state["base_logits"]
        source_probs = state["source_probs"]
        opener_probs = state["opener_probs"]
        sources = state["sources"]
        residual_scale = state["residual_scale"]
        program_mix = state["program_mix"]
        route_adjustments = state["route_adjustments"]
        opener_adjustments = state["opener_adjustments"]

        lag_weight_means = np.array(mx.mean(lag_weights, axis=(0, 1)), copy=False)
        global_lag_idx = self._global_lag_bucket_idx()
        global_lag_mean = 0.0 if global_lag_idx is None else float(lag_weight_means[global_lag_idx])
        program_slot_means = np.array(mx.mean(program_mix, axis=(0, 1)), copy=False)
        route_prob_means = {name: float(np.array(mx.mean(value), copy=False)) for name, value in source_probs.items()}
        opener_prob_means = {name: float(np.array(mx.mean(value), copy=False)) for name, value in opener_probs.items()}
        route_delta_means = {
            name: float(np.array(mx.mean(mx.abs(delta)), copy=False))
            for name, delta in route_adjustments.items()
        }
        opener_delta_means = {
            name: float(np.array(mx.mean(mx.abs(delta)), copy=False))
            for name, delta in opener_adjustments.items()
        }
        source_mass_means: dict[str, float] = {}
        for name, source in sources.items():
            if source is None:
                continue
            mass = mx.mean(mx.log1p(mx.sum(source, axis=-1)))
            source_mass_means[name] = float(np.array(mass, copy=False))
        return {
            "lag_weight_means": {
                str(lookback): float(value)
                for lookback, value in zip(self.routing_config.lag_lookbacks, lag_weight_means.tolist())
            },
            "program_slot_means": {
                str(slot_idx): float(value)
                for slot_idx, value in enumerate(program_slot_means.tolist())
            },
            "route_prob_means": route_prob_means,
            "opener_prob_means": opener_prob_means,
            "route_delta_abs_means": route_delta_means,
            "opener_delta_abs_means": opener_delta_means,
            "source_mass_means": source_mass_means,
            "bounded_lag_mass_mean": 1.0 - global_lag_mean,
            "global_lag_mass_mean": global_lag_mean,
            "residual_scale_mean": float(np.array(mx.mean(residual_scale), copy=False)),
            "base_logit_abs_mean": float(np.array(mx.mean(mx.abs(base_logits)), copy=False)),
        }


def scale_config(config: AsuraTetraConfig, scale: float) -> AsuraTetraConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        base_config=scale_conker3_config(config.base_config, scale),
    )
