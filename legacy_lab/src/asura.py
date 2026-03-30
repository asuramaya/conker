from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from conker.src.conker3 import scale_config as scale_conker3_config
from conker.src.conker9 import ConkerNineConfig, ConkerNineModel


ROUTED_SOURCE_NAMES = (
    "exact1",
    "exact2",
    "exact3",
    "special2",
    "number2",
    "urlpath2",
    "markup2",
    "attr2",
    "entity2",
    "stack2",
    "wordclass2",
    "delim2",
    "delimsub2",
    "recency",
)

ASURA_FEATURE_DIM = 11


@dataclass(frozen=True)
class AsuraConfig(ConkerNineConfig):
    source_controller_temperature: float = 1.0
    opener_controller_temperature: float = 1.0
    residual_controller_temperature: float = 1.0
    source_topk: int = 0
    candidate_floor: float = 0.0
    global_lag_cap: float = 0.5


class AsuraModel(ConkerNineModel):
    """Strict causal residual routing with lag control, source ownership, and opener control."""

    def __init__(self, vocab_size: int, config: AsuraConfig = AsuraConfig()):
        super().__init__(vocab_size=vocab_size, config=config)
        source_count = len(ROUTED_SOURCE_NAMES)
        self.asura_source_feature_weights = mx.zeros((source_count, ASURA_FEATURE_DIM), dtype=mx.float32)
        self.asura_source_mass_weights = mx.zeros((source_count,), dtype=mx.float32)
        self.asura_source_hit_weights = mx.zeros((source_count,), dtype=mx.float32)
        self.asura_source_bias = mx.zeros((source_count,), dtype=mx.float32)
        self.asura_source_abstain_feature_weights = mx.zeros((ASURA_FEATURE_DIM,), dtype=mx.float32)
        self.asura_source_abstain_bias = mx.array(0.0, dtype=mx.float32)

        self.asura_opener_feature_weights = mx.zeros((source_count, ASURA_FEATURE_DIM), dtype=mx.float32)
        self.asura_opener_mass_weights = mx.zeros((source_count,), dtype=mx.float32)
        self.asura_opener_hit_weights = mx.zeros((source_count,), dtype=mx.float32)
        self.asura_opener_bias = mx.zeros((source_count,), dtype=mx.float32)

        self.asura_residual_feature_weights = mx.zeros((ASURA_FEATURE_DIM,), dtype=mx.float32)
        self.asura_residual_bias = mx.array(0.0, dtype=mx.float32)

    @property
    def routing_config(self) -> AsuraConfig:
        return self.config

    def _global_lag_bucket_idx(self) -> int | None:
        for idx, lookback in enumerate(self.routing_config.lag_lookbacks):
            if lookback <= 0:
                return idx
        return None

    def _lag_mass_masks(self) -> tuple[mx.array, mx.array, mx.array]:
        short = np.array([1.0 if 0 < lookback <= 8 else 0.0 for lookback in self.routing_config.lag_lookbacks], dtype=np.float32)
        long_finite = np.array([1.0 if lookback >= 32 else 0.0 for lookback in self.routing_config.lag_lookbacks], dtype=np.float32)
        global_mask = np.array([1.0 if lookback <= 0 else 0.0 for lookback in self.routing_config.lag_lookbacks], dtype=np.float32)
        return mx.array(short), mx.array(long_finite), mx.array(global_mask)

    def _lag_controller(self, base_features: mx.array) -> mx.array:
        logits = (
            mx.sum(base_features[..., None, :] * self.lag_gate_feature_weights[None, None, :, :], axis=-1)
            + self.lag_gate_bias[None, None, :]
        )
        temperature = max(self.routing_config.lag_controller_temperature, 1e-4)
        global_idx = self._global_lag_bucket_idx()
        if global_idx is None:
            return mx.softmax(logits / temperature, axis=-1)
        finite_indices = [idx for idx in range(logits.shape[-1]) if idx != global_idx]
        if not finite_indices:
            return mx.ones_like(logits)

        finite_logits = mx.stack([logits[..., idx] for idx in finite_indices], axis=-1)
        finite_probs = mx.softmax(finite_logits / temperature, axis=-1)
        global_logit = logits[..., global_idx : global_idx + 1]
        global_lag_cap = max(0.0, min(1.0, self.routing_config.global_lag_cap))
        global_mass = global_lag_cap * mx.sigmoid(global_logit / temperature)
        bounded_mass = mx.array(1.0, dtype=logits.dtype) - global_mass

        weights: list[mx.array] = []
        finite_offset = 0
        for idx in range(logits.shape[-1]):
            if idx == global_idx:
                weights.append(global_mass)
            else:
                weights.append(bounded_mass * finite_probs[..., finite_offset : finite_offset + 1])
                finite_offset += 1
        return mx.concatenate(weights, axis=-1)

    def _asura_features(self, base_features: mx.array, lag_weights: mx.array) -> mx.array:
        short_mask, long_finite_mask, global_mask = self._lag_mass_masks()
        lag_peak = mx.max(lag_weights, axis=-1)
        lag_entropy = -mx.sum(
            lag_weights * mx.log(mx.maximum(lag_weights, mx.array(1e-8, dtype=lag_weights.dtype))),
            axis=-1,
        ) / max(np.log(max(lag_weights.shape[-1], 2)), 1.0)
        short_mass = mx.sum(lag_weights * short_mask[None, None, :], axis=-1)
        long_finite_mass = mx.sum(lag_weights * long_finite_mask[None, None, :], axis=-1)
        global_mass = mx.sum(lag_weights * global_mask[None, None, :], axis=-1)
        return mx.concatenate(
            [
                base_features,
                lag_peak[..., None],
                lag_entropy[..., None],
                short_mass[..., None],
                long_finite_mass[..., None],
                global_mass[..., None],
            ],
            axis=-1,
        )

    @staticmethod
    def _source_mass_and_hit(source: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        mass = mx.log1p(mx.sum(source, axis=-1))
        hit = (mx.sum(source, axis=-1) > 0).astype(dtype)
        return mass, hit

    def _source_controller_logit(
        self,
        source_idx: int,
        features: mx.array,
        source: mx.array,
        dtype: mx.Dtype,
    ) -> mx.array:
        mass, hit = self._source_mass_and_hit(source, dtype)
        return (
            mx.sum(features * self.asura_source_feature_weights[source_idx][None, None, :], axis=-1)
            + self.asura_source_mass_weights[source_idx] * mass
            + self.asura_source_hit_weights[source_idx] * hit
            + self.asura_source_bias[source_idx]
        ).astype(dtype)

    def _opener_controller_logit(
        self,
        source_idx: int,
        features: mx.array,
        source: mx.array,
        dtype: mx.Dtype,
    ) -> mx.array:
        mass, hit = self._source_mass_and_hit(source, dtype)
        return (
            mx.sum(features * self.asura_opener_feature_weights[source_idx][None, None, :], axis=-1)
            + self.asura_opener_mass_weights[source_idx] * mass
            + self.asura_opener_hit_weights[source_idx] * hit
            + self.asura_opener_bias[source_idx]
        ).astype(dtype)

    def _source_router(
        self,
        features: mx.array,
        sources: dict[str, mx.array | None],
        dtype: mx.Dtype,
    ) -> dict[str, mx.array]:
        active_items = [(name, source) for name, source in sources.items() if source is not None]
        if not active_items:
            return {}
        logits = []
        names = []
        for name, source in active_items:
            source_idx = ROUTED_SOURCE_NAMES.index(name)
            logits.append(self._source_controller_logit(source_idx, features, source, dtype))
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

    def _opener_router(
        self,
        features: mx.array,
        sources: dict[str, mx.array | None],
        dtype: mx.Dtype,
    ) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        for name, source in sources.items():
            if source is None:
                continue
            source_idx = ROUTED_SOURCE_NAMES.index(name)
            _, hit = self._source_mass_and_hit(source, dtype)
            temperature = max(self.routing_config.opener_controller_temperature, 1e-4)
            activation = mx.sigmoid(self._opener_controller_logit(source_idx, features, source, dtype) / temperature) * hit
            out[name] = activation[..., None]
        return out

    def _residual_scale(self, features: mx.array) -> mx.array:
        logit = (
            mx.sum(features * self.asura_residual_feature_weights[None, None, :], axis=-1)
            + self.asura_residual_bias
        )
        temperature = max(self.routing_config.residual_controller_temperature, 1e-4)
        return mx.sigmoid(logit / temperature)[..., None]

    def _source_dict_from_tuple(self, feature_tuple: tuple[mx.array | None, ...]) -> dict[str, mx.array | None]:
        return {name: feature for name, feature in zip(ROUTED_SOURCE_NAMES, feature_tuple)}

    def _routing_state(
        self,
        chars: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array | None], mx.array, dict[str, mx.array], dict[str, mx.array], mx.array]:
        base_logits = self.base(chars)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)
        base_features = self._support_gate_features(chars, base_logits)
        lag_weights = self._lag_controller(base_features)
        sources = self._source_dict_from_tuple(self._count_features_with_controller(chars, lag_weights))
        route_features = self._asura_features(base_features, lag_weights)
        source_probs = self._source_router(route_features, sources, base_logits.dtype)
        opener_probs = self._opener_router(route_features, sources, base_logits.dtype)
        residual_scale = self._residual_scale(route_features)
        return base_logits, sources, lag_weights, source_probs, opener_probs, residual_scale

    def controller_snapshot(self, chars: mx.array) -> dict[str, object]:
        base_logits, sources, lag_weights, source_probs, opener_probs, residual_scale = self._routing_state(chars)
        lag_weight_means = np.array(mx.mean(lag_weights, axis=(0, 1)), copy=False)
        global_lag_idx = self._global_lag_bucket_idx()
        global_lag_mean = 0.0 if global_lag_idx is None else float(lag_weight_means[global_lag_idx])
        route_prob_means = {name: float(np.array(mx.mean(value), copy=False)) for name, value in source_probs.items()}
        opener_prob_means = {name: float(np.array(mx.mean(value), copy=False)) for name, value in opener_probs.items()}
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
            "route_prob_means": route_prob_means,
            "opener_prob_means": opener_prob_means,
            "source_mass_means": source_mass_means,
            "bounded_lag_mass_mean": 1.0 - global_lag_mean,
            "global_lag_mass_mean": global_lag_mean,
            "residual_scale_mean": float(np.array(mx.mean(residual_scale), copy=False)),
            "base_logit_abs_mean": float(np.array(mx.mean(mx.abs(base_logits)), copy=False)),
        }

    def _forward_impl(
        self,
        chars: mx.array,
        return_support_activations: bool = False,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        base_logits, sources, _lag_weights, source_probs, opener_probs, residual_scale = self._routing_state(chars)
        base_log_probs = base_logits - mx.logsumexp(base_logits, axis=-1, keepdims=True)
        base_centered = self.config.base_feature_scale * (
            base_log_probs - mx.mean(base_log_probs, axis=-1, keepdims=True)
        )

        pre = mx.zeros_like(base_logits) if self.config.gate_only_mode else mx.broadcast_to(self.bias, base_logits.shape)
        candidate_mask = mx.zeros(base_logits.shape, dtype=base_logits.dtype)
        support_activations: dict[str, mx.array] = {}

        for name, source in sources.items():
            if source is None:
                continue
            weight = getattr(self, f"w_{name}")
            flag_weight = getattr(self, f"w_{name}_flag")
            route_prob = source_probs.get(name, mx.zeros(source.shape[:2] + (1,), dtype=base_logits.dtype))
            opener_prob = opener_probs.get(name, mx.zeros(source.shape[:2] + (1,), dtype=base_logits.dtype))
            term = self._source_term(
                name,
                source,
                None,
                True,
                base_logits.dtype,
                weight,
                flag_weight,
                fixed_gate=route_prob,
            )
            pre = pre + term
            source_hit = (source > 0).astype(base_logits.dtype)
            candidate_strength = self.routing_config.candidate_floor + (1.0 - self.routing_config.candidate_floor) * opener_prob
            candidate_mask = mx.maximum(candidate_mask, candidate_strength * source_hit)
            support_activations[name] = route_prob[..., 0]

        if not self.config.gate_only_mode:
            pre = pre + self.w_base * base_centered
        residual = residual_scale * candidate_mask * (self.config.residual_cap * mx.tanh(pre / self.config.residual_cap))
        return base_logits + residual, (support_activations if return_support_activations else None)


def scale_config(config: AsuraConfig, scale: float) -> AsuraConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        base_config=scale_conker3_config(config.base_config, scale),
    )
