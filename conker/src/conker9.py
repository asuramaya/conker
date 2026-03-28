from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from conker.src.conker3 import scale_config as scale_conker3_config
from conker.src.conker4b import ConkerFourBConfig, ConkerFourBModel, _lookback_causal_mask


@dataclass(frozen=True)
class ConkerNineConfig(ConkerFourBConfig):
    lag_lookbacks: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 0)
    lag_controller_temperature: float = 1.0


class ConkerNineModel(ConkerFourBModel):
    """Strict Conker-4b with a causal controller over fixed legal lag buckets."""

    def __init__(self, vocab_size: int, config: ConkerNineConfig = ConkerNineConfig()):
        super().__init__(vocab_size=vocab_size, config=config)
        bucket_count = len(config.lag_lookbacks)
        if bucket_count <= 0:
            raise ValueError("Conker-9 requires at least one lag bucket.")
        self.lag_gate_feature_weights = mx.zeros((bucket_count, 6), dtype=mx.float32)
        self.lag_gate_bias = mx.zeros((bucket_count,), dtype=mx.float32)

    @property
    def controller_config(self) -> ConkerNineConfig:
        return self.config

    def _lag_bucket_masks(self, timesteps: int) -> list[mx.array]:
        masks: list[mx.array] = []
        for lookback in self.controller_config.lag_lookbacks:
            if lookback <= 0:
                masks.append(self.causal_mask[:timesteps, :timesteps])
            else:
                masks.append(mx.array(_lookback_causal_mask(timesteps, int(lookback))))
        return masks

    def _lag_controller(self, base_features: mx.array) -> mx.array:
        logits = (
            mx.sum(base_features[..., None, :] * self.lag_gate_feature_weights[None, None, :, :], axis=-1)
            + self.lag_gate_bias[None, None, :]
        )
        temperature = max(self.controller_config.lag_controller_temperature, 1e-4)
        return mx.softmax(logits / temperature, axis=-1)

    def _blend_feature_bank(
        self,
        feature_bank: list[tuple[mx.array | None, ...]],
        lag_weights: mx.array,
    ) -> tuple[mx.array | None, ...]:
        blended: list[mx.array | None] = []
        for feature_idx in range(len(feature_bank[0])):
            example = feature_bank[0][feature_idx]
            if example is None:
                blended.append(None)
                continue
            stacked = mx.stack([bucket[feature_idx] for bucket in feature_bank], axis=-1)
            blended_feature = mx.sum(stacked * lag_weights[..., None, :], axis=-1)
            blended.append(blended_feature)
        return tuple(blended)

    def _count_features_with_controller(
        self,
        chars: mx.array,
        lag_weights: mx.array,
    ) -> tuple[mx.array | None, ...]:
        _, timesteps = chars.shape
        masks = self._lag_bucket_masks(timesteps)
        feature_bank = [ConkerFourBModel._count_features_core(self, chars, mask) for mask in masks]
        return self._blend_feature_bank(feature_bank, lag_weights)

    def _forward_impl(
        self,
        chars: mx.array,
        return_support_activations: bool = False,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        base_logits = self.base(chars)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)

        base_features = self._support_gate_features(chars, base_logits)
        lag_weights = self._lag_controller(base_features)
        (
            exact1,
            exact2,
            exact3,
            special2,
            number2,
            urlpath2,
            markup2,
            attr2,
            entity2,
            stack2,
            wordclass2,
            delim2,
            delimsub2,
            recency,
        ) = self._count_features_with_controller(chars, lag_weights)

        base_log_probs = base_logits - mx.logsumexp(base_logits, axis=-1, keepdims=True)
        base_centered = self.config.base_feature_scale * (
            base_log_probs - mx.mean(base_log_probs, axis=-1, keepdims=True)
        )
        support_gate_features = self._support_gate_features(chars, base_logits) if self.config.dynamic_support_gates else None
        support_ownership_gates = self._support_ownership_gates(
            {
                "exact1": exact1,
                "special2": special2,
                "number2": number2,
                "urlpath2": urlpath2,
                "markup2": markup2,
                "attr2": attr2,
                "entity2": entity2,
                "stack2": stack2,
                "wordclass2": wordclass2,
                "delim2": delim2,
                "delimsub2": delimsub2,
            },
            support_gate_features,
            base_logits.dtype,
        )
        support_activations: dict[str, mx.array] = {}
        support_independent_gates: dict[str, mx.array] = {}
        if self.config.support_gate_mode == "independent":
            for source_name, source in (
                ("exact1", exact1),
                ("special2", special2),
                ("number2", number2),
                ("urlpath2", urlpath2),
                ("markup2", markup2),
                ("attr2", attr2),
                ("entity2", entity2),
                ("stack2", stack2),
                ("wordclass2", wordclass2),
                ("delim2", delim2),
                ("delimsub2", delimsub2),
            ):
                gate, activation = self._independent_support_gate(
                    source_name,
                    source,
                    support_gate_features,
                    base_logits.dtype,
                )
                if gate is not None and activation is not None:
                    support_independent_gates[source_name] = gate
                    support_activations[source_name] = activation

        pre = mx.zeros_like(base_logits) if self.config.gate_only_mode else mx.broadcast_to(self.bias, base_logits.shape)
        candidate_mask = mx.zeros(base_logits.shape, dtype=base_logits.dtype)

        if exact1 is not None:
            exact1_term = self._source_term(
                "exact1",
                exact1,
                support_gate_features,
                self.config.exact1_opens_mask,
                base_logits.dtype,
                self.w_exact1,
                self.w_exact1_flag,
                fixed_gate=support_ownership_gates.get("exact1") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("exact1"),
            )
            pre = pre + exact1_term
            if self.config.exact1_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (exact1 > 0).astype(base_logits.dtype))

        if exact2 is not None:
            exact2_term = self._source_term(
                "exact2", exact2, support_gate_features, True, base_logits.dtype, self.w_exact2, self.w_exact2_flag
            )
            pre = pre + exact2_term
            candidate_mask = mx.maximum(candidate_mask, (exact2 > 0).astype(base_logits.dtype))

        if exact3 is not None:
            exact3_term = self._source_term(
                "exact3", exact3, support_gate_features, True, base_logits.dtype, self.w_exact3, self.w_exact3_flag
            )
            pre = pre + exact3_term
            candidate_mask = mx.maximum(candidate_mask, (exact3 > 0).astype(base_logits.dtype))

        if special2 is not None:
            special2_term = self._source_term(
                "special2",
                special2,
                support_gate_features,
                self.config.special2_opens_mask,
                base_logits.dtype,
                self.w_special2,
                self.w_special2_flag,
                fixed_gate=support_ownership_gates.get("special2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("special2"),
            )
            pre = pre + special2_term
            if self.config.special2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (special2 > 0).astype(base_logits.dtype))

        if number2 is not None:
            number2_term = self._source_term(
                "number2",
                number2,
                support_gate_features,
                self.config.number2_opens_mask,
                base_logits.dtype,
                self.w_number2,
                self.w_number2_flag,
                fixed_gate=support_ownership_gates.get("number2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("number2"),
            )
            pre = pre + number2_term
            if self.config.number2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (number2 > 0).astype(base_logits.dtype))

        if urlpath2 is not None:
            urlpath2_term = self._source_term(
                "urlpath2",
                urlpath2,
                support_gate_features,
                self.config.urlpath2_opens_mask,
                base_logits.dtype,
                self.w_urlpath2,
                self.w_urlpath2_flag,
                fixed_gate=support_ownership_gates.get("urlpath2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("urlpath2"),
            )
            pre = pre + urlpath2_term
            if self.config.urlpath2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (urlpath2 > 0).astype(base_logits.dtype))

        if markup2 is not None:
            markup2_term = self._source_term(
                "markup2",
                markup2,
                support_gate_features,
                self.config.markup2_opens_mask,
                base_logits.dtype,
                self.w_markup2,
                self.w_markup2_flag,
                fixed_gate=support_ownership_gates.get("markup2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("markup2"),
            )
            pre = pre + markup2_term
            if self.config.markup2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (markup2 > 0).astype(base_logits.dtype))

        if attr2 is not None:
            attr2_term = self._source_term(
                "attr2",
                attr2,
                support_gate_features,
                self.config.attr2_opens_mask,
                base_logits.dtype,
                self.w_attr2,
                self.w_attr2_flag,
                fixed_gate=support_ownership_gates.get("attr2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("attr2"),
            )
            pre = pre + attr2_term
            if self.config.attr2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (attr2 > 0).astype(base_logits.dtype))

        if entity2 is not None:
            entity2_term = self._source_term(
                "entity2",
                entity2,
                support_gate_features,
                self.config.entity2_opens_mask,
                base_logits.dtype,
                self.w_entity2,
                self.w_entity2_flag,
                fixed_gate=support_ownership_gates.get("entity2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("entity2"),
            )
            pre = pre + entity2_term
            if self.config.entity2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (entity2 > 0).astype(base_logits.dtype))

        if stack2 is not None:
            stack2_term = self._source_term(
                "stack2",
                stack2,
                support_gate_features,
                self.config.stack2_opens_mask,
                base_logits.dtype,
                self.w_stack2,
                self.w_stack2_flag,
                fixed_gate=support_ownership_gates.get("stack2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("stack2"),
            )
            pre = pre + stack2_term
            if self.config.stack2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (stack2 > 0).astype(base_logits.dtype))

        if wordclass2 is not None:
            wordclass2_term = self._source_term(
                "wordclass2",
                wordclass2,
                support_gate_features,
                self.config.wordclass2_opens_mask,
                base_logits.dtype,
                self.w_wordclass2,
                self.w_wordclass2_flag,
                fixed_gate=support_ownership_gates.get("wordclass2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("wordclass2"),
            )
            pre = pre + wordclass2_term
            if self.config.wordclass2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (wordclass2 > 0).astype(base_logits.dtype))

        if delim2 is not None:
            delim2_term = self._source_term(
                "delim2",
                delim2,
                support_gate_features,
                self.config.delim2_opens_mask,
                base_logits.dtype,
                self.w_delim2,
                self.w_delim2_flag,
                fixed_gate=support_ownership_gates.get("delim2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("delim2"),
            )
            pre = pre + delim2_term
            if self.config.delim2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (delim2 > 0).astype(base_logits.dtype))

        if delimsub2 is not None:
            delimsub2_term = self._source_term(
                "delimsub2",
                delimsub2,
                support_gate_features,
                self.config.delimsub2_opens_mask,
                base_logits.dtype,
                self.w_delimsub2,
                self.w_delimsub2_flag,
                fixed_gate=support_ownership_gates.get("delimsub2") if self.config.support_gate_mode == "softmax" else support_independent_gates.get("delimsub2"),
            )
            pre = pre + delimsub2_term
            if self.config.delimsub2_opens_mask:
                candidate_mask = mx.maximum(candidate_mask, (delimsub2 > 0).astype(base_logits.dtype))

        if recency is not None:
            recency_term = self._source_term(
                "recency", recency, support_gate_features, True, base_logits.dtype, self.w_recency, self.w_recency_flag
            )
            pre = pre + recency_term
            candidate_mask = mx.maximum(candidate_mask, (recency > 0).astype(base_logits.dtype))

        if not self.config.gate_only_mode:
            pre = pre + self.w_base * base_centered
        residual = candidate_mask * (self.config.residual_cap * mx.tanh(pre / self.config.residual_cap))
        return base_logits + residual, (support_activations if return_support_activations else None)


def scale_config(config: ConkerNineConfig, scale: float) -> ConkerNineConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        base_config=scale_conker3_config(config.base_config, scale),
    )
