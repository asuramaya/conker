from __future__ import annotations

from dataclasses import dataclass, field, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from conker.src.conker3 import ConkerThreeConfig, ConkerThreeModel, scale_config as scale_conker3_config


def _strict_causal_mask(max_seq_len: int) -> np.ndarray:
    return np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.float32), k=-1)


def _lookback_causal_mask(length: int, lookback: int) -> np.ndarray:
    if lookback <= 0:
        return _strict_causal_mask(length)
    time_idx = np.arange(length, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    return ((delta > 0) & (delta <= lookback)).astype(np.float32)


@dataclass
class ConkerSixConfig:
    base_config: ConkerThreeConfig = field(default_factory=ConkerThreeConfig)
    freeze_base: bool = False
    enable_exact3: bool = True
    exact_context_span: int = 0
    learnable_vocab_axis: bool = True
    learnable_causal_mask: bool = True
    causal_projection: str = "none"
    blend_mode: str = "learned_gate"
    gate_hidden_dim: int = 32
    gate_temperature: float = 1.0
    unigram_discount: float = 0.0
    exact1_discount: float = 0.75
    exact2_discount: float = 0.75
    exact3_discount: float = 0.75
    fixed_base_weight: float = 0.15
    fixed_exact1_weight: float = 0.10
    fixed_exact2_weight: float = 0.25
    fixed_exact3_weight: float = 0.50


class ConkerSixModel(nn.Module):
    """Normalized causal cache with Conker-3 as learned smoother/gate."""

    def __init__(self, vocab_size: int, config: ConkerSixConfig = ConkerSixConfig()):
        super().__init__()
        if config.blend_mode not in {"cache_only", "fixed_blend", "learned_gate", "witten_bell", "absolute_discount"}:
            raise ValueError(f"Unknown Conker-6 blend_mode: {config.blend_mode}")
        if config.causal_projection not in {"none", "strict_lower", "strict_lower_nonnegative"}:
            raise ValueError(f"Unknown Conker-6 causal_projection: {config.causal_projection}")
        self.vocab_size = vocab_size
        self.config = config
        self.base = ConkerThreeModel(vocab_size=vocab_size, config=config.base_config)
        if config.freeze_base:
            self.base.freeze(recurse=True)

        self.vocab_axis = mx.arange(vocab_size, dtype=mx.int32)
        self.causal_mask = mx.array(_strict_causal_mask(config.base_config.max_seq_len))

        gate_feature_dim = 12
        if config.blend_mode == "learned_gate":
            self.gate_hidden = nn.Linear(gate_feature_dim, config.gate_hidden_dim)
            self.gate_out = nn.Linear(config.gate_hidden_dim, 4)
        else:
            self.gate_hidden = None
            self.gate_out = None
        self.freeze(keys=("vocab_axis", "causal_mask"), strict=False)

    def _one_hot(self, chars: mx.array) -> mx.array:
        vocab_axis = self.vocab_axis if self.config.learnable_vocab_axis else mx.stop_gradient(self.vocab_axis)
        return mx.where(chars[..., None] == vocab_axis[None, None, :], 1.0, 0.0)

    def _cache_counts_core(
        self,
        chars: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array | None]:
        batch, timesteps = chars.shape
        one_hot = self._one_hot(chars)
        zeros = mx.zeros((batch, 1, self.vocab_size), dtype=one_hot.dtype)
        next_one_hot = mx.concatenate([one_hot[:, 1:, :], zeros], axis=1)
        prev_one_hot = mx.concatenate([zeros, one_hot[:, :-1, :]], axis=1)
        unigram = mx.matmul(mask[None, :, :], next_one_hot)

        cur_match = mx.matmul(one_hot, mx.transpose(one_hot, (0, 2, 1))) * mask[None, :, :]
        exact1 = mx.matmul(cur_match, next_one_hot)

        prev_match = mx.matmul(prev_one_hot, mx.transpose(prev_one_hot, (0, 2, 1)))
        pair_match = cur_match * prev_match
        exact2 = mx.matmul(pair_match, next_one_hot)

        exact3 = None
        if self.config.enable_exact3:
            prev2_one_hot = mx.concatenate([mx.zeros((batch, 2, self.vocab_size), dtype=one_hot.dtype), one_hot[:, :-2, :]], axis=1)
            prev2_match = mx.matmul(prev2_one_hot, mx.transpose(prev2_one_hot, (0, 2, 1)))
            exact3 = mx.matmul(pair_match * prev2_match, next_one_hot)
        return unigram, exact1, exact2, exact3

    def _cache_counts(
        self,
        chars: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array | None]:
        batch, timesteps = chars.shape
        if timesteps > self.config.base_config.max_seq_len:
            raise ValueError(
                f"Conker-6 max_seq_len={self.config.base_config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.config.exact_context_span <= 0:
            base_mask = self.causal_mask if self.config.learnable_causal_mask else mx.stop_gradient(self.causal_mask)
            mask = base_mask[:timesteps, :timesteps]
            if self.config.causal_projection != "none":
                support = mx.array(_strict_causal_mask(timesteps), dtype=mask.dtype)
                if self.config.causal_projection == "strict_lower_nonnegative":
                    mask = mx.maximum(mask, mx.array(0.0, dtype=mask.dtype))
                mask = mask * support
            return self._cache_counts_core(chars, mask)

        total_timesteps = batch * timesteps
        flat_chars = chars.reshape(1, total_timesteps)
        flat_mask = mx.array(_lookback_causal_mask(total_timesteps, self.config.exact_context_span))
        unigram, exact1, exact2, exact3 = self._cache_counts_core(flat_chars, flat_mask)
        unigram = unigram.reshape(batch, timesteps, self.vocab_size)
        exact1 = exact1.reshape(batch, timesteps, self.vocab_size)
        exact2 = exact2.reshape(batch, timesteps, self.vocab_size)
        if exact3 is not None:
            exact3 = exact3.reshape(batch, timesteps, self.vocab_size)
        return unigram, exact1, exact2, exact3

    @staticmethod
    def _normalize_counts(counts: mx.array | None) -> tuple[mx.array | None, mx.array | None]:
        if counts is None:
            return None, None
        mass = mx.sum(counts, axis=-1, keepdims=True)
        probs = counts / mx.maximum(mass, mx.array(1e-8, dtype=counts.dtype))
        return probs, mass

    def _gate_features(
        self,
        chars: mx.array,
        base_logits: mx.array,
        mass1: mx.array,
        mass2: mx.array,
        mass3: mx.array | None,
    ) -> mx.array:
        entropy, max_logit, variance = self.base._logit_features(base_logits)
        zeros = mx.zeros_like(entropy)
        if self.base.config.enable_linear:
            states, _ = self.base._linear_states(chars)
            abs_states = mx.abs(states)
            total_energy = mx.mean(abs_states, axis=-1)
            if self.base.non_osc_modes > 0:
                non_osc_energy = mx.mean(abs_states[:, :, : self.base.non_osc_modes], axis=-1)
            else:
                non_osc_energy = zeros
            if self.base.osc_mode_count > 0:
                osc_energy = mx.mean(abs_states[:, :, self.base.non_osc_modes :], axis=-1)
            else:
                osc_energy = zeros
        else:
            total_energy = zeros
            non_osc_energy = zeros
            osc_energy = zeros
        mass3_safe = mass3 if mass3 is not None else mx.zeros_like(mass1)
        return mx.stack(
            [
                entropy,
                max_logit,
                variance,
                total_energy,
                non_osc_energy,
                osc_energy,
                mx.squeeze(mx.log1p(mass1), axis=-1),
                mx.squeeze(mx.log1p(mass2), axis=-1),
                mx.squeeze(mx.log1p(mass3_safe), axis=-1),
                mx.squeeze((mass1 > 0).astype(base_logits.dtype), axis=-1),
                mx.squeeze((mass2 > 0).astype(base_logits.dtype), axis=-1),
                mx.squeeze((mass3_safe > 0).astype(base_logits.dtype), axis=-1),
            ],
            axis=-1,
        )

    def _fixed_component_weights(
        self,
        base_probs: mx.array,
        mass1: mx.array,
        mass2: mx.array,
        mass3: mx.array | None,
    ) -> mx.array:
        dtype = base_probs.dtype
        hit1 = (mass1 > 0).astype(dtype)
        hit2 = (mass2 > 0).astype(dtype)
        if mass3 is not None:
            hit3 = (mass3 > 0).astype(dtype)
        else:
            hit3 = mx.zeros_like(hit2)
        weights = mx.concatenate(
            [
                mx.ones_like(hit1) * self.config.fixed_base_weight,
                hit1 * self.config.fixed_exact1_weight,
                hit2 * self.config.fixed_exact2_weight,
                hit3 * self.config.fixed_exact3_weight,
            ],
            axis=-1,
        )
        denom = mx.maximum(mx.sum(weights, axis=-1, keepdims=True), mx.array(1e-8, dtype=dtype))
        return weights / denom

    def _cache_only_weights(
        self,
        base_probs: mx.array,
        mass1: mx.array,
        mass2: mx.array,
        mass3: mx.array | None,
    ) -> mx.array:
        dtype = base_probs.dtype
        hit1 = (mass1 > 0).astype(dtype)
        hit2 = (mass2 > 0).astype(dtype)
        if mass3 is not None:
            hit3 = (mass3 > 0).astype(dtype)
        else:
            hit3 = mx.zeros_like(hit2)
        base_w = 1.0 - mx.maximum(hit1, mx.maximum(hit2, hit3))
        remaining = 1.0 - base_w
        cache_weights = mx.concatenate(
            [
                mx.zeros_like(base_w),
                remaining * (1.0 - hit2) * hit1,
                remaining * hit2 * (1.0 - hit3),
                remaining * hit3,
            ],
            axis=-1,
        )
        return mx.concatenate([base_w, cache_weights[..., 1:]], axis=-1)

    def _component_weights(
        self,
        chars: mx.array,
        base_logits: mx.array,
        mass1: mx.array,
        mass2: mx.array,
        mass3: mx.array | None,
    ) -> mx.array:
        if self.config.blend_mode == "cache_only":
            return self._cache_only_weights(base_logits, mass1, mass2, mass3)
        if self.config.blend_mode == "fixed_blend":
            return self._fixed_component_weights(base_logits, mass1, mass2, mass3)
        if self.config.blend_mode in {"witten_bell", "absolute_discount"}:
            raise RuntimeError("Discounted backoff modes do not use learned component weights.")

        features = self._gate_features(chars, base_logits, mass1, mass2, mass3)
        hidden = mx.tanh(self.gate_hidden(features))
        logits = self.gate_out(hidden)
        unavailable = [
            mx.zeros(mass1.shape, dtype=mx.bool_),
            mass1 <= 0,
            mass2 <= 0,
            mx.ones(mass2.shape, dtype=mx.bool_) if mass3 is None else (mass3 <= 0),
        ]
        masked_logits = []
        neg_inf = mx.array(-1e9, dtype=base_logits.dtype)
        for idx, mask in enumerate(unavailable):
            component_logits = logits[:, :, idx : idx + 1] / max(self.config.gate_temperature, 1e-4)
            masked_logits.append(mx.where(mask, neg_inf, component_logits))
        return mx.softmax(mx.concatenate(masked_logits, axis=-1), axis=-1)

    @staticmethod
    def _witten_bell_backoff(counts: mx.array, lower_probs: mx.array) -> mx.array:
        total = mx.sum(counts, axis=-1, keepdims=True)
        distinct = mx.sum((counts > 0).astype(counts.dtype), axis=-1, keepdims=True)
        lambda_hist = total / mx.maximum(total + distinct, mx.array(1e-8, dtype=counts.dtype))
        mle = counts / mx.maximum(total, mx.array(1e-8, dtype=counts.dtype))
        return lambda_hist * mle + (1.0 - lambda_hist) * lower_probs

    @staticmethod
    def _absolute_discount_backoff(counts: mx.array, lower_probs: mx.array, discount: float) -> mx.array:
        total = mx.sum(counts, axis=-1, keepdims=True)
        distinct = mx.sum((counts > 0).astype(counts.dtype), axis=-1, keepdims=True)
        discount_arr = mx.array(discount, dtype=counts.dtype)
        discounted = mx.maximum(counts - discount_arr, mx.array(0.0, dtype=counts.dtype))
        mle = discounted / mx.maximum(total, mx.array(1e-8, dtype=counts.dtype))
        backoff_mass = (discount_arr * distinct) / mx.maximum(total, mx.array(1e-8, dtype=counts.dtype))
        return mle + backoff_mass * lower_probs

    def _discounted_cache_probs(
        self,
        base_probs: mx.array,
        unigram: mx.array,
        exact1: mx.array,
        exact2: mx.array,
        exact3: mx.array | None,
    ) -> mx.array:
        uniform = mx.ones_like(base_probs) * (1.0 / self.vocab_size)
        if self.config.blend_mode == "witten_bell":
            p_uni = self._witten_bell_backoff(unigram, uniform)
            p1 = self._witten_bell_backoff(exact1, p_uni)
            p2 = self._witten_bell_backoff(exact2, p1)
            return self._witten_bell_backoff(exact3, p2) if exact3 is not None else p2
        p_uni = self._absolute_discount_backoff(unigram, uniform, self.config.unigram_discount)
        p1 = self._absolute_discount_backoff(exact1, p_uni, self.config.exact1_discount)
        p2 = self._absolute_discount_backoff(exact2, p1, self.config.exact2_discount)
        return self._absolute_discount_backoff(exact3, p2, self.config.exact3_discount) if exact3 is not None else p2

    def __call__(self, chars: mx.array) -> mx.array:
        base_logits = self.base(chars)
        if self.config.freeze_base:
            base_logits = mx.stop_gradient(base_logits)
        base_probs = mx.softmax(base_logits, axis=-1)

        unigram, exact1, exact2, exact3 = self._cache_counts(chars)
        if self.config.blend_mode in {"witten_bell", "absolute_discount"}:
            mixed = self._discounted_cache_probs(base_probs, unigram, exact1, exact2, exact3)
            mixed = mixed / mx.maximum(mx.sum(mixed, axis=-1, keepdims=True), mx.array(1e-8, dtype=mixed.dtype))
            return mx.log(mx.maximum(mixed, mx.array(1e-8, dtype=mixed.dtype)))
        probs1, mass1 = self._normalize_counts(exact1)
        probs2, mass2 = self._normalize_counts(exact2)
        probs3, mass3 = self._normalize_counts(exact3)
        if self.config.blend_mode == "cache_only":
            base_probs = mx.ones_like(base_probs) * (1.0 / self.vocab_size)

        weights = self._component_weights(chars, base_logits, mass1, mass2, mass3)
        components = [base_probs, probs1, probs2]
        if probs3 is not None:
            components.append(probs3)
        else:
            components.append(mx.zeros_like(base_probs))
        mixed = mx.zeros_like(base_probs)
        for idx, probs in enumerate(components):
            mixed = mixed + weights[:, :, idx : idx + 1] * probs
        mixed = mixed / mx.maximum(mx.sum(mixed, axis=-1, keepdims=True), mx.array(1e-8, dtype=mixed.dtype))
        return mx.log(mx.maximum(mixed, mx.array(1e-8, dtype=mixed.dtype)))


def scale_config(config: ConkerSixConfig, scale: float) -> ConkerSixConfig:
    if scale == 1.0:
        return config
    return replace(config, base_config=scale_conker3_config(config.base_config, scale))
