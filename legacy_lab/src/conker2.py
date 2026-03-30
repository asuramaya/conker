from __future__ import annotations

from dataclasses import dataclass, field
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from carving_machine.config import HierarchicalCarverConfig
from carving_machine.models import HierarchicalCarverModel, MLP


def _logspace_half_lives(start: float, end: float, count: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(start), np.log(end), count, dtype=np.float32))


@dataclass(frozen=True)
class ConkerTwoConfig:
    embedding_dim: int = 32
    linear_modes: int = 256
    max_seq_len: int = 256
    linear_half_life_min: float = 1.5
    linear_half_life_max: float = 512.0
    linear_hidden: tuple[int, ...] = (128,)
    mixer_hidden: tuple[int, ...] = (32,)
    mixer_bias_scale: float = 0.25
    mix_mode: str = "learned"
    mix_space: str = "logit"
    use_bias: bool = True
    enable_linear: bool = True
    enable_correction: bool = True
    share_embedding: bool = True
    linear_impl: str = "kernel"
    correction: HierarchicalCarverConfig = field(
        default_factory=lambda: HierarchicalCarverConfig(
            fast_size=128,
            mid_size=256,
            slow_size=512,
            controller_width=128,
            fast_sample_size=64,
            mid_sample_size=48,
            slow_sample_size=48,
            readout_hidden=(128,),
            aux_source="zeros",
        )
    )


class ConkerTwoModel(nn.Module):
    """Frozen linear multiscale substrate plus a tiny frozen nonlinear correction expert."""

    def __init__(self, vocab_size: int, config: ConkerTwoConfig = ConkerTwoConfig()):
        super().__init__()
        if not config.enable_linear and not config.enable_correction:
            raise ValueError("Conker-2 must enable at least one expert path.")
        if config.mix_mode not in {"learned", "equal"}:
            raise ValueError(f"Unknown Conker-2 mix_mode: {config.mix_mode}")
        if config.mix_space not in {"logit", "probability"}:
            raise ValueError(f"Unknown Conker-2 mix_space: {config.mix_space}")
        if config.linear_impl not in {"kernel", "fft"}:
            raise ValueError(f"Unknown Conker-2 linear_impl: {config.linear_impl}")

        self.vocab_size = vocab_size
        self.config = config

        self.linear_embedding = None
        self.linear_in_proj = None
        self.linear_decays = None
        self.linear_kernel = None
        self.linear_readout = None
        self.correction = None
        self.mixer = None
        self.bias_proj = None

        if config.enable_correction:
            self.correction = HierarchicalCarverModel(
                vocab_size=vocab_size,
                embedding_dim=config.embedding_dim,
                config=config.correction,
            )
            self.correction.freeze_static()

        if config.enable_linear:
            if self.correction is None or not config.share_embedding:
                self.linear_embedding = nn.Embedding(vocab_size, config.embedding_dim)

            rng = np.random.default_rng(42)
            in_proj = rng.standard_normal((config.embedding_dim, config.linear_modes), dtype=np.float32)
            in_proj *= 1.0 / math.sqrt(config.embedding_dim)
            self.linear_in_proj = mx.array(in_proj)

            half_lives = _logspace_half_lives(
                config.linear_half_life_min,
                config.linear_half_life_max,
                config.linear_modes,
            )
            decays = np.exp(np.log(0.5, dtype=np.float32) / half_lives)
            self.linear_decays = mx.array(decays.astype(np.float32))

            if config.linear_impl == "kernel":
                time_idx = np.arange(config.max_seq_len, dtype=np.int32)
                delta = time_idx[:, None] - time_idx[None, :]
                mask = delta >= 0
                safe_delta = np.where(mask, delta, 0).astype(np.float32)
                kernel = np.power(decays[None, None, :], safe_delta[..., None], dtype=np.float32)
                kernel = np.where(mask[..., None], kernel, 0.0).astype(np.float32)
                # Store as [modes, T, T] so each mode gets a batched matmul.
                self.linear_kernel = mx.array(np.transpose(kernel, (2, 0, 1)))

            self.linear_readout = MLP(
                config.linear_modes + config.embedding_dim,
                config.linear_hidden,
                vocab_size,
            )

        if config.enable_linear and config.enable_correction:
            if config.mix_mode == "learned":
                self.mixer = MLP(8, config.mixer_hidden, 2)
            if config.use_bias:
                self.bias_proj = nn.Linear(8, vocab_size)

        freeze_keys = [key for key in ("linear_in_proj", "linear_decays", "linear_kernel") if getattr(self, key) is not None]
        if freeze_keys:
            self.freeze(keys=freeze_keys, strict=False)

    @staticmethod
    def _logit_features(logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        max_logit = mx.max(logits, axis=-1)
        centered = logits - mx.mean(logits, axis=-1, keepdims=True)
        variance = mx.mean(centered * centered, axis=-1)
        return entropy, max_logit, variance

    def _embed(self, chars: mx.array) -> mx.array:
        if self.correction is not None and self.config.share_embedding:
            return self.correction.embedding(chars)
        if self.linear_embedding is None:
            raise RuntimeError("Conker-2 linear path has no embedding table.")
        return self.linear_embedding(chars)

    def _linear_logits(self, chars: mx.array) -> mx.array:
        _, timesteps = chars.shape
        if timesteps > self.config.max_seq_len:
            raise ValueError(
                f"Conker-2 max_seq_len={self.config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.linear_in_proj is None or self.linear_readout is None:
            raise RuntimeError("Conker-2 linear path is disabled.")
        x = self._embed(chars)
        drive = mx.matmul(x, self.linear_in_proj)
        if self.config.linear_impl == "kernel":
            if self.linear_kernel is None:
                raise RuntimeError("Conker-2 kernel path is missing its materialized kernel.")
            kernels = self.linear_kernel[:, :timesteps, :timesteps]
            drive_mb = mx.transpose(drive, (2, 0, 1))
            states_mb = mx.matmul(drive_mb, mx.transpose(kernels, (0, 2, 1)))
            states = mx.transpose(states_mb, (1, 2, 0))
        else:
            states = self._linear_states_fft(drive, timesteps)
        return self.linear_readout(mx.concatenate([states, x], axis=-1))

    def _linear_states_fft(self, drive: mx.array, timesteps: int) -> mx.array:
        if self.linear_decays is None:
            raise RuntimeError("Conker-2 FFT path is missing linear decays.")
        drive_mb = mx.transpose(drive, (0, 2, 1))
        n_fft = 1 << int(math.ceil(math.log2(max(2 * timesteps - 1, 1))))
        time = mx.arange(timesteps, dtype=drive.dtype)
        kernel = mx.power(self.linear_decays[:, None], time[None, :])
        drive_f = mx.fft.rfft(drive_mb, n=n_fft, axis=-1)
        kernel_f = mx.fft.rfft(kernel[None, :, :], n=n_fft, axis=-1)
        states_mb = mx.fft.irfft(drive_f * kernel_f, n=n_fft, axis=-1)[..., :timesteps]
        return mx.transpose(states_mb, (0, 2, 1))

    def __call__(self, chars: mx.array) -> mx.array:
        logits_linear = self._linear_logits(chars) if self.config.enable_linear else None
        logits_corr = self.correction(chars) if self.correction is not None else None

        if logits_linear is None:
            return logits_corr
        if logits_corr is None:
            return logits_linear

        ent_l, max_l, var_l = self._logit_features(logits_linear)
        ent_c, max_c, var_c = self._logit_features(logits_corr)
        features = mx.stack(
            [
                ent_l,
                ent_c,
                max_l,
                max_c,
                var_l,
                var_c,
                ent_l - ent_c,
                max_l - max_c,
            ],
            axis=-1,
        )

        if self.mixer is None:
            mix = mx.full((*features.shape[:-1], 2), 0.5, dtype=features.dtype)
        else:
            mix = mx.softmax(self.mixer(features), axis=-1)

        if self.bias_proj is None:
            residual_bias = mx.zeros_like(logits_linear)
        else:
            residual_bias = self.config.mixer_bias_scale * self.bias_proj(features)

        if self.config.mix_space == "logit":
            return mix[..., 0:1] * logits_linear + mix[..., 1:2] * logits_corr + residual_bias

        probs = mix[..., 0:1] * mx.softmax(logits_linear, axis=-1) + mix[..., 1:2] * mx.softmax(logits_corr, axis=-1)
        return mx.log(mx.maximum(probs, 1e-8)) + residual_bias
