from __future__ import annotations

from dataclasses import dataclass, field, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from conker.src.conker4b import ConkerFourBConfig, ConkerFourBModel, scale_config as scale_conker4b_config


def _teacher_mask(length: int, mode: str) -> np.ndarray:
    if mode == "future":
        return np.triu(np.ones((length, length), dtype=np.float32), k=1)
    if mode == "bidirectional":
        mask = np.ones((length, length), dtype=np.float32)
        np.fill_diagonal(mask, 0.0)
        return mask
    raise ValueError(f"Unknown Conker-7 teacher_mask_mode: {mode}")


@dataclass
class ConkerSevenConfig:
    student_config: ConkerFourBConfig = field(default_factory=ConkerFourBConfig)
    teacher_mask_mode: str = "future"
    teacher_weight: float = 0.5
    teacher_start_step: int = 0
    teacher_enable_exact2: bool = True
    teacher_enable_exact3: bool = True
    teacher_enable_special2: bool = False
    teacher_enable_number2: bool = False
    teacher_enable_markup2: bool = False
    teacher_enable_attr2: bool = False
    teacher_enable_delim2: bool = False
    teacher_exact2_weight: float = 1.0
    teacher_exact3_weight: float = 2.0
    teacher_special2_weight: float = 1.0
    teacher_number2_weight: float = 1.0
    teacher_markup2_weight: float = 1.0
    teacher_attr2_weight: float = 1.0
    teacher_delim2_weight: float = 0.5


class ConkerSevenModel(nn.Module):
    """Legal causal Conker-4b student with future-aware training-time distillation."""

    def __init__(self, vocab_size: int, config: ConkerSevenConfig = ConkerSevenConfig()):
        super().__init__()
        if config.teacher_mask_mode not in {"future", "bidirectional"}:
            raise ValueError(f"Unknown Conker-7 teacher_mask_mode: {config.teacher_mask_mode}")
        self.vocab_size = vocab_size
        self.config = config
        self.student = ConkerFourBModel(vocab_size=vocab_size, config=config.student_config)
        self.train_step = mx.array(0.0, dtype=mx.float32)
        self.freeze(keys=("train_step",), strict=False)

    def __call__(self, chars: mx.array) -> mx.array:
        return self.student(chars)

    def set_train_step(self, step: int) -> None:
        self.train_step = mx.array(float(step), dtype=mx.float32)

    def _teacher_probs(self, x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
        teacher_context = mx.concatenate([x, y[:, -1:]], axis=1)
        mask = mx.array(_teacher_mask(teacher_context.shape[1], self.config.teacher_mask_mode))
        (
            _exact1,
            exact2,
            exact3,
            special2,
            number2,
            _urlpath2,
            markup2,
            attr2,
            _entity2,
            _stack2,
            _wordclass2,
            delim2,
            _delimsub2,
            _recency,
        ) = self.student._count_features_core(teacher_context, mask)

        teacher_counts = mx.zeros((x.shape[0], x.shape[1], self.vocab_size), dtype=mx.float32)
        if self.config.teacher_enable_exact2 and exact2 is not None:
            teacher_counts = teacher_counts + self.config.teacher_exact2_weight * exact2[:, :-1, :]
        if self.config.teacher_enable_exact3 and exact3 is not None:
            teacher_counts = teacher_counts + self.config.teacher_exact3_weight * exact3[:, :-1, :]
        if self.config.teacher_enable_special2 and special2 is not None:
            teacher_counts = teacher_counts + self.config.teacher_special2_weight * special2[:, :-1, :]
        if self.config.teacher_enable_number2 and number2 is not None:
            teacher_counts = teacher_counts + self.config.teacher_number2_weight * number2[:, :-1, :]
        if self.config.teacher_enable_markup2 and markup2 is not None:
            teacher_counts = teacher_counts + self.config.teacher_markup2_weight * markup2[:, :-1, :]
        if self.config.teacher_enable_attr2 and attr2 is not None:
            teacher_counts = teacher_counts + self.config.teacher_attr2_weight * attr2[:, :-1, :]
        if self.config.teacher_enable_delim2 and delim2 is not None:
            teacher_counts = teacher_counts + self.config.teacher_delim2_weight * delim2[:, :-1, :]

        teacher_mass = mx.sum(teacher_counts, axis=-1)
        teacher_probs = teacher_counts / mx.maximum(teacher_mass[..., None], mx.array(1e-8, dtype=teacher_counts.dtype))
        return mx.stop_gradient(teacher_probs), mx.stop_gradient(teacher_mass)

    def supervised_loss(self, x: mx.array, y: mx.array) -> mx.array:
        logits, support_activations = self.student._forward_impl(x, return_support_activations=True)
        batch_size, timesteps, vocab_size = logits.shape
        ce_loss = mx.mean(
            nn.losses.cross_entropy(
                logits.reshape(batch_size * timesteps, vocab_size),
                y.reshape(batch_size * timesteps),
            )
        )

        loss = ce_loss
        if self.config.teacher_weight > 0.0 and float(np.array(self.train_step)) >= self.config.teacher_start_step:
            teacher_probs, teacher_mass = self._teacher_probs(x, y)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            teacher_ce = -mx.sum(teacher_probs * log_probs, axis=-1)
            active = (teacher_mass > 0).astype(logits.dtype)
            active_count = mx.sum(active)
            distill = mx.sum(teacher_ce * active) / mx.maximum(active_count, mx.array(1.0, dtype=logits.dtype))
            loss = loss + self.config.teacher_weight * distill

        if (
            self.student.config.support_overlap_penalty > 0.0
            and support_activations is not None
            and len(support_activations) >= 2
        ):
            acts = mx.stack(list(support_activations.values()), axis=-1)
            sum_acts = mx.sum(acts, axis=-1)
            pairwise = 0.5 * (sum_acts * sum_acts - mx.sum(acts * acts, axis=-1))
            denom = max((acts.shape[-1] * (acts.shape[-1] - 1)) / 2.0, 1.0)
            loss = loss + self.student.config.support_overlap_penalty * mx.mean(pairwise / denom)
        return loss


def scale_config(config: ConkerSevenConfig, scale: float) -> ConkerSevenConfig:
    if scale == 1.0:
        return config
    return replace(config, student_config=scale_conker4b_config(config.student_config, scale))
