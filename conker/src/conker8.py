from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from conker.src.conker3 import scale_config as scale_conker3_config
from conker.src.conker4b import ConkerFourBConfig, ConkerFourBModel, _lookback_causal_mask


@dataclass(frozen=True)
class ConkerEightConfig(ConkerFourBConfig):
    learn_lag_profile: bool = True
    lag_profile_span: float = 0.5
    learn_delimiter_mask: bool = True
    learn_number_mask: bool = True
    learn_special_mask: bool = True
    learn_urlpath_mask: bool = True
    learn_markup_mask: bool = True
    learn_attr_mask: bool = True
    learn_entity_mask: bool = True
    support_mask_span: float = 0.5


class ConkerEightModel(ConkerFourBModel):
    """Strict Conker-4b with explicit learned causal/profile structure."""

    def __init__(self, vocab_size: int, config: ConkerEightConfig = ConkerEightConfig()):
        super().__init__(vocab_size=vocab_size, config=config)
        self.causal_lag_logits = mx.zeros((config.base_config.max_seq_len,), dtype=mx.float32)
        self.delimiter_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)
        self.number_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)
        self.special_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)
        self.urlpath_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)
        self.markup_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)
        self.attr_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)
        self.entity_mask_logits = mx.zeros((vocab_size,), dtype=mx.float32)

    @property
    def structure_config(self) -> ConkerEightConfig:
        return self.config

    def _weighted_binary_mask(
        self,
        base_mask: mx.array,
        logits: mx.array,
        enabled: bool,
    ) -> mx.array:
        if not enabled:
            return base_mask
        scale = 1.0 + self.structure_config.support_mask_span * mx.tanh(logits)
        return base_mask * scale.astype(base_mask.dtype)

    def _weighted_support_masks(self) -> dict[str, mx.array]:
        return {
            "delimiter_mask": self._weighted_binary_mask(
                self.delimiter_mask,
                self.delimiter_mask_logits,
                self.structure_config.learn_delimiter_mask,
            ),
            "number_mask": self._weighted_binary_mask(
                self.number_mask,
                self.number_mask_logits,
                self.structure_config.learn_number_mask,
            ),
            "special_mask": self._weighted_binary_mask(
                self.special_mask,
                self.special_mask_logits,
                self.structure_config.learn_special_mask,
            ),
            "urlpath_mask": self._weighted_binary_mask(
                self.urlpath_mask,
                self.urlpath_mask_logits,
                self.structure_config.learn_urlpath_mask,
            ),
            "markup_mask": self._weighted_binary_mask(
                self.markup_mask,
                self.markup_mask_logits,
                self.structure_config.learn_markup_mask,
            ),
            "attr_mask": self._weighted_binary_mask(
                self.attr_mask,
                self.attr_mask_logits,
                self.structure_config.learn_attr_mask,
            ),
            "entity_mask": self._weighted_binary_mask(
                self.entity_mask,
                self.entity_mask_logits,
                self.structure_config.learn_entity_mask,
            ),
        }

    def _weighted_causal_mask(self, length: int, lookback: int = 0) -> mx.array:
        if length > self.structure_config.base_config.max_seq_len:
            raise ValueError(
                f"Conker-8 learned lag profile max_seq_len={self.structure_config.base_config.max_seq_len} "
                f"is smaller than requested length={length}."
            )
        time_idx = mx.arange(length, dtype=mx.int32)
        delta = time_idx[:, None] - time_idx[None, :]
        mask = (delta > 0).astype(mx.float32)
        if lookback > 0:
            mask = mask * (delta <= lookback).astype(mx.float32)
        if not self.structure_config.learn_lag_profile:
            return mask
        lag_weights = 1.0 + self.structure_config.lag_profile_span * mx.tanh(self.causal_lag_logits[:length])
        safe_delta = mx.maximum(delta, 0)
        return mask * lag_weights[safe_delta]

    def _count_features_core(
        self,
        chars: mx.array,
        mask: mx.array,
    ) -> tuple[
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
    ]:
        weighted_masks = self._weighted_support_masks()
        batch, timesteps = chars.shape
        one_hot = self._one_hot(chars)
        zeros = mx.zeros((batch, 1, self.vocab_size), dtype=one_hot.dtype)
        next_one_hot = mx.concatenate([one_hot[:, 1:, :], zeros], axis=1)
        prev_one_hot = mx.concatenate([zeros, one_hot[:, :-1, :]], axis=1)

        exact1 = exact2 = exact3 = special2 = number2 = urlpath2 = markup2 = attr2 = entity2 = stack2 = wordclass2 = delim2 = delimsub2 = recency = None

        if any(
            (
                self.config.enable_exact1,
                self.config.enable_exact2,
                self.config.enable_exact3,
                self.config.enable_special2,
                self.config.enable_number2,
                self.config.enable_urlpath2,
                self.config.enable_markup2,
                self.config.enable_attr2,
                self.config.enable_entity2,
            )
        ):
            cur_match = mx.matmul(one_hot, mx.transpose(one_hot, (0, 2, 1))) * mask[None, :, :]
            if self.config.enable_exact1:
                exact1 = mx.matmul(cur_match, next_one_hot)
            if any(
                (
                    self.config.enable_exact2,
                    self.config.enable_exact3,
                    self.config.enable_special2,
                    self.config.enable_number2,
                    self.config.enable_urlpath2,
                    self.config.enable_markup2,
                    self.config.enable_attr2,
                    self.config.enable_entity2,
                )
            ):
                prev_match = mx.matmul(prev_one_hot, mx.transpose(prev_one_hot, (0, 2, 1)))
                pair_match = cur_match * prev_match
                if self.config.enable_exact2:
                    exact2 = mx.matmul(pair_match, next_one_hot)
                if self.config.enable_exact3:
                    prev2_one_hot = mx.concatenate(
                        [mx.zeros((batch, 2, self.vocab_size), dtype=one_hot.dtype), one_hot[:, :-2, :]],
                        axis=1,
                    )
                    prev2_match = mx.matmul(prev2_one_hot, mx.transpose(prev2_one_hot, (0, 2, 1)))
                    exact3 = mx.matmul(pair_match * prev2_match, next_one_hot)
                if self.config.enable_special2:
                    special2 = mx.matmul(pair_match, next_one_hot * weighted_masks["special_mask"][None, None, :])
                if self.config.enable_number2:
                    number2 = mx.matmul(pair_match, next_one_hot * weighted_masks["number_mask"][None, None, :])
                if self.config.enable_urlpath2:
                    urlpath2 = mx.matmul(pair_match, next_one_hot * weighted_masks["urlpath_mask"][None, None, :])
                if self.config.enable_markup2:
                    markup2 = mx.matmul(pair_match, next_one_hot * weighted_masks["markup_mask"][None, None, :])
                if self.config.enable_attr2:
                    attr2 = mx.matmul(pair_match, next_one_hot * weighted_masks["attr_mask"][None, None, :])
                if self.config.enable_entity2:
                    entity2 = mx.matmul(pair_match, next_one_hot * weighted_masks["entity_mask"][None, None, :])

        class_ids = self.token_class_ids[chars]
        prev_class_ids = mx.concatenate([mx.zeros((batch, 1), dtype=class_ids.dtype), class_ids[:, :-1]], axis=1)
        next_class_ids = mx.concatenate([class_ids[:, 1:], mx.zeros((batch, 1), dtype=class_ids.dtype)], axis=1)

        if self.config.enable_wordclass2:
            next_class_one_hot = (next_class_ids[..., None] == self.class_axis[None, None, :]).astype(one_hot.dtype)
            cur_class_match = (class_ids[:, :, None] == class_ids[:, None, :]).astype(one_hot.dtype) * mask[None, :, :]
            prev_class_match = (prev_class_ids[:, :, None] == prev_class_ids[:, None, :]).astype(one_hot.dtype)
            class2_counts = mx.matmul(cur_class_match * prev_class_match, next_class_one_hot)
            wordclass2 = mx.matmul(class2_counts, self.wordclass_token_mask)

        if self.config.enable_delim2:
            next_delim = next_one_hot * weighted_masks["delimiter_mask"][None, None, :]
            cur_class_match = (class_ids[:, :, None] == class_ids[:, None, :]).astype(one_hot.dtype) * mask[None, :, :]
            prev_class_match = (prev_class_ids[:, :, None] == prev_class_ids[:, None, :]).astype(one_hot.dtype)
            delim2 = mx.matmul(cur_class_match * prev_class_match, next_delim)

        if self.config.enable_delimsub2:
            delim_subtype_ids = self.delim_subtype_ids[chars]
            prev_delim_subtype_ids = mx.concatenate(
                [mx.zeros((batch, 1), dtype=delim_subtype_ids.dtype), delim_subtype_ids[:, :-1]],
                axis=1,
            )
            next_delim_subtype_ids = mx.concatenate(
                [delim_subtype_ids[:, 1:], mx.zeros((batch, 1), dtype=delim_subtype_ids.dtype)],
                axis=1,
            )
            next_delim_subtype_one_hot = (
                next_delim_subtype_ids[..., None] == self.delim_subtype_axis[None, None, :]
            ).astype(one_hot.dtype)
            cur_delim_match = (
                delim_subtype_ids[:, :, None] == delim_subtype_ids[:, None, :]
            ).astype(one_hot.dtype) * mask[None, :, :]
            prev_delim_match = (
                prev_delim_subtype_ids[:, :, None] == prev_delim_subtype_ids[:, None, :]
            ).astype(one_hot.dtype)
            delimsub_counts = mx.matmul(cur_delim_match * prev_delim_match, next_delim_subtype_one_hot)
            delimsub2 = mx.matmul(delimsub_counts, self.delim_subtype_token_mask)

        if self.config.enable_recency:
            recency_kernel = self._weighted_causal_mask(timesteps)
            if self.config.exact_context_span > 0:
                recency_kernel = recency_kernel * mx.array(_lookback_causal_mask(timesteps, self.config.exact_context_span))
            recency = mx.matmul(mx.broadcast_to(recency_kernel[None, :, :], (batch, timesteps, timesteps)), one_hot)

        if self.config.enable_stack2:
            char_np = np.array(chars, dtype=np.int32, copy=False)
            open_np = np.array(self.stack_open_ids, dtype=np.int32, copy=False)
            close_np = np.array(self.stack_close_ids, dtype=np.int32, copy=False)
            closer_mask_np = np.array(self.stack_closer_token_mask, dtype=np.float32, copy=False)
            stack_counts = np.zeros((batch, timesteps, self.vocab_size), dtype=np.float32)
            for b in range(batch):
                stack: list[int] = []
                for t in range(timesteps):
                    token_id = int(char_np[b, t])
                    close_type = int(close_np[token_id])
                    if close_type != 0 and stack and stack[-1] == close_type:
                        stack.pop()
                    open_type = int(open_np[token_id])
                    if open_type != 0:
                        stack.append(open_type)
                    if stack:
                        top = stack[-1]
                        stack_counts[b, t, :] = closer_mask_np[top] * float(len(stack))
            stack2 = mx.array(stack_counts)

        return exact1, exact2, exact3, special2, number2, urlpath2, markup2, attr2, entity2, stack2, wordclass2, delim2, delimsub2, recency

    def _count_features(
        self,
        chars: mx.array,
    ) -> tuple[
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
    ]:
        batch, timesteps = chars.shape
        if timesteps > self.config.base_config.max_seq_len:
            raise ValueError(
                f"Conker-8 max_seq_len={self.config.base_config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.config.exact_context_span > 0:
            raise ValueError("Conker-8 initial rebuild does not support exact_context_span > 0.")
        mask = self._weighted_causal_mask(timesteps)
        return self._count_features_core(chars, mask)


def scale_config(config: ConkerEightConfig, scale: float) -> ConkerEightConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        base_config=scale_conker3_config(config.base_config, scale),
    )
