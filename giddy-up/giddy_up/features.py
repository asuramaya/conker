from __future__ import annotations

import mlx.core as mx
import numpy as np


def structure_proxy_feature_arrays(base_probs: mx.array, p_trigram: mx.array) -> dict[str, mx.array]:
    trigram_log_probs = mx.log(mx.maximum(p_trigram, mx.array(1e-8, dtype=p_trigram.dtype)))
    trigram_entropy = -mx.sum(p_trigram * trigram_log_probs, axis=-1)
    trigram_peak = mx.max(p_trigram, axis=-1)
    trigram_top4 = mx.sort(p_trigram, axis=-1)[..., -4:]
    trigram_top4_mass = mx.sum(trigram_top4, axis=-1)
    base_top4 = mx.argsort(base_probs, axis=-1)[..., -4:]
    agreement_mass = mx.sum(mx.take_along_axis(p_trigram, base_top4, axis=-1), axis=-1)
    vocab_log = mx.array(float(np.log(max(int(p_trigram.shape[-1]), 2))), dtype=p_trigram.dtype)
    normalized_entropy = trigram_entropy / mx.maximum(vocab_log, mx.array(1e-8, dtype=p_trigram.dtype))
    trigram_candidate4_soft = trigram_top4_mass * mx.maximum(
        mx.array(1.0, dtype=p_trigram.dtype) - normalized_entropy,
        mx.array(0.0, dtype=p_trigram.dtype),
    )
    base_top = mx.argmax(base_probs, axis=-1)
    trigram_top = mx.argmax(p_trigram, axis=-1)
    top_agree = (base_top == trigram_top).astype(mx.float32)
    return {
        "entropy": trigram_entropy,
        "peak": trigram_peak,
        "candidate4": trigram_candidate4_soft,
        "agreement": top_agree,
        "agreement_mass": agreement_mass,
    }
