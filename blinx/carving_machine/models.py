from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import (
    AuxSource,
    CarverConfig,
    FrozenReadoutConfig,
    GRUConfig,
    HierarchicalAuxSource,
    HierarchicalCarverConfig,
    HormonalHierarchicalCarverConfig,
    RoutedHierarchicalConfig,
    SamplePolicy,
    StateSource,
)
from .reservoir import ReservoirBundle, build_dense_matrix


def _fixed_projection(rows: int, cols: int) -> mx.array:
    return mx.array(np.random.randn(rows, cols).astype(np.float32) / math.sqrt(rows))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int):
        super().__init__()
        self.layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev, hidden_dim))
            prev = hidden_dim
        self.out = nn.Linear(prev, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.gelu(layer(x))
        return self.out(x)


class CarverModel(nn.Module):
    """
    Flexible carver model.

    The important abstraction is not "masking" by itself, but:

    - a frozen substrate (`state`)
    - a learned side-channel (`mask`)
    - a readout that can consume either the sculpted state, the raw state,
      zeros, the mask, or a random second view
    """

    def __init__(
        self,
        reservoir: ReservoirBundle,
        vocab_size: int,
        embedding_dim: int,
        reservoir_size: int,
        config: CarverConfig,
    ):
        super().__init__()
        self.Wr = reservoir.Wr
        self.Wi = reservoir.Wi
        self.reservoir_size = reservoir_size
        self.sample_size = config.sample_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.controller_proj = _fixed_projection(reservoir_size, config.projection_dim)
        self.aux_proj = _fixed_projection(reservoir_size, config.sample_size)

        self.c1 = nn.Linear(config.projection_dim + embedding_dim, config.controller_width)
        self.c2 = nn.Linear(config.controller_width, config.controller_width)
        self.c3 = nn.Linear(config.controller_width, reservoir_size)
        self.c3.weight = mx.zeros_like(self.c3.weight)
        self.c3.bias = mx.full(self.c3.bias.shape, config.mask_bias)

        self.sample_idx = mx.array(np.random.choice(reservoir_size, config.sample_size, replace=False))
        self.readout = MLP(config.sample_size * 2, config.readout_hidden, vocab_size)

        self.state_source: StateSource = config.state_source
        self.aux_source: AuxSource = config.aux_source
        self.sample_policy: SamplePolicy = config.sample_policy

    def freeze_static(self) -> None:
        self.freeze(
            keys=["Wr", "Wi", "controller_proj", "aux_proj", "sample_idx"],
            strict=False,
        )

    def set_readout_mode(self, state_source: StateSource, aux_source: AuxSource) -> None:
        self.state_source = state_source
        self.aux_source = aux_source

    def _controller_outputs(self, state: mx.array, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        hidden = mx.concatenate([state @ self.controller_proj, x], axis=-1)
        hidden = nn.gelu(self.c1(hidden))
        hidden = nn.gelu(self.c2(hidden))
        logits = self.c3(hidden)
        mask = mx.sigmoid(logits)
        return hidden, logits, mask

    def _controller(self, state: mx.array, x: mx.array) -> mx.array:
        _, _, mask = self._controller_outputs(state, x)
        return mask

    def _sample_indices(self, mask: mx.array) -> mx.array:
        if self.sample_policy == "random":
            return self.sample_idx
        if self.sample_policy == "boundary":
            distance = mx.abs(mask - 0.5)
            return mx.argsort(distance, axis=-1)[:, : self.sample_size]
        if self.sample_policy == "mask_variance":
            mean = mx.mean(mask, axis=0)
            centered = mask - mean
            variance = mx.mean(centered * centered, axis=0)
            return mx.argsort(-variance)[: self.sample_size]
        raise ValueError(f"Unknown sample policy: {self.sample_policy}")

    def _gather(self, values: mx.array, sample_idx: mx.array) -> mx.array:
        if sample_idx.ndim == 1:
            return values[:, sample_idx]
        return mx.take_along_axis(values, sample_idx, axis=-1)

    def _state_features(self, state: mx.array, mask: mx.array, sample_idx: mx.array) -> mx.array:
        if self.state_source == "sculpted":
            chosen = state * (1.0 - mask)
            return self._gather(chosen, sample_idx)
        if self.state_source == "raw":
            return self._gather(state, sample_idx)
        if self.state_source == "zeros":
            return mx.zeros((state.shape[0], self.sample_size))
        raise ValueError(f"Unknown state source: {self.state_source}")

    def _aux_features(self, state: mx.array, mask: mx.array, sample_idx: mx.array) -> mx.array:
        if self.aux_source == "mask":
            return self._gather(mask, sample_idx)
        if self.aux_source == "zeros":
            return mx.zeros((state.shape[0], self.sample_size))
        if self.aux_source == "random":
            return state @ self.aux_proj
        raise ValueError(f"Unknown aux source: {self.aux_source}")

    def _step(self, state: mx.array, x: mx.array) -> tuple[mx.array, mx.array]:
        state = mx.stop_gradient(mx.tanh(state @ self.Wr.T + x @ self.Wi.T))
        mask = self._controller(state, x)
        return state, mask

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        state = mx.zeros((batch_size, self.reservoir_size))
        logits = []
        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            state, mask = self._step(state, x)
            sample_idx = self._sample_indices(mask)
            readout_input = mx.concatenate(
                [
                    self._state_features(state, mask, sample_idx),
                    self._aux_features(state, mask, sample_idx),
                ],
                axis=-1,
            )
            logits.append(self.readout(readout_input))
        return mx.stack(logits, axis=1)

    def get_masks(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        state = mx.zeros((batch_size, self.reservoir_size))
        masks = []
        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            state, mask = self._step(state, x)
            masks.append(mask)
        return mx.stack(masks, axis=1)

    def inspect_controller(self, chars: mx.array) -> dict[str, mx.array]:
        batch_size, timesteps = chars.shape
        state = mx.zeros((batch_size, self.reservoir_size))
        masks = []
        logits = []
        hidden_states = []
        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            state = mx.stop_gradient(mx.tanh(state @ self.Wr.T + x @ self.Wi.T))
            hidden, logit, mask = self._controller_outputs(state, x)
            masks.append(mask)
            logits.append(logit)
            hidden_states.append(hidden)
        return {
            "mask": mx.stack(masks, axis=1),
            "logit": mx.stack(logits, axis=1),
            "hidden": mx.stack(hidden_states, axis=1),
        }


class GRUModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, config: GRUConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Wz = nn.Linear(embedding_dim + config.hidden_size, config.hidden_size)
        self.Wr_g = nn.Linear(embedding_dim + config.hidden_size, config.hidden_size)
        self.Wn = nn.Linear(embedding_dim + config.hidden_size, config.hidden_size)
        self.readout = MLP(config.hidden_size, config.readout_hidden, vocab_size)

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        hidden = mx.zeros((batch_size, self.hidden_size))
        logits = []
        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            xh = mx.concatenate([x, hidden], axis=-1)
            z = mx.sigmoid(self.Wz(xh))
            r = mx.sigmoid(self.Wr_g(xh))
            n = mx.tanh(self.Wn(mx.concatenate([x, r * hidden], axis=-1)))
            hidden = (1.0 - z) * hidden + z * n
            logits.append(self.readout(hidden))
        return mx.stack(logits, axis=1)


class HierarchicalCarverModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, config: HierarchicalCarverConfig):
        super().__init__()
        self.fast_size = config.fast_size
        self.mid_size = config.mid_size
        self.slow_size = config.slow_size
        self.slow_update_stride = max(config.slow_update_stride, 1)
        self.aux_source: HierarchicalAuxSource = config.aux_source
        self.use_pathway_gates = config.use_pathway_gates
        self.gate_fast_mid = config.gate_fast_mid
        self.gate_mid_slow = config.gate_mid_slow
        self.pathway_gate_source = config.pathway_gate_source
        controller_view_dim = config.controller_view_dim or config.controller_width
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.Wf = build_dense_matrix(
            config.fast_size,
            config.fast_connectivity,
            config.fast_spectral_radius,
            config.seed,
            topology=config.fast_topology,
            rewire_prob=config.fast_rewire_prob,
        )
        self.Wm = build_dense_matrix(
            config.mid_size,
            config.mid_connectivity,
            config.mid_spectral_radius,
            config.seed + 1,
            topology=config.mid_topology,
            rewire_prob=config.mid_rewire_prob,
        )
        self.Ws = build_dense_matrix(
            config.slow_size,
            config.slow_connectivity,
            config.slow_spectral_radius,
            config.seed + 2,
            topology=config.slow_topology,
            rewire_prob=config.slow_rewire_prob,
        )

        rng = np.random.default_rng(config.seed)
        self.Wif = mx.array(
            rng.standard_normal((config.fast_size, embedding_dim), dtype=np.float32) * config.input_scale
        )
        self.Wu1 = mx.array(
            rng.standard_normal((config.mid_size, config.fast_size), dtype=np.float32) * config.upward_scale
        )
        self.Wu2 = mx.array(
            rng.standard_normal((config.slow_size, config.mid_size), dtype=np.float32) * config.upward_scale
        )
        self.ps = _fixed_projection(config.slow_size, controller_view_dim)
        self.pm = _fixed_projection(config.mid_size, controller_view_dim)
        self.aux_proj = _fixed_projection(config.fast_size, config.fast_sample_size)

        self.p1 = nn.Linear(controller_view_dim * 2, config.controller_width)
        self.p2 = nn.Linear(config.controller_width, config.controller_width)
        self.p3 = nn.Linear(config.controller_width, config.fast_size)
        self.p3.weight = mx.zeros_like(self.p3.weight)
        self.p3.bias = mx.full(self.p3.bias.shape, config.mask_bias)
        if self.use_pathway_gates:
            gate_input_dim = controller_view_dim if self.pathway_gate_source == "slow" else 1
            self.g_fm = nn.Linear(gate_input_dim, 1)
            self.g_ms = nn.Linear(gate_input_dim, 1)
            self.g_fm.weight = mx.zeros_like(self.g_fm.weight)
            self.g_fm.bias = mx.full(self.g_fm.bias.shape, config.pathway_gate_bias)
            self.g_ms.weight = mx.zeros_like(self.g_ms.weight)
            self.g_ms.bias = mx.full(self.g_ms.bias.shape, config.pathway_gate_bias)

        sample_rng = np.random.default_rng(config.seed + 10)
        self.sf = mx.array(sample_rng.choice(config.fast_size, config.fast_sample_size, replace=False))
        self.sm = mx.array(sample_rng.choice(config.mid_size, config.mid_sample_size, replace=False))
        self.ss = mx.array(sample_rng.choice(config.slow_size, config.slow_sample_size, replace=False))

        readout_in = config.fast_sample_size * 2 + config.mid_sample_size + config.slow_sample_size
        self.readout = MLP(readout_in, config.readout_hidden, vocab_size)

    def freeze_static(self) -> None:
        self.freeze(
            keys=["Wf", "Wm", "Ws", "Wif", "Wu1", "Wu2", "ps", "pm", "aux_proj", "sf", "sm", "ss"],
            strict=False,
        )

    def _apply_pathway_gate_heads(self, gate_input: mx.array, batch_size: int) -> tuple[mx.array, mx.array]:
        ones = mx.ones((batch_size, 1))
        gate_fm = mx.sigmoid(self.g_fm(gate_input)) if self.gate_fast_mid else ones
        gate_ms = mx.sigmoid(self.g_ms(gate_input)) if self.gate_mid_slow else ones
        return gate_fm, gate_ms

    def _pathway_gates(self, slow: mx.array) -> tuple[mx.array, mx.array]:
        if not self.use_pathway_gates:
            ones = mx.ones((slow.shape[0], 1))
            return ones, ones
        if self.pathway_gate_source != "slow":
            raise ValueError("Slow-state pathway gates requested for a non-slow gate source.")
        slow_view = slow @ self.ps
        return self._apply_pathway_gate_heads(slow_view, slow.shape[0])

    def _surprise_pathway_gates(self, surprise: mx.array) -> tuple[mx.array, mx.array]:
        if not self.use_pathway_gates:
            ones = mx.ones((surprise.shape[0], 1))
            return ones, ones
        surprise_level = mx.mean(mx.abs(surprise), axis=-1, keepdims=True)
        return self._apply_pathway_gate_heads(surprise_level, surprise.shape[0])

    def _default_pathway_gates(self, batch_size: int) -> tuple[mx.array, mx.array]:
        if not self.use_pathway_gates:
            ones = mx.ones((batch_size, 1))
            return ones, ones
        if self.pathway_gate_source == "slow":
            return self._apply_pathway_gate_heads(mx.zeros((batch_size, self.ps.shape[1])), batch_size)
        return self._apply_pathway_gate_heads(mx.zeros((batch_size, 1)), batch_size)

    def _aux_features(self, fast: mx.array, prediction: mx.array) -> mx.array:
        if self.aux_source == "prediction":
            return prediction[:, self.sf]
        if self.aux_source == "zeros":
            return mx.zeros((fast.shape[0], self.sf.shape[0]))
        if self.aux_source == "random":
            return fast @ self.aux_proj
        raise ValueError(f"Unknown hierarchical aux source: {self.aux_source}")

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        gate_fm, gate_ms = self._default_pathway_gates(batch_size)
        logits = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            if self.use_pathway_gates and self.pathway_gate_source == "slow":
                gate_fm, gate_ms = self._pathway_gates(slow)
            fast = mx.stop_gradient(mx.tanh(fast @ self.Wf.T + x @ self.Wif.T))
            mid = mx.stop_gradient(mx.tanh(mid @ self.Wm.T + gate_fm * (fast @ self.Wu1.T)))
            if timestep % self.slow_update_stride == 0:
                slow = mx.stop_gradient(mx.tanh(slow @ self.Ws.T + gate_ms * (mid @ self.Wu2.T)))

            hidden = mx.concatenate([slow @ self.ps, mid @ self.pm], axis=-1)
            hidden = nn.gelu(self.p1(hidden))
            hidden = nn.gelu(self.p2(hidden))
            prediction = mx.sigmoid(self.p3(hidden))

            surprise = fast * (1.0 - prediction)
            if self.use_pathway_gates and self.pathway_gate_source == "surprise":
                gate_fm, gate_ms = self._surprise_pathway_gates(surprise)
            readout_input = mx.concatenate(
                [
                    surprise[:, self.sf],
                    self._aux_features(fast, prediction),
                    mid[:, self.sm],
                    slow[:, self.ss],
                ],
                axis=-1,
            )
            logits.append(self.readout(readout_input))

        return mx.stack(logits, axis=1)


class DelayLineHierarchicalModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, config: HierarchicalCarverConfig):
        super().__init__()
        self.fast_size = config.fast_size
        self.mid_size = config.mid_size
        self.slow_size = config.slow_size
        self.slow_update_stride = max(config.slow_update_stride, 1)
        self.aux_source: HierarchicalAuxSource = config.aux_source
        controller_view_dim = config.controller_view_dim or config.controller_width

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.ps = _fixed_projection(config.slow_size, controller_view_dim)
        self.pm = _fixed_projection(config.mid_size, controller_view_dim)
        self.aux_proj = _fixed_projection(config.fast_size, config.fast_sample_size)

        self.fast_delay_idx = mx.array(np.arange(math.ceil(config.fast_size / embedding_dim), dtype=np.int32))
        self.mid_delay_idx = mx.array(np.arange(math.ceil(config.mid_size / embedding_dim), dtype=np.int32))
        self.slow_delay_idx = mx.array(np.arange(math.ceil(config.slow_size / embedding_dim), dtype=np.int32))
        self.max_delay_len = int(max(self.fast_delay_idx.shape[0], self.mid_delay_idx.shape[0], self.slow_delay_idx.shape[0]))

        self.p1 = nn.Linear(controller_view_dim * 2, config.controller_width)
        self.p2 = nn.Linear(config.controller_width, config.controller_width)
        self.p3 = nn.Linear(config.controller_width, config.fast_size)
        self.p3.weight = mx.zeros_like(self.p3.weight)
        self.p3.bias = mx.full(self.p3.bias.shape, config.mask_bias)

        sample_rng = np.random.default_rng(config.seed + 10)
        self.sf = mx.array(sample_rng.choice(config.fast_size, config.fast_sample_size, replace=False))
        self.sm = mx.array(sample_rng.choice(config.mid_size, config.mid_sample_size, replace=False))
        self.ss = mx.array(sample_rng.choice(config.slow_size, config.slow_sample_size, replace=False))

        readout_in = config.fast_sample_size * 2 + config.mid_sample_size + config.slow_sample_size
        self.readout = MLP(readout_in, config.readout_hidden, vocab_size)

    def freeze_static(self) -> None:
        self.freeze(
            keys=[
                "ps",
                "pm",
                "aux_proj",
                "fast_delay_idx",
                "mid_delay_idx",
                "slow_delay_idx",
                "sf",
                "sm",
                "ss",
            ],
            strict=False,
        )

    def _delay_state(self, history: mx.array, delay_idx: mx.array, size: int) -> mx.array:
        delayed = mx.take(history, delay_idx, axis=1)
        flat = mx.reshape(delayed, (history.shape[0], -1))
        if flat.shape[1] < size:
            flat = mx.concatenate([flat, mx.zeros((history.shape[0], size - flat.shape[1]))], axis=-1)
        return mx.tanh(flat[:, :size])

    def _aux_features(self, fast: mx.array, prediction: mx.array) -> mx.array:
        if self.aux_source == "prediction":
            return prediction[:, self.sf]
        if self.aux_source == "zeros":
            return mx.zeros((fast.shape[0], self.sf.shape[0]))
        if self.aux_source == "random":
            return fast @ self.aux_proj
        raise ValueError(f"Unknown hierarchical aux source: {self.aux_source}")

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        slow = mx.zeros((batch_size, self.slow_size))
        logits = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            history = mx.concatenate([x[:, None, :], history[:, :-1, :]], axis=1)
            fast = self._delay_state(history, self.fast_delay_idx, self.fast_size)
            mid = self._delay_state(history, self.mid_delay_idx, self.mid_size)
            if timestep % self.slow_update_stride == 0:
                slow = self._delay_state(history, self.slow_delay_idx, self.slow_size)

            hidden = mx.concatenate([slow @ self.ps, mid @ self.pm], axis=-1)
            hidden = nn.gelu(self.p1(hidden))
            hidden = nn.gelu(self.p2(hidden))
            prediction = mx.sigmoid(self.p3(hidden))
            surprise = fast * (1.0 - prediction)

            readout_input = mx.concatenate(
                [
                    surprise[:, self.sf],
                    self._aux_features(fast, prediction),
                    mid[:, self.sm],
                    slow[:, self.ss],
                ],
                axis=-1,
            )
            logits.append(self.readout(readout_input))

        return mx.stack(logits, axis=1)


class HybridDelayHierarchicalModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, config: HierarchicalCarverConfig):
        super().__init__()
        self.fast_size = config.fast_size
        self.mid_size = config.mid_size
        self.slow_size = config.slow_size
        self.slow_update_stride = max(config.slow_update_stride, 1)
        self.aux_source: HierarchicalAuxSource = config.aux_source
        controller_view_dim = config.controller_view_dim or config.controller_width

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Wm = build_dense_matrix(
            config.mid_size,
            config.mid_connectivity,
            config.mid_spectral_radius,
            config.seed + 1,
            topology=config.mid_topology,
            rewire_prob=config.mid_rewire_prob,
        )
        self.Ws = build_dense_matrix(
            config.slow_size,
            config.slow_connectivity,
            config.slow_spectral_radius,
            config.seed + 2,
            topology=config.slow_topology,
            rewire_prob=config.slow_rewire_prob,
        )

        rng = np.random.default_rng(config.seed)
        self.Wu1 = mx.array(
            rng.standard_normal((config.mid_size, config.fast_size), dtype=np.float32) * config.upward_scale
        )
        self.Wu2 = mx.array(
            rng.standard_normal((config.slow_size, config.mid_size), dtype=np.float32) * config.upward_scale
        )
        self.ps = _fixed_projection(config.slow_size, controller_view_dim)
        self.pm = _fixed_projection(config.mid_size, controller_view_dim)
        self.aux_proj = _fixed_projection(config.fast_size, config.fast_sample_size)

        self.fast_delay_idx = mx.array(np.arange(math.ceil(config.fast_size / embedding_dim), dtype=np.int32))
        self.max_delay_len = int(self.fast_delay_idx.shape[0])

        self.p1 = nn.Linear(controller_view_dim * 2, config.controller_width)
        self.p2 = nn.Linear(config.controller_width, config.controller_width)
        self.p3 = nn.Linear(config.controller_width, config.fast_size)
        self.p3.weight = mx.zeros_like(self.p3.weight)
        self.p3.bias = mx.full(self.p3.bias.shape, config.mask_bias)

        sample_rng = np.random.default_rng(config.seed + 10)
        self.sf = mx.array(sample_rng.choice(config.fast_size, config.fast_sample_size, replace=False))
        self.sm = mx.array(sample_rng.choice(config.mid_size, config.mid_sample_size, replace=False))
        self.ss = mx.array(sample_rng.choice(config.slow_size, config.slow_sample_size, replace=False))

        readout_in = config.fast_sample_size * 2 + config.mid_sample_size + config.slow_sample_size
        self.readout = MLP(readout_in, config.readout_hidden, vocab_size)

    def freeze_static(self) -> None:
        self.freeze(
            keys=[
                "Wm",
                "Ws",
                "Wu1",
                "Wu2",
                "ps",
                "pm",
                "aux_proj",
                "fast_delay_idx",
                "sf",
                "sm",
                "ss",
            ],
            strict=False,
        )

    def _delay_state(self, history: mx.array) -> mx.array:
        delayed = mx.take(history, self.fast_delay_idx, axis=1)
        flat = mx.reshape(delayed, (history.shape[0], -1))
        if flat.shape[1] < self.fast_size:
            flat = mx.concatenate([flat, mx.zeros((history.shape[0], self.fast_size - flat.shape[1]))], axis=-1)
        return mx.tanh(flat[:, : self.fast_size])

    def _aux_features(self, fast: mx.array, prediction: mx.array) -> mx.array:
        if self.aux_source == "prediction":
            return prediction[:, self.sf]
        if self.aux_source == "zeros":
            return mx.zeros((fast.shape[0], self.sf.shape[0]))
        if self.aux_source == "random":
            return fast @ self.aux_proj
        raise ValueError(f"Unknown hierarchical aux source: {self.aux_source}")

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        logits = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            history = mx.concatenate([x[:, None, :], history[:, :-1, :]], axis=1)
            fast = self._delay_state(history)
            mid = mx.stop_gradient(mx.tanh(mid @ self.Wm.T + fast @ self.Wu1.T))
            if timestep % self.slow_update_stride == 0:
                slow = mx.stop_gradient(mx.tanh(slow @ self.Ws.T + mid @ self.Wu2.T))

            hidden = mx.concatenate([slow @ self.ps, mid @ self.pm], axis=-1)
            hidden = nn.gelu(self.p1(hidden))
            hidden = nn.gelu(self.p2(hidden))
            prediction = mx.sigmoid(self.p3(hidden))
            surprise = fast * (1.0 - prediction)

            readout_input = mx.concatenate(
                [
                    surprise[:, self.sf],
                    self._aux_features(fast, prediction),
                    mid[:, self.sm],
                    slow[:, self.ss],
                ],
                axis=-1,
            )
            logits.append(self.readout(readout_input))

        return mx.stack(logits, axis=1)


class MixedMemoryHierarchicalModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, config: HierarchicalCarverConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.fast_size = config.fast_size
        self.mid_size = config.mid_size
        self.slow_size = config.slow_size
        self.slow_update_stride = max(config.slow_update_stride, 1)
        self.aux_source: HierarchicalAuxSource = config.aux_source
        self.use_hypothesis_error_channel = config.use_hypothesis_error_channel
        self.hypothesis_loss_weight = config.hypothesis_loss_weight
        self.use_predictive_residual_channel = config.use_predictive_residual_channel
        self.use_predictive_output_channel = config.use_predictive_output_channel
        self.use_random_third_channel = config.use_random_third_channel
        self.random_third_size = config.random_third_sample_size or config.fast_sample_size
        self.predictive_residual_horizons = tuple(sorted(set(config.predictive_residual_horizons)))
        self.predictive_residual_loss_weight = config.predictive_residual_loss_weight
        self.fast_memory_mode = config.fast_memory_mode
        self.mid_memory_mode = config.mid_memory_mode
        self.slow_memory_mode = config.slow_memory_mode
        controller_view_dim = config.controller_view_dim or config.controller_width

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        rng = np.random.default_rng(config.seed)

        if self.fast_memory_mode == "recurrent":
            self.Wf = build_dense_matrix(
                config.fast_size,
                config.fast_connectivity,
                config.fast_spectral_radius,
                config.seed,
                topology=config.fast_topology,
                rewire_prob=config.fast_rewire_prob,
            )
            self.Wif = mx.array(
                rng.standard_normal((config.fast_size, embedding_dim), dtype=np.float32) * config.input_scale
            )
        if self.mid_memory_mode == "recurrent":
            self.Wm = build_dense_matrix(
                config.mid_size,
                config.mid_connectivity,
                config.mid_spectral_radius,
                config.seed + 1,
                topology=config.mid_topology,
                rewire_prob=config.mid_rewire_prob,
            )
            self.Wu1 = mx.array(
                rng.standard_normal((config.mid_size, config.fast_size), dtype=np.float32) * config.upward_scale
            )
        if self.slow_memory_mode == "recurrent":
            self.Ws = build_dense_matrix(
                config.slow_size,
                config.slow_connectivity,
                config.slow_spectral_radius,
                config.seed + 2,
                topology=config.slow_topology,
                rewire_prob=config.slow_rewire_prob,
            )
            self.Wu2 = mx.array(
                rng.standard_normal((config.slow_size, config.mid_size), dtype=np.float32) * config.upward_scale
            )

        self.ps = _fixed_projection(config.slow_size, controller_view_dim)
        self.pm = _fixed_projection(config.mid_size, controller_view_dim)
        self.aux_proj = _fixed_projection(config.fast_size, config.fast_sample_size)

        self.fast_delay_idx = mx.array(np.arange(math.ceil(config.fast_size / embedding_dim), dtype=np.int32))
        self.mid_delay_idx = mx.array(np.arange(math.ceil(config.mid_size / embedding_dim), dtype=np.int32))
        self.slow_delay_idx = mx.array(np.arange(math.ceil(config.slow_size / embedding_dim), dtype=np.int32))
        self.max_delay_len = int(max(self.fast_delay_idx.shape[0], self.mid_delay_idx.shape[0], self.slow_delay_idx.shape[0]))

        self.p1 = nn.Linear(controller_view_dim * 2, config.controller_width)
        self.p2 = nn.Linear(config.controller_width, config.controller_width)
        self.p3 = nn.Linear(config.controller_width, config.fast_size)
        self.p3.weight = mx.zeros_like(self.p3.weight)
        self.p3.bias = mx.full(self.p3.bias.shape, config.mask_bias)

        sample_rng = np.random.default_rng(config.seed + 10)
        self.sf = mx.array(sample_rng.choice(config.fast_size, config.fast_sample_size, replace=False))
        self.sm = mx.array(sample_rng.choice(config.mid_size, config.mid_sample_size, replace=False))
        self.ss = mx.array(sample_rng.choice(config.slow_size, config.slow_sample_size, replace=False))

        readout_in = config.fast_sample_size * 2 + config.mid_sample_size + config.slow_sample_size
        if self.use_hypothesis_error_channel:
            self.hypothesis = MLP(config.controller_width, config.hypothesis_hidden, config.fast_sample_size)
            readout_in += config.fast_sample_size
        if self.use_predictive_residual_channel:
            residual_dim = len(self.predictive_residual_horizons) * vocab_size
            self.predictive_residual_predictor = MLP(
                config.controller_width,
                config.predictive_residual_hidden,
                residual_dim,
            )
            self.predictive_residual_proj = nn.Linear(residual_dim, config.fast_sample_size)
            readout_in += config.fast_sample_size
        if self.use_predictive_output_channel:
            output_dim = len(self.predictive_residual_horizons) * vocab_size
            self.predictive_output_predictor = MLP(
                config.controller_width,
                config.predictive_residual_hidden,
                output_dim,
            )
            self.predictive_output_proj = nn.Linear(output_dim, config.fast_sample_size)
            readout_in += config.fast_sample_size
        if self.use_random_third_channel:
            self.random_third_proj = mx.array(
                rng.standard_normal((config.controller_width, self.random_third_size), dtype=np.float32) * 0.05
            )
            readout_in += self.random_third_size
        self.readout = MLP(readout_in, config.readout_hidden, vocab_size)

    @staticmethod
    def _feature_norm(features: mx.array) -> mx.array:
        return mx.sqrt(mx.sum(features * features, axis=-1) + 1e-8)

    def freeze_static(self) -> None:
        keys = ["ps", "pm", "aux_proj", "fast_delay_idx", "mid_delay_idx", "slow_delay_idx", "sf", "sm", "ss"]
        for key in ("Wf", "Wif", "Wm", "Wu1", "Ws", "Wu2"):
            if hasattr(self, key):
                keys.append(key)
        self.freeze(keys=keys, strict=False)

    def _delay_state(self, history: mx.array, delay_idx: mx.array, size: int) -> mx.array:
        delayed = mx.take(history, delay_idx, axis=1)
        flat = mx.reshape(delayed, (history.shape[0], -1))
        if flat.shape[1] < size:
            flat = mx.concatenate([flat, mx.zeros((history.shape[0], size - flat.shape[1]))], axis=-1)
        return mx.tanh(flat[:, :size])

    def _aux_features(self, fast: mx.array, prediction: mx.array) -> mx.array:
        if self.aux_source == "prediction":
            return prediction[:, self.sf]
        if self.aux_source == "zeros":
            return mx.zeros((fast.shape[0], self.sf.shape[0]))
        if self.aux_source == "random":
            return fast @ self.aux_proj
        raise ValueError(f"Unknown hierarchical aux source: {self.aux_source}")

    def _one_hot(self, tokens: mx.array) -> mx.array:
        classes = mx.arange(self.vocab_size, dtype=mx.int32)
        return (tokens[..., None] == classes).astype(mx.float32)

    def _encode_step(
        self,
        history: mx.array,
        x: mx.array,
        fast: mx.array,
        mid: mx.array,
        slow: mx.array,
        timestep: int,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        history = mx.concatenate([x[:, None, :], history[:, :-1, :]], axis=1)

        if self.fast_memory_mode == "recurrent":
            fast = mx.stop_gradient(mx.tanh(fast @ self.Wf.T + x @ self.Wif.T))
        else:
            fast = self._delay_state(history, self.fast_delay_idx, self.fast_size)

        if self.mid_memory_mode == "recurrent":
            mid = mx.stop_gradient(mx.tanh(mid @ self.Wm.T + fast @ self.Wu1.T))
        else:
            mid = self._delay_state(history, self.mid_delay_idx, self.mid_size)

        if timestep % self.slow_update_stride == 0:
            if self.slow_memory_mode == "recurrent":
                slow = mx.stop_gradient(mx.tanh(slow @ self.Ws.T + mid @ self.Wu2.T))
            else:
                slow = self._delay_state(history, self.slow_delay_idx, self.slow_size)

        hidden = mx.concatenate([slow @ self.ps, mid @ self.pm], axis=-1)
        hidden = nn.gelu(self.p1(hidden))
        hidden = nn.gelu(self.p2(hidden))
        prediction = mx.sigmoid(self.p3(hidden))
        surprise = fast * (1.0 - prediction)
        aux_features = self._aux_features(fast, prediction)
        readout_features = mx.concatenate(
            [
                surprise[:, self.sf],
                aux_features,
                mid[:, self.sm],
                slow[:, self.ss],
            ],
            axis=-1,
        )
        return history, fast, mid, slow, hidden, aux_features, readout_features

    def _forward_with_hypothesis(
        self,
        chars: mx.array,
        collect_targets: bool,
    ) -> tuple[mx.array, mx.array | None, mx.array | None]:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        prev_hypothesis = mx.zeros((batch_size, self.sf.shape[0]))
        logits = []
        hypothesis_preds = []
        aux_targets = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            history, fast, mid, slow, hidden, aux_features, base_features = self._encode_step(
                history, x, fast, mid, slow, timestep
            )
            error_features = mx.stop_gradient(aux_features - prev_hypothesis)
            logits.append(self.readout(mx.concatenate([base_features, error_features], axis=-1)))

            hypothesis = self.hypothesis(hidden)
            hypothesis_preds.append(hypothesis)
            if collect_targets:
                aux_targets.append(mx.stop_gradient(aux_features))
            prev_hypothesis = hypothesis

        hypothesis_stack = mx.stack(hypothesis_preds, axis=1)
        target_stack = mx.stack(aux_targets, axis=1) if collect_targets else None
        return mx.stack(logits, axis=1), hypothesis_stack, target_stack

    def _forward_with_predictive_residual(
        self,
        chars: mx.array,
        collect_predictions: bool,
    ) -> tuple[mx.array, mx.array | None]:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        logits = []
        predictor_probs = []
        horizon_count = len(self.predictive_residual_horizons)

        for timestep in range(timesteps):
            current_one_hot = self._one_hot(chars[:, timestep])
            residual_parts = []
            for horizon_idx, horizon in enumerate(self.predictive_residual_horizons):
                if timestep >= horizon:
                    residual_parts.append(current_one_hot - predictor_probs[timestep - horizon][:, horizon_idx, :])
                else:
                    residual_parts.append(mx.zeros((batch_size, self.vocab_size)))
            residual_cat = mx.concatenate(residual_parts, axis=-1)

            x = self.embedding(chars[:, timestep])
            history, fast, mid, slow, hidden, _, base_features = self._encode_step(
                history, x, fast, mid, slow, timestep
            )
            residual_features = self.predictive_residual_proj(mx.stop_gradient(residual_cat))
            logits.append(self.readout(mx.concatenate([base_features, residual_features], axis=-1)))

            predictor_logits = self.predictive_residual_predictor(hidden)
            predictor_prob = mx.softmax(
                predictor_logits.reshape(batch_size, horizon_count, self.vocab_size),
                axis=-1,
            )
            predictor_probs.append(predictor_prob)

        prediction_stack = mx.stack(predictor_probs, axis=1) if collect_predictions else None
        return mx.stack(logits, axis=1), prediction_stack

    def _forward_with_predictive_output(
        self,
        chars: mx.array,
        collect_predictions: bool,
    ) -> tuple[mx.array, mx.array | None]:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        logits = []
        predictor_probs = []
        horizon_count = len(self.predictive_residual_horizons)

        for timestep in range(timesteps):
            forecast_parts = []
            for horizon_idx, horizon in enumerate(self.predictive_residual_horizons):
                if timestep >= horizon:
                    forecast_parts.append(predictor_probs[timestep - horizon][:, horizon_idx, :])
                else:
                    forecast_parts.append(mx.zeros((batch_size, self.vocab_size)))
            forecast_cat = mx.concatenate(forecast_parts, axis=-1)

            x = self.embedding(chars[:, timestep])
            history, fast, mid, slow, hidden, _, base_features = self._encode_step(
                history, x, fast, mid, slow, timestep
            )
            output_features = self.predictive_output_proj(mx.stop_gradient(forecast_cat))
            logits.append(self.readout(mx.concatenate([base_features, output_features], axis=-1)))

            predictor_logits = self.predictive_output_predictor(hidden)
            predictor_prob = mx.softmax(
                predictor_logits.reshape(batch_size, horizon_count, self.vocab_size),
                axis=-1,
            )
            predictor_probs.append(predictor_prob)

        prediction_stack = mx.stack(predictor_probs, axis=1) if collect_predictions else None
        return mx.stack(logits, axis=1), prediction_stack

    def supervised_loss(self, x: mx.array, y: mx.array) -> mx.array:
        if self.use_predictive_output_channel:
            logits, prediction_stack = self._forward_with_predictive_output(x, collect_predictions=True)
            batch_size, timesteps, vocab_size = logits.shape
            main_loss = mx.mean(
                nn.losses.cross_entropy(logits.reshape(batch_size * timesteps, vocab_size), y.reshape(batch_size * timesteps))
            )
            if prediction_stack is None:
                return main_loss
            observed = mx.concatenate([x, y[:, -1:]], axis=1)
            brier_terms = []
            for horizon_idx, horizon in enumerate(self.predictive_residual_horizons):
                valid_steps = timesteps + 1 - horizon
                if valid_steps <= 0:
                    continue
                predicted = prediction_stack[:, :valid_steps, horizon_idx, :]
                targets = self._one_hot(observed[:, horizon : timesteps + 1])
                brier_terms.append(mx.mean((predicted - targets) ** 2))
            if not brier_terms:
                return main_loss
            output_loss = mx.mean(mx.stack(brier_terms))
            return main_loss + self.predictive_residual_loss_weight * output_loss
        if self.use_predictive_residual_channel:
            logits, prediction_stack = self._forward_with_predictive_residual(x, collect_predictions=True)
            batch_size, timesteps, vocab_size = logits.shape
            main_loss = mx.mean(
                nn.losses.cross_entropy(logits.reshape(batch_size * timesteps, vocab_size), y.reshape(batch_size * timesteps))
            )
            if prediction_stack is None:
                return main_loss
            observed = mx.concatenate([x, y[:, -1:]], axis=1)
            brier_terms = []
            for horizon_idx, horizon in enumerate(self.predictive_residual_horizons):
                valid_steps = timesteps + 1 - horizon
                if valid_steps <= 0:
                    continue
                predicted = prediction_stack[:, :valid_steps, horizon_idx, :]
                targets = self._one_hot(observed[:, horizon : timesteps + 1])
                brier_terms.append(mx.mean((predicted - targets) ** 2))
            if not brier_terms:
                return main_loss
            residual_loss = mx.mean(mx.stack(brier_terms))
            return main_loss + self.predictive_residual_loss_weight * residual_loss
        if not self.use_hypothesis_error_channel:
            logits = self(x)
            batch_size, timesteps, vocab_size = logits.shape
            return mx.mean(
                nn.losses.cross_entropy(logits.reshape(batch_size * timesteps, vocab_size), y.reshape(batch_size * timesteps))
            )
        logits, hypothesis, aux_targets = self._forward_with_hypothesis(x, collect_targets=True)
        batch_size, timesteps, vocab_size = logits.shape
        main_loss = mx.mean(
            nn.losses.cross_entropy(logits.reshape(batch_size * timesteps, vocab_size), y.reshape(batch_size * timesteps))
        )
        if aux_targets is None or timesteps < 2:
            return main_loss
        prediction_loss = mx.mean((hypothesis[:, :-1, :] - aux_targets[:, 1:, :]) ** 2)
        return main_loss + self.hypothesis_loss_weight * prediction_loss

    def __call__(self, chars: mx.array) -> mx.array:
        if self.use_random_third_channel:
            batch_size, timesteps = chars.shape
            history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
            fast = mx.zeros((batch_size, self.fast_size))
            mid = mx.zeros((batch_size, self.mid_size))
            slow = mx.zeros((batch_size, self.slow_size))
            logits = []

            for timestep in range(timesteps):
                x = self.embedding(chars[:, timestep])
                history, fast, mid, slow, hidden, _, base_features = self._encode_step(
                    history, x, fast, mid, slow, timestep
                )
                random_features = mx.stop_gradient(hidden @ self.random_third_proj)
                logits.append(self.readout(mx.concatenate([base_features, random_features], axis=-1)))
            return mx.stack(logits, axis=1)
        if self.use_predictive_output_channel:
            logits, _ = self._forward_with_predictive_output(chars, collect_predictions=False)
            return logits
        if self.use_predictive_residual_channel:
            logits, _ = self._forward_with_predictive_residual(chars, collect_predictions=False)
            return logits
        if self.use_hypothesis_error_channel:
            logits, _, _ = self._forward_with_hypothesis(chars, collect_targets=False)
            return logits
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        logits = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            history, fast, mid, slow, _, _, readout_input = self._encode_step(history, x, fast, mid, slow, timestep)
            logits.append(self.readout(readout_input))

        return mx.stack(logits, axis=1)

    def channel_trace(self, chars: mx.array) -> dict[str, mx.array]:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        aux_norms = []
        hidden_norms = []
        third_norms = []
        third_delta_norms = []
        prev_third = None

        if self.use_predictive_output_channel:
            predictor_probs: list[mx.array] = []
            horizon_count = len(self.predictive_residual_horizons)
            for timestep in range(timesteps):
                forecast_parts = []
                for horizon_idx, horizon in enumerate(self.predictive_residual_horizons):
                    if timestep >= horizon:
                        forecast_parts.append(predictor_probs[timestep - horizon][:, horizon_idx, :])
                    else:
                        forecast_parts.append(mx.zeros((batch_size, self.vocab_size)))
                forecast_cat = mx.concatenate(forecast_parts, axis=-1)

                x = self.embedding(chars[:, timestep])
                history, fast, mid, slow, hidden, aux_features, _ = self._encode_step(
                    history, x, fast, mid, slow, timestep
                )
                third_features = self.predictive_output_proj(mx.stop_gradient(forecast_cat))
                aux_norms.append(self._feature_norm(aux_features))
                hidden_norms.append(self._feature_norm(hidden))
                third_norms.append(self._feature_norm(third_features))
                if prev_third is None:
                    third_delta_norms.append(mx.zeros((batch_size,)))
                else:
                    third_delta_norms.append(self._feature_norm(third_features - prev_third))
                prev_third = third_features

                predictor_logits = self.predictive_output_predictor(hidden)
                predictor_prob = mx.softmax(
                    predictor_logits.reshape(batch_size, horizon_count, self.vocab_size),
                    axis=-1,
                )
                predictor_probs.append(predictor_prob)
        elif self.use_predictive_residual_channel:
            predictor_probs = []
            horizon_count = len(self.predictive_residual_horizons)
            for timestep in range(timesteps):
                current_one_hot = self._one_hot(chars[:, timestep])
                residual_parts = []
                for horizon_idx, horizon in enumerate(self.predictive_residual_horizons):
                    if timestep >= horizon:
                        residual_parts.append(current_one_hot - predictor_probs[timestep - horizon][:, horizon_idx, :])
                    else:
                        residual_parts.append(mx.zeros((batch_size, self.vocab_size)))
                residual_cat = mx.concatenate(residual_parts, axis=-1)

                x = self.embedding(chars[:, timestep])
                history, fast, mid, slow, hidden, aux_features, _ = self._encode_step(
                    history, x, fast, mid, slow, timestep
                )
                third_features = self.predictive_residual_proj(mx.stop_gradient(residual_cat))
                aux_norms.append(self._feature_norm(aux_features))
                hidden_norms.append(self._feature_norm(hidden))
                third_norms.append(self._feature_norm(third_features))
                if prev_third is None:
                    third_delta_norms.append(mx.zeros((batch_size,)))
                else:
                    third_delta_norms.append(self._feature_norm(third_features - prev_third))
                prev_third = third_features

                predictor_logits = self.predictive_residual_predictor(hidden)
                predictor_prob = mx.softmax(
                    predictor_logits.reshape(batch_size, horizon_count, self.vocab_size),
                    axis=-1,
                )
                predictor_probs.append(predictor_prob)
        elif self.use_hypothesis_error_channel:
            prev_hypothesis = mx.zeros((batch_size, self.sf.shape[0]))
            for timestep in range(timesteps):
                x = self.embedding(chars[:, timestep])
                history, fast, mid, slow, hidden, aux_features, _ = self._encode_step(
                    history, x, fast, mid, slow, timestep
                )
                third_features = mx.stop_gradient(aux_features - prev_hypothesis)
                aux_norms.append(self._feature_norm(aux_features))
                hidden_norms.append(self._feature_norm(hidden))
                third_norms.append(self._feature_norm(third_features))
                if prev_third is None:
                    third_delta_norms.append(mx.zeros((batch_size,)))
                else:
                    third_delta_norms.append(self._feature_norm(third_features - prev_third))
                prev_third = third_features
                prev_hypothesis = self.hypothesis(hidden)
        else:
            for timestep in range(timesteps):
                x = self.embedding(chars[:, timestep])
                history, fast, mid, slow, hidden, aux_features, _ = self._encode_step(
                    history, x, fast, mid, slow, timestep
                )
                if self.use_random_third_channel:
                    third_features = mx.stop_gradient(hidden @ self.random_third_proj)
                else:
                    third_features = mx.zeros((batch_size, self.random_third_size))
                aux_norms.append(self._feature_norm(aux_features))
                hidden_norms.append(self._feature_norm(hidden))
                third_norms.append(self._feature_norm(third_features))
                if prev_third is None:
                    third_delta_norms.append(mx.zeros((batch_size,)))
                else:
                    third_delta_norms.append(self._feature_norm(third_features - prev_third))
                prev_third = third_features

        return {
            "aux_norm": mx.stack(aux_norms, axis=1),
            "hidden_norm": mx.stack(hidden_norms, axis=1),
            "third_norm": mx.stack(third_norms, axis=1),
            "third_delta_norm": mx.stack(third_delta_norms, axis=1),
        }


class RoutedHierarchicalModel(nn.Module):
    BRANCH_NAMES = (
        "v6",
        "fast_delay",
        "fast_mid_delay",
        "fast_only_recurrent",
    )

    def __init__(self, vocab_size: int, embedding_dim: int, config: RoutedHierarchicalConfig):
        super().__init__()
        self.fast_size = config.fast_size
        self.mid_size = config.mid_size
        self.slow_size = config.slow_size
        self.slow_update_stride = max(config.slow_update_stride, 1)
        self.aux_source: HierarchicalAuxSource = config.aux_source
        self.router_mode = config.router_mode
        self.branch_count = len(self.BRANCH_NAMES)
        controller_view_dim = config.controller_view_dim or config.controller_width

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.Wf = build_dense_matrix(
            config.fast_size,
            config.fast_connectivity,
            config.fast_spectral_radius,
            config.seed,
            topology=config.fast_topology,
            rewire_prob=config.fast_rewire_prob,
        )
        self.Wm = build_dense_matrix(
            config.mid_size,
            config.mid_connectivity,
            config.mid_spectral_radius,
            config.seed + 1,
            topology=config.mid_topology,
            rewire_prob=config.mid_rewire_prob,
        )
        self.Ws = build_dense_matrix(
            config.slow_size,
            config.slow_connectivity,
            config.slow_spectral_radius,
            config.seed + 2,
            topology=config.slow_topology,
            rewire_prob=config.slow_rewire_prob,
        )

        rng = np.random.default_rng(config.seed)
        self.Wif = mx.array(
            rng.standard_normal((config.fast_size, embedding_dim), dtype=np.float32) * config.input_scale
        )
        self.Wu1 = mx.array(
            rng.standard_normal((config.mid_size, config.fast_size), dtype=np.float32) * config.upward_scale
        )
        self.Wu2 = mx.array(
            rng.standard_normal((config.slow_size, config.mid_size), dtype=np.float32) * config.upward_scale
        )
        self.ps = _fixed_projection(config.slow_size, controller_view_dim)
        self.pm = _fixed_projection(config.mid_size, controller_view_dim)
        self.aux_proj = _fixed_projection(config.fast_size, config.fast_sample_size)

        self.fast_delay_idx = mx.array(np.arange(math.ceil(config.fast_size / embedding_dim), dtype=np.int32))
        self.mid_delay_idx = mx.array(np.arange(math.ceil(config.mid_size / embedding_dim), dtype=np.int32))
        self.slow_delay_idx = mx.array(np.arange(math.ceil(config.slow_size / embedding_dim), dtype=np.int32))
        self.max_delay_len = int(max(self.fast_delay_idx.shape[0], self.mid_delay_idx.shape[0], self.slow_delay_idx.shape[0]))

        self.p1 = nn.Linear(controller_view_dim * 2, config.controller_width)
        self.p2 = nn.Linear(config.controller_width, config.controller_width)
        self.p3 = nn.Linear(config.controller_width, config.fast_size)
        self.p3.weight = mx.zeros_like(self.p3.weight)
        self.p3.bias = mx.full(self.p3.bias.shape, config.mask_bias)

        sample_rng = np.random.default_rng(config.seed + 10)
        self.sf = mx.array(sample_rng.choice(config.fast_size, config.fast_sample_size, replace=False))
        self.sm = mx.array(sample_rng.choice(config.mid_size, config.mid_sample_size, replace=False))
        self.ss = mx.array(sample_rng.choice(config.slow_size, config.slow_sample_size, replace=False))

        self.feature_dim = config.fast_sample_size * 2 + config.mid_sample_size + config.slow_sample_size
        self.readout = MLP(self.feature_dim, config.readout_hidden, vocab_size)

        summary_dim = config.controller_width + 3
        if self.router_mode == "static":
            self.router_static = nn.Linear(1, self.branch_count)
            self.router_static.weight = mx.zeros_like(self.router_static.weight)
            self.router_static.bias = mx.zeros_like(self.router_static.bias)
        elif self.router_mode == "learned":
            self.router = MLP(summary_dim * self.branch_count, config.router_hidden, self.branch_count)

    def freeze_static(self) -> None:
        self.freeze(
            keys=[
                "Wf",
                "Wm",
                "Ws",
                "Wif",
                "Wu1",
                "Wu2",
                "ps",
                "pm",
                "aux_proj",
                "fast_delay_idx",
                "mid_delay_idx",
                "slow_delay_idx",
                "sf",
                "sm",
                "ss",
            ],
            strict=False,
        )

    def _delay_state(self, history: mx.array, delay_idx: mx.array, size: int) -> mx.array:
        delayed = mx.take(history, delay_idx, axis=1)
        flat = mx.reshape(delayed, (history.shape[0], -1))
        if flat.shape[1] < size:
            flat = mx.concatenate([flat, mx.zeros((history.shape[0], size - flat.shape[1]))], axis=-1)
        return mx.tanh(flat[:, :size])

    def _aux_features(self, fast: mx.array, prediction: mx.array) -> mx.array:
        if self.aux_source == "prediction":
            return prediction[:, self.sf]
        if self.aux_source == "zeros":
            return mx.zeros((fast.shape[0], self.sf.shape[0]))
        if self.aux_source == "random":
            return fast @ self.aux_proj
        raise ValueError(f"Unknown hierarchical aux source: {self.aux_source}")

    def _branch_outputs(self, fast: mx.array, mid: mx.array, slow: mx.array) -> tuple[mx.array, mx.array]:
        hidden = mx.concatenate([slow @ self.ps, mid @ self.pm], axis=-1)
        hidden = nn.gelu(self.p1(hidden))
        hidden = nn.gelu(self.p2(hidden))
        prediction = mx.sigmoid(self.p3(hidden))
        surprise = fast * (1.0 - prediction)
        features = mx.concatenate(
            [
                surprise[:, self.sf],
                self._aux_features(fast, prediction),
                mid[:, self.sm],
                slow[:, self.ss],
            ],
            axis=-1,
        )
        provisional_logits = self.readout(features)
        probs = mx.softmax(provisional_logits, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1, keepdims=True)
        sorted_probs = mx.sort(probs, axis=-1)
        margin = sorted_probs[:, -1:] - sorted_probs[:, -2:-1]
        surprise_level = mx.mean(mx.abs(surprise), axis=-1, keepdims=True)
        summary = mx.concatenate([hidden, entropy, margin, surprise_level], axis=-1)
        return features, summary

    def _routing_weights(self, summaries: list[mx.array]) -> mx.array:
        batch_size = summaries[0].shape[0]
        if self.router_mode == "equal":
            return mx.full((batch_size, self.branch_count), 1.0 / self.branch_count)
        if self.router_mode == "static":
            logits = self.router_static(mx.ones((batch_size, 1)))
            return mx.softmax(logits, axis=-1)
        if self.router_mode == "learned":
            router_input = mx.concatenate(summaries, axis=-1)
            logits = self.router(router_input)
            return mx.softmax(logits, axis=-1)
        raise ValueError(f"Unknown router mode: {self.router_mode}")

    def _forward(self, chars: mx.array, collect_routes: bool = False) -> tuple[mx.array, mx.array | None]:
        batch_size, timesteps = chars.shape
        history = mx.zeros((batch_size, self.max_delay_len, self.embedding.weight.shape[1]))
        fast_r = mx.zeros((batch_size, self.fast_size))
        mid_r_v6 = mx.zeros((batch_size, self.mid_size))
        mid_r_fast_delay = mx.zeros((batch_size, self.mid_size))
        slow_r_v6 = mx.zeros((batch_size, self.slow_size))
        slow_r_fast_delay = mx.zeros((batch_size, self.slow_size))
        slow_r_fast_mid_delay = mx.zeros((batch_size, self.slow_size))
        logits = []
        routes = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            history = mx.concatenate([x[:, None, :], history[:, :-1, :]], axis=1)

            fast_r = mx.stop_gradient(mx.tanh(fast_r @ self.Wf.T + x @ self.Wif.T))
            fast_d = self._delay_state(history, self.fast_delay_idx, self.fast_size)
            mid_d = self._delay_state(history, self.mid_delay_idx, self.mid_size)
            slow_d = self._delay_state(history, self.slow_delay_idx, self.slow_size)

            mid_r_v6 = mx.stop_gradient(mx.tanh(mid_r_v6 @ self.Wm.T + fast_r @ self.Wu1.T))
            mid_r_fast_delay = mx.stop_gradient(mx.tanh(mid_r_fast_delay @ self.Wm.T + fast_d @ self.Wu1.T))
            if timestep % self.slow_update_stride == 0:
                slow_r_v6 = mx.stop_gradient(mx.tanh(slow_r_v6 @ self.Ws.T + mid_r_v6 @ self.Wu2.T))
                slow_r_fast_delay = mx.stop_gradient(mx.tanh(slow_r_fast_delay @ self.Ws.T + mid_r_fast_delay @ self.Wu2.T))
                slow_r_fast_mid_delay = mx.stop_gradient(mx.tanh(slow_r_fast_mid_delay @ self.Ws.T + mid_d @ self.Wu2.T))

            branch_states = [
                (fast_r, mid_r_v6, slow_r_v6),
                (fast_d, mid_r_fast_delay, slow_r_fast_delay),
                (fast_d, mid_d, slow_r_fast_mid_delay),
                (fast_r, mid_d, slow_d),
            ]

            branch_features = []
            branch_summaries = []
            for fast, mid, slow in branch_states:
                features, summary = self._branch_outputs(fast, mid, slow)
                branch_features.append(features)
                branch_summaries.append(summary)

            weights = self._routing_weights(branch_summaries)
            mixed_features = mx.zeros_like(branch_features[0])
            for idx, features in enumerate(branch_features):
                mixed_features = mixed_features + weights[:, idx : idx + 1] * features
            logits.append(self.readout(mixed_features))
            if collect_routes:
                routes.append(weights)

        route_trace = mx.stack(routes, axis=1) if collect_routes else None
        return mx.stack(logits, axis=1), route_trace

    def __call__(self, chars: mx.array) -> mx.array:
        logits, _ = self._forward(chars, collect_routes=False)
        return logits

    def route_trace(self, chars: mx.array) -> mx.array:
        _, routes = self._forward(chars, collect_routes=True)
        if routes is None:
            raise RuntimeError("Route trace requested without collection.")
        return routes


class HormonalHierarchicalCarverModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, config: HormonalHierarchicalCarverConfig):
        super().__init__()
        self.fast_size = config.fast_size
        self.mid_size = config.mid_size
        self.slow_size = config.slow_size
        self.hormone_count = config.hormone_count
        self.noise_std = config.noise_std
        self.slow_update_stride = max(config.slow_update_stride, 1)
        self.use_hormone_predictor = config.use_hormone_predictor
        self.include_hormones_in_readout = config.include_hormones_in_readout
        controller_view_dim = config.controller_view_dim or config.controller_width
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.Wf = build_dense_matrix(
            config.fast_size,
            config.fast_connectivity,
            config.fast_spectral_radius,
            config.seed,
            topology=config.fast_topology,
            rewire_prob=config.fast_rewire_prob,
        )
        self.Wm = build_dense_matrix(
            config.mid_size,
            config.mid_connectivity,
            config.mid_spectral_radius,
            config.seed + 1,
            topology=config.mid_topology,
            rewire_prob=config.mid_rewire_prob,
        )
        self.Ws = build_dense_matrix(
            config.slow_size,
            config.slow_connectivity,
            config.slow_spectral_radius,
            config.seed + 2,
            topology=config.slow_topology,
            rewire_prob=config.slow_rewire_prob,
        )

        rng = np.random.default_rng(config.seed)
        self.Wif = mx.array(
            rng.standard_normal((config.fast_size, embedding_dim), dtype=np.float32) * config.input_scale
        )
        self.Wu1 = mx.array(
            rng.standard_normal((config.mid_size, config.fast_size), dtype=np.float32) * config.upward_scale
        )
        self.Wu2 = mx.array(
            rng.standard_normal((config.slow_size, config.mid_size), dtype=np.float32) * config.upward_scale
        )

        self.Wh = _fixed_projection(config.slow_size, config.hormone_count)
        self.Rf = _fixed_projection(config.hormone_count, config.fast_size)
        self.Rm = _fixed_projection(config.hormone_count, config.mid_size)
        self.ps = _fixed_projection(config.slow_size, controller_view_dim)
        self.pm = _fixed_projection(config.mid_size, controller_view_dim)

        predictor_in = config.hormone_count if config.use_hormone_predictor else controller_view_dim * 2
        self.p1 = nn.Linear(predictor_in, config.controller_width)
        self.p2 = nn.Linear(config.controller_width, config.controller_width)
        self.p3 = nn.Linear(config.controller_width, config.fast_size)
        self.p3.weight = mx.zeros_like(self.p3.weight)
        self.p3.bias = mx.full(self.p3.bias.shape, config.mask_bias)

        sample_rng = np.random.default_rng(config.seed + 10)
        self.sf = mx.array(sample_rng.choice(config.fast_size, config.fast_sample_size, replace=False))
        self.sm = mx.array(sample_rng.choice(config.mid_size, config.mid_sample_size, replace=False))
        self.ss = mx.array(sample_rng.choice(config.slow_size, config.slow_sample_size, replace=False))

        readout_in = config.fast_sample_size * 2 + config.mid_sample_size + config.slow_sample_size
        if config.include_hormones_in_readout:
            readout_in += config.hormone_count
        self.readout = MLP(readout_in, config.readout_hidden, vocab_size)

    def freeze_static(self) -> None:
        self.freeze(
            keys=["Wf", "Wm", "Ws", "Wif", "Wu1", "Wu2", "Wh", "Rf", "Rm", "ps", "pm", "sf", "sm", "ss"],
            strict=False,
        )

    def _noise(self, shape: tuple[int, int]) -> mx.array:
        if self.noise_std <= 0:
            return mx.zeros(shape)
        return mx.random.normal(shape) * self.noise_std

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        fast = mx.zeros((batch_size, self.fast_size))
        mid = mx.zeros((batch_size, self.mid_size))
        slow = mx.zeros((batch_size, self.slow_size))
        logits = []

        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])

            hormones = mx.tanh(slow @ self.Wh)
            gate_fast = mx.sigmoid(hormones @ self.Rf)
            gate_mid = mx.sigmoid(hormones @ self.Rm)

            fast_pre = fast @ self.Wf.T + x @ self.Wif.T + self._noise(fast.shape)
            fast = mx.stop_gradient(mx.tanh(fast_pre) * gate_fast)

            mid_pre = mid @ self.Wm.T + fast @ self.Wu1.T + self._noise(mid.shape)
            mid = mx.stop_gradient(mx.tanh(mid_pre) * gate_mid)

            if timestep % self.slow_update_stride == 0:
                slow_pre = slow @ self.Ws.T + mid @ self.Wu2.T
                if self.noise_std > 0:
                    slow_pre = slow_pre + self._noise(slow.shape) * 0.5
                slow = mx.stop_gradient(mx.tanh(slow_pre))

            if self.use_hormone_predictor:
                predictor = hormones
            else:
                predictor = mx.concatenate([slow @ self.ps, mid @ self.pm], axis=-1)

            hidden = nn.gelu(self.p1(predictor))
            hidden = nn.gelu(self.p2(hidden))
            prediction = mx.sigmoid(self.p3(hidden))
            surprise = fast * (1.0 - prediction)

            parts = [
                surprise[:, self.sf],
                prediction[:, self.sf],
                mid[:, self.sm],
                slow[:, self.ss],
            ]
            if self.include_hormones_in_readout:
                parts.append(hormones)
            logits.append(self.readout(mx.concatenate(parts, axis=-1)))

        return mx.stack(logits, axis=1)


class FrozenReadoutModel(nn.Module):
    def __init__(
        self,
        reservoir: ReservoirBundle,
        vocab_size: int,
        embedding_dim: int,
        reservoir_size: int,
        config: FrozenReadoutConfig,
    ):
        super().__init__()
        self.Wr = reservoir.Wr
        self.Wi = reservoir.Wi
        self.reservoir_size = reservoir_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.sample_size = config.sample_size
        if config.sample_size is not None:
            self.sample_idx = mx.array(np.random.choice(reservoir_size, config.sample_size, replace=False))
            readout_in = config.sample_size
        else:
            self.sample_idx = None
            readout_in = reservoir_size
        self.readout = MLP(readout_in, config.readout_hidden, vocab_size)

    def freeze_static(self) -> None:
        keys = ["Wr", "Wi"]
        if self.sample_idx is not None:
            keys.append("sample_idx")
        self.freeze(keys=keys, strict=False)

    def __call__(self, chars: mx.array) -> mx.array:
        batch_size, timesteps = chars.shape
        state = mx.zeros((batch_size, self.reservoir_size))
        logits = []
        for timestep in range(timesteps):
            x = self.embedding(chars[:, timestep])
            state = mx.stop_gradient(mx.tanh(state @ self.Wr.T + x @ self.Wi.T))
            features = state if self.sample_idx is None else state[:, self.sample_idx]
            logits.append(self.readout(features))
        return mx.stack(logits, axis=1)


@dataclass(frozen=True)
class ModelPreset:
    name: str
    description: str
    kind: str
    carver: CarverConfig | None = None
    hierarchical: HierarchicalCarverConfig | None = None
    hormonal: HormonalHierarchicalCarverConfig | None = None
    routed: RoutedHierarchicalConfig | None = None
    gru: GRUConfig | None = None
    frozen: FrozenReadoutConfig | None = None
