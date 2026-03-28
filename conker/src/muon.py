from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


class Muon:
    def __init__(
        self,
        keys: list[str],
        params: dict[str, mx.array],
        *,
        momentum: float,
        backend_steps: int,
        momentum_warmup_start: float,
        momentum_warmup_steps: int,
        weight_decay: float,
    ):
        self.keys = keys
        self.momentum = momentum
        self.backend_steps = backend_steps
        self.momentum_warmup_start = momentum_warmup_start
        self.momentum_warmup_steps = momentum_warmup_steps
        self.weight_decay = weight_decay
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def current_momentum(self, step: int) -> float:
        if self.momentum_warmup_steps > 0:
            t = min(step / self.momentum_warmup_steps, 1.0)
            return (1.0 - t) * self.momentum_warmup_start + t * self.momentum
        return self.momentum

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], *, step: int, lr: float) -> dict[str, mx.array]:
        momentum = self.current_momentum(step)
        updated: dict[str, mx.array] = {}
        for k in self.keys:
            if k not in grads:
                continue
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            updated[k] = p * (1.0 - lr * self.weight_decay) - lr * (g_ortho * scale).astype(p.dtype)
        return updated


class SplitMuonAdam:
    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        adam_eps: float = 1e-8,
        muon_momentum: float = 0.95,
        muon_backend_steps: int = 5,
        muon_momentum_warmup_start: float = 0.85,
        muon_momentum_warmup_steps: int = 500,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        params = dict(nn.utils.tree_flatten(model.parameters()))
        trainable_names = {k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())}

        self.embed_keys = [
            k for k in trainable_names if k in params and params[k].ndim == 2 and "embedding" in k
        ]
        self.matrix_keys = [
            k
            for k in trainable_names
            if k in params and params[k].ndim == 2 and k not in self.embed_keys
        ]
        self.scalar_keys = [
            k for k in trainable_names if k in params and k not in self.embed_keys and k not in self.matrix_keys
        ]

        self.muon = Muon(
            self.matrix_keys,
            params,
            momentum=muon_momentum,
            backend_steps=muon_backend_steps,
            momentum_warmup_start=muon_momentum_warmup_start,
            momentum_warmup_steps=muon_momentum_warmup_steps,
            weight_decay=weight_decay,
        )
        self.adam_embed = optim.Adam(
            learning_rate=learning_rate,
            betas=[beta1, beta2],
            eps=adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=learning_rate,
            betas=[beta1, beta2],
            eps=adam_eps,
            bias_correction=True,
        )

    def _decay_params(self, params: dict[str, mx.array], keys: list[str]) -> dict[str, mx.array]:
        return {k: params[k] * (1.0 - self.learning_rate * self.weight_decay) for k in keys}

    def step(self, model: nn.Module, grads_tree: dict, step: int) -> None:
        params = dict(nn.utils.tree_flatten(model.parameters()))
        grads = dict(nn.utils.tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr=self.learning_rate))

        embed_grads = {k: grads[k] for k in self.embed_keys if k in grads}
        if embed_grads:
            self.adam_embed.learning_rate = self.learning_rate
            updated.update(self.adam_embed.apply_gradients(embed_grads, self._decay_params(params, list(embed_grads.keys()))))

        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        if scalar_grads:
            self.adam_scalar.learning_rate = self.learning_rate
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, self._decay_params(params, list(scalar_grads.keys()))))

        model.update(nn.utils.tree_unflatten(list(updated.items())))
        mx.eval(
            model.parameters(),
            self.adam_embed.state,
            self.adam_scalar.state,
            self.muon.buffers,
        )
