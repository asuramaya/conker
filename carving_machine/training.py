from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import OuroborosConfig, TrainConfig
from .data import CharDataset, Text8Data
from .models import HierarchicalCarverModel


@dataclass
class RunMetrics:
    seed: int
    params: int
    train_loss: float
    test_loss: float
    overfit_pct: float
    train_time_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random_module = getattr(mx, "random", None)
    if random_module is not None and hasattr(random_module, "seed"):
        random_module.seed(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(value.size for _, value in nn.utils.tree_flatten(model.trainable_parameters()))


def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    batch_size, timesteps, vocab_size = logits.shape
    return mx.mean(nn.losses.cross_entropy(logits.reshape(batch_size * timesteps, vocab_size), y.reshape(batch_size * timesteps)))


def train_loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    custom_loss = getattr(model, "supervised_loss", None)
    if callable(custom_loss):
        return custom_loss(x, y)
    return loss_fn(model, x, y)


def evaluate(model: nn.Module, dataset: Text8Data, train_config: TrainConfig, split: str) -> float:
    total = 0.0
    for _ in range(train_config.eval_batches):
        x, y = dataset.batch(split, train_config.batch_size, train_config.seq_len)
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        total += loss.item()
    return total / train_config.eval_batches


def evaluate_ouroboros(
    model: nn.Module,
    dataset: CharDataset,
    train_config: TrainConfig,
    ouroboros_config: OuroborosConfig,
    split: str = "test",
    near_boundaries: bool = False,
) -> dict[str, float]:
    curve = evaluate_ouroboros_curve(
        model,
        dataset,
        train_config,
        ouroboros_config,
        checkpoints=(ouroboros_config.rollout_len,),
        split=split,
        near_boundaries=near_boundaries,
    )
    return curve["checkpoints"][str(ouroboros_config.rollout_len)]


def evaluate_ouroboros_curve(
    model: nn.Module,
    dataset: CharDataset,
    train_config: TrainConfig,
    ouroboros_config: OuroborosConfig,
    checkpoints: tuple[int, ...],
    split: str = "test",
    near_boundaries: bool = False,
) -> dict[str, Any]:
    rollout_len = max(checkpoints)
    prompts, targets = dataset.rollout_batch(
        split=split,
        batch_size=ouroboros_config.num_prompts,
        prompt_len=ouroboros_config.prompt_len,
        rollout_len=rollout_len,
        near_boundaries=near_boundaries,
        boundary_band=ouroboros_config.boundary_band,
    )
    return evaluate_ouroboros_curve_batch(
        model,
        train_config,
        prompts,
        targets,
        checkpoints=checkpoints,
    )


def evaluate_ouroboros_curve_batch(
    model: nn.Module,
    train_config: TrainConfig,
    prompts: mx.array,
    targets: mx.array,
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    context = prompts
    rollout_len = max(checkpoints)
    total_loss = 0.0
    total_correct = 0.0
    checkpoint_set = set(checkpoints)
    curve: dict[str, dict[str, float]] = {}

    for step in range(rollout_len):
        logits = model(context)
        next_logits = logits[:, -1, :]
        target = targets[:, step]
        losses = nn.losses.cross_entropy(next_logits, target)
        prediction = mx.argmax(next_logits, axis=-1)
        mx.eval(losses, prediction)
        total_loss += float(mx.sum(losses).item())
        total_correct += float(mx.sum(prediction == target).item())
        context = mx.concatenate([context, prediction[:, None]], axis=1)
        if context.shape[1] > train_config.seq_len:
            context = context[:, -train_config.seq_len :]
        step_count = step + 1
        if step_count in checkpoint_set:
            total_tokens = int(prompts.shape[0]) * step_count
            curve[str(step_count)] = {
                "rollout_loss": total_loss / total_tokens,
                "match_rate": total_correct / total_tokens,
            }

    return {
        "checkpoints": curve,
    }


def _log_softmax(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _hierarchical_step(
    model: HierarchicalCarverModel,
    token: mx.array,
    fast: mx.array,
    mid: mx.array,
    slow: mx.array,
    timestep: int,
    Wf: mx.array,
    Wm: mx.array,
    Ws: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    x = model.embedding(token)
    gate_fm, gate_ms = model._pathway_gates(slow)
    fast = mx.tanh(fast @ Wf.T + x @ model.Wif.T)
    mid = mx.tanh(mid @ Wm.T + gate_fm * (fast @ model.Wu1.T))
    if timestep % model.slow_update_stride == 0:
        slow = mx.tanh(slow @ Ws.T + gate_ms * (mid @ model.Wu2.T))

    hidden = mx.concatenate([slow @ model.ps, mid @ model.pm], axis=-1)
    hidden = nn.gelu(model.p1(hidden))
    hidden = nn.gelu(model.p2(hidden))
    prediction = mx.sigmoid(model.p3(hidden))
    surprise = fast * (1.0 - prediction)
    readout_input = mx.concatenate(
        [
            surprise[:, model.sf],
            model._aux_features(fast, prediction),
            mid[:, model.sm],
            slow[:, model.ss],
        ],
        axis=-1,
    )
    logits = model.readout(readout_input)
    return fast, mid, slow, logits


def _hierarchical_warmup(
    model: HierarchicalCarverModel,
    prompt: mx.array,
    Wf: mx.array,
    Wm: mx.array,
    Ws: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    batch_size, prompt_len = prompt.shape
    fast = mx.zeros((batch_size, model.fast_size))
    mid = mx.zeros((batch_size, model.mid_size))
    slow = mx.zeros((batch_size, model.slow_size))
    logits = mx.zeros((batch_size, model.readout.out.weight.shape[0]))

    for timestep in range(prompt_len):
        fast, mid, slow, logits = _hierarchical_step(
            model,
            prompt[:, timestep],
            fast,
            mid,
            slow,
            timestep,
            Wf,
            Wm,
            Ws,
        )
    return fast, mid, slow, logits


def _hebbian_update(
    W: mx.array,
    state: mx.array,
    plasticity_rate: float,
    support_mask: mx.array,
) -> mx.array:
    if plasticity_rate <= 0:
        return W
    corr = (state.T @ state) / max(state.shape[0], 1)
    return W + (plasticity_rate * corr * support_mask)


def _homeostatic_hebbian_update(
    W: mx.array,
    state: mx.array,
    plasticity_rate: float,
    support_mask: mx.array,
    decay_rate: float,
    target_W: mx.array,
) -> mx.array:
    corr = (state.T @ state) / max(state.shape[0], 1)
    delta = plasticity_rate * corr * support_mask
    if decay_rate > 0:
        delta = delta - decay_rate * (W - target_W) * support_mask
    return W + delta


def _anchor_decay_update(
    W: mx.array,
    decay_rate: float,
    support_mask: mx.array,
    target_W: mx.array,
) -> mx.array:
    if decay_rate <= 0:
        return W
    return W - decay_rate * (W - target_W) * support_mask


def _mean_normalized_entropy(log_probs: mx.array) -> float:
    probs = mx.exp(log_probs)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    vocab = max(int(log_probs.shape[-1]), 2)
    normalized = entropy / np.log(vocab)
    mx.eval(normalized)
    return float(mx.mean(normalized).item())


def _approx_spectral_radius(matrix: mx.array, seed: int = 0, iterations: int = 8) -> float:
    W = np.array(matrix)
    if W.size == 0 or not np.any(W):
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(W.shape[0], dtype=np.float32)
    for _ in range(iterations):
        Wv = W @ v
        norm = np.linalg.norm(Wv)
        if norm <= 1e-12:
            return 0.0
        v = Wv / norm
    return float(np.linalg.norm(W @ v))


def evaluate_hebbian_ouroboros_curve(
    model: HierarchicalCarverModel,
    dataset: CharDataset,
    train_config: TrainConfig,
    ouroboros_config: OuroborosConfig,
    plasticity_rate: float,
    checkpoints: tuple[int, ...],
    split: str = "test",
    near_boundaries: bool = False,
) -> dict[str, Any]:
    rollout_len = max(checkpoints)
    prompts, targets = dataset.rollout_batch(
        split=split,
        batch_size=ouroboros_config.num_prompts,
        prompt_len=ouroboros_config.prompt_len,
        rollout_len=rollout_len,
        near_boundaries=near_boundaries,
        boundary_band=ouroboros_config.boundary_band,
    )

    base_Wf = mx.array(np.array(model.Wf))
    base_Wm = mx.array(np.array(model.Wm))
    base_Ws = mx.array(np.array(model.Ws))
    mask_f = (mx.abs(base_Wf) > 0).astype(base_Wf.dtype)
    mask_m = (mx.abs(base_Wm) > 0).astype(base_Wm.dtype)
    mask_s = (mx.abs(base_Ws) > 0).astype(base_Ws.dtype)

    self_Wf = base_Wf
    self_Wm = base_Wm
    self_Ws = base_Ws
    teacher_Wf = base_Wf
    teacher_Wm = base_Wm
    teacher_Ws = base_Ws

    self_fast, self_mid, self_slow, self_logits = _hierarchical_warmup(model, prompts, self_Wf, self_Wm, self_Ws)
    teacher_fast, teacher_mid, teacher_slow, teacher_logits = self_fast, self_mid, self_slow, self_logits

    checkpoint_set = set(checkpoints)
    curve: dict[str, dict[str, Any]] = {}
    step_curves = {
        "rollout_loss": [],
        "match_rate": [],
        "teacher_kl": [],
        "teacher_argmax_agreement": [],
    }

    total_loss = 0.0
    total_correct = 0.0
    total_kl = 0.0
    total_teacher_agreement = 0.0

    for step in range(rollout_len):
        target = targets[:, step]
        losses = nn.losses.cross_entropy(self_logits, target)
        self_prediction = mx.argmax(self_logits, axis=-1)
        teacher_prediction = mx.argmax(teacher_logits, axis=-1)
        self_log_probs = _log_softmax(self_logits)
        teacher_log_probs = _log_softmax(teacher_logits)
        teacher_probs = mx.exp(teacher_log_probs)
        kl = mx.sum(teacher_probs * (teacher_log_probs - self_log_probs), axis=-1)
        agreement = (self_prediction == teacher_prediction)
        correct = (self_prediction == target)

        mx.eval(losses, self_prediction, teacher_prediction, kl, agreement, correct)

        step_loss = float(mx.mean(losses).item())
        step_match = float(mx.mean(correct).item())
        step_kl = float(mx.mean(kl).item())
        step_agreement = float(mx.mean(agreement).item())

        step_curves["rollout_loss"].append(step_loss)
        step_curves["match_rate"].append(step_match)
        step_curves["teacher_kl"].append(step_kl)
        step_curves["teacher_argmax_agreement"].append(step_agreement)

        total_loss += float(mx.sum(losses).item())
        total_correct += float(mx.sum(correct).item())
        total_kl += float(mx.sum(kl).item())
        total_teacher_agreement += float(mx.sum(agreement).item())

        self_Wf = _hebbian_update(self_Wf, self_fast, plasticity_rate, mask_f)
        self_Wm = _hebbian_update(self_Wm, self_mid, plasticity_rate, mask_m)
        self_Ws = _hebbian_update(self_Ws, self_slow, plasticity_rate, mask_s)
        teacher_Wf = _hebbian_update(teacher_Wf, teacher_fast, plasticity_rate, mask_f)
        teacher_Wm = _hebbian_update(teacher_Wm, teacher_mid, plasticity_rate, mask_m)
        teacher_Ws = _hebbian_update(teacher_Ws, teacher_slow, plasticity_rate, mask_s)

        step_count = step + 1
        if step_count in checkpoint_set:
            total_tokens = ouroboros_config.num_prompts * step_count
            curve[str(step_count)] = {
                "rollout_loss": total_loss / total_tokens,
                "match_rate": total_correct / total_tokens,
                "teacher_kl": total_kl / total_tokens,
                "teacher_argmax_agreement": total_teacher_agreement / total_tokens,
                "spectral_radius": {
                    "fast": _approx_spectral_radius(self_Wf, seed=step_count + 11),
                    "mid": _approx_spectral_radius(self_Wm, seed=step_count + 17),
                    "slow": _approx_spectral_radius(self_Ws, seed=step_count + 23),
                },
                "state_rms": {
                    "fast": float(mx.sqrt(mx.mean(self_fast * self_fast)).item()),
                    "mid": float(mx.sqrt(mx.mean(self_mid * self_mid)).item()),
                    "slow": float(mx.sqrt(mx.mean(self_slow * self_slow)).item()),
                },
            }

        if step == rollout_len - 1:
            continue

        next_timestep = ouroboros_config.prompt_len + step
        self_fast, self_mid, self_slow, self_logits = _hierarchical_step(
            model,
            self_prediction,
            self_fast,
            self_mid,
            self_slow,
            next_timestep,
            self_Wf,
            self_Wm,
            self_Ws,
        )
        teacher_fast, teacher_mid, teacher_slow, teacher_logits = _hierarchical_step(
            model,
            target,
            teacher_fast,
            teacher_mid,
            teacher_slow,
            next_timestep,
            teacher_Wf,
            teacher_Wm,
            teacher_Ws,
        )

    return {
        "plasticity_rate": plasticity_rate,
        "checkpoints": curve,
        "step_curves": step_curves,
    }


def evaluate_adaptive_hebbian_ouroboros_curve(
    model: HierarchicalCarverModel,
    dataset: CharDataset,
    train_config: TrainConfig,
    ouroboros_config: OuroborosConfig,
    base_rate: float,
    surprise_scale: float,
    decay_rate: float,
    checkpoints: tuple[int, ...],
    split: str = "test",
    near_boundaries: bool = False,
) -> dict[str, Any]:
    rollout_len = max(checkpoints)
    prompts, targets = dataset.rollout_batch(
        split=split,
        batch_size=ouroboros_config.num_prompts,
        prompt_len=ouroboros_config.prompt_len,
        rollout_len=rollout_len,
        near_boundaries=near_boundaries,
        boundary_band=ouroboros_config.boundary_band,
    )

    base_Wf = mx.array(np.array(model.Wf))
    base_Wm = mx.array(np.array(model.Wm))
    base_Ws = mx.array(np.array(model.Ws))
    mask_f = (mx.abs(base_Wf) > 0).astype(base_Wf.dtype)
    mask_m = (mx.abs(base_Wm) > 0).astype(base_Wm.dtype)
    mask_s = (mx.abs(base_Ws) > 0).astype(base_Ws.dtype)

    self_Wf = base_Wf
    self_Wm = base_Wm
    self_Ws = base_Ws
    teacher_Wf = base_Wf
    teacher_Wm = base_Wm
    teacher_Ws = base_Ws

    self_fast, self_mid, self_slow, self_logits = _hierarchical_warmup(model, prompts, self_Wf, self_Wm, self_Ws)
    teacher_fast, teacher_mid, teacher_slow, teacher_logits = self_fast, self_mid, self_slow, self_logits

    checkpoint_set = set(checkpoints)
    curve: dict[str, dict[str, Any]] = {}
    step_curves = {
        "rollout_loss": [],
        "match_rate": [],
        "teacher_kl": [],
        "teacher_argmax_agreement": [],
        "self_rate": [],
        "teacher_rate": [],
        "self_entropy": [],
        "teacher_entropy": [],
    }

    total_loss = 0.0
    total_correct = 0.0
    total_kl = 0.0
    total_teacher_agreement = 0.0
    total_self_rate = 0.0
    total_teacher_rate = 0.0
    total_self_entropy = 0.0
    total_teacher_entropy = 0.0

    for step in range(rollout_len):
        target = targets[:, step]
        losses = nn.losses.cross_entropy(self_logits, target)
        self_prediction = mx.argmax(self_logits, axis=-1)
        teacher_prediction = mx.argmax(teacher_logits, axis=-1)
        self_log_probs = _log_softmax(self_logits)
        teacher_log_probs = _log_softmax(teacher_logits)
        teacher_probs = mx.exp(teacher_log_probs)
        kl = mx.sum(teacher_probs * (teacher_log_probs - self_log_probs), axis=-1)
        agreement = self_prediction == teacher_prediction
        correct = self_prediction == target

        mx.eval(losses, self_prediction, teacher_prediction, kl, agreement, correct)

        self_entropy = _mean_normalized_entropy(self_log_probs)
        teacher_entropy = _mean_normalized_entropy(teacher_log_probs)
        self_rate = base_rate + surprise_scale * self_entropy
        teacher_rate = base_rate + surprise_scale * teacher_entropy

        step_loss = float(mx.mean(losses).item())
        step_match = float(mx.mean(correct).item())
        step_kl = float(mx.mean(kl).item())
        step_agreement = float(mx.mean(agreement).item())

        step_curves["rollout_loss"].append(step_loss)
        step_curves["match_rate"].append(step_match)
        step_curves["teacher_kl"].append(step_kl)
        step_curves["teacher_argmax_agreement"].append(step_agreement)
        step_curves["self_rate"].append(self_rate)
        step_curves["teacher_rate"].append(teacher_rate)
        step_curves["self_entropy"].append(self_entropy)
        step_curves["teacher_entropy"].append(teacher_entropy)

        total_loss += float(mx.sum(losses).item())
        total_correct += float(mx.sum(correct).item())
        total_kl += float(mx.sum(kl).item())
        total_teacher_agreement += float(mx.sum(agreement).item())
        total_self_rate += self_rate
        total_teacher_rate += teacher_rate
        total_self_entropy += self_entropy
        total_teacher_entropy += teacher_entropy

        self_Wf = _homeostatic_hebbian_update(self_Wf, self_fast, self_rate, mask_f, decay_rate, base_Wf)
        self_Wm = _homeostatic_hebbian_update(self_Wm, self_mid, self_rate, mask_m, decay_rate, base_Wm)
        self_Ws = _homeostatic_hebbian_update(self_Ws, self_slow, self_rate, mask_s, decay_rate, base_Ws)
        teacher_Wf = _homeostatic_hebbian_update(teacher_Wf, teacher_fast, teacher_rate, mask_f, decay_rate, base_Wf)
        teacher_Wm = _homeostatic_hebbian_update(teacher_Wm, teacher_mid, teacher_rate, mask_m, decay_rate, base_Wm)
        teacher_Ws = _homeostatic_hebbian_update(teacher_Ws, teacher_slow, teacher_rate, mask_s, decay_rate, base_Ws)

        step_count = step + 1
        if step_count in checkpoint_set:
            total_tokens = ouroboros_config.num_prompts * step_count
            curve[str(step_count)] = {
                "rollout_loss": total_loss / total_tokens,
                "match_rate": total_correct / total_tokens,
                "teacher_kl": total_kl / total_tokens,
                "teacher_argmax_agreement": total_teacher_agreement / total_tokens,
                "mean_self_rate": total_self_rate / step_count,
                "mean_teacher_rate": total_teacher_rate / step_count,
                "mean_self_entropy": total_self_entropy / step_count,
                "mean_teacher_entropy": total_teacher_entropy / step_count,
                "spectral_radius": {
                    "fast": _approx_spectral_radius(self_Wf, seed=step_count + 31),
                    "mid": _approx_spectral_radius(self_Wm, seed=step_count + 37),
                    "slow": _approx_spectral_radius(self_Ws, seed=step_count + 43),
                },
                "state_rms": {
                    "fast": float(mx.sqrt(mx.mean(self_fast * self_fast)).item()),
                    "mid": float(mx.sqrt(mx.mean(self_mid * self_mid)).item()),
                    "slow": float(mx.sqrt(mx.mean(self_slow * self_slow)).item()),
                },
            }

        if step == rollout_len - 1:
            continue

        next_timestep = ouroboros_config.prompt_len + step
        self_fast, self_mid, self_slow, self_logits = _hierarchical_step(
            model,
            self_prediction,
            self_fast,
            self_mid,
            self_slow,
            next_timestep,
            self_Wf,
            self_Wm,
            self_Ws,
        )
        teacher_fast, teacher_mid, teacher_slow, teacher_logits = _hierarchical_step(
            model,
            target,
            teacher_fast,
            teacher_mid,
            teacher_slow,
            next_timestep,
            teacher_Wf,
            teacher_Wm,
            teacher_Ws,
        )

    return {
        "base_rate": base_rate,
        "surprise_scale": surprise_scale,
        "decay_rate": decay_rate,
        "checkpoints": curve,
        "step_curves": step_curves,
    }


def evaluate_sleep_cycle_ouroboros_curve(
    model: HierarchicalCarverModel,
    dataset: CharDataset,
    train_config: TrainConfig,
    ouroboros_config: OuroborosConfig,
    base_rate: float,
    surprise_scale: float,
    wake_decay_rate: float,
    wake_chunk: int,
    sws_replay: int,
    sws_decay_rate: float,
    rem_replay: int,
    rem_decay_rate: float,
    rem_corruption_rate: float,
    checkpoints: tuple[int, ...],
    split: str = "test",
    near_boundaries: bool = False,
    rng_seed: int = 0,
) -> dict[str, Any]:
    rollout_len = max(checkpoints)
    prompts, targets = dataset.rollout_batch(
        split=split,
        batch_size=ouroboros_config.num_prompts,
        prompt_len=ouroboros_config.prompt_len,
        rollout_len=rollout_len,
        near_boundaries=near_boundaries,
        boundary_band=ouroboros_config.boundary_band,
    )

    base_Wf = mx.array(np.array(model.Wf))
    base_Wm = mx.array(np.array(model.Wm))
    base_Ws = mx.array(np.array(model.Ws))
    mask_f = (mx.abs(base_Wf) > 0).astype(base_Wf.dtype)
    mask_m = (mx.abs(base_Wm) > 0).astype(base_Wm.dtype)
    mask_s = (mx.abs(base_Ws) > 0).astype(base_Ws.dtype)

    self_Wf = base_Wf
    self_Wm = base_Wm
    self_Ws = base_Ws
    teacher_Wf = base_Wf
    teacher_Wm = base_Wm
    teacher_Ws = base_Ws

    self_fast, self_mid, self_slow, self_logits = _hierarchical_warmup(model, prompts, self_Wf, self_Wm, self_Ws)
    teacher_fast, teacher_mid, teacher_slow, teacher_logits = self_fast, self_mid, self_slow, self_logits

    checkpoint_set = set(checkpoints)
    curve: dict[str, dict[str, Any]] = {}
    step_curves = {
        "rollout_loss": [],
        "match_rate": [],
        "teacher_kl": [],
        "teacher_argmax_agreement": [],
        "self_rate": [],
        "teacher_rate": [],
        "phase": [],
    }

    total_loss = 0.0
    total_correct = 0.0
    total_kl = 0.0
    total_teacher_agreement = 0.0
    total_self_rate = 0.0
    total_teacher_rate = 0.0
    wake_steps_done = 0
    actual_steps = 0
    cycle_count = 0
    history_len = max(wake_chunk, sws_replay, rem_replay, 1)
    self_recent: list[np.ndarray] = []
    teacher_recent: list[np.ndarray] = []
    rng = np.random.default_rng(rng_seed)

    def append_recent(buffer: list[np.ndarray], token: mx.array) -> None:
        buffer.append(np.array(token, dtype=np.int32))
        if len(buffer) > history_len:
            del buffer[:-history_len]

    def recent_matrix(buffer: list[np.ndarray], length: int) -> np.ndarray | None:
        if length <= 0 or not buffer:
            return None
        tail = buffer[-min(length, len(buffer)) :]
        return np.stack(tail, axis=1)

    def corrupt_replay(tokens: np.ndarray) -> np.ndarray:
        if tokens.size == 0:
            return tokens
        corrupted = tokens.copy()
        if corrupted.shape[1] > 1:
            perm = rng.permutation(corrupted.shape[1])
            corrupted = corrupted[:, perm]
        if rem_corruption_rate > 0:
            corrupt_mask = rng.random(corrupted.shape) < rem_corruption_rate
            random_tokens = rng.integers(0, len(dataset.idx_to_char), size=corrupted.shape, dtype=np.int32)
            corrupted = np.where(corrupt_mask, random_tokens, corrupted)
        return corrupted.astype(np.int32)

    def run_sleep_phase(
        self_tokens: np.ndarray | None,
        teacher_tokens: np.ndarray | None,
        decay_rate: float,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        nonlocal self_fast, self_mid, self_slow, self_logits
        nonlocal teacher_fast, teacher_mid, teacher_slow, teacher_logits
        nonlocal self_Wf, self_Wm, self_Ws, teacher_Wf, teacher_Wm, teacher_Ws
        nonlocal actual_steps
        if self_tokens is None or teacher_tokens is None:
            return (
                self_fast,
                self_mid,
                self_slow,
                self_logits,
                teacher_fast,
                teacher_mid,
                teacher_slow,
                teacher_logits,
                self_Wf,
                self_Wm,
                self_Ws,
            )
        replay_len = min(self_tokens.shape[1], teacher_tokens.shape[1])
        for idx in range(replay_len):
            self_token = mx.array(self_tokens[:, idx])
            teacher_token = mx.array(teacher_tokens[:, idx])
            self_fast, self_mid, self_slow, self_logits = _hierarchical_step(
                model,
                self_token,
                self_fast,
                self_mid,
                self_slow,
                ouroboros_config.prompt_len + actual_steps,
                self_Wf,
                self_Wm,
                self_Ws,
            )
            teacher_fast, teacher_mid, teacher_slow, teacher_logits = _hierarchical_step(
                model,
                teacher_token,
                teacher_fast,
                teacher_mid,
                teacher_slow,
                ouroboros_config.prompt_len + actual_steps,
                teacher_Wf,
                teacher_Wm,
                teacher_Ws,
            )
            self_Wf = _anchor_decay_update(self_Wf, decay_rate, mask_f, base_Wf)
            self_Wm = _anchor_decay_update(self_Wm, decay_rate, mask_m, base_Wm)
            self_Ws = _anchor_decay_update(self_Ws, decay_rate, mask_s, base_Ws)
            teacher_Wf = _anchor_decay_update(teacher_Wf, decay_rate, mask_f, base_Wf)
            teacher_Wm = _anchor_decay_update(teacher_Wm, decay_rate, mask_m, base_Wm)
            teacher_Ws = _anchor_decay_update(teacher_Ws, decay_rate, mask_s, base_Ws)
            actual_steps += 1
        return (
            self_fast,
            self_mid,
            self_slow,
            self_logits,
            teacher_fast,
            teacher_mid,
            teacher_slow,
            teacher_logits,
            self_Wf,
            self_Wm,
            self_Ws,
        )

    while wake_steps_done < rollout_len:
        cycle_count += 1
        cycle_wake = min(wake_chunk, rollout_len - wake_steps_done)
        for _ in range(cycle_wake):
            target = targets[:, wake_steps_done]
            losses = nn.losses.cross_entropy(self_logits, target)
            self_prediction = mx.argmax(self_logits, axis=-1)
            teacher_prediction = mx.argmax(teacher_logits, axis=-1)
            self_log_probs = _log_softmax(self_logits)
            teacher_log_probs = _log_softmax(teacher_logits)
            teacher_probs = mx.exp(teacher_log_probs)
            kl = mx.sum(teacher_probs * (teacher_log_probs - self_log_probs), axis=-1)
            agreement = self_prediction == teacher_prediction
            correct = self_prediction == target

            mx.eval(losses, self_prediction, teacher_prediction, kl, agreement, correct, target)

            self_entropy = _mean_normalized_entropy(self_log_probs)
            teacher_entropy = _mean_normalized_entropy(teacher_log_probs)
            self_rate = base_rate + surprise_scale * self_entropy
            teacher_rate = base_rate + surprise_scale * teacher_entropy

            step_loss = float(mx.mean(losses).item())
            step_match = float(mx.mean(correct).item())
            step_kl = float(mx.mean(kl).item())
            step_agreement = float(mx.mean(agreement).item())

            step_curves["rollout_loss"].append(step_loss)
            step_curves["match_rate"].append(step_match)
            step_curves["teacher_kl"].append(step_kl)
            step_curves["teacher_argmax_agreement"].append(step_agreement)
            step_curves["self_rate"].append(self_rate)
            step_curves["teacher_rate"].append(teacher_rate)
            step_curves["phase"].append("wake")

            total_loss += float(mx.sum(losses).item())
            total_correct += float(mx.sum(correct).item())
            total_kl += float(mx.sum(kl).item())
            total_teacher_agreement += float(mx.sum(agreement).item())
            total_self_rate += self_rate
            total_teacher_rate += teacher_rate

            append_recent(self_recent, self_prediction)
            append_recent(teacher_recent, target)

            self_Wf = _homeostatic_hebbian_update(self_Wf, self_fast, self_rate, mask_f, wake_decay_rate, base_Wf)
            self_Wm = _homeostatic_hebbian_update(self_Wm, self_mid, self_rate, mask_m, wake_decay_rate, base_Wm)
            self_Ws = _homeostatic_hebbian_update(self_Ws, self_slow, self_rate, mask_s, wake_decay_rate, base_Ws)
            teacher_Wf = _homeostatic_hebbian_update(teacher_Wf, teacher_fast, teacher_rate, mask_f, wake_decay_rate, base_Wf)
            teacher_Wm = _homeostatic_hebbian_update(teacher_Wm, teacher_mid, teacher_rate, mask_m, wake_decay_rate, base_Wm)
            teacher_Ws = _homeostatic_hebbian_update(teacher_Ws, teacher_slow, teacher_rate, mask_s, wake_decay_rate, base_Ws)

            wake_steps_done += 1
            actual_steps += 1
            if wake_steps_done in checkpoint_set:
                total_tokens = ouroboros_config.num_prompts * wake_steps_done
                curve[str(wake_steps_done)] = {
                    "rollout_loss": total_loss / total_tokens,
                    "match_rate": total_correct / total_tokens,
                    "teacher_kl": total_kl / total_tokens,
                    "teacher_argmax_agreement": total_teacher_agreement / total_tokens,
                    "mean_self_rate": total_self_rate / wake_steps_done,
                    "mean_teacher_rate": total_teacher_rate / wake_steps_done,
                    "spectral_radius": {
                        "fast": _approx_spectral_radius(self_Wf, seed=wake_steps_done + 51),
                        "mid": _approx_spectral_radius(self_Wm, seed=wake_steps_done + 57),
                        "slow": _approx_spectral_radius(self_Ws, seed=wake_steps_done + 63),
                    },
                    "state_rms": {
                        "fast": float(mx.sqrt(mx.mean(self_fast * self_fast)).item()),
                        "mid": float(mx.sqrt(mx.mean(self_mid * self_mid)).item()),
                        "slow": float(mx.sqrt(mx.mean(self_slow * self_slow)).item()),
                    },
                    "actual_steps": actual_steps,
                    "cycle_count": cycle_count,
                }

            if wake_steps_done == rollout_len:
                break

            self_fast, self_mid, self_slow, self_logits = _hierarchical_step(
                model,
                self_prediction,
                self_fast,
                self_mid,
                self_slow,
                ouroboros_config.prompt_len + actual_steps,
                self_Wf,
                self_Wm,
                self_Ws,
            )
            teacher_fast, teacher_mid, teacher_slow, teacher_logits = _hierarchical_step(
                model,
                target,
                teacher_fast,
                teacher_mid,
                teacher_slow,
                ouroboros_config.prompt_len + actual_steps,
                teacher_Wf,
                teacher_Wm,
                teacher_Ws,
            )

        if wake_steps_done == rollout_len:
            break

        self_replay = recent_matrix(self_recent, sws_replay)
        teacher_replay = recent_matrix(teacher_recent, sws_replay)
        run_sleep_phase(self_replay, teacher_replay, sws_decay_rate)

        self_rem = recent_matrix(self_recent, rem_replay)
        teacher_rem = recent_matrix(teacher_recent, rem_replay)
        if self_rem is not None and teacher_rem is not None:
            run_sleep_phase(corrupt_replay(self_rem), corrupt_replay(teacher_rem), rem_decay_rate)

    return {
        "base_rate": base_rate,
        "surprise_scale": surprise_scale,
        "wake_decay_rate": wake_decay_rate,
        "wake_chunk": wake_chunk,
        "sws_replay": sws_replay,
        "sws_decay_rate": sws_decay_rate,
        "rem_replay": rem_replay,
        "rem_decay_rate": rem_decay_rate,
        "rem_corruption_rate": rem_corruption_rate,
        "checkpoints": curve,
        "step_curves": step_curves,
    }


def train_model(
    model: nn.Module,
    dataset: Text8Data,
    train_config: TrainConfig,
    seed: int,
    label: str,
    on_step: Callable[[int, nn.Module, list[float]], None] | None = None,
) -> RunMetrics:
    params = count_trainable_params(model)
    value_and_grad = nn.value_and_grad(model, train_loss_fn)
    optimizer = optim.AdamW(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    seed_everything(seed + 1000)
    losses: list[float] = []
    best = float("inf")
    start = time.time()

    for step in range(1, train_config.steps + 1):
        x, y = dataset.batch("train", train_config.batch_size, train_config.seq_len)
        loss, grads = value_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, max_norm=train_config.grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        current = loss.item()
        losses.append(current)
        if current < best:
            best = current

        if on_step is not None:
            on_step(step, model, losses)

        if step % train_config.log_every == 0:
            recent = float(np.mean(losses[-train_config.log_every :]))
            speed = (step * train_config.batch_size * train_config.seq_len) / max(time.time() - start, 1e-9)
            print(f"      {step:5d} | loss {recent:.4f} | best {best:.4f} | {speed:.0f} ch/s")

    elapsed = time.time() - start
    train_loss = evaluate(model, dataset, train_config, "train")
    test_loss = evaluate(model, dataset, train_config, "test")
    return RunMetrics(
        seed=seed,
        params=params,
        train_loss=train_loss,
        test_loss=test_loss,
        overfit_pct=(test_loss / train_loss - 1.0) * 100.0,
        train_time_sec=elapsed,
    )


def summarize_runs(label: str, runs: list[RunMetrics]) -> dict[str, Any]:
    tests = [run.test_loss for run in runs]
    overfits = [run.overfit_pct for run in runs]
    times = [run.train_time_sec for run in runs]
    return {
        "label": label,
        "params": runs[0].params if runs else 0,
        "test_mean": float(np.mean(tests)) if tests else 0.0,
        "test_std": float(np.std(tests)) if tests else 0.0,
        "overfit_mean": float(np.mean(overfits)) if overfits else 0.0,
        "time_mean_sec": float(np.mean(times)) if times else 0.0,
        "runs": [run.to_dict() for run in runs],
    }
