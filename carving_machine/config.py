from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ReservoirTopology = Literal["erdos_renyi", "small_world"]
TokenizerMode = Literal["char", "bpe_1024"]
StateSource = Literal["sculpted", "raw", "zeros"]
AuxSource = Literal["mask", "zeros", "random"]
HierarchicalAuxSource = Literal["prediction", "zeros", "random"]
SamplePolicy = Literal["random", "boundary", "mask_variance"]
RuntimeProfile = Literal["full", "pilot"]
PathwayGateSource = Literal["slow", "surprise"]
LayerMemoryMode = Literal["recurrent", "delay"]
RouterMode = Literal["equal", "static", "learned"]


@dataclass(frozen=True)
class Text8Config:
    path: str | None = None
    limit_chars: int = 10_000_000
    split: float = 0.9
    alphabet: str = " abcdefghijklmnopqrstuvwxyz"
    tokenizer: TokenizerMode = "char"
    bpe_vocab_size: int = 1024
    bpe_cache_path: str | None = None


@dataclass(frozen=True)
class RegimeSwitchConfig:
    root: str | None = None
    book_files: tuple[str, ...] = (
        "pg3200.txt",
        "pg5600.txt",
        "pg200.txt",
        "pg11800.txt",
    )
    split: float = 0.9
    alphabet: str = " abcdefghijklmnopqrstuvwxyz"
    tokenizer: TokenizerMode = "char"
    bpe_vocab_size: int = 1024
    bpe_cache_path: str | None = None
    skip_chars: int = 4_096
    chars_per_book: int = 1_500_000
    total_chars: int = 4_000_000
    block_chars: int = 2_048
    block_jitter: int = 512
    seed: int = 42


@dataclass(frozen=True)
class OuroborosConfig:
    prompt_len: int = 64
    rollout_len: int = 128
    num_prompts: int = 16
    boundary_band: int = 128


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    steps: int = 5_000
    log_every: int = 500
    eval_batches: int = 300
    seeds: tuple[int, ...] = (42,)


@dataclass(frozen=True)
class ReservoirConfig:
    size: int = 5_000
    connectivity: float = 0.1
    spectral_radius: float = 0.99
    embedding_dim: int = 32
    topology: ReservoirTopology = "erdos_renyi"
    rewire_prob: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class CarverConfig:
    projection_dim: int = 512
    controller_width: int = 512
    sample_size: int = 500
    sample_policy: SamplePolicy = "random"
    readout_hidden: tuple[int, ...] = (128,)
    state_source: StateSource = "sculpted"
    aux_source: AuxSource = "mask"
    mask_bias: float = -3.0


@dataclass(frozen=True)
class HierarchicalCarverConfig:
    fast_size: int = 1_000
    fast_spectral_radius: float = 0.7
    fast_connectivity: float = 0.15
    fast_topology: ReservoirTopology = "erdos_renyi"
    fast_rewire_prob: float = 0.1
    mid_size: int = 2_000
    mid_spectral_radius: float = 0.95
    mid_connectivity: float = 0.08
    mid_topology: ReservoirTopology = "erdos_renyi"
    mid_rewire_prob: float = 0.1
    slow_size: int = 5_000
    slow_spectral_radius: float = 0.999
    slow_connectivity: float = 0.04
    slow_topology: ReservoirTopology = "erdos_renyi"
    slow_rewire_prob: float = 0.1
    input_scale: float = 0.1
    upward_scale: float = 0.05
    slow_update_stride: int = 1
    controller_view_dim: int | None = None
    controller_width: int = 256
    fast_sample_size: int = 200
    mid_sample_size: int = 100
    slow_sample_size: int = 100
    readout_hidden: tuple[int, ...] = (128,)
    mask_bias: float = -3.0
    aux_source: HierarchicalAuxSource = "prediction"
    use_hypothesis_error_channel: bool = False
    hypothesis_hidden: tuple[int, ...] = (64,)
    hypothesis_loss_weight: float = 0.25
    use_predictive_residual_channel: bool = False
    use_predictive_output_channel: bool = False
    use_random_third_channel: bool = False
    random_third_sample_size: int | None = None
    predictive_residual_horizons: tuple[int, ...] = (1, 2, 4, 8)
    predictive_residual_hidden: tuple[int, ...] = (64,)
    predictive_residual_loss_weight: float = 0.25
    fast_memory_mode: LayerMemoryMode = "recurrent"
    mid_memory_mode: LayerMemoryMode = "recurrent"
    slow_memory_mode: LayerMemoryMode = "recurrent"
    use_pathway_gates: bool = False
    gate_fast_mid: bool = True
    gate_mid_slow: bool = True
    pathway_gate_source: PathwayGateSource = "slow"
    pathway_gate_bias: float = 1.5
    seed: int = 42


@dataclass(frozen=True)
class RoutedHierarchicalConfig:
    fast_size: int = 1_000
    fast_spectral_radius: float = 0.7
    fast_connectivity: float = 0.15
    fast_topology: ReservoirTopology = "erdos_renyi"
    fast_rewire_prob: float = 0.1
    mid_size: int = 2_000
    mid_spectral_radius: float = 0.95
    mid_connectivity: float = 0.08
    mid_topology: ReservoirTopology = "erdos_renyi"
    mid_rewire_prob: float = 0.1
    slow_size: int = 5_000
    slow_spectral_radius: float = 0.999
    slow_connectivity: float = 0.04
    slow_topology: ReservoirTopology = "erdos_renyi"
    slow_rewire_prob: float = 0.1
    input_scale: float = 0.1
    upward_scale: float = 0.05
    slow_update_stride: int = 1
    controller_view_dim: int | None = None
    controller_width: int = 256
    fast_sample_size: int = 200
    mid_sample_size: int = 100
    slow_sample_size: int = 100
    readout_hidden: tuple[int, ...] = (128,)
    mask_bias: float = -3.0
    aux_source: HierarchicalAuxSource = "prediction"
    router_mode: RouterMode = "learned"
    router_hidden: tuple[int, ...] = (128,)
    seed: int = 42


@dataclass(frozen=True)
class HormonalHierarchicalCarverConfig:
    fast_size: int = 1_000
    fast_spectral_radius: float = 0.7
    fast_connectivity: float = 0.15
    fast_topology: ReservoirTopology = "erdos_renyi"
    fast_rewire_prob: float = 0.1
    mid_size: int = 2_000
    mid_spectral_radius: float = 0.95
    mid_connectivity: float = 0.08
    mid_topology: ReservoirTopology = "erdos_renyi"
    mid_rewire_prob: float = 0.1
    slow_size: int = 5_000
    slow_spectral_radius: float = 0.999
    slow_connectivity: float = 0.04
    slow_topology: ReservoirTopology = "erdos_renyi"
    slow_rewire_prob: float = 0.1
    input_scale: float = 0.1
    upward_scale: float = 0.05
    slow_update_stride: int = 1
    controller_view_dim: int | None = None
    controller_width: int = 256
    hormone_count: int = 50
    noise_std: float = 0.0
    use_hormone_predictor: bool = False
    include_hormones_in_readout: bool = False
    fast_sample_size: int = 200
    mid_sample_size: int = 100
    slow_sample_size: int = 100
    readout_hidden: tuple[int, ...] = (128,)
    mask_bias: float = -3.0
    seed: int = 42


@dataclass(frozen=True)
class GRUConfig:
    hidden_size: int = 800
    readout_hidden: tuple[int, ...] = (1478,)


@dataclass(frozen=True)
class FrozenReadoutConfig:
    sample_size: int | None = 500
    readout_hidden: tuple[int, ...] = (128,)


@dataclass(frozen=True)
class RuntimeConfig:
    data: Text8Config = field(default_factory=Text8Config)
    regime_switch: RegimeSwitchConfig = field(default_factory=RegimeSwitchConfig)
    ouroboros: OuroborosConfig = field(default_factory=OuroborosConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    profile: RuntimeProfile = "full"


TRAIN_PROFILES: dict[RuntimeProfile, TrainConfig] = {
    "full": TrainConfig(),
    "pilot": TrainConfig(
        seq_len=64,
        batch_size=24,
        steps=1_000,
        log_every=100,
        eval_batches=50,
        seeds=(42,),
    ),
}


TRAIN_PROFILE_DESCRIPTIONS: dict[RuntimeProfile, str] = {
    "full": "baseline research run: 5k steps, 300 eval batches",
    "pilot": "fast probe: 1k steps, 50 eval batches, 1 seed",
}


def train_config_for_profile(profile: RuntimeProfile) -> TrainConfig:
    return TRAIN_PROFILES[profile]
