from __future__ import annotations

from typing import Iterable, List, Tuple

from conker.src.blinx5 import (
    DEFAULT_MASK_POLICY,
    DEFAULT_PHASES,
    MASK_POLICY_ADAPTIVE,
    Blinx5Compressed,
    Blinx5Phase,
    compress as _compress,
    phase_names as _phase_names,
    roundtrip_ok as _roundtrip_ok,
)


BRANCH_NAME = "blinx-5c"
MASK_POLICY = MASK_POLICY_ADAPTIVE


def compress(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx5Phase, ...] = DEFAULT_PHASES,
) -> Blinx5Compressed:
    return _compress(
        data,
        max_rounds=max_rounds,
        phases=phases,
        mask_policy=MASK_POLICY,
    )


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx5Phase, ...] = DEFAULT_PHASES,
) -> Tuple[Blinx5Compressed, bool]:
    return _roundtrip_ok(
        data,
        max_rounds=max_rounds,
        phases=phases,
        mask_policy=MASK_POLICY,
    )


def phase_names(phases: Iterable[Blinx5Phase]) -> List[str]:
    return _phase_names(phases)
