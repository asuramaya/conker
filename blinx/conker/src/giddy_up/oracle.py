from __future__ import annotations

import sys
from pathlib import Path


CANONICAL_ROOT = Path(__file__).resolve().parents[4] / "giddy-up"
if not CANONICAL_ROOT.exists():
    raise ModuleNotFoundError("Canonical giddy-up repo not found at ../giddy-up relative to blinx.")
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))

from giddy_up.oracle import (
    DEFAULT_SCAN_ROOTS,
    OracleCorpusStats,
    OracleFileStats,
    OracleRadiusStats,
    _contexts_for_radius,
    _iter_files,
    analyze_oracle,
)

__all__ = [
    "DEFAULT_SCAN_ROOTS",
    "OracleCorpusStats",
    "OracleFileStats",
    "OracleRadiusStats",
    "_contexts_for_radius",
    "_iter_files",
    "analyze_oracle",
]
