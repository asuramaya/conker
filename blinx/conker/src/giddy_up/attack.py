from __future__ import annotations

import sys
from pathlib import Path


CANONICAL_ROOT = Path(__file__).resolve().parents[4] / "giddy-up"
if not CANONICAL_ROOT.exists():
    raise ModuleNotFoundError("Canonical giddy-up repo not found at ../giddy-up relative to blinx.")
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))

from giddy_up.attack import (
    ORACLE_POSITION_EXPORT_FIELDS,
    ORACLE_POSITION_EXPORT_SCHEMA_VERSION,
    OracleAttackCorpusStats,
    OracleAttackFileStats,
    OracleAttackRadiusStats,
    OraclePositionLabel,
    analyze_oracle_attack,
    iter_oracle_position_labels,
    oracle_position_export_record,
)

__all__ = [
    "ORACLE_POSITION_EXPORT_FIELDS",
    "ORACLE_POSITION_EXPORT_SCHEMA_VERSION",
    "OracleAttackCorpusStats",
    "OracleAttackFileStats",
    "OracleAttackRadiusStats",
    "OraclePositionLabel",
    "analyze_oracle_attack",
    "iter_oracle_position_labels",
    "oracle_position_export_record",
]
