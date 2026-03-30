from __future__ import annotations

import sys
from pathlib import Path


CANONICAL_ROOT = Path(__file__).resolve().parents[4] / "giddy-up"
if not CANONICAL_ROOT.exists():
    raise ModuleNotFoundError(
        "Canonical giddy-up repo not found at ../giddy-up relative to conker."
    )
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))

from giddy_up.features import structure_proxy_feature_arrays

__all__ = ["structure_proxy_feature_arrays"]
