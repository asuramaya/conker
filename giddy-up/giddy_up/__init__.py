from .attack import (
    ORACLE_POSITION_EXPORT_FIELDS,
    ORACLE_POSITION_EXPORT_SCHEMA_VERSION,
    analyze_oracle_attack,
    iter_oracle_position_labels,
    oracle_position_export_record,
)
from .oracle import DEFAULT_SCAN_ROOTS, analyze_oracle

__all__ = [
    "DEFAULT_SCAN_ROOTS",
    "ORACLE_POSITION_EXPORT_FIELDS",
    "ORACLE_POSITION_EXPORT_SCHEMA_VERSION",
    "analyze_oracle",
    "analyze_oracle_attack",
    "iter_oracle_position_labels",
    "oracle_position_export_record",
    "structure_proxy_feature_arrays",
]


def structure_proxy_feature_arrays(*args, **kwargs):
    from .features import structure_proxy_feature_arrays as impl

    return impl(*args, **kwargs)
