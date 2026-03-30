from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from giddy_up.attack import (
    ORACLE_POSITION_EXPORT_SCHEMA_VERSION,
    iter_oracle_position_labels,
    oracle_position_export_record,
)
from giddy_up.oracle import DEFAULT_SCAN_ROOTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-position oracle labels as JSONL.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to scan. Defaults to the local giddy-up repo surface.",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument(
        "--required-radius-max",
        type=int,
        help=(
            "Scan bidirectional contexts up to this radius when computing required_radius. "
            "Defaults to the export radius."
        ),
    )
    parser.add_argument("--jsonl-out", required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-positions-per-file", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.paths] if args.paths else [REPO_ROOT / root for root in DEFAULT_SCAN_ROOTS]
    output_path = Path(args.jsonl_out)
    required_radius_max = args.required_radius_max if args.required_radius_max is not None else args.radius

    row_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_oracle_position_labels(
            input_paths,
            args.radius,
            max_files=args.max_files,
            max_positions_per_file=args.max_positions_per_file,
            required_radius_max=required_radius_max,
        ):
            handle.write(json.dumps(oracle_position_export_record(row)) + "\n")
            row_count += 1

    summary = {
        "branch": "giddy-up-oracle-position-export",
        "goal": "per-position oracle label stream",
        "schema_version": ORACLE_POSITION_EXPORT_SCHEMA_VERSION,
        "scan_roots": [str(path) for path in input_paths],
        "radius": args.radius,
        "required_radius_max": required_radius_max,
        "row_count": row_count,
        "jsonl_out": str(output_path),
        "max_files": args.max_files,
        "max_positions_per_file": args.max_positions_per_file,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
