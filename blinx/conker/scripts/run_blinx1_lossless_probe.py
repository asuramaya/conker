from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.blinx1 import roundtrip_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BLINX-1 lossless hole-punch probe.")
    parser.add_argument("input", help="Path to a file to compress losslessly.")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--min-occurrences", type=int, default=2)
    parser.add_argument("--min-removed", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("profit", "discovery"),
        default="profit",
    )
    parser.add_argument(
        "--candidate-radii",
        default="1,2,3,4",
        help="Comma-separated context radii to try each round.",
    )
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    data = input_path.read_bytes()
    candidate_radii = tuple(
        radius
        for radius in (
            int(part.strip()) for part in args.candidate_radii.split(",") if part.strip()
        )
        if radius > 0
    )
    compressed, ok = roundtrip_ok(
        data,
        max_rounds=args.max_rounds,
        min_occurrences=args.min_occurrences,
        min_removed=args.min_removed,
        candidate_radii=candidate_radii,
        selection_mode=args.selection_mode,
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "branch": "blinx-1",
        "goal": "lossless compression",
        "input_path": str(input_path),
        "selection_mode": args.selection_mode,
        "candidate_radii": list(candidate_radii),
        "roundtrip_ok": ok,
        "source_zlib_bytes": source_zlib_bytes,
        **compressed.stats(),
    }
    result["delta_vs_source_bytes"] = result["zlib_bytes"] - source_zlib_bytes
    text = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
