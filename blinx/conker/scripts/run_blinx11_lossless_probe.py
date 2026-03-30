from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.blinx11 import DEFAULT_PHASES, phase_names, roundtrip_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BLINX-11 lane-aware candidate lossless-compression probe."
    )
    parser.add_argument("input", help="Path to a file to compress losslessly.")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--branch-limit", type=int, default=16)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    data = input_path.read_bytes()
    compressed, ok = roundtrip_ok(
        data,
        max_rounds=args.max_rounds,
        phases=DEFAULT_PHASES,
        branch_limit=args.branch_limit,
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "branch": "blinx-11",
        "admission": "lane_aware_entropy",
        "goal": "lossless compression",
        "input_path": str(input_path),
        "phase_program": phase_names(DEFAULT_PHASES),
        "roundtrip_ok": ok,
        "source_zlib_bytes": source_zlib_bytes,
        "branch_limit": args.branch_limit,
        **compressed.stats(),
    }
    result["delta_vs_source_bytes"] = int(result["zlib_bytes"]) - source_zlib_bytes
    text = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
