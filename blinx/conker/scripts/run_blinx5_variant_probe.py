from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src import blinx5a, blinx5b, blinx5c


VARIANTS = {
    "5a": blinx5a,
    "5b": blinx5b,
    "5c": blinx5c,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a BLINX-5 variant lossless-compression probe."
    )
    parser.add_argument("input", help="Path to a file to compress losslessly.")
    parser.add_argument("--variant", choices=tuple(VARIANTS.keys()), default="5c")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = VARIANTS[args.variant]
    input_path = Path(args.input)
    data = input_path.read_bytes()
    compressed, ok = module.roundtrip_ok(
        data,
        max_rounds=args.max_rounds,
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "branch": module.BRANCH_NAME,
        "goal": "lossless compression",
        "input_path": str(input_path),
        "phase_program": module.phase_names(module.DEFAULT_PHASES),
        "roundtrip_ok": ok,
        "source_zlib_bytes": source_zlib_bytes,
        **compressed.stats(),
    }
    result["delta_vs_source_bytes"] = int(result["zlib_bytes"]) - source_zlib_bytes
    text = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
