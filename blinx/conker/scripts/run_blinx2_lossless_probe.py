from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.blinx2 import roundtrip_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BLINX-2 global-rule lossless probe.")
    parser.add_argument("input", help="Path to a file to compress losslessly.")
    parser.add_argument("--selection-mode", choices=("profit", "discovery"), default="profit")
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--min-occurrences", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--top-candidates", type=int, default=64)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    data = input_path.read_bytes()
    compressed, ok = roundtrip_ok(
        data,
        min_len=args.min_len,
        max_len=args.max_len,
        min_occurrences=args.min_occurrences,
        max_rules=args.max_rules,
        top_candidates=args.top_candidates,
        selection_mode=args.selection_mode,
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "branch": "blinx-2",
        "goal": "lossless compression",
        "input_path": str(input_path),
        "selection_mode": args.selection_mode,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_occurrences": args.min_occurrences,
        "max_rules": args.max_rules,
        "top_candidates": args.top_candidates,
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
