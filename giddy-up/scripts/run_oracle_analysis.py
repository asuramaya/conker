from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from giddy_up.oracle import DEFAULT_SCAN_ROOTS, analyze_oracle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure bidirectional uniqueness over local files.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to scan. Defaults to the local giddy-up repo surface.",
    )
    parser.add_argument("--radii", default="1,2,3,4")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    return parser.parse_args()


def _stats_to_dict(stats):
    return {
        "radius": stats.radius,
        "positions": stats.positions,
        "contexts": stats.contexts,
        "deterministic_contexts": stats.deterministic_contexts,
        "globally_unique_contexts": stats.globally_unique_contexts,
        "deterministic_positions": stats.deterministic_positions,
        "globally_unique_positions": stats.globally_unique_positions,
        "ambiguous_positions": stats.ambiguous_positions,
        "deterministic_fraction": stats.deterministic_fraction,
        "globally_unique_fraction": stats.globally_unique_fraction,
        "candidate_leq_2_fraction": stats.candidate_leq_2_fraction,
        "candidate_leq_4_fraction": stats.candidate_leq_4_fraction,
        "candidate_leq_8_fraction": stats.candidate_leq_8_fraction,
        "mean_branching_factor": stats.mean_branching_factor,
        "max_branching_factor": stats.max_branching_factor,
    }


def main() -> None:
    args = parse_args()
    radii = tuple(
        radius
        for radius in (int(part.strip()) for part in args.radii.split(",") if part.strip())
        if radius > 0
    )
    input_paths = [Path(path) for path in args.paths] if args.paths else [REPO_ROOT / root for root in DEFAULT_SCAN_ROOTS]
    corpus = analyze_oracle(input_paths, radii)

    result = {
        "branch": "giddy-up-oracle",
        "goal": "bidirectional uniqueness analysis",
        "scan_roots": [str(path) for path in input_paths],
        "radii": list(corpus.radii),
        "file_count": corpus.file_count,
        "total_bytes": corpus.total_bytes,
        "files": [
            {
                "path": file_stats.path,
                "size": file_stats.size,
                "radii": [_stats_to_dict(radius_stats) for radius_stats in file_stats.radii],
            }
            for file_stats in corpus.files
        ],
    }

    text = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    if args.md_out:
        lines = [
            "# Giddy-Up Oracle Analysis",
            "",
            f"- files scanned: {corpus.file_count}",
            f"- total bytes: {corpus.total_bytes}",
            f"- radii: {', '.join(str(radius) for radius in corpus.radii)}",
            "",
            "## Top Files By Deterministic Fraction",
        ]
        best_rows = []
        for file_stats in corpus.files:
            best = max(file_stats.radii, key=lambda item: item.deterministic_fraction, default=None)
            if best is not None:
                best_rows.append((best.deterministic_fraction, file_stats.path, best))
        for fraction, path, best in sorted(best_rows, reverse=True)[:8]:
            lines.append(
                f"- `{path}` radius `{best.radius}`: deterministic_fraction={fraction:.4f}, "
                f"globally_unique_fraction={best.globally_unique_fraction:.4f}, "
                f"candidate_leq_4_fraction={best.candidate_leq_4_fraction:.4f}, "
                f"positions={best.positions}, mean_branching={best.mean_branching_factor:.2f}"
            )
        lines.extend(
            [
                "",
                "## Corpus Summary",
            ]
        )
        for radius in corpus.radii:
            radius_rows = [file_stats.radii[corpus.radii.index(radius)] for file_stats in corpus.files]
            avg_det = sum(row.deterministic_fraction for row in radius_rows) / len(radius_rows) if radius_rows else 0.0
            avg_unique = sum(row.globally_unique_fraction for row in radius_rows) / len(radius_rows) if radius_rows else 0.0
            lines.append(
                f"- radius `{radius}`: mean deterministic_fraction={avg_det:.4f}, "
                f"mean globally_unique_fraction={avg_unique:.4f}, "
                f"mean candidate_leq_4_fraction="
                f"{(sum(row.candidate_leq_4_fraction for row in radius_rows) / len(radius_rows)) if radius_rows else 0.0:.4f}, "
                f"mean candidate_leq_8_fraction="
                f"{(sum(row.candidate_leq_8_fraction for row in radius_rows) / len(radius_rows)) if radius_rows else 0.0:.4f}"
            )
        Path(args.md_out).write_text("\n".join(lines) + "\n")
    print(text)


if __name__ == "__main__":
    main()
