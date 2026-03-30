from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.giddy_up.attack import analyze_oracle_attack
from conker.src.giddy_up.oracle import DEFAULT_SCAN_ROOTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attack BLINX oracle claims with leave-one-out, left-only, and rulebook-cost audits.")
    parser.add_argument("paths", nargs="*", help="Files or directories to scan. Defaults to the core BLINX tree.")
    parser.add_argument("--radii", default="1,2,3,4")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    return parser.parse_args()


def _stats_to_dict(stats):
    return {
        "radius": stats.radius,
        "positions": stats.positions,
        "bidi_inclusive_deterministic_fraction": stats.bidi_inclusive_deterministic_fraction,
        "bidi_leaveout_deterministic_fraction": stats.bidi_leaveout_deterministic_fraction,
        "bidi_inclusive_candidate4_fraction": stats.bidi_inclusive_candidate4_fraction,
        "bidi_leaveout_candidate4_fraction": stats.bidi_leaveout_candidate4_fraction,
        "left_leaveout_deterministic_fraction": stats.left_leaveout_deterministic_fraction,
        "left_leaveout_candidate4_fraction": stats.left_leaveout_candidate4_fraction,
        "self_inclusion_deterministic_uplift": stats.self_inclusion_deterministic_uplift,
        "self_inclusion_candidate4_uplift": stats.self_inclusion_candidate4_uplift,
        "future_context_deterministic_uplift": stats.future_context_deterministic_uplift,
        "future_context_candidate4_uplift": stats.future_context_candidate4_uplift,
        "leaveout_support_changed_fraction": stats.leaveout_support_changed_fraction,
        "rulebook_key_count": stats.rulebook_key_count,
        "rulebook_raw_bytes": stats.rulebook_raw_bytes,
        "rulebook_zlib_bytes": stats.rulebook_zlib_bytes,
        "removed_count": stats.removed_count,
        "removed_fraction": stats.removed_fraction,
        "mask_bytes": stats.mask_bytes,
        "naive_net_removed_bytes": stats.naive_net_removed_bytes,
    }


def main() -> None:
    args = parse_args()
    radii = tuple(
        radius
        for radius in (int(part.strip()) for part in args.radii.split(",") if part.strip())
        if radius > 0
    )
    input_paths = [Path(path) for path in args.paths] if args.paths else [REPO_ROOT / root for root in DEFAULT_SCAN_ROOTS]
    corpus = analyze_oracle_attack(input_paths, radii)

    result = {
        "branch": "blinx-oracle-attack",
        "goal": "attack bidirectional oracle claims with leave-one-out, prefix-only, and rulebook-cost audits",
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
            "# BLINX Oracle Attack",
            "",
            f"- files scanned: {corpus.file_count}",
            f"- total bytes: {corpus.total_bytes}",
            f"- radii: {', '.join(str(radius) for radius in corpus.radii)}",
            "",
            "## Corpus Summary",
        ]
        for radius in corpus.radii:
            radius_rows = [file_stats.radii[corpus.radii.index(radius)] for file_stats in corpus.files]
            if not radius_rows:
                continue
            mean = lambda attr: sum(getattr(row, attr) for row in radius_rows) / len(radius_rows)
            lines.append(
                f"- radius `{radius}`: "
                f"mean bidi inclusive det={mean('bidi_inclusive_deterministic_fraction'):.4f}, "
                f"mean bidi leave-one-out det={mean('bidi_leaveout_deterministic_fraction'):.4f}, "
                f"mean left-only leave-one-out det={mean('left_leaveout_deterministic_fraction'):.4f}, "
                f"mean self-inclusion candidate<=4 uplift={mean('self_inclusion_candidate4_uplift'):.4f}, "
                f"mean future-context candidate<=4 uplift={mean('future_context_candidate4_uplift'):.4f}, "
                f"mean naive net removed bytes={mean('naive_net_removed_bytes'):.2f}"
            )
        lines.extend(["", "## Top Files By Self-Inclusion Uplift"])
        top_self = []
        for file_stats in corpus.files:
            best = max(file_stats.radii, key=lambda item: item.self_inclusion_candidate4_uplift, default=None)
            if best is not None:
                top_self.append((best.self_inclusion_candidate4_uplift, file_stats.path, best))
        for uplift, path, best in sorted(top_self, reverse=True)[:8]:
            lines.append(
                f"- `{path}` radius `{best.radius}`: self_inclusion_candidate4_uplift={uplift:.4f}, "
                f"bidi_inclusive_candidate4={best.bidi_inclusive_candidate4_fraction:.4f}, "
                f"bidi_leaveout_candidate4={best.bidi_leaveout_candidate4_fraction:.4f}"
            )
        lines.extend(["", "## Top Files By Future-Context Uplift"])
        top_future = []
        for file_stats in corpus.files:
            best = max(file_stats.radii, key=lambda item: item.future_context_candidate4_uplift, default=None)
            if best is not None:
                top_future.append((best.future_context_candidate4_uplift, file_stats.path, best))
        for uplift, path, best in sorted(top_future, reverse=True)[:8]:
            lines.append(
                f"- `{path}` radius `{best.radius}`: future_context_candidate4_uplift={uplift:.4f}, "
                f"bidi_leaveout_candidate4={best.bidi_leaveout_candidate4_fraction:.4f}, "
                f"left_leaveout_candidate4={best.left_leaveout_candidate4_fraction:.4f}"
            )
        lines.extend(["", "## Worst Rulebook Break-Even Cases"])
        worst_rulebook = []
        for file_stats in corpus.files:
            worst = min(file_stats.radii, key=lambda item: item.naive_net_removed_bytes, default=None)
            if worst is not None:
                worst_rulebook.append((worst.naive_net_removed_bytes, file_stats.path, worst))
        for net_bytes, path, worst in sorted(worst_rulebook)[:8]:
            lines.append(
                f"- `{path}` radius `{worst.radius}`: naive_net_removed_bytes={net_bytes}, "
                f"removed_count={worst.removed_count}, rulebook_zlib_bytes={worst.rulebook_zlib_bytes}, mask_bytes={worst.mask_bytes}"
            )
        Path(args.md_out).write_text("\n".join(lines) + "\n")
    print(text)


if __name__ == "__main__":
    main()
