from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import statistics
import sys
import zlib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conker.src.blinx1 import roundtrip_ok


DEFAULT_FILES = (
    "README.md",
    "HISTORY.md",
    "ROADMAP.md",
    "conker/docs/BLINX1.md",
    "conker/docs/CURRENT_FRONTIER.md",
    "conker/docs/COMPRESSION_MATRIX.md",
    "conker/docs/CONKER6.md",
    "conker/src/conker4b.py",
    "conker/src/conker10.py",
    "carving_machine/models.py",
)

DEFAULT_RADII = (
    (1,),
    (2,),
    (1, 2),
    (1, 2, 3, 4),
    (2, 4),
    (2, 4, 8),
)

PRESETS = {
    "smoke": {
        "selection_modes": ("profit", "discovery"),
        "radii": ((1,), (1, 2), (1, 2, 3, 4)),
        "min_occurrences": (2, 4),
        "min_removed": (4, 16),
        "max_rounds": (4, 8),
    },
    "medium": {
        "selection_modes": ("profit", "discovery"),
        "radii": ((1,), (2,), (1, 2), (1, 2, 3, 4)),
        "min_occurrences": (2, 3, 6),
        "min_removed": (4, 16, 32),
        "max_rounds": (4, 8),
    },
    "wide": {
        "selection_modes": ("profit", "discovery"),
        "radii": DEFAULT_RADII,
        "min_occurrences": (2, 3, 4, 6),
        "min_removed": (1, 4, 8, 16, 32),
        "max_rounds": (4, 8, 16),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BLINX-1 ablation matrix over local files.")
    parser.add_argument("--json-out", default="conker/out/blinx1_matrix.json")
    parser.add_argument("--md-out", default="conker/out/blinx1_matrix.md")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--preset", choices=tuple(PRESETS.keys()), default="medium")
    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
    )
    return parser.parse_args()


def run_case(task: dict[str, object]) -> dict[str, object]:
    path = Path(task["path"])
    data = path.read_bytes()
    candidate_radii = tuple(task["candidate_radii"])
    compressed, ok = roundtrip_ok(
        data,
        max_rounds=int(task["max_rounds"]),
        min_occurrences=int(task["min_occurrences"]),
        min_removed=int(task["min_removed"]),
        candidate_radii=candidate_radii,
        selection_mode=str(task["selection_mode"]),
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "path": str(path),
        "selection_mode": task["selection_mode"],
        "candidate_radii": list(candidate_radii),
        "max_rounds": int(task["max_rounds"]),
        "min_occurrences": int(task["min_occurrences"]),
        "min_removed": int(task["min_removed"]),
        "roundtrip_ok": ok,
        "source_zlib_bytes": source_zlib_bytes,
        **compressed.stats(),
    }
    result["delta_vs_source_bytes"] = int(result["zlib_bytes"]) - source_zlib_bytes
    result["improves_over_source"] = result["delta_vs_source_bytes"] < 0
    return result


def build_tasks(preset_name: str) -> list[dict[str, object]]:
    preset = PRESETS[preset_name]
    tasks: list[dict[str, object]] = []
    for path in DEFAULT_FILES:
        for selection_mode in preset["selection_modes"]:
            for candidate_radii in preset["radii"]:
                for min_occurrences in preset["min_occurrences"]:
                    for min_removed in preset["min_removed"]:
                        for max_rounds in preset["max_rounds"]:
                            tasks.append(
                                {
                                    "path": path,
                                    "selection_mode": selection_mode,
                                    "candidate_radii": candidate_radii,
                                    "min_occurrences": min_occurrences,
                                    "min_removed": min_removed,
                                    "max_rounds": max_rounds,
                                }
                            )
    return tasks


def summarize(results: list[dict[str, object]], preset_name: str) -> dict[str, object]:
    valid = [result for result in results if result["roundtrip_ok"]]
    deltas = [int(result["delta_vs_source_bytes"]) for result in valid]
    improvements = [result for result in valid if result["improves_over_source"]]
    best = sorted(valid, key=lambda result: (int(result["delta_vs_source_bytes"]), int(result["zlib_bytes"])))[:20]
    by_mode: dict[str, list[dict[str, object]]] = {"profit": [], "discovery": []}
    for result in valid:
        by_mode[str(result["selection_mode"])].append(result)
    mode_summary = {}
    for mode, rows in by_mode.items():
        if not rows:
            mode_summary[mode] = {"count": 0}
            continue
        mode_summary[mode] = {
            "count": len(rows),
            "best_delta_vs_source_bytes": min(int(row["delta_vs_source_bytes"]) for row in rows),
            "mean_delta_vs_source_bytes": statistics.mean(int(row["delta_vs_source_bytes"]) for row in rows),
            "mean_rounds": statistics.mean(int(row["round_count"]) for row in rows),
            "improvement_count": sum(1 for row in rows if row["improves_over_source"]),
        }
    per_file_best = {}
    for path in sorted({str(result["path"]) for result in valid}):
        rows = [result for result in valid if result["path"] == path]
        per_file_best[path] = min(
            rows,
            key=lambda result: (int(result["delta_vs_source_bytes"]), int(result["zlib_bytes"])),
        )
    return {
        "preset": preset_name,
        "task_count": len(results),
        "valid_count": len(valid),
        "improvement_count": len(improvements),
        "best_delta_vs_source_bytes": min(deltas) if deltas else None,
        "mean_delta_vs_source_bytes": statistics.mean(deltas) if deltas else None,
        "mode_summary": mode_summary,
        "best_overall": best,
        "best_by_file": per_file_best,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# BLINX-1 Ablation Matrix",
        "",
        f"- preset: `{summary['preset']}`",
        f"- tasks: `{summary['task_count']}`",
        f"- valid roundtrips: `{summary['valid_count']}`",
        f"- profitable cases vs source zlib: `{summary['improvement_count']}`",
        f"- best delta vs source bytes: `{summary['best_delta_vs_source_bytes']}`",
        f"- mean delta vs source bytes: `{summary['mean_delta_vs_source_bytes']}`",
        "",
        "## Modes",
        "",
    ]
    for mode, mode_summary in summary["mode_summary"].items():
        lines.append(f"- `{mode}`: `{json.dumps(mode_summary, sort_keys=True)}`")
    lines.extend(["", "## Best Overall", ""])
    for row in summary["best_overall"][:10]:
        lines.append(
            "- `{path}` mode=`{selection_mode}` radii=`{candidate_radii}` min_occ=`{min_occurrences}` min_removed=`{min_removed}` max_rounds=`{max_rounds}` delta=`{delta_vs_source_bytes}` rounds=`{round_count}` removed_frac=`{removed_fraction:.4f}`".format(
                **row
            )
        )
    lines.extend(["", "## Best By File", ""])
    for path, row in summary["best_by_file"].items():
        lines.append(
            "- `{}`: mode=`{}` radii=`{}` delta=`{}` rounds=`{}` removed_frac=`{:.4f}`".format(
                path,
                row["selection_mode"],
                row["candidate_radii"],
                row["delta_vs_source_bytes"],
                row["round_count"],
                row["removed_fraction"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args.preset)
    results: list[dict[str, object]] = []
    executor_cls = ProcessPoolExecutor
    if args.executor == "thread":
        executor_cls = ThreadPoolExecutor
    if args.executor == "auto":
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_case, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
        except PermissionError:
            results.clear()
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_case, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
    else:
        with executor_cls(max_workers=args.workers) as executor:
            futures = [executor.submit(run_case, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(
        key=lambda result: (
            str(result["path"]),
            str(result["selection_mode"]),
            tuple(result["candidate_radii"]),
            int(result["min_occurrences"]),
            int(result["min_removed"]),
            int(result["max_rounds"]),
        )
    )
    summary = summarize(results, args.preset)
    payload = {"summary": summary, "results": results}
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n")
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
