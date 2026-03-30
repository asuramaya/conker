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

from conker.src.blinx2 import roundtrip_ok


DEFAULT_FILES = (
    "README.md",
    "HISTORY.md",
    "ROADMAP.md",
    "conker/docs/BLINX1.md",
    "conker/docs/CURRENT_FRONTIER.md",
    "conker/docs/CONKER6.md",
    "conker/src/conker4b.py",
    "conker/src/conker10.py",
    "carving_machine/models.py",
)

PRESETS = {
    "smoke": {
        "selection_modes": ("profit", "discovery"),
        "min_lens": (4, 6),
        "max_lens": (8, 12),
        "min_occurrences": (2, 4),
        "max_rules": (4, 8),
        "top_candidates": (32,),
    },
    "medium": {
        "selection_modes": ("profit", "discovery"),
        "min_lens": (4, 6),
        "max_lens": (8, 12, 16),
        "min_occurrences": (2, 3, 4),
        "max_rules": (4, 8, 16),
        "top_candidates": (32, 64),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BLINX-2 global-rule matrix.")
    parser.add_argument("--json-out", default="conker/out/blinx2_matrix.json")
    parser.add_argument("--md-out", default="conker/out/blinx2_matrix.md")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--preset", choices=tuple(PRESETS.keys()), default="smoke")
    parser.add_argument("--executor", choices=("auto", "process", "thread"), default="auto")
    return parser.parse_args()


def run_case(task: dict[str, object]) -> dict[str, object]:
    path = Path(task["path"])
    data = path.read_bytes()
    compressed, ok = roundtrip_ok(
        data,
        min_len=int(task["min_len"]),
        max_len=int(task["max_len"]),
        min_occurrences=int(task["min_occurrences"]),
        max_rules=int(task["max_rules"]),
        top_candidates=int(task["top_candidates"]),
        selection_mode=str(task["selection_mode"]),
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "path": str(path),
        "selection_mode": task["selection_mode"],
        "min_len": int(task["min_len"]),
        "max_len": int(task["max_len"]),
        "min_occurrences": int(task["min_occurrences"]),
        "max_rules": int(task["max_rules"]),
        "top_candidates": int(task["top_candidates"]),
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
            for min_len in preset["min_lens"]:
                for max_len in preset["max_lens"]:
                    if max_len < min_len:
                        continue
                    for min_occurrences in preset["min_occurrences"]:
                        for max_rules in preset["max_rules"]:
                            for top_candidates in preset["top_candidates"]:
                                tasks.append(
                                    {
                                        "path": path,
                                        "selection_mode": selection_mode,
                                        "min_len": min_len,
                                        "max_len": max_len,
                                        "min_occurrences": min_occurrences,
                                        "max_rules": max_rules,
                                        "top_candidates": top_candidates,
                                    }
                                )
    return tasks


def summarize(results: list[dict[str, object]], preset_name: str) -> dict[str, object]:
    valid = [result for result in results if result["roundtrip_ok"]]
    deltas = [int(result["delta_vs_source_bytes"]) for result in valid]
    improvements = [result for result in valid if result["improves_over_source"]]
    best = sorted(valid, key=lambda result: (int(result["delta_vs_source_bytes"]), int(result["zlib_bytes"])))[:20]
    by_mode = {"profit": [], "discovery": []}
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
            "mean_rule_count": statistics.mean(int(row["rule_count"]) for row in rows),
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
        "# BLINX-2 Ablation Matrix",
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
            "- `{path}` mode=`{selection_mode}` len=`{min_len}-{max_len}` min_occ=`{min_occurrences}` max_rules=`{max_rules}` delta=`{delta_vs_source_bytes}` rules=`{rule_count}` replaced_frac=`{replaced_fraction:.4f}`".format(
                **row
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
            int(result["min_len"]),
            int(result["max_len"]),
            int(result["min_occurrences"]),
            int(result["max_rules"]),
            int(result["top_candidates"]),
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
