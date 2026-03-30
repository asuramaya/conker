from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
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

DEFAULT_FILES = (
    "README.md",
    "conker/docs/BLINX1.md",
    "conker/docs/BLINX4.md",
    "conker/docs/BLINX5.md",
    "conker/docs/BLINX_ORACLE.md",
    "conker/src/blinx3.py",
    "conker/src/blinx5.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a BLINX-5 variant ablation matrix over local files."
    )
    parser.add_argument("--json-out", default="conker/out/blinx5_variant_matrix.json")
    parser.add_argument("--md-out", default="conker/out/blinx5_variant_matrix.md")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=8)
    return parser.parse_args()


def run_case(task: dict[str, object]) -> dict[str, object]:
    module = VARIANTS[str(task["variant"])]
    path = REPO_ROOT / str(task["path"])
    data = path.read_bytes()
    compressed, ok = module.roundtrip_ok(
        data,
        max_rounds=int(task["max_rounds"]),
    )
    source_zlib_bytes = len(zlib.compress(data, 9))
    result = {
        "path": str(path.relative_to(REPO_ROOT)),
        "variant": str(task["variant"]),
        "branch": module.BRANCH_NAME,
        "mask_policy": module.MASK_POLICY,
        "max_rounds": int(task["max_rounds"]),
        "roundtrip_ok": ok,
        "source_zlib_bytes": source_zlib_bytes,
        **compressed.stats(),
    }
    result["delta_vs_source_bytes"] = int(result["zlib_bytes"]) - source_zlib_bytes
    result["improves_over_source"] = result["delta_vs_source_bytes"] < 0
    return result


def build_tasks(max_rounds: int) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for path in DEFAULT_FILES:
        for variant in VARIANTS:
            tasks.append(
                {
                    "path": path,
                    "variant": variant,
                    "max_rounds": max_rounds,
                }
            )
    return tasks


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    valid = [result for result in results if result["roundtrip_ok"]]
    by_variant: dict[str, list[dict[str, object]]] = {variant: [] for variant in VARIANTS}
    for result in valid:
        by_variant[str(result["variant"])].append(result)
    variant_summary = {}
    for variant, rows in by_variant.items():
        if not rows:
            variant_summary[variant] = {"count": 0}
            continue
        variant_summary[variant] = {
            "count": len(rows),
            "best_delta_vs_source_bytes": min(int(row["delta_vs_source_bytes"]) for row in rows),
            "mean_delta_vs_source_bytes": statistics.mean(
                int(row["delta_vs_source_bytes"]) for row in rows
            ),
            "mean_round_count": statistics.mean(int(row["round_count"]) for row in rows),
            "improvement_count": sum(1 for row in rows if row["improves_over_source"]),
        }
    best_by_file = {}
    for path in sorted({str(result["path"]) for result in valid}):
        rows = [result for result in valid if result["path"] == path]
        best_by_file[path] = min(
            rows,
            key=lambda result: (int(result["delta_vs_source_bytes"]), int(result["zlib_bytes"])),
        )
    return {
        "task_count": len(results),
        "valid_count": len(valid),
        "variant_summary": variant_summary,
        "best_by_file": best_by_file,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# BLINX-5 Variant Ablation",
        "",
        f"- tasks: `{summary['task_count']}`",
        f"- valid roundtrips: `{summary['valid_count']}`",
        "",
        "## Variants",
        "",
    ]
    for variant, variant_summary in summary["variant_summary"].items():
        lines.append(f"- `{variant}`: `{json.dumps(variant_summary, sort_keys=True)}`")
    lines.extend(["", "## Best By File", ""])
    for path, row in summary["best_by_file"].items():
        lines.append(
            "- `{}`: variant=`{}` delta=`{}` rounds=`{}` mask_formats=`{}`".format(
                path,
                row["variant"],
                row["delta_vs_source_bytes"],
                row["round_count"],
                row["mask_format_counts"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args.max_rounds)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(
        key=lambda result: (
            str(result["path"]),
            str(result["variant"]),
        )
    )
    summary = summarize(results)
    payload = {"summary": summary, "results": results}
    json_out = REPO_ROOT / str(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n")
    md_out = REPO_ROOT / str(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
