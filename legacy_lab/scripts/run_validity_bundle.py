#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_repo_dir(name: str, env_var: str | None = None) -> Path:
    candidates: list[Path] = []
    if env_var:
        raw_value = os.environ.get(env_var)
        raw = Path(raw_value) if raw_value else None
        if raw:
            candidates.append(raw)
        candidates.append(workspace_root() / name)
    candidates.extend(
        [
            workspace_root() / name,
            workspace_root().parent / name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate repo '{name}'. Checked: {', '.join(str(path) for path in candidates)}")


def dumps_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(data) + "\n", encoding="utf-8")


def build_template_payload(bundle_id: str, candidate_id: str, summary: str) -> dict[str, Any]:
    claim = {
        "candidate_id": candidate_id,
        "requested_label": "Tier-1 reviewed",
        "summary": summary,
    }
    metrics = {
        "primary": {
            "bridge_bpb": None,
            "full_eval_bpb": None,
            "full_eval_int6_bpb": None,
            "artifact_bytes": None,
        }
    }
    provenance = {
        "source_repo": str((workspace_root() / "conker").resolve()),
        "source_root": str((workspace_root() / "conker" / "out").resolve()),
        "notes": [
            "Fill in the exact checkpoint, JSON result files, and reproduction commands for this candidate.",
            "Detector outputs from conker-detect should be attached via the manifest attachments list.",
        ],
    }
    audits = {
        "tier1": {
            "status": "pass",
            "summary": "Claim metadata present; fill in the concrete protocol and reproduction notes.",
        },
        "tier2": {
            "status": "missing",
            "summary": "Attach conker-detect structural or artifact-boundary outputs here.",
        },
        "tier3": {
            "status": "missing",
            "summary": "Attach conker-detect legality output here when a replay adapter exists.",
        },
    }
    manifest = {
        "bundle_id": bundle_id,
        "claim": "claim.json",
        "metrics": "metrics.json",
        "provenance": "provenance.json",
        "audits": "audits.json",
        "attachments": [],
    }
    return {
        "claim": claim,
        "metrics": metrics,
        "provenance": provenance,
        "audits": audits,
        "manifest": manifest,
    }


def cmd_template(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_template_payload(
        bundle_id=args.bundle_id,
        candidate_id=args.candidate_id or args.bundle_id,
        summary=args.summary,
    )
    write_json(out_dir / "claim.json", payload["claim"])
    write_json(out_dir / "metrics.json", payload["metrics"])
    write_json(out_dir / "provenance.json", payload["provenance"])
    write_json(out_dir / "audits.json", payload["audits"])
    write_json(out_dir / "validity_manifest.json", payload["manifest"])
    print(dumps_json({"status": "ok", "out_dir": str(out_dir), "manifest": str(out_dir / "validity_manifest.json")}))
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    ledger_root = find_repo_dir("conker-ledger", env_var="CONKER_LEDGER_ROOT")
    sys.path.insert(0, str((ledger_root / "src").resolve()))
    from conker_ledger.ledger import write_validity_bundle  # type: ignore

    manifest = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    result = write_validity_bundle(manifest, out_dir)
    text = dumps_json(result)
    print(text)
    if args.json:
        Path(args.json).resolve().write_text(text + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thin umbrella helper for conker -> conker-detect -> conker-ledger validity workflow."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_template = sub.add_parser("template", help="Write a starter validity manifest bundle in a directory")
    p_template.add_argument("out_dir")
    p_template.add_argument("--bundle-id", required=True)
    p_template.add_argument("--candidate-id")
    p_template.add_argument(
        "--summary",
        default="Fill in the candidate summary, then attach detector outputs and concrete metric evidence.",
    )
    p_template.set_defaults(func=cmd_template)

    p_bundle = sub.add_parser("bundle", help="Assemble a validity bundle via the sibling conker-ledger repo")
    p_bundle.add_argument("manifest")
    p_bundle.add_argument("out_dir")
    p_bundle.add_argument("--json")
    p_bundle.set_defaults(func=cmd_bundle)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
