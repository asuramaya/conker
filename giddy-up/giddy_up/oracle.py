from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SCAN_ROOTS = (
    "README.md",
    "docs",
    "giddy_up",
    "scripts",
)


@dataclass(frozen=True)
class OracleRadiusStats:
    radius: int
    positions: int
    contexts: int
    deterministic_contexts: int
    globally_unique_contexts: int
    deterministic_positions: int
    globally_unique_positions: int
    ambiguous_positions: int
    deterministic_fraction: float
    globally_unique_fraction: float
    candidate_leq_2_fraction: float
    candidate_leq_4_fraction: float
    candidate_leq_8_fraction: float
    mean_branching_factor: float
    max_branching_factor: int


@dataclass(frozen=True)
class OracleFileStats:
    path: str
    size: int
    radii: list[OracleRadiusStats]


@dataclass(frozen=True)
class OracleCorpusStats:
    file_count: int
    total_bytes: int
    radii: list[int]
    files: list[OracleFileStats]


def _iter_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            if "__pycache__" in child.parts or ".git" in child.parts or "conker/out" in child.as_posix():
                continue
            files.append(child)
    return files


def _contexts_for_radius(data: bytes, radius: int) -> list[tuple[bytes, int]]:
    if len(data) < 2 * radius + 1:
        return []
    return [
        (data[index - radius : index] + data[index + 1 : index + 1 + radius], data[index])
        for index in range(radius, len(data) - radius)
    ]


def analyze_oracle(paths: Iterable[Path], radii: Iterable[int]) -> OracleCorpusStats:
    selected_files = _iter_files(paths)
    radius_list = sorted({radius for radius in radii if radius > 0})
    corpus_maps: dict[int, dict[bytes, Counter[int]]] = {
        radius: defaultdict(Counter) for radius in radius_list
    }
    per_file_contexts: dict[tuple[str, int], list[tuple[bytes, int]]] = {}
    total_bytes = 0

    for path in selected_files:
        data = path.read_bytes()
        total_bytes += len(data)
        for radius in radius_list:
            contexts = _contexts_for_radius(data, radius)
            per_file_contexts[(str(path), radius)] = contexts
            for context, center in contexts:
                corpus_maps[radius][context][center] += 1

    file_stats: list[OracleFileStats] = []
    for path in selected_files:
        data = path.read_bytes()
        radius_stats: list[OracleRadiusStats] = []
        for radius in radius_list:
            contexts = per_file_contexts[(str(path), radius)]
            total_positions = len(contexts)
            seen_contexts: set[bytes] = set()
            deterministic_contexts = 0
            globally_unique_contexts = 0
            deterministic_positions = 0
            globally_unique_positions = 0
            candidate_leq_2_positions = 0
            candidate_leq_4_positions = 0
            candidate_leq_8_positions = 0
            branching_sum = 0
            max_branching = 0
            for context, _center in contexts:
                seen_contexts.add(context)
                support = corpus_maps[radius][context]
                branching = len(support)
                branching_sum += branching
                if branching > max_branching:
                    max_branching = branching
                if branching == 1:
                    deterministic_positions += 1
                if branching <= 2:
                    candidate_leq_2_positions += 1
                if branching <= 4:
                    candidate_leq_4_positions += 1
                if branching <= 8:
                    candidate_leq_8_positions += 1
                if sum(support.values()) == 1:
                    globally_unique_positions += 1
            for context in seen_contexts:
                support = corpus_maps[radius][context]
                if len(support) == 1:
                    deterministic_contexts += 1
                if sum(support.values()) == 1:
                    globally_unique_contexts += 1
            ambiguous_positions = total_positions - deterministic_positions
            contexts_count = len(seen_contexts)
            mean_branching = (branching_sum / total_positions) if total_positions else 0.0
            radius_stats.append(
                OracleRadiusStats(
                    radius=radius,
                    positions=total_positions,
                    contexts=contexts_count,
                    deterministic_contexts=deterministic_contexts,
                    globally_unique_contexts=globally_unique_contexts,
                    deterministic_positions=deterministic_positions,
                    globally_unique_positions=globally_unique_positions,
                    ambiguous_positions=ambiguous_positions,
                    deterministic_fraction=(
                        deterministic_positions / total_positions if total_positions else 0.0
                    ),
                    globally_unique_fraction=(
                        globally_unique_positions / total_positions if total_positions else 0.0
                    ),
                    candidate_leq_2_fraction=(
                        candidate_leq_2_positions / total_positions if total_positions else 0.0
                    ),
                    candidate_leq_4_fraction=(
                        candidate_leq_4_positions / total_positions if total_positions else 0.0
                    ),
                    candidate_leq_8_fraction=(
                        candidate_leq_8_positions / total_positions if total_positions else 0.0
                    ),
                    mean_branching_factor=mean_branching,
                    max_branching_factor=max_branching,
                )
            )
        file_stats.append(OracleFileStats(path=str(path), size=len(data), radii=radius_stats))

    return OracleCorpusStats(
        file_count=len(selected_files),
        total_bytes=total_bytes,
        radii=radius_list,
        files=file_stats,
    )
