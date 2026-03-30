from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import zlib

from ._codec import build_unique_context_dictionary, select_removals
from .oracle import _contexts_for_radius, _iter_files


@dataclass(frozen=True)
class OracleAttackRadiusStats:
    radius: int
    positions: int
    bidi_inclusive_deterministic_fraction: float
    bidi_leaveout_deterministic_fraction: float
    bidi_inclusive_candidate4_fraction: float
    bidi_leaveout_candidate4_fraction: float
    left_leaveout_deterministic_fraction: float
    left_leaveout_candidate4_fraction: float
    self_inclusion_deterministic_uplift: float
    self_inclusion_candidate4_uplift: float
    future_context_deterministic_uplift: float
    future_context_candidate4_uplift: float
    leaveout_support_changed_fraction: float
    rulebook_key_count: int
    rulebook_raw_bytes: int
    rulebook_zlib_bytes: int
    removed_count: int
    removed_fraction: float
    mask_bytes: int
    naive_net_removed_bytes: int


@dataclass(frozen=True)
class OraclePositionLabel:
    path: str
    size: int
    radius: int
    position: int
    center: int
    center_hex: str
    left_context_hex: str
    right_context_hex: str
    bidi_inclusive_support_size: int
    bidi_leaveout_support_size: int
    left_leaveout_support_size: int
    teacher_candidate_tokens: list[int]
    teacher_candidate_counts: list[int]
    teacher_candidate_frequencies: list[float]
    bidi_inclusive_candidate4: bool
    bidi_leaveout_candidate4: bool
    left_leaveout_candidate4: bool
    candidate_set_leq_4: bool
    candidate_set_leq_8: bool
    required_radius: int | None
    future_uplift: int
    self_inclusion_uplift: int
    clean_bridge_score: float
    memory_trust: float
    bridge_confidence: float
    self_inclusion_support_changed: bool
    future_context_support_changed: bool


ORACLE_POSITION_EXPORT_SCHEMA_VERSION = 3
ORACLE_POSITION_EXPORT_FIELDS = (
    "schema_version",
    "path",
    "size",
    "radius",
    "position",
    "center",
    "center_hex",
    "left_context_hex",
    "right_context_hex",
    "bidi_inclusive_support_size",
    "bidi_leaveout_support_size",
    "left_leaveout_support_size",
    "teacher_candidate_tokens",
    "teacher_candidate_counts",
    "teacher_candidate_frequencies",
    "bidi_inclusive_candidate4",
    "bidi_leaveout_candidate4",
    "left_leaveout_candidate4",
    "candidate_set_leq_4",
    "candidate_set_leq_8",
    "required_radius",
    "future_uplift",
    "self_inclusion_uplift",
    "clean_bridge_score",
    "memory_trust",
    "bridge_confidence",
    "self_inclusion_support_changed",
    "future_context_support_changed",
)


def oracle_position_export_record(label: OraclePositionLabel) -> dict[str, object]:
    return {
        "schema_version": ORACLE_POSITION_EXPORT_SCHEMA_VERSION,
        "path": label.path,
        "size": label.size,
        "radius": label.radius,
        "position": label.position,
        "center": label.center,
        "center_hex": label.center_hex,
        "left_context_hex": label.left_context_hex,
        "right_context_hex": label.right_context_hex,
        "bidi_inclusive_support_size": label.bidi_inclusive_support_size,
        "bidi_leaveout_support_size": label.bidi_leaveout_support_size,
        "left_leaveout_support_size": label.left_leaveout_support_size,
        "teacher_candidate_tokens": label.teacher_candidate_tokens,
        "teacher_candidate_counts": label.teacher_candidate_counts,
        "teacher_candidate_frequencies": label.teacher_candidate_frequencies,
        "bidi_inclusive_candidate4": label.bidi_inclusive_candidate4,
        "bidi_leaveout_candidate4": label.bidi_leaveout_candidate4,
        "left_leaveout_candidate4": label.left_leaveout_candidate4,
        "candidate_set_leq_4": label.candidate_set_leq_4,
        "candidate_set_leq_8": label.candidate_set_leq_8,
        "required_radius": label.required_radius,
        "future_uplift": label.future_uplift,
        "self_inclusion_uplift": label.self_inclusion_uplift,
        "clean_bridge_score": label.clean_bridge_score,
        "memory_trust": label.memory_trust,
        "bridge_confidence": label.bridge_confidence,
        "self_inclusion_support_changed": label.self_inclusion_support_changed,
        "future_context_support_changed": label.future_context_support_changed,
    }


@dataclass(frozen=True)
class OracleAttackFileStats:
    path: str
    size: int
    radii: list[OracleAttackRadiusStats]


@dataclass(frozen=True)
class OracleAttackCorpusStats:
    file_count: int
    total_bytes: int
    radii: list[int]
    files: list[OracleAttackFileStats]


def _support_size(counter: Counter[int]) -> int:
    return sum(1 for count in counter.values() if count > 0)


def _subtract_counter(global_counter: Counter[int], local_counter: Counter[int]) -> Counter[int]:
    out: Counter[int] = Counter()
    for value, count in global_counter.items():
        remaining = count - local_counter.get(value, 0)
        if remaining > 0:
            out[value] = remaining
    return out


def _serialize_dictionary(dictionary: dict[bytes, int]) -> bytes:
    out = bytearray()
    for key in sorted(dictionary):
        out.extend(len(key).to_bytes(2, "little"))
        out.extend(key)
        out.append(int(dictionary[key]) & 0xFF)
    return bytes(out)


def iter_oracle_position_labels(
    paths: Iterable[Path],
    radius: int,
    *,
    max_files: int | None = None,
    max_positions_per_file: int | None = None,
    required_radius_max: int | None = None,
) -> Iterable[OraclePositionLabel]:
    selected_files = _iter_files(paths)
    if max_files is not None:
        selected_files = selected_files[:max_files]
    if radius <= 0:
        return

    required_radius_cap = radius if required_radius_max is None else max(radius, required_radius_max)
    required_radii = list(range(1, required_radius_cap + 1))

    bidi_global: dict[bytes, Counter[int]] = defaultdict(Counter)
    left_global: dict[bytes, Counter[int]] = defaultdict(Counter)
    bidi_global_by_radius: dict[int, dict[bytes, Counter[int]]] = {
        required_radius: defaultdict(Counter) for required_radius in required_radii
    }
    per_file_bidi: dict[str, list[tuple[bytes, int]]] = {}
    per_file_left: dict[str, list[tuple[bytes, int]]] = {}
    per_file_bidi_by_radius: dict[str, dict[int, list[tuple[bytes, int]]]] = {}
    per_file_bidi_local: dict[str, dict[bytes, Counter[int]]] = {}
    per_file_left_local: dict[str, dict[bytes, Counter[int]]] = {}
    file_bytes: dict[str, bytes] = {}

    for path in selected_files:
        data = path.read_bytes()
        path_str = str(path)
        file_bytes[path_str] = data
        bidi_contexts = _contexts_for_radius(data, radius)
        bidi_contexts_by_radius: dict[int, list[tuple[bytes, int]]] = {}
        for required_radius in required_radii:
            bidi_contexts_by_radius[required_radius] = _contexts_for_radius(data, required_radius)
        left_contexts = [
            (data[index - radius : index], data[index])
            for index in range(radius, len(data) - radius)
        ] if len(data) >= 2 * radius + 1 else []
        per_file_bidi[path_str] = bidi_contexts
        per_file_left[path_str] = left_contexts
        per_file_bidi_by_radius[path_str] = bidi_contexts_by_radius

        bidi_local: dict[bytes, Counter[int]] = defaultdict(Counter)
        for context, center in bidi_contexts:
            bidi_global[context][center] += 1
            bidi_local[context][center] += 1
        for required_radius, contexts in bidi_contexts_by_radius.items():
            for context, center in contexts:
                bidi_global_by_radius[required_radius][context][center] += 1
        per_file_bidi_local[path_str] = bidi_local

        left_local: dict[bytes, Counter[int]] = defaultdict(Counter)
        for context, center in left_contexts:
            left_global[context][center] += 1
            left_local[context][center] += 1
        per_file_left_local[path_str] = left_local

    for path in selected_files:
        path_str = str(path)
        data = file_bytes[path_str]
        bidi_contexts = per_file_bidi[path_str]
        left_contexts = per_file_left[path_str]
        bidi_contexts_by_radius = per_file_bidi_by_radius[path_str]
        bidi_local = per_file_bidi_local[path_str]
        left_local = per_file_left_local[path_str]
        total_positions = len(bidi_contexts)
        limit = total_positions if max_positions_per_file is None else min(total_positions, max_positions_per_file)
        for position, ((bidi_context, center), (left_context, _left_center)) in enumerate(
            zip(bidi_contexts, left_contexts)
        ):
            if position >= limit:
                break
            center_index = position + radius
            global_bidi = bidi_global[bidi_context]
            leaveout_bidi = _subtract_counter(global_bidi, bidi_local[bidi_context])
            global_left = left_global[left_context]
            leaveout_left = _subtract_counter(global_left, left_local[left_context])
            bidi_inclusive_support_size = _support_size(global_bidi)
            bidi_leaveout_support_size = _support_size(leaveout_bidi)
            left_leaveout_support_size = _support_size(leaveout_left)
            ranked_candidates = sorted(
                leaveout_bidi.items(),
                key=lambda item: (-item[1], item[0]),
            )
            teacher_candidate_tokens = [token for token, _ in ranked_candidates]
            teacher_candidate_counts = [count for _, count in ranked_candidates]
            teacher_candidate_total = sum(teacher_candidate_counts)
            teacher_candidate_frequencies = (
                [count / teacher_candidate_total for count in teacher_candidate_counts]
                if teacher_candidate_total > 0
                else []
            )
            candidate_set_leq_4 = bidi_inclusive_support_size <= 4
            candidate_set_leq_8 = bidi_inclusive_support_size <= 8
            future_uplift = int(bidi_leaveout_support_size == 1) - int(left_leaveout_support_size == 1)
            self_inclusion_uplift = int(bidi_inclusive_support_size == 1) - int(
                bidi_leaveout_support_size == 1
            )
            memory_trust = 1.0 / left_leaveout_support_size if left_leaveout_support_size > 0 else 0.0
            bridge_confidence = 1.0 / bidi_leaveout_support_size if bidi_leaveout_support_size > 0 else 0.0
            clean_bridge_score = (
                (2.0 * memory_trust * bridge_confidence) / (memory_trust + bridge_confidence)
                if memory_trust > 0.0 and bridge_confidence > 0.0
                else 0.0
            )
            required_radius_found: int | None = None
            for required_radius in required_radii:
                contexts_for_radius = bidi_contexts_by_radius[required_radius]
                required_index = center_index - required_radius
                if required_index < 0 or required_index >= len(contexts_for_radius):
                    continue
                required_context, required_center = contexts_for_radius[required_index]
                if required_center != center:
                    continue
                support = bidi_global_by_radius[required_radius][required_context]
                if _support_size(support) == 1:
                    required_radius_found = required_radius
                    break
            yield OraclePositionLabel(
                path=path_str,
                size=len(data),
                radius=radius,
                position=position,
                center=center,
                center_hex=f"{center:02x}",
                left_context_hex=left_context.hex(),
                right_context_hex=bidi_context[radius:].hex(),
                bidi_inclusive_support_size=bidi_inclusive_support_size,
                bidi_leaveout_support_size=bidi_leaveout_support_size,
                left_leaveout_support_size=left_leaveout_support_size,
                teacher_candidate_tokens=teacher_candidate_tokens,
                teacher_candidate_counts=teacher_candidate_counts,
                teacher_candidate_frequencies=teacher_candidate_frequencies,
                bidi_inclusive_candidate4=bidi_inclusive_support_size <= 4,
                bidi_leaveout_candidate4=0 < bidi_leaveout_support_size <= 4,
                left_leaveout_candidate4=0 < left_leaveout_support_size <= 4,
                candidate_set_leq_4=candidate_set_leq_4,
                candidate_set_leq_8=candidate_set_leq_8,
                required_radius=required_radius_found,
                future_uplift=future_uplift,
                self_inclusion_uplift=self_inclusion_uplift,
                clean_bridge_score=clean_bridge_score,
                memory_trust=memory_trust,
                bridge_confidence=bridge_confidence,
                self_inclusion_support_changed=(
                    bidi_inclusive_support_size != bidi_leaveout_support_size
                ),
                future_context_support_changed=(
                    bidi_leaveout_support_size != left_leaveout_support_size
                ),
            )


def analyze_oracle_attack(paths: Iterable[Path], radii: Iterable[int]) -> OracleAttackCorpusStats:
    selected_files = _iter_files(paths)
    radius_list = sorted({radius for radius in radii if radius > 0})
    total_bytes = 0

    bidi_global: dict[int, dict[bytes, Counter[int]]] = {
        radius: defaultdict(Counter) for radius in radius_list
    }
    left_global: dict[int, dict[bytes, Counter[int]]] = {
        radius: defaultdict(Counter) for radius in radius_list
    }
    per_file_bidi: dict[tuple[str, int], list[tuple[bytes, int]]] = {}
    per_file_left: dict[tuple[str, int], list[tuple[bytes, int]]] = {}
    per_file_bidi_local: dict[tuple[str, int], dict[bytes, Counter[int]]] = {}
    per_file_left_local: dict[tuple[str, int], dict[bytes, Counter[int]]] = {}
    file_bytes: dict[str, bytes] = {}

    for path in selected_files:
        data = path.read_bytes()
        file_bytes[str(path)] = data
        total_bytes += len(data)
        for radius in radius_list:
            bidi_contexts = _contexts_for_radius(data, radius)
            left_contexts = [
                (data[index - radius : index], data[index])
                for index in range(radius, len(data) - radius)
            ] if len(data) >= 2 * radius + 1 else []
            per_file_bidi[(str(path), radius)] = bidi_contexts
            per_file_left[(str(path), radius)] = left_contexts

            bidi_local: dict[bytes, Counter[int]] = defaultdict(Counter)
            for context, center in bidi_contexts:
                bidi_global[radius][context][center] += 1
                bidi_local[context][center] += 1
            per_file_bidi_local[(str(path), radius)] = bidi_local

            left_local: dict[bytes, Counter[int]] = defaultdict(Counter)
            for context, center in left_contexts:
                left_global[radius][context][center] += 1
                left_local[context][center] += 1
            per_file_left_local[(str(path), radius)] = left_local

    file_stats: list[OracleAttackFileStats] = []
    for path in selected_files:
        path_str = str(path)
        data = file_bytes[path_str]
        radius_rows: list[OracleAttackRadiusStats] = []
        for radius in radius_list:
            bidi_contexts = per_file_bidi[(path_str, radius)]
            left_contexts = per_file_left[(path_str, radius)]
            bidi_local = per_file_bidi_local[(path_str, radius)]
            left_local = per_file_left_local[(path_str, radius)]
            total_positions = len(bidi_contexts)

            bidi_inclusive_det = 0
            bidi_leaveout_det = 0
            bidi_inclusive_c4 = 0
            bidi_leaveout_c4 = 0
            left_leaveout_det = 0
            left_leaveout_c4 = 0
            leaveout_support_changed = 0

            for index, ((bidi_context, _), (left_context, _left_center)) in enumerate(zip(bidi_contexts, left_contexts)):
                global_bidi = bidi_global[radius][bidi_context]
                leaveout_bidi = _subtract_counter(global_bidi, bidi_local[bidi_context])
                global_bidi_support = _support_size(global_bidi)
                leaveout_bidi_support = _support_size(leaveout_bidi)
                if global_bidi_support == 1:
                    bidi_inclusive_det += 1
                if global_bidi_support <= 4:
                    bidi_inclusive_c4 += 1
                if leaveout_bidi_support == 1:
                    bidi_leaveout_det += 1
                if 0 < leaveout_bidi_support <= 4:
                    bidi_leaveout_c4 += 1
                if leaveout_bidi_support != global_bidi_support:
                    leaveout_support_changed += 1

                global_left = left_global[radius][left_context]
                leaveout_left = _subtract_counter(global_left, left_local[left_context])
                leaveout_left_support = _support_size(leaveout_left)
                if leaveout_left_support == 1:
                    left_leaveout_det += 1
                if 0 < leaveout_left_support <= 4:
                    left_leaveout_c4 += 1

            dictionary = build_unique_context_dictionary(data, radius, min_occurrences=2)
            removed_mask, _survivors, removed_count, used_dictionary = select_removals(
                data,
                dictionary,
                context_radius=radius,
            )
            serialized_rulebook = _serialize_dictionary(used_dictionary)
            rulebook_zlib_bytes = len(zlib.compress(serialized_rulebook, 9))
            mask_bytes = (len(removed_mask) + 7) // 8

            radius_rows.append(
                OracleAttackRadiusStats(
                    radius=radius,
                    positions=total_positions,
                    bidi_inclusive_deterministic_fraction=(
                        bidi_inclusive_det / total_positions if total_positions else 0.0
                    ),
                    bidi_leaveout_deterministic_fraction=(
                        bidi_leaveout_det / total_positions if total_positions else 0.0
                    ),
                    bidi_inclusive_candidate4_fraction=(
                        bidi_inclusive_c4 / total_positions if total_positions else 0.0
                    ),
                    bidi_leaveout_candidate4_fraction=(
                        bidi_leaveout_c4 / total_positions if total_positions else 0.0
                    ),
                    left_leaveout_deterministic_fraction=(
                        left_leaveout_det / total_positions if total_positions else 0.0
                    ),
                    left_leaveout_candidate4_fraction=(
                        left_leaveout_c4 / total_positions if total_positions else 0.0
                    ),
                    self_inclusion_deterministic_uplift=(
                        (bidi_inclusive_det - bidi_leaveout_det) / total_positions if total_positions else 0.0
                    ),
                    self_inclusion_candidate4_uplift=(
                        (bidi_inclusive_c4 - bidi_leaveout_c4) / total_positions if total_positions else 0.0
                    ),
                    future_context_deterministic_uplift=(
                        (bidi_leaveout_det - left_leaveout_det) / total_positions if total_positions else 0.0
                    ),
                    future_context_candidate4_uplift=(
                        (bidi_leaveout_c4 - left_leaveout_c4) / total_positions if total_positions else 0.0
                    ),
                    leaveout_support_changed_fraction=(
                        leaveout_support_changed / total_positions if total_positions else 0.0
                    ),
                    rulebook_key_count=len(used_dictionary),
                    rulebook_raw_bytes=len(serialized_rulebook),
                    rulebook_zlib_bytes=rulebook_zlib_bytes,
                    removed_count=removed_count,
                    removed_fraction=(removed_count / len(data) if data else 0.0),
                    mask_bytes=mask_bytes,
                    naive_net_removed_bytes=removed_count - rulebook_zlib_bytes - mask_bytes,
                )
            )

        file_stats.append(OracleAttackFileStats(path=path_str, size=len(data), radii=radius_rows))

    return OracleAttackCorpusStats(
        file_count=len(selected_files),
        total_bytes=total_bytes,
        radii=radius_list,
        files=file_stats,
    )
