from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from io import BytesIO
import math
import struct
from typing import Dict, Iterable, List, Tuple
import zlib

from conker.src.blinx4 import (
    DEFAULT_PHASES,
    Blinx4Phase as Blinx11Phase,
    _better_candidate,
    _phase_active,
    phase_names as _phase_names,
)


MAGIC = b"BL11"
VERSION = 1


@dataclass
class Blinx11Round:
    length_before: int
    context_radius: int
    removed_mask: bytes
    branch_codes: bytes
    branch_count: int
    dictionary: Dict[bytes, bytes]
    removed_count: int
    branch_limit: int
    phase_name: str = "unknown"
    dictionary_cap: int | None = None
    zlib_after_round: int | None = None
    zlib_gain: int | None = None
    schedule_score: float | None = None


@dataclass
class LaneCandidate:
    options: bytes
    total: int
    lane: str
    entropy: float
    dominant_share: float


LANE_PRIORITY = {
    "exact": 0,
    "tight": 1,
    "structured": 2,
    "broad": 3,
    "relaxed": 4,
}


@dataclass(frozen=True)
class LaneProfile:
    name: str
    max_support: int
    max_entropy: float
    min_dominant_share: float
    lane_bias: float


LANE_PROFILES: Tuple[LaneProfile, ...] = (
    LaneProfile(name="exact", max_support=1, max_entropy=0.0, min_dominant_share=1.0, lane_bias=2.6),
    LaneProfile(name="tight", max_support=2, max_entropy=0.95, min_dominant_share=0.80, lane_bias=1.8),
    LaneProfile(name="structured", max_support=4, max_entropy=1.60, min_dominant_share=0.65, lane_bias=1.1),
    LaneProfile(name="broad", max_support=8, max_entropy=2.20, min_dominant_share=0.50, lane_bias=0.5),
    LaneProfile(name="relaxed", max_support=16, max_entropy=2.85, min_dominant_share=0.35, lane_bias=0.2),
)


@dataclass
class CandidateStep:
    survivors: bytes
    round_state: Blinx11Round
    zlib_bytes: int
    gain: int


@dataclass
class Blinx11Compressed:
    original_length: int
    final_bytes: bytes
    rounds: List[Blinx11Round]

    def serialize(self) -> bytes:
        out = BytesIO()
        out.write(MAGIC)
        out.write(
            struct.pack(
                "<BIII",
                VERSION,
                self.original_length,
                len(self.rounds),
                len(self.final_bytes),
            )
        )
        for round_state in self.rounds:
            out.write(
                struct.pack(
                    "<IIIII",
                    round_state.length_before,
                    round_state.context_radius,
                    len(round_state.dictionary),
                    len(round_state.removed_mask),
                    len(round_state.branch_codes),
                )
            )
            key_len = round_state.context_radius * 2
            for key in sorted(round_state.dictionary):
                options = round_state.dictionary[key]
                if len(key) != key_len:
                    raise ValueError("BLX11 key length mismatch for round radius")
                out.write(struct.pack("<B", len(options)))
                out.write(key)
                out.write(options)
            out.write(round_state.removed_mask)
            out.write(round_state.branch_codes)
        out.write(self.final_bytes)
        return out.getvalue()

    def zlib_size(self, level: int = 9) -> int:
        return len(zlib.compress(self.serialize(), level))

    def stats(self) -> dict[str, object]:
        total_removed = sum(round_state.removed_count for round_state in self.rounds)
        total_branch = sum(round_state.branch_count for round_state in self.rounds)
        phase_counts: Dict[str, int] = {}
        for round_state in self.rounds:
            phase_counts[round_state.phase_name] = phase_counts.get(round_state.phase_name, 0) + 1
        return {
            "original_length": self.original_length,
            "final_length": len(self.final_bytes),
            "serialized_bytes": len(self.serialize()),
            "zlib_bytes": self.zlib_size(),
            "round_count": len(self.rounds),
            "total_removed": total_removed,
            "branch_count": total_branch,
            "removed_fraction": (
                float(total_removed) / float(self.original_length)
                if self.original_length
                else 0.0
            ),
            "dictionary_key_count": sum(len(round_state.dictionary) for round_state in self.rounds),
            "phase_counts": phase_counts,
            "rounds": [
                {
                    "phase_name": round_state.phase_name,
                    "length_before": round_state.length_before,
                    "context_radius": round_state.context_radius,
                    "dictionary_size": len(round_state.dictionary),
                    "dictionary_cap": round_state.dictionary_cap,
                    "removed_count": round_state.removed_count,
                    "branch_count": round_state.branch_count,
                    "branch_limit": round_state.branch_limit,
                    "zlib_after_round": round_state.zlib_after_round,
                    "zlib_gain": round_state.zlib_gain,
                    "schedule_score": round_state.schedule_score,
                }
                for round_state in self.rounds
            ],
        }


def _pack_removed_mask(mask: List[bool]) -> bytes:
    width = (len(mask) + 7) // 8
    payload = bytearray(width)
    for index, bit in enumerate(mask):
        if bit:
            payload[index // 8] |= 1 << (index % 8)
    return bytes(payload)


def _unpack_removed_mask(payload: bytes, length: int) -> List[bool]:
    mask = [False] * length
    for index in range(length):
        mask[index] = bool(payload[index // 8] & (1 << (index % 8)))
    return mask


def _lane_for_bucket(bucket: Dict[int, int]) -> tuple[str, float, float] | None:
    if not bucket:
        return None
    total = sum(bucket.values())
    support = len(bucket)
    counts = sorted(bucket.values(), reverse=True)
    dominant_share = counts[0] / float(total)
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = float(count) / float(total)
        entropy -= probability * math.log2(probability)
    for profile in LANE_PROFILES:
        if (
            support <= profile.max_support
            and entropy <= profile.max_entropy
            and dominant_share >= profile.min_dominant_share
        ):
            return profile.name, entropy, dominant_share
    return None


def _build_lane_context_candidates(
    data: bytes,
    *,
    context_radius: int,
    min_occurrences: int,
    branch_limit: int,
) -> Dict[bytes, LaneCandidate]:
    counts: Dict[bytes, Dict[int, int]] = {}
    if len(data) < 2 * context_radius + 1:
        return {}
    for index in range(context_radius, len(data) - context_radius):
        key = data[index - context_radius : index] + data[index + 1 : index + 1 + context_radius]
        bucket = counts.setdefault(key, {})
        bucket[data[index]] = bucket.get(data[index], 0) + 1
    out: Dict[bytes, LaneCandidate] = {}
    for key, bucket in counts.items():
        total = sum(bucket.values())
        if total < min_occurrences:
            continue
        lane_data = _lane_for_bucket(bucket)
        if lane_data is None:
            continue
        lane_name, entropy, dominant_share = lane_data
        if len(bucket) > branch_limit:
            continue
        ordered = [value for value, _count in sorted(bucket.items(), key=lambda item: (-item[1], item[0]))]
        out[key] = LaneCandidate(
            options=bytes(ordered),
            total=total,
            lane=lane_name,
            entropy=entropy,
            dominant_share=dominant_share,
        )
    return out


def _prune_branch_dictionary(
    candidates: Dict[bytes, LaneCandidate],
    *,
    dictionary_cap: int | None,
) -> Dict[bytes, LaneCandidate]:
    items = sorted(
        candidates.items(),
        key=lambda item: (
            LANE_PRIORITY.get(item[1].lane, 99),
            -item[1].total,
            len(item[1].options),
            item[1].entropy,
            len(item[0]),
            item[0],
        ),
    )
    if dictionary_cap is not None and dictionary_cap > 0:
        items = items[:dictionary_cap]
    return {key: value for key, value in items}


@dataclass
class ScheduledRemoval:
    index: int
    start: int
    end: int
    key: bytes
    options: bytes
    code: int
    weight: float
    lane: str
    entropy: float
    dominant_share: float


def _branch_position_score(*, total: int, option_count: int, code: int, lane: str, entropy: float, dominant_share: float) -> float:
    # Favor high-confidence lanes early. The goal is to admit more candidates,
    # not to spend the round budget on noisy ties.
    lane_bonus = {name: profile.lane_bias for name, profile in zip(
        [profile.name for profile in LANE_PROFILES], LANE_PROFILES
    )}.get(lane, 0.0)
    support_bonus = min(3.5, math.log2(float(total) + 1.0))
    option_bonus = 1.0 / float(option_count)
    entropy_penalty = 0.35 * entropy
    dominance_bonus = 1.8 * dominant_share
    mode_bonus = 0.1 if code == 0 else 0.0
    return 3.0 + lane_bonus + support_bonus + option_bonus + dominance_bonus + mode_bonus - entropy_penalty


def _build_scheduled_removals(
    data: bytes,
    dictionary: Dict[bytes, LaneCandidate],
    context_radius: int,
) -> List[ScheduledRemoval]:
    out: List[ScheduledRemoval] = []
    for index in range(context_radius, len(data) - context_radius):
        key = data[index - context_radius : index] + data[index + 1 : index + 1 + context_radius]
        info = dictionary.get(key)
        if info is None:
            continue
        options = info.options
        try:
            code = options.index(data[index])
        except ValueError:
            continue
        out.append(
            ScheduledRemoval(
                index=index,
                start=index,
                end=index + context_radius,
                key=key,
                options=options,
                code=code,
                weight=_branch_position_score(
                    total=info.total,
                    option_count=len(options),
                    code=code,
                    lane=info.lane,
                    entropy=info.entropy,
                    dominant_share=info.dominant_share,
                ),
                lane=info.lane,
                entropy=info.entropy,
                dominant_share=info.dominant_share,
            )
        )
    return out


def _select_branch_removals(
    data: bytes,
    dictionary: Dict[bytes, LaneCandidate],
    context_radius: int,
) -> Tuple[List[bool], bytes, bytes, int, Dict[bytes, bytes], float]:
    candidates = _build_scheduled_removals(
        data,
        dictionary,
        context_radius=context_radius,
    )
    if not candidates:
        return [False] * len(data), data, b"", 0, {}, 0.0
    candidates.sort(key=lambda item: (item.end, item.index))
    ends = [candidate.end for candidate in candidates]
    prev: List[int] = []
    for candidate in candidates:
        prev.append(bisect_right(ends, candidate.start - 1) - 1)
    dp = [0.0] * (len(candidates) + 1)
    take = [False] * len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        include = candidate.weight + dp[prev[idx - 1] + 1]
        exclude = dp[idx - 1]
        if include > exclude:
            dp[idx] = include
            take[idx - 1] = True
        else:
            dp[idx] = exclude
    chosen: List[ScheduledRemoval] = []
    idx = len(candidates) - 1
    while idx >= 0:
        include = candidates[idx].weight + dp[prev[idx] + 1]
        exclude = dp[idx]
        if take[idx] and include > exclude:
            chosen.append(candidates[idx])
            idx = prev[idx]
        else:
            idx -= 1
    chosen.reverse()
    removed = [False] * len(data)
    branch_codes: List[int] = []
    used_keys: Dict[bytes, bytes] = {}
    chosen_indices = {candidate.index: candidate for candidate in chosen}
    survivors = bytearray()
    for index, value in enumerate(data):
        candidate = chosen_indices.get(index)
        if candidate is None:
            survivors.append(value)
            continue
        removed[index] = True
        branch_codes.append(candidate.code)
        used_keys[candidate.key] = candidate.options
    return (
        removed,
        bytes(survivors),
        bytes(branch_codes),
        len(chosen),
        used_keys,
        float(sum(candidate.weight for candidate in chosen)),
    )


def _evaluate_phase(
    data: bytes,
    current: bytes,
    rounds: List[Blinx11Round],
    *,
    phase: Blinx11Phase,
    current_zlib: int,
    branch_limit: int,
) -> CandidateStep | None:
    best: CandidateStep | None = None
    for context_radius in phase.candidate_radii:
        candidates = _build_lane_context_candidates(
            current,
            context_radius=context_radius,
            min_occurrences=phase.min_occurrences,
            branch_limit=branch_limit,
        )
        if not candidates:
            continue
        for dictionary_cap in phase.dictionary_caps:
            dictionary = _prune_branch_dictionary(candidates, dictionary_cap=dictionary_cap)
            if not dictionary:
                continue
            removed_mask, survivors, branch_codes, removed_count, used_dictionary, schedule_score = _select_branch_removals(
                current,
                dictionary,
                context_radius=context_radius,
            )
            if removed_count < phase.min_removed or not used_dictionary:
                continue
            round_state = Blinx11Round(
                length_before=len(current),
                context_radius=context_radius,
                removed_mask=_pack_removed_mask(removed_mask),
                branch_codes=branch_codes,
                branch_count=len(branch_codes),
                dictionary=used_dictionary,
                removed_count=removed_count,
                branch_limit=branch_limit,
                phase_name=phase.name,
                dictionary_cap=dictionary_cap,
                schedule_score=schedule_score,
            )
            candidate_zlib = Blinx11Compressed(
                original_length=len(data),
                final_bytes=survivors,
                rounds=rounds + [round_state],
            ).zlib_size()
            gain = current_zlib - candidate_zlib
            if gain < phase.min_gain:
                continue
            round_state.zlib_after_round = candidate_zlib
            round_state.zlib_gain = gain
            candidate = CandidateStep(
                survivors=survivors,
                round_state=round_state,
                zlib_bytes=candidate_zlib,
                gain=gain,
            )
            if _better_candidate(best, candidate, selection_mode=phase.selection_mode):
                best = candidate
    return best


def compress(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx11Phase, ...] = DEFAULT_PHASES,
    branch_limit: int = 8,
) -> Blinx11Compressed:
    rounds: List[Blinx11Round] = []
    current = data
    current_zlib = Blinx11Compressed(original_length=len(data), final_bytes=current, rounds=rounds).zlib_size()
    for _ in range(max_rounds):
        survivor_ratio = float(len(current)) / float(len(data)) if data else 0.0
        best: CandidateStep | None = None
        for phase in phases:
            if not _phase_active(phase, survivor_ratio=survivor_ratio):
                continue
            candidate = _evaluate_phase(
                data,
                current,
                rounds,
                phase=phase,
                current_zlib=current_zlib,
                branch_limit=branch_limit,
            )
            if candidate is None:
                continue
            if best is None or (
                candidate.gain,
                candidate.round_state.removed_count,
                -candidate.round_state.context_radius,
            ) > (
                best.gain,
                best.round_state.removed_count,
                -best.round_state.context_radius,
            ):
                best = candidate
        if best is None:
            break
        current = best.survivors
        rounds.append(best.round_state)
        current_zlib = best.zlib_bytes
    return Blinx11Compressed(
        original_length=len(data),
        final_bytes=current,
        rounds=rounds,
    )


def deserialize(payload: bytes) -> Blinx11Compressed:
    stream = BytesIO(payload)
    if stream.read(4) != MAGIC:
        raise ValueError("Not a BLX11 payload")
    version, original_length, round_count, final_length = struct.unpack("<BIII", stream.read(13))
    if version != VERSION:
        raise ValueError(f"Unsupported BLX11 version: {version}")
    rounds: List[Blinx11Round] = []
    for _ in range(round_count):
        length_before, context_radius, entry_count, mask_size, branch_size = struct.unpack("<IIIII", stream.read(20))
        dictionary: Dict[bytes, bytes] = {}
        key_len = context_radius * 2
        for _ in range(entry_count):
            option_count = struct.unpack("<B", stream.read(1))[0]
            key = stream.read(key_len)
            options = stream.read(option_count)
            dictionary[key] = options
        removed_mask = stream.read(mask_size)
        removed_count = sum(bool(removed_mask[index // 8] & (1 << (index % 8))) for index in range(length_before))
        branch_codes = stream.read(branch_size)
        rounds.append(
            Blinx11Round(
                length_before=length_before,
                context_radius=context_radius,
                removed_mask=removed_mask,
                branch_codes=branch_codes,
                branch_count=removed_count,
                dictionary=dictionary,
                removed_count=removed_count,
                branch_limit=16,
                phase_name="replay",
            )
        )
    final_bytes = stream.read(final_length)
    return Blinx11Compressed(original_length=original_length, final_bytes=final_bytes, rounds=rounds)


def decompress(compressed: Blinx11Compressed) -> bytes:
    current = compressed.final_bytes
    for round_state in reversed(compressed.rounds):
        removed_mask = _unpack_removed_mask(round_state.removed_mask, round_state.length_before)
        branch_codes = round_state.branch_codes
        branch_index = 0
        rebuilt: List[int | None] = [None] * round_state.length_before
        survivor_index = 0
        for index, removed in enumerate(removed_mask):
            if not removed:
                rebuilt[index] = current[survivor_index]
                survivor_index += 1
        if survivor_index != len(current):
            raise ValueError("BLX11 survivor length mismatch during rebuild")
        for index, removed in enumerate(removed_mask):
            if not removed:
                continue
            radius = round_state.context_radius
            if index < radius or index >= len(rebuilt) - radius:
                raise ValueError("BLX11 attempted to remove boundary byte")
            left = rebuilt[index - radius : index]
            right = rebuilt[index + 1 : index + 1 + radius]
            if any(value is None for value in left) or any(value is None for value in right):
                raise ValueError("BLX11 encountered overlapping removals within a round")
            key = bytes(left + right)  # type: ignore[arg-type]
            options = round_state.dictionary.get(key)
            if options is None:
                raise ValueError(f"BLX11 missing dictionary entry for context {key}")
            code = branch_codes[branch_index]
            branch_index += 1
            if code >= len(options):
                raise ValueError("BLX11 branch code outside option set")
            rebuilt[index] = options[code]
        current = bytes(rebuilt)  # type: ignore[arg-type]
    if len(current) != compressed.original_length:
        raise ValueError("BLX11 output length mismatch")
    return current


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx11Phase, ...] = DEFAULT_PHASES,
    branch_limit: int = 16,
) -> Tuple[Blinx11Compressed, bool]:
    compressed = compress(
        data,
        max_rounds=max_rounds,
        phases=phases,
        branch_limit=branch_limit,
    )
    payload = compressed.serialize()
    rebuilt = decompress(deserialize(payload))
    return compressed, rebuilt == data


def phase_names(phases: Iterable[Blinx11Phase]) -> List[str]:
    return _phase_names(phases)
