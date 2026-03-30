from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from conker.src import blinx6 as _blinx6
from conker.src.blinx4 import (
    _better_candidate,
    _build_unique_context_candidates,
    _phase_active,
    _prune_dictionary,
)


DEFAULT_PHASES = _blinx6.DEFAULT_PHASES
Blinx7Phase = _blinx6.Blinx6Phase

DICT_POLICY_GLOBAL = "global"
DICT_POLICY_TYPED = "typed"
DICT_POLICY_ADAPTIVE = "adaptive"
DEFAULT_DICT_POLICY = DICT_POLICY_ADAPTIVE


@dataclass
class Blinx7Round(_blinx6.Blinx6Round):
    dictionary_policy: str = DICT_POLICY_GLOBAL


@dataclass
class CandidateStep:
    survivors: bytes
    round_state: Blinx7Round
    zlib_bytes: int
    gain: int


@dataclass
class Blinx7Compressed(_blinx6.Blinx6Compressed):
    rounds: List[Blinx7Round]

    def stats(self) -> dict[str, object]:
        payload = super().stats()
        payload["wire_format"] = "blx6-compatible"
        policy_counts: Dict[str, int] = {}
        for index, round_state in enumerate(self.rounds):
            policy = getattr(round_state, "dictionary_policy", DICT_POLICY_GLOBAL)
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
            payload["rounds"][index]["dictionary_policy"] = policy  # type: ignore[index]
        payload["dictionary_policy_counts"] = policy_counts
        return payload


def _byte_class(value: int) -> int:
    if value in (9, 10, 13, 32):
        return 0
    if 48 <= value <= 57:
        return 1
    if 65 <= value <= 90:
        return 2
    if 97 <= value <= 122:
        return 3
    if value in b"()[]{}<>":
        return 4
    if value in b"'\"`":
        return 5
    if value in b"._-/:\\,;|":
        return 6
    if 33 <= value <= 126:
        return 7
    return 8


def _bucket_id(key: bytes, value: int) -> Tuple[int, int, int, int]:
    radius = len(key) // 2
    left = key[:radius]
    right = key[radius:]
    left_edge = _byte_class(left[0]) if left else 9
    right_edge = _byte_class(right[-1]) if right else 9
    return (radius, left_edge, _byte_class(value), right_edge)


def _prune_dictionary_typed(
    candidates: Dict[bytes, Tuple[int, int]],
    *,
    dictionary_cap: int | None,
) -> Dict[bytes, int]:
    buckets: Dict[Tuple[int, int, int, int], List[Tuple[bytes, int, int]]] = {}
    for key, (value, count) in candidates.items():
        bucket = _bucket_id(key, value)
        buckets.setdefault(bucket, []).append((key, value, count))

    for items in buckets.values():
        items.sort(key=lambda item: (-item[2], len(item[0]), item[0], item[1]))

    if dictionary_cap is None or dictionary_cap <= 0:
        selected = [item for bucket in buckets.values() for item in bucket]
        selected.sort(key=lambda item: (-item[2], len(item[0]), item[0], item[1]))
        return {key: value for key, value, _ in selected}

    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (-item[1][0][2], len(item[1]), item[0]),
    )
    selected: List[Tuple[bytes, int, int]] = []
    consumed: Dict[Tuple[int, int, int, int], int] = {}

    # First pass: keep one high-count entry from as many type buckets as the cap allows.
    for bucket, items in ordered_buckets:
        if len(selected) >= dictionary_cap:
            break
        selected.append(items[0])
        consumed[bucket] = 1

    extras: List[Tuple[bytes, int, int]] = []
    for bucket, items in buckets.items():
        extras.extend(items[consumed.get(bucket, 0) :])
    extras.sort(key=lambda item: (-item[2], len(item[0]), item[0], item[1]))

    for item in extras:
        if len(selected) >= dictionary_cap:
            break
        selected.append(item)
    return {key: value for key, value, _ in selected}


def _dictionary_variants(
    candidates: Dict[bytes, Tuple[int, int]],
    *,
    dictionary_cap: int | None,
    dictionary_policy: str,
) -> List[Tuple[str, Dict[bytes, int]]]:
    if dictionary_policy == DICT_POLICY_GLOBAL:
        return [(DICT_POLICY_GLOBAL, _prune_dictionary(candidates, dictionary_cap=dictionary_cap))]
    if dictionary_policy == DICT_POLICY_TYPED:
        return [(DICT_POLICY_TYPED, _prune_dictionary_typed(candidates, dictionary_cap=dictionary_cap))]
    if dictionary_policy != DICT_POLICY_ADAPTIVE:
        raise ValueError(f"Unknown BLINX-7 dictionary policy: {dictionary_policy}")

    variants = [
        (DICT_POLICY_GLOBAL, _prune_dictionary(candidates, dictionary_cap=dictionary_cap)),
        (DICT_POLICY_TYPED, _prune_dictionary_typed(candidates, dictionary_cap=dictionary_cap)),
    ]
    deduped: List[Tuple[str, Dict[bytes, int]]] = []
    seen: set[Tuple[Tuple[bytes, int], ...]] = set()
    for policy_name, dictionary in variants:
        fingerprint = tuple(sorted(dictionary.items()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append((policy_name, dictionary))
    return deduped


def _evaluate_phase(
    data: bytes,
    current: bytes,
    rounds: List[Blinx7Round],
    *,
    phase: Blinx7Phase,
    current_zlib: int,
    mask_policy: str,
    dictionary_policy: str,
) -> CandidateStep | None:
    best: CandidateStep | None = None
    for context_radius in phase.candidate_radii:
        candidates = _build_unique_context_candidates(
            current,
            context_radius=context_radius,
            min_occurrences=phase.min_occurrences,
        )
        if not candidates:
            continue
        for dictionary_cap in phase.dictionary_caps:
            for policy_name, dictionary in _dictionary_variants(
                candidates,
                dictionary_cap=dictionary_cap,
                dictionary_policy=dictionary_policy,
            ):
                if not dictionary:
                    continue
                removed_mask, removed_positions, survivors, removed_count, used_dictionary = _blinx6._select_removals(
                    current,
                    dictionary,
                    context_radius=context_radius,
                )
                if removed_count < phase.min_removed or not used_dictionary:
                    continue
                for base_round in _blinx6._round_variants(
                    length_before=len(current),
                    context_radius=context_radius,
                    removed_mask=removed_mask,
                    removed_positions=removed_positions,
                    dictionary=used_dictionary,
                    removed_count=removed_count,
                    phase_name=phase.name,
                    dictionary_cap=dictionary_cap,
                    pair_rule_budget=0,
                    mask_policy=mask_policy,
                ):
                    for pair_rule_budget in phase.pair_rule_budgets:
                        round_state = Blinx7Round(
                            length_before=base_round.length_before,
                            context_radius=base_round.context_radius,
                            removed_payload=base_round.removed_payload,
                            removed_positions=base_round.removed_positions,
                            removed_count=base_round.removed_count,
                            phase_name=base_round.phase_name,
                            dictionary=base_round.dictionary,
                            dictionary_cap=base_round.dictionary_cap,
                            pair_rule_budget=pair_rule_budget,
                            mask_format=base_round.mask_format,
                            dictionary_policy=policy_name,
                        )
                        candidate_zlib = Blinx7Compressed(
                            original_length=len(data),
                            final_bytes=survivors,
                            rounds=rounds + [round_state],
                            pair_rule_budget=pair_rule_budget,
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
    phases: Tuple[Blinx7Phase, ...] = DEFAULT_PHASES,
    mask_policy: str = _blinx6.DEFAULT_MASK_POLICY,
    dictionary_policy: str = DEFAULT_DICT_POLICY,
) -> Blinx7Compressed:
    rounds: List[Blinx7Round] = []
    current = data
    current_pair_rule_budget = 0
    current_zlib = Blinx7Compressed(
        original_length=len(data),
        final_bytes=current,
        rounds=rounds,
        pair_rule_budget=current_pair_rule_budget,
    ).zlib_size()
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
                mask_policy=mask_policy,
                dictionary_policy=dictionary_policy,
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
        if best.round_state.pair_rule_budget is not None:
            current_pair_rule_budget = best.round_state.pair_rule_budget
    return Blinx7Compressed(
        original_length=len(data),
        final_bytes=current,
        rounds=rounds,
        pair_rule_budget=current_pair_rule_budget,
    )


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx7Phase, ...] = DEFAULT_PHASES,
    mask_policy: str = _blinx6.DEFAULT_MASK_POLICY,
    dictionary_policy: str = DEFAULT_DICT_POLICY,
) -> Tuple[Blinx7Compressed, bool]:
    compressed = compress(
        data,
        max_rounds=max_rounds,
        phases=phases,
        mask_policy=mask_policy,
        dictionary_policy=dictionary_policy,
    )
    payload = compressed.serialize()
    rebuilt = _blinx6.decompress(_blinx6.deserialize(payload))
    return compressed, rebuilt == data


def phase_names(phases: Iterable[Blinx7Phase]) -> List[str]:
    return _blinx6.phase_names(phases)
