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
from conker.src.blinx7 import (
    DEFAULT_PHASES,
    Blinx7Phase as Blinx8Phase,
    CandidateStep,
    DICT_POLICY_ADAPTIVE,
    DICT_POLICY_GLOBAL,
    DICT_POLICY_TYPED,
    Blinx7Compressed,
    Blinx7Round,
    _bucket_id,
    _dictionary_variants,
    _prune_dictionary_typed,
)


DICT_POLICY_ORTHO = "orthogonal"
DEFAULT_DICT_POLICY = DICT_POLICY_ADAPTIVE


@dataclass
class Blinx8Round(Blinx7Round):
    dictionary_policy: str = DICT_POLICY_GLOBAL


@dataclass
class Blinx8Compressed(Blinx7Compressed):
    rounds: List[Blinx8Round]


def _slot_for_bucket(bucket: Tuple[int, int, int, int]) -> int:
    _radius, left_edge, center, right_edge = bucket
    if center == 1:
        return 0  # numeric
    if center in (4, 5) or left_edge in (4, 5) or right_edge in (4, 5):
        return 1  # boundary / quote
    if center in (2, 3) and left_edge in (0, 2, 3, 6, 7) and right_edge in (0, 2, 3, 6, 7):
        return 2  # text-like
    return 3  # fallback


def _prune_dictionary_orthogonal(
    candidates: Dict[bytes, Tuple[int, int]],
    *,
    dictionary_cap: int | None,
) -> Dict[bytes, int]:
    slots: Dict[int, List[Tuple[bytes, int, int]]] = {}
    for key, (value, count) in candidates.items():
        slot = _slot_for_bucket(_bucket_id(key, value))
        slots.setdefault(slot, []).append((key, value, count))

    for items in slots.values():
        items.sort(key=lambda item: (-item[2], len(item[0]), item[0], item[1]))

    if dictionary_cap is None or dictionary_cap <= 0:
        selected = [item for items in slots.values() for item in items]
        selected.sort(key=lambda item: (-item[2], len(item[0]), item[0], item[1]))
        return {key: value for key, value, _ in selected}

    slot_order = sorted(
        slots,
        key=lambda slot: (-slots[slot][0][2], len(slots[slot]), slot),
    )
    selected: List[Tuple[bytes, int, int]] = []
    consumed: Dict[int, int] = {}

    # First pass: force representation from distinct slots.
    for slot in slot_order:
        if len(selected) >= dictionary_cap:
            break
        selected.append(slots[slot][0])
        consumed[slot] = 1

    # Second pass: fill remaining budget round-robin by slot, preserving slot diversity.
    cursor = 0
    while len(selected) < dictionary_cap:
        advanced = False
        for slot in slot_order:
            index = consumed.get(slot, 0)
            items = slots.get(slot, [])
            if index >= len(items):
                continue
            selected.append(items[index])
            consumed[slot] = index + 1
            advanced = True
            if len(selected) >= dictionary_cap:
                break
        if not advanced:
            break
        cursor += 1
        if cursor > dictionary_cap * 2:
            break

    return {key: value for key, value, _ in selected}


def _dictionary_variants_orthogonal(
    candidates: Dict[bytes, Tuple[int, int]],
    *,
    dictionary_cap: int | None,
    dictionary_policy: str,
) -> List[Tuple[str, Dict[bytes, int]]]:
    if dictionary_policy == DICT_POLICY_ORTHO:
        return [(DICT_POLICY_ORTHO, _prune_dictionary_orthogonal(candidates, dictionary_cap=dictionary_cap))]
    if dictionary_policy == DICT_POLICY_GLOBAL:
        return [(DICT_POLICY_GLOBAL, _prune_dictionary(candidates, dictionary_cap=dictionary_cap))]
    if dictionary_policy == DICT_POLICY_TYPED:
        return [(DICT_POLICY_TYPED, _prune_dictionary_typed(candidates, dictionary_cap=dictionary_cap))]
    if dictionary_policy != DICT_POLICY_ADAPTIVE:
        raise ValueError(f"Unknown BLINX-8 dictionary policy: {dictionary_policy}")

    variants = [
        (DICT_POLICY_GLOBAL, _prune_dictionary(candidates, dictionary_cap=dictionary_cap)),
        (DICT_POLICY_TYPED, _prune_dictionary_typed(candidates, dictionary_cap=dictionary_cap)),
        (DICT_POLICY_ORTHO, _prune_dictionary_orthogonal(candidates, dictionary_cap=dictionary_cap)),
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
    rounds: List[Blinx8Round],
    *,
    phase: Blinx8Phase,
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
            for policy_name, dictionary in _dictionary_variants_orthogonal(
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
                        round_state = Blinx8Round(
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
                        candidate_zlib = Blinx8Compressed(
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
    phases: Tuple[Blinx8Phase, ...] = DEFAULT_PHASES,
    mask_policy: str = _blinx6.DEFAULT_MASK_POLICY,
    dictionary_policy: str = DEFAULT_DICT_POLICY,
) -> Blinx8Compressed:
    rounds: List[Blinx8Round] = []
    current = data
    current_pair_rule_budget = 0
    current_zlib = Blinx8Compressed(
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
    return Blinx8Compressed(
        original_length=len(data),
        final_bytes=current,
        rounds=rounds,
        pair_rule_budget=current_pair_rule_budget,
    )


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx8Phase, ...] = DEFAULT_PHASES,
    mask_policy: str = _blinx6.DEFAULT_MASK_POLICY,
    dictionary_policy: str = DEFAULT_DICT_POLICY,
) -> Tuple[Blinx8Compressed, bool]:
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


def phase_names(phases: Iterable[Blinx8Phase]) -> List[str]:
    return _blinx6.phase_names(phases)
