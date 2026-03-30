from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import struct
from typing import Dict, Iterable, List, Tuple
import zlib


MAGIC = b"BLX4"
VERSION = 1
GRAMMAR_BASE_SYMBOL = 256


@dataclass(frozen=True)
class Blinx4Phase:
    name: str
    selection_mode: str
    candidate_radii: Tuple[int, ...]
    dictionary_caps: Tuple[int | None, ...]
    pair_rule_budgets: Tuple[int, ...]
    min_occurrences: int
    min_removed: int
    min_gain: int = 0
    min_survivor_ratio: float = 0.0
    max_survivor_ratio: float = 1.0


DEFAULT_PHASES: Tuple[Blinx4Phase, ...] = (
    Blinx4Phase(
        name="probe",
        selection_mode="discovery",
        candidate_radii=(1, 2, 3, 4),
        dictionary_caps=(128, 64, None),
        pair_rule_budgets=(32, 64, 128),
        min_occurrences=2,
        min_removed=8,
        min_gain=0,
        min_survivor_ratio=0.78,
        max_survivor_ratio=1.0,
    ),
    Blinx4Phase(
        name="harvest",
        selection_mode="profit",
        candidate_radii=(2, 3, 4),
        dictionary_caps=(32, 64, 128),
        pair_rule_budgets=(32, 64, 128, 192),
        min_occurrences=2,
        min_removed=8,
        min_gain=1,
        min_survivor_ratio=0.55,
        max_survivor_ratio=1.0,
    ),
    Blinx4Phase(
        name="consolidate",
        selection_mode="profit",
        candidate_radii=(1, 2, 3),
        dictionary_caps=(16, 32, 64),
        pair_rule_budgets=(16, 32, 64, 128),
        min_occurrences=3,
        min_removed=6,
        min_gain=1,
        min_survivor_ratio=0.30,
        max_survivor_ratio=0.82,
    ),
    Blinx4Phase(
        name="refine",
        selection_mode="profit",
        candidate_radii=(1, 2),
        dictionary_caps=(8, 16, 32),
        pair_rule_budgets=(16, 32, 64),
        min_occurrences=4,
        min_removed=4,
        min_gain=1,
        min_survivor_ratio=0.0,
        max_survivor_ratio=0.58,
    ),
)


@dataclass
class Blinx4Round:
    length_before: int
    context_radius: int
    removed_mask: bytes
    dictionary: Dict[bytes, int]
    removed_count: int
    phase_name: str = "unknown"
    dictionary_cap: int | None = None
    pair_rule_budget: int | None = None
    zlib_after_round: int | None = None
    zlib_gain: int | None = None


@dataclass
class PairGrammar:
    rules: List[Tuple[int, int]]
    encoded_sequences: List[List[int]]

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    @property
    def symbol_count(self) -> int:
        return sum(len(sequence) for sequence in self.encoded_sequences)


@dataclass
class Blinx4Compressed:
    original_length: int
    final_bytes: bytes
    rounds: List[Blinx4Round]
    pair_rule_budget: int = 128

    def _flatten_keys(self) -> List[bytes]:
        keys: List[bytes] = []
        for round_state in self.rounds:
            for key in sorted(round_state.dictionary):
                keys.append(key)
        return keys

    def _build_pair_grammar(self, keys: List[bytes]) -> PairGrammar:
        sequences: List[List[int]] = [list(key) for key in keys]
        rules: List[Tuple[int, int]] = []
        next_symbol = GRAMMAR_BASE_SYMBOL
        while len(rules) < self.pair_rule_budget:
            pair_counts: Dict[Tuple[int, int], int] = {}
            for sequence in sequences:
                for index in range(len(sequence) - 1):
                    pair = (sequence[index], sequence[index + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            pair, count = max(pair_counts.items(), key=lambda item: item[1])
            if count < 2:
                break
            new_symbol = next_symbol
            next_symbol += 1
            replaced_sequences: List[List[int]] = []
            for sequence in sequences:
                rebuilt: List[int] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and sequence[index] == pair[0]
                        and sequence[index + 1] == pair[1]
                    ):
                        rebuilt.append(new_symbol)
                        index += 2
                    else:
                        rebuilt.append(sequence[index])
                        index += 1
                replaced_sequences.append(rebuilt)
            rules.append(pair)
            sequences = replaced_sequences
        return PairGrammar(rules=rules, encoded_sequences=sequences)

    def serialize(self) -> bytes:
        out = BytesIO()
        out.write(MAGIC)
        out.write(
            struct.pack(
                "<BIIII",
                VERSION,
                self.original_length,
                len(self.rounds),
                self.pair_rule_budget,
                len(self.final_bytes),
            )
        )

        keys = self._flatten_keys()
        grammar = self._build_pair_grammar(keys)
        out.write(struct.pack("<I", grammar.rule_count))
        for left, right in grammar.rules:
            out.write(struct.pack("<HH", left, right))

        sequence_iter = iter(grammar.encoded_sequences)
        out.write(struct.pack("<I", len(keys)))
        for round_state in self.rounds:
            round_keys = sorted(round_state.dictionary)
            out.write(
                struct.pack(
                    "<IIII",
                    round_state.length_before,
                    round_state.context_radius,
                    len(round_keys),
                    len(round_state.removed_mask),
                )
            )
            for key in round_keys:
                encoded = next(sequence_iter)
                out.write(struct.pack("<BH", round_state.dictionary[key], len(encoded)))
                for symbol in encoded:
                    out.write(struct.pack("<H", symbol))
            out.write(round_state.removed_mask)
        out.write(self.final_bytes)
        return out.getvalue()

    def zlib_size(self, level: int = 9) -> int:
        return len(zlib.compress(self.serialize(), level))

    def stats(self) -> dict[str, object]:
        total_removed = sum(round_state.removed_count for round_state in self.rounds)
        keys = self._flatten_keys()
        grammar = self._build_pair_grammar(keys)
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
            "removed_fraction": (
                float(total_removed) / float(self.original_length)
                if self.original_length
                else 0.0
            ),
            "pair_rule_budget": self.pair_rule_budget,
            "pair_rule_count": grammar.rule_count,
            "pair_symbol_count": grammar.symbol_count,
            "dictionary_key_count": len(keys),
            "phase_counts": phase_counts,
            "rounds": [
                {
                    "phase_name": round_state.phase_name,
                    "length_before": round_state.length_before,
                    "context_radius": round_state.context_radius,
                    "dictionary_size": len(round_state.dictionary),
                    "dictionary_cap": round_state.dictionary_cap,
                    "pair_rule_budget": round_state.pair_rule_budget,
                    "removed_count": round_state.removed_count,
                    "zlib_after_round": round_state.zlib_after_round,
                    "zlib_gain": round_state.zlib_gain,
                    "removed_fraction": (
                        float(round_state.removed_count) / float(round_state.length_before)
                        if round_state.length_before
                        else 0.0
                    ),
                }
                for round_state in self.rounds
            ],
        }


@dataclass
class CandidateStep:
    survivors: bytes
    round_state: Blinx4Round
    zlib_bytes: int
    gain: int


def _pack_removed_mask(mask: List[bool]) -> bytes:
    width = (len(mask) + 7) // 8
    payload = bytearray(width)
    for index, bit in enumerate(mask):
        if bit:
            payload[index // 8] |= 1 << (index % 8)
    return bytes(payload)


def _build_unique_context_candidates(
    data: bytes,
    *,
    context_radius: int,
    min_occurrences: int,
) -> Dict[bytes, Tuple[int, int]]:
    counts: Dict[bytes, Dict[int, int]] = {}
    if len(data) < 2 * context_radius + 1:
        return {}
    for index in range(context_radius, len(data) - context_radius):
        key = data[index - context_radius : index] + data[index + 1 : index + 1 + context_radius]
        bucket = counts.setdefault(key, {})
        bucket[data[index]] = bucket.get(data[index], 0) + 1
    unique: Dict[bytes, Tuple[int, int]] = {}
    for key, bucket in counts.items():
        if len(bucket) != 1:
            continue
        value, count = next(iter(bucket.items()))
        if count >= min_occurrences:
            unique[key] = (value, count)
    return unique


def _prune_dictionary(
    candidates: Dict[bytes, Tuple[int, int]],
    *,
    dictionary_cap: int | None,
) -> Dict[bytes, int]:
    items = sorted(candidates.items(), key=lambda item: (-item[1][1], len(item[0]), item[0]))
    if dictionary_cap is not None and dictionary_cap > 0:
        items = items[:dictionary_cap]
    return {key: value for key, (value, _) in items}


def _select_removals(
    data: bytes,
    dictionary: Dict[bytes, int],
    context_radius: int,
) -> Tuple[List[bool], bytes, int, Dict[bytes, int]]:
    removed = [False] * len(data)
    survivors = bytearray()
    removed_count = 0
    last_removed = -10**9
    used_keys: Dict[bytes, int] = {}
    index = 0
    while index < len(data):
        key = (
            data[index - context_radius : index] + data[index + 1 : index + 1 + context_radius]
            if context_radius <= index < len(data) - context_radius
            else None
        )
        can_remove = (
            key is not None
            and index - last_removed > context_radius
            and key in dictionary
            and dictionary[key] == data[index]
        )
        if can_remove:
            removed[index] = True
            removed_count += 1
            last_removed = index
            used_keys[key] = dictionary[key]
        else:
            survivors.append(data[index])
        index += 1
    return removed, bytes(survivors), removed_count, used_keys


def _phase_active(
    phase: Blinx4Phase,
    *,
    survivor_ratio: float,
) -> bool:
    return phase.min_survivor_ratio <= survivor_ratio <= phase.max_survivor_ratio


def _better_candidate(
    current_best: CandidateStep | None,
    candidate: CandidateStep,
    *,
    selection_mode: str,
) -> bool:
    if current_best is None:
        return True
    if selection_mode == "profit":
        return (
            candidate.gain,
            candidate.round_state.removed_count,
            -candidate.round_state.context_radius,
        ) > (
            current_best.gain,
            current_best.round_state.removed_count,
            -current_best.round_state.context_radius,
        )
    if selection_mode == "discovery":
        return (
            candidate.round_state.removed_count,
            candidate.gain,
            -candidate.round_state.context_radius,
        ) > (
            current_best.round_state.removed_count,
            current_best.gain,
            -current_best.round_state.context_radius,
        )
    raise ValueError(f"Unknown BLX4 selection mode: {selection_mode}")


def _evaluate_phase(
    data: bytes,
    current: bytes,
    rounds: List[Blinx4Round],
    *,
    phase: Blinx4Phase,
    current_zlib: int,
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
            dictionary = _prune_dictionary(candidates, dictionary_cap=dictionary_cap)
            if not dictionary:
                continue
            removed_mask, survivors, removed_count, used_dictionary = _select_removals(
                current,
                dictionary,
                context_radius=context_radius,
            )
            if removed_count < phase.min_removed or not used_dictionary:
                continue
            for pair_rule_budget in phase.pair_rule_budgets:
                round_state = Blinx4Round(
                    length_before=len(current),
                    context_radius=context_radius,
                    removed_mask=_pack_removed_mask(removed_mask),
                    dictionary=used_dictionary,
                    removed_count=removed_count,
                    phase_name=phase.name,
                    dictionary_cap=dictionary_cap,
                    pair_rule_budget=pair_rule_budget,
                )
                candidate_zlib = Blinx4Compressed(
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
    phases: Tuple[Blinx4Phase, ...] = DEFAULT_PHASES,
) -> Blinx4Compressed:
    rounds: List[Blinx4Round] = []
    current = data
    current_pair_rule_budget = 0
    current_zlib = Blinx4Compressed(
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
    return Blinx4Compressed(
        original_length=len(data),
        final_bytes=current,
        rounds=rounds,
        pair_rule_budget=current_pair_rule_budget,
    )


def _expand_symbol(symbol: int, rules: List[Tuple[int, int]], memo: Dict[int, bytes]) -> bytes:
    if symbol < GRAMMAR_BASE_SYMBOL:
        return bytes((symbol,))
    if symbol in memo:
        return memo[symbol]
    index = symbol - GRAMMAR_BASE_SYMBOL
    left, right = rules[index]
    expanded = _expand_symbol(left, rules, memo) + _expand_symbol(right, rules, memo)
    memo[symbol] = expanded
    return expanded


def _read_exact(stream: BytesIO, size: int) -> bytes:
    blob = stream.read(size)
    if len(blob) != size:
        raise ValueError("Unexpected end of BLX4 payload")
    return blob


def deserialize(payload: bytes) -> Blinx4Compressed:
    stream = BytesIO(payload)
    if _read_exact(stream, 4) != MAGIC:
        raise ValueError("Not a BLX4 payload")
    version, original_length, round_count, pair_rule_budget, final_length = struct.unpack(
        "<BIIII",
        _read_exact(stream, 17),
    )
    if version != VERSION:
        raise ValueError(f"Unsupported BLX4 version: {version}")
    grammar_count = struct.unpack("<I", _read_exact(stream, 4))[0]
    grammar_rules = [struct.unpack("<HH", _read_exact(stream, 4)) for _ in range(grammar_count)]
    total_key_count = struct.unpack("<I", _read_exact(stream, 4))[0]
    memo: Dict[int, bytes] = {}
    rounds: List[Blinx4Round] = []
    keys_read = 0
    for _ in range(round_count):
        length_before, context_radius, entry_count, mask_size = struct.unpack(
            "<IIII",
            _read_exact(stream, 16),
        )
        dictionary: Dict[bytes, int] = {}
        for _ in range(entry_count):
            value, encoded_len = struct.unpack("<BH", _read_exact(stream, 3))
            encoded = [struct.unpack("<H", _read_exact(stream, 2))[0] for _ in range(encoded_len)]
            key = b"".join(_expand_symbol(symbol, grammar_rules, memo) for symbol in encoded)
            dictionary[key] = value
            keys_read += 1
        removed_mask = _read_exact(stream, mask_size)
        removed_count = sum(
            bool(removed_mask[index // 8] & (1 << (index % 8)))
            for index in range(length_before)
        )
        rounds.append(
            Blinx4Round(
                length_before=length_before,
                context_radius=context_radius,
                removed_mask=removed_mask,
                dictionary=dictionary,
                removed_count=removed_count,
                phase_name="replay",
            )
        )
    if keys_read != total_key_count:
        raise ValueError("BLX4 key count mismatch")
    final_bytes = _read_exact(stream, final_length)
    return Blinx4Compressed(
        original_length=original_length,
        final_bytes=final_bytes,
        rounds=rounds,
        pair_rule_budget=pair_rule_budget,
    )


def _unpack_removed_mask(payload: bytes, length: int) -> List[bool]:
    mask = [False] * length
    for index in range(length):
        mask[index] = bool(payload[index // 8] & (1 << (index % 8)))
    return mask


def decompress(compressed: Blinx4Compressed) -> bytes:
    current = compressed.final_bytes
    for round_state in reversed(compressed.rounds):
        removed_mask = _unpack_removed_mask(round_state.removed_mask, round_state.length_before)
        rebuilt: List[int | None] = [None] * round_state.length_before
        survivor_index = 0
        for index, removed in enumerate(removed_mask):
            if not removed:
                rebuilt[index] = current[survivor_index]
                survivor_index += 1
        if survivor_index != len(current):
            raise ValueError("BLX4 survivor length mismatch during rebuild")
        for index, removed in enumerate(removed_mask):
            if not removed:
                continue
            radius = round_state.context_radius
            if index < radius or index >= len(rebuilt) - radius:
                raise ValueError("BLX4 attempted to remove boundary byte")
            left = rebuilt[index - radius : index]
            right = rebuilt[index + 1 : index + 1 + radius]
            if any(value is None for value in left) or any(value is None for value in right):
                raise ValueError("BLX4 encountered overlapping removals within a round")
            key = bytes(left + right)  # type: ignore[arg-type]
            if key not in round_state.dictionary:
                raise ValueError(f"BLX4 missing dictionary entry for context {key}")
            rebuilt[index] = round_state.dictionary[key]
        current = bytes(rebuilt)  # type: ignore[arg-type]
    if len(current) != compressed.original_length:
        raise ValueError("BLX4 output length mismatch")
    return current


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx4Phase, ...] = DEFAULT_PHASES,
) -> Tuple[Blinx4Compressed, bool]:
    compressed = compress(
        data,
        max_rounds=max_rounds,
        phases=phases,
    )
    payload = compressed.serialize()
    rebuilt = decompress(deserialize(payload))
    return compressed, rebuilt == data


def phase_names(phases: Iterable[Blinx4Phase]) -> List[str]:
    return [phase.name for phase in phases]
