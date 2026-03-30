from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import struct
from typing import Dict, Iterable, List, Tuple
import zlib

from conker.src.blinx4 import (
    DEFAULT_PHASES,
    Blinx4Phase as Blinx6Phase,
    _better_candidate,
    _build_unique_context_candidates,
    _phase_active,
    _prune_dictionary,
    phase_names as _phase_names,
)


MAGIC = b"BLX6"
VERSION = 1
GRAMMAR_BASE_SYMBOL = 256

MASK_FORMAT_BITSET = 0
MASK_FORMAT_POSDELTA = 1
MASK_POLICY_BITSET = "bitset"
MASK_POLICY_POSDELTA = "posdelta"
MASK_POLICY_ADAPTIVE = "adaptive"
DEFAULT_MASK_POLICY = MASK_POLICY_ADAPTIVE
MASK_FORMAT_NAMES = {
    MASK_FORMAT_BITSET: "bitset",
    MASK_FORMAT_POSDELTA: "posdelta",
}

DICT_MODE_DIRECT = 0
DICT_MODE_SHARED = 1
DICT_MODE_NAMES = {
    DICT_MODE_DIRECT: "direct",
    DICT_MODE_SHARED: "shared",
}


def _pack_version_mode(version: int, dictionary_mode: int) -> int:
    return version | (dictionary_mode << 7)


def _unpack_version_mode(version_mode: int) -> Tuple[int, int]:
    return version_mode & 0x7F, (version_mode >> 7) & 0x01


@dataclass
class Blinx6Round:
    length_before: int
    context_radius: int
    removed_payload: bytes
    removed_positions: Tuple[int, ...]
    removed_count: int
    phase_name: str = "unknown"
    dictionary: Dict[bytes, int] | None = None
    dictionary_cap: int | None = None
    pair_rule_budget: int | None = None
    mask_format: int = MASK_FORMAT_BITSET
    zlib_after_round: int | None = None
    zlib_gain: int | None = None

    def __post_init__(self) -> None:
        if self.dictionary is None:
            self.dictionary = {}


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
class CandidateStep:
    survivors: bytes
    round_state: Blinx6Round
    zlib_bytes: int
    gain: int


@dataclass
class Blinx6Compressed:
    original_length: int
    final_bytes: bytes
    rounds: List[Blinx6Round]
    pair_rule_budget: int = 0

    def _flatten_keys(self) -> List[bytes]:
        keys: List[bytes] = []
        for round_state in self.rounds:
            for key in sorted(round_state.dictionary or {}):
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

    def _global_entry_plan(self) -> Tuple[List[Tuple[bytes, int]], List[Tuple[int, ...]]]:
        entries: List[Tuple[bytes, int]] = []
        entry_to_id: Dict[Tuple[bytes, int], int] = {}
        round_entry_ids: List[Tuple[int, ...]] = []
        for round_state in self.rounds:
            ids: List[int] = []
            for key in sorted(round_state.dictionary or {}):
                value = (round_state.dictionary or {})[key]
                entry = (key, value)
                entry_id = entry_to_id.get(entry)
                if entry_id is None:
                    entry_id = len(entries)
                    entries.append(entry)
                    entry_to_id[entry] = entry_id
                ids.append(entry_id)
            ids.sort()
            round_entry_ids.append(tuple(ids))
        return entries, round_entry_ids

    def _render_direct_body(self) -> bytes:
        out = BytesIO()
        keys = self._flatten_keys()
        grammar = self._build_pair_grammar(keys)
        out.write(struct.pack("<I", grammar.rule_count))
        for left, right in grammar.rules:
            out.write(struct.pack("<HH", left, right))

        sequence_iter = iter(grammar.encoded_sequences)
        out.write(struct.pack("<I", len(keys)))
        for round_state in self.rounds:
            round_keys = sorted(round_state.dictionary or {})
            out.write(
                struct.pack(
                    "<IIIII",
                    round_state.length_before,
                    round_state.context_radius,
                    len(round_keys),
                    round_state.mask_format,
                    len(round_state.removed_payload),
                )
            )
            for key in round_keys:
                encoded = next(sequence_iter)
                out.write(struct.pack("<BH", (round_state.dictionary or {})[key], len(encoded)))
                for symbol in encoded:
                    out.write(struct.pack("<H", symbol))
            out.write(round_state.removed_payload)
        out.write(self.final_bytes)
        return out.getvalue()

    def _render_shared_body(self) -> bytes:
        out = BytesIO()
        entries, round_entry_ids = self._global_entry_plan()
        keys = [key for key, _ in entries]
        grammar = self._build_pair_grammar(keys)
        out.write(struct.pack("<I", grammar.rule_count))
        for left, right in grammar.rules:
            out.write(struct.pack("<HH", left, right))

        sequence_iter = iter(grammar.encoded_sequences)
        out.write(struct.pack("<I", len(entries)))
        for key, value in entries:
            encoded = next(sequence_iter)
            out.write(struct.pack("<BH", value, len(encoded)))
            for symbol in encoded:
                out.write(struct.pack("<H", symbol))

        for round_state, entry_ids in zip(self.rounds, round_entry_ids):
            entry_payload = _encode_index_list(entry_ids)
            out.write(
                struct.pack(
                    "<IIIIII",
                    round_state.length_before,
                    round_state.context_radius,
                    len(entry_ids),
                    round_state.mask_format,
                    len(round_state.removed_payload),
                    len(entry_payload),
                )
            )
            out.write(entry_payload)
            out.write(round_state.removed_payload)
        out.write(self.final_bytes)
        return out.getvalue()

    def _payload_candidates(self) -> Dict[int, bytes]:
        direct = BytesIO()
        direct.write(MAGIC)
        direct.write(
            struct.pack(
                "<BIIII",
                _pack_version_mode(VERSION, DICT_MODE_DIRECT),
                self.original_length,
                len(self.rounds),
                self.pair_rule_budget,
                len(self.final_bytes),
            )
        )
        direct.write(self._render_direct_body())

        shared = BytesIO()
        shared.write(MAGIC)
        shared.write(
            struct.pack(
                "<BIIII",
                _pack_version_mode(VERSION, DICT_MODE_SHARED),
                self.original_length,
                len(self.rounds),
                self.pair_rule_budget,
                len(self.final_bytes),
            )
        )
        shared.write(self._render_shared_body())
        return {
            DICT_MODE_DIRECT: direct.getvalue(),
            DICT_MODE_SHARED: shared.getvalue(),
        }

    def _mode_costs(self, level: int = 9) -> Dict[int, int]:
        payloads = self._payload_candidates()
        return {mode: len(zlib.compress(payload, level)) for mode, payload in payloads.items()}

    def _choose_mode(self, level: int = 9) -> Tuple[int, bytes, Dict[int, int]]:
        payloads = self._payload_candidates()
        costs = {mode: len(zlib.compress(payload, level)) for mode, payload in payloads.items()}
        chosen_mode = min(
            payloads,
            key=lambda mode: (costs[mode], len(payloads[mode]), mode),
        )
        return chosen_mode, payloads[chosen_mode], costs

    def serialize(self, level: int = 9) -> bytes:
        _, payload, _ = self._choose_mode(level=level)
        return payload

    def zlib_size(self, level: int = 9) -> int:
        _, _, costs = self._choose_mode(level=level)
        return min(costs.values())

    def stats(self) -> dict[str, object]:
        total_removed = sum(round_state.removed_count for round_state in self.rounds)
        keys = self._flatten_keys()
        direct_grammar = self._build_pair_grammar(keys)
        entries, _ = self._global_entry_plan()
        chosen_mode, payload, costs = self._choose_mode(level=9)
        phase_counts: Dict[str, int] = {}
        mask_format_counts: Dict[str, int] = {}
        mask_payload_bytes = 0
        for round_state in self.rounds:
            phase_counts[round_state.phase_name] = phase_counts.get(round_state.phase_name, 0) + 1
            name = MASK_FORMAT_NAMES.get(round_state.mask_format, f"unknown:{round_state.mask_format}")
            mask_format_counts[name] = mask_format_counts.get(name, 0) + 1
            mask_payload_bytes += len(round_state.removed_payload)
        return {
            "original_length": self.original_length,
            "final_length": len(self.final_bytes),
            "serialized_bytes": len(payload),
            "zlib_bytes": costs[chosen_mode],
            "round_count": len(self.rounds),
            "total_removed": total_removed,
            "removed_fraction": (
                float(total_removed) / float(self.original_length)
                if self.original_length
                else 0.0
            ),
            "pair_rule_budget": self.pair_rule_budget,
            "pair_rule_count": direct_grammar.rule_count,
            "pair_symbol_count": direct_grammar.symbol_count,
            "dictionary_key_count": len(keys),
            "dictionary_mode": DICT_MODE_NAMES[chosen_mode],
            "direct_zlib_bytes": costs[DICT_MODE_DIRECT],
            "shared_zlib_bytes": costs[DICT_MODE_SHARED],
            "shared_entry_count": len(entries),
            "phase_counts": phase_counts,
            "mask_format_counts": mask_format_counts,
            "mask_payload_bytes": mask_payload_bytes,
            "rounds": [
                {
                    "phase_name": round_state.phase_name,
                    "length_before": round_state.length_before,
                    "context_radius": round_state.context_radius,
                    "dictionary_size": len(round_state.dictionary or {}),
                    "dictionary_cap": round_state.dictionary_cap,
                    "pair_rule_budget": round_state.pair_rule_budget,
                    "mask_format": MASK_FORMAT_NAMES.get(round_state.mask_format, round_state.mask_format),
                    "mask_payload_bytes": len(round_state.removed_payload),
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


def _pack_removed_mask(mask: List[bool]) -> bytes:
    width = (len(mask) + 7) // 8
    payload = bytearray(width)
    for index, bit in enumerate(mask):
        if bit:
            payload[index // 8] |= 1 << (index % 8)
    return bytes(payload)


def _positions_from_bitset(payload: bytes, length: int) -> Tuple[int, ...]:
    positions: List[int] = []
    for index in range(length):
        if payload[index // 8] & (1 << (index % 8)):
            positions.append(index)
    return tuple(positions)


def _encode_varint(value: int) -> bytes:
    if value <= 0:
        raise ValueError("BLX6 varint expects a positive integer")
    out = bytearray()
    current = value
    while current >= 0x80:
        out.append((current & 0x7F) | 0x80)
        current >>= 7
    out.append(current)
    return bytes(out)


def _decode_varint(stream: BytesIO) -> int:
    shift = 0
    value = 0
    while True:
        blob = stream.read(1)
        if len(blob) != 1:
            raise ValueError("Unexpected end of BLX6 varint payload")
        byte = blob[0]
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
        if shift > 63:
            raise ValueError("BLX6 varint overflow")
    if value <= 0:
        raise ValueError("BLX6 varint decoded a non-positive delta")
    return value


def _encode_removed_positions(positions: Tuple[int, ...]) -> bytes:
    out = bytearray()
    previous = -1
    for position in positions:
        delta = position - previous
        out.extend(_encode_varint(delta))
        previous = position
    return bytes(out)


def _decode_removed_positions(payload: bytes, length_before: int) -> Tuple[int, ...]:
    positions: List[int] = []
    previous = -1
    stream = BytesIO(payload)
    while stream.tell() < len(payload):
        delta = _decode_varint(stream)
        position = previous + delta
        if position < 0 or position >= length_before:
            raise ValueError("BLX6 decoded a removed position outside the round length")
        positions.append(position)
        previous = position
    return tuple(positions)


def _encode_index_list(indices: Tuple[int, ...]) -> bytes:
    out = bytearray()
    previous = -1
    for index in indices:
        delta = index - previous
        out.extend(_encode_varint(delta))
        previous = index
    return bytes(out)


def _decode_index_list(payload: bytes, max_index: int) -> Tuple[int, ...]:
    indices: List[int] = []
    previous = -1
    stream = BytesIO(payload)
    while stream.tell() < len(payload):
        delta = _decode_varint(stream)
        index = previous + delta
        if index < 0 or index >= max_index:
            raise ValueError("BLX6 decoded an entry index outside the shared table")
        indices.append(index)
        previous = index
    return tuple(indices)


def _select_removals(
    data: bytes,
    dictionary: Dict[bytes, int],
    context_radius: int,
) -> Tuple[List[bool], Tuple[int, ...], bytes, int, Dict[bytes, int]]:
    removed = [False] * len(data)
    positions: List[int] = []
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
            positions.append(index)
            removed_count += 1
            last_removed = index
            used_keys[key] = dictionary[key]
        else:
            survivors.append(data[index])
        index += 1
    return removed, tuple(positions), bytes(survivors), removed_count, used_keys


def _round_variants(
    *,
    length_before: int,
    context_radius: int,
    removed_mask: List[bool],
    removed_positions: Tuple[int, ...],
    dictionary: Dict[bytes, int],
    removed_count: int,
    phase_name: str,
    dictionary_cap: int | None,
    pair_rule_budget: int,
    mask_policy: str,
) -> List[Blinx6Round]:
    bitset_payload = _pack_removed_mask(removed_mask)
    posdelta_payload = _encode_removed_positions(removed_positions)
    if mask_policy == MASK_POLICY_BITSET:
        return [
            Blinx6Round(
                length_before=length_before,
                context_radius=context_radius,
                removed_payload=bitset_payload,
                removed_positions=removed_positions,
                removed_count=removed_count,
                phase_name=phase_name,
                dictionary=dictionary,
                dictionary_cap=dictionary_cap,
                pair_rule_budget=pair_rule_budget,
                mask_format=MASK_FORMAT_BITSET,
            )
        ]
    if mask_policy == MASK_POLICY_POSDELTA:
        return [
            Blinx6Round(
                length_before=length_before,
                context_radius=context_radius,
                removed_payload=posdelta_payload,
                removed_positions=removed_positions,
                removed_count=removed_count,
                phase_name=phase_name,
                dictionary=dictionary,
                dictionary_cap=dictionary_cap,
                pair_rule_budget=pair_rule_budget,
                mask_format=MASK_FORMAT_POSDELTA,
            )
        ]
    if mask_policy != MASK_POLICY_ADAPTIVE:
        raise ValueError(f"Unknown BLX6 mask policy: {mask_policy}")
    variants = [
        Blinx6Round(
            length_before=length_before,
            context_radius=context_radius,
            removed_payload=bitset_payload,
            removed_positions=removed_positions,
            removed_count=removed_count,
            phase_name=phase_name,
            dictionary=dictionary,
            dictionary_cap=dictionary_cap,
            pair_rule_budget=pair_rule_budget,
            mask_format=MASK_FORMAT_BITSET,
        )
    ]
    if posdelta_payload != bitset_payload:
        variants.append(
            Blinx6Round(
                length_before=length_before,
                context_radius=context_radius,
                removed_payload=posdelta_payload,
                removed_positions=removed_positions,
                removed_count=removed_count,
                phase_name=phase_name,
                dictionary=dictionary,
                dictionary_cap=dictionary_cap,
                pair_rule_budget=pair_rule_budget,
                mask_format=MASK_FORMAT_POSDELTA,
            )
        )
    return variants


def _evaluate_phase(
    data: bytes,
    current: bytes,
    rounds: List[Blinx6Round],
    *,
    phase: Blinx6Phase,
    current_zlib: int,
    mask_policy: str,
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
            removed_mask, removed_positions, survivors, removed_count, used_dictionary = _select_removals(
                current,
                dictionary,
                context_radius=context_radius,
            )
            if removed_count < phase.min_removed or not used_dictionary:
                continue
            for pair_rule_budget in phase.pair_rule_budgets:
                for round_state in _round_variants(
                    length_before=len(current),
                    context_radius=context_radius,
                    removed_mask=removed_mask,
                    removed_positions=removed_positions,
                    dictionary=used_dictionary,
                    removed_count=removed_count,
                    phase_name=phase.name,
                    dictionary_cap=dictionary_cap,
                    pair_rule_budget=pair_rule_budget,
                    mask_policy=mask_policy,
                ):
                    candidate_zlib = Blinx6Compressed(
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
    phases: Tuple[Blinx6Phase, ...] = DEFAULT_PHASES,
    mask_policy: str = DEFAULT_MASK_POLICY,
) -> Blinx6Compressed:
    rounds: List[Blinx6Round] = []
    current = data
    current_pair_rule_budget = 0
    current_zlib = Blinx6Compressed(
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
    return Blinx6Compressed(
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
        raise ValueError("Unexpected end of BLX6 payload")
    return blob


def deserialize(payload: bytes) -> Blinx6Compressed:
    stream = BytesIO(payload)
    if _read_exact(stream, 4) != MAGIC:
        raise ValueError("Not a BLX6 payload")
    version_mode, original_length, round_count, pair_rule_budget, final_length = struct.unpack(
        "<BIIII",
        _read_exact(stream, 17),
    )
    version, dictionary_mode = _unpack_version_mode(version_mode)
    if version != VERSION:
        raise ValueError(f"Unsupported BLX6 version: {version}")
    grammar_count = struct.unpack("<I", _read_exact(stream, 4))[0]
    grammar_rules = [struct.unpack("<HH", _read_exact(stream, 4)) for _ in range(grammar_count)]
    memo: Dict[int, bytes] = {}
    rounds: List[Blinx6Round] = []

    if dictionary_mode == DICT_MODE_DIRECT:
        total_key_count = struct.unpack("<I", _read_exact(stream, 4))[0]
        keys_read = 0
        for _ in range(round_count):
            length_before, context_radius, entry_count, mask_format, payload_size = struct.unpack(
                "<IIIII",
                _read_exact(stream, 20),
            )
            dictionary: Dict[bytes, int] = {}
            for _ in range(entry_count):
                value, encoded_len = struct.unpack("<BH", _read_exact(stream, 3))
                encoded = [struct.unpack("<H", _read_exact(stream, 2))[0] for _ in range(encoded_len)]
                key = b"".join(_expand_symbol(symbol, grammar_rules, memo) for symbol in encoded)
                dictionary[key] = value
                keys_read += 1
            removed_payload = _read_exact(stream, payload_size)
            if mask_format == MASK_FORMAT_BITSET:
                removed_positions = _positions_from_bitset(removed_payload, length_before)
            elif mask_format == MASK_FORMAT_POSDELTA:
                removed_positions = _decode_removed_positions(removed_payload, length_before)
            else:
                raise ValueError(f"Unsupported BLX6 mask format: {mask_format}")
            rounds.append(
                Blinx6Round(
                    length_before=length_before,
                    context_radius=context_radius,
                    removed_payload=removed_payload,
                    removed_positions=removed_positions,
                    removed_count=len(removed_positions),
                    phase_name="replay",
                    dictionary=dictionary,
                    mask_format=mask_format,
                )
            )
        if keys_read != total_key_count:
            raise ValueError("BLX6 direct key count mismatch")
    elif dictionary_mode == DICT_MODE_SHARED:
        entry_count = struct.unpack("<I", _read_exact(stream, 4))[0]
        shared_entries: List[Tuple[bytes, int]] = []
        for _ in range(entry_count):
            value, encoded_len = struct.unpack("<BH", _read_exact(stream, 3))
            encoded = [struct.unpack("<H", _read_exact(stream, 2))[0] for _ in range(encoded_len)]
            key = b"".join(_expand_symbol(symbol, grammar_rules, memo) for symbol in encoded)
            shared_entries.append((key, value))
        for _ in range(round_count):
            length_before, context_radius, local_entry_count, mask_format, payload_size, index_payload_size = struct.unpack(
                "<IIIIII",
                _read_exact(stream, 24),
            )
            entry_payload = _read_exact(stream, index_payload_size)
            removed_payload = _read_exact(stream, payload_size)
            entry_ids = _decode_index_list(entry_payload, len(shared_entries))
            if len(entry_ids) != local_entry_count:
                raise ValueError("BLX6 shared entry count mismatch")
            dictionary = {shared_entries[entry_id][0]: shared_entries[entry_id][1] for entry_id in entry_ids}
            if mask_format == MASK_FORMAT_BITSET:
                removed_positions = _positions_from_bitset(removed_payload, length_before)
            elif mask_format == MASK_FORMAT_POSDELTA:
                removed_positions = _decode_removed_positions(removed_payload, length_before)
            else:
                raise ValueError(f"Unsupported BLX6 mask format: {mask_format}")
            rounds.append(
                Blinx6Round(
                    length_before=length_before,
                    context_radius=context_radius,
                    removed_payload=removed_payload,
                    removed_positions=removed_positions,
                    removed_count=len(removed_positions),
                    phase_name="replay",
                    dictionary=dictionary,
                    mask_format=mask_format,
                )
            )
    else:
        raise ValueError(f"Unsupported BLX6 dictionary mode: {dictionary_mode}")

    final_bytes = _read_exact(stream, final_length)
    return Blinx6Compressed(
        original_length=original_length,
        final_bytes=final_bytes,
        rounds=rounds,
        pair_rule_budget=pair_rule_budget,
    )


def decompress(compressed: Blinx6Compressed) -> bytes:
    current = compressed.final_bytes
    for round_state in reversed(compressed.rounds):
        removed_positions = round_state.removed_positions
        next_removed_index = 0
        next_removed = removed_positions[next_removed_index] if removed_positions else None
        rebuilt: List[int | None] = [None] * round_state.length_before
        survivor_index = 0
        for index in range(round_state.length_before):
            if next_removed is not None and index == next_removed:
                next_removed_index += 1
                next_removed = (
                    removed_positions[next_removed_index]
                    if next_removed_index < len(removed_positions)
                    else None
                )
                continue
            rebuilt[index] = current[survivor_index]
            survivor_index += 1
        if survivor_index != len(current):
            raise ValueError("BLX6 survivor length mismatch during rebuild")
        for index in removed_positions:
            radius = round_state.context_radius
            if index < radius or index >= len(rebuilt) - radius:
                raise ValueError("BLX6 attempted to remove boundary byte")
            left = rebuilt[index - radius : index]
            right = rebuilt[index + 1 : index + 1 + radius]
            if any(value is None for value in left) or any(value is None for value in right):
                raise ValueError("BLX6 encountered overlapping removals within a round")
            key = bytes(left + right)  # type: ignore[arg-type]
            if key not in (round_state.dictionary or {}):
                raise ValueError(f"BLX6 missing dictionary entry for context {key}")
            rebuilt[index] = (round_state.dictionary or {})[key]
        current = bytes(rebuilt)  # type: ignore[arg-type]
    if len(current) != compressed.original_length:
        raise ValueError("BLX6 output length mismatch")
    return current


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    phases: Tuple[Blinx6Phase, ...] = DEFAULT_PHASES,
    mask_policy: str = DEFAULT_MASK_POLICY,
) -> Tuple[Blinx6Compressed, bool]:
    compressed = compress(
        data,
        max_rounds=max_rounds,
        phases=phases,
        mask_policy=mask_policy,
    )
    payload = compressed.serialize()
    rebuilt = decompress(deserialize(payload))
    return compressed, rebuilt == data


def phase_names(phases: Iterable[Blinx6Phase]) -> List[str]:
    return _phase_names(phases)
