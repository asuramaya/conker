from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import struct
import zlib


MAGIC = b"BLX1"
VERSION = 2


@dataclass
class Blinx1Round:
    length_before: int
    context_radius: int
    removed_mask: bytes
    dictionary: dict[bytes, int]
    removed_count: int
    zlib_after_round: int | None = None


@dataclass
class Blinx1Compressed:
    original_length: int
    final_bytes: bytes
    rounds: list[Blinx1Round]

    def serialize(self) -> bytes:
        out = BytesIO()
        out.write(MAGIC)
        out.write(struct.pack("<BII", VERSION, self.original_length, len(self.rounds)))
        for round_state in self.rounds:
            out.write(
                struct.pack(
                    "<IIII",
                    round_state.length_before,
                    round_state.context_radius,
                    len(round_state.dictionary),
                    len(round_state.removed_mask),
                )
            )
            for key, value in sorted(round_state.dictionary.items()):
                out.write(key)
                out.write(bytes((value,)))
            out.write(round_state.removed_mask)
        out.write(struct.pack("<I", len(self.final_bytes)))
        out.write(self.final_bytes)
        return out.getvalue()

    def zlib_size(self, level: int = 9) -> int:
        return len(zlib.compress(self.serialize(), level))

    def stats(self) -> dict[str, object]:
        total_removed = sum(round_state.removed_count for round_state in self.rounds)
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
            "rounds": [
                {
                    "length_before": round_state.length_before,
                    "context_radius": round_state.context_radius,
                    "dictionary_size": len(round_state.dictionary),
                    "removed_count": round_state.removed_count,
                    "zlib_after_round": round_state.zlib_after_round,
                    "removed_fraction": (
                        float(round_state.removed_count) / float(round_state.length_before)
                        if round_state.length_before
                        else 0.0
                    ),
                }
                for round_state in self.rounds
            ],
        }


def _pack_removed_mask(mask: list[bool]) -> bytes:
    width = (len(mask) + 7) // 8
    payload = bytearray(width)
    for index, bit in enumerate(mask):
        if bit:
            payload[index // 8] |= 1 << (index % 8)
    return bytes(payload)


def _unpack_removed_mask(payload: bytes, length: int) -> list[bool]:
    mask = [False] * length
    for index in range(length):
        mask[index] = bool(payload[index // 8] & (1 << (index % 8)))
    return mask


def _read_exact(stream: BytesIO, size: int) -> bytes:
    blob = stream.read(size)
    if len(blob) != size:
        raise ValueError("Unexpected end of BLX1 payload")
    return blob


def _build_unique_context_dictionary(
    data: bytes, context_radius: int, min_occurrences: int
) -> dict[bytes, int]:
    counts: dict[bytes, dict[int, int]] = {}
    if len(data) < 2 * context_radius + 1:
        return {}
    for index in range(context_radius, len(data) - context_radius):
        key = data[index - context_radius : index] + data[index + 1 : index + 1 + context_radius]
        bucket = counts.setdefault(key, {})
        bucket[data[index]] = bucket.get(data[index], 0) + 1
    unique: dict[bytes, int] = {}
    for key, bucket in counts.items():
        if len(bucket) != 1:
            continue
        value, count = next(iter(bucket.items()))
        if count >= min_occurrences:
            unique[key] = value
    return unique


def _select_removals(
    data: bytes, dictionary: dict[bytes, int], context_radius: int
) -> tuple[list[bool], bytes, int, dict[bytes, int]]:
    removed = [False] * len(data)
    survivors = bytearray()
    removed_count = 0
    last_removed = -10**9
    used_keys: dict[bytes, int] = {}
    index = 0
    while index < len(data):
        can_remove = (
            context_radius <= index < len(data) - context_radius
            and index - last_removed > context_radius
            and (
                data[index - context_radius : index]
                + data[index + 1 : index + 1 + context_radius]
            )
            in dictionary
            and dictionary[
                data[index - context_radius : index]
                + data[index + 1 : index + 1 + context_radius]
            ]
            == data[index]
        )
        if can_remove:
            key = data[index - context_radius : index] + data[index + 1 : index + 1 + context_radius]
            removed[index] = True
            removed_count += 1
            last_removed = index
            used_keys[key] = dictionary[key]
        else:
            survivors.append(data[index])
        index += 1
    return removed, bytes(survivors), removed_count, used_keys


def punch_round(
    data: bytes,
    *,
    context_radius: int = 1,
    min_occurrences: int = 2,
) -> tuple[bytes, Blinx1Round] | None:
    if len(data) < 2 * context_radius + 1:
        return None
    dictionary = _build_unique_context_dictionary(
        data,
        context_radius=context_radius,
        min_occurrences=min_occurrences,
    )
    if not dictionary:
        return None
    removed_mask, survivors, removed_count, used_dictionary = _select_removals(
        data,
        dictionary,
        context_radius=context_radius,
    )
    if removed_count == 0:
        return None
    round_state = Blinx1Round(
        length_before=len(data),
        context_radius=context_radius,
        removed_mask=_pack_removed_mask(removed_mask),
        dictionary=used_dictionary,
        removed_count=removed_count,
    )
    return survivors, round_state


def compress(
    data: bytes,
    *,
    max_rounds: int = 8,
    min_occurrences: int = 2,
    min_removed: int = 8,
    candidate_radii: tuple[int, ...] = (1, 2, 3, 4),
    selection_mode: str = "profit",
) -> Blinx1Compressed:
    rounds: list[Blinx1Round] = []
    current = data
    current_zlib = Blinx1Compressed(original_length=len(data), final_bytes=current, rounds=rounds).zlib_size()
    for _ in range(max_rounds):
        best_step: tuple[bytes, Blinx1Round] | None = None
        best_zlib = current_zlib
        best_removed = -1
        for context_radius in candidate_radii:
            step = punch_round(
                current,
                context_radius=context_radius,
                min_occurrences=min_occurrences,
            )
            if step is None:
                continue
            next_current, round_state = step
            if round_state.removed_count < min_removed:
                continue
            candidate_rounds = rounds + [round_state]
            candidate_zlib = Blinx1Compressed(
                original_length=len(data),
                final_bytes=next_current,
                rounds=candidate_rounds,
            ).zlib_size()
            round_state.zlib_after_round = candidate_zlib
            if selection_mode == "profit":
                if candidate_zlib < best_zlib:
                    best_step = (next_current, round_state)
                    best_zlib = candidate_zlib
            elif selection_mode == "discovery":
                if (
                    round_state.removed_count > best_removed
                    or (
                        round_state.removed_count == best_removed
                        and candidate_zlib < best_zlib
                    )
                ):
                    best_step = (next_current, round_state)
                    best_zlib = candidate_zlib
                    best_removed = round_state.removed_count
            else:
                raise ValueError(f"Unknown BLX1 selection mode: {selection_mode}")
        if best_step is None:
            break
        next_current, round_state = best_step
        current = next_current
        rounds.append(round_state)
        current_zlib = best_zlib
    return Blinx1Compressed(
        original_length=len(data),
        final_bytes=current,
        rounds=rounds,
    )


def deserialize(payload: bytes) -> Blinx1Compressed:
    stream = BytesIO(payload)
    if _read_exact(stream, 4) != MAGIC:
        raise ValueError("Not a BLX1 payload")
    version, original_length, round_count = struct.unpack("<BII", _read_exact(stream, 9))
    if version not in (1, VERSION):
        raise ValueError(f"Unsupported BLX1 version: {version}")
    rounds: list[Blinx1Round] = []
    for _ in range(round_count):
        if version == 1:
            length_before, dictionary_size, mask_size = struct.unpack(
                "<III", _read_exact(stream, 12)
            )
            context_radius = 1
        else:
            length_before, context_radius, dictionary_size, mask_size = struct.unpack(
                "<IIII", _read_exact(stream, 16)
            )
        dictionary: dict[bytes, int] = {}
        key_width = 2 * context_radius
        for _ in range(dictionary_size):
            if version == 1:
                left, right, value = _read_exact(stream, 3)
                dictionary[bytes((left, right))] = value
            else:
                key = _read_exact(stream, key_width)
                value = _read_exact(stream, 1)[0]
                dictionary[key] = value
        removed_mask = _read_exact(stream, mask_size)
        rounds.append(
            Blinx1Round(
                length_before=length_before,
                context_radius=context_radius,
                removed_mask=removed_mask,
                dictionary=dictionary,
                removed_count=sum(_unpack_removed_mask(removed_mask, length_before)),
            )
        )
    final_length = struct.unpack("<I", _read_exact(stream, 4))[0]
    final_bytes = _read_exact(stream, final_length)
    return Blinx1Compressed(
        original_length=original_length,
        final_bytes=final_bytes,
        rounds=rounds,
    )


def decompress(compressed: Blinx1Compressed) -> bytes:
    current = compressed.final_bytes
    for round_state in reversed(compressed.rounds):
        removed_mask = _unpack_removed_mask(round_state.removed_mask, round_state.length_before)
        rebuilt = [None] * round_state.length_before
        survivor_index = 0
        for index, removed in enumerate(removed_mask):
            if not removed:
                rebuilt[index] = current[survivor_index]
                survivor_index += 1
        if survivor_index != len(current):
            raise ValueError("BLX1 survivor length mismatch during rebuild")
        for index, removed in enumerate(removed_mask):
            if not removed:
                continue
            context_radius = round_state.context_radius
            if index < context_radius or index >= len(rebuilt) - context_radius:
                raise ValueError("BLX1 attempted to remove boundary byte")
            left = rebuilt[index - context_radius : index]
            right = rebuilt[index + 1 : index + 1 + context_radius]
            if any(value is None for value in left) or any(value is None for value in right):
                raise ValueError("BLX1 encountered overlapping removals within a round")
            key = bytes(left + right)  # type: ignore[arg-type]
            if key not in round_state.dictionary:
                raise ValueError(f"BLX1 missing dictionary entry for context {key}")
            rebuilt[index] = round_state.dictionary[key]
        current = bytes(rebuilt)  # type: ignore[arg-type]
    if len(current) != compressed.original_length:
        raise ValueError("BLX1 output length mismatch")
    return current


def roundtrip_ok(
    data: bytes,
    *,
    max_rounds: int = 8,
    min_occurrences: int = 2,
    min_removed: int = 8,
    candidate_radii: tuple[int, ...] = (1, 2, 3, 4),
    selection_mode: str = "profit",
) -> tuple[Blinx1Compressed, bool]:
    compressed = compress(
        data,
        max_rounds=max_rounds,
        min_occurrences=min_occurrences,
        min_removed=min_removed,
        candidate_radii=candidate_radii,
        selection_mode=selection_mode,
    )
    rebuilt = decompress(compressed)
    return compressed, rebuilt == data
