from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import struct
from typing import Union
import zlib


MAGIC = b"BLX2"
VERSION = 1
RAW_TAG = 0
RULE_TAG = 1


@dataclass(frozen=True)
class RawChunk:
    data: bytes


@dataclass(frozen=True)
class RuleRef:
    rule_id: int


StreamItem = Union[RawChunk, RuleRef]


@dataclass
class Blinx2Rule:
    rule_id: int
    phrase: bytes
    match_count: int
    replaced_bytes: int
    zlib_after_rule: int | None = None


@dataclass
class Blinx2Compressed:
    original_length: int
    rules: list[Blinx2Rule]
    stream: list[StreamItem]

    def serialize(self) -> bytes:
        out = BytesIO()
        out.write(MAGIC)
        out.write(struct.pack("<BII", VERSION, self.original_length, len(self.rules)))
        for rule in self.rules:
            out.write(struct.pack("<HI", len(rule.phrase), rule.match_count))
            out.write(rule.phrase)
        out.write(struct.pack("<I", len(self.stream)))
        for item in self.stream:
            if isinstance(item, RawChunk):
                out.write(bytes((RAW_TAG,)))
                out.write(struct.pack("<I", len(item.data)))
                out.write(item.data)
            else:
                out.write(bytes((RULE_TAG,)))
                out.write(struct.pack("<H", item.rule_id))
        return out.getvalue()

    def zlib_size(self, level: int = 9) -> int:
        return len(zlib.compress(self.serialize(), level))

    def decoded_bytes(self) -> bytes:
        rulebook = {rule.rule_id: rule.phrase for rule in self.rules}
        out = bytearray()
        for item in self.stream:
            if isinstance(item, RawChunk):
                out.extend(item.data)
            else:
                out.extend(rulebook[item.rule_id])
        return bytes(out)

    def stats(self) -> dict[str, object]:
        replaced_total = sum(rule.replaced_bytes for rule in self.rules)
        return {
            "original_length": self.original_length,
            "final_stream_length": sum(
                len(item.data) if isinstance(item, RawChunk) else 1 for item in self.stream
            ),
            "serialized_bytes": len(self.serialize()),
            "zlib_bytes": self.zlib_size(),
            "rule_count": len(self.rules),
            "replaced_bytes_total": replaced_total,
            "replaced_fraction": (
                float(replaced_total) / float(self.original_length)
                if self.original_length
                else 0.0
            ),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "phrase_len": len(rule.phrase),
                    "match_count": rule.match_count,
                    "replaced_bytes": rule.replaced_bytes,
                    "zlib_after_rule": rule.zlib_after_rule,
                }
                for rule in self.rules
            ],
        }


def _merge_stream(stream: list[StreamItem]) -> list[StreamItem]:
    merged: list[StreamItem] = []
    buffer = bytearray()
    for item in stream:
        if isinstance(item, RawChunk):
            if item.data:
                buffer.extend(item.data)
            continue
        if buffer:
            merged.append(RawChunk(bytes(buffer)))
            buffer.clear()
        merged.append(item)
    if buffer:
        merged.append(RawChunk(bytes(buffer)))
    return merged


def _iter_candidate_counts(
    stream: list[StreamItem],
    *,
    min_len: int,
    max_len: int,
) -> dict[bytes, int]:
    counts: dict[bytes, int] = {}
    for item in stream:
        if not isinstance(item, RawChunk):
            continue
        data = item.data
        for length in range(min_len, max_len + 1):
            if len(data) < length:
                continue
            for start in range(0, len(data) - length + 1):
                phrase = data[start : start + length]
                counts[phrase] = counts.get(phrase, 0) + 1
    return counts


def _rough_phrase_score(phrase: bytes, count: int) -> int:
    phrase_len = len(phrase)
    return count * max(0, phrase_len - 1) - phrase_len - 8


def _top_candidates(
    stream: list[StreamItem],
    *,
    min_len: int,
    max_len: int,
    min_occurrences: int,
    top_k: int,
) -> list[tuple[bytes, int]]:
    counts = _iter_candidate_counts(stream, min_len=min_len, max_len=max_len)
    candidates = [
        (phrase, count)
        for phrase, count in counts.items()
        if count >= min_occurrences and _rough_phrase_score(phrase, count) > 0
    ]
    candidates.sort(key=lambda item: (_rough_phrase_score(item[0], item[1]), len(item[0]), item[1]), reverse=True)
    return candidates[:top_k]


def _apply_rule(
    stream: list[StreamItem],
    *,
    phrase: bytes,
    rule_id: int,
) -> tuple[list[StreamItem], int, int]:
    new_stream: list[StreamItem] = []
    match_count = 0
    replaced_bytes = 0
    phrase_len = len(phrase)
    for item in stream:
        if not isinstance(item, RawChunk):
            new_stream.append(item)
            continue
        data = item.data
        cursor = 0
        chunk_parts: list[StreamItem] = []
        while cursor < len(data):
            found = data.find(phrase, cursor)
            if found < 0:
                tail = data[cursor:]
                if tail:
                    chunk_parts.append(RawChunk(tail))
                break
            prefix = data[cursor:found]
            if prefix:
                chunk_parts.append(RawChunk(prefix))
            chunk_parts.append(RuleRef(rule_id))
            match_count += 1
            replaced_bytes += phrase_len
            cursor = found + phrase_len
        new_stream.extend(chunk_parts)
    return _merge_stream(new_stream), match_count, replaced_bytes


def compress(
    data: bytes,
    *,
    min_len: int = 4,
    max_len: int = 12,
    min_occurrences: int = 2,
    max_rules: int = 8,
    top_candidates: int = 64,
    selection_mode: str = "profit",
) -> Blinx2Compressed:
    stream: list[StreamItem] = [RawChunk(data)]
    rules: list[Blinx2Rule] = []
    current = Blinx2Compressed(original_length=len(data), rules=rules.copy(), stream=stream).zlib_size()
    for rule_id in range(max_rules):
        candidates = _top_candidates(
            stream,
            min_len=min_len,
            max_len=max_len,
            min_occurrences=min_occurrences,
            top_k=top_candidates,
        )
        if not candidates:
            break
        best_stream: list[StreamItem] | None = None
        best_rule: Blinx2Rule | None = None
        best_zlib = current
        best_replaced = -1
        for phrase, _ in candidates:
            candidate_stream, match_count, replaced_bytes = _apply_rule(
                stream,
                phrase=phrase,
                rule_id=rule_id,
            )
            if match_count < min_occurrences:
                continue
            candidate_rule = Blinx2Rule(
                rule_id=rule_id,
                phrase=phrase,
                match_count=match_count,
                replaced_bytes=replaced_bytes,
            )
            candidate_rules = rules + [candidate_rule]
            candidate_zlib = Blinx2Compressed(
                original_length=len(data),
                rules=candidate_rules,
                stream=candidate_stream,
            ).zlib_size()
            candidate_rule.zlib_after_rule = candidate_zlib
            if selection_mode == "profit":
                if candidate_zlib < best_zlib:
                    best_zlib = candidate_zlib
                    best_stream = candidate_stream
                    best_rule = candidate_rule
            elif selection_mode == "discovery":
                if (
                    replaced_bytes > best_replaced
                    or (replaced_bytes == best_replaced and candidate_zlib < best_zlib)
                ):
                    best_replaced = replaced_bytes
                    best_zlib = candidate_zlib
                    best_stream = candidate_stream
                    best_rule = candidate_rule
            else:
                raise ValueError(f"Unknown BLX2 selection mode: {selection_mode}")
        if best_stream is None or best_rule is None:
            break
        stream = best_stream
        rules.append(best_rule)
        current = best_zlib
    return Blinx2Compressed(
        original_length=len(data),
        rules=rules,
        stream=stream,
    )


def decompress(compressed: Blinx2Compressed) -> bytes:
    rebuilt = compressed.decoded_bytes()
    if len(rebuilt) != compressed.original_length:
        raise ValueError("BLX2 output length mismatch")
    return rebuilt


def roundtrip_ok(
    data: bytes,
    *,
    min_len: int = 4,
    max_len: int = 12,
    min_occurrences: int = 2,
    max_rules: int = 8,
    top_candidates: int = 64,
    selection_mode: str = "profit",
) -> tuple[Blinx2Compressed, bool]:
    compressed = compress(
        data,
        min_len=min_len,
        max_len=max_len,
        min_occurrences=min_occurrences,
        max_rules=max_rules,
        top_candidates=top_candidates,
        selection_mode=selection_mode,
    )
    rebuilt = decompress(compressed)
    return compressed, rebuilt == data
