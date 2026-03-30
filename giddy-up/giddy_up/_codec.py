from __future__ import annotations


def build_unique_context_dictionary(data: bytes, context_radius: int, min_occurrences: int) -> dict[bytes, int]:
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


def select_removals(
    data: bytes, dictionary: dict[bytes, int], context_radius: int
) -> tuple[list[bool], bytes, int, dict[bytes, int]]:
    removed = [False] * len(data)
    survivors = bytearray()
    removed_count = 0
    last_removed = -10**9
    used_keys: dict[bytes, int] = {}
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
