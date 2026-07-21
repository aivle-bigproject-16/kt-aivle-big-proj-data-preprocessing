from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import TypeVar

from .models import Sample

T = TypeVar("T")


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def order_key(seed: int, scope: str, item: str) -> str:
    return sha1_hex(f"{seed}|{scope}|{item}")


def sample_id_for(modality: str, original_stem: str, image_relative_posix: str) -> str:
    normalized = image_relative_posix.replace("\\", "/")
    return f"{modality}__{original_stem}__{sha1_hex(normalized)[:8]}"


def normalize_battery_id(value: str) -> str:
    if not value.isdigit():
        raise ValueError(f"battery_id must be digits: {value!r}")
    return str(int(value))


def natural_id_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value.casefold(), value)


def canonical_stratum_key(key: object) -> str:
    if isinstance(key, tuple):
        values = [int(value) if isinstance(value, bool) else value for value in key]
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return str(key)


def quantile_bins(values: Mapping[str, float], bins: int = 5) -> dict[str, int]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    ranked = sorted(values.items(), key=lambda pair: (pair[1], pair[0].casefold(), pair[0]))
    n = len(ranked)
    return {item: min(bins - 1, (i * bins) // n) for i, (item, _) in enumerate(ranked)} if n else {}


def largest_remainder(
    total: int,
    weights: Mapping[T, float],
    caps: Mapping[T, int],
) -> dict[T, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if sum(caps.get(k, 0) for k in weights) < total:
        raise ValueError("insufficient capacity")
    weight_sum = sum(weights.values())
    if total and weight_sum <= 0:
        raise ValueError("sum of weights must be positive")
    raw = {k: total * weights[k] / weight_sum if weight_sum else 0.0 for k in weights}
    alloc = {k: min(math.floor(raw[k]), caps.get(k, 0)) for k in weights}
    while sum(alloc.values()) < total:
        candidates = [k for k in weights if alloc[k] < caps.get(k, 0)]
        if not candidates:
            raise ValueError("insufficient capacity")
        candidates.sort(
            key=lambda k: (
                -(raw[k] - alloc[k]),
                -weights[k],
                canonical_stratum_key(k),
            )
        )
        alloc[candidates[0]] += 1
    return alloc


def stratified_sample(
    items: Sequence[Sample], count: int, seed: int, scope: str
) -> list[Sample]:
    if count >= len(items):
        return sorted(items, key=lambda sample: sample.sample_id)
    if count < 0:
        raise ValueError("count must be non-negative")
    pos = [item for item in items if item.is_defect]
    neg = [item for item in items if not item.is_defect]
    wanted_pos = round(count * len(pos) / len(items)) if items else 0
    wanted_pos = min(len(pos), max(0, wanted_pos))
    if count - wanted_pos > len(neg):
        wanted_pos = count - len(neg)
    pos.sort(key=lambda s: (order_key(seed, scope + ":p", s.sample_id), s.sample_id))
    neg.sort(key=lambda s: (order_key(seed, scope + ":n", s.sample_id), s.sample_id))
    return sorted(pos[:wanted_pos] + neg[: count - wanted_pos], key=lambda s: s.sample_id)
