from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from .models import Sample


AREA_BIN_ORDER = (
    "zero",
    "lt_0.1pct",
    "0.1_1pct",
    "1_5pct",
    "5_10pct",
    "10_25pct",
    "25_40pct",
    "40_50pct",
    "ge_50pct",
)

CT_PRE_SPLIT_AREA_THRESHOLD = 0.25
CT_PRE_SPLIT_POLICY_PRECISION = 8
CT_PRE_SPLIT_EXCLUSION_REASON = "ct_porosity_bbox_max_ratio_ge_0.25"


def porosity_area_bin(value: float) -> str:
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"porosity area ratio must be finite and >=0: {value!r}")
    if value == 0:
        return "zero"
    if value < 0.001:
        return "lt_0.1pct"
    if value < 0.01:
        return "0.1_1pct"
    if value < 0.05:
        return "1_5pct"
    if value < 0.10:
        return "5_10pct"
    if value < 0.25:
        return "10_25pct"
    if value < 0.40:
        return "25_40pct"
    if value < 0.50:
        return "40_50pct"
    return "ge_50pct"


def is_ct_pre_split_excluded(sample: Sample) -> bool:
    """Return whether v4.0 excludes a CT image before any ID split.

    The policy uses the largest component bbox computed from the same normalized
    8-decimal coordinates written to YOLO. The boundary is inclusive. RGB is
    deliberately unaffected.
    """
    return (
        sample.modality == "CT"
        and sample.porosity_bbox_max_ratio >= CT_PRE_SPLIT_AREA_THRESHOLD
    )


def ct_area_features(samples: Iterable[Sample]) -> Counter[str]:
    features: Counter[str] = Counter()
    for sample in samples:
        if sample.modality != "CT":
            continue
        area_bin = porosity_area_bin(sample.porosity_area_sum_ratio)
        features["images"] += 1
        features[f"axis:{sample.axis}"] += 1
        features[f"area:{area_bin}"] += 1
        features[f"joint:{sample.axis}|{area_bin}"] += 1
    return features
