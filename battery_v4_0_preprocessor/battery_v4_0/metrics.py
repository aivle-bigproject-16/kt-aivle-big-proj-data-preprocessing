from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import IdStats, Sample


@dataclass(frozen=True)
class SampleMetrics:
    images: int
    defect_images: int
    annotations: int
    defect_image_ratio: float
    annotations_per_image: float
    normal_ratio: float


def sample_metrics(samples: Iterable[Sample]) -> SampleMetrics:
    materialized = list(samples)
    images = len(materialized)
    defect_images = sum(sample.is_defect for sample in materialized)
    annotations = sum(len(sample.det_lines) for sample in materialized)
    return SampleMetrics(
        images=images,
        defect_images=defect_images,
        annotations=annotations,
        defect_image_ratio=defect_images / images if images else 0.0,
        annotations_per_image=annotations / images if images else 0.0,
        normal_ratio=(images - defect_images) / images if images else 0.0,
    )


def selected_samples(
    ids: Iterable[IdStats],
    group_name: str = "",
    selected_for: Callable[[str, IdStats], Iterable[Sample]] | None = None,
) -> list[Sample]:
    selector = selected_for or (lambda _name, stats: stats.selected_samples)
    return [sample for stats in ids for sample in selector(group_name, stats)]


def group_selected_samples(
    ids: Iterable[IdStats],
    group_name: Callable[[IdStats], str],
) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for stats in ids:
        groups[group_name(stats)].extend(stats.selected_samples)
    return dict(groups)


def split_fold_group_name(stats: IdStats) -> str:
    if stats.modality == "CT":
        return "CT_test" if stats.split_role == "test" else f"CT_fold_{stats.fold_id}"
    return f"EXT_{stats.split_role}"


def report_dataset_group_name(stats: IdStats) -> str:
    if stats.modality == "CT":
        return "CT_test" if stats.split_role == "test" else "CT_development"
    return f"EXT_{stats.split_role}"
