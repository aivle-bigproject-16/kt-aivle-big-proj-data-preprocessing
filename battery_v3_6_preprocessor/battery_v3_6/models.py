from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import sample_metrics


@dataclass(frozen=True)
class ParsedName:
    modality: str
    battery_id: str
    axis: str = ""
    form: str = ""
    frame: str = ""


@dataclass
class ConvertedDefect:
    det_lines: list[str] = field(default_factory=list)
    seg_lines: list[str] = field(default_factory=list)
    det_valid: bool = False
    seg_valid: bool = False
    min_extent_applied: bool = False
    multipart_split_count: int = 0
    polygon_count: int = 0
    polygon_area_ratio: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class Sample:
    sample_id: str
    modality: str = ""
    battery_id: str = ""
    axis: str = ""
    original_stem: str = ""
    image_path: Path | None = None
    json_path: Path | None = None
    image_relative_posix: str = ""
    json_relative_posix: str = ""
    image_w: int = 0
    image_h: int = 0
    roi_w: int | None = None
    roi_h: int | None = None
    application: str = ""
    original_is_normal: Any = None
    is_normal_interpreted: bool = False
    defects: list[dict[str, Any]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    det_lines: list[str] = field(default_factory=list)
    seg_lines: list[str] = field(default_factory=list)
    included_det: bool = False
    included_seg: bool = False
    exclusion_reason_det: str = ""
    exclusion_reason_seg: str = ""
    pixel_hash: str = ""
    json_sha256: str = ""
    duplicate_group_id: str = ""
    duplicate_action: str = ""
    original_defect_count: int = 0
    yolo_det_instance_count: int = 0
    yolo_seg_instance_count: int = 0
    multipart_split_count: int = 0
    min_extent_applied: bool = False
    porosity_polygon_count: int = 0
    porosity_area_sum_ratio: float = 0.0
    selected_rgb_160: bool = False
    split_role: str = ""
    fold_id: str = ""
    selected: bool = False
    scanned_pair: bool = False
    raw_parsed_image: bool = False
    issues: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_defect(self) -> bool:
        return bool(self.det_lines)

    @classmethod
    def stub(cls, sample_id: str, is_defect: bool) -> "Sample":
        return cls(sample_id=sample_id, det_lines=["x"] if is_defect else [])


@dataclass
class IdStats:
    modality: str
    battery_id: str
    application: str
    samples: list[Sample]
    selected_samples: list[Sample] = field(default_factory=list)
    split_role: str = ""
    fold_id: str = ""

    @property
    def valid_images(self) -> int:
        return len(self.samples)

    @property
    def defect_images(self) -> int:
        return sample_metrics(self.samples).defect_images

    @property
    def defect_image_ratio(self) -> float:
        return sample_metrics(self.samples).defect_image_ratio

    @property
    def annotations(self) -> int:
        return sample_metrics(self.samples).annotations

    @property
    def annotations_per_image(self) -> float:
        return sample_metrics(self.samples).annotations_per_image

    @property
    def has_damaged(self) -> bool:
        return any("Damaged" in s.class_names for s in self.samples)

    @property
    def has_pollution(self) -> bool:
        return any("Pollution" in s.class_names for s in self.samples)

    def axis_count(self, axis: str) -> int:
        return sum(s.axis == axis for s in self.samples)
