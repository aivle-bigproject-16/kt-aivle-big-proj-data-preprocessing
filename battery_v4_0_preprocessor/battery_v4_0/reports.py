from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .deterministic import natural_id_key
from .ct_area import (
    AREA_BIN_ORDER,
    CT_PRE_SPLIT_AREA_THRESHOLD,
    porosity_area_bin,
)
from .metrics import (
    group_selected_samples,
    report_dataset_group_name,
    sample_metrics,
    selected_samples as collect_selected_samples,
    split_fold_group_name,
)
from .models import IdStats, Sample
from .scan import ScanResult
from .selection import (
    CT_ID_POSITIVE_RATE_BIN_ORDER,
    SelectionResult,
    ct_id_positive_rate,
    ct_id_positive_rate_bin,
)

MANIFEST_FIELDS = [
    "sample_id", "original_stem", "modality", "battery_id", "axis", "application",
    "orig_image_path", "orig_json_path", "orig_json_relative_path", "output_image_name",
    "output_label_stem", "image_w", "image_h", "roi_w", "roi_h", "original_is_normal",
    "is_normal_interpreted", "has_damaged", "has_pollution", "has_porosity",
    "defect_image_ratio_id", "pixel_hash", "json_sha256", "dup_group_id", "dup_action",
    "selected_rgb_160", "split_role", "fold_id", "included_det", "included_seg",
    "pre_split_eligible", "pre_split_exclusion_reason",
    "exclusion_reason_det", "exclusion_reason_seg", "original_defect_count",
    "yolo_det_instance_count", "yolo_seg_instance_count", "multipart_split_count",
    "min_extent_applied", "porosity_polygon_count", "porosity_area_sum_ratio",
    "porosity_bbox_max_ratio",
    "porosity_area_bin", "ct_stratum",
]

CSV_SCHEMAS: dict[str, list[str]] = {
    "id_scan_report.csv": ["modality", "battery_id", "application", "scanned_pairs", "valid_images", "pre_split_eligible_images", "pre_split_excluded_images", "x_valid", "y_valid", "z_valid", "ct_axis_set_valid", "defect_image_ratio", "annotations_per_image"],
    "split_axis_balance.csv": ["split", "x_images", "y_images", "z_images", "total_images", "x_ratio", "y_ratio", "z_ratio", "x_gap", "y_gap", "z_gap", "axis_max_gap", "axis_sum_gap"],
    "selected_battery_ids_candidate.csv": ["modality", "battery_id", "split_role", "fold_id", "application", "n_valid_images", "n_selected_images", "x_count", "y_count", "z_count", "defect_image_ratio", "annotations_per_image", "has_damaged", "has_pollution", "ct_id_positive_rate_bin"],
    "selected_battery_ids.csv": ["modality", "battery_id", "split_role", "fold_id", "application", "n_valid_images", "n_selected_images", "x_count", "y_count", "z_count", "defect_image_ratio", "annotations_per_image", "has_damaged", "has_pollution", "ct_id_positive_rate_bin"],
    "ct_id_positive_rate_stratification.csv": ["scope", "positive_rate_bin", "candidate_id_count", "target_test_id_count", "actual_test_id_count", "development_id_count", "status"],
    "test_battery_ids_candidate.csv": ["modality", "battery_id"],
    "test_battery_ids.csv": ["modality", "battery_id"],
    "matching_issues.csv": ["split", "stem", "image_path", "json_candidate_paths", "image_count", "json_count", "candidate_count", "reason"],
    "corrupt_images.csv": ["path", "error"],
    "pixel_duplicates.csv": ["dup_group_id", "pixel_hash", "modality", "battery_id", "sample_id", "action"],
    "json_anomalies.csv": ["sample_id", "path", "issue", "detail"],
    "class_name_variants.csv": ["sample_id", "raw_class", "action"],
    "polygon_issues.csv": ["sample_id", "defect_index", "issue"],
    "discarded_stats.csv": ["reason", "count"],
    "dryrun_warnings.csv": ["guard"],
    "quality_exceptions.csv": ["warning_id", "warning_code", "observed_value", "threshold", "status", "reviewer", "reviewed_at", "reason"],
    "review_warnings.csv": ["warning", "reviewer", "reviewed_at", "status"],
    "class_balance_report.csv": ["row_type", "dataset", "class_name", "battery_ids", "images", "annotations", "defect_image_ratio", "annotations_per_image", "normal_ratio"],
    "task_dataset_comparison.csv": ["dataset", "detection_images", "segmentation_images", "difference", "det_defect_ratio", "seg_defect_ratio"],
    "split_id_leakage_audit.csv": ["left", "right", "id_overlap", "pixel_hash_overlap", "status"],
    "ct_split_area_distribution.csv": ["split", "axis", "area_bin", "image_count", "within_split_share"],
    "ct_split_balance_summary.csv": ["split", "id_count", "image_count", "target_image_ratio", "actual_image_ratio"],
    "ct_large_area_review.csv": ["sample_id", "battery_id", "axis", "porosity_polygon_count", "porosity_area_sum_ratio", "area_bin", "orig_image_path", "orig_json_path"],
    "ct_id_exclusions.csv": ["battery_id", "gate", "observed", "threshold", "reason"],
    "ct_bbox_exclusions.csv": ["sample_id", "battery_id", "axis", "porosity_polygon_count", "porosity_area_sum_ratio", "porosity_bbox_max_ratio", "area_bin", "orig_image_path", "orig_json_path", "reason"],
    "manifest.csv": MANIFEST_FIELDS,
}


def fmt_float(value: float) -> str:
    return f"{value:.8f}"


def quality_exception_rows(warnings: Iterable[str]) -> list[dict[str, str]]:
    """Create stable, auditable exception rows for §17.2 quality warnings."""
    rows: list[dict[str, str]] = []
    for warning in sorted(set(warnings)):
        if "annotations_per_image deviation=" in warning:
            code, threshold = "annotation_density_deviation", "relative_deviation<=0.20"
        elif "required class missing" in warning:
            code, threshold = "required_class_missing", "class_count>=1"
        elif "det/seg gate" in warning:
            code, threshold = "det_seg_distribution", "seg_exclusion<=0.07;ratio_diff<=0.03"
        else:
            code, threshold = "quality_gate", "plan_v4.0"
        canonical = json.dumps(
            {"warning_code": code, "observed_value": warning, "threshold": threshold},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append({
            "warning_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "warning_code": code,
            "observed_value": warning,
            "threshold": threshold,
            "status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "reason": "",
        })
    return rows


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _report_row_key(filename: str, fields: list[str], row: dict[str, Any]) -> tuple[Any, ...]:
    full_row = tuple(str(row.get(field, "")) for field in fields)
    if filename in {
        "id_scan_report.csv",
        "selected_battery_ids_candidate.csv",
        "selected_battery_ids.csv",
        "test_battery_ids_candidate.csv",
        "test_battery_ids.csv",
    }:
        return (str(row.get("modality", "")), natural_id_key(str(row.get("battery_id", ""))), full_row)
    if filename == "ct_id_positive_rate_stratification.csv":
        scope = str(row.get("scope", ""))
        scope_rank = 0 if scope == "overall" else (2 if scope == "CT_test" else 1)
        bin_rank = list(CT_ID_POSITIVE_RATE_BIN_ORDER).index(str(row.get("positive_rate_bin", "")))
        return (scope_rank, scope, bin_rank, full_row)
    if filename in {"json_anomalies.csv", "class_name_variants.csv", "polygon_issues.csv"}:
        issue = row.get("issue", row.get("raw_class", ""))
        return (str(row.get("sample_id", "")), str(issue), full_row)
    if filename == "manifest.csv":
        return (str(row.get("sample_id", "")), full_row)
    if filename == "matching_issues.csv":
        return (str(row.get("image_path", "")).replace("\\", "/"), str(row.get("reason", "")), full_row)
    if filename == "corrupt_images.csv":
        return (str(row.get("path", "")).replace("\\", "/"), str(row.get("error", "")), full_row)
    if filename == "pixel_duplicates.csv":
        return (str(row.get("dup_group_id", "")), str(row.get("sample_id", "")), full_row)
    if filename == "split_id_leakage_audit.csv":
        return (str(row.get("left", "")), str(row.get("right", "")), full_row)
    return full_row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _selected_metrics(stats: IdStats) -> tuple[float, float]:
    metrics = sample_metrics(stats.selected_samples)
    return metrics.defect_image_ratio, metrics.annotations_per_image


def selected_id_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    rows = []
    for stats in selection.ids:
        ratio, annotations = _selected_metrics(stats)
        rows.append({
            "modality": stats.modality,
            "battery_id": stats.battery_id,
            "split_role": stats.split_role,
            "fold_id": stats.fold_id,
            "application": stats.application,
            "n_valid_images": len(stats.samples),
            "n_selected_images": len(stats.selected_samples),
            "x_count": sum(sample.axis == "x" for sample in stats.selected_samples),
            "y_count": sum(sample.axis == "y" for sample in stats.selected_samples),
            "z_count": sum(sample.axis == "z" for sample in stats.selected_samples),
            "defect_image_ratio": fmt_float(ratio),
            "annotations_per_image": fmt_float(annotations),
            "has_damaged": stats.has_damaged,
            "has_pollution": stats.has_pollution,
            "ct_id_positive_rate_bin": (
                ct_id_positive_rate_bin(ct_id_positive_rate(stats)) if stats.modality == "CT" else ""
            ),
        })
    return rows


def _id_scan_rows(scan: ScanResult) -> list[dict[str, Any]]:
    all_groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    valid_groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    valid_ids = {id(sample) for sample in scan.valid_samples}
    for sample in scan.samples:
        if sample.modality in {"CT", "EXT"} and sample.battery_id:
            all_groups[(sample.modality, sample.battery_id)].append(sample)
        if id(sample) in valid_ids:
            valid_groups[(sample.modality, sample.battery_id)].append(sample)
    rows = []
    for key in sorted(set(all_groups) | set(valid_groups), key=lambda value: (value[0], natural_id_key(value[1]))):
        group, valid = all_groups.get(key, []), valid_groups.get(key, [])
        applications = Counter(sample.application for sample in valid)
        application = min(applications, key=lambda value: (-applications[value], value)) if applications else ""
        eligible = [sample for sample in valid if sample.pre_split_eligible]
        metrics = sample_metrics(eligible)
        axis_counts = {axis: sum(sample.axis == axis for sample in eligible) for axis in "xyz"}
        rows.append({
            "modality": key[0], "battery_id": key[1], "application": application,
            "scanned_pairs": len(group), "valid_images": len(valid),
            "pre_split_eligible_images": len(eligible),
            "pre_split_excluded_images": len(valid) - len(eligible),
            "x_valid": axis_counts["x"], "y_valid": axis_counts["y"], "z_valid": axis_counts["z"],
            "ct_axis_set_valid": key[0] != "CT" or all(sample.axis in "xyz" for sample in eligible),
            "defect_image_ratio": fmt_float(metrics.defect_image_ratio), "annotations_per_image": fmt_float(metrics.annotations_per_image),
        })
    return rows


def _ct_axis_balance_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    groups = {
        role: [
            sample
            for stats in selection.ids
            if stats.modality == "CT" and stats.split_role == role
            for sample in stats.selected_samples
        ]
        for role in ("development", "test")
    }
    counts = {
        role: {axis: sum(sample.axis == axis for sample in samples) for axis in "xyz"}
        for role, samples in groups.items()
    }
    ratios = {}
    for role, axis_counts in counts.items():
        total = sum(axis_counts.values())
        ratios[role] = {axis: axis_counts[axis] / total if total else 0.0 for axis in "xyz"}
    gaps = {axis: abs(ratios["test"][axis] - ratios["development"][axis]) for axis in "xyz"}
    rows = []
    for role in ("development", "test"):
        total = sum(counts[role].values())
        rows.append({
            "split": role,
            "x_images": counts[role]["x"],
            "y_images": counts[role]["y"],
            "z_images": counts[role]["z"],
            "total_images": total,
            "x_ratio": fmt_float(ratios[role]["x"]),
            "y_ratio": fmt_float(ratios[role]["y"]),
            "z_ratio": fmt_float(ratios[role]["z"]),
            "x_gap": fmt_float(gaps["x"]),
            "y_gap": fmt_float(gaps["y"]),
            "z_gap": fmt_float(gaps["z"]),
            "axis_max_gap": fmt_float(max(gaps.values())),
            "axis_sum_gap": fmt_float(sum(gaps.values())),
        })
    return rows


def _manifest_rows(scan: ScanResult, selection: SelectionResult) -> list[dict[str, Any]]:
    valid_by_id: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in scan.valid_samples:
        if not sample.pre_split_eligible:
            continue
        valid_by_id[(sample.modality, sample.battery_id)].append(sample)
    ratios = {key: sample_metrics(group).defect_image_ratio for key, group in valid_by_id.items()}
    id_roles = {(stats.modality, stats.battery_id): (stats.split_role, stats.fold_id) for stats in selection.ids}
    rows = []
    for sample in scan.samples:
        suffix = ".jpg" if sample.modality == "CT" else (sample.image_path.suffix.lower() if sample.image_path else "")
        classes = set(sample.class_names)
        role, fold = id_roles.get((sample.modality, sample.battery_id), (sample.split_role, sample.fold_id))
        final_det = sample.included_det and sample.selected
        final_seg = sample.included_seg and sample.selected
        det_reason = sample.exclusion_reason_det
        seg_reason = sample.exclusion_reason_seg
        if not sample.pre_split_eligible:
            det_reason = sample.pre_split_exclusion_reason
            seg_reason = sample.pre_split_exclusion_reason
        elif sample.included_det and not sample.selected:
            det_reason = "id_sampling_cap" if (sample.modality, sample.battery_id) in id_roles else "rgb_id_not_selected"
        if sample.pre_split_eligible and sample.included_seg and not sample.selected:
            seg_reason = "id_sampling_cap" if (sample.modality, sample.battery_id) in id_roles else "rgb_id_not_selected"
        rows.append({
            "sample_id": sample.sample_id, "original_stem": sample.original_stem,
            "modality": sample.modality, "battery_id": sample.battery_id, "axis": sample.axis,
            "application": sample.application, "orig_image_path": str(sample.image_path or ""),
            "orig_json_path": str(sample.json_path or ""), "orig_json_relative_path": sample.json_relative_posix,
            "output_image_name": sample.sample_id + suffix, "output_label_stem": sample.sample_id,
            "image_w": sample.image_w, "image_h": sample.image_h, "roi_w": sample.roi_w or "", "roi_h": sample.roi_h or "",
            "original_is_normal": sample.original_is_normal, "is_normal_interpreted": sample.is_normal_interpreted,
            "has_damaged": "Damaged" in classes, "has_pollution": "Pollution" in classes, "has_porosity": "porosity" in classes,
            "defect_image_ratio_id": fmt_float(ratios.get((sample.modality, sample.battery_id), 0.0)),
            "pixel_hash": sample.pixel_hash, "json_sha256": sample.json_sha256,
            "dup_group_id": sample.duplicate_group_id, "dup_action": sample.duplicate_action,
            "selected_rgb_160": sample.selected_rgb_160, "split_role": role, "fold_id": fold,
            "included_det": final_det, "included_seg": final_seg,
            "exclusion_reason_det": det_reason, "exclusion_reason_seg": seg_reason,
            "original_defect_count": sample.original_defect_count,
            "yolo_det_instance_count": sample.yolo_det_instance_count,
            "yolo_seg_instance_count": sample.yolo_seg_instance_count,
            "multipart_split_count": sample.multipart_split_count, "min_extent_applied": sample.min_extent_applied,
            "porosity_polygon_count": sample.porosity_polygon_count if sample.modality == "CT" else "",
            "porosity_area_sum_ratio": fmt_float(sample.porosity_area_sum_ratio) if sample.modality == "CT" else "",
            "porosity_bbox_max_ratio": fmt_float(sample.porosity_bbox_max_ratio) if sample.modality == "CT" else "",
            "porosity_area_bin": porosity_area_bin(sample.porosity_area_sum_ratio) if sample.modality == "CT" else "",
            "ct_stratum": f"{sample.axis}|{porosity_area_bin(sample.porosity_area_sum_ratio)}" if sample.modality == "CT" else "",
            "pre_split_eligible": sample.pre_split_eligible,
            "pre_split_exclusion_reason": sample.pre_split_exclusion_reason,
        })
    return rows


def _ct_area_distribution_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    groups = group_selected_samples(selection.ids, split_fold_group_name)
    rows = []
    for split, samples in sorted(groups.items()):
        if not split.startswith("CT_"):
            continue
        counts = Counter((sample.axis, porosity_area_bin(sample.porosity_area_sum_ratio)) for sample in samples)
        total = len(samples)
        for axis in "xyz":
            for area_bin in AREA_BIN_ORDER:
                count = counts[(axis, area_bin)]
                if count:
                    rows.append({"split": split, "axis": axis, "area_bin": area_bin, "image_count": count, "within_split_share": fmt_float(count / total)})
    return rows


def _ct_split_balance_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    groups = group_selected_samples(selection.ids, split_fold_group_name)
    ct_groups = {name: samples for name, samples in groups.items() if name.startswith("CT_")}
    development_total = sum(len(samples) for name, samples in ct_groups.items() if name.startswith("CT_fold_"))
    rows = []
    for name, samples in sorted(ct_groups.items()):
        is_fold = name.startswith("CT_fold_")
        rows.append({
            "split": name,
            "id_count": len({sample.battery_id for sample in samples}),
            "image_count": len(samples),
            "target_image_ratio": fmt_float(0.2) if is_fold else "",
            "actual_image_ratio": fmt_float(len(samples) / development_total) if is_fold and development_total else "",
        })
    return rows


def _ct_positive_rate_stratification_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    """CT ID 양성률 구간별 후보/목표test/실제test/development 개수와 상태.

    overall 행은 test 실제 count가 quota와 같으면 PASS, 아니면 FAIL(하드 게이트).
    fold 행은 감사용이므로 INFO로만 기록한다.
    """
    ct_ids = [stats for stats in selection.ids if stats.modality == "CT"]
    quotas = selection.ct_test_positive_rate_quotas

    def bin_of(stats: IdStats) -> str:
        return ct_id_positive_rate_bin(ct_id_positive_rate(stats))

    candidate = Counter(bin_of(stats) for stats in ct_ids)
    test_ids = [stats for stats in ct_ids if stats.split_role == "test"]
    development_ids = [stats for stats in ct_ids if stats.split_role == "development"]
    test_counts = Counter(bin_of(stats) for stats in test_ids)
    development_counts = Counter(bin_of(stats) for stats in development_ids)

    rows: list[dict[str, Any]] = []
    for name in CT_ID_POSITIVE_RATE_BIN_ORDER:
        target = quotas.get(name, 0)
        actual = test_counts.get(name, 0)
        rows.append({
            "scope": "overall",
            "positive_rate_bin": name,
            "candidate_id_count": candidate.get(name, 0),
            "target_test_id_count": target,
            "actual_test_id_count": actual,
            "development_id_count": development_counts.get(name, 0),
            "status": "PASS" if actual == target else "FAIL",
        })

    fold_ids: dict[str, list[IdStats]] = defaultdict(list)
    for stats in development_ids:
        fold_ids[stats.fold_id].append(stats)
    for fold_id in sorted(fold_ids):
        fold_counts = Counter(bin_of(stats) for stats in fold_ids[fold_id])
        for name in CT_ID_POSITIVE_RATE_BIN_ORDER:
            rows.append({
                "scope": f"CT_fold_{fold_id}",
                "positive_rate_bin": name,
                "candidate_id_count": candidate.get(name, 0),
                "target_test_id_count": "",
                "actual_test_id_count": "",
                "development_id_count": fold_counts.get(name, 0),
                "status": "INFO",
            })

    for name in CT_ID_POSITIVE_RATE_BIN_ORDER:
        rows.append({
            "scope": "CT_test",
            "positive_rate_bin": name,
            "candidate_id_count": candidate.get(name, 0),
            "target_test_id_count": quotas.get(name, 0),
            "actual_test_id_count": test_counts.get(name, 0),
            "development_id_count": "",
            "status": "INFO",
        })
    return rows


def _ct_large_area_review_rows(scan: ScanResult) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "battery_id": sample.battery_id,
            "axis": sample.axis,
            "porosity_polygon_count": sample.porosity_polygon_count,
            "porosity_area_sum_ratio": fmt_float(sample.porosity_area_sum_ratio),
            "area_bin": porosity_area_bin(sample.porosity_area_sum_ratio),
            "orig_image_path": str(sample.image_path or ""),
            "orig_json_path": str(sample.json_path or ""),
        }
        for sample in scan.valid_samples
        if sample.modality == "CT" and sample.porosity_area_sum_ratio >= 0.40
    ]


def _ct_bbox_exclusion_rows(scan: ScanResult) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "battery_id": sample.battery_id,
            "axis": sample.axis,
            "porosity_polygon_count": sample.porosity_polygon_count,
            "porosity_area_sum_ratio": fmt_float(sample.porosity_area_sum_ratio),
            "porosity_bbox_max_ratio": fmt_float(sample.porosity_bbox_max_ratio),
            "area_bin": porosity_area_bin(sample.porosity_area_sum_ratio),
            "orig_image_path": str(sample.image_path or ""),
            "orig_json_path": str(sample.json_path or ""),
            "reason": sample.pre_split_exclusion_reason,
        }
        for sample in scan.valid_samples
        if not sample.pre_split_eligible
    ]


def _classes_for(modality: str) -> list[tuple[int, str]]:
    return [(0, "porosity")] if modality == "CT" else [(0, "Damaged"), (1, "Pollution")]


def _line_class_count(samples: list[Sample], task: str, class_id: int) -> int:
    lines = (line for sample in samples for line in (sample.det_lines if task == "det" else sample.seg_lines))
    return sum(line.split(maxsplit=1)[0] == str(class_id) for line in lines)


def quality_warnings(scan: ScanResult, selection: SelectionResult) -> tuple[list[str], list[str]]:
    """(gate, review) 두 리스트를 반환한다.

    gate = §17.2 하드 품질 게이트(밀도 ±20%, 필수 클래스 부재, det/seg 게이트). approve/execute를 하드 블록한다.
    review = §17.3 검토 경고(면적 구간 쏠림, fold 이미지 비율 5%p, Test 100장 미만, Test 제외율 20% 이상).
             `review_warnings.csv`에 감사 기록으로 남지만 approve/execute를 자동 차단하지 않는다.
    계획서 §17.2/§17.3 분류를 코드로 그대로 반영한다. 어느 경고도 숨기지 않으며 누수 게이트는 건드리지 않는다(§22.8).
    """
    gate = list(selection.warnings)
    review = list(selection.review_warnings)
    leaked_large = [
        sample.sample_id
        for sample in selection.selected_samples
        if sample.modality == "CT"
        and sample.porosity_bbox_max_ratio >= CT_PRE_SPLIT_AREA_THRESHOLD
    ]
    if leaked_large:
        raise RuntimeError(
            f"structural CT bbox policy gate failed: {len(leaked_large)} selected image(s) >=25%"
        )
    original_large = [
        sample
        for sample in scan.samples
        if sample.included_det
        and sample.modality == "CT"
        and sample.porosity_area_sum_ratio >= 0.40
    ]
    if original_large:
        review.append(
            f"CT original population: {len(original_large)} valid image(s) have porosity area >=40%; "
            "see ct_large_area_review.csv"
        )
    split_population = [sample for sample in scan.valid_samples if sample.pre_split_eligible]
    applications: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in split_population:
        if sample.application:
            applications[(sample.modality, sample.battery_id)].add(sample.application)
    for (modality, battery_id), values in sorted(applications.items()):
        if len(values) > 1:
            review.append(
                f"mixed application within ID {modality}/{battery_id}: {','.join(sorted(values))}"
            )
    eligible_ids = {sample.sample_id for sample in split_population}
    repaired = sum(
        row.get("issue") == "polygon_repaired_or_clipped" and row.get("sample_id") in eligible_ids
        for row in scan.polygon_issues
    )
    multipart = sum(sample.multipart_split_count for sample in split_population)
    min_extent = sum(sample.min_extent_applied for sample in split_population)
    if repaired or multipart or min_extent:
        review.append(
            f"geometry review totals: repaired_or_clipped={repaired}, "
            f"multipart_extra_pieces={multipart}, min_extent_images={min_extent}"
        )
    groups = group_selected_samples(selection.ids, split_fold_group_name)
    all_selected = selection.selected_samples
    groups.update({
        "CT_overall": [sample for sample in all_selected if sample.modality == "CT"],
        "EXT_overall": [sample for sample in all_selected if sample.modality == "EXT"],
    })
    for stats in selection.ids:
        if stats.split_role != "test":
            continue
        if len(stats.samples) < 100:
            review.append(f"Test {stats.modality}/{stats.battery_id}: valid images {len(stats.samples)} < 100")
        denominator = scan.raw_image_counts.get((stats.modality, stats.battery_id), 0)
        exclusion = 1 - len(stats.samples) / denominator if denominator else 1.0
        if exclusion >= 0.20 - 1e-12:
            review.append(f"Test {stats.modality}/{stats.battery_id}: exclusion={exclusion:.8f} (raw={denominator}, valid={len(stats.samples)})")
    for name, samples in groups.items():
        modality = "CT" if name.startswith("CT") else "EXT"
        role_requires_class = name == "CT_test" or name.startswith("CT_fold_") or name in {"EXT_val", "EXT_test"}
        if role_requires_class:
            for class_id, class_name in _classes_for(modality):
                for task in ("det", "seg"):
                    if _line_class_count(samples, task, class_id) == 0:
                        gate.append(f"{name}:{task}: required class missing: {class_name}")
        det_samples = [sample for sample in samples if sample.included_det]
        seg_samples = [sample for sample in samples if sample.included_seg]
        exclusion_rate = 1 - len(seg_samples) / len(det_samples) if det_samples else 0.0
        det_ratio = sample_metrics(det_samples).defect_image_ratio
        seg_ratio = sample_metrics(seg_samples).defect_image_ratio
        if exclusion_rate > 0.07 + 1e-12 or abs(det_ratio - seg_ratio) > 0.03 + 1e-12:
            gate.append(f"{name}: det/seg gate exclusion={exclusion_rate:.8f}, ratio_diff={abs(det_ratio-seg_ratio):.8f}")
    return sorted(set(gate)), sorted(set(review))


def leakage_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    groups = group_selected_samples(selection.ids, split_fold_group_name)
    rows = []
    names = sorted(groups)
    for index, left in enumerate(names):
        left_ids = {(sample.modality, sample.battery_id) for sample in groups[left]}
        left_hashes = {sample.pixel_hash for sample in groups[left] if sample.pixel_hash}
        for right in names[index + 1 :]:
            if left.split("_")[0] != right.split("_")[0]:
                continue
            right_ids = {(sample.modality, sample.battery_id) for sample in groups[right]}
            right_hashes = {sample.pixel_hash for sample in groups[right] if sample.pixel_hash}
            id_overlap = len(left_ids & right_ids)
            hash_overlap = len(left_hashes & right_hashes)
            rows.append({"left": left, "right": right, "id_overlap": id_overlap, "pixel_hash_overlap": hash_overlap, "status": "PASS" if id_overlap == 0 and hash_overlap == 0 else "FAIL"})
    return rows


def _balance_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    rows = []
    for dataset, samples in sorted(group_selected_samples(selection.ids, report_dataset_group_name).items()):
        modality = "CT" if dataset.startswith("CT") else "EXT"
        det_samples = [sample for sample in samples if sample.included_det]
        metrics = sample_metrics(det_samples)
        rows.append({
            "row_type": "dataset", "dataset": dataset, "class_name": "ALL",
            "battery_ids": len({sample.battery_id for sample in samples}), "images": metrics.images,
            "annotations": metrics.annotations,
            "defect_image_ratio": fmt_float(metrics.defect_image_ratio),
            "annotations_per_image": fmt_float(metrics.annotations_per_image),
            "normal_ratio": fmt_float(metrics.normal_ratio),
        })
        for class_id, class_name in _classes_for(modality):
            class_annotations = _line_class_count(det_samples, "det", class_id)
            class_images = sum(any(line.startswith(str(class_id) + " ") for line in sample.det_lines) for sample in det_samples)
            rows.append({"row_type": "class", "dataset": dataset, "class_name": class_name, "battery_ids": "", "images": class_images, "annotations": class_annotations, "defect_image_ratio": "", "annotations_per_image": "", "normal_ratio": ""})
    return rows


def _comparison_rows(selection: SelectionResult) -> list[dict[str, Any]]:
    rows = []
    for dataset, samples in sorted(group_selected_samples(selection.ids, report_dataset_group_name).items()):
        det = [sample for sample in samples if sample.included_det]
        seg = [sample for sample in samples if sample.included_seg]
        rows.append({
            "dataset": dataset, "detection_images": len(det), "segmentation_images": len(seg), "difference": len(det) - len(seg),
            "det_defect_ratio": fmt_float(sample_metrics(det).defect_image_ratio),
            "seg_defect_ratio": fmt_float(sample_metrics(seg).defect_image_ratio),
        })
    return rows


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _eda_markdown(scan: ScanResult, selection: SelectionResult, warnings: list[str], manifest_rows: list[dict[str, Any]]) -> str:
    report_groups = group_selected_samples(selection.ids, report_dataset_group_name)
    report_groups.update({
        name: samples
        for name, samples in group_selected_samples(selection.ids, split_fold_group_name).items()
        if name.startswith("CT_fold_")
    })
    summary_rows = []
    bbox_rows = []
    for dataset, samples in sorted(report_groups.items()):
        det = [sample for sample in samples if sample.included_det]
        seg = [sample for sample in samples if sample.included_seg]
        metrics = sample_metrics(det)
        widths, heights, aspects = [], [], []
        for sample in det:
            for line in sample.det_lines:
                fields = line.split()
                if len(fields) == 5:
                    width, height = float(fields[3]), float(fields[4])
                    widths.append(width)
                    heights.append(height)
                    aspects.append(width / height if height else 0.0)
        summary_rows.append([
            dataset,
            len({sample.battery_id for sample in samples}),
            len(det),
            len(seg),
            fmt_float(metrics.defect_image_ratio),
            fmt_float(metrics.annotations_per_image),
            fmt_float(metrics.normal_ratio),
            f"{sum(sample.axis == 'x' for sample in samples)}/{sum(sample.axis == 'y' for sample in samples)}/{sum(sample.axis == 'z' for sample in samples)}",
        ])
        bbox_rows.append([
            dataset, len(widths), fmt_float(median(widths) if widths else 0.0),
            fmt_float(median(heights) if heights else 0.0),
            fmt_float(_percentile(aspects, 0.05)), fmt_float(median(aspects) if aspects else 0.0), fmt_float(_percentile(aspects, 0.95)),
        ])
    density_rows = []
    gate_groups = group_selected_samples(selection.ids, split_fold_group_name)
    ct_dev = [stats for stats in selection.ids if stats.modality == "CT" and stats.split_role == "development"]
    ct_metrics = sample_metrics(collect_selected_samples(ct_dev))
    ct_target = ct_metrics.annotations_per_image
    ct_defect_target = ct_metrics.defect_image_ratio
    ext_trainval = [stats for stats in selection.ids if stats.modality == "EXT" and stats.split_role in {"train", "val"}]
    ext_metrics = sample_metrics(collect_selected_samples(ext_trainval))
    ext_target = ext_metrics.annotations_per_image
    ext_defect_target = ext_metrics.defect_image_ratio
    for name, samples in sorted(gate_groups.items()):
        target = ct_target if name.startswith("CT_fold_") else ext_target if name in {"EXT_train", "EXT_val"} else 0.0
        achieved_metrics = sample_metrics(samples)
        achieved = achieved_metrics.annotations_per_image
        defect_target = ct_defect_target if name.startswith("CT_fold_") else ext_defect_target if name in {"EXT_train", "EXT_val"} else 0.0
        defect_achieved = achieved_metrics.defect_image_ratio
        deviation = abs(achieved / target - 1) if target else 0.0
        density_rows.append([name, fmt_float(defect_target), fmt_float(defect_achieved), fmt_float(target), fmt_float(achieved), fmt_float(deviation), "PASS" if target == 0 or deviation <= 0.20 + 1e-12 or name.endswith("test") else "FAIL"])
    anomaly_counts = Counter(row["issue"] for row in scan.json_anomalies)
    polygon_counts = Counter(row["issue"] for row in scan.polygon_issues)
    discarded = Counter(row["exclusion_reason_det"] for row in manifest_rows if not row["included_det"])
    applications = Counter((stats.split_role, stats.application) for stats in selection.ids if stats.modality == "EXT")
    application_rows = [[role, application, count] for (role, application), count in sorted(applications.items())]
    processing_rows = [
        ["pixel duplicate rows", len(scan.pixel_duplicates)],
        ["JSON anomaly rows", len(scan.json_anomalies)],
        ["polygon issue rows", len(scan.polygon_issues)],
        ["300-image cap discards", discarded.get("id_sampling_cap", 0)],
        ["original defects (selected)", sum(sample.original_defect_count for sample in selection.selected_samples)],
        ["YOLO det instances", sample_metrics(selection.selected_samples).annotations],
        ["YOLO seg instances", sum(len(sample.seg_lines) for sample in selection.selected_samples)],
        ["min-extent images", sum(sample.min_extent_applied for sample in selection.selected_samples)],
    ]
    issue_rows = [[f"json:{issue}", count] for issue, count in sorted(anomaly_counts.items())]
    issue_rows += [[f"polygon:{issue}", count] for issue, count in sorted(polygon_counts.items())]

    # CT ID 양성률 층화: image-micro(전체 이미지 기준)와 ID-macro(ID별 평균)를 분리해 보인다.
    ct_ids = [stats for stats in selection.ids if stats.modality == "CT"]
    ct_test_ids = [stats for stats in ct_ids if stats.split_role == "test"]
    ct_dev_ids = [stats for stats in ct_ids if stats.split_role == "development"]
    pr_rows = []
    quotas = selection.ct_test_positive_rate_quotas
    for name in CT_ID_POSITIVE_RATE_BIN_ORDER:
        cand = sum(1 for stats in ct_ids if ct_id_positive_rate_bin(ct_id_positive_rate(stats)) == name)
        tgt = quotas.get(name, 0)
        act = sum(1 for stats in ct_test_ids if ct_id_positive_rate_bin(ct_id_positive_rate(stats)) == name)
        dev = sum(1 for stats in ct_dev_ids if ct_id_positive_rate_bin(ct_id_positive_rate(stats)) == name)
        pr_rows.append([name, cand, tgt, act, dev])

    def _macro(ids: list[IdStats]) -> tuple[float, float, float, float, float]:
        if not ids:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        rates = sorted(ct_id_positive_rate(stats) for stats in ids)
        mean = sum(rates) / len(rates)
        zero_share = sum(1 for value in rates if value <= 1e-12) / len(rates)
        return (mean, median(rates), rates[0], rates[-1], zero_share)

    dev_macro = _macro(ct_dev_ids)
    test_macro = _macro(ct_test_ids)
    dev_micro = sample_metrics(collect_selected_samples(ct_dev_ids)).defect_image_ratio
    test_micro = sample_metrics(collect_selected_samples(ct_test_ids)).defect_image_ratio
    macro_rows = [
        ["development", fmt_float(dev_micro), fmt_float(dev_macro[0]), fmt_float(dev_macro[1]), fmt_float(dev_macro[2]), fmt_float(dev_macro[3]), fmt_float(dev_macro[4])],
        ["test", fmt_float(test_micro), fmt_float(test_macro[0]), fmt_float(test_macro[1]), fmt_float(test_macro[2]), fmt_float(test_macro[3]), fmt_float(test_macro[4])],
    ]
    return (
        "# v4.0 post-crop EDA (dry-run estimate)\n\n"
        "`execute` 전에는 CT crop 결과를 쓰지 않으므로 픽셀 통계가 아닌 확정 ROI/YOLO 변환 결과를 집계한다.\n\n"
        "## Dataset summary\n\n"
        + _markdown_table(["dataset", "IDs", "det images", "seg images", "defect ratio", "ann/image", "normal ratio", "x/y/z"], summary_rows)
        + "\n\n## Annotation density guard\n\n"
        + _markdown_table(["group", "defect target", "defect achieved", "ann target", "ann achieved", "ann relative deviation", "status"], density_rows)
        + "\n\n## Normalized bbox and aspect ratio\n\n"
        + _markdown_table(["dataset", "boxes", "median w", "median h", "aspect p05", "aspect median", "aspect p95"], bbox_rows)
        + "\n\n## EXT application ID counts\n\n"
        + _markdown_table(["split", "application", "IDs"], application_rows)
        + "\n\n## Processing totals\n\n"
        + _markdown_table(["metric", "count"], processing_rows)
        + "\n\n## Issue counts\n\n"
        + (_markdown_table(["issue", "count"], issue_rows) if issue_rows else "No JSON or polygon issues.")
        + "\n\n## CT ID positive-rate stratification\n\n"
        + "구간 경계는 하한 포함·상한 미포함이다: zero=0, very_low=(0,0.01), low_mid=[0.01,0.30), "
        + "mid_high=[0.30,0.60), very_high=[0.60,1.00]. 분모는 pre-split 및 CT ID gate 이후 이미지 수다.\n\n"
        + _markdown_table(["positive_rate_bin", "candidate IDs", "target test IDs", "actual test IDs", "development IDs"], pr_rows)
        + "\n\nimage-micro(전체 이미지 기준 양성률)와 ID-macro(ID별 양성률 평균)는 서로 다른 지표다.\n\n"
        + _markdown_table(["scope", "image-micro positive", "ID-macro mean", "ID-macro median", "ID-macro min", "ID-macro max", "완전음성 ID 비율"], macro_rows)
        + f"\n\n## Quality gate warnings\n\n{len(warnings)} warning(s). See `dryrun_warnings.csv`.\n"
    )


def write_reports(scan: ScanResult, selection: SelectionResult, report_dir: Path) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_warnings, review_warnings = quality_warnings(scan, selection)
    # §17.3 검토 경고는 scan 단계 review 경고와 합쳐 acknowledge 가능한 review_warnings.csv로 내보낸다.
    review_rows = list(scan.review_warnings) + [
        {"warning": warning, "reviewer": "", "reviewed_at": "", "status": "pending"}
        for warning in review_warnings
    ]
    exception_rows = quality_exception_rows(gate_warnings)
    manifest_rows = _manifest_rows(scan, selection)
    bbox_exclusion_rows = _ct_bbox_exclusion_rows(scan)
    area_distribution_rows = _ct_area_distribution_rows(selection)
    valid_ct = [sample for sample in scan.valid_samples if sample.modality == "CT"]
    eligible_ct = [sample for sample in valid_ct if sample.pre_split_eligible]
    excluded_ct = [sample for sample in valid_ct if not sample.pre_split_eligible]
    if len(valid_ct) != len(eligible_ct) + len(excluded_ct):
        raise RuntimeError("structural report gate failed: CT valid != eligible + excluded")
    manifest_excluded_ids = {
        row["sample_id"]
        for row in manifest_rows
        if row["modality"] == "CT" and not row["pre_split_eligible"]
    }
    bbox_excluded_ids = {row["sample_id"] for row in bbox_exclusion_rows}
    if manifest_excluded_ids != bbox_excluded_ids:
        raise RuntimeError("structural report gate failed: manifest and ct_bbox_exclusions differ")
    manifest_selected_ct = sum(
        row["modality"] == "CT" and row["included_det"] for row in manifest_rows
    )
    distributed_ct = sum(int(row["image_count"]) for row in area_distribution_rows)
    if manifest_selected_ct != distributed_ct:
        raise RuntimeError("structural report gate failed: manifest and CT distribution counts differ")
    discarded = Counter(row["exclusion_reason_det"] for row in manifest_rows if not row["included_det"])
    selected_rows = selected_id_rows(selection)
    test_rows = [{"modality": row["modality"], "battery_id": row["battery_id"]} for row in selected_rows if row["split_role"] == "test"]
    rows_by_file: dict[str, Iterable[dict[str, Any]]] = {
        "id_scan_report.csv": _id_scan_rows(scan),
        "split_axis_balance.csv": _ct_axis_balance_rows(selection),
        "selected_battery_ids_candidate.csv": selected_rows,
        "test_battery_ids_candidate.csv": test_rows,
        "matching_issues.csv": scan.matching_issues,
        "corrupt_images.csv": scan.corrupt_images,
        "pixel_duplicates.csv": scan.pixel_duplicates,
        "json_anomalies.csv": scan.json_anomalies,
        "class_name_variants.csv": scan.class_name_variants,
        "polygon_issues.csv": scan.polygon_issues,
        "discarded_stats.csv": [{"reason": reason, "count": count} for reason, count in sorted(discarded.items())],
        "dryrun_warnings.csv": [{"guard": warning} for warning in gate_warnings],
        "quality_exceptions.csv": exception_rows,
        "review_warnings.csv": review_rows,
        "class_balance_report.csv": _balance_rows(selection),
        "task_dataset_comparison.csv": _comparison_rows(selection),
        "split_id_leakage_audit.csv": leakage_rows(selection),
        "ct_split_area_distribution.csv": area_distribution_rows,
        "ct_split_balance_summary.csv": _ct_split_balance_rows(selection),
        "ct_large_area_review.csv": _ct_large_area_review_rows(scan),
        "ct_id_exclusions.csv": list(selection.ct_id_gate_rows),
        "ct_id_positive_rate_stratification.csv": _ct_positive_rate_stratification_rows(selection),
        "ct_bbox_exclusions.csv": bbox_exclusion_rows,
        "manifest.csv": manifest_rows,
    }
    for filename, fields in CSV_SCHEMAS.items():
        if filename in {"selected_battery_ids.csv", "test_battery_ids.csv"}:
            continue
        rows = list(rows_by_file.get(filename, []))
        rows.sort(key=lambda row: _report_row_key(filename, fields, row))
        write_csv(report_dir / filename, fields, rows)
    modality_counts = Counter(sample.modality for sample in scan.samples)
    valid_counts = Counter(sample.modality for sample in scan.valid_samples)
    raw_ids = {modality: len({battery_id for current_modality, battery_id in scan.raw_image_counts if current_modality == modality}) for modality in ("CT", "EXT")}
    valid_ids = {modality: len({sample.battery_id for sample in scan.valid_samples if sample.modality == modality}) for modality in ("CT", "EXT")}
    (report_dir / "scan_summary.md").write_text(
        "# v4.0 dry-run scan summary\n\n"
        f"- Raw fingerprint: `{scan.raw_fingerprint}`\n"
        f"- Scanned matched samples: CT {modality_counts['CT']:,}, EXT {modality_counts['EXT']:,}\n"
        f"- Valid detection samples: CT {valid_counts['CT']:,}, EXT {valid_counts['EXT']:,}\n"
        f"- Parsed raw IDs: CT {raw_ids['CT']:,}, EXT {raw_ids['EXT']:,}\n"
        f"- Valid IDs: CT {valid_ids['CT']:,}, EXT {valid_ids['EXT']:,}\n"
        f"- CT IDs dropped by ID gates: {len(selection.ct_id_gate_rows):,}\n"
        f"- Selected IDs: CT {len({stats.battery_id for stats in selection.ids if stats.modality == 'CT'}):,}, EXT {len({stats.battery_id for stats in selection.ids if stats.modality == 'EXT'}):,}\n"
        f"- CT pre-split bbox exclusions (>=25%): {sum(not sample.pre_split_eligible for sample in scan.valid_samples if sample.modality == 'CT'):,}\n"
        f"- Matching issues: {len(scan.matching_issues):,}\n"
        f"- Corrupt images: {len(scan.corrupt_images):,}\n"
        f"- Pixel duplicate rows: {len(scan.pixel_duplicates):,}\n"
        f"- JSON anomalies: {len(scan.json_anomalies):,}\n"
        f"- Polygon issues: {len(scan.polygon_issues):,}\n"
        f"- Undefined class variants: {len(scan.class_name_variants):,}\n"
        f"- Quality warnings: {len(gate_warnings):,}\n"
        f"- Review warnings: {len(review_rows):,}\n",
        encoding="utf-8",
        newline="\n",
    )
    (report_dir / "ID_parse_rule.md").write_text(
        "# ID parse rule\n\n- CT: `CT_cell_<form>_<battery_id>_<axis>_<frame>`\n- EXT: `RGB_cell_<form>_<battery_id>_<frame>`\n- Leading zeros are removed. `image_info.id` is not used.\n",
        encoding="utf-8", newline="\n",
    )
    (report_dir / "eda_v3_postcrop.md").write_text(_eda_markdown(scan, selection, gate_warnings, manifest_rows), encoding="utf-8", newline="\n")
    (report_dir / "raw_fingerprint.sha256").write_text(scan.raw_fingerprint + "\n", encoding="ascii", newline="\n")
    return gate_warnings
