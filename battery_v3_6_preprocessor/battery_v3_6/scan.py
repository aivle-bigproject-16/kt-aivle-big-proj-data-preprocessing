from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .deterministic import sample_id_for
from .geometry import convert_defect, parse_points
from .models import Sample
from .parsing import (
    canonical_class,
    deduplicate_defects,
    find_json_fields,
    is_ignored_outline,
    normalize_application,
    parse_filename,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SOURCE_MARKERS = {"01.원천데이터", "원천데이터"}
LABEL_MARKERS = {"02.라벨링데이터", "라벨링데이터"}


@dataclass
class ScanResult:
    raw_root: Path
    samples: list[Sample] = field(default_factory=list)
    matching_issues: list[dict[str, Any]] = field(default_factory=list)
    corrupt_images: list[dict[str, Any]] = field(default_factory=list)
    json_anomalies: list[dict[str, Any]] = field(default_factory=list)
    class_name_variants: list[dict[str, Any]] = field(default_factory=list)
    polygon_issues: list[dict[str, Any]] = field(default_factory=list)
    pixel_duplicates: list[dict[str, Any]] = field(default_factory=list)
    review_warnings: list[dict[str, Any]] = field(default_factory=list)
    raw_fingerprint: str = ""
    raw_image_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def valid_samples(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.included_det and not sample.duplicate_action.startswith("exclude")]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha1(path: Path) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha1()
        digest.update(struct.pack(">II", rgb.width, rgb.height))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def decode_dimensions_and_pixel_sha1(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        rgb = image.convert("RGB")
        digest = hashlib.sha1()
        digest.update(struct.pack(">II", width, height))
        digest.update(rgb.tobytes())
        return width, height, digest.hexdigest()


def _relative_after_marker(path: Path, markers: set[str]) -> tuple[str, Path] | None:
    parts = path.parts
    split = next((part for part in parts if part.casefold() in {"training", "validation"}), "")
    for index, part in enumerate(parts):
        if part.casefold() in {marker.casefold() for marker in markers}:
            return split, Path(*parts[index + 1 :])
    return None


def _match_key(path: Path, markers: set[str]) -> tuple[str, str, str] | None:
    relative = _relative_after_marker(path, markers)
    if relative is None:
        return None
    split, _tail = relative
    # 이미지-JSON 페어링은 (구역, stem)만으로 한다. 실제 AI-Hub 데이터는 원천/라벨 트리의 중간
    # 폴더명이 비대칭(예: TS_..._images_1 vs TL_..._label)이라 상위경로를 키에 넣으면 교집합이 0이 되어
    # 모든 쌍이 미매칭 처리된다(실측 확인). stem은 modality 접두어(ct_/rgb_)를 포함해 (구역,stem) 조합이
    # 충돌 없이 유일하다(실측 276,170쌍 전부 1:1, 충돌 0). 중간 성분은 "" 로 비워 다운스트림 인덱스를 유지한다.
    return split.casefold(), "", path.stem.casefold()


def _relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _json_battery_id(data_info: dict[str, Any]) -> str | None:
    value = data_info.get("battery_ids", data_info.get("battery_id"))
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        text = str(value).strip()
        if text.isdigit():
            return str(int(text))
    return None


def _image_dimensions(image_info: dict[str, Any]) -> tuple[int | None, int | None]:
    width = image_info.get("width")
    height = image_info.get("height")
    if isinstance(width, int) and not isinstance(width, bool) and isinstance(height, int) and not isinstance(height, bool):
        return width, height
    return None, None


def _add_anomaly(result: ScanResult, sample: Sample, issue: str, detail: str = "") -> None:
    result.json_anomalies.append({"sample_id": sample.sample_id, "path": str(sample.json_path or ""), "issue": issue, "detail": detail})
    sample.issues.append((issue, detail))


def _label_signature(sample: Sample) -> str:
    items = []
    for defect in sample.defects:
        raw_name = defect.get("name")
        name = canonical_class(raw_name, sample.modality)
        if name:
            items.append([name, defect.get("points")])
    items.sort(key=lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    payload = [items, sample.original_is_normal, [sample.roi_w, sample.roi_h] if sample.modality == "CT" else None]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _process_pair(result: ScanResult, image_path: Path, json_path: Path) -> Sample:
    raw_root = result.raw_root
    image_rel = _relative_posix(raw_root, image_path)
    parsed = parse_filename(image_path.name)
    modality = parsed.modality if parsed else "UNKNOWN"
    sample = Sample(
        sample_id=sample_id_for(modality, image_path.stem, image_rel),
        modality=modality,
        battery_id=parsed.battery_id if parsed else "",
        axis=parsed.axis if parsed else "",
        original_stem=image_path.stem,
        image_path=image_path,
        json_path=json_path,
        image_relative_posix=image_rel,
        json_relative_posix=_relative_posix(raw_root, json_path),
        scanned_pair=True,
        raw_parsed_image=parsed is not None,
    )
    if parsed is None:
        source_relative = _relative_after_marker(image_path, SOURCE_MARKERS)
        result.matching_issues.append({
            "split": source_relative[0].casefold() if source_relative else "",
            "stem": image_path.stem,
            "image_path": str(image_path),
            "json_candidate_paths": str(json_path),
            "image_count": 1,
            "json_count": 1,
            "candidate_count": 1,
            "reason": "filename_parse_error",
        })
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "filename_parse_error"
        return sample
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object")
    except Exception as exc:
        _add_anomaly(result, sample, "json_parse_error", str(exc))
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "json_parse_error"
        return sample
    data_info, image_info, defects, is_normal = find_json_fields(payload)
    sample.original_is_normal = is_normal
    sample.application = normalize_application(data_info.get("application"))
    sample.original_defect_count = len(defects)
    sample.json_sha256 = sha256_file(json_path)
    json_id = _json_battery_id(data_info)
    if json_id != sample.battery_id:
        _add_anomaly(result, sample, "battery_id_mismatch", f"filename={sample.battery_id};json={json_id}")
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "battery_id_mismatch"
        return sample
    data_type = str(data_info.get("data_type", "")).strip().casefold()
    json_modality = "CT" if data_type == "ct" else "EXT" if data_type in {"rgb", "ext", "exterior"} else ""
    if json_modality and json_modality != sample.modality:
        _add_anomaly(result, sample, "modality_mismatch", f"filename={sample.modality};json={json_modality}")
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "modality_mismatch"
        return sample
    json_filename = image_info.get("file_name")
    if not isinstance(json_filename, str) or Path(json_filename).stem != image_path.stem:
        _add_anomaly(result, sample, "image_filename_mismatch", str(json_filename))
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "image_filename_mismatch"
        return sample
    try:
        sample.image_w, sample.image_h, sample.pixel_hash = decode_dimensions_and_pixel_sha1(image_path)
    except Exception as exc:
        result.corrupt_images.append({"path": str(image_path), "error": str(exc)})
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "corrupt_image"
        return sample
    expected_w, expected_h = _image_dimensions(image_info)
    if (expected_w, expected_h) != (sample.image_w, sample.image_h):
        _add_anomaly(result, sample, "image_size_mismatch", f"actual={sample.image_w}x{sample.image_h};json={expected_w}x{expected_h}")
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "image_size_mismatch"
        return sample
    if sample.modality == "CT":
        roi = data_info.get("roi")
        if roi is None:
            _add_anomaly(result, sample, "missing_ct_roi")
            sample.exclusion_reason_det = sample.exclusion_reason_seg = "missing_ct_roi"
            return sample
        valid_roi = (
            isinstance(roi, list) and len(roi) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in roi)
            and 0 < roi[0] <= sample.image_w and 0 < roi[1] <= sample.image_h
        )
        if not valid_roi:
            _add_anomaly(result, sample, "invalid_ct_roi", repr(roi))
            sample.exclusion_reason_det = sample.exclusion_reason_seg = "invalid_ct_roi"
            return sample
        sample.roi_w, sample.roi_h = roi
    else:
        sample.roi_w = sample.image_w
        sample.roi_h = sample.image_h
    deduped, removed = deduplicate_defects(defects)
    sample.defects = deduped
    if removed:
        _add_anomaly(result, sample, "duplicate_defect_in_array", str(removed))
    if not isinstance(is_normal, bool):
        _add_anomaly(result, sample, "invalid_is_normal_type", type(is_normal).__name__)
    undefined = []
    target_defects: list[tuple[int, str, dict[str, Any]]] = []
    for index, defect in enumerate(deduped):
        raw_class = defect.get("name")
        if is_ignored_outline(raw_class):
            continue
        canonical = canonical_class(raw_class, sample.modality)
        if canonical is None:
            undefined.append((index, raw_class))
            continue
        points, point_issue = parse_points(defect.get("points"))
        if point_issue:
            result.polygon_issues.append({"sample_id": sample.sample_id, "defect_index": index, "issue": point_issue})
            target_defects.append((index, canonical, defect))
            continue
        target_defects.append((index, canonical, defect))
    if undefined:
        for _, raw_class in undefined:
            result.class_name_variants.append({"sample_id": sample.sample_id, "raw_class": str(raw_class), "action": "exclude_image"})
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "undefined_class"
        return sample
    if is_normal is True and target_defects:
        _add_anomaly(result, sample, "is_normal_true_with_defect")
    if not target_defects:
        if is_normal is True:
            sample.is_normal_interpreted = True
            sample.included_det = sample.included_seg = True
            return sample
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "no_valid_target_and_not_explicit_normal"
        return sample
    all_det = True
    all_seg = True
    class_ids = {"porosity": 0, "Damaged": 0, "Pollution": 1}
    for index, canonical, defect in target_defects:
        try:
            converted = convert_defect(class_ids[canonical], defect.get("points"), sample.roi_w or 0, sample.roi_h or 0)
        except Exception as exc:
            result.polygon_issues.append({
                "sample_id": sample.sample_id,
                "defect_index": index,
                "issue": f"polygon_exception:{type(exc).__name__}",
            })
            all_det = False
            all_seg = False
            continue
        for issue in converted.issues:
            result.polygon_issues.append({"sample_id": sample.sample_id, "defect_index": index, "issue": issue})
        if converted.multipart_split_count:
            for piece_index in range(converted.multipart_split_count + 1):
                result.polygon_issues.append({
                    "sample_id": sample.sample_id,
                    "defect_index": index,
                    "issue": f"multipart_piece:{piece_index}",
                })
        all_det = all_det and converted.det_valid
        all_seg = all_seg and converted.seg_valid
        sample.det_lines.extend(converted.det_lines)
        sample.seg_lines.extend(converted.seg_lines)
        sample.class_names.extend([canonical] * len(converted.det_lines))
        sample.multipart_split_count += converted.multipart_split_count
        sample.min_extent_applied = sample.min_extent_applied or converted.min_extent_applied
        if sample.modality == "CT" and canonical == "porosity":
            sample.porosity_polygon_count += converted.polygon_count
            sample.porosity_area_sum_ratio += converted.polygon_area_ratio
    sample.included_det = bool(target_defects) and all_det
    sample.included_seg = bool(target_defects) and all_seg
    if not sample.included_det:
        sample.det_lines.clear()
        sample.class_names.clear()
        sample.exclusion_reason_det = "invalid_target_polygon"
    if not sample.included_seg:
        sample.seg_lines.clear()
        sample.exclusion_reason_seg = "segmentation_polygon_repair_failed"
    sample.yolo_det_instance_count = len(sample.det_lines)
    sample.yolo_seg_instance_count = len(sample.seg_lines)
    return sample


def _unmatched_sample(result: ScanResult, image_path: Path, reason: str) -> Sample:
    image_rel = _relative_posix(result.raw_root, image_path)
    parsed = parse_filename(image_path.name)
    modality = parsed.modality if parsed else "UNKNOWN"
    sample = Sample(
        sample_id=sample_id_for(modality, image_path.stem, image_rel),
        modality=modality,
        battery_id=parsed.battery_id if parsed else "",
        axis=parsed.axis if parsed else "",
        original_stem=image_path.stem,
        image_path=image_path,
        image_relative_posix=image_rel,
        raw_parsed_image=parsed is not None,
        exclusion_reason_det=reason,
        exclusion_reason_seg=reason,
    )
    try:
        sample.image_w, sample.image_h, sample.pixel_hash = decode_dimensions_and_pixel_sha1(image_path)
    except Exception as exc:
        result.corrupt_images.append({"path": str(image_path), "error": str(exc)})
        sample.exclusion_reason_det = sample.exclusion_reason_seg = "corrupt_image"
    return sample


def _apply_pixel_duplicate_rules(result: ScanResult) -> None:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in result.samples:
        if sample.pixel_hash and sample.included_det:
            groups[sample.pixel_hash].append(sample)
    duplicate_groups = [(hash_value, samples) for hash_value, samples in groups.items() if len(samples) > 1]
    duplicate_groups.sort(key=lambda pair: pair[0])
    cross_id_groups = 0
    cross_id_samples = 0
    for number, (hash_value, samples) in enumerate(duplicate_groups, start=1):
        group_id = f"dup_{number:06d}"
        samples.sort(key=lambda s: (s.original_stem.casefold(), str(s.image_path).casefold()))
        battery_ids = {sample.battery_id for sample in samples}
        if len(battery_ids) > 1:
            cross_id_groups += 1
            cross_id_samples += len(samples)
            for sample in samples:
                sample.duplicate_group_id = group_id
                sample.duplicate_action = "exclude_cross_id_group"
                sample.included_det = sample.included_seg = False
                sample.exclusion_reason_det = sample.exclusion_reason_seg = "cross_id_pixel_duplicate"
        else:
            signatures = {_label_signature(sample) for sample in samples}
            if len(signatures) > 1:
                for sample in samples:
                    sample.duplicate_group_id = group_id
                    sample.duplicate_action = "exclude_same_id_label_conflict"
                    sample.included_det = sample.included_seg = False
                    sample.exclusion_reason_det = sample.exclusion_reason_seg = "exclude_same_id_label_conflict"
                result.review_warnings.append({"warning": f"same-ID pixel label conflict: {group_id}", "reviewer": "", "reviewed_at": "", "status": "pending"})
            else:
                for index, sample in enumerate(samples):
                    sample.duplicate_group_id = group_id
                    sample.duplicate_action = "keep_representative" if index == 0 else "exclude_same_id_duplicate"
                    if index:
                        sample.included_det = sample.included_seg = False
                        sample.exclusion_reason_det = sample.exclusion_reason_seg = "same_id_pixel_duplicate"
        for sample in samples:
            result.pixel_duplicates.append({
                "dup_group_id": group_id,
                "pixel_hash": hash_value,
                "modality": sample.modality,
                "battery_id": sample.battery_id,
                "sample_id": sample.sample_id,
                "action": sample.duplicate_action,
            })
    denominator = sum(sample.raw_parsed_image for sample in result.samples)
    if cross_id_groups > 50 or (denominator and cross_id_samples / denominator > 0.01):
        result.review_warnings.append({"warning": f"cross-ID pixel duplicates: groups={cross_id_groups}, samples={cross_id_samples}", "reviewer": "", "reviewed_at": "", "status": "pending"})


ISSUE_KEYS = ("matching_issues", "corrupt_images", "json_anomalies", "polygon_issues", "class_name_variants")


def _scan_pair_worker(payload: tuple[Path, Path, Path]) -> tuple[Sample, dict[str, list[dict[str, Any]]]]:
    """1:1 이미지-JSON 쌍을 독립 처리한다(멀티프로세스 워커).

    공유 상태를 만지지 않고 로컬 ScanResult에 이슈를 담아 반환한다. 메인이 결정론적으로 병합한다.
    반환 값·부작용은 직렬 `_process_pair`와 동일하다.
    """
    raw_root, image_path, json_path = payload
    local = ScanResult(raw_root=raw_root)
    sample = _process_pair(local, image_path, json_path)
    return sample, {key: getattr(local, key) for key in ISSUE_KEYS}


def _fingerprint_worker(payload: tuple[Path, Path]) -> tuple[str, int, str]:
    raw_root, path = payload
    return _relative_posix(raw_root, path), path.stat().st_size, sha256_file(path)


def compute_raw_fingerprint(raw_root: Path, paths: list[Path], jobs: int = 1) -> str:
    if jobs and jobs > 1 and paths:
        chunk = max(1, len(paths) // (jobs * 8))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            triples = list(executor.map(_fingerprint_worker, [(raw_root, path) for path in paths], chunksize=chunk))
    else:
        triples = [(_relative_posix(raw_root, path), path.stat().st_size, sha256_file(path)) for path in paths]
    records = [f"{relative}\t{size}\t{digest}" for relative, size, digest in sorted(triples, key=lambda item: item[0])]
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def scan_dataset(raw_root: Path, jobs: int = 1) -> ScanResult:
    raw_root = raw_root.resolve()
    result = ScanResult(raw_root=raw_root)
    # os.walk로 전수 열거(pathlib rglob+is_file 대비 ~수백배 빠름). filenames는 이미 파일만 반환하므로
    # per-file is_file() stat이 불필요하다. 파일 집합·정렬 결과는 rglob 방식과 동일 → 산출물/fingerprint 불변.
    all_files = [Path(dirpath) / name for dirpath, _dirs, filenames in os.walk(raw_root) for name in filenames]
    all_files.sort(key=lambda p: _relative_posix(raw_root, p))
    images = [path for path in all_files if path.suffix.casefold() in IMAGE_SUFFIXES]
    jsons = [path for path in all_files if path.suffix.casefold() == ".json"]
    image_map: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    json_map: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    loose_json_stems: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in images:
        parsed = parse_filename(path.name)
        if parsed:
            key_count = (parsed.modality, parsed.battery_id)
            result.raw_image_counts[key_count] = result.raw_image_counts.get(key_count, 0) + 1
        key = _match_key(path, SOURCE_MARKERS)
        if key:
            image_map[key].append(path)
    for path in jsons:
        key = _match_key(path, LABEL_MARKERS)
        if key:
            json_map[key].append(path)
            loose_json_stems[(key[0], key[2])].append(path)
    single_pairs: list[tuple[Path, Path, Path]] = []
    for key in sorted(set(image_map) | set(json_map)):
        image_paths = image_map.get(key, [])
        json_paths = json_map.get(key, [])
        if len(image_paths) == 1 and len(json_paths) == 1:
            single_pairs.append((raw_root, image_paths[0], json_paths[0]))
            continue
        candidates = loose_json_stems.get((key[0], key[2]), []) if image_paths and not json_paths else json_paths
        reason = "cross_folder_stem_candidate" if image_paths and not json_paths and candidates else "matching_cardinality"
        result.matching_issues.append({
            "split": key[0],
            "stem": key[2],
            "image_path": "|".join(str(p) for p in image_paths),
            "json_candidate_paths": "|".join(str(p) for p in candidates),
            "image_count": len(image_paths),
            "json_count": len(json_paths),
            "candidate_count": len(candidates),
            "reason": reason,
        })
        result.samples.extend(_unmatched_sample(result, path, reason) for path in image_paths)
    # 1:1 쌍 처리가 스캔 비용의 대부분(이미지 디코드+픽셀해시). jobs>1이면 프로세스 분산.
    # 결과 순서는 이후 samples.sort / reports rows.sort / fingerprint sort가 정규화하므로 병렬/직렬 산출이 동일하다.
    if jobs and jobs > 1 and single_pairs:
        chunk = max(1, len(single_pairs) // (jobs * 8))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for sample, issues in executor.map(_scan_pair_worker, single_pairs, chunksize=chunk):
                result.samples.append(sample)
                for issue_key in ISSUE_KEYS:
                    getattr(result, issue_key).extend(issues[issue_key])
    else:
        for _root, image_path, json_path in single_pairs:
            result.samples.append(_process_pair(result, image_path, json_path))
    _apply_pixel_duplicate_rules(result)
    result.samples.sort(key=lambda sample: sample.sample_id)
    for sample in result.samples:
        sample.defects = []  # duplicate analysis is complete; release raw annotation objects
    result.raw_fingerprint = compute_raw_fingerprint(raw_root, images + jsons, jobs)
    return result
