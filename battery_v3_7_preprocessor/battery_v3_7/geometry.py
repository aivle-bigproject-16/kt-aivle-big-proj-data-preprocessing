from __future__ import annotations

import math
from collections.abc import Iterable

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box

from .models import ConvertedDefect


def parse_points(raw: object) -> tuple[list[tuple[float, float]], str | None]:
    if not isinstance(raw, list):
        return [], "invalid_points_structure"
    if raw and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in raw):
        pairs = list(raw)
    else:
        if len(raw) % 2:
            return [], "odd_flat_points_length"
        pairs = list(zip(raw[::2], raw[1::2]))
    points: list[tuple[float, float]] = []
    for x, y in pairs:
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return [], "non_numeric_coordinate"
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return [], "non_finite_coordinate"
        point = (float(x), float(y))
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(set(points)) < 3:
        return [], "fewer_than_3_unique_points"
    return points, None


def _polygons(geometry: object) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        result: list[Polygon] = []
        for part in geometry.geoms:
            result.extend(_polygons(part))
        return result
    return []


def _fmt(value: float) -> str:
    value = min(1.0, max(0.0, value))
    if abs(value) < 0.5e-8:
        value = 0.0
    return f"{value:.8f}"


def _det_line(class_id: int, bounds: tuple[float, float, float, float], fw: float, fh: float) -> str:
    x1, y1, x2, y2 = bounds
    return " ".join(
        [
            str(class_id),
            _fmt((x1 + x2) / (2 * fw)),
            _fmt((y1 + y2) / (2 * fh)),
            _fmt((x2 - x1) / fw),
            _fmt((y2 - y1) / fh),
        ]
    )


def _serialized_bbox_ratio(coordinates: Iterable[tuple[float, float]], fw: float, fh: float) -> float:
    """Round normalized vertices exactly as YOLO segmentation, then take bounds."""
    normalized = [(float(_fmt(x / fw)), float(_fmt(y / fh))) for x, y in coordinates]
    if not normalized:
        return 0.0
    xs, ys = zip(*normalized)
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _seg_line(class_id: int, coordinates: Iterable[tuple[float, float]], fw: float, fh: float) -> str:
    fields = [str(class_id)]
    for x, y in coordinates:
        fields.extend((_fmt(x / fw), _fmt(y / fh)))
    return " ".join(fields)


def _normalize_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if points and points[0] == points[-1]:
        points = points[:-1]
    if not points:
        return points
    variants: list[list[tuple[float, float]]] = []
    for seq in (points, list(reversed(points))):
        start = min(range(len(seq)), key=lambda i: (seq[i][0], seq[i][1], i))
        variants.append(seq[start:] + seq[:start])
    return min(variants)


def convert_defect(class_id: int, raw_points: object, fw: int, fh: int) -> ConvertedDefect:
    result = ConvertedDefect()
    if fw <= 0 or fh <= 0:
        result.issues.append("invalid_frame")
        return result
    points, issue = parse_points(raw_points)
    if issue:
        result.issues.append(issue)
        return result
    frame = box(0, 0, fw, fh)
    original = Polygon(points)
    changed = not original.is_valid
    repaired = original if original.is_valid else original.buffer(0)
    fully_unchanged = original.is_valid and frame.covers(original) and original.area > 0
    clipped = repaired.intersection(frame) if not repaired.is_empty else repaired
    if not repaired.is_empty and not frame.covers(repaired):
        changed = True
    polygons = [p for p in _polygons(clipped) if not p.is_empty and p.area > 0]
    if polygons:
        bbox_ratios: list[float] = []
        if changed:
            result.issues.append("polygon_repaired_or_clipped")
        if changed or len(polygons) > 1:
            polygons.sort(key=lambda p: (*p.bounds, p.area, tuple(p.exterior.coords)))
        for polygon in polygons:
            coordinates = points if fully_unchanged and len(polygons) == 1 else list(polygon.exterior.coords)[:-1]
            coordinates = [(min(fw, max(0.0, x)), min(fh, max(0.0, y))) for x, y in coordinates]
            if changed:
                coordinates = _normalize_ring(coordinates)
            if len(set(coordinates)) >= 3:
                result.seg_lines.append(_seg_line(class_id, coordinates, fw, fh))
                result.det_lines.append(_det_line(class_id, polygon.bounds, fw, fh))
                bbox_ratios.append(_serialized_bbox_ratio(coordinates, fw, fh))
        result.seg_valid = bool(result.seg_lines)
        result.det_valid = bool(result.det_lines)
        result.multipart_split_count = max(0, len(result.det_lines) - 1)
        result.polygon_count = len(result.seg_lines)
        result.polygon_area_ratio = sum(polygon.area for polygon in polygons) / (fw * fh)
        result.bbox_max_ratio = max(bbox_ratios, default=0.0)
        return result

    line = LineString(points + [points[0]])
    intersection = line.intersection(frame)
    if intersection.is_empty:
        result.issues.extend(("invalid_detection_bbox", "empty_or_zero_area_after_clip"))
        return result
    x1, y1, x2, y2 = intersection.bounds
    min_extent = False
    if x2 <= x1:
        x1, x2 = max(0.0, x1 - 0.5), min(float(fw), x2 + 0.5)
        min_extent = True
    if y2 <= y1:
        y1, y2 = max(0.0, y1 - 0.5), min(float(fh), y2 + 0.5)
        min_extent = True
    if x2 > x1 and y2 > y1:
        result.det_lines.append(_det_line(class_id, (x1, y1, x2, y2), fw, fh))
        result.det_valid = True
        result.min_extent_applied = min_extent
    else:
        result.issues.append("invalid_detection_bbox")
    result.issues.append("empty_or_zero_area_after_clip")
    return result
