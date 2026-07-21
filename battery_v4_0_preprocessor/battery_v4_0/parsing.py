from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .deterministic import normalize_battery_id
from .models import ParsedName

def parse_filename(filename: str) -> ParsedName | None:
    stem = Path(filename).stem
    if "_module_" in stem.casefold() or "_cell_" not in stem.casefold():
        return None
    parts = stem.split("_")
    if len(parts) >= 6 and parts[0].casefold() == "ct" and parts[1].casefold() == "cell":
        form, battery_id, axis, frame = "_".join(parts[2:-3]), parts[-3], parts[-2].casefold(), parts[-1]
        if form and battery_id.isdigit() and axis in {"x", "y", "z"} and frame:
            return ParsedName("CT", normalize_battery_id(battery_id), axis, form, frame)
    if len(parts) >= 5 and parts[0].casefold() == "rgb" and parts[1].casefold() == "cell":
        form, battery_id, frame = "_".join(parts[2:-2]), parts[-2], parts[-1]
        if form and battery_id.isdigit() and frame:
            return ParsedName("EXT", normalize_battery_id(battery_id), "", form, frame)
    return None


def normalize_class_token(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split()).casefold()


def canonical_class(raw: Any, modality: str) -> str | None:
    if not isinstance(raw, str):
        return None
    token = normalize_class_token(raw)
    if token == "battery outline":
        return None
    if modality == "CT":
        return "porosity" if token == "porosity" else None
    aliases = {
        "damaged": "Damaged",
        "damage": "Damaged",
        "pollution": "Pollution",
        "contamination": "Pollution",
    }
    return aliases.get(token)


def is_ignored_outline(raw: Any) -> bool:
    return isinstance(raw, str) and normalize_class_token(raw) == "battery outline"


def normalize_application(raw: Any) -> str:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return "빈값"
    token = text.casefold()
    if token == "가전" or token in {"home", "consumer", "appliance"}:
        return "가전"
    if token == "산업" or token in {"industrial", "industry", "ev"}:
        return "산업"
    return text


def choose_application(values: list[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value.casefold(), value)) if counts else "빈값"


def defect_key(defect: dict[str, Any]) -> str:
    return json.dumps([defect.get("name"), defect.get("points")], ensure_ascii=False, separators=(",", ":"))


def deduplicate_defects(defects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for defect in defects:
        key = defect_key(defect)
        if key not in seen:
            seen.add(key)
            kept.append(defect)
    return kept, len(defects) - len(kept)


def find_json_fields(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Any]:
    data_info = payload.get("data_info") if isinstance(payload.get("data_info"), dict) else {}
    image_info = payload.get("image_info") if isinstance(payload.get("image_info"), dict) else {}
    defects = payload.get("defects") if isinstance(payload.get("defects"), list) else []
    is_normal = image_info.get("is_normal", payload.get("is_normal", data_info.get("is_normal")))
    return data_info, image_info, [d for d in defects if isinstance(d, dict)], is_normal
