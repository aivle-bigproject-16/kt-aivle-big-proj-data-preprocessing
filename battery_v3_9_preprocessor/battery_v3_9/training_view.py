from __future__ import annotations

import os
import shutil
import csv
from pathlib import Path


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _image_map(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(directory.iterdir()) if path.is_file()}


def _link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(f"hardlink failed for {source}; rerun with --copy") from exc


def _verify_leakage(dataset_root: Path, role_members: dict[str, set[str]]) -> None:
    audit = dataset_root / "reports" / "split_id_leakage_audit.csv"
    if audit.exists():
        with audit.open("r", encoding="utf-8-sig", newline="") as stream:
            failed = [row for row in csv.DictReader(stream) if row.get("status") != "PASS"]
        if failed:
            raise ValueError(f"master leakage audit contains {len(failed)} failure(s)")
    manifest = dataset_root / "reports" / "manifest.csv"
    if not manifest.exists():
        return
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    role_ids = {role: {rows[sid]["battery_id"] for sid in members if sid in rows} for role, members in role_members.items()}
    role_hashes = {role: {rows[sid]["pixel_hash"] for sid in members if sid in rows and rows[sid]["pixel_hash"]} for role, members in role_members.items()}
    names = sorted(role_members)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if role_ids[left] & role_ids[right] or role_hashes[left] & role_hashes[right]:
                raise ValueError(f"leakage detected between {left} and {right}")


def build_training_view(
    dataset_root: Path,
    output: Path,
    modality: str,
    task: str,
    fold: int | None = None,
    include_test: bool = False,
    copy: bool = False,
) -> None:
    dataset_root, output = dataset_root.resolve(), output.resolve()
    modality = modality.upper()
    if modality not in {"CT", "EXT"}:
        raise ValueError("modality must be CT or EXT")
    if task not in {"det", "seg"}:
        raise ValueError("task must be det or seg")
    if output.exists():
        raise FileExistsError(f"training view already exists: {output}")
    if modality == "CT" and (fold is None or fold not in range(5)):
        raise ValueError("CT requires --fold 0..4")
    if modality == "EXT" and fold is not None:
        raise ValueError("EXT does not use --fold")
    trainval = dataset_root / modality / "trainval"
    test_root = dataset_root / modality / "test"
    if modality == "CT":
        split_root = trainval / "folds" / f"fold_{fold}"
    else:
        split_root = trainval / "splits"
    roles = {"train": _read_ids(split_root / "train.txt"), "val": _read_ids(split_root / "val.txt")}
    if include_test:
        roles["test"] = _read_ids(test_root / "test.txt")
    train_images = _image_map(trainval / "images")
    test_images = _image_map(test_root / "images") if include_test else {}
    train_labels = trainval / f"labels_{task}"
    test_labels = test_root / f"labels_{task}"
    try:
        seen: dict[str, str] = {}
        role_members: dict[str, set[str]] = {role: set() for role in roles}
        for role, sample_ids in roles.items():
            image_lookup = test_images if role == "test" else train_images
            label_root = test_labels if role == "test" else train_labels
            for sample_id in sample_ids:
                label = label_root / f"{sample_id}.txt"
                image = image_lookup.get(sample_id)
                if not label.exists():
                    continue  # task-specific image sets are allowed by v3.9
                if image is None:
                    raise FileNotFoundError(f"image missing for {sample_id}")
                if sample_id in seen:
                    raise ValueError(f"sample leakage between {seen[sample_id]} and {role}: {sample_id}")
                seen[sample_id] = role
                role_members[role].add(sample_id)
                _link_or_copy(image, output / "images" / role / image.name, copy)
                _link_or_copy(label, output / "labels" / role / label.name, copy)
        names = ["porosity"] if modality == "CT" else ["Damaged", "Pollution"]
        lines = [
            f"path: {output.as_posix()}",
            "train: images/train",
            "val: images/val",
        ]
        if include_test:
            lines.append("test: images/test")
        lines.extend((f"nc: {len(names)}", "names: [" + ", ".join(names) + "]"))
        (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        _verify_leakage(dataset_root, role_members)
        for role in roles:
            image_stems = {path.stem for path in (output / "images" / role).iterdir()} if (output / "images" / role).exists() else set()
            label_stems = {path.stem for path in (output / "labels" / role).iterdir()} if (output / "labels" / role).exists() else set()
            if image_stems != label_stems:
                raise RuntimeError(f"image-label mismatch in {role}")
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
