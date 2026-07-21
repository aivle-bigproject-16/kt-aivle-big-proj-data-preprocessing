from __future__ import annotations

import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from .models import Sample
from .scan import sha256_file
from .selection import SelectionResult


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _save_image(sample: Sample, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sample.image_path is None:
        raise ValueError(f"missing source image for {sample.sample_id}")
    if sample.modality == "EXT":
        shutil.copy2(sample.image_path, destination)
        return
    with Image.open(sample.image_path) as image:
        mode = "L" if image.mode == "L" else "RGB"
        cropped = image.convert(mode).crop((0, 0, sample.roi_w, sample.roi_h))
        save_args = dict(format="JPEG", quality=95, optimize=False, progressive=False)
        if mode == "RGB":
            save_args["subsampling"] = 0
        cropped.save(destination, **save_args)


def _sample_destination(root: Path, sample: Sample) -> tuple[Path, Path, Path, Path]:
    bucket = "test" if sample.split_role == "test" else "trainval"
    base = root / sample.modality / bucket
    extension = ".jpg" if sample.modality == "CT" else (sample.image_path.suffix.lower() if sample.image_path else ".jpg")
    return (
        base / "images" / f"{sample.sample_id}{extension}",
        base / "labels_det" / f"{sample.sample_id}.txt",
        base / "labels_seg" / f"{sample.sample_id}.txt",
        base / "labels_json" / sample.json_relative_posix,
    )


def _write_one_sample(payload: tuple[Path, Sample]) -> None:
    """샘플 1개의 이미지·det/seg TXT·JSON 사본을 쓰고 JSON 해시를 역검증한다(멀티프로세스 워커).

    출력 경로가 sample_id로 유일하고 서로 독립이라 순서 무관. 예외는 상위로 전파돼 staging이 폐기된다.
    """
    root, sample = payload
    image_dest, det_dest, seg_dest, json_dest = _sample_destination(root, sample)
    _save_image(sample, image_dest)
    if sample.included_det:
        _write_text(det_dest, "\n".join(sample.det_lines) + ("\n" if sample.det_lines else ""))
    if sample.included_seg:
        _write_text(seg_dest, "\n".join(sample.seg_lines) + ("\n" if sample.seg_lines else ""))
    if sample.json_path is None:
        raise ValueError(f"missing source JSON for {sample.sample_id}")
    json_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.json_path, json_dest)
    if sha256_file(json_dest) != sample.json_sha256:
        raise RuntimeError(f"JSON hash mismatch after copy: {sample.sample_id}")


def _write_samples(root: Path, selection: SelectionResult, jobs: int = 1) -> None:
    tasks = [(root, sample) for stats in selection.ids for sample in stats.selected_samples]
    # CT 크롭+재인코딩이 지배적 비용. 파일별 독립이라 프로세스 분산해도 산출물 동일.
    if jobs and jobs > 1 and tasks:
        chunk = max(1, len(tasks) // (jobs * 8))
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for _ in executor.map(_write_one_sample, tasks, chunksize=chunk):
                pass
    else:
        for task in tasks:
            _write_one_sample(task)


def _write_membership(root: Path, selection: SelectionResult) -> None:
    ct_dev = [stats for stats in selection.ids if stats.modality == "CT" and stats.split_role == "development"]
    ct_test = [stats for stats in selection.ids if stats.modality == "CT" and stats.split_role == "test"]
    for fold in range(5):
        val_ids = {stats.battery_id for stats in ct_dev if stats.fold_id == str(fold)}
        train = sorted(sample.sample_id for stats in ct_dev if stats.battery_id not in val_ids for sample in stats.selected_samples)
        val = sorted(sample.sample_id for stats in ct_dev if stats.battery_id in val_ids for sample in stats.selected_samples)
        _write_text(root / f"CT/trainval/folds/fold_{fold}/train.txt", "\n".join(train) + "\n")
        _write_text(root / f"CT/trainval/folds/fold_{fold}/val.txt", "\n".join(val) + "\n")
    _write_text(root / "CT/test/test.txt", "\n".join(sorted(sample.sample_id for stats in ct_test for sample in stats.selected_samples)) + "\n")
    ext = [stats for stats in selection.ids if stats.modality == "EXT"]
    for role in ("train", "val"):
        ids = sorted(sample.sample_id for stats in ext if stats.split_role == role for sample in stats.selected_samples)
        _write_text(root / f"EXT/trainval/splits/{role}.txt", "\n".join(ids) + "\n")
    _write_text(root / "EXT/test/test.txt", "\n".join(sorted(sample.sample_id for stats in ext if stats.split_role == "test" for sample in stats.selected_samples)) + "\n")
    for modality, classes in (("CT", "porosity\n"), ("EXT", "Damaged\nPollution\n")):
        _write_text(root / modality / "trainval/classes.txt", classes)
        _write_text(root / modality / "test/classes.txt", classes)


def _verify_output(root: Path, selection: SelectionResult) -> None:
    seen: set[str] = set()
    for stats in selection.ids:
        for sample in stats.selected_samples:
            if sample.sample_id in seen:
                raise RuntimeError(f"duplicate output sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
            image, det, seg, json_path = _sample_destination(root, sample)
            if not image.exists() or not json_path.exists():
                raise RuntimeError(f"missing output pair: {sample.sample_id}")
            if det.exists() != sample.included_det or seg.exists() != sample.included_seg:
                raise RuntimeError(f"task label presence mismatch: {sample.sample_id}")
            for label in (det, seg):
                if not label.exists():
                    continue
                raw_text = label.read_text(encoding="utf-8")
                if raw_text and not raw_text.endswith("\n"):
                    raise RuntimeError(f"YOLO label missing final newline: {label}")
                if sample.is_normal_interpreted and raw_text:
                    raise RuntimeError(f"normal image label must be empty: {label}")
                if not sample.is_normal_interpreted and not raw_text:
                    raise RuntimeError(f"defect image label must not be empty: {label}")
                valid_classes = {0} if sample.modality == "CT" else {0, 1}
                for line in raw_text.splitlines():
                    fields = line.split()
                    is_detection = label == det
                    if is_detection and len(fields) != 5:
                        raise RuntimeError(f"invalid YOLO line in {label}")
                    if not is_detection and (len(fields) < 7 or (len(fields) - 1) % 2 != 0):
                        raise RuntimeError(f"invalid segmentation coordinate pairs in {label}")
                    try:
                        class_id = int(fields[0])
                        coordinates = [float(value) for value in fields[1:]]
                    except ValueError as exc:
                        raise RuntimeError(f"invalid YOLO numeric field in {label}") from exc
                    if class_id not in valid_classes:
                        raise RuntimeError(f"invalid class ID in {label}: {class_id}")
                    if any(not (0.0 <= value <= 1.0) for value in coordinates):
                        raise RuntimeError(f"YOLO coordinate out of range in {label}")


def _zip_tree(root: Path, source: Path, destination: Path) -> None:
    # 수록 대상이 PNG·JPG로 이미 압축돼 있어 deflate 이득이 거의 없다.
    # v3.6 산출물 표본 측정(각 1,600 파일): deflate level 6은 CT 3.7%, EXT 1.2%만 줄이면서
    # 처리량이 31~36 MB/s에 그쳐 약 93GB 기준 40분대의 직렬 구간을 만든다.
    # STORED는 144~314 MB/s로 같은 구간을 크게 줄이고, 늘어난 용량은 업로드 시간 1분 수준이다.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())


def write_dataset(root: Path, selection: SelectionResult, jobs: int = 1) -> None:
    _write_samples(root, selection, jobs)
    _write_membership(root, selection)
    _verify_output(root, selection)
    _write_text(root / "README.md", "# Battery v3.9 dataset\n\nTest data is for final evaluation only. Do not use it for training or model selection.\n")
    _write_text(root / "requirements.lock", "Pillow==12.2.0\nshapely==2.1.2\n")
    source_package = Path(__file__).resolve().parent
    shutil.copytree(
        source_package,
        root / "battery_v3_9",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    wrapper = (
        "from pathlib import Path\n"
        "from battery_v3_9.training_view import build_training_view\n"
        "from battery_v3_9.cli import training_view_main\n\n"
        "if __name__ == '__main__':\n"
        "    training_view_main(Path(__file__).resolve().parent)\n"
    )
    _write_text(root / "prepare_training_view.py", wrapper)
    for modality, label in (("CT", "CT"), ("EXT", "EXT")):
        _zip_tree(root, root / modality / "trainval", root / f"battery_{label}_v3_trainval.zip")
        _zip_tree(root, root / modality / "test", root / f"battery_{label}_v3_test.zip")
