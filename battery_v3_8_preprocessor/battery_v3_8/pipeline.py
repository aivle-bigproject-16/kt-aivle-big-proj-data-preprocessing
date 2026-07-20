from __future__ import annotations

import json
import hashlib
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .ct_area import (
    CT_PRE_SPLIT_AREA_THRESHOLD,
    CT_PRE_SPLIT_EXCLUSION_REASON,
    CT_PRE_SPLIT_POLICY_PRECISION,
)
from .output import write_dataset
from .reports import (
    CSV_SCHEMAS,
    leakage_rows,
    quality_exception_rows,
    read_csv,
    selected_id_rows,
    write_csv,
    write_reports,
)
from .scan import scan_dataset
from .selection import SelectionResult, assign_dataset


def dry_run(raw_root: Path, work_dir: Path, seed: int = 42, jobs: int = 1) -> tuple[SelectionResult, list[str]]:
    work_dir = work_dir.resolve()
    report_dir = work_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    scan = scan_dataset(raw_root, jobs=jobs)
    selection = assign_dataset(scan.valid_samples, seed=seed)
    leak_rows = leakage_rows(selection)
    failed = [row for row in leak_rows if row["status"] == "FAIL"]
    if failed:
        write_csv(report_dir / "split_id_leakage_audit.csv", CSV_SCHEMAS["split_id_leakage_audit.csv"], leak_rows)
        raise RuntimeError(f"structural leakage gate failed: {len(failed)} pair(s)")
    warnings = write_reports(scan, selection, report_dir)
    return selection, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pre_split_policy() -> dict[str, object]:
    return {
        "modality": "CT",
        "metric": "porosity_bbox_max_ratio",
        "precision": CT_PRE_SPLIT_POLICY_PRECISION,
        "operator": ">=",
        "threshold": CT_PRE_SPLIT_AREA_THRESHOLD,
        "reason": CT_PRE_SPLIT_EXCLUSION_REASON,
    }


def _validate_quality_exceptions(warnings: list[str], rows: list[dict[str, str]]) -> None:
    expected = {row["warning_id"]: row for row in quality_exception_rows(warnings)}
    actual = {row.get("warning_id", ""): row for row in rows}
    if set(actual) != set(expected):
        raise RuntimeError("quality exception warning IDs differ from dry-run warnings")
    for warning_id, expected_row in expected.items():
        row = actual[warning_id]
        for field in ("warning_code", "observed_value", "threshold"):
            if row.get(field) != expected_row[field]:
                raise RuntimeError(f"quality exception {warning_id} changed field: {field}")
        if row.get("status") != "approved_exception":
            raise RuntimeError(f"quality exception {warning_id} is not approved_exception")
        if not all(row.get(field, "").strip() for field in ("reviewer", "reviewed_at", "reason")):
            raise RuntimeError(f"quality exception {warning_id} is missing audit fields")


def approve_selection(work_dir: Path, approved_by: str, seed: int = 42) -> None:
    report_dir = work_dir.resolve() / "reports"
    required = [
        report_dir / "selected_battery_ids_candidate.csv",
        report_dir / "test_battery_ids_candidate.csv",
        report_dir / "dryrun_warnings.csv",
        report_dir / "review_warnings.csv",
        report_dir / "quality_exceptions.csv",
        report_dir / "raw_fingerprint.sha256",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dry-run artifacts: " + ", ".join(missing))
    dry_warnings = [row.get("guard", "") for row in read_csv(report_dir / "dryrun_warnings.csv")]
    exception_rows = read_csv(report_dir / "quality_exceptions.csv")
    if dry_warnings:
        _validate_quality_exceptions(dry_warnings, exception_rows)
    elif exception_rows:
        raise RuntimeError("quality exceptions exist without dry-run quality warnings")
    destinations = [report_dir / "selected_battery_ids.csv", report_dir / "test_battery_ids.csv", report_dir / "approval.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("approval artifacts already exist; dataset version must not be overwritten")
    shutil.copy2(report_dir / "selected_battery_ids_candidate.csv", destinations[0])
    shutil.copy2(report_dir / "test_battery_ids_candidate.csv", destinations[1])
    approval = {
        "approved_by": approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "raw_fingerprint": (report_dir / "raw_fingerprint.sha256").read_text(encoding="ascii").strip(),
        "pre_split_policy": _pre_split_policy(),
        "id_statistics_fingerprint": _sha256(report_dir / "selected_battery_ids_candidate.csv"),
        "artifact_sha256": {
            "selected_battery_ids.csv": _sha256(destinations[0]),
            "test_battery_ids.csv": _sha256(destinations[1]),
            "review_warnings.csv": _sha256(report_dir / "review_warnings.csv"),
            "quality_exceptions.csv": _sha256(report_dir / "quality_exceptions.csv"),
        },
    }
    destinations[2].write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _locked_tests(report_dir: Path) -> dict[str, set[str]]:
    rows = read_csv(report_dir / "test_battery_ids.csv")
    result = {"CT": set(), "EXT": set()}
    for row in rows:
        result.setdefault(row["modality"], set()).add(row["battery_id"])
    return result


def _mapping(rows: list[dict[str, str]]) -> dict[tuple[str, str], tuple[str, str]]:
    return {(row["modality"], row["battery_id"]): (row["split_role"], row["fold_id"]) for row in rows}


def execute(
    raw_root: Path,
    work_dir: Path,
    output: Path,
    seed: int = 42,
    keep_failed_staging: bool = False,
    jobs: int = 1,
) -> None:
    work_dir, output = work_dir.resolve(), output.resolve()
    report_dir = work_dir / "reports"
    approval_path = report_dir / "approval.json"
    if not approval_path.exists():
        raise FileNotFoundError("approval.json is required; run approve-selection first")
    if output.exists():
        raise FileExistsError(f"final output already exists: {output}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("seed") != seed:
        raise RuntimeError("seed differs from approval")
    expected_policy = _pre_split_policy()
    if approval.get("pre_split_policy") != expected_policy:
        raise RuntimeError("pre-split policy differs from approval or is missing")
    artifact_hashes = approval.get("artifact_sha256")
    required_artifacts = {
        "selected_battery_ids.csv",
        "test_battery_ids.csv",
        "review_warnings.csv",
        "quality_exceptions.csv",
    }
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != required_artifacts:
        raise RuntimeError("approval artifact hashes are missing or incomplete")
    for filename, expected_hash in artifact_hashes.items():
        path = report_dir / filename
        if not path.exists() or _sha256(path) != expected_hash:
            raise RuntimeError(f"approved artifact changed: {filename}")
    run_id = uuid.uuid4().hex[:12]
    staging = output.parent / f"{output.name}.staging-{run_id}"
    failed_dir = work_dir / "failed_runs" / run_id
    try:
        scan = scan_dataset(raw_root, jobs=jobs)
        if scan.raw_fingerprint != approval.get("raw_fingerprint"):
            raise RuntimeError("raw dataset fingerprint differs from approval")
        selection = assign_dataset(scan.valid_samples, seed=seed, locked_tests=_locked_tests(report_dir))
        approved_mapping = _mapping(read_csv(report_dir / "selected_battery_ids.csv"))
        current_mapping = _mapping([{key: str(value) for key, value in row.items()} for row in selected_id_rows(selection)])
        if current_mapping != approved_mapping:
            raise RuntimeError("recomputed split/fold mapping differs from approved selection")
        warnings = write_reports(scan, selection, staging / "reports")
        if _sha256(staging / "reports" / "selected_battery_ids_candidate.csv") != approval.get("id_statistics_fingerprint"):
            raise RuntimeError("recomputed ID statistics differ from approval")
        if warnings:
            _validate_quality_exceptions(warnings, read_csv(report_dir / "quality_exceptions.csv"))
        elif read_csv(report_dir / "quality_exceptions.csv"):
            raise RuntimeError("approved quality exceptions no longer match current warnings")
        approved_review = {row.get("warning", "") for row in read_csv(report_dir / "review_warnings.csv")}
        current_review = {row.get("warning", "") for row in read_csv(staging / "reports" / "review_warnings.csv")}
        if current_review != approved_review:
            raise RuntimeError("recomputed review warnings differ from approval review basis")
        for filename in ("selected_battery_ids.csv", "test_battery_ids.csv", "approval.json"):
            shutil.copy2(report_dir / filename, staging / "reports" / filename)
        write_dataset(staging, selection, jobs=jobs)
        staging.replace(output)
    except Exception as exc:
        failed_dir.mkdir(parents=True, exist_ok=True)
        (failed_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8", newline="\n")
        if staging.exists() and (staging / "reports").exists():
            shutil.copytree(staging / "reports", failed_dir / "reports", dirs_exist_ok=True)
        if staging.exists() and not keep_failed_staging:
            shutil.rmtree(staging)
        raise
