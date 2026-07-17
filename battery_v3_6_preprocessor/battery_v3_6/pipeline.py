from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .output import write_dataset
from .reports import CSV_SCHEMAS, leakage_rows, read_csv, selected_id_rows, write_csv, write_reports
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


def approve_selection(work_dir: Path, approved_by: str, seed: int = 42) -> None:
    report_dir = work_dir.resolve() / "reports"
    required = [
        report_dir / "selected_battery_ids_candidate.csv",
        report_dir / "test_battery_ids_candidate.csv",
        report_dir / "dryrun_warnings.csv",
        report_dir / "review_warnings.csv",
        report_dir / "raw_fingerprint.sha256",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dry-run artifacts: " + ", ".join(missing))
    dry_warnings = read_csv(report_dir / "dryrun_warnings.csv")
    if dry_warnings:
        raise RuntimeError(f"cannot approve: {len(dry_warnings)} dry-run quality warning(s)")
    review = read_csv(report_dir / "review_warnings.csv")
    pending = [row for row in review if row.get("status") != "acknowledged"]
    if pending:
        raise RuntimeError(f"cannot approve: {len(pending)} review warning(s) not acknowledged")
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
        if warnings:
            raise RuntimeError(f"execute blocked by {len(warnings)} quality gate warning(s)")
        review = read_csv(report_dir / "review_warnings.csv")
        if any(row.get("status") != "acknowledged" for row in review):
            raise RuntimeError("execute blocked by unacknowledged review warning(s)")
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
