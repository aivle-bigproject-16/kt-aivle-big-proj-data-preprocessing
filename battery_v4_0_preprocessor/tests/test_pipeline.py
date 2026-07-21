import json
import tempfile
import unittest
from pathlib import Path

from battery_v4_0.pipeline import approve_selection, execute
from battery_v4_0.reports import CSV_SCHEMAS, quality_exception_rows, write_csv


class PipelineTests(unittest.TestCase):
    def _write_minimal_reports(self, reports: Path, warning: str | None = None) -> None:
        reports.mkdir(parents=True)
        selected = {field: "" for field in CSV_SCHEMAS["selected_battery_ids_candidate.csv"]}
        selected.update({"modality": "CT", "battery_id": "1", "split_role": "test"})
        write_csv(reports / "selected_battery_ids_candidate.csv", CSV_SCHEMAS["selected_battery_ids_candidate.csv"], [selected])
        write_csv(reports / "test_battery_ids_candidate.csv", CSV_SCHEMAS["test_battery_ids_candidate.csv"], [{"modality": "CT", "battery_id": "1"}])
        write_csv(reports / "dryrun_warnings.csv", CSV_SCHEMAS["dryrun_warnings.csv"], [{"guard": warning}] if warning else [])
        write_csv(reports / "review_warnings.csv", CSV_SCHEMAS["review_warnings.csv"], [{"warning": "review only", "reviewer": "", "reviewed_at": "", "status": "pending"}])
        rows = quality_exception_rows([warning] if warning else [])
        write_csv(reports / "quality_exceptions.csv", CSV_SCHEMAS["quality_exceptions.csv"], rows)
        strat = [
            {"scope": "overall", "positive_rate_bin": name, "candidate_id_count": 0,
             "target_test_id_count": 0, "actual_test_id_count": 0, "development_id_count": 0, "status": "PASS"}
            for name in ("zero", "very_low", "low_mid", "mid_high", "very_high")
        ]
        write_csv(reports / "ct_id_positive_rate_stratification.csv", CSV_SCHEMAS["ct_id_positive_rate_stratification.csv"], strat)
        (reports / "raw_fingerprint.sha256").write_text("abc\n", encoding="ascii")

    def test_approval_records_v37_pre_split_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            reports = work_dir / "reports"
            self._write_minimal_reports(reports)

            approve_selection(work_dir, "reviewer", seed=42)

            approval = json.loads((reports / "approval.json").read_text(encoding="utf-8"))
            self.assertEqual(
                approval["pre_split_policy"],
                {
                    "modality": "CT",
                    "metric": "porosity_bbox_max_ratio",
                    "precision": 8,
                    "operator": ">=",
                    "threshold": 0.25,
                    "reason": "ct_porosity_bbox_max_ratio_ge_0.25",
                },
            )
            self.assertEqual(
                set(approval["artifact_sha256"]),
                {"selected_battery_ids.csv", "test_battery_ids.csv", "review_warnings.csv", "quality_exceptions.csv", "ct_id_positive_rate_stratification.csv"},
            )
            self.assertEqual(len(approval["id_statistics_fingerprint"]), 64)

    def test_approved_quality_exception_allows_approval_with_audit_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            warning = "CT_fold: annotations_per_image deviation=0.30000000 target=1.00000000 achieved=1.30000000"
            self._write_minimal_reports(reports, warning)
            rows = quality_exception_rows([warning])
            rows[0].update({
                "status": "approved_exception",
                "reviewer": "qa",
                "reviewed_at": "2026-07-20T00:00:00Z",
                "reason": "accepted imbalance",
            })
            write_csv(reports / "quality_exceptions.csv", CSV_SCHEMAS["quality_exceptions.csv"], rows)
            approve_selection(Path(tmp), "reviewer", seed=42)
            self.assertTrue((reports / "approval.json").exists())

    def test_quality_exception_without_audit_fields_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            warning = "CT_fold: annotations_per_image deviation=0.30000000 target=1.00000000 achieved=1.30000000"
            self._write_minimal_reports(reports, warning)
            rows = quality_exception_rows([warning])
            rows[0]["status"] = "approved_exception"
            write_csv(reports / "quality_exceptions.csv", CSV_SCHEMAS["quality_exceptions.csv"], rows)
            with self.assertRaisesRegex(RuntimeError, "missing audit fields"):
                approve_selection(Path(tmp), "reviewer", seed=42)

    def test_execute_rejects_approved_membership_file_mutation_before_scanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "work" / "reports"
            self._write_minimal_reports(reports)
            approve_selection(root / "work", "reviewer", seed=42)
            with (reports / "selected_battery_ids.csv").open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "approved artifact changed"):
                execute(root / "raw", root / "work", root / "output", seed=42)


if __name__ == "__main__":
    unittest.main()
