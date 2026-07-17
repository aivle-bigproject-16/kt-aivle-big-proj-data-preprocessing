import csv
import tempfile
import unittest
from pathlib import Path

from battery_v3_6.models import IdStats, Sample
from battery_v3_6.reports import write_reports
from battery_v3_6.scan import ScanResult
from battery_v3_6.selection import SelectionResult


class ReportTests(unittest.TestCase):
    def test_report_rows_are_sorted_and_eda_has_each_ct_fold(self):
        samples = []
        ids = []
        for fold in range(5):
            sample = Sample(
                sample_id=f"CT__sample_{fold}__0000000{fold}",
                modality="CT",
                battery_id=str(fold + 1),
                axis="x",
                included_det=True,
                included_seg=True,
                det_lines=["0 0.50000000 0.50000000 0.20000000 0.20000000"],
                seg_lines=["0 0.10000000 0.10000000 0.20000000 0.10000000 0.10000000 0.20000000"],
                class_names=["porosity"],
                selected=True,
                split_role="development",
                fold_id=str(fold),
                pixel_hash=f"hash-{fold}",
            )
            samples.append(sample)
            ids.append(IdStats("CT", str(fold + 1), "산업", [sample], [sample], "development", str(fold)))

        scan = ScanResult(raw_root=Path("."), samples=samples)
        scan.json_anomalies = [
            {"sample_id": "z", "path": "z.json", "issue": "z_issue", "detail": ""},
            {"sample_id": "a", "path": "a.json", "issue": "a_issue", "detail": ""},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            write_reports(scan, SelectionResult(ids), report_dir)

            with (report_dir / "json_anomalies.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["sample_id"] for row in rows], ["a", "z"])

            eda = (report_dir / "eda_v3_postcrop.md").read_text(encoding="utf-8")
            summary = eda.split("## Annotation density guard", 1)[0]
            for fold in range(5):
                self.assertIn(f"CT_fold_{fold}", summary)

            scan_summary = (report_dir / "scan_summary.md").read_text(encoding="utf-8")
            self.assertIn("JSON anomalies: 2", scan_summary)

    def test_reports_ct_test_and_development_axis_balance(self):
        development_sample = Sample(
            sample_id="CT__development__00000001",
            modality="CT",
            battery_id="1",
            axis="x",
            included_det=True,
            selected=True,
            split_role="development",
        )
        test_sample = Sample(
            sample_id="CT__test__00000002",
            modality="CT",
            battery_id="2",
            axis="y",
            included_det=True,
            selected=True,
            split_role="test",
        )
        ids = [
            IdStats("CT", "1", "산업", [development_sample], [development_sample], "development", "0"),
            IdStats("CT", "2", "산업", [test_sample], [test_sample], "test", ""),
        ]
        scan = ScanResult(raw_root=Path("."), samples=[development_sample, test_sample])

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            write_reports(scan, SelectionResult(ids), report_dir)

            with (report_dir / "split_axis_balance.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["split"] for row in rows], ["development", "test"])
            self.assertEqual(rows[0]["x_ratio"], "1.00000000")
            self.assertEqual(rows[1]["y_ratio"], "1.00000000")
            self.assertEqual(rows[0]["axis_max_gap"], "1.00000000")
            self.assertEqual(rows[1]["axis_sum_gap"], "2.00000000")
            self.assertEqual(
                {axis: rows[0][f"{axis}_gap"] for axis in "xyz"},
                {"x": "1.00000000", "y": "1.00000000", "z": "0.00000000"},
            )
            self.assertEqual(
                {axis: rows[0][f"{axis}_gap"] for axis in "xyz"},
                {"x": "1.00000000", "y": "1.00000000", "z": "0.00000000"},
            )


if __name__ == "__main__":
    unittest.main()

