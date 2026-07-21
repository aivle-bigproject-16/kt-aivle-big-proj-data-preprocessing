import csv
import tempfile
import unittest
from pathlib import Path

from battery_v4_0.models import IdStats, Sample
from battery_v4_0.reports import write_reports
from battery_v4_0.scan import ScanResult
from battery_v4_0.selection import SelectionResult


class ReportTests(unittest.TestCase):
    def test_selected_bbox_policy_leak_is_non_overridable_structural_failure(self):
        leaked = Sample(
            sample_id="CT__leaked__00000001", modality="CT", battery_id="1", axis="x",
            included_det=True, included_seg=True, selected=True,
            porosity_bbox_max_ratio=0.25,
        )
        stats = IdStats("CT", "1", "industrial", [leaked], [leaked], "test", "")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "structural CT bbox policy gate failed"):
                write_reports(
                    ScanResult(raw_root=Path("."), samples=[leaked]),
                    SelectionResult([stats]),
                    Path(tmp),
                )

    def test_original_ge_40pct_images_are_recorded_as_review_warning(self):
        removed = Sample(
            sample_id="CT__large__00000001", modality="CT", battery_id="134", axis="z",
            included_det=True, porosity_area_sum_ratio=0.40, pre_split_eligible=False,
            porosity_bbox_max_ratio=0.40,
            pre_split_exclusion_reason="ct_porosity_bbox_max_ratio_ge_0.25",
        )
        scan = ScanResult(raw_root=Path("."), samples=[removed])
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            write_reports(scan, SelectionResult([]), report_dir)
            with (report_dir / "review_warnings.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertIn("1 valid image(s)", rows[0]["warning"])
            self.assertEqual(rows[0]["status"], "pending")

    def test_reports_preserve_pre_split_exclusion_lineage(self):
        kept = Sample(
            sample_id="CT__kept__00000001", modality="CT", battery_id="1", axis="x",
            included_det=True, included_seg=True, selected=True,
            split_role="development", fold_id="0", porosity_area_sum_ratio=0.30,
            porosity_bbox_max_ratio=0.24999999,
        )
        removed = Sample(
            sample_id="CT__removed__00000002", modality="CT", battery_id="1", axis="z",
            included_det=True, included_seg=True, selected=False,
            porosity_area_sum_ratio=0.10, porosity_bbox_max_ratio=0.25,
            pre_split_eligible=False,
            pre_split_exclusion_reason="ct_porosity_bbox_max_ratio_ge_0.25",
        )
        ids = [IdStats("CT", "1", "industrial", [kept], [kept], "development", "0")]
        scan = ScanResult(raw_root=Path("."), samples=[kept, removed])
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            write_reports(scan, SelectionResult(ids), report_dir)
            with (report_dir / "ct_bbox_exclusions.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                excluded_rows = list(csv.DictReader(stream))
            self.assertEqual([row["sample_id"] for row in excluded_rows], [removed.sample_id])
            with (report_dir / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}
            self.assertEqual(manifest[removed.sample_id]["included_det"], "False")
            self.assertEqual(
                manifest[removed.sample_id]["exclusion_reason_det"],
                "ct_porosity_bbox_max_ratio_ge_0.25",
            )
            self.assertEqual(manifest[removed.sample_id]["porosity_bbox_max_ratio"], "0.25000000")

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

    def test_ct_positive_rate_stratification_report_and_bin_column(self):
        # 양성률 구간을 넘나드는 CT ID 7개: test 2, development 5 형태를 흉내낸다.
        rates = [("1", 0.0, "test"), ("2", 0.0, "development"), ("3", 0.15, "test"),
                 ("4", 0.15, "development"), ("5", 0.45, "development"),
                 ("6", 0.70, "development"), ("7", 0.70, "development")]
        ids = []
        all_samples = []
        for bid, ratio, role in rates:
            n_def = round(100 * ratio)
            samples = [
                Sample(sample_id=f"CT_{bid}_x_{i}", modality="CT", battery_id=bid, axis="x",
                       included_det=True, included_seg=True,
                       det_lines=["0 0.5 0.5 0.1 0.1"] if i < n_def else [],
                       selected=True, split_role=role, fold_id="", pixel_hash=f"{bid}-{i}")
                for i in range(100)
            ]
            all_samples.extend(samples)
            st = IdStats("CT", bid, "산업", samples, samples, role, "")
            ids.append(st)
        selection = SelectionResult(ids, ct_test_positive_rate_quotas={"zero": 1, "very_low": 0, "low_mid": 1, "mid_high": 0, "very_high": 0})
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            write_reports(ScanResult(raw_root=Path("."), samples=all_samples), selection, report_dir)

            with (report_dir / "ct_id_positive_rate_stratification.csv").open(encoding="utf-8-sig", newline="") as s:
                rows = list(csv.DictReader(s))
            overall = [r for r in rows if r["scope"] == "overall"]
            self.assertEqual([r["positive_rate_bin"] for r in overall],
                             ["zero", "very_low", "low_mid", "mid_high", "very_high"])
            zero_row = next(r for r in overall if r["positive_rate_bin"] == "zero")
            self.assertEqual(zero_row["actual_test_id_count"], "1")
            self.assertEqual(zero_row["status"], "PASS")

            with (report_dir / "selected_battery_ids_candidate.csv").open(encoding="utf-8-sig", newline="") as s:
                sel_rows = list(csv.DictReader(s))
            self.assertIn("ct_id_positive_rate_bin", sel_rows[0])
            by_id = {r["battery_id"]: r["ct_id_positive_rate_bin"] for r in sel_rows}
            self.assertEqual(by_id["1"], "zero")
            self.assertEqual(by_id["6"], "very_high")

            eda = (report_dir / "eda_v3_postcrop.md").read_text(encoding="utf-8")
            self.assertIn("CT ID positive-rate stratification", eda)
            self.assertIn("image-micro", eda)

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
                {axis: rows[1][f"{axis}_gap"] for axis in "xyz"},
                {"x": "1.00000000", "y": "1.00000000", "z": "0.00000000"},
            )
            self.assertEqual(
                {axis: rows[0][f"{axis}_gap"] for axis in "xyz"},
                {"x": "1.00000000", "y": "1.00000000", "z": "0.00000000"},
            )


if __name__ == "__main__":
    unittest.main()
