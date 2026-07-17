import unittest

from battery_v3_6.models import Sample
from battery_v3_6.selection import assign_dataset


def sample(sample_id: str, modality: str, battery_id: str, axis: str, defect: bool) -> Sample:
    return Sample(
        sample_id=sample_id,
        modality=modality,
        battery_id=battery_id,
        axis=axis,
        application="가전",
        det_lines=["0 0.5 0.5 0.1 0.1"] if defect else [],
        seg_lines=["0 0.1 0.1 0.2 0.1 0.1 0.2"] if defect else [],
        class_names=["porosity" if modality == "CT" else "Damaged"] if defect else [],
        included_det=True,
        included_seg=True,
        pixel_hash=sample_id,
    )


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = []
        for battery in range(1, 48):
            for axis in "xyz":
                for frame in range(100):
                    sid = f"CT_{battery}_{axis}_{frame}"
                    cls.samples.append(sample(sid, "CT", str(battery), axis, frame % 5 == 0))
        for battery in range(1, 161):
            for frame in range(10):
                sid = f"EXT_{battery}_{frame}"
                cls.samples.append(sample(sid, "EXT", str(battery), "", frame < 2))

    def test_assigns_exact_id_quotas_and_axis_samples(self):
        result = assign_dataset(self.samples, seed=42)
        ct = [stats for stats in result.ids if stats.modality == "CT"]
        ext = [stats for stats in result.ids if stats.modality == "EXT"]
        self.assertEqual(sum(stats.split_role == "test" for stats in ct), 7)
        self.assertEqual(sum(stats.split_role == "development" for stats in ct), 40)
        self.assertEqual(sorted(sum(stats.fold_id == str(i) for stats in ct) for i in range(5)), [8] * 5)
        self.assertTrue(all(len(stats.selected_samples) == 300 for stats in ct if stats.split_role == "development"))
        self.assertEqual(
            (sum(stats.split_role == "train" for stats in ext), sum(stats.split_role == "val" for stats in ext), sum(stats.split_role == "test" for stats in ext)),
            (128, 16, 16),
        )
        self.assertEqual(result.warnings, [])

        repeated = assign_dataset(self.samples, seed=42)
        self.assertEqual(
            {stats.battery_id: stats.fold_id for stats in ct if stats.split_role == "development"},
            {stats.battery_id: stats.fold_id for stats in repeated.ids if stats.modality == "CT" and stats.split_role == "development"},
        )

    def test_ct_development_uses_every_valid_axis_image(self):
        samples = list(self.samples)
        samples.append(sample("CT_8_x_100", "CT", "8", "x", True))

        result = assign_dataset(
            samples,
            seed=42,
            locked_tests={"CT": {str(value) for value in range(1, 8)}},
        )

        battery_8 = next(
            stats
            for stats in result.ids
            if stats.modality == "CT" and stats.battery_id == "8"
        )
        self.assertEqual(battery_8.split_role, "development")
        self.assertEqual(len(battery_8.samples), 301)
        self.assertEqual(len(battery_8.selected_samples), 301)
        self.assertEqual(
            {axis: sum(item.axis == axis for item in battery_8.selected_samples) for axis in "xyz"},
            {"x": 101, "y": 100, "z": 100},
        )

    def test_ct_test_matches_development_axis_ratios(self):
        ct_samples = []
        axis_by_range = ((range(1, 8), "x"), (range(8, 28), "y"), (range(28, 48), "z"))
        for batteries, axis in axis_by_range:
            for battery in batteries:
                ct_samples.append(sample(f"CT_{battery}_{axis}_0", "CT", str(battery), axis, True))
        ext_samples = [item for item in self.samples if item.modality == "EXT"]

        result = assign_dataset(ct_samples + ext_samples, seed=42)

        ct_test = [stats for stats in result.ids if stats.modality == "CT" and stats.split_role == "test"]
        test_axis_id_counts = {
            axis: sum(any(item.axis == axis for item in stats.samples) for stats in ct_test)
            for axis in "xyz"
        }
        self.assertEqual(test_axis_id_counts, {"x": 1, "y": 3, "z": 3})

    def test_ct_area_concentration_is_review_not_gate(self):
        # 한 개발 ID가 40_50pct 면적 구간을 독점하면(§9.7) 면적 쏠림 경고가 나야 한다.
        # 그 경고는 §17.3 검토 경고(review_warnings)여야 하고, §17.2 하드 게이트(warnings)에는 없어야 한다.
        ct_samples = []
        for battery in range(1, 48):
            for axis in "xyz":
                for frame in range(20):
                    item = sample(f"CT_{battery}_{axis}_{frame}", "CT", str(battery), axis, frame % 5 == 0)
                    if str(battery) == "20":
                        item.porosity_area_sum_ratio = 0.45  # 40_50pct 구간
                    ct_samples.append(item)
        ext_samples = [item for item in self.samples if item.modality == "EXT"]

        result = assign_dataset(
            ct_samples + ext_samples,
            seed=42,
            locked_tests={"CT": {str(value) for value in range(1, 8)}},
        )

        self.assertFalse(any("CT_area_bin" in warning for warning in result.warnings))
        self.assertTrue(any("CT_area_bin[40_50pct]" in warning for warning in result.review_warnings))

    def test_ct_test_minimizes_defect_ratio_gap_after_axis_balance(self):
        ct_samples = []
        for battery in range(1, 48):
            for axis in "xyz":
                ct_samples.append(
                    sample(
                        f"CT_{battery}_{axis}_0",
                        "CT",
                        str(battery),
                        axis,
                        battery <= 7,
                    )
                )
        ext_samples = [item for item in self.samples if item.modality == "EXT"]

        result = assign_dataset(ct_samples + ext_samples, seed=42)

        ct_test = [stats for stats in result.ids if stats.modality == "CT" and stats.split_role == "test"]
        defect_bearing_test_ids = sum(bool(stats.selected_samples[0].det_lines) for stats in ct_test)
        self.assertEqual(defect_bearing_test_ids, 1)


if __name__ == "__main__":
    unittest.main()

