import unittest
from collections import Counter
from unittest import mock

from battery_v3_8 import selection as sel
from battery_v3_8.models import IdStats, Sample
from battery_v3_8.selection import assign_dataset


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


def build_full_fixture() -> list[Sample]:
    """47 CT IDs and 160 RGB IDs, the smallest set assign_dataset accepts."""
    samples = []
    for battery in range(1, 48):
        for axis in "xyz":
            for frame in range(100):
                samples.append(sample(f"CT_{battery}_{axis}_{frame}", "CT", str(battery), axis, frame % 5 == 0))
    for battery in range(1, 161):
        for frame in range(10):
            samples.append(sample(f"EXT_{battery}_{frame}", "EXT", str(battery), "", frame < 2))
    return samples


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = build_full_fixture()

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

    def test_ct_bbox_ge25_is_excluded_before_split_and_below_boundary_remains(self):
        samples = list(self.samples)
        excluded = sample("CT_8_x_large", "CT", "8", "x", True)
        excluded.porosity_bbox_max_ratio = 0.25
        retained = sample("CT_8_x_below", "CT", "8", "x", True)
        retained.porosity_bbox_max_ratio = 0.24999999
        samples.extend([excluded, retained])

        result = assign_dataset(
            samples,
            seed=42,
            locked_tests={"CT": {str(value) for value in range(1, 8)}},
        )
        battery_8 = next(
            stats for stats in result.ids
            if stats.modality == "CT" and stats.battery_id == "8"
        )
        self.assertNotIn(excluded, battery_8.samples)
        self.assertNotIn(excluded, battery_8.selected_samples)
        self.assertFalse(excluded.pre_split_eligible)
        self.assertEqual(excluded.pre_split_exclusion_reason, "ct_porosity_bbox_max_ratio_ge_0.25")
        self.assertIn(retained, battery_8.selected_samples)
        self.assertTrue(retained.pre_split_eligible)

    def test_ct_bbox_policy_is_applied_to_locked_test_before_metrics(self):
        samples = list(self.samples)
        excluded = sample("CT_1_x_large", "CT", "1", "x", True)
        excluded.porosity_bbox_max_ratio = 0.40
        samples.append(excluded)
        result = assign_dataset(
            samples,
            seed=42,
            locked_tests={"CT": {str(value) for value in range(1, 8)}},
        )
        test_id = next(
            stats for stats in result.ids
            if stats.modality == "CT" and stats.battery_id == "1"
        )
        self.assertEqual(test_id.split_role, "test")
        self.assertNotIn(excluded, test_id.samples)
        self.assertNotIn(excluded, result.selected_samples)

    def test_ct_id_losing_every_image_is_dropped_and_structure_adapts(self):
        """v3.8 sizes the split from the surviving IDs instead of demanding 47.

        An ID whose images are all excluded by the image rule is removed by the
        contamination gate, and the remaining 46 IDs form five folds of seven
        with the rest going to Test.
        """
        samples = list(self.samples)
        for item in samples:
            if item.modality == "CT" and item.battery_id == "47":
                item.porosity_bbox_max_ratio = 0.25
        result = assign_dataset(samples, seed=42)
        ct_ids = {stats.battery_id for stats in result.ids if stats.modality == "CT"}
        self.assertNotIn("47", ct_ids)
        self.assertEqual(len(ct_ids), 46)
        # An ID that loses every image is already absent from the split population,
        # so the image rule removes it before the ID gate can see it.
        folds = Counter(
            stats.fold_id for stats in result.ids
            if stats.modality == "CT" and stats.split_role == "development"
        )
        self.assertEqual(sorted(folds.values()), [7, 7, 7, 7, 7])
        test_ids = [stats for stats in result.ids if stats.modality == "CT" and stats.split_role == "test"]
        self.assertEqual(len(test_ids), 11)

    def test_ct_id_density_gate_ignores_a_degenerate_interquartile_range(self):
        """With few defect-bearing IDs, Q3 equals Q1 and no outlier is defined.

        Firing the fence there would drop every ID that carries an annotation,
        so the gate has to stand down instead.
        """
        result = assign_dataset(list(self.samples), seed=42)
        self.assertEqual(
            [row for row in result.ct_id_gate_rows if row["gate"] == "density_outlier"],
            [],
        )

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


class CtBalanceWiringTests(unittest.TestCase):
    """Guard the CT balance behaviour that a v3.8 dry-run had to uncover twice.

    Every assertion here failed silently before: the suite passed while the
    pipeline produced a fold deviation at the gate boundary and a Test split
    2.78x denser than development. A regression in this wiring would otherwise
    only surface after a 37 minute dry-run.
    """

    @staticmethod
    def _stats(battery_id, images, annotations_per_image, defect_ratio):
        samples = []
        defect_count = round(images * defect_ratio)
        per_defect = annotations_per_image * images / defect_count if defect_count else 0
        for index in range(images):
            defect = index < defect_count
            count = round(per_defect) if defect else 0
            samples.append(Sample(
                sample_id=f"CT_{battery_id}_x_{index}",
                modality="CT", battery_id=battery_id, axis="x", application="가전",
                det_lines=["x"] * count, included_det=True, included_seg=True,
            ))
        stats = IdStats("CT", battery_id, "가전", samples)
        stats.selected_samples = list(samples)
        return stats

    def _groups(self):
        # Group 0 carries the dense ID plus two mid-density members; group 1 is
        # empty of annotations. Trading a mid member for an empty one dilutes
        # group 0, but the two sit in different defect-ratio strata, so the trade
        # is only reachable once the stratum constraint is lifted.
        layout = {
            "0": [("dense", 100, 2.0, 0.9), ("m1", 100, 0.5, 0.5), ("m2", 100, 0.5, 0.5)],
            "1": [("z1", 100, 0.0, 0.0), ("z2", 100, 0.0, 0.0), ("z3", 100, 0.0, 0.0)],
            "2": [("m3", 100, 0.3, 0.3), ("m4", 100, 0.3, 0.3), ("m5", 100, 0.3, 0.3)],
        }
        return {
            name: [self._stats(bid, imgs, apd, ratio) for bid, imgs, apd, ratio in members]
            for name, members in layout.items()
        }

    @staticmethod
    def _worst(groups, target):
        from battery_v3_8.metrics import sample_metrics, selected_samples as collect
        return max(
            abs(sample_metrics(collect(members, name, None)).annotations_per_image / target - 1)
            for name, members in groups.items()
        )

    def test_lifting_the_stratum_constraint_lets_the_swap_reduce_density_spread(self):
        from battery_v3_8.metrics import sample_metrics, selected_samples as collect

        stratum = lambda stats: round(sel._id_selected_ratio(stats) * 10)
        constrained = self._groups()
        target = sample_metrics(
            collect([stats for members in constrained.values() for stats in members])
        ).annotations_per_image

        sel._swap_density(constrained, stratum, target, enforce_stratum=True)
        relaxed = self._groups()
        sel._swap_density(relaxed, stratum, target, enforce_stratum=False)

        self.assertLess(self._worst(relaxed, target), self._worst(constrained, target))

    def test_only_the_ct_swap_lifts_the_stratum_constraint(self):
        """CT must reach the swap with enforce_stratum disabled, RGB must not.

        Configuring this at the call site once let an offline check measure the
        constrained behaviour while the pipeline ran without it, so the wiring
        itself is asserted rather than only its effect. One assign_dataset run
        covers both paths because the run is expensive.
        """
        with mock.patch.object(sel, "_swap_density", wraps=sel._swap_density) as spy:
            assign_dataset(build_full_fixture(), seed=42)
        lifted = [call for call in spy.call_args_list if call.kwargs.get("enforce_stratum") is False]
        kept = [call for call in spy.call_args_list if call.kwargs.get("enforce_stratum") is not False]
        self.assertEqual(len(lifted), 1, "CT fold swap should run once with the constraint lifted")
        self.assertTrue(kept, "RGB density swap should still enforce the stratum")

    def test_test_objective_prefers_the_split_matching_development_density(self):
        """A lexicographic tuple never reached the density term; a sum does."""
        balanced = [self._stats("t1", 100, 0.2, 0.2)]
        dense = [self._stats("t2", 100, 2.0, 0.2)]
        development = [self._stats(f"d{i}", 100, 0.2, 0.2) for i in range(4)]
        self.assertLess(
            sel._ct_test_objective(balanced, development),
            sel._ct_test_objective(dense, development),
        )


if __name__ == "__main__":
    unittest.main()
