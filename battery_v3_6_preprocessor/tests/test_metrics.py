import unittest

from battery_v3_6.metrics import (
    group_selected_samples,
    sample_metrics,
    selected_samples,
    split_fold_group_name,
)
from battery_v3_6.models import IdStats, Sample


class MetricTests(unittest.TestCase):
    def test_sample_metrics_use_one_canonical_definition(self):
        samples = [
            Sample(sample_id="defect-2", det_lines=["0 box", "0 box"]),
            Sample(sample_id="defect-1", det_lines=["0 box"]),
            Sample(sample_id="normal"),
        ]

        metrics = sample_metrics(samples)

        self.assertEqual(metrics.images, 3)
        self.assertEqual(metrics.defect_images, 2)
        self.assertEqual(metrics.annotations, 3)
        self.assertAlmostEqual(metrics.defect_image_ratio, 2 / 3)
        self.assertAlmostEqual(metrics.annotations_per_image, 1.0)
        self.assertAlmostEqual(metrics.normal_ratio, 1 / 3)

    def test_selected_samples_and_groups_share_the_same_membership_source(self):
        fold_zero = Sample(sample_id="fold-zero", modality="CT", battery_id="1")
        fold_one = Sample(sample_id="fold-one", modality="CT", battery_id="2")
        ids = [
            IdStats("CT", "1", "산업", [fold_zero], [fold_zero], "development", "0"),
            IdStats("CT", "2", "산업", [fold_one], [fold_one], "development", "1"),
        ]

        self.assertEqual(
            [sample.sample_id for sample in selected_samples(ids)],
            ["fold-zero", "fold-one"],
        )
        groups = group_selected_samples(
            ids,
            split_fold_group_name,
        )
        self.assertEqual(sorted(groups), ["CT_fold_0", "CT_fold_1"])
        self.assertEqual(groups["CT_fold_0"][0].sample_id, "fold-zero")
        self.assertEqual(groups["CT_fold_1"][0].sample_id, "fold-one")


if __name__ == "__main__":
    unittest.main()

