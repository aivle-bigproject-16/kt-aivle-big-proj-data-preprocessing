import unittest

from battery_v4_0.ct_area import is_ct_pre_split_excluded, porosity_area_bin
from battery_v4_0.models import Sample
from battery_v4_0.geometry import convert_defect


class CTAreaTests(unittest.TestCase):
    def test_area_bin_boundaries(self):
        self.assertEqual(porosity_area_bin(0), "zero")
        self.assertEqual(porosity_area_bin(0.0009), "lt_0.1pct")
        self.assertEqual(porosity_area_bin(0.001), "0.1_1pct")
        self.assertEqual(porosity_area_bin(0.25), "25_40pct")
        self.assertEqual(porosity_area_bin(0.40), "40_50pct")
        self.assertEqual(porosity_area_bin(0.50), "ge_50pct")

    def test_converted_polygon_reports_roi_area_ratio(self):
        converted = convert_defect(0, [0, 0, 50, 0, 50, 50, 0, 50], 100, 100)
        self.assertTrue(converted.seg_valid)
        self.assertEqual(converted.polygon_count, 1)
        self.assertAlmostEqual(converted.polygon_area_ratio, 0.25)
        self.assertEqual(converted.bbox_max_ratio, 0.25)

    def test_bbox_ratio_uses_each_serialized_component_not_polygon_area_sum(self):
        converted = convert_defect(
            0,
            [0, 0, 40, 40, 0, 40, 40, 0, 60, 60, 100, 100, 60, 100, 100, 60],
            100,
            100,
        )
        self.assertEqual(converted.polygon_count, 2)
        self.assertLess(converted.polygon_area_ratio, 0.25)
        self.assertEqual(converted.bbox_max_ratio, 0.25)

    def test_bbox_rounds_normalized_vertices_before_taking_bounds(self):
        converted = convert_defect(
            0,
            [0.0000006, 0, 50.0000004, 0, 50.0000004, 50, 0.0000006, 50],
            100,
            100,
        )
        self.assertEqual(converted.bbox_max_ratio, 0.49999999 * 0.5)
        self.assertLess(converted.bbox_max_ratio, 0.25)

    def test_pre_split_exclusion_boundary_is_inclusive_and_ct_only(self):
        below = Sample("below", modality="CT", porosity_bbox_max_ratio=0.24999999)
        boundary = Sample("boundary", modality="CT", porosity_bbox_max_ratio=0.25)
        rgb = Sample("rgb", modality="EXT", porosity_bbox_max_ratio=0.90)
        self.assertFalse(is_ct_pre_split_excluded(below))
        self.assertTrue(is_ct_pre_split_excluded(boundary))
        self.assertFalse(is_ct_pre_split_excluded(rgb))

    def test_non_finite_bbox_ratio_is_rejected_by_split_policy(self):
        from battery_v4_0.selection import apply_pre_split_policy

        invalid = Sample("invalid", modality="CT", porosity_bbox_max_ratio=float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            apply_pre_split_policy([invalid])


if __name__ == "__main__":
    unittest.main()
