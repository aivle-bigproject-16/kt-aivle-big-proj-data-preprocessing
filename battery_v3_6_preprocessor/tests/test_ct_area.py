import unittest

from battery_v3_6.ct_area import porosity_area_bin
from battery_v3_6.geometry import convert_defect


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


if __name__ == "__main__":
    unittest.main()
