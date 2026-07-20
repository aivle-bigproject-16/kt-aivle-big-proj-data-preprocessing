import tempfile
import unittest
from pathlib import Path

from PIL import Image

from battery_v3_8.deterministic import (
    largest_remainder,
    normalize_battery_id,
    order_key,
    quantile_bins,
    sample_id_for,
    stratified_sample,
)
from battery_v3_8.geometry import convert_defect
from battery_v3_8.models import Sample
from battery_v3_8.parsing import (
    canonical_class,
    deduplicate_defects,
    normalize_application,
    parse_filename,
)


class DeterministicTests(unittest.TestCase):
    def test_order_key_and_sample_id_follow_contract(self):
        self.assertEqual(
            order_key(42, "rgb160", "907"),
            "10b0421e94775aa0726aaeff20445d7cf83a040e",
        )
        self.assertEqual(
            sample_id_for("EXT", "RGB_cell_A_0907_1", "Training/a.jpg"),
            "EXT__RGB_cell_A_0907_1__a49c03c9",
        )

    def test_quantile_bins_are_rank_based_and_stable(self):
        self.assertEqual(
            quantile_bins({"3": 0.2, "1": 0.1, "2": 0.1}, bins=2),
            {"1": 0, "2": 0, "3": 1},
        )

    def test_largest_remainder_respects_caps(self):
        self.assertEqual(
            largest_remainder(5, {"a": 8, "b": 2}, {"a": 3, "b": 2}),
            {"a": 3, "b": 2},
        )
        with self.assertRaises(ValueError):
            largest_remainder(6, {"a": 8, "b": 2}, {"a": 3, "b": 2})

    def test_stratified_sample_preserves_positive_count(self):
        items = [
            Sample.stub(f"s{i}", is_defect=i < 3) for i in range(10)
        ]
        chosen = stratified_sample(items, 5, 42, "EXT:1:train")
        self.assertEqual(len(chosen), 5)
        self.assertEqual(sum(s.is_defect for s in chosen), 2)
        self.assertEqual([s.sample_id for s in chosen], sorted(s.sample_id for s in chosen))


class ParsingTests(unittest.TestCase):
    def test_filename_grammar_and_id_normalization(self):
        ct = parse_filename("CT_cell_pouch_0042_x_0001.jpg")
        ext = parse_filename("RGB_cell_prismatic_0907_123.png")
        self.assertEqual((ct.modality, ct.battery_id, ct.axis), ("CT", "42", "x"))
        self.assertEqual((ext.modality, ext.battery_id, ext.axis), ("EXT", "907", ""))
        self.assertEqual(normalize_battery_id("0000"), "0")
        self.assertIsNone(parse_filename("RGB_module_x_1_1.jpg"))

    def test_class_and_application_aliases(self):
        self.assertEqual(canonical_class(" contamination ", "EXT"), "Pollution")
        self.assertEqual(canonical_class("Damage", "EXT"), "Damaged")
        self.assertEqual(canonical_class("porosity", "CT"), "porosity")
        self.assertIsNone(canonical_class("battery_outline", "CT"))
        self.assertEqual(normalize_application(" EV "), "산업")
        self.assertEqual(normalize_application(""), "빈값")

    def test_defect_array_dedup_keeps_first_item(self):
        defects = [
            {"name": "Pollution", "points": [[1, 2], [3, 4], [5, 6]], "x": 1},
            {"name": "Pollution", "points": [[1, 2], [3, 4], [5, 6]], "x": 2},
        ]
        kept, removed = deduplicate_defects(defects)
        self.assertEqual(removed, 1)
        self.assertEqual(kept[0]["x"], 1)


class GeometryTests(unittest.TestCase):
    def test_positive_polygon_produces_det_and_seg_lines(self):
        result = convert_defect(0, [[10, 10], [30, 10], [30, 20], [10, 20]], 100, 100)
        self.assertTrue(result.det_valid)
        self.assertTrue(result.seg_valid)
        self.assertEqual(result.det_lines, ["0 0.20000000 0.15000000 0.20000000 0.10000000"])
        self.assertEqual(result.seg_lines[0], "0 0.10000000 0.10000000 0.30000000 0.10000000 0.30000000 0.20000000 0.10000000 0.20000000")

    def test_linear_scratch_uses_min_extent_for_detection_only(self):
        result = convert_defect(1, [[10, 10], [20, 10], [30, 10]], 100, 100)
        self.assertTrue(result.det_valid)
        self.assertFalse(result.seg_valid)
        self.assertTrue(result.min_extent_applied)
        self.assertEqual(result.det_lines, ["1 0.20000000 0.10000000 0.20000000 0.01000000"])

    def test_outside_polygon_is_invalid(self):
        result = convert_defect(0, [[-5, -5], [-2, -5], [-2, -2]], 100, 100)
        self.assertFalse(result.det_valid)
        self.assertFalse(result.seg_valid)


class PixelHashTests(unittest.TestCase):
    def test_hash_uses_decoded_rgb_pixels_not_file_bytes(self):
        from battery_v3_8.scan import pixel_sha1

        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.png"
            b = Path(tmp) / "b.bmp"
            image = Image.new("L", (2, 2), 128)
            image.save(a)
            image.save(b)
            self.assertEqual(pixel_sha1(a), pixel_sha1(b))


if __name__ == "__main__":
    unittest.main()

