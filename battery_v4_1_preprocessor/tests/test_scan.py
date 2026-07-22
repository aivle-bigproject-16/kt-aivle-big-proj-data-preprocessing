import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from battery_v4_1.scan import scan_dataset


def write_pair(root: Path, stem: str, payload: dict, color: tuple[int, int, int]) -> None:
    image_dir = root / "Training/01.원천데이터/group"
    json_dir = root / "Training/02.라벨링데이터/group"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color).save(image_dir / f"{stem}.png")
    payload["image_info"]["file_name"] = f"{stem}.png"
    payload["image_info"]["width"] = 20
    payload["image_info"]["height"] = 10
    (json_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


class ScanTests(unittest.TestCase):
    def test_filename_parse_failure_is_written_to_matching_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root, "BAD_cell_pouch_0007_1", {
                "data_info": {"battery_ids": "7", "data_type": "RGB", "application": "home"},
                "image_info": {"is_normal": True},
                "defects": [],
            }, (1, 2, 3))

            result = scan_dataset(root)

            self.assertEqual(len(result.valid_samples), 0)
            self.assertTrue(any(row["reason"] == "filename_parse_error" for row in result.matching_issues))

    def test_reads_is_normal_from_image_info_and_deduplicates_defects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            defect = {"name": "Pollution", "points": [[1, 1], [5, 1], [5, 5]]}
            write_pair(root, "RGB_cell_pouch_0007_1", {
                "data_info": {"battery_ids": "7", "data_type": "RGB", "application": "home"},
                "image_info": {"is_normal": False},
                "defects": [defect, defect],
            }, (10, 20, 30))
            write_pair(root, "RGB_cell_pouch_0007_2", {
                "data_info": {"battery_ids": "7", "data_type": "RGB", "application": "home"},
                "image_info": {"is_normal": True},
                "defects": [],
            }, (11, 20, 30))

            result = scan_dataset(root)

            self.assertEqual(len(result.valid_samples), 2)
            self.assertEqual(sum(sample.is_normal_interpreted for sample in result.valid_samples), 1)
            self.assertEqual(len(result.valid_samples[0].det_lines), 1)
            self.assertTrue(any(row["issue"] == "duplicate_defect_in_array" for row in result.json_anomalies))

    def test_cross_id_identical_pixels_excludes_whole_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for battery_id in ("1", "2"):
                stem = f"RGB_cell_pouch_{battery_id}_1"
                write_pair(root, stem, {
                    "data_info": {"battery_ids": battery_id, "data_type": "RGB", "application": "home"},
                    "image_info": {"is_normal": True},
                    "defects": [],
                }, (50, 60, 70))

            result = scan_dataset(root)

            self.assertEqual(result.valid_samples, [])
            self.assertEqual({row["action"] for row in result.pixel_duplicates}, {"exclude_cross_id_group"})


if __name__ == "__main__":
    unittest.main()
