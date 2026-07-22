import tempfile
import unittest
from pathlib import Path

from PIL import Image

from battery_v4_1.training_view import build_training_view


class TrainingViewTests(unittest.TestCase):
    def test_builds_view_and_excludes_test_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            out = Path(tmp) / "view"
            (root / "EXT/trainval/images").mkdir(parents=True)
            (root / "EXT/trainval/labels_det").mkdir(parents=True)
            (root / "EXT/trainval/splits").mkdir(parents=True)
            for sid in ("a", "b"):
                Image.new("RGB", (2, 2)).save(root / f"EXT/trainval/images/{sid}.jpg")
                (root / f"EXT/trainval/labels_det/{sid}.txt").write_text("", encoding="utf-8")
            (root / "EXT/trainval/splits/train.txt").write_text("a\n", encoding="utf-8")
            (root / "EXT/trainval/splits/val.txt").write_text("b\n", encoding="utf-8")

            build_training_view(root, out, "EXT", "det", copy=True)

            yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("names: [Damaged, Pollution]", yaml_text)
            self.assertNotIn("test:", yaml_text)
            self.assertTrue((out / "images/train/a.jpg").exists())
            self.assertTrue((out / "labels/val/b.txt").exists())

    def test_refuses_to_overwrite_existing_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            out = Path(tmp) / "view"
            root.mkdir()
            out.mkdir()
            with self.assertRaises(FileExistsError):
                build_training_view(root, out, "EXT", "det", copy=True)


if __name__ == "__main__":
    unittest.main()
