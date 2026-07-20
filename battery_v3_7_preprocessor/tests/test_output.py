import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from battery_v3_7.models import IdStats, Sample
from battery_v3_7.output import write_dataset
from battery_v3_7.selection import SelectionResult


class OutputTests(unittest.TestCase):
    def _sample(self, base: Path, sample_id: str, det_lines: list[str], seg_lines: list[str], *, normal: bool = False) -> Sample:
        source_image = base / f"{sample_id}.png"
        source_json = base / f"{sample_id}.json"
        Image.new("RGB", (4, 4), (1, 2, 3)).save(source_image)
        json_bytes = b'{}'
        source_json.write_bytes(json_bytes)
        return Sample(
            sample_id=sample_id,
            modality="EXT",
            battery_id="1",
            image_path=source_image,
            json_path=source_json,
            json_relative_posix=f"labels/{sample_id}.json",
            json_sha256=hashlib.sha256(json_bytes).hexdigest(),
            included_det=True,
            included_seg=True,
            is_normal_interpreted=normal,
            det_lines=det_lines,
            seg_lines=seg_lines,
            split_role="train",
            selected=True,
        )

    def test_writes_dataset_and_preserves_json_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_image = base / "source.png"
            source_json = base / "source.json"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(source_image)
            json_bytes = b'{"exact":  1}\r\n'
            source_json.write_bytes(json_bytes)
            sample = Sample(
                sample_id="EXT__one__12345678",
                modality="EXT",
                battery_id="1",
                image_path=source_image,
                json_path=source_json,
                json_relative_posix="Training/02.라벨링데이터/g/source.json",
                json_sha256=hashlib.sha256(json_bytes).hexdigest(),
                included_det=True,
                included_seg=True,
                is_normal_interpreted=True,
                split_role="train",
                selected=True,
            )
            stats = IdStats("EXT", "1", "가전", [sample], [sample], "train", "")
            output = base / "dataset"

            write_dataset(output, SelectionResult([stats]))

            copied = output / "EXT/trainval/labels_json/Training/02.라벨링데이터/g/source.json"
            self.assertEqual(copied.read_bytes(), json_bytes)
            self.assertTrue((output / "EXT/trainval/labels_det/EXT__one__12345678.txt").exists())
            self.assertTrue((output / "battery_v3_7/cli.py").exists())
            self.assertTrue((output / "battery_EXT_v3_trainval.zip").exists())

    def test_rejects_invalid_class_id_during_reverse_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = self._sample(
                base,
                "EXT__invalid_class__00000001",
                ["9 0.50000000 0.50000000 0.20000000 0.20000000"],
                ["9 0.10000000 0.10000000 0.20000000 0.10000000 0.10000000 0.20000000"],
            )
            stats = IdStats("EXT", "1", "가전", [sample], [sample], "train", "")

            with self.assertRaisesRegex(RuntimeError, "class ID"):
                write_dataset(base / "dataset", SelectionResult([stats]))

    def test_rejects_malformed_segmentation_coordinate_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = self._sample(
                base,
                "EXT__bad_seg__00000002",
                ["0 0.50000000 0.50000000 0.20000000 0.20000000"],
                ["0 0.10000000 0.10000000 0.20000000 0.10000000 0.10000000 0.20000000 0.30000000"],
            )
            stats = IdStats("EXT", "1", "가전", [sample], [sample], "train", "")

            with self.assertRaisesRegex(RuntimeError, "segmentation coordinate pairs"):
                write_dataset(base / "dataset", SelectionResult([stats]))

    def test_rejects_nonempty_label_for_explicit_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sample = self._sample(
                base,
                "EXT__normal__00000003",
                ["0 0.50000000 0.50000000 0.20000000 0.20000000"],
                ["0 0.10000000 0.10000000 0.20000000 0.10000000 0.10000000 0.20000000"],
                normal=True,
            )
            stats = IdStats("EXT", "1", "가전", [sample], [sample], "train", "")

            with self.assertRaisesRegex(RuntimeError, "normal image label"):
                write_dataset(base / "dataset", SelectionResult([stats]))


if __name__ == "__main__":
    unittest.main()

