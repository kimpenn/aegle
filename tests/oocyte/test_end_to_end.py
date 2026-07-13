import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from aegle.oocyte import DONOR13_V6, detect_oocytes, load_candidate_mask


class TestOocyteEndToEnd(unittest.TestCase):
    def _write_ome(self, path, uchl1):
        channels = np.stack([uchl1, np.zeros_like(uchl1)])
        tifffile.imwrite(path, channels, ome=True, metadata={"axes": "CYX"})

    def test_writes_standalone_single_sample_deliverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yy, xx = np.mgrid[:512, :512]
            uchl1 = np.full((512, 512), 100.0, dtype=np.float32)
            uchl1 += (
                12000.0
                * np.exp(-((yy - 256) ** 2 + (xx - 256) ** 2) / (2 * 22**2))
            ).astype(np.float32)
            image_path = root / "synthetic.ome.tiff"
            self._write_ome(image_path, uchl1)

            result = detect_oocytes(
                image_path,
                sample_id="synthetic",
                out_dir=root / "output",
                config=DONOR13_V6,
                channel_index=0,
            )

            self.assertEqual(len(result.coarse_candidates), 1)
            self.assertEqual(len(result.candidates), 1)
            self.assertTrue(bool(result.candidates.iloc[0]["accepted"]))
            mask_path = root / "output" / result.candidates.iloc[0]["mask_path"]
            persisted = load_candidate_mask(mask_path)
            self.assertGreater(int(persisted.mask.sum()), 2000)
            labels = tifffile.imread(result.artifact_paths["labels"])
            self.assertEqual(labels.shape, (512, 512))
            self.assertEqual(int(labels.max()), 1)
            summary = json.loads(result.artifact_paths["summary"].read_text())
            self.assertEqual(summary["accepted_candidate_count"], 1)
            self.assertTrue(result.artifact_paths["overview"].is_file())
            self.assertTrue(result.artifact_paths["duplicate_suspects"].is_file())

    def test_blank_channel_writes_zero_label_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "blank.ome.tiff"
            self._write_ome(
                image_path,
                np.full((256, 256), 100.0, dtype=np.float32),
            )

            result = detect_oocytes(
                image_path,
                sample_id="blank",
                out_dir=root / "output",
                config=DONOR13_V6,
                channel_index=0,
            )

            self.assertTrue(result.candidates.empty)
            self.assertIn("accepted", result.candidates.columns)
            labels = tifffile.imread(result.artifact_paths["labels"])
            self.assertFalse(labels.any())


if __name__ == "__main__":
    unittest.main()
