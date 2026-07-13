import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte import detect_oocyte_batch, load_sample_manifest


class TestOocyteBatch(unittest.TestCase):
    def _write_ome(self, path):
        yy, xx = np.mgrid[:256, :256]
        uchl1 = np.full((256, 256), 100.0, dtype=np.float32)
        uchl1 += (
            12000.0
            * np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 22**2))
        ).astype(np.float32)
        channels = np.stack([uchl1, np.zeros_like(uchl1)])
        tifffile.imwrite(path, channels, ome=True, metadata={"axes": "CYX"})

    def test_isolates_sample_failures_and_records_disabled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "valid.ome.tiff"
            self._write_ome(image_path)
            manifest = root / "samples.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "valid",
                        "image_path": image_path,
                        "channel_index": 0,
                        "pixel_size_um": 0.5,
                        "enabled": True,
                    },
                    {
                        "sample_id": "missing",
                        "image_path": root / "missing.ome.tiff",
                        "channel_index": 0,
                        "pixel_size_um": 0.5,
                        "enabled": True,
                    },
                    {
                        "sample_id": "disabled",
                        "image_path": root / "unused.ome.tiff",
                        "channel_index": None,
                        "pixel_size_um": 0.5,
                        "enabled": False,
                    },
                ]
            ).to_csv(manifest, index=False)

            result = detect_oocyte_batch(
                manifest,
                out_dir=root / "output",
                jobs=1,
                continue_on_error=True,
            )

            statuses = dict(zip(result.summary["sample_id"], result.summary["status"]))
            self.assertEqual(
                statuses,
                {"valid": "complete", "missing": "failed", "disabled": "skipped"},
            )
            self.assertEqual(result.failed_count, 1)
            self.assertTrue(result.artifact_paths["batch_summary_csv"].is_file())
            self.assertTrue(result.artifact_paths["spatial_qc_atlas"].is_file())
            self.assertTrue(result.artifact_paths["spatial_qc_index"].is_file())
            self.assertTrue((root / "output/valid/oocyte_labels.ome.tiff").is_file())

            resumed = detect_oocyte_batch(
                manifest,
                out_dir=root / "output",
                jobs=1,
                continue_on_error=True,
            )
            valid = resumed.summary[resumed.summary["sample_id"] == "valid"].iloc[0]
            self.assertTrue(bool(valid["resumed"]))

    def test_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "samples.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "duplicate",
                        "image_path": "first.ome.tiff",
                        "channel_index": 0,
                        "pixel_size_um": 0.5,
                    },
                    {
                        "sample_id": "duplicate",
                        "image_path": "second.ome.tiff",
                        "channel_index": 0,
                        "pixel_size_um": 0.5,
                    },
                ]
            ).to_csv(manifest, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                load_sample_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
