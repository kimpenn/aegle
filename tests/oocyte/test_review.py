import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte import detect_oocyte_batch, generate_review_pack


class TestOocyteReviewPack(unittest.TestCase):
    def test_generates_indexed_persisted_mask_pages_and_reference_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yy, xx = np.mgrid[:256, :256]
            uchl1 = np.full((256, 256), 100.0, dtype=np.float32)
            uchl1 += (
                12000.0
                * np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 22**2))
            ).astype(np.float32)
            image_path = root / "sample.ome.tiff"
            tifffile.imwrite(
                image_path,
                np.stack([uchl1, np.zeros_like(uchl1)]),
                ome=True,
                metadata={"axes": "CYX"},
            )
            manifest = root / "samples.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "sample",
                        "image_path": image_path,
                        "channel_index": 0,
                        "pixel_size_um": 0.5,
                    }
                ]
            ).to_csv(manifest, index=False)
            batch_dir = root / "output"
            detect_oocyte_batch(manifest, out_dir=batch_dir, jobs=1)
            references = root / "references.jsonl"
            records = [
                {
                    "sample_id": "sample",
                    "oocyte_id": 1,
                    "center": [128, 128],
                    "final_score": 0.9,
                    "quality": "high",
                },
                {
                    "sample_id": "sample",
                    "oocyte_id": 2,
                    "center": [10, 10],
                    "final_score": 0.8,
                    "quality": "high",
                },
            ]
            references.write_text("\n".join(json.dumps(row) for row in records) + "\n")

            result = generate_review_pack(
                batch_dir,
                references_path=references,
                columns=1,
                rows=1,
            )

            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.novel_count, 0)
            self.assertEqual(result.missed_reference_count, 1)
            accepted = pd.read_csv(result.artifact_paths["accepted_candidates"])
            missed = pd.read_csv(result.artifact_paths["missed_references"])
            self.assertEqual(int(accepted.iloc[0]["review_rank"]), 1)
            self.assertEqual(int(accepted.iloc[0]["panel_index"]), 1)
            self.assertEqual(int(missed.iloc[0]["reference_oocyte_id"]), 2)
            self.assertTrue((result.artifact_paths["accepted_pages"] / "page_01.png").is_file())
            self.assertTrue((result.artifact_paths["missed_pages"] / "page_01.png").is_file())


if __name__ == "__main__":
    unittest.main()
