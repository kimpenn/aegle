import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte import DONOR13_V6
from aegle.oocyte.io import save_candidate_mask
from aegle.oocyte.models import BoundingBox, SegmentationMetrics
from aegle.oocyte.report import (
    _raw_mask_thumbnail,
    algorithm_document_html,
    generate_html_reports,
)


class TestHtmlReport(unittest.TestCase):
    def test_algorithm_document_contains_inline_diagrams_and_contract(self):
        document = algorithm_document_html()
        self.assertGreaterEqual(document.count("<svg"), 2)
        self.assertIn("Secondary rescue", document)
        self.assertIn("DeepCell", document)
        self.assertIn("donor13_v6_rescue_v1", document)

    def test_generates_sample_page_from_persisted_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_id = "synthetic-1"
            sample_dir = root / sample_id
            sample_dir.mkdir()
            image = np.full((2, 401, 401), 100, dtype=np.uint16)
            yy, xx = np.ogrid[:401, :401]
            image[0, (yy - 200) ** 2 + (xx - 200) ** 2 <= 45**2] = 12000
            image_path = root / "source.ome.tiff"
            tifffile.imwrite(image_path, image, metadata={"axes": "CYX"})
            mask = (yy - 200) ** 2 + (xx - 200) ** 2 <= 45**2
            bbox = BoundingBox(150, 150, 251, 251)
            cropped = np.asarray(mask[150:251, 150:251], dtype=np.bool_)
            metrics = SegmentationMetrics(
                threshold_method="triangle",
                base_threshold=500.0,
                annulus_floor=100.0,
                threshold=500.0,
                selection_mode="center_component",
                area_px=int(cropped.sum()),
                equivalent_diameter_um=45.0,
                major_axis_um=45.0,
                minor_axis_um=45.0,
                eccentricity=0.0,
                solidity=0.99,
                circularity=0.95,
                centroid_y_px=50.0,
                centroid_x_px=50.0,
                centroid_offset_px=0.0,
                mean_intensity=12000.0,
                max_intensity=12000.0,
            )
            masks_dir = sample_dir / "masks"
            save_candidate_mask(
                masks_dir / "det_0000.npz",
                mask=cropped,
                bbox=bbox,
                image_shape_yx=(401, 401),
                sample_id=sample_id,
                candidate_id="det_0000",
                profile_name=DONOR13_V6.profile_name,
                profile_fingerprint=DONOR13_V6.fingerprint(),
                metrics=metrics,
            )
            pd.DataFrame(
                [
                    {
                        "detector_component_id": "det_0000",
                        "accepted": True,
                        "detector_score": 0.91,
                        "center_x": 200,
                        "center_y": 200,
                        "local_equivalent_diameter_um": 45.0,
                        "local_circularity": 0.95,
                        "local_solidity": 0.99,
                        "mask_path": "masks/det_0000.npz",
                    }
                ]
            ).to_csv(sample_dir / "candidates.csv", index=False)
            (sample_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "source_image": str(image_path),
                        "resolved_channel_index": 0,
                    }
                )
            )
            (sample_dir / "summary.json").write_text(
                json.dumps({"image_shape_yx": [401, 401]})
            )
            pd.DataFrame([{"sample_id": sample_id, "status": "complete"}]).to_csv(
                root / "batch_summary.csv", index=False
            )

            result = generate_html_reports(root)

            page = result.sample_pages[sample_id]
            content = page.read_text()
            self.assertIn("det_0000", content)
            self.assertIn("Export CSV", content)
            self.assertIn("Whole-slide position", content)
            self.assertIn("Hide mask", content)
            self.assertIn("class=\"raw-thumbnail\"", content)
            self.assertIn("api/patch.webp?", content)
            self.assertIn("Biologist note presets", content)
            self.assertIn("Halo artifact", content)
            self.assertIn("halo_artifact", content)
            self.assertIn("True oocyte, bad mask", content)
            self.assertIn(
                "true_oocyte; mask_truncated; mask_off_target",
                content,
            )
            self.assertIn("Non-oocyte tissue", content)
            self.assertIn("Import JSON", content)
            self.assertIn("oocyte_precision_review", content)
            self.assertIn("candidate_table_sha256", content)
            self.assertTrue((sample_dir / "html_assets/oocyte-0001.webp").is_file())
            thumbnail_manifest_path = sample_dir / "html_assets/manifest.json"
            thumbnail_manifest = json.loads(thumbnail_manifest_path.read_text())
            self.assertEqual(1, thumbnail_manifest["schema_version"])
            self.assertIn("oocyte-0001.webp", thumbnail_manifest["entries"])
            self.assertTrue(result.algorithm_document.is_file())
            self.assertTrue(result.batch_index.is_file())

            with mock.patch(
                "aegle.oocyte.report._raw_mask_thumbnail",
                side_effect=AssertionError("unchanged thumbnail should be reused"),
            ):
                generate_html_reports(root)

            thumbnail_manifest_path.unlink()
            with mock.patch(
                "aegle.oocyte.report._raw_mask_thumbnail",
                wraps=_raw_mask_thumbnail,
            ) as renderer:
                generate_html_reports(root)
            renderer.assert_called_once()

            shifted_mask = (yy - 200) ** 2 + (xx - 202) ** 2 <= 45**2
            save_candidate_mask(
                masks_dir / "det_0000.npz",
                mask=np.asarray(shifted_mask[150:251, 150:251], dtype=np.bool_),
                bbox=bbox,
                image_shape_yx=(401, 401),
                sample_id=sample_id,
                candidate_id="det_0000",
                profile_name=DONOR13_V6.profile_name,
                profile_fingerprint=DONOR13_V6.fingerprint(),
                metrics=metrics,
            )
            with mock.patch(
                "aegle.oocyte.report._raw_mask_thumbnail",
                wraps=_raw_mask_thumbnail,
            ) as renderer:
                generate_html_reports(root)
            renderer.assert_called_once()

            candidates_path = sample_dir / "candidates.csv"
            candidates = pd.read_csv(candidates_path)
            candidates.loc[0, "center_x"] = 210
            candidates.to_csv(candidates_path, index=False)
            with mock.patch(
                "aegle.oocyte.report._raw_mask_thumbnail",
                wraps=_raw_mask_thumbnail,
            ) as renderer:
                generate_html_reports(root)
            renderer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
