import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte.manual_seed_finalize import finalize_manual_seed_review
from aegle.oocyte.recall_review import analyze_recall_review
from aegle.oocyte.shape_recovery_finalize import finalize_shape_recovery_review
from aegle.oocyte.shape_recovery_review import generate_shape_recovery_review
from tests.oocyte.test_recall_review import RecallReviewFixture


def _prepare_shape_case(root: Path):
    fixture = RecallReviewFixture(root)
    image = tifffile.imread(fixture.image_path)
    yy, xx = np.ogrid[:201, :201]
    patch = np.full((201, 201), 100, dtype=np.uint16)
    target = (yy - 100) ** 2 + (xx - 100) ** 2 <= 30**2
    fragment = (yy - 106) ** 2 + (xx - 112) ** 2 <= 14**2
    distance = np.sqrt((yy - 100) ** 2 + (xx - 100) ** 2)
    angle = (np.arctan2(yy - 100, xx - 100) + 2 * np.pi) % (2 * np.pi)
    sector = (distance >= 45) & (distance <= 85) & (angle < 0.75)
    patch[target] = 3500
    patch[fragment] = 7000
    patch[sector] = 5000
    image[0, 60:261, 60:261] = patch
    tifffile.imwrite(
        fixture.image_path,
        image,
        ome=True,
        metadata={"axes": "CYX"},
    )
    bundle = fixture.generate()
    metadata = json.loads(bundle.metadata_path.read_text())
    recall_path = root / "recall.json"
    recall_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "oocyte_recall",
                "sample": metadata["review_identity"],
                "windows": [],
                "missing_oocytes": [
                    {
                        "annotation_id": "shape-1",
                        "window_id": metadata["windows"][0]["window_id"],
                        "x": 160,
                        "y": 160,
                        "notes": "",
                    }
                ],
            }
        )
    )
    analysis_dir = root / "analysis"
    analyze_recall_review(fixture.sample_dir, recall_path, analysis_dir)
    page = (analysis_dir / "manual_seed_review.html").read_text()
    match = re.search(
        r'<script id="seed-data" type="application/json">(.*?)</script>',
        page,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("manual-seed review page is missing embedded data")
    embedded = json.loads(match.group(1))
    embedded["rows"][0]["manual_mask_choice"] = "accept_manual_expanded"
    embedded["rows"][0]["manual_notes"] = "boundary is incomplete"
    manual_path = root / "manual.json"
    manual_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "manual_seed_mask_review",
                "identity": embedded["identity"],
                "exported_at": "2026-07-11T12:00:00.000Z",
                "rows": embedded["rows"],
            }
        )
    )
    result = generate_shape_recovery_review(
        fixture.sample_dir,
        recall_path,
        manual_path,
        root / "shape-review",
    )
    return fixture, analysis_dir, manual_path, result


class TestShapeRecoveryReview(unittest.TestCase):
    def test_generates_only_masks_changed_by_shape_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, result = _prepare_shape_case(root)

            self.assertEqual(result.card_count, 1)
            table = pd.read_csv(result.candidates_path)
            self.assertEqual(table.loc[0, "current_percentile"], 95.0)
            self.assertEqual(table.loc[0, "shape_recovery_percentile"], 85.0)
            self.assertGreater(table.loc[0, "shape_to_current_area_ratio"], 4.0)
            page = result.page_path.read_text()
            self.assertIn("Use recovery", page)
            self.assertIn("Keep v4", page)
            self.assertTrue((result.assets_dir / "shape-001.webp").is_file())

    def test_finalizes_shape_replacement_without_modifying_v1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, analysis_dir, manual_path, shape_pack = _prepare_shape_case(root)
            base = finalize_manual_seed_review(
                fixture.sample_dir,
                manual_path,
                root / "finalized-v1",
                analysis_dir=analysis_dir,
                tile_shape_yx=(16, 16),
            )
            base_manifest_before = base.manifest_path.read_bytes()
            base_mask_before = (base.out_dir / "reviewed_masks/manual_seed_001.npz").read_bytes()
            page = shape_pack.page_path.read_text()
            match = re.search(
                r'<script id="shape-data" type="application/json">(.*?)</script>',
                page,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match)
            embedded = json.loads(match.group(1))
            embedded["rows"][0]["shape_review_choice"] = "accept_shape_recovery"
            embedded["rows"][0]["shape_review_notes"] = "approved"
            shape_review_path = root / "shape-review.json"
            shape_review_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "manual_seed_shape_recovery_review",
                        "identity": embedded["identity"],
                        "exported_at": "2026-07-12T12:00:00.000Z",
                        "rows": embedded["rows"],
                    }
                )
            )

            result = finalize_shape_recovery_review(
                fixture.sample_dir,
                shape_review_path,
                base.out_dir,
                root / "finalized-v2",
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.boundary_warning_count, 0)
            self.assertEqual(result.delta_labels.label_count, 1)
            self.assertEqual(result.combined_labels.label_count, 2)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["shape_replacement_count"], 1)
            self.assertEqual(manifest["shape_addition_count"], 0)
            self.assertEqual(manifest["combined_label_count"], 2)
            self.assertFalse(manifest["base_v1_outputs_modified"])
            decisions = pd.read_csv(result.decisions_path)
            self.assertEqual(decisions.loc[0, "final_source"], "shape_recovery")
            self.assertEqual(base_manifest_before, base.manifest_path.read_bytes())
            self.assertEqual(
                base_mask_before,
                (base.out_dir / "reviewed_masks/manual_seed_001.npz").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
