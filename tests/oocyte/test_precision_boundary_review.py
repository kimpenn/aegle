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
from aegle.oocyte.precision_boundary_review import (
    BOUNDARY_RECOVERY_PARAMETERS,
    generate_precision_boundary_review,
)
from tests.oocyte.test_recall_review import RecallReviewFixture


def _precision_review(fixture: RecallReviewFixture, identity: dict) -> Path:
    row = pd.read_csv(fixture.sample_dir / "html_candidates.csv").iloc[0]
    path = fixture.root / "precision-review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "oocyte_precision_review",
                "identity": identity,
                "exported_at": "2026-07-12T12:00:00Z",
                "rows": [
                    {
                        "review_key": "baseline_v6:accepted-1",
                        "display_id": "#001",
                        "detector_component_id": "accepted-1",
                        "detection_pass": "baseline_v6",
                        "center_x": float(row["center_x"]),
                        "center_y": float(row["center_y"]),
                        "manual_status": "reject",
                        "manual_notes": (
                            "true_oocyte; mask_truncated; mask_off_target"
                        ),
                    }
                ],
            }
        )
    )
    return path


def _replace_with_fragment_case(fixture: RecallReviewFixture) -> None:
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
    image[0, 280:481, 280:481] = patch
    tifffile.imwrite(fixture.image_path, image, ome=True, metadata={"axes": "CYX"})

    bbox = BoundingBox(378, 372, 407, 401)
    cropped = np.asarray(fragment[92:121, 98:127], dtype=np.bool_)
    metrics = SegmentationMetrics(
        threshold_method="triangle",
        base_threshold=500.0,
        annulus_floor=5000.0,
        threshold=5000.0,
        selection_mode="center_component",
        area_px=int(cropped.sum()),
        equivalent_diameter_um=14.0,
        major_axis_um=14.0,
        minor_axis_um=14.0,
        eccentricity=0.0,
        solidity=0.99,
        circularity=0.95,
        centroid_y_px=14.0,
        centroid_x_px=14.0,
        centroid_offset_px=13.4,
        mean_intensity=7000.0,
        max_intensity=7000.0,
    )
    save_candidate_mask(
        fixture.sample_dir / "masks/accepted.npz",
        mask=cropped,
        bbox=bbox,
        image_shape_yx=(768, 768),
        sample_id=fixture.sample_id,
        candidate_id="accepted-1",
        profile_name=DONOR13_V6.profile_name,
        profile_fingerprint=DONOR13_V6.fingerprint(),
        metrics=metrics,
        implementation_version="test-v1",
    )
    candidates_path = fixture.sample_dir / "html_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    candidates.loc[0, ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]] = [
        378,
        372,
        407,
        401,
    ]
    candidates.to_csv(candidates_path, index=False)


class TestPrecisionBoundaryReview(unittest.TestCase):
    def test_generates_shape_gated_replacement_without_modifying_current_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            _replace_with_fragment_case(fixture)
            bundle = fixture.generate()
            identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
            review_path = _precision_review(fixture, identity)
            current_before = (
                fixture.sample_dir / "masks/accepted.npz"
            ).read_bytes()

            result = generate_precision_boundary_review(
                fixture.sample_dir,
                review_path,
                fixture.sample_dir / "recall_analysis_precision_boundary_v1",
            )

            self.assertEqual(result.card_count, 1)
            self.assertEqual(result.proposal_count, 1)
            self.assertEqual(result.manual_only_count, 0)
            table = pd.read_csv(result.candidates_path)
            self.assertFalse(bool(table.loc[0, "conservative_available"]))
            self.assertTrue(bool(table.loc[0, "expanded_available"]))
            self.assertGreater(table.loc[0, "expanded_area_ratio"], 4.0)
            self.assertGreater(table.loc[0, "expanded_current_overlap"], 0.95)
            self.assertIn("Use expanded", result.page_path.read_text())
            self.assertTrue(
                (result.assets_dir / "boundary-001.webp").is_file()
            )
            self.assertEqual(
                current_before,
                (fixture.sample_dir / "masks/accepted.npz").read_bytes(),
            )

    def test_marks_nonexpanding_candidate_manual_only_and_rejects_stale_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            bundle = fixture.generate()
            identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
            review_path = _precision_review(fixture, identity)

            with mock.patch.dict(
                BOUNDARY_RECOVERY_PARAMETERS,
                {"min_area_growth_ratio": 2.0},
            ):
                result = generate_precision_boundary_review(
                    fixture.sample_dir,
                    review_path,
                    fixture.sample_dir / "recall_analysis_precision_boundary_v1",
                )
            self.assertEqual(result.proposal_count, 0)
            self.assertEqual(result.manual_only_count, 1)
            table = pd.read_csv(result.candidates_path)
            self.assertEqual(table.loc[0, "proposal_status"], "manual_required")

            payload = json.loads(review_path.read_text())
            payload["identity"]["candidate_table_sha256"] = "stale"
            stale_path = fixture.root / "stale-precision-review.json"
            stale_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "candidate_table_sha256"):
                generate_precision_boundary_review(
                    fixture.sample_dir,
                    stale_path,
                    fixture.sample_dir / "stale-boundary-review",
                )


if __name__ == "__main__":
    unittest.main()
