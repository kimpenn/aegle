import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte import DONOR13_V6
from aegle.oocyte.io import load_candidate_mask, save_candidate_mask
from aegle.oocyte.models import BoundingBox, SegmentationMetrics
from aegle.oocyte.precision_boundary_finalize import (
    finalize_precision_boundary_review,
)
from aegle.oocyte.precision_boundary_review import (
    generate_precision_boundary_review,
)
from aegle.oocyte.precision_manual_boundary_finalize import (
    finalize_precision_manual_boundary_review,
)
from aegle.oocyte.precision_manual_boundary_review import (
    generate_precision_manual_boundary_review,
)
from tests.oocyte.test_recall_review import RecallReviewFixture


def _add_manual_target(fixture: RecallReviewFixture) -> None:
    yy, xx = np.ogrid[:25, :25]
    fragment = np.asarray((yy - 12) ** 2 + (xx - 12) ** 2 <= 10**2, dtype=np.bool_)
    metrics = SegmentationMetrics(
        threshold_method="triangle",
        base_threshold=500.0,
        annulus_floor=5000.0,
        threshold=5000.0,
        selection_mode="center_component",
        area_px=int(fragment.sum()),
        equivalent_diameter_um=10.0,
        major_axis_um=10.0,
        minor_axis_um=10.0,
        eccentricity=0.0,
        solidity=0.99,
        circularity=0.95,
        centroid_y_px=12.0,
        centroid_x_px=12.0,
        centroid_offset_px=0.0,
        mean_intensity=10000.0,
        max_intensity=10000.0,
    )
    save_candidate_mask(
        fixture.sample_dir / "masks/manual-target.npz",
        mask=fragment,
        bbox=BoundingBox(148, 148, 173, 173),
        image_shape_yx=(768, 768),
        sample_id=fixture.sample_id,
        candidate_id="manual-target",
        profile_name=DONOR13_V6.profile_name,
        profile_fingerprint=DONOR13_V6.fingerprint(),
        metrics=metrics,
        implementation_version="test-v1",
    )
    candidates_path = fixture.sample_dir / "html_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    candidates = pd.concat(
        [
            candidates,
            pd.DataFrame(
                [
                    {
                        "detector_component_id": "manual-target",
                        "display_id": "#002",
                        "accepted": True,
                        "detector_score": 0.8,
                        "center_x": 160,
                        "center_y": 160,
                        "component_centroid_x": 160,
                        "component_centroid_y": 160,
                        "bbox_x0": 148,
                        "bbox_y0": 148,
                        "bbox_x1": 173,
                        "bbox_y1": 173,
                        "mask_path": "masks/manual-target.npz",
                        "mask_source_dir": str(fixture.sample_dir),
                        "detection_pass": "baseline_v6",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    candidates.to_csv(candidates_path, index=False)


def _precision_review(fixture: RecallReviewFixture, identity: dict) -> Path:
    candidates = pd.read_csv(fixture.sample_dir / "html_candidates.csv")
    rows = []
    for row in candidates.to_dict("records"):
        target = str(row["detector_component_id"]) == "manual-target"
        rows.append(
            {
                "review_key": f"baseline_v6:{row['detector_component_id']}",
                "display_id": str(row["display_id"]),
                "detector_component_id": str(row["detector_component_id"]),
                "detection_pass": "baseline_v6",
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "manual_status": "reject" if target else "accept",
                "manual_notes": (
                    "true_oocyte; mask_truncated; mask_off_target" if target else ""
                ),
            }
        )
    path = fixture.root / "precision-review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "oocyte_precision_review",
                "identity": identity,
                "exported_at": "2026-07-12T12:00:00Z",
                "rows": rows,
            }
        )
    )
    return path


def _embedded_payload(page_path: Path, element_id: str) -> dict:
    match = re.search(
        rf'<script id="{re.escape(element_id)}" type="application/json">(.*?)</script>',
        page_path.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"page is missing {element_id}")
    return json.loads(match.group(1))


def _prepare_base(root: Path):
    fixture = RecallReviewFixture(root)
    _add_manual_target(fixture)
    bundle = fixture.generate()
    identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
    precision_path = _precision_review(fixture, identity)
    boundary_pack = generate_precision_boundary_review(
        fixture.sample_dir,
        precision_path,
        root / "boundary-pack",
    )
    embedded = _embedded_payload(boundary_pack.page_path, "boundary-data")
    boundary_rows = embedded["rows"]
    boundary_rows[0]["boundary_review_choice"] = "needs_manual"
    boundary_rows[0]["boundary_review_notes"] = "trace the complete target"
    boundary_path = root / "boundary-review.json"
    boundary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "oocyte_precision_boundary_review",
                "identity": embedded["identity"],
                "exported_at": "2026-07-12T13:00:00Z",
                "rows": boundary_rows,
            }
        )
    )
    base_dir = root / "precision-resolved-v1"
    finalize_precision_boundary_review(
        fixture.sample_dir,
        precision_path,
        boundary_path,
        base_dir,
        tile_shape_yx=(16, 16),
    )
    return fixture, base_dir


def _manual_review_payload(page_path: Path, *, self_intersecting: bool = False) -> dict:
    embedded = _embedded_payload(page_path, "manual-boundary-data")
    row = embedded["rows"][0]
    center_x = float(row["center_x"])
    center_y = float(row["center_y"])
    if self_intersecting:
        vertices = [
            [center_x - 30, center_y - 30],
            [center_x + 30, center_y + 30],
            [center_x - 30, center_y + 30],
            [center_x + 30, center_y - 30],
        ]
    else:
        vertices = [
            [
                center_x + 34 * math.cos(2 * math.pi * index / 16),
                center_y + 34 * math.sin(2 * math.pi * index / 16),
            ]
            for index in range(16)
        ]
    row["manual_boundary_choice"] = "accept_manual_contour"
    row["manual_boundary_notes"] = "reviewed synthetic contour"
    row["vertices_xy"] = vertices
    row["vertex_count"] = len(vertices)
    return {
        "schema_version": 1,
        "review_type": "oocyte_precision_manual_boundary_review",
        "identity": embedded["identity"],
        "exported_at": "2026-07-12T14:00:00Z",
        "rows": [row],
    }


class TestPrecisionManualBoundary(unittest.TestCase):
    def test_generates_identity_bound_polygon_editor_without_modifying_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, base_dir = _prepare_base(root)
            base_manifest_before = (base_dir / "precision_resolved_manifest.json").read_bytes()

            result = generate_precision_manual_boundary_review(
                fixture.sample_dir,
                base_dir,
                root / "manual-boundary-pack",
                patch_radius_px=128,
            )

            self.assertEqual(result.card_count, 1)
            self.assertIn("Trace only the intended", result.page_path.read_text())
            self.assertIn("Accept contour", result.page_path.read_text())
            table = pd.read_csv(result.candidates_path)
            self.assertEqual(table.loc[0, "review_key"], "baseline_v6:manual-target")
            self.assertTrue((result.assets_dir / table.loc[0, "raw_asset_name"]).is_file())
            self.assertTrue(
                (result.assets_dir / table.loc[0, "context_asset_name"]).is_file()
            )
            self.assertEqual(
                base_manifest_before,
                (base_dir / "precision_resolved_manifest.json").read_bytes(),
            )

    def test_finalizes_polygon_with_base_carry_forward_and_zero_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, base_dir = _prepare_base(root)
            pack = generate_precision_manual_boundary_review(
                fixture.sample_dir,
                base_dir,
                root / "manual-boundary-pack",
                patch_radius_px=128,
            )
            payload = _manual_review_payload(pack.page_path)
            review_path = root / "manual-boundary-review.json"
            review_path.write_text(json.dumps(payload))
            base_manifest_before = (base_dir / "precision_resolved_manifest.json").read_bytes()

            result = finalize_precision_manual_boundary_review(
                fixture.sample_dir,
                base_dir,
                review_path,
                root / "precision-resolved-v2",
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.resolved_count, 2)
            self.assertEqual(result.manual_added_count, 1)
            self.assertEqual(result.manual_excluded_count, 0)
            self.assertEqual(set(tifffile.imread(result.labels.image_path).ravel()), {0, 1, 2})
            candidates = pd.read_csv(result.candidates_path)
            self.assertEqual(len(candidates), 2)
            manual = candidates[
                candidates["review_key"] == "baseline_v6:manual-target"
            ].iloc[0]
            mask = load_candidate_mask(result.out_dir / manual["mask_path"])
            self.assertTrue(mask.metadata["reviewed_manual_polygon"])
            self.assertTrue(mask.metadata["precision_resolved"])
            self.assertTrue(mask.mask.any())
            manifest = json.loads(result.manifest_path.read_text())
            self.assertFalse(manifest["release_ready"])
            self.assertTrue(manifest["manual_boundary_complete"])
            self.assertFalse(manifest["recall_complete"])
            self.assertEqual(manifest["resolved_label_count"], 2)
            self.assertEqual(manifest["overlap_audit_row_count"], 0)
            self.assertEqual(
                base_manifest_before,
                (base_dir / "precision_resolved_manifest.json").read_bytes(),
            )

    def test_rejects_self_intersection_and_stale_candidate_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, base_dir = _prepare_base(root)
            pack = generate_precision_manual_boundary_review(
                fixture.sample_dir,
                base_dir,
                root / "manual-boundary-pack",
                patch_radius_px=128,
            )
            payload = _manual_review_payload(
                pack.page_path,
                self_intersecting=True,
            )
            review_path = root / "self-intersection.json"
            review_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "self-intersects"):
                finalize_precision_manual_boundary_review(
                    fixture.sample_dir,
                    base_dir,
                    review_path,
                    root / "invalid-v2",
                    tile_shape_yx=(16, 16),
                )

            valid_payload = _manual_review_payload(pack.page_path)
            stale_review_path = root / "stale-review.json"
            stale_review_path.write_text(json.dumps(valid_payload))
            pack.candidates_path.write_text(pack.candidates_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "candidate table SHA-256"):
                finalize_precision_manual_boundary_review(
                    fixture.sample_dir,
                    base_dir,
                    stale_review_path,
                    root / "stale-v2",
                    tile_shape_yx=(16, 16),
                )


if __name__ == "__main__":
    unittest.main()
