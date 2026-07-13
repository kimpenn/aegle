import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte.io import load_candidate_mask
from aegle.oocyte.manual_seed_finalize import finalize_manual_seed_review
from aegle.oocyte.recall_manual_boundary_finalize import (
    finalize_recall_manual_boundary_review,
)
from aegle.oocyte.recall_manual_boundary_review import (
    generate_recall_manual_boundary_review,
)
from aegle.oocyte.recall_review import analyze_recall_review
from tests.oocyte.test_manual_seed_finalize import _manual_review_payload
from tests.oocyte.test_recall_review import RecallReviewFixture


def _embedded_payload(page_path: Path) -> dict:
    match = re.search(
        r'<script id="manual-boundary-data" type="application/json">(.*?)</script>',
        page_path.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("Recall manual-boundary page is missing its payload")
    return json.loads(match.group(1))


def _add_second_missed_oocyte(fixture: RecallReviewFixture) -> None:
    image = tifffile.imread(fixture.image_path)
    yy, xx = np.ogrid[:768, :768]
    target = (yy - 600) ** 2 + (xx - 600) ** 2 <= 28**2
    image[0, target] = 9000
    tifffile.imwrite(
        fixture.image_path,
        image,
        ome=True,
        metadata={"axes": "CYX"},
    )
    manifest_path = fixture.sample_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_image_size_bytes"] = fixture.image_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))


def _prepare_manual_boundary(root: Path):
    fixture = RecallReviewFixture(root)
    _add_second_missed_oocyte(fixture)
    bundle = fixture.generate()
    metadata = json.loads(bundle.metadata_path.read_text())
    recall_path = root / "recall-review.json"
    recall_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "oocyte_recall",
                "sample": metadata["review_identity"],
                "windows": [],
                "missing_oocytes": [
                    {
                        "annotation_id": "accepted-miss",
                        "window_id": metadata["windows"][0]["window_id"],
                        "x": 160,
                        "y": 160,
                        "notes": "safe automatic boundary",
                    },
                    {
                        "annotation_id": "manual-boundary-miss",
                        "window_id": metadata["windows"][0]["window_id"],
                        "x": 600,
                        "y": 600,
                        "notes": "two bad automatic boundaries",
                    },
                ],
            }
        )
    )
    analysis_dir = root / "analysis"
    analyze_recall_review(fixture.sample_dir, recall_path, analysis_dir)
    manual_payload = _manual_review_payload(analysis_dir / "manual_seed_review.html")
    manual_payload["rows"][0]["manual_mask_choice"] = "accept_manual_expanded"
    manual_payload["rows"][1]["manual_mask_choice"] = "neither"
    manual_payload["rows"][1]["manual_notes"] = (
        "true_oocyte; conservative_under-segmented; "
        "expanded_over-segmented_into_neighbor; needs_manual_boundary"
    )
    manual_review_path = root / "manual-seed-review.json"
    manual_review_path.write_text(json.dumps(manual_payload))
    base_dir = root / "reviewed-manual-seed-v1"
    finalize_manual_seed_review(
        fixture.sample_dir,
        manual_review_path,
        base_dir,
        tile_shape_yx=(16, 16),
    )
    pack = generate_recall_manual_boundary_review(
        fixture.sample_dir,
        manual_review_path,
        base_dir,
        root / "manual-boundary-pack",
        patch_radius_px=128,
    )
    return fixture, manual_review_path, base_dir, pack


class TestRecallManualBoundary(unittest.TestCase):
    def test_generates_and_finalizes_identity_bound_recall_contour(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, _, base_dir, pack = _prepare_manual_boundary(root)
            base_manifest = base_dir / "manual_seed_finalize_manifest.json"
            base_manifest_before = base_manifest.read_bytes()

            self.assertEqual(pack.card_count, 1)
            page = pack.page_path.read_text()
            self.assertIn("oocyte_recall_manual_boundary_review", page)
            self.assertIn("expanded proposal was rejected", page)
            candidates = pd.read_csv(pack.candidates_path)
            self.assertEqual(candidates.loc[0, "review_key"], "manual-boundary-miss")
            self.assertEqual(int(candidates.loc[0, "review_index"]), 2)
            self.assertEqual(candidates.loc[0, "display_id"], "#R002")
            self.assertTrue(
                (pack.assets_dir / candidates.loc[0, "raw_asset_name"]).is_file()
            )
            self.assertTrue(
                (pack.assets_dir / candidates.loc[0, "context_asset_name"]).is_file()
            )

            embedded = _embedded_payload(pack.page_path)
            row = embedded["rows"][0]
            center_x = float(row["center_x"])
            center_y = float(row["center_y"])
            bad_row = dict(row)
            bad_row["manual_boundary_choice"] = "accept_manual_contour"
            bad_row["manual_boundary_notes"] = "self-intersecting test"
            bad_row["vertices_xy"] = [
                [center_x - 30, center_y - 30],
                [center_x + 30, center_y + 30],
                [center_x - 30, center_y + 30],
                [center_x + 30, center_y - 30],
            ]
            bad_path = root / "bad-contour.json"
            bad_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "oocyte_recall_manual_boundary_review",
                        "identity": embedded["identity"],
                        "rows": [bad_row],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "self-intersects"):
                finalize_recall_manual_boundary_review(
                    fixture.sample_dir,
                    base_dir,
                    bad_path,
                    root / "bad-v2",
                    tile_shape_yx=(16, 16),
                )

            vertices = [
                [
                    center_x + 30 * math.cos(2 * math.pi * index / 20),
                    center_y + 30 * math.sin(2 * math.pi * index / 20),
                ]
                for index in range(20)
            ]
            row["manual_boundary_choice"] = "accept_manual_contour"
            row["manual_boundary_notes"] = "reviewed Recall contour"
            row["vertices_xy"] = vertices
            row["vertex_count"] = len(vertices)
            review_path = root / "recall-manual-boundary-review.json"
            review_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "oocyte_recall_manual_boundary_review",
                        "identity": embedded["identity"],
                        "exported_at": "2026-07-12T18:00:00Z",
                        "rows": [row],
                    }
                )
            )
            result = finalize_recall_manual_boundary_review(
                fixture.sample_dir,
                base_dir,
                review_path,
                root / "reviewed-manual-seed-v2",
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.manual_added_count, 1)
            self.assertEqual(result.manual_excluded_count, 0)
            self.assertEqual(result.combined_label_count, 3)
            self.assertEqual(result.labels.overlap_pixel_count, 0)
            self.assertEqual(base_manifest_before, base_manifest.read_bytes())
            contour_table = pd.read_csv(result.candidates_path)
            contour_path = result.out_dir / contour_table.loc[0, "mask_path"]
            contour = load_candidate_mask(contour_path)
            self.assertTrue(
                contour.mask[
                    600 - contour.bbox.y0,
                    600 - contour.bbox.x0,
                ]
            )
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["base_label_count"], 2)
            self.assertEqual(manifest["combined_label_count"], 3)
            self.assertEqual(manifest["remaining_manual_boundary_count"], 0)
            self.assertTrue(
                all(
                    hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
                    == record["sha256"]
                    for record in manifest["artifacts"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
