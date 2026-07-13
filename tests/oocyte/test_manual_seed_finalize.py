import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import tifffile

from aegle.oocyte.io import load_candidate_mask
from aegle.oocyte.manual_seed_finalize import finalize_manual_seed_review
from aegle.oocyte.recall_review import analyze_recall_review
from tests.oocyte.test_recall_review import RecallReviewFixture


def _manual_review_payload(page_path: Path) -> dict:
    match = re.search(
        r'<script id="seed-data" type="application/json">(.*?)</script>',
        page_path.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("manual-seed page is missing its JSON payload")
    embedded = json.loads(match.group(1))
    return {
        "schema_version": 1,
        "review_type": "manual_seed_mask_review",
        "identity": embedded["identity"],
        "exported_at": "2026-07-11T12:00:00.000Z",
        "rows": embedded["rows"],
    }


def _prepare_review(
    root: Path,
    *,
    click_xy: tuple[int, int] = (160, 160),
) -> tuple[RecallReviewFixture, Path, dict]:
    fixture = RecallReviewFixture(root)
    bundle = fixture.generate()
    metadata = json.loads(bundle.metadata_path.read_text())
    recall_review_path = root / "recall-review.json"
    recall_review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "oocyte_recall",
                "sample": metadata["review_identity"],
                "windows": [
                    {
                        "window_id": metadata["windows"][0]["window_id"],
                        "status": "has_misses",
                    }
                ],
                "missing_oocytes": [
                    {
                        "annotation_id": "manual-1",
                        "window_id": metadata["windows"][0]["window_id"],
                        "x": click_xy[0],
                        "y": click_xy[1],
                        "notes": "synthetic missed circle",
                    }
                ],
            }
        )
    )
    analysis_dir = root / "analysis"
    analyze_recall_review(fixture.sample_dir, recall_review_path, analysis_dir)
    payload = _manual_review_payload(analysis_dir / "manual_seed_review.html")
    return fixture, analysis_dir, payload


class TestManualSeedFinalize(unittest.TestCase):
    def test_writes_reviewed_delta_and_versioned_combined_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, analysis_dir, payload = _prepare_review(root)
            payload["rows"][0]["manual_mask_choice"] = "accept_manual_expanded"
            payload["rows"][0]["manual_notes"] = (
                "The expanded mask still misses part of the cell boundary"
            )
            review_path = root / "manual-review.json"
            review_path.write_text(json.dumps(payload))
            candidates_before = (fixture.sample_dir / "html_candidates.csv").read_bytes()
            analysis_before = (analysis_dir / "recall_failure_analysis.csv").read_bytes()

            result = finalize_manual_seed_review(
                fixture.sample_dir,
                review_path,
                root / "finalized",
                analysis_dir=analysis_dir,
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.boundary_warning_count, 1)
            self.assertEqual(result.delta_labels.label_count, 1)
            self.assertIsNotNone(result.combined_labels)
            self.assertEqual(result.combined_labels.label_count, 2)
            self.assertEqual(tifffile.imread(result.delta_labels.image_path).shape, (768, 768))
            self.assertEqual(
                set(tifffile.imread(result.combined_labels.image_path).ravel()),
                {0, 1, 2},
            )
            decisions = pd.read_csv(result.decisions_path)
            candidates = pd.read_csv(result.candidates_path)
            self.assertTrue(bool(decisions.loc[0, "boundary_warning"]))
            self.assertEqual(candidates.loc[0, "display_id"], "#R001")
            mask_path = result.out_dir / candidates.loc[0, "mask_path"]
            reviewed = load_candidate_mask(mask_path)
            self.assertTrue(reviewed.metadata["reviewed_manual_seed"])
            self.assertFalse(reviewed.metadata["provisional_only"])
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["accepted_manual_mask_count"], 1)
            self.assertEqual(manifest["combined_label_count"], 2)
            self.assertFalse(manifest["production_outputs_modified"])
            self.assertEqual(
                candidates_before,
                (fixture.sample_dir / "html_candidates.csv").read_bytes(),
            )
            self.assertEqual(
                analysis_before,
                (analysis_dir / "recall_failure_analysis.csv").read_bytes(),
            )

    def test_rejects_stale_analysis_identity_and_missing_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, analysis_dir, payload = _prepare_review(root)
            payload["rows"][0]["manual_mask_choice"] = "accept_manual_expanded"

            stale = json.loads(json.dumps(payload))
            stale["identity"]["analysis_sha256"] = "0" * 64
            stale_path = root / "stale.json"
            stale_path.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                finalize_manual_seed_review(
                    fixture.sample_dir,
                    stale_path,
                    root / "stale-out",
                    analysis_dir=analysis_dir,
                    tile_shape_yx=(16, 16),
                )

            missing = json.loads(json.dumps(payload))
            missing["rows"][0]["manual_mask_choice"] = ""
            missing_path = root / "missing.json"
            missing_path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, "invalid or missing"):
                finalize_manual_seed_review(
                    fixture.sample_dir,
                    missing_path,
                    root / "missing-out",
                    analysis_dir=analysis_dir,
                    tile_shape_yx=(16, 16),
                )

    def test_rejects_reviewed_mask_that_duplicates_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, analysis_dir, payload = _prepare_review(
                root,
                click_xy=(380, 380),
            )
            payload["rows"][0]["manual_mask_choice"] = "accept_manual_expanded"
            review_path = root / "duplicate-review.json"
            review_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "blocking overlap"):
                finalize_manual_seed_review(
                    fixture.sample_dir,
                    review_path,
                    root / "duplicate-out",
                    analysis_dir=analysis_dir,
                    tile_shape_yx=(16, 16),
                )

if __name__ == "__main__":
    unittest.main()
