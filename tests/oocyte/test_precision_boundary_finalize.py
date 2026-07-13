import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import tifffile

from aegle.oocyte.io import load_candidate_mask
from aegle.oocyte.precision_boundary_finalize import (
    finalize_precision_boundary_review,
)
from aegle.oocyte.precision_boundary_review import (
    generate_precision_boundary_review,
)
from tests.oocyte.test_precision_boundary_review import (
    _precision_review,
    _replace_with_fragment_case,
)
from tests.oocyte.test_recall_review import RecallReviewFixture


def _boundary_review_payload(page_path: Path, *, choice: str) -> dict:
    match = re.search(
        r'<script id="boundary-data" type="application/json">(.*?)</script>',
        page_path.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("boundary-review page is missing its JSON payload")
    embedded = json.loads(match.group(1))
    rows = embedded["rows"]
    for row in rows:
        row["boundary_review_choice"] = choice
        row["boundary_review_notes"] = "synthetic decision"
    return {
        "schema_version": 1,
        "review_type": "oocyte_precision_boundary_review",
        "identity": embedded["identity"],
        "exported_at": "2026-07-12T13:00:00Z",
        "rows": rows,
    }


class TestPrecisionBoundaryFinalize(unittest.TestCase):
    def test_finalizes_selected_proposal_without_modifying_source_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecallReviewFixture(root)
            _replace_with_fragment_case(fixture)
            bundle = fixture.generate()
            identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
            precision_path = _precision_review(fixture, identity)
            pack = generate_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                root / "boundary-pack",
            )
            payload = _boundary_review_payload(
                pack.page_path,
                choice="use_expanded",
            )
            boundary_path = root / "boundary-review.json"
            boundary_path.write_text(json.dumps(payload))
            current_before = (fixture.sample_dir / "masks/accepted.npz").read_bytes()
            proposal_path = Path(payload["rows"][0]["expanded_mask_path"])
            proposal_before = proposal_path.read_bytes()

            result = finalize_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                boundary_path,
                root / "precision-resolved",
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.resolved_count, 1)
            self.assertEqual(result.unresolved_manual_count, 0)
            self.assertEqual(result.excluded_count, 0)
            self.assertEqual(set(tifffile.imread(result.labels.image_path).ravel()), {0, 1})
            candidates = pd.read_csv(result.candidates_path)
            self.assertEqual(
                candidates.loc[0, "precision_resolution_source"],
                "precision_boundary_expanded",
            )
            reviewed = load_candidate_mask(
                result.out_dir / candidates.loc[0, "mask_path"]
            )
            self.assertTrue(reviewed.metadata["precision_resolved"])
            self.assertFalse(reviewed.metadata["provisional_only"])
            self.assertEqual(
                reviewed.metadata["precision_boundary_choice"],
                "use_expanded",
            )
            manifest = json.loads(result.manifest_path.read_text())
            self.assertFalse(manifest["release_ready"])
            self.assertTrue(manifest["precision_complete"])
            self.assertTrue(manifest["manual_boundary_complete"])
            self.assertFalse(manifest["recall_complete"])
            self.assertEqual(manifest["resolved_label_count"], 1)
            self.assertEqual(manifest["label_export"]["overlap_pixel_count"], 0)
            self.assertEqual(
                current_before,
                (fixture.sample_dir / "masks/accepted.npz").read_bytes(),
            )
            self.assertEqual(proposal_before, proposal_path.read_bytes())
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                finalize_precision_boundary_review(
                    fixture.sample_dir,
                    precision_path,
                    boundary_path,
                    root / "precision-resolved",
                    tile_shape_yx=(16, 16),
                )

    def test_preserves_needs_manual_as_unresolved_without_a_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecallReviewFixture(root)
            bundle = fixture.generate()
            identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
            precision_path = _precision_review(fixture, identity)
            pack = generate_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                root / "boundary-pack",
            )
            payload = _boundary_review_payload(
                pack.page_path,
                choice="needs_manual",
            )
            boundary_path = root / "boundary-review.json"
            boundary_path.write_text(json.dumps(payload))

            result = finalize_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                boundary_path,
                root / "precision-resolved",
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.resolved_count, 0)
            self.assertEqual(result.unresolved_manual_count, 1)
            self.assertEqual(result.excluded_count, 0)
            self.assertEqual(set(tifffile.imread(result.labels.image_path).ravel()), {0})
            queue = pd.read_csv(result.manual_queue_path)
            self.assertEqual(queue.loc[0, "review_key"], "baseline_v6:accepted-1")
            manifest = json.loads(result.manifest_path.read_text())
            self.assertFalse(manifest["manual_boundary_complete"])
            self.assertEqual(
                manifest["unresolved_review_keys"],
                ["baseline_v6:accepted-1"],
            )

    def test_carries_forward_a_direct_precision_accept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecallReviewFixture(root)
            bundle = fixture.generate()
            identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
            precision_path = _precision_review(fixture, identity)
            precision = json.loads(precision_path.read_text())
            precision["rows"][0]["manual_status"] = "accept"
            precision["rows"][0]["manual_notes"] = ""
            precision_path.write_text(json.dumps(precision))
            pack = generate_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                root / "empty-boundary-pack",
            )
            self.assertEqual(pack.card_count, 0)
            self.assertIsNotNone(pack.automatic_review_path)
            boundary_path = pack.automatic_review_path
            assert boundary_path is not None
            self.assertTrue(boundary_path.is_file())
            self.assertEqual(
                list(pd.read_csv(pack.candidates_path).columns),
                [
                    "boundary_index",
                    "review_key",
                    "display_id",
                    "detector_component_id",
                    "detection_pass",
                    "x",
                    "y",
                    "current_mask_path",
                    "conservative_available",
                    "expanded_available",
                ],
            )

            result = finalize_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                boundary_path,
                root / "precision-resolved",
                tile_shape_yx=(16, 16),
            )

            candidates = pd.read_csv(result.candidates_path)
            self.assertEqual(result.resolved_count, 1)
            self.assertEqual(
                candidates.loc[0, "precision_resolution_source"],
                "precision_accept_current",
            )

    def test_rejects_stale_boundary_candidate_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecallReviewFixture(root)
            _replace_with_fragment_case(fixture)
            bundle = fixture.generate()
            identity = json.loads(bundle.metadata_path.read_text())["review_identity"]
            precision_path = _precision_review(fixture, identity)
            pack = generate_precision_boundary_review(
                fixture.sample_dir,
                precision_path,
                root / "boundary-pack",
            )
            payload = _boundary_review_payload(
                pack.page_path,
                choice="use_expanded",
            )
            boundary_path = root / "boundary-review.json"
            boundary_path.write_text(json.dumps(payload))
            pack.candidates_path.write_text(pack.candidates_path.read_text() + "\n")

            with self.assertRaisesRegex(ValueError, "candidate table SHA-256"):
                finalize_precision_boundary_review(
                    fixture.sample_dir,
                    precision_path,
                    boundary_path,
                    root / "precision-resolved",
                    tile_shape_yx=(16, 16),
                )


if __name__ == "__main__":
    unittest.main()
