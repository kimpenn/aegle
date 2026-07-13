import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aegle.oocyte.precision_manual_boundary_finalize import (
    finalize_precision_manual_boundary_review,
)
from aegle.oocyte.precision_manual_boundary_review import (
    generate_precision_manual_boundary_review,
)
from aegle.oocyte.manual_seed_finalize import finalize_manual_seed_review
from aegle.oocyte.recall_review import (
    RecallReviewRuntime,
    _load_sample,
    analyze_recall_review,
    generate_recall_review_bundle,
)
from aegle.oocyte.recall_review_batch import (
    generate_batch_recall_review_bundle,
)
from tests.oocyte.test_precision_manual_boundary import (
    _manual_review_payload,
    _prepare_base,
)
from tests.oocyte.test_manual_seed_finalize import (
    _manual_review_payload as _manual_seed_review_payload,
)


def _prepare_v2(root: Path):
    fixture, base_dir = _prepare_base(root)
    pack = generate_precision_manual_boundary_review(
        fixture.sample_dir,
        base_dir,
        root / "manual-boundary-pack",
        patch_radius_px=128,
    )
    review_path = root / "manual-boundary-review.json"
    review_path.write_text(json.dumps(_manual_review_payload(pack.page_path)))
    v2_dir = root / "precision-resolved-v2"
    finalize_precision_manual_boundary_review(
        fixture.sample_dir,
        base_dir,
        review_path,
        v2_dir,
        tile_shape_yx=(16, 16),
    )
    return fixture, base_dir, v2_dir


def _prepare_v2_excluding_target(root: Path):
    fixture, base_dir = _prepare_base(root)
    pack = generate_precision_manual_boundary_review(
        fixture.sample_dir,
        base_dir,
        root / "manual-boundary-pack",
        patch_radius_px=128,
    )
    payload = _manual_review_payload(pack.page_path)
    payload["rows"][0]["manual_boundary_choice"] = "exclude"
    payload["rows"][0]["vertices_xy"] = []
    payload["rows"][0]["vertex_count"] = 0
    review_path = root / "manual-boundary-review.json"
    review_path.write_text(json.dumps(payload))
    v2_dir = root / "precision-resolved-v2"
    finalize_precision_manual_boundary_review(
        fixture.sample_dir,
        base_dir,
        review_path,
        v2_dir,
        tile_shape_yx=(16, 16),
    )
    return fixture, v2_dir


class TestRecallReviewedOverlay(unittest.TestCase):
    def test_reviewed_overlay_drives_masks_coverage_and_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, _, v2_dir = _prepare_v2(root)
            bundle = generate_recall_review_bundle(
                fixture.sample_dir,
                overlay_dir=v2_dir,
                window_radius_px=128,
                window_stride_px=192,
                overview_downsample=8,
            )
            metadata = json.loads(bundle.metadata_path.read_text())

            self.assertEqual(
                metadata["review_identity"]["overlay_delivery_name"],
                "precision_resolved_v2",
            )
            self.assertEqual(metadata["mask_overlay"]["candidate_count"], 2)
            self.assertEqual(len(metadata["candidates"]), 2)
            manual_row = next(
                row
                for row in metadata["candidates"]
                if row["detector_component_id"] == "manual-target"
            )
            self.assertEqual(
                manual_row["resolution_source"],
                "precision_manual_contour",
            )

            with RecallReviewRuntime(
                fixture.sample_dir,
                overlay_dir=v2_dir,
            ) as runtime:
                self.assertEqual(len(runtime.sample.candidates), 2)
                self.assertEqual(len(runtime.sample.detector_candidates), 2)
                # This point lies outside the rejected 10 px fragment but inside
                # the reviewed 34 px polygon.
                self.assertTrue(runtime._point_covered(188.0, 160.0))
                probe = runtime.probe(188.0, 160.0)
                self.assertTrue(probe["already_covered"])
                self.assertEqual(probe["failure_class"], "already_covered")
                payload = runtime.window_payload((160, 160), 128)
                target = next(
                    row
                    for row in payload["candidates"]
                    if row["detector_component_id"] == "manual-target"
                )
                self.assertEqual(
                    target["resolution_source"],
                    "precision_manual_contour",
                )
                overlay = np.asarray(
                    Image.open(
                        io.BytesIO(runtime.render_overlay((160, 160), 128))
                    )
                )
                self.assertGreater(int(overlay[128, 156, 3]), 0)

    def test_rejects_unresolved_delivery_and_pre_overlay_review_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, base_dir, v2_dir = _prepare_v2(root)
            with self.assertRaisesRegex(ValueError, "unresolved manual"):
                generate_recall_review_bundle(
                    fixture.sample_dir,
                    overlay_dir=base_dir,
                    window_radius_px=128,
                    window_stride_px=192,
                    overview_downsample=8,
                )

            bundle = generate_recall_review_bundle(
                fixture.sample_dir,
                overlay_dir=v2_dir,
                window_radius_px=128,
                window_stride_px=192,
                overview_downsample=8,
            )
            old_identity = _load_sample(fixture.sample_dir).review_identity
            old_review_path = root / "old-recall-review.json"
            old_review_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "oocyte_recall",
                        "sample": old_identity,
                        "windows": [],
                        "missing_oocytes": [],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "overlay_"):
                analyze_recall_review(
                    fixture.sample_dir,
                    old_review_path,
                    root / "old-analysis",
                )

            current_review_path = root / "current-recall-review.json"
            current_review_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "oocyte_recall",
                        "sample": json.loads(bundle.metadata_path.read_text())[
                            "review_identity"
                        ],
                        "windows": [],
                        "missing_oocytes": [],
                    }
                )
            )
            table_path = analyze_recall_review(
                fixture.sample_dir,
                current_review_path,
                root / "current-analysis",
            )
            self.assertTrue(table_path.is_file())
            analysis_summary = json.loads(
                (table_path.parent / "summary.json").read_text()
            )
            self.assertEqual(
                analysis_summary["sample"]["overlay_delivery_name"],
                "precision_resolved_v2",
            )

    def test_batch_no_generate_recovers_bound_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, _, v2_dir = _prepare_v2(root)
            generate_recall_review_bundle(
                fixture.sample_dir,
                overlay_dir=v2_dir,
                window_radius_px=128,
                window_stride_px=192,
                overview_downsample=8,
            )

            batch = generate_batch_recall_review_bundle(
                root,
                sample_ids=[fixture.sample_id],
                generate_samples=False,
            )

            self.assertEqual(batch.total_candidate_count, 2)
            manifest = json.loads(batch.manifest_path.read_text())
            self.assertEqual(
                manifest["samples"][0]["overlay_name"],
                "precision_resolved_v2",
            )
            self.assertIn("precision_resolved_v2", batch.index_path.read_text())

    def test_manual_seed_finalizer_carries_forward_reviewed_overlay_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, v2_dir = _prepare_v2_excluding_target(root)
            bundle = generate_recall_review_bundle(
                fixture.sample_dir,
                overlay_dir=v2_dir,
                window_radius_px=128,
                window_stride_px=192,
                overview_downsample=8,
            )
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
                                "annotation_id": "restored-target",
                                "window_id": metadata["windows"][0]["window_id"],
                                "x": 160,
                                "y": 160,
                                "notes": "reviewed Recall miss",
                            }
                        ],
                    }
                )
            )
            analysis_dir = root / "analysis"
            analyze_recall_review(fixture.sample_dir, recall_path, analysis_dir)
            manual_payload = _manual_seed_review_payload(
                analysis_dir / "manual_seed_review.html"
            )
            manual_payload["rows"][0][
                "manual_mask_choice"
            ] = "accept_manual_expanded"
            manual_path = root / "manual-review.json"
            manual_path.write_text(json.dumps(manual_payload))

            result = finalize_manual_seed_review(
                fixture.sample_dir,
                manual_path,
                root / "finalized",
                tile_shape_yx=(16, 16),
            )

            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.combined_labels.label_count, 2)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["production_candidate_count"], 1)
            self.assertEqual(
                manifest["sample"]["overlay_delivery_name"],
                "precision_resolved_v2",
            )


if __name__ == "__main__":
    unittest.main()
