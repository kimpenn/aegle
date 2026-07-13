import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from aegle.oocyte import DONOR13_V6
from aegle.oocyte.io import save_candidate_mask
from aegle.oocyte.models import BoundingBox, ExtractedPatch, SegmentationMetrics
from aegle.oocyte.recall_review import (
    DEFAULT_OVERVIEW_DOWNSAMPLE,
    DEFAULT_WINDOW_RADIUS_PX,
    DEFAULT_WINDOW_STRIDE_PX,
    RecallReviewRuntime,
    _axis_centers,
    _handler_for,
    _segment_manual_seed_patch,
    analyze_recall_review,
    classify_recall_failure,
    generate_recall_review_bundle,
)


class RecallReviewFixture:
    def __init__(self, root: Path, sample_id: str = "synthetic-recall"):
        self.root = root
        self.sample_id = sample_id
        self.sample_dir = root / self.sample_id
        self.sample_dir.mkdir()
        image = np.full((2, 768, 768), 100, dtype=np.uint16)
        yy, xx = np.ogrid[:768, :768]
        accepted = (yy - 380) ** 2 + (xx - 380) ** 2 <= 36**2
        missed = (yy - 160) ** 2 + (xx - 160) ** 2 <= 32**2
        image[0, accepted] = 12000
        image[0, missed] = 10000
        self.image_path = root / f"{sample_id}-source.ome.tiff"
        tifffile.imwrite(
            self.image_path,
            image,
            ome=True,
            metadata={"axes": "CYX"},
        )
        bbox = BoundingBox(344, 344, 417, 417)
        cropped = np.asarray(accepted[344:417, 344:417], dtype=np.bool_)
        metrics = SegmentationMetrics(
            threshold_method="triangle",
            base_threshold=500.0,
            annulus_floor=100.0,
            threshold=500.0,
            selection_mode="center_component",
            area_px=int(cropped.sum()),
            equivalent_diameter_um=36.0,
            major_axis_um=36.0,
            minor_axis_um=36.0,
            eccentricity=0.0,
            solidity=0.99,
            circularity=0.95,
            centroid_y_px=36.0,
            centroid_x_px=36.0,
            centroid_offset_px=0.0,
            mean_intensity=12000.0,
            max_intensity=12000.0,
        )
        mask_path = self.sample_dir / "masks/accepted.npz"
        save_candidate_mask(
            mask_path,
            mask=cropped,
            bbox=bbox,
            image_shape_yx=(768, 768),
            sample_id=self.sample_id,
            candidate_id="accepted-1",
            profile_name=DONOR13_V6.profile_name,
            profile_fingerprint=DONOR13_V6.fingerprint(),
            metrics=metrics,
            implementation_version="test-v1",
        )
        pd.DataFrame(
            [
                {
                    "detector_component_id": "accepted-1",
                    "display_id": "#001",
                    "accepted": True,
                    "detector_score": 0.9,
                    "center_x": 380,
                    "center_y": 380,
                    "component_centroid_x": 380,
                    "component_centroid_y": 380,
                    "bbox_x0": 344,
                    "bbox_y0": 344,
                    "bbox_x1": 417,
                    "bbox_y1": 417,
                    "mask_path": "masks/accepted.npz",
                    "mask_source_dir": str(self.sample_dir),
                    "detection_pass": "baseline_v6",
                }
            ]
        ).to_csv(self.sample_dir / "html_candidates.csv", index=False)
        pd.DataFrame(
            [
                {
                    "detector_component_id": "accepted-1",
                    "accepted": True,
                    "detector_score": 0.9,
                    "center_x": 380,
                    "center_y": 380,
                    "component_centroid_x": 380,
                    "component_centroid_y": 380,
                },
                {
                    "detector_component_id": "rejected-1",
                    "accepted": False,
                    "detector_score": 0.42,
                    "center_x": 160,
                    "center_y": 160,
                    "component_centroid_x": 160,
                    "component_centroid_y": 160,
                },
            ]
        ).to_csv(self.sample_dir / "candidates.csv", index=False)
        pd.DataFrame(
            [
                {
                    "detector_component_id": "coarse-1",
                    "coarse_center_x": 160,
                    "coarse_center_y": 160,
                }
            ]
        ).to_csv(self.sample_dir / "coarse_candidates.csv", index=False)
        pd.DataFrame(
            columns=["detector_component_id", "duplicate_of"]
        ).to_csv(self.sample_dir / "combined_duplicate_suppressed.csv", index=False)
        (self.sample_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "sample_id": self.sample_id,
                    "source_image": str(self.image_path),
                    "source_image_size_bytes": self.image_path.stat().st_size,
                    "resolved_channel_index": 0,
                    "resolved_config": DONOR13_V6.to_dict(),
                    "profile_fingerprint": DONOR13_V6.fingerprint(),
                    "implementation_version": "test-v1",
                }
            )
        )
        (self.sample_dir / "summary.json").write_text(
            json.dumps(
                {
                    "sample_id": self.sample_id,
                    "image_shape_yx": [768, 768],
                    "profile_name": DONOR13_V6.profile_name,
                }
            )
        )

    def generate(self):
        return generate_recall_review_bundle(
            self.sample_dir,
            window_radius_px=128,
            window_stride_px=192,
            overview_downsample=8,
        )


class TestRecallFailureClassification(unittest.TestCase):
    def test_classifies_each_detector_stage(self):
        common = {
            "nearest_suppressed_distance_px": None,
            "nearest_coarse_distance_px": None,
            "nearest_refined_distance_px": None,
        }
        self.assertEqual(
            classify_recall_failure(already_covered=True, **common),
            "already_covered",
        )
        self.assertEqual(
            classify_recall_failure(
                already_covered=False,
                nearest_suppressed_distance_px=10,
                nearest_coarse_distance_px=10,
                nearest_refined_distance_px=10,
            ),
            "dedup_error",
        )
        self.assertEqual(
            classify_recall_failure(already_covered=False, **common),
            "proposal_miss",
        )
        self.assertEqual(
            classify_recall_failure(
                already_covered=False,
                nearest_suppressed_distance_px=None,
                nearest_coarse_distance_px=20,
                nearest_refined_distance_px=None,
            ),
            "segmentation_miss",
        )
        self.assertEqual(
            classify_recall_failure(
                already_covered=False,
                nearest_suppressed_distance_px=None,
                nearest_coarse_distance_px=300,
                nearest_refined_distance_px=15,
            ),
            "acceptance_miss",
        )

    def test_manual_seed_prefers_the_component_nearest_the_click(self):
        yy, xx = np.ogrid[:361, :361]
        patch = np.full((361, 361), 100.0, dtype=np.float32)
        target = ((yy - 180) ** 2 + (xx - 180) ** 2 <= 34**2) & (
            (yy - 180) ** 2 + (xx - 180) ** 2 >= 19**2
        )
        target &= ~((xx >= 176) & (xx <= 184) & (yy < 180))
        larger_neighbor = (yy - 70) ** 2 + (xx - 70) ** 2 <= 45**2
        patch[target] = 10000.0
        patch[larger_neighbor] = 10000.0

        result = _segment_manual_seed_patch(
            patch,
            annulus_floor_percentile=95.0,
        )

        self.assertEqual(
            result.metrics.selection_mode,
            "manual_seed_nearest_component",
        )
        self.assertLess(result.metrics.centroid_offset_px, 35.0)
        self.assertFalse(bool(result.mask[70, 70]))

    def test_manual_seed_watershed_splits_touching_oocytes(self):
        yy, xx = np.ogrid[:201, :201]
        patch = np.full((201, 201), 100.0, dtype=np.float32)
        upper = (yy - 100) ** 2 + (xx - 100) ** 2 <= 29**2
        lower = (yy - 145) ** 2 + (xx - 108) ** 2 <= 28**2
        patch[upper | lower] = 10000.0

        result = _segment_manual_seed_patch(
            patch,
            annulus_floor_percentile=80.0,
            annulus_inner_px=40,
            annulus_outer_px=90,
        )

        self.assertEqual(
            result.metrics.selection_mode,
            "manual_seed_watershed_component",
        )
        self.assertTrue(bool(result.mask[100, 100]))
        self.assertFalse(bool(result.mask[145, 108]))

    def test_shape_recovery_is_opt_in_for_large_round_expansion(self):
        yy, xx = np.ogrid[:201, :201]
        patch = np.full((201, 201), 100.0, dtype=np.float32)
        target = (yy - 100) ** 2 + (xx - 100) ** 2 <= 30**2
        bright_fragment = (yy - 106) ** 2 + (xx - 112) ** 2 <= 14**2
        distance = np.sqrt((yy - 100) ** 2 + (xx - 100) ** 2)
        angle = (np.arctan2(yy - 100, xx - 100) + 2 * np.pi) % (2 * np.pi)
        bright_annulus_sector = (
            (distance >= 45) & (distance <= 85) & (angle < 0.75)
        )
        patch[target] = 3500.0
        patch[bright_fragment] = 7000.0
        patch[bright_annulus_sector] = 5000.0

        class Source:
            def read_patch(self, _center, _radius):
                return ExtractedPatch(
                    image=patch,
                    bbox=BoundingBox(0, 0, 201, 201),
                    image_shape_yx=(201, 201),
                    padding_tblr=(0, 0, 0, 0),
                )

        runtime = object.__new__(RecallReviewRuntime)
        runtime.source = Source()
        standard = runtime.segment_manual_provisionals(100, 100)
        recovered = runtime.segment_manual_provisionals(
            100,
            100,
            allow_shape_recovery=True,
        )

        self.assertEqual(standard.expanded_percentile, 95.0)
        self.assertEqual(recovered.expanded_percentile, 85.0)
        self.assertGreater(
            recovered.expanded.metrics.area_px,
            4 * standard.expanded.metrics.area_px,
        )
        self.assertGreater(recovered.expanded.metrics.circularity, 0.8)
        self.assertGreater(recovered.expanded.metrics.solidity, 0.9)


class TestRecallReviewBundle(unittest.TestCase):
    def test_survey_defaults_cover_real_13_21_shape_in_117_windows(self):
        self.assertEqual(DEFAULT_WINDOW_RADIUS_PX, 1280)
        self.assertEqual(DEFAULT_WINDOW_STRIDE_PX, 2304)
        self.assertEqual(DEFAULT_OVERVIEW_DOWNSAMPLE, 16)

        def assert_axis_coverage(length):
            centers = _axis_centers(
                length,
                DEFAULT_WINDOW_RADIUS_PX,
                DEFAULT_WINDOW_STRIDE_PX,
            )
            intervals = [
                (
                    max(0, center - DEFAULT_WINDOW_RADIUS_PX),
                    min(length, center + DEFAULT_WINDOW_RADIUS_PX + 1),
                )
                for center in centers
            ]
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], length)
            self.assertTrue(
                all(right[0] <= left[1] for left, right in zip(intervals, intervals[1:]))
            )
            return centers

        x_centers = assert_axis_coverage(29665)
        y_centers = assert_axis_coverage(20705)
        self.assertEqual(len(x_centers), 13)
        self.assertEqual(len(y_centers), 9)
        self.assertEqual(len(x_centers) * len(y_centers), 117)

    def test_generates_coordinate_faithful_bundle_and_exact_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            bundle = fixture.generate()

            self.assertTrue(bundle.page_path.is_file())
            self.assertTrue(bundle.overview_path.is_file())
            metadata = json.loads(bundle.metadata_path.read_text())
            self.assertEqual(metadata["image_width"], 768)
            self.assertEqual(metadata["image_height"], 768)
            self.assertIsInstance(
                metadata["review_identity"]["source_image_mtime_ns"],
                str,
            )
            self.assertGreater(bundle.window_count, 4)
            identity = metadata["review_identity"]
            self.assertEqual(identity["recall_coverage_profile"], "custom")
            self.assertEqual(identity["recall_window_count"], bundle.window_count)
            self.assertEqual(len(identity["recall_window_geometry_sha256"]), 64)
            self.assertIn("Add missed oocyte", bundle.page_path.read_text())
            self.assertIn("Next unreviewed", bundle.page_path.read_text())
            self.assertIn(
                "recall_window_geometry_sha256",
                bundle.page_path.read_text(),
            )

            with RecallReviewRuntime(fixture.sample_dir) as runtime:
                patch_bytes = runtime.render_patch((380, 380), 128, "local")
                patch = Image.open(io.BytesIO(patch_bytes))
                self.assertEqual(patch.size, (257, 257))
                overlay_bytes = runtime.render_overlay((380, 380), 128)
                overlay = np.asarray(Image.open(io.BytesIO(overlay_bytes)))
                self.assertEqual(overlay.shape, (257, 257, 4))
                self.assertGreater(int((overlay[..., 3] > 0).sum()), 100)
                edge = Image.open(
                    io.BytesIO(runtime.render_patch((0, 0), 128, "local"))
                )
                self.assertEqual(edge.size, (257, 257))
                payload = runtime.window_payload((380, 380), 128)
                self.assertEqual(len(payload["candidates"]), 1)
                self.assertEqual(payload["candidates"][0]["display_id"], "#001")

    def test_probe_and_offline_ingestion_preserve_production_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            bundle = fixture.generate()
            candidates_before = (fixture.sample_dir / "html_candidates.csv").read_bytes()
            metadata = json.loads(bundle.metadata_path.read_text())
            with RecallReviewRuntime(fixture.sample_dir) as runtime:
                covered = runtime.probe(380, 380)
                missed = runtime.probe(160, 160)
            self.assertEqual(covered["failure_class"], "already_covered")
            self.assertEqual(missed["failure_class"], "acceptance_miss")
            self.assertIsNotNone(missed["manual_conservative_metrics"])
            self.assertIsNotNone(missed["manual_expanded_metrics"])
            self.assertIn(
                missed["manual_conservative_percentile"],
                (95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0),
            )

            review_path = Path(directory) / "review.json"
            review_path.write_text(
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
                                "x": 160,
                                "y": 160,
                                "notes": "synthetic missed circle",
                            }
                        ],
                    }
                )
            )
            table_path = analyze_recall_review(
                fixture.sample_dir,
                review_path,
                Path(directory) / "analysis",
            )
            table = pd.read_csv(table_path)
            self.assertEqual(table.loc[0, "failure_class"], "acceptance_miss")
            self.assertTrue(
                Path(table.loc[0, "manual_conservative_mask_path"]).is_file()
            )
            self.assertTrue(
                Path(table.loc[0, "manual_expanded_mask_path"]).is_file()
            )
            self.assertTrue(
                (Path(directory) / "analysis/manual_seed_review.html").is_file()
            )
            self.assertIn(
                'href="../recall_review.html"',
                (Path(directory) / "analysis/manual_seed_review.html").read_text(),
            )
            self.assertTrue(
                (Path(directory) / "analysis/review_assets/seed-001.webp").is_file()
            )
            analysis_summary = json.loads(
                (Path(directory) / "analysis/summary.json").read_text()
            )
            self.assertEqual(
                analysis_summary["sample"]["recall_window_geometry_sha256"],
                metadata["review_identity"]["recall_window_geometry_sha256"],
            )
            self.assertEqual(
                candidates_before,
                (fixture.sample_dir / "html_candidates.csv").read_bytes(),
            )

    def test_http_routes_return_health_images_and_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            fixture.generate()
            with RecallReviewRuntime(fixture.sample_dir) as runtime:
                server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(runtime))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                try:
                    health = json.loads(urlopen(base + "/health", timeout=3).read())
                    self.assertTrue(health["read_only"])
                    patch = urlopen(
                        base + "/api/patch.webp?x=380&y=380&radius=128",
                        timeout=3,
                    ).read()
                    self.assertGreater(len(patch), 1000)
                    with self.assertRaises(HTTPError) as context:
                        urlopen(base + "/api/patch.webp?x=-1&y=0&radius=128")
                    self.assertEqual(context.exception.code, 400)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)

    def test_ingestion_rejects_wrong_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            bundle = fixture.generate()
            metadata = json.loads(bundle.metadata_path.read_text())
            metadata["review_identity"]["sample_id"] = "wrong-sample"
            review_path = Path(directory) / "wrong.json"
            review_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "oocyte_recall",
                        "sample": metadata["review_identity"],
                        "windows": [],
                        "missing_oocytes": [],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "sample_id"):
                analyze_recall_review(
                    fixture.sample_dir,
                    review_path,
                    Path(directory) / "analysis",
                )

    def test_ingestion_rejects_review_from_a_different_coverage_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            bundle = fixture.generate()
            metadata = json.loads(bundle.metadata_path.read_text())
            stale_identity = dict(metadata["review_identity"])
            stale_identity["recall_window_geometry_sha256"] = "0" * 64
            review_path = Path(directory) / "stale-grid.json"
            review_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_type": "oocyte_recall",
                        "sample": stale_identity,
                        "windows": [],
                        "missing_oocytes": [],
                    }
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "recall_window_geometry_sha256",
            ):
                analyze_recall_review(
                    fixture.sample_dir,
                    review_path,
                    Path(directory) / "analysis",
                )

    def test_http_window_route_accepts_survey_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            fixture.generate()
            with RecallReviewRuntime(fixture.sample_dir) as runtime:
                server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(runtime))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                try:
                    payload = json.loads(
                        urlopen(
                            base + "/api/window?x=380&y=380&radius=1280",
                            timeout=3,
                        ).read()
                    )
                    self.assertEqual(
                        payload["bbox"],
                        {"x0": 0, "y0": 0, "x1": 768, "y1": 768},
                    )
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)

    def test_runtime_rejects_metadata_when_grid_changes_without_new_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecallReviewFixture(Path(directory))
            bundle = fixture.generate()
            metadata = json.loads(bundle.metadata_path.read_text())
            metadata["windows"][0]["center_x"] += 1
            bundle.metadata_path.write_text(json.dumps(metadata))

            with self.assertRaisesRegex(ValueError, "coverage identity"):
                RecallReviewRuntime(fixture.sample_dir)


if __name__ == "__main__":
    unittest.main()
