import base64
import json
import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from PIL import Image

from aegle.oocyte.profiling import profile_oocyte_labels
from aegle.oocyte.release import build_oocyte_release, validate_oocyte_release


MAPPING_COLUMNS = [
    "label",
    "detector_component_id",
    "detector_score",
    "acceptance_mode",
    "center_x",
    "center_y",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "mask_path",
    "assigned_pixel_count",
    "overlap_pixel_count",
]


def _write_antibodies(path: Path) -> None:
    pd.DataFrame(
        [
            {"version": 2, "channel_id": "Channel:0:0", "antibody_name": "DAPI"},
            {"version": 2, "channel_id": "Channel:0:1", "antibody_name": "UCHL1"},
        ]
    ).to_csv(path, sep="\t", index=False)


def _mapping_row(labels: np.ndarray) -> dict:
    yy, xx = np.nonzero(labels == 1)
    return {
        "label": 1,
        "detector_component_id": "det_0001",
        "detector_score": 0.9,
        "acceptance_mode": "reviewed",
        "center_x": float(xx.mean()),
        "center_y": float(yy.mean()),
        "bbox_x0": int(xx.min()),
        "bbox_y0": int(yy.min()),
        "bbox_x1": int(xx.max()) + 1,
        "bbox_y1": int(yy.max()) + 1,
        "mask_path": "masks/det_0001.npz",
        "assigned_pixel_count": int(len(xx)),
        "overlap_pixel_count": 0,
    }


def _write_sample(root: Path, sample_id: str, *, positive: bool) -> dict:
    sample_root = root / sample_id
    sample_root.mkdir()
    raw = np.zeros((2, 32, 40), dtype=np.uint16)
    raw[0] = 10
    raw[1] = 20
    labels = np.zeros((32, 40), dtype=np.uint16)
    if positive:
        labels[8:18, 12:25] = 1
        raw[1, labels == 1] = 2000
    image_path = sample_root / "raw.ome.tiff"
    labels_path = sample_root / "labels.ome.tiff"
    mapping_path = sample_root / "mapping.csv"
    candidates_path = sample_root / "candidates.csv"
    antibodies_path = sample_root / "antibodies.tsv"
    tifffile.imwrite(image_path, raw, ome=True, metadata={"axes": "CYX"})
    tifffile.imwrite(labels_path, labels, ome=True, metadata={"axes": "YX"})
    _write_antibodies(antibodies_path)
    rows = [_mapping_row(labels)] if positive else []
    pd.DataFrame(rows, columns=MAPPING_COLUMNS).to_csv(mapping_path, index=False)
    pd.DataFrame(rows, columns=MAPPING_COLUMNS).to_csv(candidates_path, index=False)
    if positive:
        masks_dir = sample_root / "masks"
        masks_dir.mkdir()
        row = rows[0]
        x0, y0, x1, y1 = (
            int(row[name])
            for name in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
        )
        np.savez_compressed(
            masks_dir / "det_0001.npz",
            mask=labels[y0:y1, x0:x1] == 1,
            bbox_xyxy=np.asarray((x0, y0, x1, y1), dtype=np.int64),
            image_shape_yx=np.asarray(labels.shape, dtype=np.int64),
            metadata_json=np.asarray("{}"),
        )
    profiling_dir = sample_root / "profiling"
    profile_oocyte_labels(
        sample_id=sample_id,
        image_path=image_path,
        antibodies_path=antibodies_path,
        label_path=labels_path,
        mapping_path=mapping_path,
        out_dir=profiling_dir,
        pixel_size_um=0.5,
        label_scan_height_px=8,
    )
    review_path = sample_root / "review.json"
    provenance_path = sample_root / "provenance.json"
    review_path.write_text(json.dumps({"sample_id": sample_id}) + "\n")
    provenance_path.write_text(json.dumps({"source": "test"}) + "\n")
    entry = {
        "sample_id": sample_id,
        "role": "positive" if positive else "negative_control",
        "image": str(image_path),
        "antibodies": str(antibodies_path),
        "final_labels": str(labels_path),
        "final_mapping": str(mapping_path),
        "final_candidates": str(candidates_path),
        "profiling_dir": str(profiling_dir),
        "review_exports": [str(review_path)],
        "provenance_files": [str(provenance_path)],
    }
    if not positive:
        detector_path = sample_root / "detector.csv"
        rescue_path = sample_root / "rescue.csv"
        pd.DataFrame([{"accepted": False}]).to_csv(detector_path, index=False)
        pd.DataFrame([{"rescue_status": "failed_score"}]).to_csv(
            rescue_path,
            index=False,
        )
        entry["detector_candidates"] = str(detector_path)
        entry["rescue_diagnostics"] = str(rescue_path)
    return entry


class TestOocyteRelease(unittest.TestCase):
    def test_builds_and_validates_positive_and_negative_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            algorithm_path = root / "algorithm.html"
            algorithm_path.write_text("<html><body>algorithm</body></html>\n")
            spec_path = root / "release.yaml"
            spec_path.write_text(
                yaml.safe_dump(
                    {
                        "release_name": "synthetic_v1",
                        "algorithm_document": str(algorithm_path),
                        "samples": [
                            _write_sample(root, "positive", positive=True),
                            _write_sample(root, "negative", positive=False),
                        ],
                    },
                    sort_keys=False,
                )
            )
            release_dir = root / "release"

            result = build_oocyte_release(spec_path, release_dir)

            self.assertEqual(result.sample_count, 2)
            self.assertEqual(result.oocyte_count, 1)
            self.assertEqual(result.validation["negative_control_count"], 1)
            self.assertTrue((release_dir / "release_spec.yaml").is_file())
            self.assertTrue(
                (
                    release_dir
                    / "samples/positive/final/masks/mask-0001__det_0001.npz"
                ).is_file()
            )
            negative = pd.read_csv(
                release_dir / "samples/negative/final/oocyte_labels.csv"
            )
            self.assertTrue(negative.empty)
            batch = pd.read_csv(release_dir / "batch_oocyte_by_marker.csv")
            self.assertEqual(batch["oocyte_id"].tolist(), ["positive__det_0001"])
            positive_console = (
                release_dir / "samples/positive/review_console.html"
            ).read_text()
            negative_console = (
                release_dir / "samples/negative/review_console.html"
            ).read_text()
            self.assertEqual(
                len(
                    re.findall(
                        r"<article\b[^>]*\bdata-embedded-review-card(?:\s|>)",
                        positive_console,
                    )
                ),
                1,
            )
            encoded_images = re.findall(
                r"data:image/webp;base64,([A-Za-z0-9+/=]+)",
                positive_console,
            )
            self.assertEqual(len(encoded_images), 4)
            image_sizes = []
            for encoded in encoded_images:
                with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                    self.assertEqual(image.format, "WEBP")
                    image_sizes.append(image.size)
            self.assertEqual(image_sizes.count((360, 360)), 2)
            self.assertEqual(image_sizes.count((3, 2)), 2)
            self.assertEqual(positive_console.count("data-global-overview"), 1)
            self.assertEqual(
                len(
                    re.findall(
                        r"<circle\b[^>]*\bdata-global-hotspot(?:\s|>)",
                        positive_console,
                    )
                ),
                1,
            )
            self.assertIn('href="#oocyte-001"', positive_console)
            self.assertIn('id="oocyte-001"', positive_console)
            self.assertIn('data-card-target="oocyte-001"', positive_console)
            self.assertIn("best>Number(nearest.getAttribute('r'))**2", positive_console)
            self.assertIn(
                "hotspots.toggleAttribute('hidden',showRaw)",
                positive_console,
            )
            self.assertIn("Hide mask", positive_console)
            self.assertIn("Hide masks", positive_console)
            self.assertIn(".image-frame img[hidden]{display:none}", positive_console)
            self.assertIn("profiling/oocyte_by_marker.csv", positive_console)
            self.assertNotIn("/api/", positive_console)
            self.assertIn("No final oocytes", negative_console)
            self.assertIn("validated no-oocyte negative control", negative_console)
            negative_images = re.findall(
                r"data:image/webp;base64,([A-Za-z0-9+/=]+)",
                negative_console,
            )
            self.assertEqual(len(negative_images), 1)
            with Image.open(BytesIO(base64.b64decode(negative_images[0]))) as image:
                self.assertEqual(image.format, "WEBP")
                self.assertEqual(image.size, (3, 2))
            self.assertEqual(negative_console.count("data-global-overview"), 1)
            self.assertEqual(
                len(
                    re.findall(
                        r"<circle\b[^>]*\bdata-global-hotspot(?:\s|>)",
                        negative_console,
                    )
                ),
                0,
            )
            self.assertIn("No final masks in this negative control", negative_console)
            positive_manifest = json.loads(
                (
                    release_dir
                    / "samples/positive/sample_release_manifest.json"
                ).read_text()
            )
            negative_manifest = json.loads(
                (
                    release_dir
                    / "samples/negative/sample_release_manifest.json"
                ).read_text()
            )
            self.assertEqual(
                positive_manifest["embedded_console"]["overview_webp_count"],
                2,
            )
            self.assertEqual(
                positive_manifest["embedded_console"]["global_hotspot_count"],
                1,
            )
            self.assertEqual(
                negative_manifest["embedded_console"]["overview_webp_count"],
                1,
            )
            self.assertEqual(validate_oocyte_release(release_dir)["status"], "valid")
            with self.assertRaises(FileExistsError):
                build_oocyte_release(spec_path, release_dir)

            readme = release_dir / "README.md"
            readme.write_text(readme.read_text() + "tampered\n")
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                validate_oocyte_release(release_dir)

    def test_rejects_nonzero_negative_control_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = _write_sample(root, "negative", positive=False)
            pd.DataFrame([{"accepted": True}]).to_csv(
                entry["detector_candidates"],
                index=False,
            )
            spec_path = root / "release.yaml"
            spec_path.write_text(
                yaml.safe_dump(
                    {"release_name": "invalid_v1", "samples": [entry]},
                    sort_keys=False,
                )
            )

            with self.assertRaisesRegex(ValueError, "accepted detector objects"):
                build_oocyte_release(spec_path, root / "release")
            self.assertFalse((root / "release").exists())


if __name__ == "__main__":
    unittest.main()
