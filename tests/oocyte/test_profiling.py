import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte.profiling import profile_oocyte_labels


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


def _write_antibodies(path: Path, names: list[str]) -> None:
    rows = [
        {
            "version": 2,
            "channel_id": f"Channel:0:{index}",
            "antibody_name": name,
        }
        for index, name in enumerate(names)
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_mapping(path: Path, labels: np.ndarray, label_ids: list[int]) -> None:
    rows = []
    for label_id in label_ids:
        yy, xx = np.nonzero(labels == label_id)
        rows.append(
            {
                "label": label_id,
                "detector_component_id": f"det_{label_id:04d}",
                "detector_score": 0.9,
                "acceptance_mode": "reviewed",
                "center_x": int(np.round(xx.mean())),
                "center_y": int(np.round(yy.mean())),
                "bbox_x0": max(0, int(xx.min()) - 1),
                "bbox_y0": max(0, int(yy.min()) - 1),
                "bbox_x1": min(labels.shape[1], int(xx.max()) + 2),
                "bbox_y1": min(labels.shape[0], int(yy.max()) + 2),
                "mask_path": f"masks/det_{label_id:04d}.npz",
                "assigned_pixel_count": int(len(xx)),
                "overlap_pixel_count": 0,
            }
        )
    pd.DataFrame(rows, columns=MAPPING_COLUMNS).to_csv(path, index=False)


class TestOocyteProfiling(unittest.TestCase):
    def test_profiles_sparse_labels_edges_duplicate_markers_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            height, width = 32, 40
            yy, xx = np.mgrid[:height, :width]
            labels = np.zeros((height, width), dtype=np.uint16)
            labels[0:4, 0:5] = 2
            labels[20:28, 30:39] = 7
            channels = np.stack(
                [
                    (yy * 10 + xx).astype(np.uint16),
                    (100 + yy * 2 + xx * 3).astype(np.uint16),
                    (1000 + yy + xx * 4).astype(np.uint16),
                ]
            )
            image_path = root / "raw.ome.tiff"
            label_path = root / "labels.ome.tiff"
            mapping_path = root / "mapping.csv"
            antibodies_path = root / "antibodies.tsv"
            candidates_path = root / "candidates.csv"
            tifffile.imwrite(
                image_path,
                channels,
                ome=True,
                metadata={"axes": "CYX"},
            )
            tifffile.imwrite(
                label_path,
                labels,
                ome=True,
                metadata={"axes": "YX"},
                tile=(16, 16),
                compression="zlib",
            )
            _write_mapping(mapping_path, labels, [2, 7])
            _write_antibodies(antibodies_path, ["DAPI", "UCHL1", "UCHL1"])
            pd.DataFrame(
                [
                    {
                        "detector_component_id": "det_0002",
                        "detection_pass": "baseline_v6",
                        "boundary_warning": "False",
                    },
                    {
                        "detector_component_id": "det_0007",
                        "detection_pass": "manual_seed",
                        "boundary_warning": "True",
                    },
                ]
            ).to_csv(candidates_path, index=False)

            result = profile_oocyte_labels(
                sample_id="sample-a",
                image_path=image_path,
                antibodies_path=antibodies_path,
                label_path=label_path,
                mapping_path=mapping_path,
                candidates_path=candidates_path,
                out_dir=root / "profiling",
                pixel_size_um=0.5,
                max_region_height_px=8,
                merge_gap_px=0,
                label_scan_height_px=8,
            )

            markers = pd.read_csv(result.artifact_paths["markers"])
            metadata = pd.read_csv(result.artifact_paths["metadata"])
            overview = pd.read_csv(result.artifact_paths["overview"])
            channel_manifest = pd.read_csv(result.artifact_paths["channels"])
            manifest = json.loads(result.artifact_paths["manifest"].read_text())

            self.assertEqual(result.oocyte_count, 2)
            self.assertEqual(result.channel_count, 3)
            self.assertEqual(
                list(markers.columns),
                [
                    "sample_id",
                    "oocyte_id",
                    "label_id",
                    "DAPI",
                    "UCHL1",
                    "UCHL1_1",
                ],
            )
            for row_index, label_id in enumerate([2, 7]):
                mask = labels == label_id
                self.assertEqual(markers.loc[row_index, "label_id"], label_id)
                self.assertAlmostEqual(markers.loc[row_index, "DAPI"], channels[0][mask].mean())
                self.assertAlmostEqual(markers.loc[row_index, "UCHL1"], channels[1][mask].mean())
                self.assertAlmostEqual(markers.loc[row_index, "UCHL1_1"], channels[2][mask].mean())
                self.assertEqual(metadata.loc[row_index, "area_px"], int(mask.sum()))
                self.assertAlmostEqual(
                    metadata.loc[row_index, "area_um2"],
                    float(mask.sum()) * 0.25,
                )
            self.assertEqual(metadata.loc[0, "bbox_x0"], 0)
            self.assertEqual(metadata.loc[0, "bbox_y0"], 0)
            self.assertEqual(metadata.loc[1, "detection_pass"], "manual_seed")
            self.assertTrue(bool(metadata.loc[1, "boundary_warning"]))
            self.assertEqual(len(overview), 2)
            self.assertEqual(
                channel_manifest["measurement_class"].tolist(),
                ["nuclear_stain", "protein_marker", "protein_marker"],
            )
            self.assertEqual(manifest["measurement"]["statistic"], "raw_within_mask_mean")
            self.assertEqual(manifest["bounded_read_plan"]["largest_region_shape_yx"][0], 8)
            for filename, artifact in manifest["artifacts"].items():
                path = result.output_dir / filename
                self.assertEqual(artifact["size_bytes"], path.stat().st_size)
                self.assertEqual(
                    artifact["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_zero_label_control_writes_header_only_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels = np.zeros((2, 16, 16), dtype=np.uint16)
            labels = np.zeros((16, 16), dtype=np.uint16)
            image_path = root / "raw.ome.tiff"
            label_path = root / "labels.ome.tiff"
            mapping_path = root / "mapping.csv"
            antibodies_path = root / "antibodies.tsv"
            tifffile.imwrite(image_path, channels, ome=True, metadata={"axes": "CYX"})
            tifffile.imwrite(label_path, labels, ome=True, metadata={"axes": "YX"})
            pd.DataFrame(columns=MAPPING_COLUMNS).to_csv(mapping_path, index=False)
            _write_antibodies(antibodies_path, ["DAPI", "UCHL1"])

            result = profile_oocyte_labels(
                sample_id="negative",
                image_path=image_path,
                antibodies_path=antibodies_path,
                label_path=label_path,
                mapping_path=mapping_path,
                out_dir=root / "profiling",
                pixel_size_um=0.5,
                label_scan_height_px=8,
            )

            markers = pd.read_csv(result.artifact_paths["markers"])
            metadata = pd.read_csv(result.artifact_paths["metadata"])
            self.assertTrue(markers.empty)
            self.assertTrue(metadata.empty)
            self.assertEqual(
                list(markers.columns),
                ["sample_id", "oocyte_id", "label_id", "DAPI", "UCHL1"],
            )
            self.assertEqual(result.oocyte_count, 0)

    def test_rejects_unmapped_label_and_assigned_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels = np.ones((2, 16, 16), dtype=np.uint16)
            labels = np.zeros((16, 16), dtype=np.uint16)
            labels[2:6, 2:6] = 1
            image_path = root / "raw.ome.tiff"
            label_path = root / "labels.ome.tiff"
            mapping_path = root / "mapping.csv"
            antibodies_path = root / "antibodies.tsv"
            tifffile.imwrite(image_path, channels, ome=True, metadata={"axes": "CYX"})
            tifffile.imwrite(label_path, labels, ome=True, metadata={"axes": "YX"})
            pd.DataFrame(columns=MAPPING_COLUMNS).to_csv(mapping_path, index=False)
            _write_antibodies(antibodies_path, ["DAPI", "UCHL1"])
            with self.assertRaisesRegex(ValueError, "label sets differ"):
                profile_oocyte_labels(
                    sample_id="bad",
                    image_path=image_path,
                    antibodies_path=antibodies_path,
                    label_path=label_path,
                    mapping_path=mapping_path,
                    out_dir=root / "unmapped",
                    pixel_size_um=0.5,
                )

            _write_mapping(mapping_path, labels, [1])
            mapping = pd.read_csv(mapping_path)
            mapping.loc[0, "assigned_pixel_count"] += 1
            mapping.to_csv(mapping_path, index=False)
            with self.assertRaisesRegex(ValueError, "assigned_pixel_count mismatch"):
                profile_oocyte_labels(
                    sample_id="bad",
                    image_path=image_path,
                    antibodies_path=antibodies_path,
                    label_path=label_path,
                    mapping_path=mapping_path,
                    out_dir=root / "bad-count",
                    pixel_size_um=0.5,
                )

    def test_rejects_channel_table_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels = np.ones((2, 16, 16), dtype=np.uint16)
            labels = np.zeros((16, 16), dtype=np.uint16)
            image_path = root / "raw.ome.tiff"
            label_path = root / "labels.ome.tiff"
            mapping_path = root / "mapping.csv"
            antibodies_path = root / "antibodies.tsv"
            tifffile.imwrite(image_path, channels, ome=True, metadata={"axes": "CYX"})
            tifffile.imwrite(label_path, labels, ome=True, metadata={"axes": "YX"})
            pd.DataFrame(columns=MAPPING_COLUMNS).to_csv(mapping_path, index=False)
            _write_antibodies(antibodies_path, ["UCHL1"])

            with self.assertRaisesRegex(ValueError, "row count"):
                profile_oocyte_labels(
                    sample_id="bad",
                    image_path=image_path,
                    antibodies_path=antibodies_path,
                    label_path=label_path,
                    mapping_path=mapping_path,
                    out_dir=root / "profiling",
                    pixel_size_um=0.5,
                )


if __name__ == "__main__":
    unittest.main()
