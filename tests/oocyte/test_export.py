import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from aegle.oocyte import DONOR13_V6
from aegle.oocyte.export import export_whole_slide_labels
from aegle.oocyte.io import save_candidate_mask
from aegle.oocyte.models import BoundingBox, SegmentationMetrics


def _metrics(area_px):
    return SegmentationMetrics(
        threshold_method="triangle",
        base_threshold=10.0,
        annulus_floor=5.0,
        threshold=10.0,
        selection_mode="center_component",
        area_px=area_px,
        equivalent_diameter_um=20.0,
        major_axis_um=20.0,
        minor_axis_um=20.0,
        eccentricity=0.0,
        solidity=1.0,
        circularity=1.0,
        centroid_y_px=4.0,
        centroid_x_px=4.0,
        centroid_offset_px=0.0,
        mean_intensity=1000.0,
        max_intensity=1200.0,
    )


class TestWholeSlideLabelExport(unittest.TestCase):
    def test_composes_persisted_masks_with_score_ordered_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp)
            masks_dir = sample_dir / "masks"
            first = np.ones((8, 8), dtype=np.bool_)
            second = np.ones((8, 8), dtype=np.bool_)
            for candidate_id, mask, bbox in (
                ("det_0000", first, BoundingBox(10, 10, 18, 18)),
                ("det_0001", second, BoundingBox(14, 14, 22, 22)),
            ):
                save_candidate_mask(
                    masks_dir / f"{candidate_id}.npz",
                    mask=mask,
                    bbox=bbox,
                    image_shape_yx=(64, 64),
                    sample_id="test",
                    candidate_id=candidate_id,
                    profile_name=DONOR13_V6.profile_name,
                    profile_fingerprint=DONOR13_V6.fingerprint(),
                    metrics=_metrics(int(mask.sum())),
                )
            candidates = pd.DataFrame(
                [
                    {
                        "detector_component_id": "det_0001",
                        "accepted": True,
                        "detector_score": 0.7,
                        "acceptance_mode": "strict",
                        "center_x": 18,
                        "center_y": 18,
                        "mask_path": "masks/det_0001.npz",
                    },
                    {
                        "detector_component_id": "det_0000",
                        "accepted": True,
                        "detector_score": 0.9,
                        "acceptance_mode": "strict",
                        "center_x": 14,
                        "center_y": 14,
                        "mask_path": "masks/det_0000.npz",
                    },
                ]
            )

            result = export_whole_slide_labels(
                candidates,
                sample_dir=sample_dir,
                image_shape_yx=(64, 64),
                image_path=sample_dir / "oocyte_labels.ome.tiff",
                mapping_path=sample_dir / "oocyte_labels.csv",
                tile_shape_yx=(16, 16),
            )

            labels = tifffile.imread(result.image_path)
            mapping = pd.read_csv(result.mapping_path)
            self.assertEqual(result.label_count, 2)
            self.assertEqual(int((labels == 1).sum()), 64)
            self.assertEqual(int((labels == 2).sum()), 48)
            self.assertEqual(result.overlap_pixel_count, 16)
            self.assertEqual(mapping.iloc[0]["detector_component_id"], "det_0000")
            self.assertEqual(mapping.iloc[1]["overlap_pixel_count"], 16)


if __name__ == "__main__":
    unittest.main()
