import unittest

import numpy as np
import pandas as pd

from aegle.oocyte import (
    DONOR13_V6,
    detect_coarse_candidates,
    refine_candidates_from_array,
)
from aegle.oocyte.detection import (
    _deduplicate_coarse_rows,
    build_downsampled_mean_map_from_array,
)


class TestCoarseDetection(unittest.TestCase):
    def test_strip_downsampling_computes_block_means(self):
        image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        result = build_downsampled_mean_map_from_array(image, DONOR13_V6.coarse)
        expected = image.reshape(2, 8, 2, 8).mean(axis=(1, 3))
        np.testing.assert_array_equal(result, expected)

    def test_coarse_row_dedup_prefers_peak_at_equal_intensity(self):
        common = {
            "coarse_max_ds": 100.0,
            "coarse_area_ds": 30,
            "coarse_center_x": 100,
            "coarse_center_y": 100,
        }
        rows = [
            {**common, "coarse_seed_kind": "global_peak"},
            {**common, "coarse_seed_kind": "peak"},
        ]
        result = _deduplicate_coarse_rows(rows, merge_distance_px=60.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["coarse_seed_kind"], "peak")

    def test_detects_bright_synthetic_regions(self):
        yy, xx = np.ogrid[:256, :256]
        image = np.full((256, 256), 100.0, dtype=np.float32)
        image[(yy - 70) ** 2 + (xx - 80) ** 2 <= 7**2] = 8000.0
        image[(yy - 180) ** 2 + (xx - 170) ** 2 <= 8**2] = 12000.0

        result = detect_coarse_candidates(image, DONOR13_V6)

        self.assertGreaterEqual(len(result.candidates), 2)
        centers = result.candidates[["coarse_center_x", "coarse_center_y"]].to_numpy()
        for expected_x, expected_y in ((80 * 8 + 4, 70 * 8 + 4), (170 * 8 + 4, 180 * 8 + 4)):
            distances = np.hypot(
                centers[:, 0] - expected_x,
                centers[:, 1] - expected_y,
            )
            self.assertLessEqual(float(distances.min()), 8.0)
        self.assertEqual(result.mask.dtype, np.bool_)
        self.assertEqual(result.contrast.shape, image.shape)

    def test_refines_two_instances_from_one_coarse_patch(self):
        yy, xx = np.mgrid[:701, :701]
        image = np.full((701, 701), 100.0, dtype=np.float32)
        image += (
            10000.0
            * np.exp(-((yy - 350) ** 2 + (xx - 280) ** 2) / (2 * 22**2))
        ).astype(np.float32)
        image += (
            12000.0
            * np.exp(-((yy - 350) ** 2 + (xx - 420) ** 2) / (2 * 22**2))
        ).astype(np.float32)
        coarse = pd.DataFrame(
            [
                {
                    "detector_component_id": "det_0000",
                    "coarse_center_x": 350,
                    "coarse_center_y": 350,
                    "coarse_max_ds": 12000.0,
                }
            ]
        )

        result = refine_candidates_from_array(image, coarse, DONOR13_V6)

        accepted = result.candidates[result.candidates["accepted"]]
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(result.candidate_masks), 2)
        centers = accepted[["center_x", "center_y"]].to_numpy()
        for expected_x in (280, 420):
            distances = np.hypot(centers[:, 0] - expected_x, centers[:, 1] - 350)
            self.assertLessEqual(float(distances.min()), 2.0)
        for candidate_mask in result.candidate_masks.values():
            self.assertGreater(int(candidate_mask.mask.sum()), 2000)

    def test_rejects_tiny_bright_punctum_during_refinement(self):
        yy, xx = np.mgrid[:401, :401]
        image = np.full((401, 401), 100.0, dtype=np.float32)
        image += (
            12000.0
            * np.exp(-((yy - 200) ** 2 + (xx - 200) ** 2) / (2 * 2**2))
        ).astype(np.float32)
        coarse = pd.DataFrame(
            [
                {
                    "detector_component_id": "det_0000",
                    "coarse_center_x": 200,
                    "coarse_center_y": 200,
                    "coarse_max_ds": 12000.0,
                }
            ]
        )

        result = refine_candidates_from_array(image, coarse, DONOR13_V6)

        self.assertTrue(result.candidates.empty)
        self.assertEqual(result.candidate_masks, {})


if __name__ == "__main__":
    unittest.main()
