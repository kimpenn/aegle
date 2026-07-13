import unittest

import numpy as np

from aegle.oocyte import DONOR13_V6, segment_oocyte_patch


def circular_patch(
    *,
    shape=(361, 361),
    circles=((180, 180, 40, 12000.0),),
    background=200.0,
):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    image = np.full(shape, background, dtype=np.float32)
    for center_y, center_x, radius, intensity in circles:
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        image[mask] = intensity
    return image


class TestLocalSegmentation(unittest.TestCase):
    def test_segments_centered_bright_circle(self):
        result = segment_oocyte_patch(circular_patch(), DONOR13_V6)

        self.assertEqual(result.metrics.selection_mode, "center_component")
        self.assertGreater(result.metrics.circularity, 0.85)
        self.assertGreater(result.metrics.solidity, 0.98)
        self.assertGreater(result.metrics.equivalent_diameter_um, 38.0)
        self.assertLess(result.metrics.equivalent_diameter_um, 44.0)
        self.assertLess(result.metrics.centroid_offset_px, 0.1)

    def test_center_component_wins_over_larger_neighbor(self):
        patch = circular_patch(
            circles=(
                (180, 180, 35, 12000.0),
                (180, 285, 45, 12000.0),
            )
        )
        result = segment_oocyte_patch(patch, DONOR13_V6)

        self.assertEqual(result.metrics.selection_mode, "center_component")
        self.assertLess(result.metrics.equivalent_diameter_um, 42.0)
        self.assertLess(result.metrics.centroid_offset_px, 1.0)

    def test_uses_distance_weighted_fallback_when_center_is_background(self):
        patch = circular_patch(circles=((180, 220, 30, 12000.0),))
        result = segment_oocyte_patch(patch, DONOR13_V6)

        self.assertEqual(result.metrics.selection_mode, "distance_weighted_fallback")
        self.assertAlmostEqual(result.metrics.centroid_x_px, 220.0, delta=1.0)

    def test_rejects_non_finite_patch(self):
        patch = circular_patch()
        patch[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            segment_oocyte_patch(patch, DONOR13_V6)


if __name__ == "__main__":
    unittest.main()
