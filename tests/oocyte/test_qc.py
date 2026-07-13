import unittest

import pandas as pd

from aegle.oocyte import accepted_duplicate_suspects


class TestOocyteSpatialQc(unittest.TestCase):
    def test_flags_only_diameter_scaled_overlaps(self):
        candidates = pd.DataFrame(
            [
                {
                    "detector_component_id": "det_0000",
                    "accepted": True,
                    "center_x": 100,
                    "center_y": 100,
                    "local_equivalent_diameter_um": 40.0,
                },
                {
                    "detector_component_id": "det_0001",
                    "accepted": True,
                    "center_x": 150,
                    "center_y": 100,
                    "local_equivalent_diameter_um": 40.0,
                },
                {
                    "detector_component_id": "det_0002",
                    "accepted": True,
                    "center_x": 400,
                    "center_y": 400,
                    "local_equivalent_diameter_um": 40.0,
                },
            ]
        )

        result = accepted_duplicate_suspects(candidates, pixel_size_um=0.5)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["detector_component_id_a"], "det_0000")
        self.assertEqual(result.iloc[0]["detector_component_id_b"], "det_0001")
        self.assertAlmostEqual(result.iloc[0]["overlap_fraction_smaller"], 0.75)


if __name__ == "__main__":
    unittest.main()
