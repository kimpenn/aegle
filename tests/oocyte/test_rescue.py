import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aegle.oocyte import DONOR13_V6_RESCUE_V1
from aegle.oocyte.models import BoundingBox
from aegle.oocyte.rescue import (
    _baseline_fallback_is_eligible,
    _final_acceptance_reason,
    suppress_accepted_mask_duplicates,
)


def candidate(**overrides):
    row = {
        "local_equivalent_diameter_um": 32.0,
        "local_circularity": 0.85,
        "local_solidity": 0.96,
        "local_eccentricity": 0.30,
        "local_max_intensity": 12000.0,
        "mean_to_background_ratio": 4.0,
        "detector_score": 0.70,
        "local_centroid_offset_px": 2.0,
    }
    row.update(overrides)
    return row


class TestSecondaryRescueAcceptance(unittest.TestCase):
    def setUp(self):
        self.config = DONOR13_V6_RESCUE_V1.secondary_rescue
        assert self.config is not None

    def test_standard_candidate_is_accepted(self):
        self.assertEqual(
            _final_acceptance_reason(candidate(), self.config),
            "accepted_standard_pre_dedup",
        )

    def test_bright_irregular_rule_does_not_lower_global_shape_floor(self):
        self.assertEqual(
            _final_acceptance_reason(
                candidate(
                    local_circularity=0.60,
                    local_solidity=0.84,
                    detector_score=0.44,
                    local_max_intensity=9000.0,
                ),
                self.config,
            ),
            "accepted_bright_irregular_pre_dedup",
        )

    def test_low_intensity_elongated_tissue_is_rejected(self):
        reason = _final_acceptance_reason(
            candidate(
                local_eccentricity=0.87,
                local_max_intensity=2500.0,
                mean_to_background_ratio=2.0,
            ),
            self.config,
        )
        self.assertIn("low_intensity_elongation", reason)
        self.assertTrue(reason.startswith("failed_"))

    def test_very_bright_centered_fragment_is_review_gated(self):
        self.assertEqual(
            _final_acceptance_reason(
                candidate(
                    local_equivalent_diameter_um=21.0,
                    local_circularity=0.46,
                    local_solidity=0.75,
                    local_max_intensity=10000.0,
                    mean_to_background_ratio=7.0,
                    detector_score=0.40,
                    local_centroid_offset_px=1.0,
                ),
                self.config,
            ),
            "accepted_bright_fragment_pre_dedup",
        )

    def test_original_p99_mask_can_win_when_p95_merges_neighbors(self):
        self.assertTrue(
            _baseline_fallback_is_eligible(
                candidate(
                    local_circularity=0.64,
                    local_solidity=0.89,
                    local_eccentricity=0.80,
                    detector_score=0.38,
                    local_centroid_offset_px=29.0,
                ),
                self.config,
            )
        )

    def test_actual_mask_overlap_suppresses_only_the_lower_score(self):
        table = pd.DataFrame(
            [
                {
                    "detector_component_id": "high",
                    "accepted": True,
                    "acceptance_mode": "strict",
                    "detector_score": 0.9,
                    "center_x": 10,
                    "center_y": 10,
                },
                {
                    "detector_component_id": "duplicate",
                    "accepted": True,
                    "acceptance_mode": "strict",
                    "detector_score": 0.8,
                    "center_x": 12,
                    "center_y": 12,
                },
                {
                    "detector_component_id": "neighbor",
                    "accepted": True,
                    "acceptance_mode": "strict",
                    "detector_score": 0.7,
                    "center_x": 18,
                    "center_y": 18,
                },
            ]
        )
        full = np.ones((10, 10), dtype=np.bool_)
        empty_overlap = np.ones((10, 10), dtype=np.bool_)
        bbox = BoundingBox(0, 0, 10, 10)
        masks = {
            "high": SimpleNamespace(mask=full, bbox=bbox),
            "duplicate": SimpleNamespace(mask=full.copy(), bbox=bbox),
            "neighbor": SimpleNamespace(
                mask=empty_overlap,
                bbox=BoundingBox(20, 20, 30, 30),
            ),
        }

        result, diagnostics = suppress_accepted_mask_duplicates(
            table,
            masks,
            overlap_fraction=0.25,
            max_centroid_distance_px=20.0,
        )

        accepted = result.set_index("detector_component_id")["accepted"]
        self.assertTrue(bool(accepted["high"]))
        self.assertFalse(bool(accepted["duplicate"]))
        self.assertTrue(bool(accepted["neighbor"]))
        self.assertEqual(diagnostics.iloc[0]["duplicate_of"], "high")

    def test_small_low_confidence_object_is_rejected(self):
        reason = _final_acceptance_reason(
            candidate(
                local_equivalent_diameter_um=19.0,
                local_max_intensity=1800.0,
                detector_score=0.46,
            ),
            self.config,
        )
        self.assertIn("small_low_confidence", reason)


if __name__ == "__main__":
    unittest.main()
