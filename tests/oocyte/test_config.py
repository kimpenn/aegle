import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from aegle.oocyte.config import (
    DONOR13_V6,
    DONOR13_V6_RESCUE_V1,
    ExperimentalConfig,
    OocyteDetectionConfig,
    available_profiles,
    get_profile,
)


BASELINE_PATH = Path(__file__).parent / "data" / "donor13_v6_baseline.json"


class TestDonor13V6Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(BASELINE_PATH.read_text())

    def test_profile_is_registered_and_immutable(self):
        self.assertEqual(
            available_profiles(),
            ("donor13_v6", "donor13_v6_rescue_v1"),
        )
        self.assertIs(get_profile("donor13_v6"), DONOR13_V6)
        self.assertIs(
            get_profile("donor13_v6_rescue_v1"),
            DONOR13_V6_RESCUE_V1,
        )
        with self.assertRaises(FrozenInstanceError):
            DONOR13_V6.pixel_size_um = 1.0

    def test_profile_fingerprint_matches_frozen_baseline(self):
        self.assertEqual(
            DONOR13_V6.fingerprint(),
            self.baseline["profile_fingerprint"],
        )

    def test_v6_enables_validated_seed_families_without_border_rescue(self):
        contract = self.baseline["algorithm_contract"]
        self.assertEqual(DONOR13_V6.local.max_broad_peak_seeds_per_candidate, 3)
        self.assertEqual(DONOR13_V6.local.max_offset_ring_seeds, 8)
        self.assertEqual(DONOR13_V6.local.max_centroid_reseed_iterations, 1)
        self.assertLess(
            DONOR13_V6.local.compact_window_radius_px,
            DONOR13_V6.local.window_radius_px,
        )
        self.assertFalse(DONOR13_V6.experimental.border_rescue_enabled)
        self.assertFalse(contract["border_rescue_enabled"])

    def test_baseline_totals_equal_sum_of_samples(self):
        samples = self.baseline["samples"].values()
        totals = self.baseline["expected_totals"]
        count_fields = (
            "reference_count",
            "coarse_candidate_count",
            "refined_candidate_count",
            "accepted_candidate_count",
            "reference_recalled_by_refined",
            "reference_recalled_by_accepted",
            "accepted_candidates_near_reference",
            "accepted_candidates_novel",
        )
        self.assertEqual(len(self.baseline["samples"]), totals["sample_count"])
        for field in count_fields:
            with self.subTest(field=field):
                self.assertEqual(sum(sample[field] for sample in samples), totals[field])

    def test_representative_case_matches_acceptance_tolerances(self):
        metrics = self.baseline["representative_case"]["metrics"]
        self.assertGreaterEqual(metrics["equivalent_diameter_um"], 35.0)
        self.assertLessEqual(metrics["equivalent_diameter_um"], 39.0)
        self.assertGreaterEqual(metrics["circularity"], 0.82)
        self.assertGreaterEqual(metrics["solidity"], 0.94)
        self.assertLessEqual(metrics["centroid_offset_px"], 12.0)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown oocyte detector profile"):
            get_profile("donor13_v7")

    def test_v6_rejects_experimental_border_rescue(self):
        with self.assertRaisesRegex(ValueError, "does not support experimental border rescue"):
            OocyteDetectionConfig(
                experimental=ExperimentalConfig(border_rescue_enabled=True)
            )


if __name__ == "__main__":
    unittest.main()
