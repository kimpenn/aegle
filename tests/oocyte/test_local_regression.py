import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aegle.oocyte import (
    DONOR13_V6,
    read_ome_channel_patch,
    scan_coarse_candidates,
    scan_refined_candidates,
    segment_oocyte_patch,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RAW_IMAGE = (
    WORKSPACE_ROOT
    / "data/Ovary/D11_13/Scan1/processed_hubmap"
    / "D11_13_Scan1.er_manual_13-23_Ovary_Central_Superior_Antimesenteric_Scan1.ome.tiff"
)
EXPECTED_MASK = (
    WORKSPACE_ROOT
    / "frs-atlas-phenocycler/output/ovary_panel1/v2_4/oocyte_segmentation_poc"
    / "13-23_oocyte_680.mask.npz"
)
EXPECTED_COARSE = (
    WORKSPACE_ROOT
    / "frs-atlas-phenocycler/output/ovary_panel1/v2_4"
    / "oocyte_raw_detector_peak_seed_v6_experimental/13-23"
    / "13-23_coarse_candidates.csv"
)
EXPECTED_REFINED = EXPECTED_COARSE.with_name("13-23_refined_candidates.csv")
RUN_LOCAL = os.environ.get("AEGLE_RUN_OOCYTE_LOCAL_REGRESSION") == "1"


@unittest.skipUnless(
    RUN_LOCAL and RAW_IMAGE.exists() and EXPECTED_MASK.exists(),
    "set AEGLE_RUN_OOCYTE_LOCAL_REGRESSION=1 with local donor13 data",
)
class TestDonor13LocalRegression(unittest.TestCase):
    def test_oocyte_680_matches_research_mask(self):
        patch = read_ome_channel_patch(
            RAW_IMAGE,
            channel_index=27,
            center_xy=(11701, 20077),
            radius=DONOR13_V6.local.window_radius_px,
        )
        result = segment_oocyte_patch(patch.image, DONOR13_V6)
        with np.load(EXPECTED_MASK, allow_pickle=False) as archive:
            expected_mask = np.asarray(archive["oocyte_mask"], dtype=np.bool_)

        np.testing.assert_array_equal(result.mask, expected_mask)
        self.assertAlmostEqual(result.metrics.equivalent_diameter_um, 36.91886960131623)
        self.assertAlmostEqual(result.metrics.circularity, 0.8751328034565493)
        self.assertAlmostEqual(result.metrics.solidity, 0.9758432087511395)
        self.assertLessEqual(result.metrics.centroid_offset_px, 12.0)

    def test_13_23_coarse_proposals_match_research_output(self):
        result = scan_coarse_candidates(RAW_IMAGE, 27, DONOR13_V6)
        expected = pd.read_csv(EXPECTED_COARSE)

        pd.testing.assert_frame_equal(
            result.candidates,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(result.image_shape_yx, (26849, 17783))

    def test_13_23_refinement_matches_research_output(self):
        coarse = pd.read_csv(EXPECTED_COARSE)
        result = scan_refined_candidates(RAW_IMAGE, 27, coarse, DONOR13_V6)
        expected = pd.read_csv(EXPECTED_REFINED)
        expected = expected[result.candidates.columns]

        pd.testing.assert_frame_equal(
            result.candidates,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(len(result.candidates), 318)
        self.assertEqual(int(result.candidates["accepted"].sum()), 135)
        self.assertEqual(len(result.candidate_masks), len(result.candidates))


if __name__ == "__main__":
    unittest.main()
