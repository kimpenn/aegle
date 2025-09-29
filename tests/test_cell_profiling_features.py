import math
import unittest

import numpy as np

from aegle.extract_features import extract_features_v2_optimized
from tests.utils import (
    make_empty_patch,
    make_nucleus_only_patch,
    make_single_cell_patch,
)


class TestExtractFeaturesWithSyntheticData(unittest.TestCase):
    def test_single_cell_means(self):
        """Whole-cell, nucleus, cytoplasm means should match synthetic truth."""
        patch = make_single_cell_patch(
            shape=(48, 48),
            channels=("chan0", "chan1"),
            nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
            cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
        )

        markers, metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        self.assertEqual(list(markers.columns), list(patch.channels))
        self.assertEqual(markers.shape[0], 1)
        cell_id = markers.index[0]

        # Whole-cell mean should match expected metrics stored in the synthetic patch.
        expected = patch.expected_means[cell_id]
        for channel in patch.channels:
            with self.subTest(channel=channel):
                self.assertAlmostEqual(
                    markers.loc[cell_id, channel],
                    expected[channel]["wholecell_mean"],
                    places=6,
                )
                self.assertAlmostEqual(
                    metadata.loc[cell_id, f"{channel}_nucleus_mean"],
                    expected[channel]["nucleus_mean"],
                    places=6,
                )
                self.assertAlmostEqual(
                    metadata.loc[cell_id, f"{channel}_cytoplasm_mean"],
                    expected[channel]["cytoplasm_mean"],
                    places=6,
                )

    def test_nucleus_only_patch_has_nan_cytoplasm(self):
        """Cytoplasm mean should be NaN when whole-cell equals nucleus."""
        patch = make_nucleus_only_patch(
            shape=(40, 40),
            channels=("chan0",),
            nucleus_intensity={"chan0": 17.0},
        )

        markers, metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        cell_id = markers.index[0]
        channel = patch.channels[0]

        # Whole-cell mean equals nucleus mean since there is no cytoplasm.
        self.assertAlmostEqual(
            markers.loc[cell_id, channel],
            metadata.loc[cell_id, f"{channel}_nucleus_mean"],
            places=6,
        )
        self.assertTrue(math.isnan(metadata.loc[cell_id, f"{channel}_cytoplasm_mean"]))

    def test_empty_patch_returns_no_cells(self):
        """Extractor should return empty outputs when no labels exist."""
        patch = make_empty_patch(shape=(32, 32), channels=("chan0", "chan1"))
        markers, metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )
        self.assertTrue(markers.empty)
        self.assertTrue(metadata.empty)


if __name__ == "__main__":
    unittest.main()
