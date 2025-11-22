import unittest
import numpy as np

from aegle.extract_features import extract_features_v2_optimized
from tests.utils import make_single_cell_patch


class TestExtractFeaturesDuplicateChannels(unittest.TestCase):
    def test_duplicate_channel_names_are_made_unique(self):
        """Extractor should suffix duplicate channel names and preserve values."""
        img = np.ones((4, 4), dtype=np.float32)
        image_dict = {"dup": img}
        nucleus_mask = np.ones((4, 4), dtype=np.uint32)
        cell_mask = np.ones((4, 4), dtype=np.uint32)

        markers, metadata = extract_features_v2_optimized(
            image_dict,
            nucleus_mask,
            ["dup", "dup"],
            cell_masks=cell_mask,
        )

        # The extractor should rename to dup and dup_1 in order.
        self.assertListEqual(list(markers.columns), ["dup", "dup_1"])
        cell_id = markers.index[0]
        self.assertAlmostEqual(markers.loc[cell_id, "dup"], 1.0, places=6)
        self.assertAlmostEqual(markers.loc[cell_id, "dup_1"], 1.0, places=6)
        self.assertIn("dup_nucleus_mean", metadata.columns)
        self.assertIn("dup_1_nucleus_mean", metadata.columns)

    def test_zero_channels_returns_empty(self):
        """Calling with no channels should yield empty outputs."""
        markers, metadata = extract_features_v2_optimized(
            {}, np.zeros((8, 8), dtype=np.uint32), [], cell_masks=np.zeros((8, 8), dtype=np.uint32)
        )
        self.assertTrue(markers.empty)
        self.assertTrue(metadata.empty)

    def test_zero_masks_do_not_raise(self):
        """Zero-valued masks with channels should return empty frames."""
        img = np.ones((4, 4), dtype=np.float32)
        markers, metadata = extract_features_v2_optimized(
            {"c0": img},
            np.zeros((4, 4), dtype=np.uint32),
            ["c0"],
            cell_masks=np.zeros((4, 4), dtype=np.uint32),
        )
        self.assertTrue(markers.empty)
        self.assertTrue(metadata.empty)

    def test_mixed_dtype_inputs(self):
        """Mixed dtypes should still compute means without error."""
        image_dict = {"c0": np.ones((2, 2), dtype=np.float64)}
        nucleus = np.array([[1, 0], [0, 0]], dtype=np.uint32)
        cell_mask = nucleus.astype(np.int32)
        markers, metadata = extract_features_v2_optimized(
            image_dict, nucleus, ["c0"], cell_masks=cell_mask
        )
        cell_id = markers.index[0]
        self.assertAlmostEqual(markers.loc[cell_id, "c0"], 1.0, places=6)

    def test_no_nan_warnings_for_zero_masks(self):
        """Zero-count labels should not trigger NaN warnings in CoV."""
        img = np.zeros((2, 2), dtype=np.float32)
        markers, metadata = extract_features_v2_optimized(
            {"c0": img},
            np.array([[1, 1], [0, 0]], dtype=np.uint32),
            ["c0"],
            cell_masks=np.array([[1, 1], [0, 0]], dtype=np.uint32),
        )
        # Markers should be finite; metadata may have NaN cytoplasm means but should not raise.
        self.assertFalse(markers.isna().any().any())


if __name__ == "__main__":
    unittest.main()
