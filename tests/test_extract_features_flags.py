import unittest
import numpy as np

from aegle.extract_features import extract_features_v2_optimized


class TestExtractFeaturesFlags(unittest.TestCase):
    def setUp(self):
        self.image = np.ones((4, 4), dtype=np.uint16)
        self.nucleus = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint32)
        self.cell = self.nucleus.copy()
        self.channels = ["c0", "c1"]
        self.image_dict = {c: self.image * (i + 1) for i, c in enumerate(self.channels)}

    def test_skip_laplacian_and_cov_drop_columns(self):
        markers, metadata = extract_features_v2_optimized(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            compute_laplacian=False,
            compute_cov=False,
        )
        self.assertEqual(set(markers.columns), set(self.channels))
        self.assertNotIn("c0_cov", metadata.columns)
        self.assertNotIn("c1_cov", metadata.columns)
        self.assertFalse(any(col.endswith("_laplacian_var") for col in metadata.columns))

    def test_enabling_features_adds_columns(self):
        _, metadata = extract_features_v2_optimized(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            compute_laplacian=True,
            compute_cov=True,
        )
        for channel in self.channels:
            self.assertIn(f"{channel}_cov", metadata.columns)
            self.assertIn(f"{channel}_laplacian_var", metadata.columns)

    def test_channel_dtype_float32_propagates(self):
        markers, metadata = extract_features_v2_optimized(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            channel_dtype=np.float32,
            compute_laplacian=False,
            compute_cov=False,
        )
        for dtype in markers.dtypes:
            self.assertEqual(dtype, np.float32)
        nucleus_means = metadata[[f"{c}_nucleus_mean" for c in self.channels]]
        for dtype in nucleus_means.dtypes:
            self.assertEqual(dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
