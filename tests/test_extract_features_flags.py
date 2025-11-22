import unittest
import numpy as np

from aegle.extract_features import extract_features_v2_optimized
from aegle.gpu_utils import is_cupy_available
from tests.utils.gpu_test_utils import requires_gpu

# Only import GPU version if available
if is_cupy_available():
    from aegle.extract_features_gpu import extract_features_v2_gpu


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


class TestGPUFeatureFlags(unittest.TestCase):
    """Test GPU-specific feature flags."""

    def setUp(self):
        self.image = np.ones((4, 4), dtype=np.uint16)
        self.nucleus = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint32)
        self.cell = self.nucleus.copy()
        self.channels = ["c0", "c1"]
        self.image_dict = {c: self.image * (i + 1) for i, c in enumerate(self.channels)}

    @requires_gpu()
    def test_gpu_respects_compute_laplacian_flag(self):
        """GPU version should respect compute_laplacian flag like CPU."""
        # GPU with Laplacian disabled
        _, metadata_disabled = extract_features_v2_gpu(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            compute_laplacian=False,
        )
        self.assertFalse(any(col.endswith("_laplacian_var") for col in metadata_disabled.columns))

        # GPU with Laplacian enabled
        _, metadata_enabled = extract_features_v2_gpu(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            compute_laplacian=True,
        )
        for channel in self.channels:
            self.assertIn(f"{channel}_laplacian_var", metadata_enabled.columns)

    @requires_gpu()
    def test_gpu_respects_compute_cov_flag(self):
        """GPU version should respect compute_cov flag like CPU."""
        # GPU with CoV disabled
        _, metadata_disabled = extract_features_v2_gpu(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            compute_cov=False,
        )
        self.assertNotIn("c0_cov", metadata_disabled.columns)
        self.assertNotIn("c1_cov", metadata_disabled.columns)

        # GPU with CoV enabled
        _, metadata_enabled = extract_features_v2_gpu(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            compute_cov=True,
        )
        for channel in self.channels:
            self.assertIn(f"{channel}_cov", metadata_enabled.columns)

    @requires_gpu()
    def test_gpu_respects_channel_dtype_flag(self):
        """GPU version should respect channel_dtype flag like CPU."""
        markers, metadata = extract_features_v2_gpu(
            self.image_dict,
            self.nucleus,
            self.channels,
            cell_masks=self.cell,
            channel_dtype=np.float32,
        )

        # Check marker dtypes
        for dtype in markers.dtypes:
            self.assertEqual(dtype, np.float32)

        # Check metadata numeric columns
        nucleus_means = metadata[[f"{c}_nucleus_mean" for c in self.channels]]
        for dtype in nucleus_means.dtypes:
            self.assertEqual(dtype, np.float32)

    @requires_gpu()
    def test_gpu_batch_size_flag_accepted(self):
        """GPU version should accept gpu_batch_size parameter."""
        # Should not raise error with different batch sizes
        for batch_size in [0, 1, 2]:
            markers, metadata = extract_features_v2_gpu(
                self.image_dict,
                self.nucleus,
                self.channels,
                cell_masks=self.cell,
                gpu_batch_size=batch_size,
            )
            # Should return valid results
            self.assertEqual(len(markers), 2)  # 2 cells
            self.assertEqual(set(markers.columns), set(self.channels))

    @requires_gpu()
    def test_gpu_and_cpu_same_flags_produce_same_columns(self):
        """GPU and CPU with same flags should produce same column structure."""
        flag_combinations = [
            {"compute_laplacian": False, "compute_cov": False},
            {"compute_laplacian": True, "compute_cov": False},
            {"compute_laplacian": False, "compute_cov": True},
            {"compute_laplacian": True, "compute_cov": True},
        ]

        for flags in flag_combinations:
            with self.subTest(flags=flags):
                cpu_markers, cpu_metadata = extract_features_v2_optimized(
                    self.image_dict,
                    self.nucleus,
                    self.channels,
                    cell_masks=self.cell,
                    **flags,
                )

                gpu_markers, gpu_metadata = extract_features_v2_gpu(
                    self.image_dict,
                    self.nucleus,
                    self.channels,
                    cell_masks=self.cell,
                    **flags,
                )

                # Column names should match
                self.assertEqual(
                    set(cpu_markers.columns),
                    set(gpu_markers.columns),
                    f"Marker columns differ for flags {flags}"
                )
                self.assertEqual(
                    set(cpu_metadata.columns),
                    set(gpu_metadata.columns),
                    f"Metadata columns differ for flags {flags}"
                )


if __name__ == "__main__":
    unittest.main()
