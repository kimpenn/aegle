"""
Tests for GPU-accelerated feature extraction.

Validates that GPU version produces numerically equivalent results to CPU version
and handles edge cases correctly.
"""

import math
import unittest
import numpy as np
from unittest.mock import patch

from aegle.extract_features import extract_features_v2_optimized
from aegle.gpu_utils import is_cupy_available
from tests.utils import (
    make_empty_patch,
    make_nucleus_only_patch,
    make_single_cell_patch,
)
from tests.utils.gpu_test_utils import requires_gpu, assert_gpu_cpu_equal

# Only import GPU version if CuPy available
if is_cupy_available():
    from aegle.extract_features_gpu import extract_features_v2_gpu


class TestGPUFeatureExtractionNumericalCorrectness(unittest.TestCase):
    """Test GPU feature extraction produces same results as CPU."""

    @requires_gpu()
    def test_gpu_vs_cpu_basic_equivalence(self):
        """GPU and CPU should produce identical results on basic synthetic data."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0", "chan1", "chan2"),
            nucleus_intensity={"chan0": 100.0, "chan1": 50.0, "chan2": 25.0},
            cytoplasm_intensity={"chan0": 10.0, "chan1": 20.0, "chan2": 5.0},
        )

        # Run CPU version
        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        # Run GPU version
        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        # Compare results
        assert_gpu_cpu_equal(gpu_markers, cpu_markers)
        assert_gpu_cpu_equal(gpu_metadata, cpu_metadata)

    @requires_gpu()
    def test_gpu_vs_cpu_with_multiple_cells(self):
        """GPU should handle multiple cells correctly."""
        # Create a 2x2 grid of cells
        from tests.utils.synthetic_data_factory import make_disk_cells_patch

        cells = [
            {
                "center": (25, 25),
                "nucleus_radius": 5,
                "cell_radius": 10,
                "nucleus_intensity": {"CD3": 100, "CD8": 50, "CD45": 200},
                "cytoplasm_intensity": {"CD3": 20, "CD8": 10, "CD45": 40},
            },
            {
                "center": (25, 75),
                "nucleus_radius": 5,
                "cell_radius": 10,
                "nucleus_intensity": {"CD3": 80, "CD8": 120, "CD45": 150},
                "cytoplasm_intensity": {"CD3": 15, "CD8": 25, "CD45": 30},
            },
            {
                "center": (75, 25),
                "nucleus_radius": 5,
                "cell_radius": 10,
                "nucleus_intensity": {"CD3": 90, "CD8": 70, "CD45": 180},
                "cytoplasm_intensity": {"CD3": 18, "CD8": 12, "CD45": 35},
            },
            {
                "center": (75, 75),
                "nucleus_radius": 5,
                "cell_radius": 10,
                "nucleus_intensity": {"CD3": 110, "CD8": 90, "CD45": 160},
                "cytoplasm_intensity": {"CD3": 22, "CD8": 18, "CD45": 38},
            },
        ]

        test_patch = make_disk_cells_patch(
            shape=(100, 100),
            channels=("CD3", "CD8", "CD45"),
            cells=cells,
        )

        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            test_patch.image_dict,
            test_patch.nucleus_mask,
            test_patch.channels,
            cell_masks=test_patch.cell_mask,
        )

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            test_patch.image_dict,
            test_patch.nucleus_mask,
            test_patch.channels,
            cell_masks=test_patch.cell_mask,
        )

        # Should have 4 cells
        self.assertEqual(len(cpu_markers), 4)
        self.assertEqual(len(gpu_markers), 4)

        assert_gpu_cpu_equal(gpu_markers, cpu_markers)
        assert_gpu_cpu_equal(gpu_metadata, cpu_metadata)

    @requires_gpu()
    def test_gpu_batch_sizes(self):
        """GPU should produce same results with different batch sizes."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("c1", "c2", "c3", "c4", "c5"),
            nucleus_intensity={f"c{i}": float(i * 10) for i in range(1, 6)},
            cytoplasm_intensity={f"c{i}": float(i * 2) for i in range(1, 6)},
        )

        # Run with different batch sizes
        batch_sizes = [0, 1, 2, 5]  # 0=auto, others=explicit
        results = []

        for batch_size in batch_sizes:
            markers, metadata = extract_features_v2_gpu(
                patch.image_dict,
                patch.nucleus_mask,
                patch.channels,
                cell_masks=patch.cell_mask,
                gpu_batch_size=batch_size,
            )
            results.append((markers, metadata))

        # All results should be identical
        base_markers, base_metadata = results[0]
        for i, (markers, metadata) in enumerate(results[1:], 1):
            with self.subTest(batch_size=batch_sizes[i]):
                assert_gpu_cpu_equal(markers, base_markers,
                                   err_msg=f"Batch size {batch_sizes[i]} differs")
                assert_gpu_cpu_equal(metadata, base_metadata,
                                   err_msg=f"Batch size {batch_sizes[i]} differs")

    @requires_gpu()
    def test_gpu_with_compute_cov_enabled(self):
        """GPU should handle CoV computation correctly."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0", "chan1"),
            nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
            cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
        )

        # CPU with CoV
        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_cov=True,
        )

        # GPU with CoV
        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_cov=True,
        )

        # Check CoV columns exist
        for channel in patch.channels:
            self.assertIn(f"{channel}_cov", cpu_metadata.columns)
            self.assertIn(f"{channel}_cov", gpu_metadata.columns)

        assert_gpu_cpu_equal(gpu_markers, cpu_markers)
        assert_gpu_cpu_equal(gpu_metadata, cpu_metadata)

    @requires_gpu()
    def test_gpu_with_compute_cov_disabled(self):
        """GPU should skip CoV computation when disabled."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0", "chan1"),
            nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
            cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
        )

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_cov=False,
        )

        # CoV columns should not exist
        for channel in patch.channels:
            self.assertNotIn(f"{channel}_cov", gpu_metadata.columns)

    @requires_gpu()
    def test_gpu_with_laplacian_enabled(self):
        """GPU should handle Laplacian computation (via CPU fallback)."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0",),
            nucleus_intensity={"chan0": 50.0},
            cytoplasm_intensity={"chan0": 5.0},
        )

        # CPU with Laplacian
        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_laplacian=True,
        )

        # GPU with Laplacian (should use CPU implementation)
        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_laplacian=True,
        )

        # Check Laplacian columns exist
        self.assertIn("chan0_laplacian_var", cpu_metadata.columns)
        self.assertIn("chan0_laplacian_var", gpu_metadata.columns)

        assert_gpu_cpu_equal(gpu_markers, cpu_markers)
        assert_gpu_cpu_equal(gpu_metadata, cpu_metadata)

    @requires_gpu()
    def test_gpu_with_float32_dtype(self):
        """GPU should handle float32 channel dtype correctly."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0", "chan1"),
            nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
            cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
        )

        # Run with float32
        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            channel_dtype=np.float32,
        )

        # Results should be valid (specific values depend on implementation)
        self.assertEqual(len(gpu_markers), 1)
        self.assertEqual(list(gpu_markers.columns), list(patch.channels))


class TestGPUFeatureExtractionEdgeCases(unittest.TestCase):
    """Test GPU feature extraction handles edge cases."""

    @requires_gpu()
    def test_gpu_empty_masks(self):
        """GPU should handle empty masks gracefully."""
        patch = make_empty_patch(shape=(64, 64), channels=("chan0", "chan1"))

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        # Should return empty DataFrames
        self.assertTrue(gpu_markers.empty)
        self.assertTrue(gpu_metadata.empty)

        # Columns should still be present
        self.assertEqual(list(gpu_markers.columns), list(patch.channels))
        self.assertIn("label", gpu_metadata.columns)

    @requires_gpu()
    def test_gpu_nucleus_only_patch(self):
        """GPU should handle nucleus-only cells correctly."""
        patch = make_nucleus_only_patch(
            shape=(64, 64),
            channels=("chan0",),
            nucleus_intensity={"chan0": 50.0},
        )

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        cell_id = gpu_markers.index[0]
        channel = patch.channels[0]

        # Whole-cell mean should equal nucleus mean
        self.assertAlmostEqual(
            gpu_markers.loc[cell_id, channel],
            gpu_metadata.loc[cell_id, f"{channel}_nucleus_mean"],
            places=6,
        )

        # Cytoplasm mean should be NaN
        self.assertTrue(math.isnan(gpu_metadata.loc[cell_id, f"{channel}_cytoplasm_mean"]))

    @requires_gpu()
    def test_gpu_single_channel(self):
        """GPU should handle single-channel images."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("DAPI",),
            nucleus_intensity={"DAPI": 100.0},
            cytoplasm_intensity={"DAPI": 10.0},
        )

        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        assert_gpu_cpu_equal(gpu_markers, cpu_markers)
        assert_gpu_cpu_equal(gpu_metadata, cpu_metadata)


class TestGPUFeatureExtractionFallback(unittest.TestCase):
    """Test GPU feature extraction fallback behavior."""

    def test_gpu_fallback_when_unavailable(self):
        """Should fall back to CPU when GPU unavailable."""
        from unittest.mock import patch as mock_patch

        test_patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0",),
            nucleus_intensity={"chan0": 50.0},
            cytoplasm_intensity={"chan0": 5.0},
        )

        # Mock GPU as unavailable at the source
        with mock_patch('aegle.gpu_utils.is_cupy_available', return_value=False):
            from aegle.extract_features_gpu import extract_features_v2_gpu

            markers, metadata = extract_features_v2_gpu(
                test_patch.image_dict,
                test_patch.nucleus_mask,
                test_patch.channels,
                cell_masks=test_patch.cell_mask,
            )

            # Should still return valid results via CPU fallback
            self.assertEqual(len(markers), 1)
            self.assertEqual(list(markers.columns), list(test_patch.channels))


class TestGPUFeatureExtractionOutputSchema(unittest.TestCase):
    """Test GPU feature extraction output format matches CPU."""

    @requires_gpu()
    def test_gpu_output_column_names(self):
        """GPU output should have same column names as CPU."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("CD3", "CD8", "CD45"),
            nucleus_intensity={"CD3": 50.0, "CD8": 30.0, "CD45": 100.0},
            cytoplasm_intensity={"CD3": 5.0, "CD8": 3.0, "CD45": 10.0},
        )

        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_cov=True,
            compute_laplacian=True,
        )

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
            compute_cov=True,
            compute_laplacian=True,
        )

        # Check column names match exactly
        self.assertEqual(list(gpu_markers.columns), list(cpu_markers.columns))
        self.assertEqual(list(gpu_metadata.columns), list(cpu_metadata.columns))

    @requires_gpu()
    def test_gpu_output_dtypes(self):
        """GPU output should have compatible dtypes with CPU."""
        patch = make_single_cell_patch(
            shape=(64, 64),
            channels=("chan0",),
            nucleus_intensity={"chan0": 50.0},
            cytoplasm_intensity={"chan0": 5.0},
        )

        cpu_markers, cpu_metadata = extract_features_v2_optimized(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        gpu_markers, gpu_metadata = extract_features_v2_gpu(
            patch.image_dict,
            patch.nucleus_mask,
            patch.channels,
            cell_masks=patch.cell_mask,
        )

        # Markers should be numeric
        for col in gpu_markers.columns:
            self.assertTrue(
                np.issubdtype(gpu_markers[col].dtype, np.number),
                f"Column {col} should be numeric"
            )

        # Metadata numeric columns should match
        for col in cpu_metadata.columns:
            if np.issubdtype(cpu_metadata[col].dtype, np.number):
                self.assertTrue(
                    np.issubdtype(gpu_metadata[col].dtype, np.number),
                    f"Metadata column {col} should be numeric"
                )


if __name__ == "__main__":
    unittest.main()
