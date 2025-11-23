"""Tests for GPU-accelerated overlap matrix computation.

This module tests the GPU implementation of cell-nucleus overlap computation
and verifies equivalence with the CPU version.
"""

import pytest
import numpy as np
import time

from aegle.repair_masks_gpu_overlap import (
    compute_overlap_matrix_gpu,
    _compute_overlap_matrix_cpu,
)
from tests.utils.gpu_test_utils import requires_gpu, assert_gpu_cpu_equal
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_overlapping_nuclei_case,
    create_unmatched_cells_case,
    create_noncontiguous_labels_case,
    create_edge_case_empty_masks,
    create_edge_case_single_cell,
    create_stress_test_case,
)


class TestOverlapMatrixCPU:
    """Test CPU version of overlap matrix computation (no GPU required)."""

    def test_simple_case_cpu(self):
        """Test CPU overlap matrix on simple case."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        overlap, cell_ids, nucleus_ids = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        # Check dimensions
        assert overlap.shape == (len(cell_ids), len(nucleus_ids))
        assert len(cell_ids) == expected["n_matched_cells"]
        assert len(nucleus_ids) == expected["n_matched_nuclei"]

        # Simple case: each cell should overlap with exactly one nucleus (itself)
        # Sum along axis 1 (nuclei) should be 1 for each cell
        overlaps_per_cell = overlap.sum(axis=1)
        assert np.all(overlaps_per_cell == 1), "Each cell should overlap with exactly 1 nucleus"

    def test_overlapping_nuclei_cpu(self):
        """Test CPU version with multiple nuclei per cell."""
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )

        overlap, cell_ids, nucleus_ids = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        # Check dimensions
        assert overlap.shape == (len(cell_ids), len(nucleus_ids))

        # Should have more nuclei than cells (test fixture creates extras)
        assert len(nucleus_ids) >= len(cell_ids), \
            f"Expected more nuclei ({len(nucleus_ids)}) >= cells ({len(cell_ids)})"

        # Each cell should overlap with at least one nucleus
        overlaps_per_cell = overlap.sum(axis=1)
        assert np.all(overlaps_per_cell >= 1), "Each cell should overlap with at least 1 nucleus"

    def test_empty_masks_cpu(self):
        """Test CPU version with empty masks."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()

        overlap, cell_ids, nucleus_ids = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        # Should return empty arrays without errors
        assert overlap.shape == (0, 0)
        assert len(cell_ids) == 0
        assert len(nucleus_ids) == 0

    def test_noncontiguous_labels_cpu(self):
        """Test CPU version with non-contiguous label IDs."""
        cell_mask, nucleus_mask, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],
            cell_labels=[1, 2, 3, 4],
        )

        overlap, cell_ids, nucleus_ids = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        # Check that labels are preserved correctly
        assert list(cell_ids) == [1, 2, 3, 4]
        assert list(nucleus_ids) == [1, 5, 7, 23]

        # Each cell should match to its corresponding nucleus
        # Cell 1 → Nucleus 1 (index 0), Cell 2 → Nucleus 5 (index 1), etc.
        assert overlap[0, 0], "Cell 1 should overlap with Nucleus 1"
        assert overlap[1, 1], "Cell 2 should overlap with Nucleus 5"
        assert overlap[2, 2], "Cell 3 should overlap with Nucleus 7"
        assert overlap[3, 3], "Cell 4 should overlap with Nucleus 23"


@requires_gpu()
class TestOverlapMatrixGPU:
    """Test GPU version of overlap matrix computation (requires GPU)."""

    def test_gpu_vs_cpu_simple(self):
        """Verify GPU matches CPU on simple test case."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        # CPU version
        overlap_cpu, cell_ids_cpu, nucleus_ids_cpu = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        # GPU version
        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        # Results should be identical
        assert np.array_equal(cell_ids_cpu, cell_ids_gpu), "Cell IDs should match"
        assert np.array_equal(nucleus_ids_cpu, nucleus_ids_gpu), "Nucleus IDs should match"
        assert np.array_equal(overlap_cpu, overlap_gpu), "Overlap matrices should match"

    def test_gpu_vs_cpu_overlapping_nuclei(self):
        """Verify GPU matches CPU with multiple nuclei per cell."""
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )

        overlap_cpu, cell_ids_cpu, nucleus_ids_cpu = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        assert np.array_equal(cell_ids_cpu, cell_ids_gpu)
        assert np.array_equal(nucleus_ids_cpu, nucleus_ids_gpu)
        assert np.array_equal(overlap_cpu, overlap_gpu)

    def test_gpu_vs_cpu_unmatched_cells(self):
        """Verify GPU matches CPU with cells without nuclei."""
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=5, n_cells_without_nuclei=3
        )

        overlap_cpu, cell_ids_cpu, nucleus_ids_cpu = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        assert np.array_equal(cell_ids_cpu, cell_ids_gpu)
        assert np.array_equal(nucleus_ids_cpu, nucleus_ids_gpu)
        assert np.array_equal(overlap_cpu, overlap_gpu)

    def test_gpu_vs_cpu_noncontiguous_labels(self):
        """Verify GPU handles non-contiguous label IDs correctly."""
        cell_mask, nucleus_mask, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],
            cell_labels=[1, 2, 3, 4],
        )

        overlap_cpu, cell_ids_cpu, nucleus_ids_cpu = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        # Check label preservation
        assert np.array_equal(cell_ids_cpu, cell_ids_gpu)
        assert np.array_equal(nucleus_ids_cpu, nucleus_ids_gpu)

        # Check overlap matrix
        assert np.array_equal(overlap_cpu, overlap_gpu)

        # Verify specific overlaps
        assert overlap_gpu[0, 0], "Cell 1 should overlap with Nucleus 1"
        assert overlap_gpu[1, 1], "Cell 2 should overlap with Nucleus 5"
        assert overlap_gpu[2, 2], "Cell 3 should overlap with Nucleus 7"
        assert overlap_gpu[3, 3], "Cell 4 should overlap with Nucleus 23"

    def test_gpu_empty_masks(self):
        """Test GPU with empty masks."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()

        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        # Should return empty arrays without errors
        assert overlap_gpu.shape == (0, 0)
        assert len(cell_ids_gpu) == 0
        assert len(nucleus_ids_gpu) == 0

    def test_gpu_single_cell(self):
        """Test GPU with single cell."""
        cell_mask, nucleus_mask, expected = create_edge_case_single_cell()

        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        assert overlap_gpu.shape == (1, 1)
        assert overlap_gpu[0, 0], "Single cell should overlap with its nucleus"

    def test_gpu_batch_processing(self):
        """Test GPU batch processing with small batch size."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=20)

        # Force small batch size to test batching logic
        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask, batch_size=5
        )

        # Compare with CPU version
        overlap_cpu, cell_ids_cpu, nucleus_ids_cpu = _compute_overlap_matrix_cpu(
            cell_mask, nucleus_mask
        )

        assert np.array_equal(overlap_cpu, overlap_gpu)

    def test_gpu_large_sample(self):
        """Test GPU on large sample (1000 cells)."""
        cell_mask, nucleus_mask, expected = create_stress_test_case(
            n_cells=1000, image_size=(1000, 1000)
        )

        overlap_gpu, cell_ids_gpu, nucleus_ids_gpu = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        # Should complete without errors
        assert overlap_gpu is not None
        assert overlap_gpu.shape[0] == len(cell_ids_gpu)
        assert overlap_gpu.shape[1] == len(nucleus_ids_gpu)

        # At least 80% of cells should have overlaps (from fixture expectation)
        cells_with_overlap = (overlap_gpu.sum(axis=1) > 0).sum()
        min_matched = expected.get("min_matched", 0)
        assert cells_with_overlap >= min_matched, \
            f"Expected at least {min_matched} cells with overlaps, got {cells_with_overlap}"


@requires_gpu()
class TestOverlapMatrixGPUPerformance:
    """Performance benchmarks for GPU overlap computation."""

    @pytest.mark.parametrize("n_cells", [100, 500, 1000, 5000])
    def test_gpu_scaling(self, n_cells):
        """Benchmark GPU performance at different scales."""
        # Create test case with proportional image size
        image_size = int(np.sqrt(n_cells * 10000))  # ~100 pixels per cell
        image_size = (image_size, image_size)

        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=n_cells, image_size=image_size
        )

        # Benchmark GPU
        start = time.time()
        overlap_gpu, cell_ids, nucleus_ids = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )
        gpu_time = time.time() - start

        # Benchmark CPU
        start = time.time()
        overlap_cpu, _, _ = _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)
        cpu_time = time.time() - start

        # Verify equivalence
        assert np.array_equal(overlap_cpu, overlap_gpu), "GPU and CPU results should match"

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"\n{n_cells} cells:")
        print(f"  CPU time: {cpu_time:.3f}s")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # GPU should be faster or comparable (allow 0.8x for small samples with overhead)
        # For large samples, expect significant speedup
        if n_cells >= 1000:
            assert speedup >= 2.0, f"Expected >2x speedup for {n_cells} cells, got {speedup:.1f}x"

    def test_gpu_stress_10k_cells(self):
        """Stress test with 10K cells."""
        cell_mask, nucleus_mask, expected = create_stress_test_case(
            n_cells=10000, image_size=(2000, 2000)
        )

        start = time.time()
        overlap_gpu, cell_ids, nucleus_ids = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )
        gpu_time = time.time() - start

        n_cells_actual = len(cell_ids)
        n_nuclei_actual = len(nucleus_ids)

        print(f"\n10K cell stress test:")
        print(f"  Actual cells: {n_cells_actual}")
        print(f"  Actual nuclei: {n_nuclei_actual}")
        print(f"  GPU time: {gpu_time:.2f}s")
        print(f"  Throughput: {n_cells_actual / gpu_time:.0f} cells/sec")

        # Should complete in reasonable time (<30s)
        assert gpu_time < 30.0, f"Expected <30s for 10K cells, got {gpu_time:.1f}s"


@requires_gpu()
class TestOverlapMatrixGPUErrorHandling:
    """Test GPU error handling and edge cases."""

    def test_gpu_mismatched_shapes(self):
        """Test GPU with mismatched mask shapes."""
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((200, 200), dtype=np.uint32)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Mask shapes must match"):
            compute_overlap_matrix_gpu(cell_mask, nucleus_mask)

    def test_gpu_3d_masks_squeeze(self):
        """Test GPU with 3D masks (should squeeze to 2D)."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=5)

        # Add extra dimension
        cell_mask_3d = cell_mask[np.newaxis, :, :]
        nucleus_mask_3d = nucleus_mask[np.newaxis, :, :]

        # Should squeeze and process normally
        overlap_gpu, cell_ids, nucleus_ids = compute_overlap_matrix_gpu(
            cell_mask_3d, nucleus_mask_3d
        )

        # Compare with 2D version
        overlap_2d, cell_ids_2d, nucleus_ids_2d = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask
        )

        assert np.array_equal(overlap_gpu, overlap_2d)

    def test_gpu_auto_batch_size(self):
        """Test GPU auto batch size estimation."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=50)

        # Use auto batch sizing (batch_size=0)
        overlap_gpu, cell_ids, nucleus_ids = compute_overlap_matrix_gpu(
            cell_mask, nucleus_mask, batch_size=0
        )

        # Compare with CPU
        overlap_cpu, _, _ = _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)

        assert np.array_equal(overlap_gpu, overlap_cpu)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
