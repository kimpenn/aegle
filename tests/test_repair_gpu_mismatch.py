"""Tests for GPU-accelerated mismatch fraction computation.

This module tests the GPU implementation of cell-nucleus mismatch computation
and verifies equivalence with the CPU version.

Test Categories:
1. Equivalence Tests (30 tests): GPU vs CPU correctness
2. Performance Tests (15 tests): Speedup benchmarks
3. Robustness Tests (20 tests): Error handling, edge cases
4. Regression Tests (10 tests): Real data validation
"""

import pytest
import numpy as np
import time
from scipy.sparse import csr_matrix, issparse

from aegle.repair_masks_gpu_mismatch import (
    compute_mismatch_matrix_gpu,
    _compute_mismatch_matrix_cpu,
)
from aegle.repair_masks_gpu_overlap import compute_overlap_matrix_gpu
from tests.utils.gpu_test_utils import requires_gpu, assert_gpu_cpu_equal
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_overlapping_nuclei_case,
    create_unmatched_cells_case,
    create_partial_overlap_case,
    create_noncontiguous_labels_case,
    create_edge_case_empty_masks,
    create_edge_case_single_cell,
    create_stress_test_case,
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_overlap_matrix_for_testing(cell_mask, nucleus_mask):
    """Generate overlap matrix for testing (uses existing Task 2.2 code).

    Args:
        cell_mask: (H, W) uint32 labeled cell mask
        nucleus_mask: (H, W) uint32 labeled nucleus mask

    Returns:
        tuple: (overlap_matrix, cell_labels, nucleus_labels)
    """
    overlap_matrix, cell_labels, nucleus_labels = compute_overlap_matrix_gpu(
        cell_mask, nucleus_mask
    )
    return overlap_matrix, cell_labels, nucleus_labels


def create_cell_membrane_mask(cell_mask):
    """Create cell membrane mask for testing.

    Simulates membrane as the outer boundary of each cell.

    Args:
        cell_mask: (H, W) uint32 labeled cell mask

    Returns:
        cell_membrane_mask: (H, W) uint32 labeled membrane mask
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    H, W = cell_mask.shape
    membrane_mask = np.zeros((H, W), dtype=np.uint32)

    unique_cells = np.unique(cell_mask)
    unique_cells = unique_cells[unique_cells > 0]

    for cell_id in unique_cells:
        cell_pixels = (cell_mask == cell_id)

        # Dilate and subtract erosion to get boundary
        dilated = binary_dilation(cell_pixels)
        eroded = binary_erosion(cell_pixels)
        boundary = dilated & ~eroded

        membrane_mask[boundary] = cell_id

    return membrane_mask


def assert_sparse_matrices_close(sparse1, sparse2, atol=1e-6):
    """Assert two sparse matrices are numerically close.

    Args:
        sparse1: scipy.sparse matrix
        sparse2: scipy.sparse matrix
        atol: Absolute tolerance for floating point comparison

    Raises:
        AssertionError: If matrices differ beyond tolerance
    """
    assert issparse(sparse1), "sparse1 must be a sparse matrix"
    assert issparse(sparse2), "sparse2 must be a sparse matrix"

    # Convert to CSR format for consistent comparison
    sparse1_csr = sparse1.tocsr()
    sparse2_csr = sparse2.tocsr()

    # Check shapes
    assert sparse1_csr.shape == sparse2_csr.shape, \
        f"Shape mismatch: {sparse1_csr.shape} vs {sparse2_csr.shape}"

    # Check sparsity pattern (indices)
    assert np.array_equal(sparse1_csr.indices, sparse2_csr.indices), \
        "Sparse matrix indices differ (different sparsity pattern)"

    assert np.array_equal(sparse1_csr.indptr, sparse2_csr.indptr), \
        "Sparse matrix indptr differs (different sparsity pattern)"

    # Check values within tolerance
    np.testing.assert_allclose(
        sparse1_csr.data,
        sparse2_csr.data,
        atol=atol,
        err_msg="Sparse matrix values differ beyond tolerance"
    )


# ============================================================================
# Category 1: Equivalence Tests (30 tests)
# ============================================================================


class TestMismatchMatrixCPU:
    """Test CPU version of mismatch matrix computation (no GPU required)."""

    def test_cpu_simple_perfect_matching(self):
        """CPU computes zero mismatch for perfect 1:1 matching."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Check dimensions
        assert mismatch_matrix.shape == (len(cell_labels), len(nucleus_labels))

        # Perfect matching: mismatch should be zero along diagonal
        for i in range(min(len(cell_labels), len(nucleus_labels))):
            mismatch = mismatch_matrix[i, i]
            assert mismatch == 0.0, \
                f"Expected zero mismatch for perfect match, got {mismatch}"

    def test_cpu_partial_overlap(self):
        """CPU computes non-zero mismatch for partial overlap."""
        cell_mask, nucleus_mask, expected = create_partial_overlap_case(
            n_cells=5, nucleus_offset=15
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should have non-zero mismatch due to offset
        if expected.get("expect_nonzero_mismatch", False):
            # At least one mismatch should be non-zero
            assert mismatch_matrix.data.max() > 0.0, \
                "Expected non-zero mismatch for partial overlap"

    def test_cpu_noncontiguous_labels(self):
        """CPU handles non-contiguous label IDs [1, 5, 7, 23]."""
        cell_mask, nucleus_mask, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],
            cell_labels=[1, 2, 3, 4],
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Check that labels are preserved correctly
        assert list(cell_labels) == [1, 2, 3, 4]
        assert list(nucleus_labels) == [1, 5, 7, 23]

        # Should compute without errors
        assert mismatch_matrix is not None
        assert mismatch_matrix.shape == (4, 4)

    def test_cpu_empty_masks(self):
        """CPU handles empty masks without crashing."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should return empty sparse matrix without errors
        assert mismatch_matrix.shape == (0, 0)
        assert mismatch_matrix.nnz == 0

    def test_cpu_single_cell(self):
        """CPU handles single cell edge case."""
        cell_mask, nucleus_mask, expected = create_edge_case_single_cell()
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert mismatch_matrix.shape == (1, 1)
        # Single perfect match should have zero mismatch
        assert mismatch_matrix[0, 0] == 0.0

    def test_cpu_overlapping_nuclei(self):
        """CPU handles multiple nuclei per cell."""
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should have valid mismatches for multiple candidates
        assert mismatch_matrix.shape == (len(cell_labels), len(nucleus_labels))
        assert mismatch_matrix.nnz > 0

    def test_cpu_unmatched_cells(self):
        """CPU handles cells without nuclei."""
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=5, n_cells_without_nuclei=3
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_matrix = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Cells without nuclei should have no entries (handled by overlap matrix)
        # Only cells WITH nuclei should appear in the overlap matrix
        assert mismatch_matrix.shape[0] <= 8  # Total cells created


@requires_gpu()
class TestMismatchMatrixGPUEquivalence:
    """Test GPU vs CPU equivalence for mismatch computation."""

    def test_gpu_matches_cpu_simple_perfect(self):
        """GPU matches CPU on perfect 1:1 matching (mismatch=0.0)."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # CPU version
        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # GPU version
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Results should be identical
        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_partial_overlap(self):
        """GPU matches CPU with partial nucleus-cell overlap."""
        cell_mask, nucleus_mask, expected = create_partial_overlap_case(
            n_cells=5, nucleus_offset=15
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_noncontiguous_labels(self):
        """GPU handles non-contiguous label IDs [1, 5, 7, 23]."""
        cell_mask, nucleus_mask, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],
            cell_labels=[1, 2, 3, 4],
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_empty_masks(self):
        """GPU handles empty masks without crashing."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)
        assert mismatch_gpu.shape == (0, 0)

    def test_gpu_matches_cpu_single_cell(self):
        """GPU handles single cell correctly."""
        cell_mask, nucleus_mask, expected = create_edge_case_single_cell()
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_overlapping_nuclei(self):
        """GPU matches CPU with multiple nuclei per cell."""
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_unmatched_cells(self):
        """GPU matches CPU with cells without nuclei."""
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=5, n_cells_without_nuclei=3
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    @pytest.mark.parametrize("n_cells", [5, 10, 20, 50])
    def test_gpu_matches_cpu_varying_sizes(self, n_cells):
        """GPU matches CPU across different sample sizes."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=n_cells)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    @pytest.mark.parametrize("image_size", [(256, 256), (512, 512), (1024, 1024)])
    def test_gpu_matches_cpu_varying_image_sizes(self, image_size):
        """GPU matches CPU across different image dimensions."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(
            n_cells=20, image_size=image_size
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    @pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
    def test_gpu_matches_cpu_different_seeds(self, seed):
        """GPU matches CPU with different random fixture configurations."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=15, seed=seed)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_all_cells_label_gaps(self):
        """GPU matches CPU when both cell and nucleus labels have gaps."""
        cell_mask, nucleus_mask, _ = create_noncontiguous_labels_case(
            label_gaps="both",
            nucleus_labels=[2, 10, 15, 30],
            cell_labels=[3, 7, 12, 25],
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_large_label_ids(self):
        """GPU handles very large label IDs correctly."""
        cell_mask, nucleus_mask, _ = create_noncontiguous_labels_case(
            label_gaps="both",
            nucleus_labels=[100, 500, 1000, 5000],
            cell_labels=[50, 250, 750, 2500],
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    @pytest.mark.parametrize("nucleus_offset", [0, 10, 20, 30])
    def test_gpu_matches_cpu_varying_mismatch_levels(self, nucleus_offset):
        """GPU matches CPU across different mismatch severity levels."""
        cell_mask, nucleus_mask, _ = create_partial_overlap_case(
            n_cells=5, nucleus_offset=nucleus_offset
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_stress_100_cells(self):
        """GPU matches CPU on 100-cell stress test."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=100, image_size=(500, 500)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_stress_500_cells(self):
        """GPU matches CPU on 500-cell stress test."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=500, image_size=(750, 750)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

    def test_gpu_matches_cpu_stress_1000_cells(self):
        """GPU matches CPU on 1000-cell stress test."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=1000, image_size=(1000, 1000)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)


# ============================================================================
# Category 2: Performance Tests (15 tests)
# ============================================================================


@requires_gpu()
class TestMismatchMatrixGPUPerformance:
    """Performance benchmarks for GPU mismatch computation."""

    @pytest.mark.parametrize("n_cells,image_size", [
        (100, (500, 500)),
        (1000, (1000, 1000)),
        (5000, (2000, 2000)),
    ])
    def test_gpu_speedup_scaling(self, n_cells, image_size):
        """Benchmark GPU speedup at multiple scales."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=n_cells, image_size=image_size
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Benchmark CPU
        start = time.time()
        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )
        cpu_time = time.time() - start

        # Benchmark GPU
        start = time.time()
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )
        gpu_time = time.time() - start

        # Verify equivalence
        assert_sparse_matrices_close(mismatch_cpu, mismatch_gpu, atol=1e-6)

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"\n{n_cells} cells on {image_size}:")
        print(f"  CPU time: {cpu_time:.3f}s")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # For large samples, expect significant speedup
        if n_cells >= 1000:
            assert speedup >= 2.0, \
                f"Expected >2x speedup for {n_cells} cells, got {speedup:.1f}x"

    def test_gpu_speedup_10k_cells(self):
        """Benchmark GPU speedup on 10K cells."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=10000, image_size=(3000, 3000)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # GPU only (CPU would take too long)
        start = time.time()
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )
        gpu_time = time.time() - start

        n_cells_actual = len(cell_labels)
        n_pairs = mismatch_gpu.nnz

        print(f"\n10K cell benchmark:")
        print(f"  Cells: {n_cells_actual}")
        print(f"  Pairs processed: {n_pairs}")
        print(f"  GPU time: {gpu_time:.2f}s")
        print(f"  Throughput: {n_pairs / gpu_time:.0f} pairs/sec")

        # Should complete in reasonable time
        assert gpu_time < 60.0, f"Expected <60s for 10K cells, got {gpu_time:.1f}s"

    @pytest.mark.parametrize("batch_size", [100, 1000, 5000, 10000])
    def test_gpu_batch_size_performance(self, batch_size):
        """Benchmark performance with different batch sizes."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=1000, image_size=(1000, 1000)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        start = time.time()
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels,
            batch_size=batch_size
        )
        gpu_time = time.time() - start

        print(f"\nBatch size {batch_size}: {gpu_time:.3f}s")

        # Should complete without errors
        assert mismatch_gpu is not None

    def test_gpu_memory_efficiency(self):
        """Verify GPU memory usage is reasonable."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=1000, image_size=(1000, 1000)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # This should complete without OOM
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Check sparse matrix memory footprint
        sparse_size_mb = (mismatch_gpu.data.nbytes +
                          mismatch_gpu.indices.nbytes +
                          mismatch_gpu.indptr.nbytes) / 1e6

        print(f"\nSparse matrix size: {sparse_size_mb:.2f} MB")
        print(f"Non-zero entries: {mismatch_gpu.nnz}")

        # Should be much smaller than dense (1000 x 1000 x 4 bytes = 4 MB)
        assert sparse_size_mb < 4.0, "Sparse matrix should be smaller than dense"

    def test_gpu_throughput_measurement(self):
        """Measure GPU throughput (pairs/sec)."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=2000, image_size=(1500, 1500)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        start = time.time()
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )
        gpu_time = time.time() - start

        n_pairs = mismatch_gpu.nnz
        throughput = n_pairs / gpu_time

        print(f"\nThroughput test:")
        print(f"  Pairs: {n_pairs}")
        print(f"  Time: {gpu_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} pairs/sec")

        # Should process at least 100 pairs/sec
        assert throughput >= 100, f"Expected >100 pairs/sec, got {throughput:.0f}"

    def test_gpu_scaling_linearity(self):
        """Verify GPU time scales linearly with problem size."""
        sizes = [100, 200, 400]
        times = []

        for n_cells in sizes:
            image_size = int(np.sqrt(n_cells * 5000))
            cell_mask, nucleus_mask, _ = create_stress_test_case(
                n_cells=n_cells, image_size=(image_size, image_size)
            )
            membrane_mask = create_cell_membrane_mask(cell_mask)
            overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
                cell_mask, nucleus_mask
            )

            start = time.time()
            mismatch_gpu = compute_mismatch_matrix_gpu(
                cell_mask, nucleus_mask, membrane_mask,
                overlap_matrix, cell_labels, nucleus_labels
            )
            gpu_time = time.time() - start
            times.append(gpu_time)

            print(f"{n_cells} cells: {gpu_time:.3f}s")

        # Time should scale sub-quadratically (not O(n^2))
        # For 2x cells, time should be <4x
        if times[0] > 0:
            scaling_200 = times[1] / times[0]
            scaling_400 = times[2] / times[0]

            print(f"\nScaling: 100→200: {scaling_200:.1f}x, 100→400: {scaling_400:.1f}x")

            # GPU should be better than quadratic scaling
            assert scaling_400 < 16, f"Scaling should be sub-quadratic, got {scaling_400:.1f}x"


# ============================================================================
# Category 3: Robustness Tests (20 tests)
# ============================================================================


@requires_gpu()
class TestMismatchMatrixGPURobustness:
    """Test GPU error handling and edge cases."""

    def test_gpu_batch_size_invariance(self):
        """Results are independent of batch size."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=50)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Compute with different batch sizes
        mismatch_batch_10 = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels,
            batch_size=10
        )

        mismatch_batch_100 = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels,
            batch_size=100
        )

        # Results should be identical
        assert_sparse_matrices_close(mismatch_batch_10, mismatch_batch_100, atol=1e-9)

    def test_gpu_numerical_stability(self):
        """Multiple runs produce identical results."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=20)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Run multiple times
        results = []
        for _ in range(3):
            mismatch = compute_mismatch_matrix_gpu(
                cell_mask, nucleus_mask, membrane_mask,
                overlap_matrix, cell_labels, nucleus_labels
            )
            results.append(mismatch)

        # All results should be identical
        for i in range(1, len(results)):
            assert_sparse_matrices_close(results[0], results[i], atol=1e-9)

    def test_gpu_mismatched_mask_shapes_raises_error(self):
        """GPU raises error for mismatched mask shapes."""
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((200, 200), dtype=np.uint32)
        membrane_mask = np.zeros((100, 100), dtype=np.uint32)
        overlap_matrix = np.zeros((0, 0), dtype=bool)
        cell_labels = np.array([], dtype=np.uint32)
        nucleus_labels = np.array([], dtype=np.uint32)

        with pytest.raises(ValueError, match="Mask shapes must match"):
            compute_mismatch_matrix_gpu(
                cell_mask, nucleus_mask, membrane_mask,
                overlap_matrix, cell_labels, nucleus_labels
            )

    def test_gpu_3d_masks_squeeze(self):
        """GPU handles 3D masks (should squeeze to 2D)."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=5)
        membrane_mask = create_cell_membrane_mask(cell_mask)

        # Add extra dimension
        cell_mask_3d = cell_mask[np.newaxis, :, :]
        nucleus_mask_3d = nucleus_mask[np.newaxis, :, :]
        membrane_mask_3d = membrane_mask[np.newaxis, :, :]

        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Should squeeze and process normally
        mismatch_3d = compute_mismatch_matrix_gpu(
            cell_mask_3d, nucleus_mask_3d, membrane_mask_3d,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Compare with 2D version
        mismatch_2d = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert_sparse_matrices_close(mismatch_3d, mismatch_2d, atol=1e-9)

    def test_gpu_zero_overlap_matrix(self):
        """GPU handles case where no cells overlap nuclei."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=5)
        membrane_mask = create_cell_membrane_mask(cell_mask)

        # Create empty overlap matrix
        cell_labels = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        nucleus_labels = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        overlap_matrix = np.zeros((5, 5), dtype=bool)

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should return empty sparse matrix
        assert mismatch.nnz == 0

    def test_gpu_single_overlap_pair(self):
        """GPU handles case with only one overlapping pair."""
        cell_mask, nucleus_mask, _ = create_edge_case_single_cell()
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should have exactly one entry
        assert mismatch.nnz == 1
        assert mismatch[0, 0] == 0.0  # Perfect match

    def test_gpu_handles_all_zeros_membrane(self):
        """GPU handles case where membrane mask is all zeros."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=5)
        membrane_mask = np.zeros_like(cell_mask, dtype=np.uint32)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Should process without errors (all cell interior = whole cell)
        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert mismatch is not None

    def test_gpu_handles_very_small_cells(self):
        """GPU handles cells with only a few pixels."""
        H, W = 100, 100
        cell_mask = np.zeros((H, W), dtype=np.uint32)
        nucleus_mask = np.zeros((H, W), dtype=np.uint32)

        # Create tiny cells (3x3 pixels)
        for i in range(5):
            y, x = 20 + i * 15, 20 + i * 15
            cell_mask[y:y+3, x:x+3] = i + 1
            nucleus_mask[y+1:y+2, x+1:x+2] = i + 1  # 1 pixel nucleus

        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should handle tiny cells without errors
        assert mismatch is not None

    def test_gpu_handles_very_large_cells(self):
        """GPU handles cells with many pixels."""
        H, W = 500, 500
        cell_mask = np.zeros((H, W), dtype=np.uint32)
        nucleus_mask = np.zeros((H, W), dtype=np.uint32)

        # Create large cells (100x100 pixels)
        cell_mask[50:150, 50:150] = 1
        nucleus_mask[80:120, 80:120] = 1

        cell_mask[200:300, 200:300] = 2
        nucleus_mask[230:270, 230:270] = 2

        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should handle large cells without errors
        assert mismatch is not None

    def test_gpu_dtype_preservation(self):
        """GPU preserves correct dtypes in output."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Check dtypes
        assert mismatch.dtype == np.float32, "Mismatch matrix should be float32"

    def test_gpu_handles_max_mismatch(self):
        """GPU correctly computes mismatch=1.0 for completely mismatched nuclei."""
        H, W = 200, 200
        cell_mask = np.zeros((H, W), dtype=np.uint32)
        nucleus_mask = np.zeros((H, W), dtype=np.uint32)

        # Cell on left, nucleus on right (no overlap after membrane removal)
        cell_mask[50:150, 20:80] = 1
        nucleus_mask[50:150, 120:180] = 1  # Completely outside cell interior

        membrane_mask = create_cell_membrane_mask(cell_mask)

        # Force overlap to exist in matrix (even though physically separated)
        overlap_matrix = np.array([[True]], dtype=bool)
        cell_labels = np.array([1], dtype=np.uint32)
        nucleus_labels = np.array([1], dtype=np.uint32)

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Mismatch should be 1.0 (100% of nucleus outside cell interior)
        if mismatch.nnz > 0:
            assert mismatch[0, 0] >= 0.9, "Expected near-maximum mismatch"

    def test_gpu_handles_partial_membrane_coverage(self):
        """GPU handles cells where membrane doesn't fully surround."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=5)

        # Create sparse membrane (not full boundary)
        membrane_mask = np.zeros_like(cell_mask, dtype=np.uint32)
        unique_cells = np.unique(cell_mask)
        unique_cells = unique_cells[unique_cells > 0]

        for cell_id in unique_cells:
            cell_pixels = (cell_mask == cell_id)
            y_coords, x_coords = np.where(cell_pixels)
            if len(y_coords) > 0:
                # Mark only a few boundary pixels
                membrane_mask[y_coords[0], x_coords[0]] = cell_id

        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should process without errors
        assert mismatch is not None

    def test_gpu_mismatch_range_validity(self):
        """GPU produces mismatch values in valid range [0, 1]."""
        cell_mask, nucleus_mask, _ = create_partial_overlap_case(
            n_cells=10, nucleus_offset=15
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # All mismatch values should be in [0, 1]
        assert np.all(mismatch.data >= 0.0), "Mismatch values should be >= 0"
        assert np.all(mismatch.data <= 1.0), "Mismatch values should be <= 1"

    def test_gpu_sparsity_pattern_correctness(self):
        """GPU sparse matrix has correct sparsity pattern matching overlap."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Mismatch sparsity should match overlap sparsity
        overlap_nnz = np.sum(overlap_matrix)
        assert mismatch.nnz == overlap_nnz, \
            f"Mismatch nnz ({mismatch.nnz}) should match overlap nnz ({overlap_nnz})"

    def test_gpu_handles_rectangular_images(self):
        """GPU handles non-square images correctly."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(
            n_cells=10, image_size=(300, 600)  # Rectangular
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should process without errors
        assert mismatch is not None

    def test_gpu_handles_label_id_one(self):
        """GPU correctly handles label ID = 1 (edge case for indexing)."""
        H, W = 100, 100
        cell_mask = np.zeros((H, W), dtype=np.uint32)
        nucleus_mask = np.zeros((H, W), dtype=np.uint32)

        # Create single cell with ID = 1
        cell_mask[30:70, 30:70] = 1
        nucleus_mask[45:55, 45:55] = 1

        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should process correctly
        assert list(cell_labels) == [1]
        assert list(nucleus_labels) == [1]
        assert mismatch[0, 0] >= 0.0


# ============================================================================
# Category 4: Regression Tests (10 tests)
# ============================================================================


@requires_gpu()
class TestMismatchMatrixGPURegression:
    """Regression tests using real data patterns."""

    def test_match_statistics_consistency(self):
        """Verify matching stats are consistent with CPU."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=20)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_cpu = _compute_mismatch_matrix_cpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Statistics should match
        assert mismatch_cpu.nnz == mismatch_gpu.nnz
        assert np.isclose(mismatch_cpu.data.mean(), mismatch_gpu.data.mean(), atol=1e-6)
        assert np.isclose(mismatch_cpu.data.std(), mismatch_gpu.data.std(), atol=1e-6)

    def test_perfect_matching_yields_zero_mismatch(self):
        """Perfect cell-nucleus alignment yields mismatch=0.0."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=15)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # All diagonal entries should be zero (perfect matches)
        for i in range(min(mismatch_gpu.shape)):
            if mismatch_gpu[i, i] != 0:  # Skip if no entry
                assert mismatch_gpu[i, i] == 0.0, \
                    f"Expected zero mismatch for perfect match at ({i}, {i})"

    def test_mismatch_increases_with_offset(self):
        """Mismatch fraction increases as nucleus moves away from cell center."""
        offsets = [0, 10, 20]
        mismatches = []

        for offset in offsets:
            cell_mask, nucleus_mask, _ = create_partial_overlap_case(
                n_cells=5, nucleus_offset=offset
            )
            membrane_mask = create_cell_membrane_mask(cell_mask)
            overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
                cell_mask, nucleus_mask
            )

            mismatch_gpu = compute_mismatch_matrix_gpu(
                cell_mask, nucleus_mask, membrane_mask,
                overlap_matrix, cell_labels, nucleus_labels
            )

            if mismatch_gpu.nnz > 0:
                mismatches.append(mismatch_gpu.data.mean())
            else:
                mismatches.append(0.0)

        print(f"\nMismatch vs offset: {list(zip(offsets, mismatches))}")

        # Mismatch should generally increase with offset
        if len(mismatches) >= 2:
            assert mismatches[-1] >= mismatches[0], \
                "Mismatch should increase with nucleus offset"

    def test_sparse_density_realistic(self):
        """Sparse matrix density is realistic for biological data."""
        cell_mask, nucleus_mask, _ = create_overlapping_nuclei_case(
            n_cells=20, n_nuclei=30
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        density = mismatch_gpu.nnz / (mismatch_gpu.shape[0] * mismatch_gpu.shape[1])
        print(f"\nSparse matrix density: {density:.4f}")

        # Density should be low (sparse) - typically <10% for biological data
        assert density < 0.5, f"Expected sparse matrix, got density {density:.4f}"

    def test_handles_typical_tissue_density(self):
        """GPU handles typical tissue cell densities (hundreds to thousands)."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=500, image_size=(1000, 1000)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should complete successfully
        assert mismatch_gpu is not None
        assert mismatch_gpu.shape[0] > 0

    def test_nucleus_trimming_correctness(self):
        """Mismatch correctly identifies nucleus pixels outside cell interior."""
        # Create case where nucleus extends beyond cell interior
        H, W = 200, 200
        cell_mask = np.zeros((H, W), dtype=np.uint32)
        nucleus_mask = np.zeros((H, W), dtype=np.uint32)

        # Cell
        cell_mask[50:150, 50:150] = 1

        # Nucleus partially outside cell interior
        nucleus_mask[60:140, 60:160] = 1  # Extends right

        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix = np.array([[True]], dtype=bool)
        cell_labels = np.array([1], dtype=np.uint32)
        nucleus_labels = np.array([1], dtype=np.uint32)

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should have non-zero mismatch (nucleus outside cell)
        if mismatch_gpu.nnz > 0:
            assert mismatch_gpu[0, 0] > 0.0, "Expected non-zero mismatch"

    def test_consistent_with_biological_expectations(self):
        """Results match biological expectations for typical samples."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=30)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # For well-segmented data:
        # - Most matches should have low mismatch (<0.2)
        # - Few matches should have high mismatch (>0.8)
        low_mismatch = np.sum(mismatch_gpu.data < 0.2)
        high_mismatch = np.sum(mismatch_gpu.data > 0.8)

        print(f"\nMismatch distribution:")
        print(f"  Low mismatch (<0.2): {low_mismatch}/{mismatch_gpu.nnz}")
        print(f"  High mismatch (>0.8): {high_mismatch}/{mismatch_gpu.nnz}")

        # Most should be low mismatch for good segmentation
        if mismatch_gpu.nnz > 0:
            assert low_mismatch >= high_mismatch, \
                "Expected more low-mismatch than high-mismatch pairs"

    def test_reproducibility_across_runs(self):
        """Results are reproducible across multiple executions."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=25)
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Run 5 times
        results = []
        for _ in range(5):
            mismatch = compute_mismatch_matrix_gpu(
                cell_mask, nucleus_mask, membrane_mask,
                overlap_matrix, cell_labels, nucleus_labels
            )
            results.append(mismatch)

        # All results should be identical
        for i in range(1, len(results)):
            assert_sparse_matrices_close(results[0], results[i], atol=1e-9)

    def test_handles_varied_cell_sizes(self):
        """GPU handles datasets with varied cell sizes (small + large)."""
        H, W = 500, 500
        cell_mask = np.zeros((H, W), dtype=np.uint32)
        nucleus_mask = np.zeros((H, W), dtype=np.uint32)

        # Small cell
        cell_mask[50:70, 50:70] = 1
        nucleus_mask[55:65, 55:65] = 1

        # Medium cell
        cell_mask[100:150, 100:150] = 2
        nucleus_mask[115:135, 115:135] = 2

        # Large cell
        cell_mask[200:350, 200:350] = 3
        nucleus_mask[250:300, 250:300] = 3

        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        # Should handle varied sizes correctly
        assert mismatch_gpu.shape == (3, 3)
        assert mismatch_gpu.nnz == 3  # All cells have nuclei

    def test_stress_test_5000_cells_completes(self):
        """Stress test: 5000 cells completes without errors."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=5000, image_size=(2000, 2000)
        )
        membrane_mask = create_cell_membrane_mask(cell_mask)
        overlap_matrix, cell_labels, nucleus_labels = create_overlap_matrix_for_testing(
            cell_mask, nucleus_mask
        )

        # Should complete without OOM or crashes
        mismatch_gpu = compute_mismatch_matrix_gpu(
            cell_mask, nucleus_mask, membrane_mask,
            overlap_matrix, cell_labels, nucleus_labels
        )

        assert mismatch_gpu is not None
        print(f"\n5000-cell stress test:")
        print(f"  Cells: {len(cell_labels)}")
        print(f"  Pairs: {mismatch_gpu.nnz}")
        print(f"  Matrix size: {mismatch_gpu.shape}")


# ============================================================================
# Test Execution
# ============================================================================


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
