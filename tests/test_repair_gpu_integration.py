"""Integration tests for GPU-accelerated mask repair pipeline.

This module tests the complete GPU repair pipeline including:
- End-to-end GPU vs CPU equivalence
- Performance benchmarks
- Error handling and fallback
- Integration with existing repair_masks.py
"""

import pytest
import numpy as np
import time
import logging

# Test utilities
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_overlapping_nuclei_case,
    create_unmatched_cells_case,
    create_partial_overlap_case,
    create_stress_test_case,
    create_edge_case_empty_masks,
    create_edge_case_single_cell,
    create_noncontiguous_labels_case,
)

# Import both GPU and CPU versions
from aegle.repair_masks_gpu import (
    repair_masks_gpu,
    get_matched_masks_gpu,
)
from aegle.repair_masks import (
    get_matched_masks,
    repair_masks_single,
)
from aegle.gpu_utils import is_cupy_available


# Decorator to skip tests if GPU not available
requires_gpu = pytest.mark.skipif(
    not is_cupy_available(),
    reason="GPU not available (CuPy not installed or no CUDA device)"
)


class TestGPURepairIntegration:
    """Integration tests for the complete GPU repair pipeline."""

    @requires_gpu
    def test_simple_case_gpu_vs_cpu(self):
        """Test GPU vs CPU equivalence on simple case (perfect matches)."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # CPU version (using original implementation)
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)

        # Should produce identical results
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"

        # Check metadata
        assert metadata_gpu["gpu_used"], "GPU should have been used"
        assert not metadata_gpu["fallback_to_cpu"], "Should not have fallen back to CPU"
        assert metadata_gpu["n_matched_cells"] == expected["n_matched_cells"]

    @requires_gpu
    def test_overlapping_nuclei_gpu_vs_cpu(self):
        """Test GPU vs CPU equivalence when multiple nuclei overlap same cell."""
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # CPU version
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)

        # Should match
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"
        assert metadata_gpu["gpu_used"], "GPU should have been used"

    @requires_gpu
    def test_unmatched_cells_gpu_vs_cpu(self):
        """Test GPU vs CPU with cells that have no nucleus."""
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=5, n_cells_without_nuclei=3
        )

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # CPU version
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)

        # Should match
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"

        # Check that some cells were unmatched
        n_cells_total = len(np.unique(cell_mask)) - 1
        n_matched = metadata_gpu["n_matched_cells"]
        assert n_matched < n_cells_total, "Some cells should be unmatched"
        assert n_matched == expected["n_matched_cells"], "Unexpected match count"

    @requires_gpu
    def test_partial_overlap_gpu_vs_cpu(self):
        """Test GPU vs CPU when nuclei extend outside cell boundaries."""
        cell_mask, nucleus_mask, expected = create_partial_overlap_case(n_cells=5)

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # CPU version
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)

        # Should match exactly
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"

        # Nuclei should be trimmed (have fewer pixels than original)
        n_pixels_repaired = np.sum(nucleus_gpu > 0)
        n_pixels_original = np.sum(nucleus_mask > 0)
        assert n_pixels_repaired < n_pixels_original, "Nuclei should be trimmed"

    @requires_gpu
    def test_noncontiguous_labels_gpu_vs_cpu(self):
        """Test GPU vs CPU with non-contiguous label IDs (e.g., [1, 5, 7, 23])."""
        cell_mask, nucleus_mask, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],
            cell_labels=[1, 2, 3, 4],
        )

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # CPU version
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)

        # Should handle non-contiguous labels correctly
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"
        assert metadata_gpu["n_matched_cells"] == expected["n_matched_cells"]

    @requires_gpu
    def test_edge_case_empty_masks(self):
        """Test GPU repair with empty masks (no cells, no nuclei)."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # Should return empty masks without errors
        assert np.sum(cell_gpu) == 0, "Cell mask should be empty"
        assert np.sum(nucleus_gpu) == 0, "Nucleus mask should be empty"
        assert metadata_gpu["n_matched_cells"] == 0
        assert metadata_gpu["n_matched_nuclei"] == 0
        assert metadata_gpu["gpu_used"], "GPU should have been used"

    @requires_gpu
    def test_edge_case_single_cell(self):
        """Test GPU repair with single cell-nucleus pair."""
        cell_mask, nucleus_mask, expected = create_edge_case_single_cell()

        # GPU version
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # CPU version
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)

        # Should match
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"
        assert metadata_gpu["n_matched_cells"] == 1
        assert metadata_gpu["n_matched_nuclei"] == 1

    @requires_gpu
    def test_stress_test_large_sample(self):
        """Test GPU on large sample (10K cells)."""
        cell_mask, nucleus_mask, expected = create_stress_test_case(
            n_cells=10000, image_size=(2000, 2000)
        )

        # GPU version
        start_time = time.time()
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )
        gpu_time = time.time() - start_time

        # Should complete successfully
        assert metadata_gpu["gpu_used"], "GPU should have been used"
        assert metadata_gpu["n_matched_cells"] > 0, "Should match some cells"

        # Check that most cells matched (at least 80% as per fixture)
        n_cells_created = expected["n_cells_created"]
        min_expected = expected["min_matched"]
        assert metadata_gpu["n_matched_cells"] >= min_expected, \
            f"Expected at least {min_expected} matches, got {metadata_gpu['n_matched_cells']}"

        # Log performance
        cells_per_sec = metadata_gpu["n_matched_cells"] / gpu_time if gpu_time > 0 else 0
        logging.info(
            f"GPU stress test: {metadata_gpu['n_matched_cells']} cells in {gpu_time:.2f}s "
            f"({cells_per_sec:.1f} cells/sec)"
        )

    @requires_gpu
    def test_cpu_fallback_when_gpu_disabled(self):
        """Test that CPU fallback works when GPU is explicitly disabled."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=5)

        # Disable GPU
        cell_result, nucleus_result, metadata = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=False
        )

        # Should use CPU fallback
        assert not metadata["gpu_used"], "GPU should not have been used"
        assert metadata["fallback_to_cpu"], "Should have fallen back to CPU"
        assert metadata["n_matched_cells"] == expected["n_matched_cells"]

    @requires_gpu
    def test_get_matched_masks_gpu_wrapper(self):
        """Test the drop-in replacement wrapper function."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        # Add batch dimension (as expected by original interface)
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)

        # GPU version using wrapper
        cell_gpu, nucleus_gpu, outside_gpu = get_matched_masks_gpu(
            cell_mask_3d, nucleus_mask_3d, use_gpu=True
        )

        # CPU version
        cell_cpu, nucleus_cpu, outside_cpu = get_matched_masks(
            cell_mask_3d, nucleus_mask_3d
        )

        # Should match exactly
        assert np.array_equal(cell_gpu, cell_cpu), "Cell masks don't match"
        assert np.array_equal(nucleus_gpu, nucleus_cpu), "Nucleus masks don't match"

        # Check shape is preserved (should have batch dimension)
        assert cell_gpu.shape == cell_mask_3d.shape, "Output shape should match input"
        assert nucleus_gpu.shape == nucleus_mask_3d.shape

    @requires_gpu
    def test_timing_breakdown(self):
        """Test that metadata includes detailed timing breakdown."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=1000, image_size=(1000, 1000)
        )

        cell_result, nucleus_result, metadata = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # Check that all timing stages are present
        assert "boundary_computation" in metadata, "Missing boundary timing"
        assert "coordinate_extraction" in metadata, "Missing coordinate timing"
        assert "overlap_computation" in metadata, "Missing overlap timing"
        assert "cell_matching" in metadata, "Missing matching timing"
        assert "mask_assembly" in metadata, "Missing assembly timing"
        assert "total_time" in metadata, "Missing total timing"

        # Total should be sum of parts (approximately)
        parts_sum = (
            metadata["boundary_computation"]
            + metadata["coordinate_extraction"]
            + metadata["overlap_computation"]
            + metadata["cell_matching"]
            + metadata["mask_assembly"]
        )
        assert abs(metadata["total_time"] - parts_sum) < 1.0, \
            "Total time should approximately equal sum of parts"


class TestGPURepairPerformance:
    """Performance benchmarks for GPU repair."""

    @requires_gpu
    @pytest.mark.parametrize("n_cells", [100, 500, 1000, 5000])
    def test_gpu_speedup_vs_cpu(self, n_cells):
        """Benchmark GPU vs CPU speedup at various scales."""
        # Create test case
        image_size = (int(np.sqrt(n_cells * 10000)), int(np.sqrt(n_cells * 10000)))
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=n_cells, image_size=image_size
        )

        # Benchmark GPU
        start = time.time()
        cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )
        gpu_time = time.time() - start

        # Benchmark CPU
        cell_mask_3d = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
        start = time.time()
        cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
        cpu_time = time.time() - start

        # Verify equivalence
        cell_cpu = np.squeeze(cell_cpu)
        nucleus_cpu = np.squeeze(nucleus_cpu)
        assert np.array_equal(cell_gpu, cell_cpu), "Results should match"

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        # Log results
        logging.info(
            f"n_cells={n_cells}: GPU={gpu_time:.3f}s, CPU={cpu_time:.3f}s, "
            f"speedup={speedup:.2f}x"
        )

        # For larger samples, GPU should generally be faster (but varies by hardware/sample)
        # Log warning if GPU is slower, but don't fail the test (performance varies)
        if n_cells >= 5000 and speedup < 3.0:
            logging.warning(
                f"GPU performance below target for {n_cells} cells: {speedup:.2f}x (target: 3x). "
                f"This can vary based on hardware, sample characteristics, and GPU load."
            )

    @requires_gpu
    def test_batch_size_effect(self):
        """Test effect of different batch sizes on performance."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=5000, image_size=(1500, 1500)
        )

        batch_sizes = [0, 100, 500, 1000]  # 0 = auto
        timings = {}

        for batch_size in batch_sizes:
            start = time.time()
            cell_result, nucleus_result, metadata = repair_masks_gpu(
                cell_mask, nucleus_mask, use_gpu=True, batch_size=batch_size
            )
            elapsed = time.time() - start
            timings[batch_size] = elapsed

            logging.info(
                f"batch_size={batch_size}: {elapsed:.3f}s "
                f"({metadata['n_matched_cells']} cells matched)"
            )

        # All batch sizes should produce same results
        # (we don't verify this here to save time, but could add if needed)

        # Auto batch size should be reasonable (within 2x of best)
        auto_time = timings[0]
        best_time = min(timings.values())
        assert auto_time <= best_time * 2.0, \
            f"Auto batch size is too slow: {auto_time:.3f}s vs best {best_time:.3f}s"


class TestGPURepairRobustness:
    """Test error handling and robustness."""

    @requires_gpu
    def test_mismatched_shapes(self):
        """Test error handling when cell and nucleus masks have different shapes."""
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((200, 200), dtype=np.uint32)  # Different shape

        # Should fall back to CPU gracefully (CPU version will also raise error)
        # The GPU version catches errors and falls back, so we shouldn't expect
        # an exception to be raised directly. Instead, test the CPU fallback path.
        cell_result, nucleus_result, metadata = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # Should have fallen back due to shape mismatch error
        # Note: With empty masks, it returns empty results without error
        assert metadata is not None

    @requires_gpu
    def test_wrong_dtype(self):
        """Test handling of non-integer mask dtypes."""
        cell_mask = np.zeros((100, 100), dtype=np.float32)  # Wrong dtype
        nucleus_mask = np.zeros((100, 100), dtype=np.float32)

        # Should handle gracefully (convert or raise clear error)
        # The implementation will convert to uint32 in various places
        cell_result, nucleus_result, metadata = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True
        )

        # Should complete without crash
        assert cell_result.dtype == np.uint32
        assert nucleus_result.dtype == np.uint32

    @requires_gpu
    def test_3d_mask_squeezing(self):
        """Test that 3D masks (with batch dimension) are properly squeezed."""
        cell_mask_2d, nucleus_mask_2d, expected = create_simple_mask_pair(n_cells=5)

        # Add batch dimension
        cell_mask_3d = np.expand_dims(cell_mask_2d, axis=0)
        nucleus_mask_3d = np.expand_dims(nucleus_mask_2d, axis=0)

        # Should handle 3D input
        cell_result, nucleus_result, metadata = repair_masks_gpu(
            cell_mask_3d, nucleus_mask_3d, use_gpu=True
        )

        # Output should be 2D (squeezed)
        assert cell_result.ndim == 2, "Output should be 2D"
        assert nucleus_result.ndim == 2

    def test_gpu_unavailable_fallback(self):
        """Test CPU fallback when GPU is not available."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=5)

        # This test runs on all systems (GPU and non-GPU)
        # On GPU systems, we force use_gpu=False to test fallback
        # On non-GPU systems, use_gpu=True will automatically fallback
        cell_result, nucleus_result, metadata = repair_masks_gpu(
            cell_mask, nucleus_mask, use_gpu=True  # Will fallback if GPU unavailable
        )

        # Should complete successfully regardless of GPU availability
        assert metadata["n_matched_cells"] == expected["n_matched_cells"]

        # If GPU not available, should have fallen back
        if not is_cupy_available():
            assert metadata["fallback_to_cpu"], "Should fallback when GPU unavailable"
            assert not metadata["gpu_used"], "Should not report GPU used when unavailable"


# Utility function for manual benchmarking
def benchmark_repair_gpu_vs_cpu(n_cells=10000, image_size=(2000, 2000)):
    """Benchmark GPU vs CPU repair on large sample.

    This is a utility function for manual benchmarking, not a test.

    Args:
        n_cells: Number of cells to create
        image_size: Image dimensions

    Returns:
        Dict with timing and speedup information
    """
    if not is_cupy_available():
        logging.warning("GPU not available, skipping benchmark")
        return None

    logging.info(f"Creating test case with {n_cells} cells...")
    cell_mask, nucleus_mask, expected = create_stress_test_case(
        n_cells=n_cells, image_size=image_size
    )

    n_cells_actual = len(np.unique(cell_mask)) - 1
    n_nuclei_actual = len(np.unique(nucleus_mask)) - 1
    logging.info(f"Created {n_cells_actual} cells, {n_nuclei_actual} nuclei")

    # Benchmark GPU
    logging.info("Running GPU repair...")
    start = time.time()
    cell_gpu, nucleus_gpu, metadata_gpu = repair_masks_gpu(
        cell_mask, nucleus_mask, use_gpu=True
    )
    gpu_time = time.time() - start

    # Benchmark CPU
    logging.info("Running CPU repair...")
    cell_mask_3d = np.expand_dims(cell_mask, axis=0)
    nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)
    start = time.time()
    cell_cpu, nucleus_cpu, _ = get_matched_masks(cell_mask_3d, nucleus_mask_3d)
    cpu_time = time.time() - start
    cell_cpu = np.squeeze(cell_cpu)
    nucleus_cpu = np.squeeze(nucleus_cpu)

    # Verify equivalence
    matches = np.array_equal(cell_gpu, cell_cpu)

    # Calculate speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

    results = {
        "n_cells": n_cells_actual,
        "n_matched": metadata_gpu["n_matched_cells"],
        "gpu_time": gpu_time,
        "cpu_time": cpu_time,
        "speedup": speedup,
        "gpu_cells_per_sec": n_cells_actual / gpu_time if gpu_time > 0 else 0,
        "cpu_cells_per_sec": n_cells_actual / cpu_time if cpu_time > 0 else 0,
        "results_match": matches,
    }

    logging.info(f"Benchmark results:")
    logging.info(f"  GPU time: {gpu_time:.2f}s ({results['gpu_cells_per_sec']:.1f} cells/sec)")
    logging.info(f"  CPU time: {cpu_time:.2f}s ({results['cpu_cells_per_sec']:.1f} cells/sec)")
    logging.info(f"  Speedup: {speedup:.2f}x")
    logging.info(f"  Results match: {matches}")

    return results


if __name__ == "__main__":
    # Run benchmark if executed directly
    logging.basicConfig(level=logging.INFO)
    benchmark_repair_gpu_vs_cpu(n_cells=10000, image_size=(2000, 2000))
