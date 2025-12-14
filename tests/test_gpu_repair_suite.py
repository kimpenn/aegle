"""Comprehensive GPU repair test suite.

This test suite validates the entire GPU repair pipeline across multiple
scenarios, sizes, and edge cases. It serves as a regression test to ensure
GPU components work correctly together.

The suite includes:
- Full pipeline tests at multiple scales (100 - 10K cells)
- GPU vs CPU equivalence verification on all fixtures
- Non-contiguous label handling
- Performance scaling validation
- Memory management tests
- Error recovery tests
- CPU fallback scenarios

Run with:
    pytest tests/test_gpu_repair_suite.py -v
    pytest tests/test_gpu_repair_suite.py -v -m "not stress"  # Skip stress tests
"""

import pytest
import numpy as np
import time
import logging
from typing import Dict, Any

from aegle.repair_masks_gpu import repair_masks_gpu, get_matched_masks_gpu
from aegle.repair_masks import get_matched_masks
from tests.utils.gpu_test_utils import requires_gpu, assert_gpu_cpu_equal
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

logger = logging.getLogger(__name__)


@requires_gpu()
class TestGPURepairSuite:
    """Comprehensive GPU repair validation."""

    def test_full_pipeline_small_sample(self):
        """Test GPU pipeline on small sample (100 cells)."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(
            n_cells=100,
            image_size=(1024, 1024),
            cell_radius=30,
            nucleus_radius=15,
            seed=42
        )

        # Run GPU pipeline
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)

        # Verify basic correctness
        assert cell_matched is not None
        assert nucleus_matched is not None
        assert metadata is not None

        # Check cell counts match expected
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matched cells, got {n_matched}"

        # Verify metadata includes timing
        metadata = metadata
        assert "timing" in metadata
        assert metadata["used_gpu"] is True

    def test_full_pipeline_medium_sample(self):
        """Test GPU pipeline on medium sample (1K cells)."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(
            n_cells=1000,
            image_size=(2048, 2048),
            cell_radius=30,
            nucleus_radius=15,
            seed=43
        )

        start_time = time.time()
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
        elapsed = time.time() - start_time

        # Verify correctness
        assert cell_matched is not None
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched >= expected["n_matched_cells"] * 0.95, \
            f"Expected ~{expected['n_matched_cells']} cells, got {n_matched}"

        # Performance sanity check: should complete in reasonable time
        assert elapsed < 60, f"GPU took {elapsed:.1f}s for 1K cells (too slow)"

        logger.info(f"GPU processed 1K cells in {elapsed:.2f}s ({1000/elapsed:.1f} cells/sec)")

    @pytest.mark.stress
    def test_full_pipeline_large_sample(self):
        """Test GPU pipeline on large sample (10K cells)."""
        cell_mask, nucleus_mask, expected = create_stress_test_case(
            n_cells=10000,
            image_size=(4096, 4096),
            seed=44
        )

        start_time = time.time()
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
        elapsed = time.time() - start_time

        # Verify correctness
        assert cell_matched is not None

        # Check reasonable match rate
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched >= expected.get("min_matched", 0), \
            f"Too few matched cells: {n_matched}"

        # Performance: should be much faster than CPU (which would take minutes)
        assert elapsed < 300, f"GPU took {elapsed:.1f}s for 10K cells (too slow)"

        logger.info(f"GPU processed 10K cells in {elapsed:.2f}s ({10000/elapsed:.1f} cells/sec)")

    def test_gpu_vs_cpu_equivalence_all_fixtures(self):
        """Test GPU matches CPU on all test fixtures."""
        fixtures = [
            ("simple", create_simple_mask_pair(n_cells=50)),
            ("overlapping_nuclei", create_overlapping_nuclei_case(n_cells=20)),
            ("unmatched_cells", create_unmatched_cells_case(n_cells_with_nuclei=15)),
            ("partial_overlap", create_partial_overlap_case(n_cells=10)),
            ("empty_masks", create_edge_case_empty_masks()),
            ("single_cell", create_edge_case_single_cell()),
        ]

        for fixture_name, (cell_mask, nucleus_mask, expected) in fixtures:
            # CPU version
            result_cpu = get_matched_masks(
                cell_mask,
                nucleus_mask,
                mismatch_tolerance=0.2,
                search_radius=5
            )

            # GPU version (using wrapper for same interface)
            result_gpu = get_matched_masks_gpu(
                cell_mask,
                nucleus_mask,
                mismatch_tolerance=0.2,
                search_radius=5
            )

            # Compare cell matched masks (must be exact for label masks)
            assert np.array_equal(
                result_cpu["cell_matched_mask"],
                cell_matched
            ), f"Fixture '{fixture_name}': GPU cell mask differs from CPU"

            # Compare nucleus matched masks
            assert np.array_equal(
                result_cpu["nucleus_matched_mask"],
                nucleus_matched
            ), f"Fixture '{fixture_name}': GPU nucleus mask differs from CPU"

            logger.info(f"Fixture '{fixture_name}': GPU matches CPU ✓")

    def test_non_contiguous_labels_full_pipeline(self):
        """Test non-contiguous labels work end-to-end."""
        # Test with non-contiguous nucleus labels
        cell_mask, nucleus_mask, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],
            cell_labels=[1, 2, 3, 4]
        )

        # Run GPU pipeline
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)

        # Verify no errors and basic correctness
        assert cell_matched is not None

        # CPU version for comparison
        result_cpu = get_matched_masks(
            cell_mask,
            nucleus_mask,
            mismatch_tolerance=0.2,
            search_radius=5
        )

        # Should produce identical results
        assert np.array_equal(
            result_cpu["cell_matched_mask"],
            cell_matched
        ), "GPU non-contiguous label handling differs from CPU"

    def test_performance_scaling(self):
        """Test performance scales correctly with sample size."""
        scales = [100, 500, 1000]
        timings = []

        for n_cells in scales:
            # Create sample
            image_size = int(np.sqrt(n_cells) * 100)
            cell_mask, nucleus_mask, _ = create_simple_mask_pair(
                n_cells=n_cells,
                image_size=(image_size, image_size),
                cell_radius=30,
                nucleus_radius=15,
                seed=100 + n_cells
            )

            # Benchmark GPU
            start = time.time()
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
            elapsed = time.time() - start

            assert cell_matched is not None, f"GPU failed on {n_cells} cells"
            timings.append((n_cells, elapsed))

        # Log results
        for n_cells, elapsed in timings:
            cells_per_sec = n_cells / elapsed
            logger.info(f"{n_cells} cells: {elapsed:.2f}s ({cells_per_sec:.1f} cells/sec)")

        # Verify scaling: larger samples should have better throughput (cells/sec)
        # due to better GPU utilization
        throughputs = [n / t for n, t in timings]

        # At minimum, throughput should not degrade catastrophically
        # (allow some variance due to GPU warmup)
        assert throughputs[-1] >= throughputs[0] * 0.5, \
            "GPU throughput degrades too much with scale"

    def test_memory_efficiency(self):
        """Test GPU memory is properly managed."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        # Check initial memory
        mem_pool = cp.get_default_memory_pool()
        initial_used = mem_pool.used_bytes()

        # Process moderate sample
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(
            n_cells=500,
            image_size=(1024, 1024)
        )

        # Run GPU pipeline
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
        assert cell_matched is not None

        # Check memory after processing
        mid_used = mem_pool.used_bytes()

        # Process second sample (memory should be reused)
        result2 = repair_masks_gpu(cell_mask, nucleus_mask)
        assert result2 is not None

        final_used = mem_pool.used_bytes()

        # Memory should not grow unboundedly between runs
        # (allow some growth for caching, but not 2x)
        memory_growth = (final_used - mid_used) / max(1, mid_used - initial_used)
        assert memory_growth < 1.5, \
            f"Memory usage grew by {memory_growth:.1f}x between runs (possible leak)"

        logger.info(f"GPU memory: initial={initial_used/1e6:.1f}MB, "
                   f"mid={mid_used/1e6:.1f}MB, final={final_used/1e6:.1f}MB")

    def test_error_recovery(self):
        """Test graceful handling of GPU errors."""
        # Test with invalid input (should handle gracefully)
        cell_mask = np.array([[]], dtype=np.uint32)
        nucleus_mask = np.array([[]], dtype=np.uint32)

        # Should not crash, should return valid (empty) result or raise clear error
        try:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
            # If it succeeds, result should be valid empty result
            assert cell_matched is not None
            assert cell_matched is not None
        except (ValueError, RuntimeError) as e:
            # If it raises an error, it should be clear and descriptive
            assert len(str(e)) > 0, "Error message should be descriptive"
            logger.info(f"GPU correctly raised error on invalid input: {e}")

    def test_cpu_fallback_scenarios(self):
        """Test all CPU fallback code paths."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Test 1: Explicit CPU mode
        result_cpu = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=False)
        assert result_cpu is not None
        assert result_cpu["metadata"]["used_gpu"] is False

        # Test 2: Empty masks (should work on both CPU and GPU)
        empty_cell, empty_nucleus, _ = create_edge_case_empty_masks()
        result_empty = repair_masks_gpu(empty_cell, empty_nucleus, use_gpu=True)
        assert result_empty is not None

        # Test 3: Single cell (edge case)
        single_cell, single_nucleus, _ = create_edge_case_single_cell()
        result_single = repair_masks_gpu(single_cell, single_nucleus, use_gpu=True)
        assert result_single is not None

        logger.info("All CPU fallback scenarios tested ✓")


@requires_gpu()
class TestGPURepairRegressionPrevention:
    """Tests to prevent regressions in GPU repair functionality."""

    def test_gpu_produces_identical_masks(self):
        """Ensure GPU produces bit-identical masks to CPU."""
        # Use deterministic fixture
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=50, seed=999)

        # Run both versions
        result_cpu = get_matched_masks(cell_mask, nucleus_mask)
        result_gpu = get_matched_masks_gpu(cell_mask, nucleus_mask)

        # Masks must be bit-identical (not just similar)
        assert np.array_equal(result_cpu["cell_matched_mask"], cell_matched)
        assert np.array_equal(result_cpu["nucleus_matched_mask"], nucleus_matched)

    def test_gpu_metadata_completeness(self):
        """Ensure GPU returns complete metadata."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=20)

        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)

        # Check required metadata fields
        metadata = metadata
        required_fields = ["used_gpu", "timing", "n_cells_processed", "n_nuclei_processed"]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Check timing breakdown
        timing = metadata["timing"]
        assert "total" in timing
        assert timing["total"] > 0, "Total time should be positive"

    def test_gpu_handles_all_dtype_masks(self):
        """Ensure GPU handles various mask dtypes correctly."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Test with different dtypes
        for dtype in [np.uint8, np.uint16, np.uint32, np.int32]:
            cell_typed = cell_mask.astype(dtype)
            nucleus_typed = nucleus_mask.astype(dtype)

            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_typed, nucleus_typed)
            assert cell_matched is not None, f"Failed with dtype {dtype}"

            # Output should be consistent dtype
            assert cell_matched.dtype in [np.uint32, np.int32]


@pytest.mark.stress
@requires_gpu()
class TestGPURepairStress:
    """Stress tests for GPU repair (run separately with -m stress)."""

    def test_very_large_sample(self):
        """Test with 50K+ cell sample."""
        pytest.skip("Very large stress test - run manually on production hardware")

        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=50000,
            image_size=(8192, 8192)
        )

        start = time.time()
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
        elapsed = time.time() - start

        assert cell_matched is not None
        logger.info(f"Processed 50K cells in {elapsed:.1f}s ({50000/elapsed:.1f} cells/sec)")

    def test_extreme_label_gaps(self):
        """Test with extreme non-contiguous labels (e.g., [1, 1000000])."""
        # Create mask with huge label gaps
        cell_mask = np.zeros((500, 500), dtype=np.uint32)
        nucleus_mask = np.zeros((500, 500), dtype=np.uint32)

        # Create cells with extreme label IDs
        extreme_labels = [1, 1000, 100000, 1000000]
        spacing = 100

        for i, label_id in enumerate(extreme_labels):
            y = 100 + i * spacing // len(extreme_labels)
            x = 100 + i * spacing // len(extreme_labels)

            # Simple square cells
            cell_mask[y:y+40, x:x+40] = label_id
            nucleus_mask[y+10:y+30, x+10:x+30] = label_id

        # Should handle extreme labels without errors
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
        assert cell_matched is not None

        # Verify matches
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched == len(extreme_labels)

    def test_repeated_runs_no_memory_leak(self):
        """Run 100 iterations and verify memory doesn't grow."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        # Create test sample
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=100)

        # Run once to warm up
        _ = repair_masks_gpu(cell_mask, nucleus_mask)

        # Record baseline memory
        mem_pool = cp.get_default_memory_pool()
        baseline_memory = mem_pool.used_bytes()

        # Run many iterations
        n_iterations = 100
        for i in range(n_iterations):
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
            assert cell_matched is not None

        # Check final memory
        final_memory = mem_pool.used_bytes()

        # Memory should not grow significantly (allow 10% tolerance)
        memory_ratio = final_memory / max(baseline_memory, 1e6)  # Avoid div by 0
        assert memory_ratio < 1.1, \
            f"Memory grew {memory_ratio:.2f}x after {n_iterations} iterations (leak?)"

        logger.info(f"Memory stable after {n_iterations} iterations: "
                   f"{baseline_memory/1e6:.1f}MB → {final_memory/1e6:.1f}MB")

    def test_concurrent_gpu_operations(self):
        """Test multiple GPU operations don't interfere."""
        # This is a simplified test - true concurrency would require threading/multiprocessing
        # Here we just verify sequential operations don't corrupt state

        samples = []
        for i in range(5):
            cell_mask, nucleus_mask, _ = create_simple_mask_pair(
                n_cells=50,
                seed=1000 + i
            )
            samples.append((cell_mask, nucleus_mask))

        # Process all samples
        results = []
        for cell_mask, nucleus_mask in samples:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
            results.append(result)

        # Verify all succeeded
        assert len(results) == 5
        for result in results:
            assert cell_matched is not None
            assert cell_matched is not None

        # Re-process and verify identical results (deterministic)
        for i, (cell_mask, nucleus_mask) in enumerate(samples):
            result2 = repair_masks_gpu(cell_mask, nucleus_mask)
            assert np.array_equal(
                results[i]["cell_matched_mask"],
                result2["cell_matched_mask"]
            ), f"Non-deterministic result for sample {i}"


if __name__ == "__main__":
    # Run with: python tests/test_gpu_repair_suite.py
    pytest.main([__file__, "-v", "-s"])
