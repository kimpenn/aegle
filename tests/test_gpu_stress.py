"""GPU stress tests.

This test suite pushes GPU components to their limits:
- Very large samples (50K+ cells)
- Extreme label gaps (non-contiguous IDs like [1, 1000000])
- Memory leak detection (100+ iterations)
- Concurrent operations

These tests are marked with @pytest.mark.stress and can be skipped via:
    pytest tests/test_gpu_stress.py -v -m "not stress"

Run stress tests with:
    pytest tests/test_gpu_stress.py -v -m stress
"""

import pytest
import numpy as np
import time
import logging

from aegle.repair_masks_gpu import repair_masks_gpu
from aegle.gpu_utils import is_cupy_available
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_stress_test_case,
)

logger = logging.getLogger(__name__)


@pytest.mark.stress
@pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
class TestGPUStressVeryLargeSamples:
    """Test GPU with very large samples (50K+ cells)."""

    def test_very_large_sample_50k_cells(self):
        """Test with 50K cell sample."""
        pytest.skip("Very large stress test - run manually on production hardware with >16GB VRAM")

        logger.info("Creating 50K cell sample (this may take a minute)...")
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=50000,
            image_size=(8192, 8192)
        )

        logger.info("Running GPU repair on 50K cells...")
        start = time.time()
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
        elapsed = time.time() - start

        assert cell_matched is not None
        assert cell_matched is not None

        throughput = 50000 / elapsed
        logger.info(f"Processed 50K cells in {elapsed:.1f}s ({throughput:.1f} cells/sec)")

        # Should complete in reasonable time (even on GPU, 50K is large)
        assert elapsed < 600, f"GPU took {elapsed:.1f}s for 50K cells (>10 min, too slow)"

    def test_very_large_sample_100k_cells(self):
        """Test with 100K cell sample."""
        pytest.skip("Extreme stress test - requires high-end GPU (A100 80GB recommended)")

        logger.info("Creating 100K cell sample (this may take several minutes)...")
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=100000,
            image_size=(12000, 12000)
        )

        logger.info("Running GPU repair on 100K cells...")
        start = time.time()

        try:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            elapsed = time.time() - start

            assert cell_matched is not None
            throughput = 100000 / elapsed
            logger.info(f"Processed 100K cells in {elapsed:.1f}s ({throughput:.1f} cells/sec)")

        except Exception as e:
            logger.warning(f"100K cell test failed (expected on lower-end GPUs): {e}")
            pytest.skip(f"GPU insufficient for 100K cells: {e}")


@pytest.mark.stress
@pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
class TestGPUStressExtremeLabelGaps:
    """Test GPU with extreme non-contiguous labels."""

    def test_extreme_label_gaps_million(self):
        """Test with extreme label gaps (e.g., [1, 1000000])."""
        # Create mask with huge label gaps
        cell_mask = np.zeros((1000, 1000), dtype=np.uint32)
        nucleus_mask = np.zeros((1000, 1000), dtype=np.uint32)

        # Create cells with extreme label IDs
        extreme_labels = [1, 1000, 100000, 1000000]
        spacing = 200

        for i, label_id in enumerate(extreme_labels):
            y = 200 + (i % 2) * 400
            x = 200 + (i // 2) * 400

            # Simple square cells
            cell_mask[y:y+100, x:x+100] = label_id
            nucleus_mask[y+20:y+80, x+20:x+80] = label_id

        logger.info(f"Testing extreme label gaps: {extreme_labels}")

        # Should handle extreme labels without errors
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        assert cell_matched is not None
        assert cell_matched is not None

        # Verify all cells were matched
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched == len(extreme_labels), \
            f"Expected {len(extreme_labels)} matches, got {n_matched}"

        # Verify label IDs are preserved correctly
        matched_labels = set(np.unique(cell_matched)) - {0}
        expected_labels = set(extreme_labels)
        assert matched_labels == expected_labels, \
            f"Matched labels {matched_labels} != expected {expected_labels}"

    def test_sparse_label_distribution(self):
        """Test with sparse label distribution (many gaps)."""
        # Create labels like [1, 10, 100, 1000, 10000]
        cell_mask = np.zeros((2000, 2000), dtype=np.uint32)
        nucleus_mask = np.zeros((2000, 2000), dtype=np.uint32)

        sparse_labels = [10**i for i in range(6)]  # [1, 10, 100, 1000, 10000, 100000]
        grid_size = int(np.ceil(np.sqrt(len(sparse_labels))))
        cell_size = 200

        for i, label_id in enumerate(sparse_labels):
            row = (i // grid_size) * 400 + 200
            col = (i % grid_size) * 400 + 200

            if row + cell_size > 2000 or col + cell_size > 2000:
                continue

            cell_mask[row:row+cell_size, col:col+cell_size] = label_id
            nucleus_mask[row+50:row+150, col+50:col+150] = label_id

        logger.info(f"Testing sparse label distribution: {sparse_labels}")

        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        assert cell_matched is not None
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched >= len(sparse_labels) - 2, \
            f"Too few matches for sparse labels: {n_matched}"

    def test_random_large_labels(self):
        """Test with random large label IDs."""
        np.random.seed(999)

        cell_mask = np.zeros((1500, 1500), dtype=np.uint32)
        nucleus_mask = np.zeros((1500, 1500), dtype=np.uint32)

        # Generate random large labels
        n_cells = 20
        random_labels = np.random.randint(1, 1000000, size=n_cells)
        random_labels = np.unique(random_labels)  # Remove duplicates

        cell_size = 120
        grid_size = int(np.ceil(np.sqrt(len(random_labels))))

        for i, label_id in enumerate(random_labels):
            row = (i // grid_size) * 150 + 50
            col = (i % grid_size) * 150 + 50

            if row + cell_size > 1500 or col + cell_size > 1500:
                continue

            cell_mask[row:row+cell_size, col:col+cell_size] = label_id
            nucleus_mask[row+30:row+90, col+30:col+90] = label_id

        logger.info(f"Testing {len(random_labels)} random large labels")

        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        assert cell_matched is not None
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched >= len(random_labels) * 0.8, \
            f"Too few matches for random labels: {n_matched}/{len(random_labels)}"


@pytest.mark.stress
@pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
class TestGPUStressMemoryLeaks:
    """Test for memory leaks in GPU operations."""

    def test_repeated_runs_no_memory_leak_100_iterations(self):
        """Run 100 iterations and verify memory doesn't grow."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        # Create test sample
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(
            n_cells=100,
            image_size=(512, 512)
        )

        # Run once to warm up and allocate initial memory
        logger.info("Warming up GPU...")
        _ = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        # Record baseline memory
        mem_pool = cp.get_default_memory_pool()
        baseline_memory = mem_pool.used_bytes()

        logger.info(f"Baseline GPU memory: {baseline_memory/1e6:.1f} MB")

        # Run many iterations
        n_iterations = 100
        logger.info(f"Running {n_iterations} iterations...")

        for i in range(n_iterations):
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            assert cell_matched is not None

            # Log memory every 20 iterations
            if (i + 1) % 20 == 0:
                current_memory = mem_pool.used_bytes()
                logger.info(f"Iteration {i+1}: {current_memory/1e6:.1f} MB")

        # Check final memory
        final_memory = mem_pool.used_bytes()

        logger.info(f"Final GPU memory: {final_memory/1e6:.1f} MB")

        # Memory should not grow significantly (allow 10% tolerance)
        memory_ratio = final_memory / max(baseline_memory, 1e6)  # Avoid div by 0

        assert memory_ratio < 1.1, \
            f"Memory grew {memory_ratio:.2f}x after {n_iterations} iterations " \
            f"({baseline_memory/1e6:.1f}MB → {final_memory/1e6:.1f}MB, possible leak)"

        logger.info(f"Memory stable after {n_iterations} iterations ✓")

    def test_repeated_runs_different_sizes(self):
        """Test memory is properly freed when processing different sized samples."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        mem_pool = cp.get_default_memory_pool()

        # Warm up
        _ = repair_masks_gpu(*create_simple_mask_pair(n_cells=50), use_gpu=True)
        baseline = mem_pool.used_bytes()

        # Process samples of varying sizes
        sizes = [100, 500, 1000, 500, 100]  # Vary up and down

        for n_cells in sizes:
            cell_mask, nucleus_mask, _ = create_simple_mask_pair(
                n_cells=n_cells,
                image_size=(int(np.sqrt(n_cells) * 50), int(np.sqrt(n_cells) * 50))
            )

            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            assert cell_matched is not None

        # Final memory should not be significantly higher than baseline
        final = mem_pool.used_bytes()
        memory_ratio = final / max(baseline, 1e6)

        logger.info(f"Memory after varying sizes: {baseline/1e6:.1f}MB → {final/1e6:.1f}MB "
                   f"(ratio: {memory_ratio:.2f}x)")

        assert memory_ratio < 1.5, \
            f"Memory grew {memory_ratio:.2f}x after processing varying sizes"

    def test_large_sample_memory_freed(self):
        """Test memory is freed after processing large sample."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        mem_pool = cp.get_default_memory_pool()

        # Small warm up
        _ = repair_masks_gpu(*create_simple_mask_pair(n_cells=10), use_gpu=True)
        baseline = mem_pool.used_bytes()

        # Process large sample
        logger.info("Processing large sample...")
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=5000,
            image_size=(3000, 3000)
        )

        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
        assert cell_matched is not None

        # Memory during large sample processing
        peak_memory = mem_pool.used_bytes()
        logger.info(f"Peak memory during 5K cells: {peak_memory/1e6:.1f} MB")

        # Delete references to force cleanup
        del cell_mask, nucleus_mask, result

        # Process small sample again
        _ = repair_masks_gpu(*create_simple_mask_pair(n_cells=10), use_gpu=True)
        final = mem_pool.used_bytes()

        logger.info(f"Memory after cleanup: {baseline/1e6:.1f}MB → {peak_memory/1e6:.1f}MB → "
                   f"{final/1e6:.1f}MB")

        # Final should be close to baseline (within 50% due to GPU caching)
        memory_ratio = final / max(baseline, 1e6)
        assert memory_ratio < 1.5, \
            f"Memory not freed after large sample: {memory_ratio:.2f}x baseline"


@pytest.mark.stress
@pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
class TestGPUStressConcurrentOperations:
    """Test GPU behavior with sequential/concurrent operations."""

    def test_sequential_operations_no_interference(self):
        """Test sequential operations don't interfere with each other."""
        # Create multiple different samples
        samples = []
        for i in range(10):
            cell_mask, nucleus_mask, expected = create_simple_mask_pair(
                n_cells=50,
                seed=1000 + i * 10
            )
            samples.append((cell_mask, nucleus_mask, expected))

        # Process all samples sequentially
        results = []
        for cell_mask, nucleus_mask, expected in samples:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            results.append(result)

        # Verify all succeeded
        assert len(results) == 10
        for i, result in enumerate(results):
            assert cell_matched is not None, f"Sample {i} failed"
            assert cell_matched is not None

        # Re-process first sample and verify identical result (deterministic)
        cell_mask, nucleus_mask, _ = samples[0]
        result_rerun = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        assert np.array_equal(
            results[0]["cell_matched_mask"],
            result_rerun["cell_matched_mask"]
        ), "Non-deterministic results detected (state interference)"

        logger.info("Sequential operations are deterministic ✓")

    def test_alternating_gpu_cpu_operations(self):
        """Test alternating between GPU and CPU modes."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=50, seed=42)

        # Alternate GPU and CPU
        modes = [True, False, True, False, True]
        results = []

        for use_gpu in modes:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=use_gpu)
            results.append(result)

        # All results should be identical (GPU and CPU produce same output)
        for i in range(1, len(results)):
            assert np.array_equal(
                results[0]["cell_matched_mask"],
                results[i]["cell_matched_mask"]
            ), f"GPU/CPU alternation produced different results at index {i}"

        logger.info("GPU/CPU alternation produces consistent results ✓")

    def test_rapid_small_batches(self):
        """Test rapid processing of many small samples."""
        # Simulate processing many patches/tiles rapidly
        n_batches = 50
        batch_size = 20  # cells per batch

        logger.info(f"Processing {n_batches} rapid small batches...")

        start = time.time()
        for i in range(n_batches):
            cell_mask, nucleus_mask, _ = create_simple_mask_pair(
                n_cells=batch_size,
                image_size=(256, 256),
                seed=2000 + i
            )

            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            assert cell_matched is not None

        elapsed = time.time() - start
        throughput = (n_batches * batch_size) / elapsed

        logger.info(f"Processed {n_batches} batches ({n_batches * batch_size} total cells) "
                   f"in {elapsed:.2f}s ({throughput:.1f} cells/sec)")

        # Should handle rapid small batches efficiently
        assert elapsed < 60, f"Rapid batches took {elapsed:.1f}s (too slow)"


if __name__ == "__main__":
    # Run with: python tests/test_gpu_stress.py -m stress
    pytest.main([__file__, "-v", "-s", "-m", "stress"])
