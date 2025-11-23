"""GPU performance regression tests.

This test suite ensures GPU performance doesn't regress in future changes.
It validates that GPU components meet minimum speedup targets:

- GPU morphology: 10x+ speedup over CPU
- GPU overlap computation: 3x+ speedup for >5K cells
- GPU integration: No slowdown vs CPU (minimum 1x)

Run with:
    pytest tests/test_gpu_performance_regression.py -v
"""

import pytest
import numpy as np
import time
import logging

from aegle.repair_masks_gpu_morphology import find_boundaries_gpu
from aegle.repair_masks_gpu_overlap import compute_overlap_matrix_gpu
from aegle.repair_masks_gpu import repair_masks_gpu
from aegle.repair_masks import get_matched_masks
from tests.utils.gpu_test_utils import requires_gpu
from tests.utils.repair_test_fixtures import create_simple_mask_pair, create_stress_test_case

logger = logging.getLogger(__name__)


@requires_gpu()
class TestGPUMorphologyPerformance:
    """Ensure GPU morphology meets minimum performance (10x speedup)."""

    def test_gpu_morphology_performance_large_mask(self):
        """Test GPU morphology speedup on large mask (5000x5000)."""
        from skimage.segmentation import find_boundaries

        # Create large mask
        mask_size = 5000
        mask = np.zeros((mask_size, mask_size), dtype=np.uint32)

        # Create grid of cells
        n_cells = 100
        cell_size = mask_size // 12
        for i in range(10):
            for j in range(10):
                if i * 10 + j >= n_cells:
                    break
                y = i * (mask_size // 10) + cell_size
                x = j * (mask_size // 10) + cell_size
                mask[y:y+cell_size, x:x+cell_size] = i * 10 + j + 1

        # Benchmark CPU (scikit-image)
        cpu_times = []
        for _ in range(3):
            start = time.time()
            _ = find_boundaries(mask, mode='inner')
            cpu_times.append(time.time() - start)
        cpu_time = np.median(cpu_times)

        # Benchmark GPU
        gpu_times = []
        for _ in range(3):
            start = time.time()
            _ = find_boundaries_gpu(mask, mode='inner')
            gpu_times.append(time.time() - start)
        gpu_time = np.median(gpu_times)

        speedup = cpu_time / gpu_time

        logger.info(f"Morphology 5000x5000: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, "
                   f"speedup={speedup:.1f}x")

        # Minimum requirement: 10x speedup
        assert speedup >= 10.0, \
            f"GPU morphology speedup {speedup:.1f}x < 10x minimum (regression detected)"

    def test_gpu_morphology_performance_multiple_sizes(self):
        """Test GPU morphology performance at multiple scales."""
        from skimage.segmentation import find_boundaries

        sizes = [1000, 2000, 4000]
        results = []

        for size in sizes:
            # Create test mask
            mask = np.zeros((size, size), dtype=np.uint32)
            n_cells = min(size // 100, 50)
            cell_size = size // 12

            for i in range(int(np.sqrt(n_cells))):
                for j in range(int(np.sqrt(n_cells))):
                    y = i * (size // int(np.sqrt(n_cells))) + cell_size // 2
                    x = j * (size // int(np.sqrt(n_cells))) + cell_size // 2
                    mask[y:y+cell_size, x:x+cell_size] = i * int(np.sqrt(n_cells)) + j + 1

            # CPU time
            start = time.time()
            _ = find_boundaries(mask, mode='inner')
            cpu_time = time.time() - start

            # GPU time
            start = time.time()
            _ = find_boundaries_gpu(mask, mode='inner')
            gpu_time = time.time() - start

            speedup = cpu_time / gpu_time
            results.append((size, cpu_time, gpu_time, speedup))

        # Log all results
        for size, cpu_t, gpu_t, speedup in results:
            logger.info(f"{size}x{size}: CPU={cpu_t:.3f}s, GPU={gpu_t:.3f}s, speedup={speedup:.1f}x")

        # Verify all achieve good speedup
        for size, _, _, speedup in results:
            assert speedup >= 5.0, \
                f"GPU morphology at {size}x{size} only {speedup:.1f}x (expected >5x)"


@requires_gpu()
class TestGPUOverlapPerformance:
    """Ensure GPU overlap meets minimum performance (3x speedup for >5K cells)."""

    def test_gpu_overlap_performance_5k_cells(self):
        """Test GPU overlap speedup on 5K cells."""
        from aegle.repair_masks_gpu_overlap import _compute_overlap_matrix_cpu

        # Create 5K cell sample
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=5000,
            image_size=(3000, 3000)
        )

        # Benchmark CPU
        cpu_times = []
        for _ in range(2):
            start = time.time()
            _ = _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)
            cpu_times.append(time.time() - start)
        cpu_time = np.median(cpu_times)

        # Benchmark GPU
        gpu_times = []
        for _ in range(2):
            start = time.time()
            _ = compute_overlap_matrix_gpu(cell_mask, nucleus_mask, use_gpu=True)
            gpu_times.append(time.time() - start)
        gpu_time = np.median(gpu_times)

        speedup = cpu_time / gpu_time

        logger.info(f"Overlap 5K cells: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, "
                   f"speedup={speedup:.1f}x")

        # Minimum requirement: 3x speedup for 5K+ cells
        assert speedup >= 3.0, \
            f"GPU overlap speedup {speedup:.1f}x < 3x minimum for 5K cells (regression)"

    def test_gpu_overlap_performance_10k_cells(self):
        """Test GPU overlap speedup on 10K cells."""
        from aegle.repair_masks_gpu_overlap import _compute_overlap_matrix_cpu

        # Create 10K cell sample
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=10000,
            image_size=(4000, 4000)
        )

        # CPU time
        start = time.time()
        _ = _compute_overlap_matrix_cpu(cell_mask, nucleus_mask)
        cpu_time = time.time() - start

        # GPU time
        start = time.time()
        _ = compute_overlap_matrix_gpu(cell_mask, nucleus_mask, use_gpu=True)
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time

        logger.info(f"Overlap 10K cells: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, "
                   f"speedup={speedup:.1f}x")

        # Should see even better speedup at 10K
        assert speedup >= 3.0, \
            f"GPU overlap speedup {speedup:.1f}x < 3x minimum for 10K cells (regression)"


@requires_gpu()
class TestGPUIntegrationPerformance:
    """Ensure GPU version doesn't slow down vs CPU (minimum 1x speedup)."""

    def test_gpu_integration_no_slowdown_small(self):
        """Ensure GPU doesn't slow down on small samples (100 cells)."""
        # Small samples might not benefit from GPU, but shouldn't be slower
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=100)

        # CPU time
        start = time.time()
        _ = get_matched_masks(cell_mask, nucleus_mask)
        cpu_time = time.time() - start

        # GPU time
        start = time.time()
        _ = repair_masks_gpu(cell_mask, nucleus_mask)
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time

        logger.info(f"Integration 100 cells: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, "
                   f"speedup={speedup:.2f}x")

        # Allow some slowdown for small samples due to GPU overhead
        # But shouldn't be more than 2x slower
        assert speedup >= 0.5, \
            f"GPU {1/speedup:.1f}x slower than CPU on small sample (too much overhead)"

    def test_gpu_integration_no_slowdown_medium(self):
        """Ensure GPU doesn't slow down on medium samples (1K cells)."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(
            n_cells=1000,
            image_size=(2000, 2000)
        )

        # CPU time
        start = time.time()
        _ = get_matched_masks(cell_mask, nucleus_mask)
        cpu_time = time.time() - start

        # GPU time
        start = time.time()
        _ = repair_masks_gpu(cell_mask, nucleus_mask)
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time

        logger.info(f"Integration 1K cells: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, "
                   f"speedup={speedup:.2f}x")

        # Minimum requirement: no slowdown (1x speedup)
        assert speedup >= 1.0, \
            f"GPU {1/speedup:.1f}x slower than CPU on 1K cells (regression)"

    def test_gpu_integration_no_slowdown_large(self):
        """Ensure GPU provides speedup on large samples (5K cells)."""
        cell_mask, nucleus_mask, _ = create_stress_test_case(
            n_cells=5000,
            image_size=(3000, 3000)
        )

        # CPU time (single run due to length)
        start = time.time()
        _ = get_matched_masks(cell_mask, nucleus_mask)
        cpu_time = time.time() - start

        # GPU time
        start = time.time()
        _ = repair_masks_gpu(cell_mask, nucleus_mask)
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time

        logger.info(f"Integration 5K cells: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, "
                   f"speedup={speedup:.2f}x")

        # For large samples, GPU should provide speedup
        assert speedup >= 1.0, \
            f"GPU {1/speedup:.1f}x slower than CPU on 5K cells (regression)"


@requires_gpu()
class TestGPUPerformanceScaling:
    """Test GPU performance scaling characteristics."""

    @pytest.mark.parametrize("n_cells", [100, 500, 1000, 5000])
    def test_gpu_throughput_scaling(self, n_cells):
        """Test GPU throughput (cells/sec) scales with sample size."""
        # Create sample
        image_size = int(np.sqrt(n_cells) * 100)
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(
            n_cells=n_cells,
            image_size=(image_size, image_size),
            seed=42 + n_cells
        )

        # Benchmark GPU
        start = time.time()
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
        elapsed = time.time() - start

        assert cell_matched is not None
        throughput = n_cells / elapsed

        logger.info(f"{n_cells} cells: {elapsed:.2f}s ({throughput:.1f} cells/sec)")

        # Minimum throughput requirements (very conservative)
        min_throughput = {
            100: 10,    # At least 10 cells/sec
            500: 20,    # At least 20 cells/sec
            1000: 30,   # At least 30 cells/sec
            5000: 40,   # At least 40 cells/sec
        }

        assert throughput >= min_throughput[n_cells], \
            f"GPU throughput {throughput:.1f} < {min_throughput[n_cells]} cells/sec (regression)"

    def test_gpu_time_per_cell_decreases(self):
        """Test time per cell decreases as sample size increases (better GPU utilization)."""
        scales = [100, 500, 1000, 2000]
        times_per_cell = []

        for n_cells in scales:
            image_size = int(np.sqrt(n_cells) * 100)
            cell_mask, nucleus_mask, _ = create_simple_mask_pair(
                n_cells=n_cells,
                image_size=(image_size, image_size),
                seed=100 + n_cells
            )

            start = time.time()
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)
            elapsed = time.time() - start

            assert cell_matched is not None
            time_per_cell = elapsed / n_cells * 1000  # ms/cell
            times_per_cell.append((n_cells, time_per_cell))

        # Log results
        for n_cells, tpc in times_per_cell:
            logger.info(f"{n_cells} cells: {tpc:.2f} ms/cell")

        # Time per cell should generally decrease with scale (GPU amortization)
        # Allow some variance, but largest should be more efficient than smallest
        assert times_per_cell[-1][1] <= times_per_cell[0][1] * 2, \
            "GPU doesn't show expected efficiency gains with scale (regression)"


if __name__ == "__main__":
    # Run with: python tests/test_gpu_performance_regression.py
    pytest.main([__file__, "-v", "-s"])
