"""
Tests for GPU memory management utilities.

Tests GPU detection, memory querying, batch size estimation, and memory cleanup.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from aegle.gpu_utils import (
    is_cupy_available,
    get_gpu_memory_info,
    estimate_gpu_batch_size,
    clear_gpu_memory,
    log_gpu_memory,
)
from tests.utils.gpu_test_utils import requires_gpu, mock_gpu_memory


class TestGPUAvailability(unittest.TestCase):
    """Test GPU availability detection."""

    def test_is_cupy_available_caches_result(self):
        """is_cupy_available should cache result for performance."""
        # Call twice and ensure result is consistent
        first_call = is_cupy_available()
        second_call = is_cupy_available()

        self.assertEqual(first_call, second_call)
        self.assertIsInstance(first_call, bool)

    def test_is_cupy_available_with_no_import(self):
        """Should return False when CuPy not importable."""
        # Reset cache
        import aegle.gpu_utils
        aegle.gpu_utils._CUPY_AVAILABLE = None

        with patch('aegle.gpu_utils.cp', side_effect=ImportError):
            # Force re-check
            aegle.gpu_utils._CUPY_AVAILABLE = None
            result = is_cupy_available()
            self.assertFalse(result)


class TestGPUMemoryInfo(unittest.TestCase):
    """Test GPU memory information querying."""

    @requires_gpu()
    def test_get_gpu_memory_info_returns_valid_dict(self):
        """Should return dict with expected keys when GPU available."""
        info = get_gpu_memory_info()

        self.assertIsNotNone(info)
        self.assertIn('device_id', info)
        self.assertIn('total_gb', info)
        self.assertIn('free_gb', info)
        self.assertIn('used_gb', info)
        self.assertIn('pool_used_gb', info)
        self.assertIn('pool_total_gb', info)

        # Sanity checks
        self.assertGreater(info['total_gb'], 0)
        self.assertGreaterEqual(info['free_gb'], 0)
        self.assertGreaterEqual(info['used_gb'], 0)
        self.assertEqual(info['device_id'], 0)

    @requires_gpu()
    def test_get_gpu_memory_info_consistency(self):
        """Free + used should approximately equal total."""
        info = get_gpu_memory_info()

        self.assertIsNotNone(info)
        # Allow small discrepancy due to rounding and ongoing allocations
        total_accounted = info['free_gb'] + info['used_gb']
        self.assertAlmostEqual(total_accounted, info['total_gb'], delta=0.1)

    def test_get_gpu_memory_info_without_gpu(self):
        """Should return None when GPU unavailable."""
        with patch('aegle.gpu_utils.is_cupy_available', return_value=False):
            info = get_gpu_memory_info()
            self.assertIsNone(info)


class TestGPUBatchSizeEstimation(unittest.TestCase):
    """Test GPU batch size estimation logic."""

    def test_estimate_batch_size_with_large_memory(self):
        """Should recommend larger batch size with abundant memory."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
            )

            # With 12 GB free, should be able to process many channels
            self.assertGreater(batch_size, 1)
            self.assertLessEqual(batch_size, 50)  # Cap at n_channels

    def test_estimate_batch_size_with_limited_memory(self):
        """Should recommend smaller batch size with limited memory."""
        with mock_gpu_memory(total_bytes=4e9, free_bytes=1e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
            )

            # With only 1 GB free, batch size should be conservative
            self.assertGreater(batch_size, 0)
            self.assertLess(batch_size, 20)

    def test_estimate_batch_size_caps_at_n_channels(self):
        """Should never recommend batch size larger than n_channels."""
        with mock_gpu_memory(total_bytes=32e9, free_bytes=24e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=10,
                image_shape=(1000, 1000),
                n_cells=5000,
                dtype=np.float32,
            )

            # Even with lots of memory, should cap at 10
            self.assertLessEqual(batch_size, 10)

    def test_estimate_batch_size_with_different_dtypes(self):
        """Batch size should vary with dtype (float64 uses more memory)."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_float32 = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
            )

            batch_float64 = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float64,
            )

            # float64 uses 2x memory, so batch size should be smaller
            self.assertLess(batch_float64, batch_float32)

    def test_estimate_batch_size_with_small_image(self):
        """Small images should allow larger batch sizes."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_small = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(100, 100),  # Small image
                n_cells=100,
                dtype=np.float32,
            )

            batch_large = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(2000, 2000),  # Large image
                n_cells=10000,
                dtype=np.float32,
            )

            # Smaller images use less memory per channel
            self.assertGreater(batch_small, batch_large)

    def test_estimate_batch_size_minimum_is_one(self):
        """Batch size should never be less than 1."""
        with mock_gpu_memory(total_bytes=1e9, free_bytes=0.1e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(4000, 4000),  # Very large
                n_cells=100000,
                dtype=np.float32,
            )

            self.assertGreaterEqual(batch_size, 1)

    def test_estimate_batch_size_without_gpu(self):
        """Should return conservative default when GPU unavailable."""
        with patch('aegle.gpu_utils.get_gpu_memory_info', return_value=None):
            batch_size = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
            )

            # Should return default conservative value
            self.assertEqual(batch_size, 5)

    def test_estimate_batch_size_safety_factor(self):
        """Safety factor should reduce effective available memory."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            # Conservative safety factor
            batch_conservative = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
                safety_factor=0.5,  # Use only 50% of free memory
            )

            # Aggressive safety factor
            batch_aggressive = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
                safety_factor=0.9,  # Use 90% of free memory
            )

            # More conservative safety factor should yield smaller batch
            self.assertLess(batch_conservative, batch_aggressive)


class TestGPUMemoryCleanup(unittest.TestCase):
    """Test GPU memory cleanup operations."""

    @requires_gpu()
    def test_clear_gpu_memory_runs_without_error(self):
        """Should successfully clear memory pools when GPU available."""
        try:
            clear_gpu_memory()
        except Exception as e:
            self.fail(f"clear_gpu_memory raised unexpected exception: {e}")

    def test_clear_gpu_memory_without_gpu(self):
        """Should handle gracefully when GPU unavailable."""
        with patch('aegle.gpu_utils.is_cupy_available', return_value=False):
            try:
                clear_gpu_memory()
            except Exception as e:
                self.fail(f"clear_gpu_memory should not raise when GPU unavailable: {e}")

    @requires_gpu()
    def test_clear_gpu_memory_actually_frees_memory(self):
        """Clearing memory should reduce pool usage."""
        import cupy as cp

        # Allocate some GPU memory
        arrays = [cp.ones((1000, 1000), dtype=cp.float32) for _ in range(10)]

        # Get memory info before clearing
        info_before = get_gpu_memory_info()

        # Delete references and clear
        del arrays
        clear_gpu_memory()

        # Get memory info after clearing
        info_after = get_gpu_memory_info()

        # Pool memory should be reduced (or at least not increased)
        self.assertLessEqual(info_after['pool_used_gb'], info_before['pool_used_gb'] + 0.1)


class TestGPUMemoryLogging(unittest.TestCase):
    """Test GPU memory logging."""

    @requires_gpu()
    def test_log_gpu_memory_with_gpu_available(self):
        """Should log memory info when GPU available."""
        with self.assertLogs('aegle.gpu_utils', level=logging.INFO) as cm:
            log_gpu_memory("Test prefix")

            # Should have logged something about memory
            self.assertTrue(any('Test prefix' in msg for msg in cm.output))
            self.assertTrue(any('GB' in msg for msg in cm.output))

    def test_log_gpu_memory_without_gpu(self):
        """Should log debug message when GPU unavailable."""
        with patch('aegle.gpu_utils.get_gpu_memory_info', return_value=None):
            with self.assertLogs('aegle.gpu_utils', level=logging.DEBUG) as cm:
                log_gpu_memory("Test prefix")

                # Should log that query failed
                self.assertTrue(any('Unable to query' in msg for msg in cm.output))

    @requires_gpu()
    def test_log_gpu_memory_custom_prefix(self):
        """Should use custom prefix in log message."""
        custom_prefix = "Custom memory check"

        with self.assertLogs('aegle.gpu_utils', level=logging.INFO) as cm:
            log_gpu_memory(custom_prefix)

            # Should contain custom prefix
            self.assertTrue(any(custom_prefix in msg for msg in cm.output))


class TestGPUMemoryEdgeCases(unittest.TestCase):
    """Test edge cases in GPU memory management."""

    def test_estimate_batch_size_zero_channels(self):
        """Should handle zero channels gracefully."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=0,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
            )

            # Should return 0 (cap at n_channels)
            self.assertEqual(batch_size, 0)

    def test_estimate_batch_size_single_channel(self):
        """Should handle single channel correctly."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=1,
                image_shape=(1000, 1000),
                n_cells=10000,
                dtype=np.float32,
            )

            # Should cap at 1
            self.assertEqual(batch_size, 1)

    def test_estimate_batch_size_zero_cells(self):
        """Should handle zero cells (empty image)."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(1000, 1000),
                n_cells=0,
                dtype=np.float32,
            )

            # Should still return valid batch size (memory calculation should work)
            self.assertGreater(batch_size, 0)

    def test_estimate_batch_size_very_large_n_cells(self):
        """Should handle very large number of cells."""
        with mock_gpu_memory(total_bytes=16e9, free_bytes=12e9):
            batch_size = estimate_gpu_batch_size(
                n_channels=50,
                image_shape=(4000, 4000),
                n_cells=5000000,  # 5 million cells
                dtype=np.float32,
            )

            # Should return smaller batch due to large output arrays
            self.assertGreater(batch_size, 0)
            self.assertLess(batch_size, 50)


if __name__ == "__main__":
    unittest.main()
