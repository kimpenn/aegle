"""GPU availability and fallback tests.

This test suite validates behavior when GPU is/isn't available:
- GPU detection works correctly
- CPU fallback when GPU explicitly disabled
- CPU fallback when CuPy not installed or GPU unavailable
- Error messages are clear and actionable

Run with:
    pytest tests/test_gpu_availability.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch
import logging

from aegle.gpu_utils import is_cupy_available, get_gpu_memory_info
from aegle.repair_masks_gpu import repair_masks_gpu, get_matched_masks_gpu
from tests.utils.repair_test_fixtures import create_simple_mask_pair

logger = logging.getLogger(__name__)


class TestGPUDetection:
    """Test GPU detection works correctly."""

    def test_detect_gpu_available(self):
        """Test GPU detection returns boolean."""
        gpu_available = is_cupy_available()

        # Should return boolean
        assert isinstance(gpu_available, bool)

        # Log for debugging
        logger.info(f"GPU available: {gpu_available}")

        # If GPU is available, should be able to get memory info
        if gpu_available:
            mem_info = get_gpu_memory_info()
            assert isinstance(mem_info, dict)
            assert "total_gb" in mem_info
            assert "free_gb" in mem_info
            assert mem_info["total_gb"] > 0
            logger.info(f"GPU memory: {mem_info['total_gb']:.1f} GB total, "
                       f"{mem_info['free_gb']:.1f} GB free")

    def test_cupy_import_handling(self):
        """Test that CuPy import is handled gracefully."""
        # Try importing CuPy
        try:
            import cupy as cp
            cupy_installed = True
            logger.info("CuPy is installed")
        except ImportError:
            cupy_installed = False
            logger.info("CuPy is not installed")

        # is_cupy_available() should match actual import
        gpu_available = is_cupy_available()

        # If CuPy not installed, GPU should not be available
        if not cupy_installed:
            assert not gpu_available, \
                "GPU reported as available but CuPy not installed"

    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy not available")
    def test_gpu_memory_info_format(self):
        """Test GPU memory info has correct format."""
        mem_info = get_gpu_memory_info()

        # Check required fields
        required_fields = ["device_id", "total_gb", "free_gb", "used_gb"]
        for field in required_fields:
            assert field in mem_info, f"Missing field: {field}"

        # Check values are reasonable
        assert mem_info["total_gb"] > 0, "Total GPU memory should be positive"
        assert mem_info["free_gb"] >= 0, "Free GPU memory should be non-negative"
        assert mem_info["used_gb"] >= 0, "Used GPU memory should be non-negative"
        assert mem_info["total_gb"] >= mem_info["used_gb"], \
            "Total should be >= used"


class TestCPUFallback:
    """Test CPU fallback when GPU explicitly disabled."""

    def test_cpu_fallback_when_gpu_disabled(self):
        """Test CPU fallback when use_gpu=False."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Explicitly disable GPU
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=False)

        # Should succeed
        assert cell_matched is not None
        assert cell_matched is not None
        assert metadata is not None

        # Metadata should indicate CPU was used
        assert metadata["used_gpu"] is False

        logger.info("CPU fallback works correctly when GPU disabled")

    def test_wrapper_respects_use_gpu_flag(self):
        """Test get_matched_masks_gpu wrapper respects use_gpu flag."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Test with GPU disabled
        result_cpu = get_matched_masks_gpu(
            cell_mask,
            nucleus_mask,
            use_gpu=False
        )

        assert result_cpu is not None
        assert metadata_cpu["used_gpu"] is False

        # If GPU available, test with it enabled
        if is_cupy_available():
            result_gpu = get_matched_masks_gpu(
                cell_mask,
                nucleus_mask,
                use_gpu=True
            )

            assert cell_matched is not None
            assert metadata["used_gpu"] is True

    def test_cpu_fallback_produces_valid_results(self):
        """Test CPU fallback produces valid results (not just empty)."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=20)

        # Run with GPU disabled
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=False)

        # Verify result is valid
        assert cell_matched is not None
        assert cell_matched is not None
        assert nucleus_matched is not None

        # Check cells were actually matched
        n_matched = len(np.unique(cell_matched)) - 1
        assert n_matched > 0, "CPU fallback should produce matches"
        assert n_matched == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matches, got {n_matched}"


class TestCPUFallbackWhenGPUUnavailable:
    """Test CPU fallback when CuPy not installed or GPU unavailable."""

    def test_graceful_handling_when_cupy_missing(self):
        """Test graceful handling when CuPy is not installed."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Mock CuPy as unavailable
        with patch("aegle.gpu_utils.is_cupy_available", return_value=False):
            # Try to use GPU
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

            # Should fall back to CPU gracefully
            assert cell_matched is not None
            assert cell_matched is not None

            # Metadata should show CPU was used (fallback)
            assert metadata["used_gpu"] is False

    def test_error_messages_when_gpu_requested_but_unavailable(self):
        """Test clear error messages when GPU requested but unavailable."""
        # This test verifies that error messages are logged (not that exceptions are raised)
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        with patch("aegle.gpu_utils.is_cupy_available", return_value=False):
            with patch("logging.warning") as mock_warning:
                cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

                # Should still succeed via CPU fallback
                assert cell_matched is not None

                # Should have logged a warning about GPU unavailable
                # (Check that warning was called at least once)
                assert mock_warning.call_count > 0, \
                    "Should log warning when GPU requested but unavailable"

    @pytest.mark.skipif(is_cupy_available(), reason="GPU is available")
    def test_actual_cupy_unavailable_handling(self):
        """Test actual behavior when CuPy is not installed (only runs if GPU unavailable)."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Request GPU (should fall back to CPU)
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        # Should succeed via CPU fallback
        assert cell_matched is not None
        assert cell_matched is not None

        # Metadata should show CPU was used
        assert metadata["used_gpu"] is False


class TestGPUErrorHandling:
    """Test error handling for GPU operations."""

    @pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
    def test_invalid_input_error_handling(self):
        """Test GPU handles invalid inputs gracefully."""
        # Test with completely empty masks
        empty_mask = np.array([], dtype=np.uint32).reshape(0, 0)

        # Should handle gracefully (either succeed with empty result or raise clear error)
        try:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(empty_mask, empty_mask, use_gpu=True)
            # If it succeeds, verify result is valid
            assert cell_matched is not None
        except (ValueError, RuntimeError) as e:
            # If it raises error, verify message is descriptive
            error_msg = str(e).lower()
            assert len(error_msg) > 10, "Error message should be descriptive"
            assert any(word in error_msg for word in ["empty", "invalid", "shape"]), \
                f"Error message should mention the problem: {e}"

    @pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
    def test_mismatched_shapes_error_handling(self):
        """Test GPU handles mismatched mask shapes gracefully."""
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((50, 50), dtype=np.uint32)  # Different shape

        # Should handle gracefully
        try:
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            # If it succeeds (e.g., by resizing), verify result is valid
            assert cell_matched is not None
        except (ValueError, AssertionError) as e:
            # If it raises error, verify message mentions shape mismatch
            error_msg = str(e).lower()
            assert "shape" in error_msg or "size" in error_msg, \
                f"Error should mention shape mismatch: {e}"

    @pytest.mark.skipif(not is_cupy_available(), reason="GPU not available")
    def test_gpu_error_falls_back_to_cpu(self):
        """Test that GPU errors trigger CPU fallback."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Mock a GPU function to raise an error
        with patch("aegle.repair_masks_gpu._repair_masks_gpu_impl", side_effect=RuntimeError("Mock GPU error")):
            # Should fall back to CPU
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

            # Should still succeed
            assert cell_matched is not None
            assert cell_matched is not None

            # Metadata should show CPU was used (fallback)
            assert metadata["used_gpu"] is False


class TestConfigurationDrivenGPU:
    """Test GPU enable/disable via configuration."""

    def test_default_use_gpu_false(self):
        """Test default use_gpu=False (conservative default)."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Call without specifying use_gpu (should default to False for safety)
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask)

        assert cell_matched is not None
        # Default should be CPU (conservative)
        assert metadata["used_gpu"] is False

    def test_explicit_gpu_enable(self):
        """Test explicit GPU enable via use_gpu=True."""
        if not is_cupy_available():
            pytest.skip("GPU not available")

        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Explicitly enable GPU
        cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)

        assert cell_matched is not None
        assert metadata["used_gpu"] is True

    def test_use_gpu_flag_overrides(self):
        """Test use_gpu flag correctly overrides defaults."""
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        # Test CPU mode
        result_cpu = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=False)
        assert metadata_cpu["used_gpu"] is False

        # Test GPU mode (if available)
        if is_cupy_available():
            cell_matched, nucleus_matched, metadata = repair_masks_gpu(cell_mask, nucleus_mask, use_gpu=True)
            assert metadata["used_gpu"] is True

            # Results should be identical
            assert np.array_equal(
                result_cpu["cell_matched_mask"],
                cell_matched
            ), "CPU and GPU should produce identical results"


if __name__ == "__main__":
    # Run with: python tests/test_gpu_availability.py
    pytest.main([__file__, "-v", "-s"])
