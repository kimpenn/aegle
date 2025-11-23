"""Tests for GPU-accelerated morphological operations in mask repair.

This test suite verifies:
1. GPU vs CPU equivalence (max difference < 1e-6)
2. Various mask sizes and configurations
3. OOM handling with artificially large masks
4. CPU fallback when GPU unavailable
"""

import pytest
import numpy as np
from unittest.mock import patch

from aegle.repair_masks_gpu_morphology import (
    find_boundaries_gpu,
    compute_labeled_boundary_gpu,
    _find_boundaries_cpu,
)
from aegle.gpu_utils import is_cupy_available, clear_gpu_memory
from tests.utils.gpu_test_utils import requires_gpu


# Test fixtures for various mask configurations

def create_simple_square_mask(size=100, label=1):
    """Create a simple square mask for basic testing."""
    mask = np.zeros((size, size), dtype=np.uint32)
    margin = size // 4
    mask[margin:size-margin, margin:size-margin] = label
    return mask


def create_multi_object_mask(size=100, num_objects=5):
    """Create a mask with multiple objects for testing label preservation."""
    mask = np.zeros((size, size), dtype=np.uint32)
    object_size = size // (num_objects + 1)

    for i in range(num_objects):
        x_start = (i + 1) * object_size - object_size // 2
        x_end = x_start + object_size
        y_start = object_size
        y_end = y_start + object_size
        mask[y_start:y_end, x_start:x_end] = i + 1

    return mask


def create_nested_objects_mask(size=100):
    """Create a mask with nested objects (concentric squares)."""
    mask = np.zeros((size, size), dtype=np.uint32)
    for i, margin in enumerate([10, 25, 40]):
        mask[margin:size-margin, margin:size-margin] = i + 1
    return mask


def create_touching_objects_mask(size=100):
    """Create a mask with objects that touch each other."""
    mask = np.zeros((size, size), dtype=np.uint32)
    half = size // 2
    # Left object
    mask[:, :half] = 1
    # Right object
    mask[:, half:] = 2
    return mask


# Basic functionality tests

class TestFindBoundariesGPUBasic:
    """Basic tests for find_boundaries_gpu function."""

    def test_simple_square_inner_boundary_cpu(self):
        """Test inner boundary detection on CPU for a simple square."""
        mask = create_simple_square_mask(size=100, label=1)
        boundaries = _find_boundaries_cpu(mask, mode='inner')

        # Verify output shape
        assert boundaries.shape == mask.shape

        # Verify output type
        assert boundaries.dtype == bool

        # Verify boundaries exist
        assert boundaries.any(), "No boundaries detected"

        # Verify boundaries are only on object edges
        # Inner boundary should be inside the object
        assert boundaries[mask == 0].sum() == 0, "Boundaries detected outside object"

    def test_simple_square_inner_boundary_gpu_vs_cpu(self):
        """Test that GPU and CPU produce identical results for inner boundaries."""
        mask = create_simple_square_mask(size=100, label=1)

        # CPU version
        boundaries_cpu = _find_boundaries_cpu(mask, mode='inner')

        # GPU version (may fall back to CPU if GPU unavailable)
        boundaries_gpu = find_boundaries_gpu(mask, mode='inner')

        # Should be identical
        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            "GPU and CPU boundary detection differ"

    @requires_gpu()
    def test_multi_object_boundary_preservation(self):
        """Test that boundaries preserve original labels."""
        mask = create_multi_object_mask(size=200, num_objects=5)

        # Compute labeled boundaries
        boundary_mask = compute_labeled_boundary_gpu(mask)

        # Verify output shape
        assert boundary_mask.shape == mask.shape

        # Verify all boundary labels are present in original mask
        boundary_labels = np.unique(boundary_mask[boundary_mask > 0])
        mask_labels = np.unique(mask[mask > 0])
        assert set(boundary_labels).issubset(set(mask_labels)), \
            "Boundary contains labels not in original mask"


# GPU vs CPU equivalence tests

@requires_gpu()
class TestGPUCPUEquivalence:
    """Test that GPU results exactly match CPU for all modes and configurations."""

    @pytest.mark.parametrize("mode", ["inner", "outer", "thick"])
    def test_boundary_modes_equivalence(self, mode):
        """Test GPU vs CPU equivalence for all boundary modes."""
        mask = create_simple_square_mask(size=150, label=1)

        # CPU version
        boundaries_cpu = _find_boundaries_cpu(mask, mode=mode)

        # GPU version
        boundaries_gpu = find_boundaries_gpu(mask, mode=mode)

        # Should be exactly equal
        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            f"GPU and CPU differ for mode '{mode}'"

    @pytest.mark.parametrize("size", [50, 100, 500, 1000])
    def test_various_mask_sizes(self, size):
        """Test GPU vs CPU equivalence for various mask sizes."""
        mask = create_simple_square_mask(size=size, label=1)

        boundaries_cpu = _find_boundaries_cpu(mask, mode='inner')
        boundaries_gpu = find_boundaries_gpu(mask, mode='inner')

        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            f"GPU and CPU differ for size {size}x{size}"

    def test_multi_object_equivalence(self):
        """Test GPU vs CPU equivalence for masks with multiple objects."""
        mask = create_multi_object_mask(size=200, num_objects=10)

        boundaries_cpu = _find_boundaries_cpu(mask, mode='inner')
        boundaries_gpu = find_boundaries_gpu(mask, mode='inner')

        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            "GPU and CPU differ for multi-object mask"

    def test_nested_objects_equivalence(self):
        """Test GPU vs CPU equivalence for nested objects."""
        mask = create_nested_objects_mask(size=150)

        boundaries_cpu = _find_boundaries_cpu(mask, mode='inner')
        boundaries_gpu = find_boundaries_gpu(mask, mode='inner')

        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            "GPU and CPU differ for nested objects"

    def test_touching_objects_equivalence(self):
        """Test GPU vs CPU equivalence for touching objects."""
        mask = create_touching_objects_mask(size=120)

        boundaries_cpu = _find_boundaries_cpu(mask, mode='inner')
        boundaries_gpu = find_boundaries_gpu(mask, mode='inner')

        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            "GPU and CPU differ for touching objects"

    def test_labeled_boundary_equivalence(self):
        """Test compute_labeled_boundary_gpu vs manual CPU implementation."""
        mask = create_multi_object_mask(size=150, num_objects=7)

        # GPU version
        boundary_gpu = compute_labeled_boundary_gpu(mask)

        # Manual CPU version (from repair_masks.py)
        from skimage.segmentation import find_boundaries
        boundary_bool_cpu = find_boundaries(mask, mode='inner')
        boundary_cpu = np.zeros_like(mask, dtype=np.uint32)
        boundary_cpu[boundary_bool_cpu] = mask[boundary_bool_cpu]

        # Should be identical
        assert np.array_equal(boundary_cpu, boundary_gpu), \
            "GPU and CPU labeled boundaries differ"


# Edge cases and error handling tests

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_mask(self):
        """Test handling of empty masks."""
        mask = np.zeros((100, 100), dtype=np.uint32)

        boundaries = find_boundaries_gpu(mask, mode='inner')

        # Should return all False (no boundaries)
        assert not boundaries.any(), "Boundaries detected in empty mask"

    def test_full_mask(self):
        """Test handling of masks with no background."""
        mask = np.ones((100, 100), dtype=np.uint32)

        boundaries = find_boundaries_gpu(mask, mode='inner')

        # Full mask with single label has no inner boundaries
        # (erosion and dilation don't change labels)
        assert not boundaries.any(), "Unexpected boundaries in single-label full mask"

    def test_single_pixel_object(self):
        """Test handling of single-pixel objects."""
        mask = np.zeros((10, 10), dtype=np.uint32)
        mask[5, 5] = 1

        boundaries = find_boundaries_gpu(mask, mode='inner')

        # Single pixel should be its own boundary
        assert boundaries[5, 5], "Single pixel not detected as boundary"

    def test_invalid_mode(self):
        """Test error handling for invalid mode."""
        mask = create_simple_square_mask(size=50)

        with pytest.raises(ValueError, match="Unsupported mode"):
            find_boundaries_gpu(mask, mode='invalid_mode')

    def test_non_integer_mask_conversion(self):
        """Test automatic conversion of non-integer masks."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0

        # Should automatically convert to integer
        boundary_mask = compute_labeled_boundary_gpu(mask)

        assert boundary_mask.dtype == np.uint32
        assert boundary_mask.any(), "No boundaries detected after conversion"


# GPU-specific tests

@requires_gpu()
class TestGPUSpecific:
    """Tests specific to GPU functionality."""

    def test_gpu_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up after operations."""
        import cupy as cp
        from aegle.gpu_utils import get_gpu_memory_info

        # Get initial memory
        mem_before = get_gpu_memory_info()
        if mem_before is None:
            pytest.skip("Cannot query GPU memory")

        # Run boundary detection
        mask = create_simple_square_mask(size=2000, label=1)
        boundaries = find_boundaries_gpu(mask, mode='inner')

        # Force cleanup
        clear_gpu_memory()

        # Get final memory
        mem_after = get_gpu_memory_info()

        # Pool memory should be released (or very similar)
        # Allow for small variations
        assert abs(mem_after['pool_used_gb'] - mem_before['pool_used_gb']) < 0.1, \
            "GPU memory not properly cleaned up"

    def test_cupy_array_input(self):
        """Test that function accepts CuPy arrays as input."""
        import cupy as cp

        # Create mask on GPU
        mask_cpu = create_simple_square_mask(size=100, label=1)
        mask_gpu = cp.asarray(mask_cpu)

        # Should work with CuPy input
        boundaries = find_boundaries_gpu(mask_gpu, mode='inner')

        # Should return numpy array
        assert isinstance(boundaries, np.ndarray), "Should return numpy array"

        # Should match CPU version
        boundaries_cpu = _find_boundaries_cpu(mask_cpu, mode='inner')
        assert np.array_equal(boundaries, boundaries_cpu), \
            "CuPy input produces different results"

    def test_large_mask_performance(self):
        """Test GPU performance on large masks (not a strict benchmark)."""
        import time

        # Create large mask
        mask = create_multi_object_mask(size=5000, num_objects=10)

        # Time GPU version
        start = time.time()
        boundaries_gpu = find_boundaries_gpu(mask, mode='inner')
        gpu_time = time.time() - start

        # Time CPU version
        start = time.time()
        boundaries_cpu = _find_boundaries_cpu(mask, mode='inner')
        cpu_time = time.time() - start

        # Should produce same results
        assert np.array_equal(boundaries_cpu, boundaries_gpu), \
            "GPU and CPU produce different results for large mask"

        # Log performance (not a strict requirement, just informational)
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\nPerformance on {mask.shape}: GPU {gpu_time:.3f}s, "
                  f"CPU {cpu_time:.3f}s, speedup {speedup:.1f}x")


# CPU fallback tests

class TestCPUFallback:
    """Test CPU fallback when GPU unavailable."""

    def test_fallback_when_gpu_disabled(self):
        """Test that CPU fallback works when GPU is disabled."""
        mask = create_simple_square_mask(size=100, label=1)

        # Mock GPU as unavailable by patching the is_cupy_available import
        with patch('aegle.gpu_utils.is_cupy_available', return_value=False):
            boundaries = find_boundaries_gpu(mask, mode='inner')

        # Should still work via CPU fallback
        assert boundaries.shape == mask.shape
        assert boundaries.any(), "No boundaries detected in CPU fallback"

    @requires_gpu()
    def test_fallback_on_gpu_error(self):
        """Test CPU fallback when GPU operation fails."""
        mask = create_simple_square_mask(size=100, label=1)

        # Mock GPU error during erosion
        with patch('cupyx.scipy.ndimage.binary_erosion', side_effect=RuntimeError("GPU error")):
            # Should fall back to CPU
            boundaries = find_boundaries_gpu(mask, mode='inner')

        # Should still produce valid results via CPU fallback
        assert boundaries.shape == mask.shape
        assert boundaries.any(), "No boundaries detected after GPU error fallback"


# Memory stress tests

@requires_gpu()
class TestMemoryStress:
    """Test handling of large masks that may cause OOM."""

    def test_very_large_mask_handling(self):
        """Test handling of very large masks (may trigger OOM or fallback)."""
        # Create a large mask (100MB+)
        large_size = 10000
        mask = np.zeros((large_size, large_size), dtype=np.uint32)
        mask[100:large_size-100, 100:large_size-100] = 1

        # Should either succeed or fall back to CPU gracefully
        try:
            boundaries = find_boundaries_gpu(mask, mode='inner')
            assert boundaries.shape == mask.shape
        except RuntimeError as e:
            # OOM is acceptable for very large masks
            assert "memory" in str(e).lower() or "OOM" in str(e), \
                f"Unexpected error: {e}"

    @pytest.mark.parametrize("size", [1000, 2000, 5000])
    def test_progressive_sizes(self, size):
        """Test various sizes to find memory limits."""
        from aegle.gpu_utils import check_gpu_memory_for_masks

        # Check if we expect to have enough memory
        can_fit = check_gpu_memory_for_masks((size, size), num_masks=1)

        mask = create_simple_square_mask(size=size, label=1)

        if can_fit:
            # Should succeed
            boundaries = find_boundaries_gpu(mask, mode='inner')
            assert boundaries.shape == mask.shape
        else:
            # May fall back to CPU or raise OOM
            # Either is acceptable
            boundaries = find_boundaries_gpu(mask, mode='inner')
            # If we get here, fallback worked
            assert boundaries.shape == mask.shape


# Integration tests with repair_masks.py patterns

class TestRepairMasksIntegration:
    """Test integration with patterns from repair_masks.py."""

    def test_compute_labeled_boundary_pattern(self):
        """Test the exact pattern used in repair_masks.py."""
        # Create realistic mask
        mask = create_multi_object_mask(size=500, num_objects=20)

        # GPU version
        boundary_gpu = compute_labeled_boundary_gpu(mask)

        # Original pattern from repair_masks.py
        from aegle.repair_masks import _compute_labeled_boundary
        boundary_cpu = _compute_labeled_boundary(mask)

        # Should be identical
        assert np.array_equal(boundary_cpu, boundary_gpu), \
            "GPU version differs from repair_masks.py implementation"

    def test_realistic_segmentation_mask(self):
        """Test with a realistic segmentation mask structure."""
        # Create mask similar to real segmentation output
        mask = np.zeros((2000, 2000), dtype=np.uint32)

        # Add ~100 "cells" with irregular shapes
        np.random.seed(42)
        cell_id = 1
        for _ in range(100):
            # Random cell center
            cy = np.random.randint(50, 1950)
            cx = np.random.randint(50, 1950)

            # Random cell size
            radius = np.random.randint(10, 30)

            # Create approximate cell
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            cell_region = (y**2 + x**2) <= radius**2

            # Place in mask
            y_slice = slice(max(0, cy-radius), min(2000, cy+radius+1))
            x_slice = slice(max(0, cx-radius), min(2000, cx+radius+1))

            # Adjust region size to fit
            region_y_size = y_slice.stop - y_slice.start
            region_x_size = x_slice.stop - x_slice.start
            cell_region_fitted = cell_region[:region_y_size, :region_x_size]

            mask[y_slice, x_slice][cell_region_fitted] = cell_id
            cell_id += 1

        # Test boundary detection
        boundary_mask = compute_labeled_boundary_gpu(mask)

        # Verify properties
        assert boundary_mask.shape == mask.shape
        assert boundary_mask.dtype == np.uint32
        assert boundary_mask.max() <= mask.max()

        # Verify all boundary labels exist in original
        boundary_labels = set(np.unique(boundary_mask)) - {0}
        mask_labels = set(np.unique(mask)) - {0}
        assert boundary_labels.issubset(mask_labels), \
            "Boundary contains invalid labels"
