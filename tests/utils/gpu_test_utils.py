"""
GPU testing utilities for the test suite.

Provides helpers for GPU-specific tests including:
- Decorators for skipping tests when GPU unavailable
- Numerical comparison helpers
- Mock utilities for GPU memory testing
"""

import numpy as np
import pytest
from contextlib import contextmanager
from unittest.mock import patch

from aegle.gpu_utils import is_cupy_available


def requires_gpu():
    """
    Pytest decorator to skip tests when GPU is not available.

    Usage:
        @requires_gpu()
        def test_gpu_feature():
            # Test that requires GPU
            pass
    """
    return pytest.mark.skipif(
        not is_cupy_available(),
        reason="GPU (CuPy) not available"
    )


def skip_if_no_gpu():
    """
    Pytest decorator to skip tests when GPU is not available.

    Alias for requires_gpu() for consistency with task requirements.

    Usage:
        @skip_if_no_gpu()
        def test_gpu_feature():
            # Test that requires GPU
            pass
    """
    return requires_gpu()


def assert_gpu_cpu_equal(gpu_result, cpu_result, rtol=1e-5, atol=1e-6,
                         err_msg="GPU and CPU results differ"):
    """
    Assert that GPU and CPU results are numerically equivalent.

    Args:
        gpu_result: Result from GPU computation (pandas DataFrame or numpy array)
        cpu_result: Result from CPU computation (pandas DataFrame or numpy array)
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-6)
        err_msg: Error message prefix

    Raises:
        AssertionError: If results differ beyond tolerance
    """
    # Handle pandas DataFrames
    if hasattr(gpu_result, 'values') and hasattr(cpu_result, 'values'):
        # Check column names match
        assert list(gpu_result.columns) == list(cpu_result.columns), \
            f"{err_msg}: Column names differ"

        # Check index matches
        assert list(gpu_result.index) == list(cpu_result.index), \
            f"{err_msg}: Index values differ"

        # Compare values column by column to handle mixed types
        for col in gpu_result.columns:
            gpu_col = gpu_result[col].values
            cpu_col = cpu_result[col].values

            # Skip comparison for non-numeric columns
            if not np.issubdtype(gpu_col.dtype, np.number):
                assert (gpu_col == cpu_col).all(), \
                    f"{err_msg}: Non-numeric column '{col}' differs"
                continue

            # For numeric columns, use approximate equality
            np.testing.assert_allclose(
                gpu_col, cpu_col,
                rtol=rtol, atol=atol,
                err_msg=f"{err_msg}: Column '{col}' differs"
            )

    # Handle numpy arrays
    elif isinstance(gpu_result, np.ndarray) and isinstance(cpu_result, np.ndarray):
        np.testing.assert_allclose(
            gpu_result, cpu_result,
            rtol=rtol, atol=atol,
            err_msg=err_msg
        )

    else:
        raise TypeError(
            f"Unsupported types for comparison: {type(gpu_result)}, {type(cpu_result)}"
        )


def compare_cpu_gpu_results(cpu_result, gpu_result, rtol=1e-5, atol=1e-6):
    """
    Compare CPU and GPU results with numerical tolerance.

    This is a wrapper around assert_gpu_cpu_equal with swapped argument order
    to match the task requirements (cpu_result first, gpu_result second).

    Args:
        cpu_result: Result from CPU computation (pandas DataFrame or numpy array)
        gpu_result: Result from GPU computation (pandas DataFrame or numpy array)
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-6)

    Returns:
        bool: True if results match within tolerance

    Raises:
        AssertionError: If results differ beyond tolerance
    """
    assert_gpu_cpu_equal(
        gpu_result=gpu_result,
        cpu_result=cpu_result,
        rtol=rtol,
        atol=atol,
        err_msg="GPU and CPU results differ"
    )
    return True


def assert_arrays_close(arr1, arr2, rtol=1e-5, atol=1e-6, err_msg="Arrays differ"):
    """
    Assert that two arrays are numerically close.

    This is a convenience wrapper around numpy.testing.assert_allclose.

    Args:
        arr1: First array
        arr2: Second array
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-6)
        err_msg: Error message to display if assertion fails

    Raises:
        AssertionError: If arrays differ beyond tolerance

    Example:
        >>> arr1 = np.array([1.0, 2.0, 3.0])
        >>> arr2 = np.array([1.0000001, 2.0000001, 3.0000001])
        >>> assert_arrays_close(arr1, arr2, rtol=1e-5)
    """
    np.testing.assert_allclose(
        arr1, arr2,
        rtol=rtol,
        atol=atol,
        err_msg=err_msg
    )


@contextmanager
def mock_gpu_memory(total_bytes, free_bytes):
    """
    Context manager to mock GPU memory information.

    Useful for testing GPU memory estimation logic without requiring specific
    hardware configurations.

    Args:
        total_bytes: Total GPU memory to simulate (bytes)
        free_bytes: Free GPU memory to simulate (bytes)

    Usage:
        with mock_gpu_memory(total_bytes=16e9, free_bytes=8e9):
            batch_size = estimate_gpu_batch_size(...)
            # batch_size will be computed based on mocked memory
    """
    mock_info = {
        'device_id': 0,
        'total_gb': total_bytes / 1e9,
        'free_gb': free_bytes / 1e9,
        'used_gb': (total_bytes - free_bytes) / 1e9,
        'pool_used_gb': 0.0,
        'pool_total_gb': 0.0,
    }

    with patch('aegle.gpu_utils.get_gpu_memory_info', return_value=mock_info):
        yield mock_info


def create_test_masks(num_cells, image_shape=(100, 100)):
    """
    Create synthetic nucleus and wholecell masks for testing.

    Creates non-overlapping square cells arranged in a grid pattern.

    Args:
        num_cells: Number of cells to create
        image_shape: Shape of the output masks (height, width)

    Returns:
        tuple: (nucleus_mask, wholecell_mask) as uint32 arrays
    """
    nucleus_mask = np.zeros(image_shape, dtype=np.uint32)
    wholecell_mask = np.zeros(image_shape, dtype=np.uint32)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_cells)))
    cell_size = min(image_shape[0] // grid_size, image_shape[1] // grid_size)

    cell_id = 1
    for i in range(grid_size):
        for j in range(grid_size):
            if cell_id > num_cells:
                break

            # Calculate cell position
            y_start = i * cell_size + 2
            y_end = (i + 1) * cell_size - 2
            x_start = j * cell_size + 2
            x_end = (j + 1) * cell_size - 2

            # Create nucleus (inner square)
            nucleus_size = cell_size // 3
            ny_center = (y_start + y_end) // 2
            nx_center = (x_start + x_end) // 2
            nucleus_mask[
                ny_center - nucleus_size:ny_center + nucleus_size,
                nx_center - nucleus_size:nx_center + nucleus_size
            ] = cell_id

            # Create wholecell (outer square)
            wholecell_mask[y_start:y_end, x_start:x_end] = cell_id

            cell_id += 1

    return nucleus_mask, wholecell_mask


def create_test_channel_data(mask, channel_values):
    """
    Create synthetic channel data with known intensities per cell.

    Args:
        mask: Cell mask (uint32 array)
        channel_values: Dict mapping cell_id to intensity value
                       Example: {1: 100.0, 2: 200.0}

    Returns:
        numpy array with same shape as mask, containing specified intensities
    """
    channel = np.zeros_like(mask, dtype=np.float32)

    for cell_id, intensity in channel_values.items():
        channel[mask == cell_id] = intensity

    return channel
