"""
GPU testing helper utilities.

This module provides a compatibility layer for gpu_test_utils.py.
All functions are re-exported from gpu_test_utils for convenience.
"""

from tests.utils.gpu_test_utils import (
    requires_gpu,
    skip_if_no_gpu,
    assert_gpu_cpu_equal,
    compare_cpu_gpu_results,
    assert_arrays_close,
    mock_gpu_memory,
    create_test_masks,
    create_test_channel_data,
)

__all__ = [
    'requires_gpu',
    'skip_if_no_gpu',
    'assert_gpu_cpu_equal',
    'compare_cpu_gpu_results',
    'assert_arrays_close',
    'mock_gpu_memory',
    'create_test_masks',
    'create_test_channel_data',
]
