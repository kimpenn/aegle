"""
Test that conftest.py fixtures work correctly.

This is a simple test file to verify all fixtures load and function properly.
"""

import pytest
import numpy as np


def test_gpu_available_fixture(gpu_available):
    """Test gpu_available fixture works."""
    assert isinstance(gpu_available, bool)


def test_analysis_gpu_available_fixture(analysis_gpu_available):
    """Test analysis_gpu_available fixture works."""
    assert isinstance(analysis_gpu_available, bool)


def test_synthetic_data_small_fixture(synthetic_data_small):
    """Test synthetic_data_small fixture works."""
    assert synthetic_data_small is not None
    assert synthetic_data_small.adata is not None
    assert synthetic_data_small.adata.n_obs == 1000
    assert synthetic_data_small.adata.n_vars == 10


def test_synthetic_data_medium_fixture(synthetic_data_medium):
    """Test synthetic_data_medium fixture works."""
    assert synthetic_data_medium is not None
    assert synthetic_data_medium.adata is not None
    assert synthetic_data_medium.adata.n_obs == 10000
    assert synthetic_data_medium.adata.n_vars == 25


def test_simple_masks_fixture(simple_masks):
    """Test simple_masks fixture works."""
    cell_mask, nucleus_mask = simple_masks
    assert cell_mask is not None
    assert nucleus_mask is not None
    assert cell_mask.shape == (100, 100)
    assert nucleus_mask.shape == (100, 100)
    assert cell_mask.max() == 10


def test_medium_masks_fixture(medium_masks):
    """Test medium_masks fixture works."""
    cell_mask, nucleus_mask = medium_masks
    assert cell_mask is not None
    assert nucleus_mask is not None
    assert cell_mask.max() == 100


def test_temp_output_dir_fixture(temp_output_dir):
    """Test temp_output_dir fixture works."""
    assert temp_output_dir.exists()
    assert temp_output_dir.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
