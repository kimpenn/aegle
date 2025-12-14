"""
Pytest configuration and shared fixtures for the test suite.

This module provides:
- GPU-related fixtures (gpu_available, synthetic_data_small, etc.)
- Shared test data generators
- Common test utilities
"""

import pytest
import numpy as np
from pathlib import Path

# Import GPU utilities
try:
    from aegle.gpu_utils import is_cupy_available
    from aegle_analysis.gpu_utils import is_gpu_available
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False


# =============================================================================
# GPU Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def gpu_available():
    """
    Session-scoped fixture that checks if GPU is available.

    Returns:
        bool: True if CuPy GPU is available, False otherwise

    Example:
        def test_something(gpu_available):
            if not gpu_available:
                pytest.skip("GPU not available")
            # ... GPU-dependent test code
    """
    if not GPU_UTILS_AVAILABLE:
        return False
    return is_cupy_available()


@pytest.fixture(scope="session")
def analysis_gpu_available():
    """
    Session-scoped fixture that checks if GPU is available for analysis.

    Returns:
        bool: True if GPU is available (via CuPy or PyTorch), False otherwise

    Example:
        def test_analysis_gpu(analysis_gpu_available):
            if not analysis_gpu_available:
                pytest.skip("Analysis GPU not available")
            # ... GPU-dependent analysis test code
    """
    if not GPU_UTILS_AVAILABLE:
        return False
    return is_gpu_available()


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def synthetic_data_small():
    """
    Function-scoped fixture providing small synthetic dataset.

    Returns:
        SyntheticAnalysisData: Small dataset (1K cells, 10 markers, 3 clusters)

    Example:
        def test_with_small_data(synthetic_data_small):
            adata = synthetic_data_small.adata
            assert adata.n_obs == 1000
    """
    from tests.utils.synthetic_analysis_data import get_small_dataset
    return get_small_dataset(random_seed=42)


@pytest.fixture(scope="function")
def synthetic_data_medium():
    """
    Function-scoped fixture providing medium synthetic dataset.

    Returns:
        SyntheticAnalysisData: Medium dataset (10K cells, 25 markers, 5 clusters)

    Example:
        def test_with_medium_data(synthetic_data_medium):
            adata = synthetic_data_medium.adata
            assert adata.n_obs == 10000
    """
    from tests.utils.synthetic_analysis_data import get_medium_dataset
    return get_medium_dataset(random_seed=42)


@pytest.fixture(scope="function")
def synthetic_data_large():
    """
    Function-scoped fixture providing large synthetic dataset.

    Returns:
        SyntheticAnalysisData: Large dataset (100K cells, 50 markers, 8 clusters)

    Example:
        def test_with_large_data(synthetic_data_large):
            adata = synthetic_data_large.adata
            assert adata.n_obs == 100000
    """
    from tests.utils.synthetic_analysis_data import get_large_dataset
    return get_large_dataset(random_seed=42)


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """
    Function-scoped fixture providing a temporary output directory.

    Args:
        tmp_path: pytest built-in fixture for temporary directory

    Returns:
        Path: Temporary directory path that will be cleaned up after test

    Example:
        def test_output(temp_output_dir):
            output_file = temp_output_dir / "result.csv"
            # Write to output_file
            assert output_file.exists()
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Test Mask Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def simple_masks():
    """
    Function-scoped fixture providing simple test masks.

    Returns:
        tuple: (cell_mask, nucleus_mask) as uint32 arrays with 10 cells

    Example:
        def test_masks(simple_masks):
            cell_mask, nucleus_mask = simple_masks
            assert cell_mask.shape == nucleus_mask.shape
    """
    from tests.utils.gpu_test_utils import create_test_masks
    return create_test_masks(num_cells=10, image_shape=(100, 100))


@pytest.fixture(scope="function")
def medium_masks():
    """
    Function-scoped fixture providing medium-sized test masks.

    Returns:
        tuple: (cell_mask, nucleus_mask) as uint32 arrays with 100 cells

    Example:
        def test_with_medium_masks(medium_masks):
            cell_mask, nucleus_mask = medium_masks
            assert cell_mask.max() == 100
    """
    from tests.utils.gpu_test_utils import create_test_masks
    return create_test_masks(num_cells=100, image_shape=(500, 500))


@pytest.fixture(scope="function")
def large_masks():
    """
    Function-scoped fixture providing large test masks.

    Returns:
        tuple: (cell_mask, nucleus_mask) as uint32 arrays with 1000 cells

    Example:
        def test_with_large_masks(large_masks):
            cell_mask, nucleus_mask = large_masks
            n_cells = len(np.unique(cell_mask)) - 1
            assert n_cells == 1000
    """
    from tests.utils.gpu_test_utils import create_test_masks
    return create_test_masks(num_cells=1000, image_shape=(2000, 2000))


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers.

    This function is called by pytest during initialization.
    """
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (CuPy)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.

    This function is called by pytest after test collection.
    It automatically marks tests based on their names.
    """
    for item in items:
        # Auto-mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Auto-mark slow tests
        if "slow" in item.nodeid.lower() or "stress" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Auto-mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
