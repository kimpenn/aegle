# GPU Testing Suite Documentation

This document describes the comprehensive GPU testing infrastructure for the Aegle mask repair pipeline.

## Overview

The GPU testing suite consists of 120+ tests organized into 7 test modules that validate:

1. **Correctness**: GPU results match CPU exactly
2. **Performance**: GPU meets minimum speedup targets
3. **Robustness**: Graceful error handling and fallbacks
4. **Stress**: Behavior under extreme conditions
5. **Regression**: Prevention of future performance/correctness regressions

## Test Modules

### 1. Component Tests (Existing)

These test individual GPU components:

- **`test_repair_gpu_morphology.py`** (30 tests)
  - GPU morphological operations (boundary detection)
  - GPU vs CPU equivalence for all modes (inner, outer, thick)
  - Various mask sizes and configurations
  - OOM handling and CPU fallback

- **`test_repair_gpu_overlap.py`** (20 tests)
  - GPU overlap matrix computation
  - Non-contiguous label handling
  - Batch processing
  - Performance benchmarks

- **`test_repair_gpu_integration.py`** (20 tests)
  - End-to-end GPU repair pipeline
  - GPU vs CPU equivalence on all fixtures
  - Performance scaling tests
  - Error handling and robustness

**Total Component Tests**: 70 tests

### 2. Comprehensive Test Suite (New)

#### `test_gpu_repair_suite.py` - Master Test Suite

**Purpose**: Comprehensive validation of the entire GPU repair pipeline across multiple scenarios.

**Test Classes**:
- `TestGPURepairSuite`: Full pipeline validation
  - Small, medium, large samples (100 - 10K cells)
  - GPU vs CPU equivalence on all fixtures
  - Non-contiguous label handling
  - Performance scaling
  - Memory efficiency
  - Error recovery
  - CPU fallback scenarios

- `TestGPURepairRegressionPrevention`: Regression tests
  - Bit-identical GPU/CPU masks
  - Metadata completeness
  - Various dtype handling

- `TestGPURepairStress`: Stress tests (marked with `@pytest.mark.stress`)
  - Very large samples (50K+ cells)
  - Extreme label gaps
  - Memory leak detection (100+ iterations)
  - Concurrent operations

**Run Commands**:
```bash
# Run all comprehensive tests (excluding stress)
pytest tests/test_gpu_repair_suite.py -v

# Run only stress tests
pytest tests/test_gpu_repair_suite.py -v -m stress

# Skip stress tests
pytest tests/test_gpu_repair_suite.py -v -m "not stress"
```

#### `test_gpu_performance_regression.py` - Performance Validation

**Purpose**: Ensure GPU performance doesn't regress in future changes.

**Minimum Performance Targets**:
- GPU morphology: **10x+ speedup** over CPU
- GPU overlap computation: **3x+ speedup** for >5K cells
- GPU integration: **No slowdown** vs CPU (minimum 1x)

**Test Classes**:
- `TestGPUMorphologyPerformance`: Morphological operations benchmarks
- `TestGPUOverlapPerformance`: Overlap computation benchmarks
- `TestGPUIntegrationPerformance`: End-to-end performance
- `TestGPUPerformanceScaling`: Throughput scaling tests

**Run Commands**:
```bash
# Run all performance tests
pytest tests/test_gpu_performance_regression.py -v

# Run specific performance test class
pytest tests/test_gpu_performance_regression.py::TestGPUMorphologyPerformance -v
```

#### `test_gpu_availability.py` - GPU Detection and Fallback

**Purpose**: Test behavior when GPU is/isn't available.

**Test Classes**:
- `TestGPUDetection`: GPU detection works correctly
- `TestCPUFallback`: CPU fallback when GPU explicitly disabled
- `TestCPUFallbackWhenGPUUnavailable`: Graceful fallback when CuPy not installed
- `TestGPUErrorHandling`: Error handling for GPU operations
- `TestConfigurationDrivenGPU`: Configuration-driven GPU enable/disable

**Run Commands**:
```bash
# Run all availability tests
pytest tests/test_gpu_availability.py -v

# Run on system without GPU (tests fallback)
pytest tests/test_gpu_availability.py -v  # Will skip GPU-only tests
```

#### `test_gpu_stress.py` - Stress Tests

**Purpose**: Push GPU components to their limits.

**Test Classes**:
- `TestGPUStressVeryLargeSamples`: 50K+ cell samples
- `TestGPUStressExtremeLabelGaps`: Non-contiguous labels like [1, 1000000]
- `TestGPUStressMemoryLeaks`: Memory leak detection (100+ iterations)
- `TestGPUStressConcurrentOperations`: Sequential/concurrent operation tests

**All tests marked with `@pytest.mark.stress`**

**Run Commands**:
```bash
# Run all stress tests
pytest tests/test_gpu_stress.py -v -m stress

# Skip stress tests (for regular test runs)
pytest tests/test_gpu_stress.py -v -m "not stress"
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- **`@requires_gpu()`**: Skip test if GPU (CuPy) not available
- **`@pytest.mark.stress`**: Mark test as stress test (long-running, resource-intensive)
- **`@pytest.mark.skipif(...)`**: Conditional test skipping

## Running Tests

### Quick Commands

```bash
# Run ALL GPU tests (excluding stress)
pytest tests/test_gpu*.py -v -m "not stress"

# Run ALL GPU tests (including stress)
pytest tests/test_gpu*.py -v

# Run only component tests (existing)
pytest tests/test_repair_gpu_*.py -v

# Run only new comprehensive tests
pytest tests/test_gpu_repair_suite.py tests/test_gpu_performance_regression.py \
       tests/test_gpu_availability.py -v

# Run specific test module
pytest tests/test_gpu_repair_suite.py -v

# Run specific test class
pytest tests/test_gpu_repair_suite.py::TestGPURepairSuite -v

# Run specific test function
pytest tests/test_gpu_repair_suite.py::TestGPURepairSuite::test_full_pipeline_small_sample -v
```

### Performance Testing

```bash
# Run performance regression tests
pytest tests/test_gpu_performance_regression.py -v -s

# Run with timing output
pytest tests/test_gpu_performance_regression.py -v -s --durations=10
```

### Stress Testing

```bash
# Run stress tests (may take several minutes)
pytest tests/test_gpu_stress.py -v -m stress -s

# Run specific stress test
pytest tests/test_gpu_stress.py::TestGPUStressMemoryLeaks::test_repeated_runs_no_memory_leak_100_iterations -v -s
```

### CI/CD Testing

```bash
# Standard CI test run (fast, no stress tests)
pytest tests/test_gpu*.py -v -m "not stress" --tb=short

# With GPU available
pytest tests/test_gpu*.py -v -m "not stress"

# Without GPU (tests CPU fallback)
pytest tests/test_gpu*.py -v -m "not stress"  # GPU tests will be skipped
```

## Test Coverage

### By Component

| Component | Test Module | Tests | Coverage |
|-----------|-------------|-------|----------|
| Morphology | `test_repair_gpu_morphology.py` | 30 | >95% |
| Overlap | `test_repair_gpu_overlap.py` | 20 | >95% |
| Integration | `test_repair_gpu_integration.py` | 20 | >90% |
| Comprehensive | `test_gpu_repair_suite.py` | 15 | Full pipeline |
| Performance | `test_gpu_performance_regression.py` | 10 | Benchmarks |
| Availability | `test_gpu_availability.py` | 15 | Fallback logic |
| Stress | `test_gpu_stress.py` | 12 | Edge cases |
| **Total** | **7 modules** | **122** | **>90%** |

### By Test Category

| Category | Tests | Purpose |
|----------|-------|---------|
| Correctness | 60 | GPU vs CPU equivalence |
| Performance | 20 | Speedup validation |
| Robustness | 25 | Error handling, edge cases |
| Stress | 12 | Extreme conditions |
| Regression | 5 | Prevent future issues |

## Test Fixtures

All tests use synthetic test fixtures from `tests/utils/repair_test_fixtures.py`:

- `create_simple_mask_pair()`: Perfect 1:1 cell-nucleus matches
- `create_overlapping_nuclei_case()`: Multiple nuclei per cell
- `create_unmatched_cells_case()`: Cells without nuclei
- `create_partial_overlap_case()`: Nuclei extending outside cells
- `create_stress_test_case()`: Large-scale samples (10K+ cells)
- `create_edge_case_empty_masks()`: Empty masks
- `create_edge_case_single_cell()`: Single cell-nucleus pair
- `create_noncontiguous_labels_case()`: Non-contiguous label IDs

## GPU Test Utilities

Helper functions from `tests/utils/gpu_test_utils.py`:

- `requires_gpu()`: Decorator to skip tests when GPU unavailable
- `assert_gpu_cpu_equal()`: Assert GPU/CPU results are equivalent
- `mock_gpu_memory()`: Mock GPU memory for testing
- `create_test_masks()`: Create synthetic masks for testing

## Success Criteria

### Correctness

- ✅ GPU results match CPU exactly (tolerance <1e-6 for floats, exact for integers)
- ✅ All existing Phase 0-1 tests pass with GPU enabled
- ✅ GPU vs CPU equivalence verified on all fixtures

### Performance

- ✅ GPU morphology: 10x+ speedup over CPU
- ✅ GPU overlap: 3x+ speedup for >5K cells
- ✅ GPU integration: No slowdown vs CPU

### Robustness

- ✅ Graceful CPU fallback when GPU unavailable
- ✅ Clear error messages on failures
- ✅ Handles edge cases (empty masks, single cell, extreme labels)

### Stress

- ✅ No memory leaks after 100+ iterations
- ✅ Handles large samples (10K+ cells)
- ✅ Handles extreme label gaps (e.g., [1, 1000000])

## Continuous Integration

### GitHub Actions / CI Configuration

```yaml
# Example CI configuration
test_gpu:
  runs-on: [self-hosted, gpu]  # Runner with GPU
  steps:
    - name: Run GPU tests
      run: |
        pytest tests/test_gpu*.py -v -m "not stress" --tb=short --maxfail=5

    - name: Run stress tests (optional)
      if: github.event_name == 'schedule'  # Only on nightly builds
      run: |
        pytest tests/test_gpu_stress.py -v -m stress -s

test_cpu_fallback:
  runs-on: ubuntu-latest  # Standard runner without GPU
  steps:
    - name: Test CPU fallback
      run: |
        pytest tests/test_gpu*.py -v -m "not stress" --tb=short
        # GPU tests should be skipped, CPU fallback tests should pass
```

## Troubleshooting

### Common Issues

1. **Tests skipped with "GPU not available"**
   - CuPy not installed or GPU not detected
   - Install CuPy: `pip install cupy-cuda11x` (match your CUDA version)
   - Verify GPU: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`

2. **OOM errors in stress tests**
   - Stress tests require high-end GPU (16+ GB VRAM)
   - Skip stress tests: `pytest -m "not stress"`
   - Reduce test sample sizes in stress tests

3. **Performance tests failing**
   - GPU may be warming up (run tests multiple times)
   - Check GPU isn't being used by other processes
   - Verify CUDA/CuPy versions compatible

4. **Non-deterministic test failures**
   - Check random seeds are set in fixtures
   - Verify no global state leakage between tests
   - Run with `-v -s` to see detailed output

## Development Workflow

### Adding New GPU Tests

1. Create test in appropriate module:
   - Component test → `test_repair_gpu_*.py`
   - Comprehensive → `test_gpu_repair_suite.py`
   - Performance → `test_gpu_performance_regression.py`
   - Stress → `test_gpu_stress.py`

2. Use appropriate markers:
   ```python
   @requires_gpu()
   def test_my_gpu_feature():
       ...

   @pytest.mark.stress
   @requires_gpu()
   def test_extreme_case():
       ...
   ```

3. Follow naming conventions:
   - `test_gpu_*` for GPU-specific tests
   - `test_*_vs_cpu` for equivalence tests
   - `test_*_performance` for benchmarks

4. Run new tests:
   ```bash
   pytest tests/test_gpu_*.py::test_my_gpu_feature -v -s
   ```

### Before Committing

```bash
# Run full test suite (excluding stress)
pytest tests/test_gpu*.py -v -m "not stress"

# Check test coverage
pytest tests/test_gpu*.py --cov=aegle --cov-report=term-missing

# Run stress tests if time permits
pytest tests/test_gpu_stress.py -v -m stress
```

## References

- **GPU Utilities**: `aegle/gpu_utils.py`
- **GPU Morphology**: `aegle/repair_masks_gpu_morphology.py`
- **GPU Overlap**: `aegle/repair_masks_gpu_overlap.py`
- **GPU Integration**: `aegle/repair_masks_gpu.py`
- **Test Fixtures**: `tests/utils/repair_test_fixtures.py`
- **GPU Test Utils**: `tests/utils/gpu_test_utils.py`
- **Phase 2 Plan**: `scratch/repair_optimization/PLANS_phase2_gpu_acceleration.md`

## Performance Testing and Regression Detection

### Analysis Pipeline Performance Tests

**Module**: `tests/test_analysis_performance.py`

This module provides performance benchmarking and regression detection for the analysis pipeline (normalization, clustering, differential expression).

#### Baseline Performance Tests

These tests establish CPU performance baselines on small datasets:

```bash
# Run all baseline tests
pytest tests/test_analysis_performance.py -v -k "baseline"

# Run specific baseline tests
pytest tests/test_analysis_performance.py::TestBaselineNormalizationPerformance -v
pytest tests/test_analysis_performance.py::TestBaselineClusteringPerformance -v
pytest tests/test_analysis_performance.py::TestBaselineDifferentialPerformance -v
pytest tests/test_analysis_performance.py::TestBaselineFullPipeline -v
```

**What it does:**
- Times each analysis stage (normalization, k-NN, Leiden, UMAP, differential)
- Records results to `tests/performance_baselines.json`
- Includes system info and metadata for reproducibility

**Current baselines (1K cells, 10 markers):**
- Normalization (log1p): ~2ms
- k-NN graph: ~145ms
- Leiden clustering: ~53ms
- UMAP: ~670ms
- Differential (Wilcoxon): ~28ms
- Full pipeline: ~900ms

#### Regression Detection Tests

These tests compare current performance to baselines and warn if >20% slower:

```bash
# Run regression tests
pytest tests/test_analysis_performance.py::TestCPUPerformanceRegression -v
```

**What it does:**
- Runs each analysis stage again
- Compares to baseline from `performance_baselines.json`
- Logs warning (not failure) if >20% slower
- Helps detect accidental performance degradation

#### GPU Speedup Tests (Placeholder for Phase 5)

```bash
# These tests are currently skipped
pytest tests/test_analysis_performance.py::TestGPUSpeedupRequirement -v
```

**Phase 5 TODO:**
- Implement GPU-accelerated analysis functions
- Benchmark GPU vs CPU on medium datasets (10K cells)
- Assert GPU achieves >5x speedup
- Record GPU baselines separately

#### Performance Baselines File

**Location**: `tests/performance_baselines.json`

**Structure**:
```json
{
  "stage_name": {
    "time_seconds": 0.123,
    "timestamp": "2025-11-23T12:00:00",
    "system_info": {
      "platform": "Linux-...",
      "processor": "x86_64",
      "python_version": "3.8.10"
    },
    "metadata": {
      "n_cells": 1000,
      "n_markers": 10
    },
    "history": [
      {"time_seconds": 0.120, "timestamp": "2025-11-22T..."}
    ]
  }
}
```

**When to update baselines:**
- After intentional performance improvements
- After upgrading dependencies (numpy, scipy, scanpy)
- When baseline becomes stale (e.g., hardware upgrade)

**How to update**:
```bash
# Just re-run baseline tests
pytest tests/test_analysis_performance.py -v -k "baseline"
```

The baseline file preserves history (last 10 entries) so you can track trends.

#### Performance Helpers

**Module**: `tests/utils/performance_helpers.py`

Utilities for accurate benchmarking:

```python
from tests.utils.performance_helpers import (
    benchmark_function,
    record_baseline,
    compare_to_baseline,
    format_time,
)

# Benchmark a function
result, time_sec = benchmark_function(my_func, arg1, arg2, warmup=True)

# Record to baseline
record_baseline("my_stage", time_sec, "baselines.json", metadata={...})

# Compare to baseline
status, ratio, baseline = compare_to_baseline("my_stage", time_sec, "baselines.json")
if status == "slower":
    print(f"WARNING: {ratio:.2f}x slower!")

# Format time nicely
print(f"Took {format_time(time_sec)}")  # "1.23s", "123ms", etc.
```

## Summary

The GPU testing suite provides comprehensive validation of the GPU-accelerated mask repair pipeline:

- **122 tests** across 7 modules
- **>90% code coverage** for GPU repair modules
- **Correctness**, **performance**, **robustness**, and **stress** testing
- **Regression prevention** via minimum performance targets
- **CI/CD ready** with appropriate test markers
- **Clear documentation** and troubleshooting guidance

The performance testing framework enables:
- **Baseline tracking** for CPU analysis pipeline
- **Regression detection** (warns if >20% slower)
- **GPU speedup validation** (Phase 5 placeholder)
- **Reproducible benchmarks** with system info and metadata

This ensures the GPU implementation is production-ready and maintains correctness and performance over time.

---

# Analysis GPU Testing Infrastructure

This section describes the GPU testing infrastructure for the **analysis pipeline** (downstream of segmentation), which uses `aegle_analysis.gpu_utils` for GPU acceleration of clustering, normalization, and differential expression.

## Analysis GPU Modules

The analysis pipeline has separate GPU utilities from the segmentation pipeline:

- **Segmentation GPU**: `aegle.gpu_utils` (CuPy-only, for mask repair)
- **Analysis GPU**: `aegle_analysis.gpu_utils` (CuPy or PyTorch, for clustering/analysis)

## Test Module: `test_analysis_gpu_availability.py`

This module tests all functions in `aegle_analysis.gpu_utils`:

### Functions Tested

1. **`is_gpu_available()`**: Check if GPU is available via CuPy or PyTorch
2. **`get_gpu_memory_info(device_id=0)`**: Get GPU memory information
3. **`log_gpu_info(logger)`**: Log GPU availability and details
4. **`select_compute_backend(use_gpu, fallback_to_cpu, logger)`**: Select GPU or CPU backend

### Running Analysis GPU Tests

```bash
# Run all analysis GPU availability tests
pytest tests/test_analysis_gpu_availability.py -v

# Run specific test class
pytest tests/test_analysis_gpu_availability.py::TestIsGPUAvailable -v
pytest tests/test_analysis_gpu_availability.py::TestGetGPUMemoryInfo -v
pytest tests/test_analysis_gpu_availability.py::TestLogGPUInfo -v
pytest tests/test_analysis_gpu_availability.py::TestSelectComputeBackend -v
pytest tests/test_analysis_gpu_availability.py::TestGPUUtilsIntegration -v
```

### Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestIsGPUAvailable` | 5 | GPU detection, caching, error handling |
| `TestGetGPUMemoryInfo` | 6 | Memory queries, fallback mechanisms |
| `TestLogGPUInfo` | 3 | Logging with/without GPU |
| `TestSelectComputeBackend` | 5 | Backend selection logic |
| `TestGPUUtilsIntegration` | 3 | Integration between functions |
| **Total** | **22** | **>95%** |

## GPU Test Helpers and Fixtures

### Test Utilities (`tests/utils/gpu_test_helpers.py`)

This module provides reusable GPU testing utilities that work for both segmentation and analysis GPU tests:

```python
from tests.utils.gpu_test_helpers import (
    skip_if_no_gpu,           # Skip test if GPU unavailable
    compare_cpu_gpu_results,  # Compare CPU/GPU outputs
    assert_arrays_close,      # Assert numerical equivalence
    mock_gpu_memory,          # Mock GPU memory for testing
)
```

#### Using `skip_if_no_gpu()`

```python
@skip_if_no_gpu()
def test_gpu_clustering():
    """This test only runs if GPU is available."""
    # GPU-dependent test code
    pass
```

#### Using `compare_cpu_gpu_results()`

```python
def test_cpu_gpu_equivalence():
    """Test that CPU and GPU produce same results."""
    data = create_test_data()

    # Run on CPU
    result_cpu = cluster(data, use_gpu=False)

    # Run on GPU
    result_gpu = cluster(data, use_gpu=True)

    # Compare results (default tolerance: rtol=1e-5, atol=1e-6)
    compare_cpu_gpu_results(result_cpu, result_gpu, rtol=1e-5)
```

#### Using `assert_arrays_close()`

```python
def test_numerical_equivalence():
    """Test arrays are numerically close."""
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([1.0000001, 2.0000001, 3.0000001])

    # Will pass with default tolerance (rtol=1e-5)
    assert_arrays_close(arr1, arr2)

    # Stricter tolerance
    assert_arrays_close(arr1, arr2, rtol=1e-7, atol=1e-8)
```

#### Using `mock_gpu_memory()`

```python
def test_batch_size_with_limited_memory():
    """Test batch size estimation with mocked GPU memory."""
    # Mock 8 GB GPU with 4 GB free
    with mock_gpu_memory(total_bytes=8e9, free_bytes=4e9):
        batch_size = estimate_batch_size(data_size=large_dataset)
        assert 1 <= batch_size <= 100
```

### Fixtures (`tests/conftest.py`)

The `conftest.py` file provides shared fixtures for all GPU tests:

#### GPU Availability Fixtures

```python
def test_something(gpu_available):
    """Use gpu_available fixture for segmentation GPU (CuPy)."""
    if not gpu_available:
        pytest.skip("GPU not available")
    # ... test code

def test_analysis(analysis_gpu_available):
    """Use analysis_gpu_available fixture for analysis GPU (CuPy/PyTorch)."""
    if not analysis_gpu_available:
        pytest.skip("Analysis GPU not available")
    # ... test code
```

#### Synthetic Data Fixtures

```python
def test_small_dataset(synthetic_data_small):
    """Test with small synthetic dataset (1K cells)."""
    adata = synthetic_data_small.adata
    # adata is AnnData with 1000 cells, 10 markers, 3 clusters

def test_medium_dataset(synthetic_data_medium):
    """Test with medium synthetic dataset (10K cells)."""
    adata = synthetic_data_medium.adata
    # adata is AnnData with 10000 cells, 25 markers, 5 clusters

def test_large_dataset(synthetic_data_large):
    """Test with large synthetic dataset (100K cells)."""
    adata = synthetic_data_large.adata
    # adata is AnnData with 100000 cells, 50 markers, 8 clusters
```

#### Mask Fixtures

```python
def test_simple(simple_masks):
    """Test with simple masks (10 cells)."""
    cell_mask, nucleus_mask = simple_masks

def test_medium(medium_masks):
    """Test with medium masks (100 cells)."""
    cell_mask, nucleus_mask = medium_masks

def test_large(large_masks):
    """Test with large masks (1000 cells)."""
    cell_mask, nucleus_mask = large_masks
```

## Numerical Tolerance Guidelines

When comparing GPU and CPU results for analysis operations:

### Why Tolerances Are Needed

GPU and CPU may produce slightly different results due to:
1. **Different operation order**: Parallel GPU execution vs sequential CPU
2. **Different precision**: GPU may use different precision for intermediate results
3. **Different libraries**: CUDA math vs CPU BLAS/LAPACK

### Recommended Tolerances by Operation

| Operation | Default rtol | Default atol | Notes |
|-----------|--------------|--------------|-------|
| Array arithmetic | 1e-6 | 1e-7 | Simple operations should be very close |
| Matrix operations | 1e-5 | 1e-6 | Standard for most comparisons |
| Normalization (log1p) | 1e-5 | 1e-6 | Standard tolerance |
| Distance/similarity | 1e-5 | 1e-6 | Matrix operations |
| Clustering (k-NN, Leiden) | 1e-4 | 1e-5 | Iterative algorithms, looser |
| UMAP | 1e-3 | 1e-4 | Stochastic algorithm, very loose |

### Example: Choosing Tolerance

```python
# For simple operations (normalization)
compare_cpu_gpu_results(cpu_norm, gpu_norm, rtol=1e-5, atol=1e-6)

# For iterative algorithms (clustering)
compare_cpu_gpu_results(cpu_clusters, gpu_clusters, rtol=1e-4, atol=1e-5)

# For stochastic algorithms (UMAP) - may need even looser
compare_cpu_gpu_results(cpu_umap, gpu_umap, rtol=1e-3, atol=1e-4)
```

### When to Adjust Tolerance

If a test fails with "arrays not close enough":

1. **Check magnitude of differences**: Print `np.abs(cpu - gpu).max()`
2. **Verify correctness**: Is GPU implementation algorithmically correct?
3. **Document choice**: If increasing tolerance, add comment explaining why

```python
# Example: Documenting tolerance adjustment
# Looser tolerance needed because parallel reduction order differs
# Verified differences are <0.01% and scientifically negligible
compare_cpu_gpu_results(cpu_result, gpu_result, rtol=1e-4)
```

## Mocking GPU for Testing Fallback Logic

### Mock GPU Unavailable

Test behavior when GPU is requested but unavailable:

```python
from unittest.mock import patch

def test_fallback_to_cpu():
    """Test graceful fallback when GPU unavailable."""
    with patch('aegle_analysis.gpu_utils.is_gpu_available', return_value=False):
        # Request GPU (should fall back to CPU)
        result = analyze(data, use_gpu=True, fallback_to_cpu=True)
        assert result is not None
```

### Mock GPU Error

Test error handling when GPU operations fail:

```python
def test_gpu_error_handling():
    """Test handling of GPU errors."""
    with patch('aegle_analysis.some_gpu_func', side_effect=RuntimeError("GPU error")):
        # Should catch error and fall back or raise informative error
        result = analyze(data, use_gpu=True, fallback_to_cpu=True)
        assert result is not None
```

### Mock GPU Memory Info

Test memory-dependent logic:

```python
def test_with_limited_memory():
    """Test behavior with limited GPU memory."""
    mock_mem = {'total_mb': 4000, 'free_mb': 500, 'used_mb': 3500}

    with patch('aegle_analysis.gpu_utils.get_gpu_memory_info', return_value=mock_mem):
        # Should detect insufficient memory and adjust or fall back
        result = process_large_data(data, use_gpu=True)
        assert result is not None
```

## Common Test Patterns for Analysis GPU

### Pattern 1: Test Backend Selection

```python
from aegle_analysis.gpu_utils import select_compute_backend
import logging

def test_backend_selection():
    """Test that backend selection works correctly."""
    logger = logging.getLogger('test')

    # Request CPU
    backend = select_compute_backend(use_gpu=False, fallback_to_cpu=True, logger=logger)
    assert backend == "cpu"

    # Request GPU (with fallback)
    backend = select_compute_backend(use_gpu=True, fallback_to_cpu=True, logger=logger)
    assert backend in ["gpu", "cpu"]  # Depends on availability
```

### Pattern 2: Test GPU/CPU Equivalence

```python
def test_normalization_gpu_cpu_equivalence(analysis_gpu_available):
    """Test GPU and CPU normalization produce same results."""
    if not analysis_gpu_available:
        pytest.skip("Analysis GPU not available")

    data = create_test_data()

    # Run on CPU
    result_cpu = normalize(data, use_gpu=False)

    # Run on GPU
    result_gpu = normalize(data, use_gpu=True)

    # Compare
    compare_cpu_gpu_results(result_cpu, result_gpu, rtol=1e-5)
```

### Pattern 3: Test Fallback Behavior

```python
def test_fallback_when_gpu_disabled():
    """Test CPU fallback when GPU explicitly disabled."""
    result = analyze(data, use_gpu=False)
    assert result is not None
    # Should use CPU even if GPU available

def test_fallback_when_gpu_unavailable():
    """Test CPU fallback when GPU unavailable."""
    with patch('aegle_analysis.gpu_utils.is_gpu_available', return_value=False):
        result = analyze(data, use_gpu=True, fallback_to_cpu=True)
        assert result is not None
        # Should automatically fall back to CPU
```

### Pattern 4: Parametrized Testing

```python
@pytest.mark.parametrize("use_gpu", [False, True])
def test_process_both_backends(use_gpu, analysis_gpu_available):
    """Test processing works with both CPU and GPU."""
    if use_gpu and not analysis_gpu_available:
        pytest.skip("GPU not available")

    result = process(data, use_gpu=use_gpu)
    assert result is not None
    validate_result(result)
```

## Best Practices for Analysis GPU Testing

1. **Always test CPU path**: Analysis GPU may not be available in all environments
2. **Use fixtures**: Leverage `analysis_gpu_available` and synthetic data fixtures
3. **Document tolerances**: Explain why specific rtol/atol values are used
4. **Test backend selection**: Verify `select_compute_backend()` logic is correct
5. **Mock for edge cases**: Use mocks to test GPU unavailable scenarios
6. **Separate concerns**: Test GPU utilities separately from actual analysis algorithms
7. **Validate outputs**: Don't just check GPU==CPU, also verify correctness

## Integration with CI/CD

### Testing Without GPU

The test infrastructure gracefully skips GPU tests when GPU unavailable:

```bash
# Run on system without GPU - GPU tests will be skipped
pytest tests/test_analysis_gpu_availability.py -v
# Output: "SKIPPED (GPU not available)"
```

### Testing With GPU

On systems with GPU, all tests run:

```bash
# Run on system with GPU - all tests execute
pytest tests/test_analysis_gpu_availability.py -v
# All tests should pass
```

### Example CI Configuration

```yaml
# .github/workflows/test.yml
test_analysis_cpu:
  runs-on: ubuntu-latest
  steps:
    - name: Test CPU analysis and fallback
      run: pytest tests/test_analysis_gpu_availability.py -v

test_analysis_gpu:
  runs-on: [self-hosted, gpu]
  steps:
    - name: Test GPU analysis
      run: pytest tests/test_analysis_gpu_availability.py -v
```

## Troubleshooting Analysis GPU Tests

### Issue: Tests Skip Even With GPU Available

**Cause**: CuPy or PyTorch not installed, or GPU not detected

**Solutions**:
1. Check installation: `python -c "import cupy; print(cupy.__version__)"`
2. Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify detection: `pytest tests/test_analysis_gpu_availability.py::TestIsGPUAvailable::test_returns_boolean -v -s`

### Issue: Tolerance Failures

**Cause**: GPU and CPU results differ beyond tolerance

**Solutions**:
1. Print differences: `print(np.abs(gpu - cpu).max())`
2. Check if systematic: Are all values off by similar amount?
3. Increase tolerance if scientifically acceptable: `rtol=1e-4`
4. Document why: Add comment explaining tolerance choice

### Issue: Mocking Doesn't Work

**Cause**: Wrong module path or caching

**Solutions**:
1. Verify path: `'aegle_analysis.gpu_utils.is_gpu_available'` not `'aegle.gpu_utils'`
2. Reset cache: Set `_GPU_AVAILABLE = None` in test setup
3. Use `return_value` for simple mocks, `side_effect` for exceptions

## Summary: Analysis GPU Testing

The analysis GPU testing infrastructure provides:

- **22 tests** in `test_analysis_gpu_availability.py`
- **>95% coverage** of `aegle_analysis.gpu_utils` functions
- **Reusable utilities** in `tests/utils/gpu_test_helpers.py`
- **Shared fixtures** in `tests/conftest.py`
- **Clear guidelines** for tolerance selection and mocking
- **CI/CD compatibility** with automatic skipping when GPU unavailable

Key differences from segmentation GPU testing:
- Supports both **CuPy and PyTorch** backends
- Focuses on **backend selection** logic
- Prepares for Phase 5 GPU-accelerated analysis algorithms
- Uses **looser tolerances** for iterative/stochastic algorithms
