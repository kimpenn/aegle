# Analysis Integration Tests

This document describes the analysis integration test framework created to validate the full end-to-end analysis pipeline.

## Overview

The integration tests run the complete analysis workflow (clustering, UMAP, differential expression) on synthetic data to ensure all components work together correctly.

## Test Files

### `/tests/test_analysis_integration.py`
Main integration test file containing 9 test cases across 4 test classes:

#### `TestFullPipelineSmall` (3 tests)
Tests using 1K cell dataset for fast execution:
- `test_full_pipeline_small_completes`: Basic end-to-end test (< 30 sec)
- `test_pipeline_handles_edge_cases`: Edge cases (very low/high resolution)
- `test_pipeline_output_types`: Validate output data types

#### `TestFullPipelineMedium` (2 tests)
Tests using 10K cell dataset for thorough validation:
- `test_full_pipeline_medium_completes`: Comprehensive validation (< 2 min)
- `test_pipeline_discovers_known_clusters`: Verify cluster recovery vs ground truth

#### `TestFullPipelineReproducibility` (2 tests)
Reproducibility validation:
- `test_full_pipeline_reproducibility`: Same seed = identical results
- `test_different_seeds_produce_similar_results`: Different seeds = similar structure

#### `TestFullPipelinePerformance` (2 tests)
Performance benchmarks (marked with `@pytest.mark.slow`):
- `test_small_dataset_completes_quickly`: 1K cells in < 30 sec
- `test_medium_dataset_completes_in_reasonable_time`: 10K cells in < 2 min

### `/tests/utils/integration_helpers.py`
Helper functions for integration testing:

- `run_full_analysis()`: Run complete analysis pipeline on AnnData
- `validate_analysis_outputs()`: Check all expected outputs exist
- `compare_analysis_runs()`: Compare two runs for reproducibility
- `get_*_config()`: Configuration fixtures for tests

## Running the Tests

```bash
# Run all integration tests (takes ~5 min)
python -m pytest tests/test_analysis_integration.py -v

# Run specific test class
python -m pytest tests/test_analysis_integration.py::TestFullPipelineSmall -v

# Run specific test
python -m pytest tests/test_analysis_integration.py::TestFullPipelineSmall::test_full_pipeline_small_completes -v

# Run with performance tests (marked slow)
python -m pytest tests/test_analysis_integration.py -v -m slow
```

## Test Results

**Status**: All 9 tests passing (as of 2025-11-23)

```
tests/test_analysis_integration.py::TestFullPipelineSmall::test_full_pipeline_small_completes PASSED [ 11%]
tests/test_analysis_integration.py::TestFullPipelineSmall::test_pipeline_handles_edge_cases PASSED [ 22%]
tests/test_analysis_integration.py::TestFullPipelineSmall::test_pipeline_output_types PASSED [ 33%]
tests/test_analysis_integration.py::TestFullPipelineMedium::test_full_pipeline_medium_completes PASSED [ 44%]
tests/test_analysis_integration.py::TestFullPipelineMedium::test_pipeline_discovers_known_clusters PASSED [ 55%]
tests/test_analysis_integration.py::TestFullPipelineReproducibility::test_different_seeds_produce_similar_results PASSED [ 66%]
tests/test_analysis_integration.py::TestFullPipelineReproducibility::test_full_pipeline_reproducibility PASSED [ 77%]
tests/test_analysis_integration.py::TestFullPipelinePerformance::test_medium_dataset_completes_in_reasonable_time PASSED [ 88%]
tests/test_analysis_integration.py::TestFullPipelinePerformance::test_small_dataset_completes_quickly PASSED [100%]

Total runtime: 306 seconds (~5 minutes)
```

## What the Tests Validate

1. **Complete workflow execution**: Full pipeline from data loading → clustering → DE analysis
2. **Output structure**: All expected AnnData outputs exist (UMAP, clusters, DE results)
3. **Data types**: Correct dtypes for all outputs (floats, categoricals, etc.)
4. **Reproducibility**: Same random seed produces identical results
5. **Cluster recovery**: Pipeline can discover known cluster structure in synthetic data
6. **Performance**: Completes in reasonable time (< 30s for 1K cells, < 2 min for 10K)
7. **Edge cases**: Handles extreme parameters gracefully

## Integration with Main Pipeline

The tests use synthetic data from `/tests/utils/synthetic_analysis_data.py`, which generates:
- Realistic marker expression patterns
- Known ground truth cluster assignments
- Spatial coordinates and morphology features
- Cell and nucleus segmentation masks

This allows testing the analysis pipeline without needing real experimental data.

## Future Extensions

The integration test framework can be extended to:

1. **GPU vs CPU comparison**: Validate GPU and CPU versions produce equivalent results
2. **Different normalization methods**: Test all normalization options
3. **Batch processing**: Test analysis on multiple patches
4. **Visualization tests**: Validate plot generation (if visualization is enabled)
5. **LLM annotation tests**: Test optional LLM-based cluster annotation

## Notes

- The tests currently skip normalization (synthetic data is pre-normalized) to avoid applying log1p to negative values
- Some warnings about invalid log2 values in DE analysis are expected and don't affect test success
- Performance benchmarks are hardware-dependent but should scale linearly with cell count
