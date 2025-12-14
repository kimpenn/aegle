# Synthetic Analysis Data Generator

## Overview

`synthetic_analysis_data.py` provides factory functions to generate realistic synthetic AnnData objects for testing downstream analysis modules. This module is a **critical dependency** for all Phase 1 analysis tests (clustering, normalization, GPU acceleration, etc.).

## Purpose

- Generate reproducible single-cell expression data with known ground truth cluster labels
- Create realistic segmentation masks (cell and nucleus) for spatial visualizations
- Provide pre-configured dataset sizes for different testing scenarios
- Enable validation of analysis correctness without requiring real data

## Key Features

### 1. Realistic Data Generation

The synthetic data mimics Aegle pipeline output:

- **Expression Matrix**: Normalized marker intensities (mean-centered, unit variance)
- **Cell Metadata**: spatial coordinates (X, Y), morphology features (nucleus_area, cell_area, eccentricity)
- **Cluster Structure**: Known ground truth labels with cluster-specific marker expression patterns
- **Segmentation Masks**: Cell and nucleus masks with 1-indexed labels matching cell IDs

### 2. Predefined Datasets

Four pre-configured dataset sizes for different testing needs:

| Dataset | Cells | Markers | Clusters | Image Size | Use Case |
|---------|-------|---------|----------|------------|----------|
| Small | 1,000 | 10 | 3 | 256×256 | Fast unit tests |
| Medium | 10,000 | 25 | 5 | 512×512 | Integration tests |
| Large | 100,000 | 50 | 8 | 1024×1024 | Performance tests |
| Stress | 500,000 | 50 | 10 | 2048×2048 | Stress tests |

### 3. Reproducibility

All generators use random seeds to ensure reproducible results across test runs.

## Usage

### Basic Usage

```python
from tests.utils.synthetic_analysis_data import get_small_dataset

# Get predefined small dataset
data = get_small_dataset(random_seed=42)

# Access components
adata = data.adata  # AnnData object
cell_mask = data.cell_mask  # Segmentation mask
nucleus_mask = data.nucleus_mask  # Nucleus mask
true_clusters = data.true_clusters  # Ground truth labels
```

### Custom Dataset Creation

```python
from tests.utils.synthetic_analysis_data import create_synthetic_anndata

# Create custom dataset
data = create_synthetic_anndata(
    n_cells=5000,
    n_markers=30,
    n_clusters=4,
    random_seed=42,
    image_shape=(768, 768),
    cluster_separation=2.5,  # Higher = more distinct clusters
    noise_level=0.2,  # Lower = cleaner data
)
```

### Using in Tests

```python
import pytest
from tests.utils.synthetic_analysis_data import get_small_dataset

class TestClustering:
    @pytest.fixture
    def synthetic_data(self):
        """Fixture providing synthetic data for tests."""
        return get_small_dataset(random_seed=42)

    def test_clustering_correctness(self, synthetic_data):
        """Test that clustering recovers known clusters."""
        adata = synthetic_data.adata
        true_labels = synthetic_data.true_clusters

        # Run clustering algorithm
        predicted_labels = my_clustering_function(adata)

        # Validate against ground truth
        agreement = calculate_agreement(true_labels, predicted_labels)
        assert agreement > 0.8  # Should recover most clusters
```

## Data Structure

### AnnData Object (`adata`)

**Expression Matrix (`adata.X`):**
- Shape: `(n_cells, n_markers)`
- Type: `np.ndarray` (float32)
- Normalization: Mean-centered and unit-variance per marker

**Cell Metadata (`adata.obs`):**
```python
adata.obs.columns:
- 'cell_id': int, 1-indexed cell identifier
- 'true_cluster': str, ground truth cluster label
- 'X': float, spatial X coordinate
- 'Y': float, spatial Y coordinate
- 'nucleus_area': float, nucleus area in pixels
- 'cell_area': float, cell area in pixels
- 'eccentricity': float, cell shape eccentricity (0=circle, 1=line)
```

**Marker Metadata (`adata.var`):**
```python
adata.var.columns:
- 'marker_name': str, marker identifier (e.g., 'CD10', 'Marker01')
- 'n_clusters_specific': int, number of clusters where marker is highly expressed
```

**Unstructured Metadata (`adata.uns`):**
```python
adata.uns:
- 'n_clusters': int, number of clusters
- 'random_seed': int, seed used for generation
- 'cluster_separation': float, cluster separation parameter
- 'noise_level': float, noise level parameter
```

### Segmentation Masks

**Cell Mask (`data.cell_mask`):**
- Shape: `(height, width)`
- Type: `np.uint32`
- Values: 0 (background), 1, 2, ..., n_cells (cell IDs)

**Nucleus Mask (`data.nucleus_mask`):**
- Shape: `(height, width)`
- Type: `np.uint32`
- Values: 0 (background), 1, 2, ..., n_cells (nucleus IDs)

**Note:** Cell IDs match AnnData row indices (cell_1 → label 1)

## Cluster-Specific Expression Patterns

The generator creates realistic marker expression patterns:

1. **Primary Markers**: 2-4 markers per cluster with high expression (cluster-specific)
2. **Secondary Markers**: Additional markers shared between some clusters (moderate expression)
3. **Background Noise**: Low-level expression across all markers for all cells
4. **Spatial Clustering**: Cells of the same type are spatially clustered in the image

This ensures that:
- Clustering algorithms can recover known clusters
- Differential expression finds cluster-specific markers
- Visualizations show clear spatial patterns

## Validation

### Visual Validation

Run the validation script to generate preview images:

```bash
python tests/utils/synthetic_analysis_data.py
```

This creates preview images in `tests/debug_synthetic_analysis_previews/`:
- `{dataset}_umap.png`: UMAP visualization showing cluster separation
- `{dataset}_heatmap.png`: Marker expression heatmap by cluster
- `{dataset}_masks.png`: Cell and nucleus segmentation masks
- `{dataset}_morphology.png`: Morphology feature distributions

### Automated Tests

Comprehensive test suite validates all functionality:

```bash
# Run all tests
pytest tests/test_synthetic_analysis_data.py -v

# Skip slow tests (large/stress datasets)
pytest tests/test_synthetic_analysis_data.py -v -m "not slow"

# Run only specific test class
pytest tests/test_synthetic_analysis_data.py::TestSyntheticAnnDataCreation -v
```

## Performance Considerations

| Dataset | Generation Time | Memory Usage | Test Runtime |
|---------|-----------------|--------------|--------------|
| Small | ~1s | ~10 MB | Fast (<5s) |
| Medium | ~5s | ~100 MB | Moderate (~30s) |
| Large | ~2min | ~1 GB | Slow (~2-5min) |
| Stress | ~10min | ~5 GB | Very Slow (~10-20min) |

**Recommendations:**
- Use **small** for fast unit tests
- Use **medium** for integration tests
- Use **large** only for performance benchmarks
- Mark **stress** tests with `@pytest.mark.slow`

## API Reference

### Main Functions

#### `create_synthetic_anndata()`

```python
def create_synthetic_anndata(
    n_cells: int = 1000,
    n_markers: int = 20,
    n_clusters: int = 5,
    random_seed: int = 42,
    *,
    image_shape: Tuple[int, int] = (512, 512),
    cluster_separation: float = 2.0,
    noise_level: float = 0.3,
    add_spatial_coords: bool = True,
    add_morphology_features: bool = True,
) -> SyntheticAnalysisData
```

Generate synthetic single-cell data with known cluster structure.

**Parameters:**
- `n_cells`: Number of cells to generate
- `n_markers`: Number of protein markers (channels)
- `n_clusters`: Number of cell type clusters
- `random_seed`: Random seed for reproducibility
- `image_shape`: Height and width of synthetic image
- `cluster_separation`: How separated clusters should be (higher = more distinct)
- `noise_level`: Amount of random noise (0-1 scale)
- `add_spatial_coords`: Whether to add X, Y coordinates
- `add_morphology_features`: Whether to add morphology features

**Returns:**
- `SyntheticAnalysisData` container with adata, masks, and ground truth

#### `create_synthetic_cell_mask()`

```python
def create_synthetic_cell_mask(
    n_cells: int,
    image_shape: Tuple[int, int] = (512, 512),
    random_seed: int = 42,
    spatial_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]
```

Generate synthetic segmentation masks for spatial plots.

**Returns:**
- `cell_mask`: Cell segmentation mask
- `nucleus_mask`: Nucleus segmentation mask

### Predefined Datasets

#### `get_small_dataset(random_seed=42)`
1,000 cells, 10 markers, 3 clusters, 256×256 image

#### `get_medium_dataset(random_seed=42)`
10,000 cells, 25 markers, 5 clusters, 512×512 image

#### `get_large_dataset(random_seed=42)`
100,000 cells, 50 markers, 8 clusters, 1024×1024 image

#### `get_stress_dataset(random_seed=42)`
500,000 cells, 50 markers, 10 clusters, 2048×2048 image

### Data Container

#### `SyntheticAnalysisData`

```python
@dataclass(frozen=True)
class SyntheticAnalysisData:
    adata: anndata.AnnData  # AnnData object
    cell_mask: np.ndarray  # Cell segmentation mask
    nucleus_mask: np.ndarray  # Nucleus mask
    true_clusters: np.ndarray  # Ground truth cluster labels
    marker_cluster_specificity: dict  # Marker → cluster mappings
```

## Integration with Analysis Pipeline

This module provides test data compatible with the Aegle analysis pipeline:

1. **Data Loading**: AnnData structure matches `aegle_analysis.analysis.create_anndata()` output
2. **Clustering**: Compatible with `aegle_analysis.analysis.run_clustering()`
3. **Visualization**: Masks compatible with `aegle_analysis.visualization.plot_clustering_on_mask()`
4. **Differential Expression**: Ground truth enables validation of DE results

## Troubleshooting

### Tests are too slow

Use smaller datasets or skip slow tests:
```bash
pytest -v -m "not slow"
```

### Memory errors with large datasets

Reduce dataset size or run tests sequentially:
```bash
pytest -v --forked  # Run tests in separate processes
```

### Cluster structure not visible in UMAP

Increase `cluster_separation` parameter:
```python
data = create_synthetic_anndata(
    n_cells=1000,
    n_markers=20,
    n_clusters=5,
    cluster_separation=3.0,  # Increase from default 2.0
)
```

## Future Enhancements

Potential improvements for future development:

1. **Batch effects**: Add synthetic batch effects for batch correction testing
2. **Cell-cell interactions**: Spatial proximity patterns between cell types
3. **Rare populations**: Support for imbalanced cluster distributions
4. **Time-series**: Sequential snapshots for temporal analysis
5. **Multi-modal**: Combined protein and RNA expression patterns

## See Also

- `tests/utils/synthetic_data_factory.py` - Synthetic data for main pipeline testing
- `tests/test_synthetic_analysis_data.py` - Comprehensive test suite
- `aegle_analysis/analysis/clustering.py` - Real clustering implementation
- `scratch/PLANS_analysis_gpu_optimization.md` - Analysis GPU optimization plan
