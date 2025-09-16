# Segmentation Visualization Features

This document describes the new segmentation visualization features added to the PhenoCycler pipeline.

## Overview

The segmentation visualization module (`aegle/visualization_segmentation.py`) provides comprehensive tools for visualizing and analyzing cell segmentation results. These features help in:

1. **Quality Assessment**: Visualize segmentation quality and identify potential errors
2. **Statistical Analysis**: Analyze cell morphology and spatial distributions
3. **Method Comparison**: Compare different segmentation approaches
4. **Error Detection**: Highlight problematic segmentations

## Features

### 1. Segmentation Overlay Visualization

Creates an overlay of cell and nucleus boundaries on the original image.

**Function**: `create_segmentation_overlay()`

**Features**:
- Red boundaries for nuclei
- Green boundaries for cells
- Yellow highlighting for unmatched nuclei
- Cyan highlighting for unmatched cells
- Optional cell ID labels

**Output**: `segmentation_overlay_patch_*.png`

### 2. Error Detection Visualization

Highlights potential segmentation errors based on morphological criteria.

**Function**: `visualize_segmentation_errors()`

**Error Types**:
- **Oversized cells**: Cells larger than threshold (default: 2000 pixels²)
- **Undersized cells**: Cells smaller than threshold (default: 50 pixels²)
- **High eccentricity**: Elongated cells (eccentricity > 0.95)
- **Unmatched**: Cells without matching nuclei or vice versa

**Output**: `segmentation_errors_patch_*.png`

### 3. Quality Heatmaps

Spatial visualization of segmentation quality metrics across patches.

**Function**: `create_quality_heatmaps()`

**Metrics**:
- Cell density (cells per mm²)
- Mean cell area
- Matched fraction (nucleus-cell matching rate)

**Output**: `quality_heatmap_*.png`

### 4. Morphology Statistics

Comprehensive analysis of cell morphological properties.

**Function**: `plot_cell_morphology_stats()`

**Analyzed Properties**:
- Cell area distribution
- Cell eccentricity (shape elongation)
- Cell solidity (convexity measure)
- Nucleus/cell area ratio

**Output**: `cell_morphology_stats.png`

### 5. Segmentation Comparison

Side-by-side comparison of different segmentation methods or parameters.

**Function**: `create_segmentation_comparison()`

**Use Cases**:
- Compare original vs. repaired segmentations
- Compare different segmentation algorithms
- Evaluate parameter tuning effects

## Configuration

Enable segmentation visualization in your pipeline configuration:

```yaml
visualization:
  visualize_segmentation: true
  show_segmentation_errors: true  # Optional: enable error visualization
```

## Output Structure

When enabled, visualization outputs are saved to:

```
<output_dir>/
└── visualization/
    └── segmentation/
        ├── segmentation_overlay_patch_0.png
        ├── segmentation_errors_patch_0.png
        ├── quality_heatmap_cell_density.png
        ├── quality_heatmap_mean_cell_area.png
        ├── quality_heatmap_matched_fraction.png
        └── cell_morphology_stats.png
```

## Testing

Use the test script to visualize existing segmentation results:

```bash
python scripts/test_segmentation_viz.py <output_directory>
```

This will create a `test_segmentation_viz` folder with example visualizations.

## Customization

### Adjusting Error Thresholds

```python
# In pipeline configuration or code
thresholds = {
    'oversized': 1500,      # pixels²
    'undersized': 100,      # pixels²
    'high_eccentricity': 0.9,
    'low_solidity': 0.7
}
```

### Adding New Metrics

The module is designed to be extensible. To add new metrics:

1. Add calculation in `create_quality_heatmaps()`
2. Add visualization in `plot_cell_morphology_stats()`
3. Add error type in `visualize_segmentation_errors()`

## Best Practices

1. **For Large Images**: Visualization can be memory-intensive. Consider:
   - Processing patches individually
   - Downsampling for overview visualizations
   - Limiting the number of cells with IDs shown

2. **For Publication**: 
   - Use high DPI (300+) for output
   - Adjust color schemes for colorblind accessibility
   - Add scale bars where appropriate

3. **For Debugging**:
   - Enable `show_ids=True` for small regions
   - Use error visualization to identify systematic issues
   - Compare multiple parameter settings

## Future Enhancements

Planned features include:
- Interactive visualization with Napari
- 3D visualization for z-stacks
- Time-series analysis for dynamic imaging
- Machine learning-based quality scoring
- Automated report generation
