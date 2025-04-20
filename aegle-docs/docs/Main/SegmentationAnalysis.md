---
sidebar_position: 3
---

# Segmentation Analysis

This document describes the segmentation analysis module implemented in the Aegle pipeline. The segmentation analysis provides comprehensive evaluation of cell and nucleus segmentation results, intensity distributions across cellular compartments, and spatial density metrics.

## Overview

The segmentation analysis module performs in-depth analysis on segmentation masks and intensity data to evaluate cell and nucleus segmentation quality, characterize marker intensity distributions, and quantify spatial density patterns. This module is particularly important for assessing the quality of the segmentation pipeline and for understanding the spatial organization of cells within the tissue.

## Key Components

### 1. Mask Analysis
The module analyzes multiple types of segmentation masks:
- **Original cell mask**: Raw cell segmentation results
- **Original nucleus mask**: Raw nucleus segmentation results
- **Matched cell mask**: Cells successfully matched to nuclei
- **Matched nucleus mask**: Nuclei successfully matched to cells
- **Unmatched nucleus mask**: Nuclei that could not be matched to cells

### 2. Intensity Analysis
For each marker channel in the image, the module:
- Calculates mean intensity per object for each mask type
- Computes intensity statistics (mean, median, standard deviation)
- Quantifies bias metrics comparing original and matched masks
- Visualizes intensity distributions across different mask types

### 3. Spatial Density Analysis
The module analyzes the spatial arrangement of cells by:
- Computing global cell density metrics (objects per square mm)
- Calculating local density metrics in moving windows
- Generating density distribution visualizations
- Comparing density metrics across different mask types

## Usage

The segmentation analysis is automatically executed as part of the main pipeline when specified in the configuration. The primary entry point is the `run_segmentation_analysis` function from the `aegle.segmentation_analysis.segmentation_analysis` module:

```python
from aegle.segmentation_analysis.segmentation_analysis import run_segmentation_analysis

# Run segmentation analysis on codex_patches object
run_segmentation_analysis(codex_patches, config, args)
```

## Input Requirements

- **codex_patches**: A `CodexPatches` object containing:
  - `original_seg_res_batch`: Original segmentation masks
  - `repaired_seg_res_batch`: Repaired segmentation masks after cell-nucleus matching
  - `all_channel_patches`: Image data across all channels

- **config**: Configuration dictionary with parameters for:
  - Image resolution (microns per pixel)
  - Visualization settings
  - Analysis parameters

- **args**: Command-line arguments, primarily for output directory specification

## Output Files

Segmentation analysis generates the following outputs:

- **codex_patches_segmentation_analysis.pickle**: Serialized analysis results
- **Channel intensity distribution plots**: Visualization of marker intensity distribution for each channel across different mask types
- **Cell density visualizations**: Histograms and density plots of spatial density metrics
- **Channel intensity bias analysis**: Comparison of intensity statistics between original and matched masks
- **Updated metadata**: The patch metadata is updated with density metrics

## Performance Considerations

The intensity analysis is computationally intensive as it processes large arrays of pixel-level data. To improve performance, the module uses parallel processing via Python's `ProcessPoolExecutor` for calculating mean intensities across different mask types. However, this can lead to high memory usage for large datasets.

For very large images, consider:
- Processing fewer channels at a time
- Reducing the number of parallel processes
- Setting a limit on the number of objects analyzed

If memory errors or process terminations occur, adjust the parallelization strategy or process smaller batches of data.

## Relationship to Other Modules

The segmentation analysis module operates on the outputs of:
- The cell segmentation module which produces the initial segmentation masks
- The cell-nucleus matching algorithm which creates the matched masks

Its outputs inform:
- Quality assessment of the segmentation pipeline
- Downstream cell profiling and phenotyping

