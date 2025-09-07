---
sidebar_position: 5
---

# Pipeline Components

This document provides detailed information about the core components of the Main analysis pipeline. The pipeline transforms raw multiplex imaging data into quantified single-cell profiles through a series of carefully orchestrated processing stages.

## Pipeline Architecture

The Main pipeline consists of two primary processing phases that work together to extract meaningful biological information from large-scale multiplex imaging data:


## Phase A: Full Image Processing

This initial phase handles the complete multiplex image, preparing it for downstream analysis through channel extraction, quality assessment, and optional visualization.

### A1. Image Initialization and Loading

**Purpose**: Establish the foundational image object and load all necessary data

**Process**:
1. **CodexImage Construction**: Creates the primary image object using configuration parameters
2. **Multi-channel Loading**: Loads the complete QPTIFF/TIFF file with all channels
3. **Metadata Integration**: Associates antibody information with channel indices
4. **Memory Management**: Optimizes memory usage for large images

```python
# Core initialization
codex_image = CodexImage(config, args)
```

**Key Configuration Parameters**:
- `data.file_name`: Path to input QPTIFF file
- `data.antibodies_file`: Channel-to-antibody mapping
- `data.image_mpp`: Spatial resolution (microns per pixel)

**Outputs**: Initialized `CodexImage` object with complete channel data

### A2. Channel Statistics Generation

**Purpose**: Calculate comprehensive statistics for quality control and normalization

**Process**:
1. **Intensity Analysis**: Computes min, max, mean, median, 95th percentile per channel
2. **Quality Metrics**: Identifies problematic channels (too bright, too dim, empty)
3. **Statistical Summary**: Generates channel_stats.csv for downstream use

```python
# Generate statistics when enabled
if config['data']['generate_channel_stats']:
    codex_image.calculate_channel_statistics()
```

**Configuration Control**:
```yaml
data:
  generate_channel_stats: true  # Enable comprehensive statistics
```

**Outputs**: 
- `channel_stats.csv`: Per-channel intensity statistics
- Internal statistics for quality control

### A3. Target Channel Extraction

**Purpose**: Select and prepare biologically relevant channels for analysis

**Process**:
1. **Channel Selection**: Identifies nuclear and wholecell marker channels
2. **Channel Merging**: Combines multiple wholecell channels if specified
3. **Intensity Optimization**: Applies channel-specific preprocessing
4. **Validation**: Ensures selected channels contain meaningful signal

```python
# Extract specified channels
codex_image.extract_target_channels()
```

**Configuration Example**:
```yaml
channels:
  nuclear_channel: DAPI
  wholecell_channel: 
    - Pan-Cytokeratin
    - CD31  # Multiple channels merged
```

**Outputs**: 
- `extracted_channel_image`: Nuclear and wholecell channels ready for segmentation
- `extended_extracted_channel_image`: Additional context channels

### A4. Optional Whole Sample Visualization

**Purpose**: Generate overview visualizations for quality assessment

**Process**:
1. **RGB Composition**: Creates composite images from selected channels
2. **Contrast Enhancement**: Applies adaptive contrast for better visibility
3. **Downsampling**: Reduces image size for manageable file sizes
4. **Multi-scale Output**: Generates visualizations at different resolutions

```python
# Generate visualizations when enabled
if config['visualization']['visualize_whole_sample']:
    save_image_rgb(codex_image.extracted_channel_image, output_path)
```

**Configuration Options**:
```yaml
visualization:
  visualize_whole_sample: true
  downsample_factor: -1        # Auto-downsample
  enhance_contrast: true       # Apply contrast enhancement
```

**Outputs**:
- `whole_sample_rgb.png`: Full sample overview
- `whole_sample_enhanced.png`: Contrast-enhanced version

---

## Phase B: Patch-Based Processing

This phase divides the image into manageable patches for scalable analysis, applies quality control, and prepares data for segmentation and cell profiling.

### B1. Image Extension and Padding

**Purpose**: Prepare the image for patch extraction by ensuring proper boundaries

**Process**:
1. **Boundary Analysis**: Calculates required padding based on patch size and overlap
2. **Smart Padding**: Extends image edges using reflection or constant values
3. **Coordinate Mapping**: Maintains spatial relationships between original and extended image
4. **Memory Optimization**: Minimizes memory footprint while ensuring complete coverage

```python
# Extend image for complete patch coverage
codex_image.extend_image()
```

**Configuration Parameters**:
```yaml
patching:
  patch_height: 2000
  patch_width: 2000
  overlap: 0.1  # 10% overlap between patches
```

**Outputs**: Extended image array with proper boundaries for patch extraction

### B2. Patch Generation and Organization

**Purpose**: Create systematic patch grid for parallel processing

**Process**:
1. **Grid Calculation**: Determines optimal patch positions with specified overlap
2. **Patch Extraction**: Cuts image into individual patches with consistent dimensions
3. **Coordinate Tracking**: Records patch positions for spatial reconstruction
4. **Memory Management**: Processes patches in batches to control memory usage

```python
# Initialize patch system
codex_patches = CodexPatches(codex_image, config, args)
```

**Patch Organization**:
- **Sequential Numbering**: Patches numbered 0, 1, 2, ... for consistent reference
- **Spatial Indexing**: Grid coordinates maintained for spatial analysis
- **Overlap Handling**: Overlapping regions properly managed for seamless reconstruction

### B3. Quality Control and Filtering

**Purpose**: Identify and filter patches suitable for biological analysis

**Process**:
1. **Tissue Detection**: Identifies patches containing meaningful tissue content
2. **Noise Assessment**: Flags patches with excessive noise or artifacts
3. **Intensity Analysis**: Evaluates signal quality in nuclear and wholecell channels
4. **Metadata Generation**: Records quality metrics for each patch

```python
# Generate patch metadata with quality metrics
codex_patches.save_metadata()
```

**Quality Metrics**:
```yaml
patch_qc:
  non_zero_perc_threshold: 0.05  # Minimum tissue coverage
  mean_intensity_threshold: 1    # Minimum signal strength
  std_intensity_threshold: 1     # Minimum signal variation
```

**Quality Classifications**:
- **Informative**: High-quality patches suitable for analysis
- **Empty**: Patches with insufficient tissue content
- **Noisy**: Patches with poor signal quality
- **Bad**: Patches failing multiple quality criteria

### B4. Patch Data Persistence

**Purpose**: Save processed patches and metadata for downstream analysis

**Process**:
1. **Array Serialization**: Saves patch arrays in efficient NumPy format
2. **Metadata Export**: Generates CSV files with patch characteristics
3. **Configuration Backup**: Preserves processing parameters for reproducibility
4. **Index Generation**: Creates lookup tables for patch retrieval

```python
# Save patches and associated metadata
codex_patches.save_patches()
codex_patches.save_metadata()
```

**Outputs**:
- `extracted_channel_patches.npy`: Processed patch arrays
- `patches_metadata.csv`: Quality metrics and spatial information
- `patch_index.json`: Spatial indexing information

### B5. Optional Data Augmentation

**Purpose**: Generate test datasets with controlled perturbations for robustness testing

**Process**:
1. **Disruption Selection**: Applies specified noise or artifact types
2. **Controlled Degradation**: Adds realistic imaging artifacts
3. **Comparative Analysis**: Enables testing of algorithm robustness
4. **Validation Dataset**: Creates challenging test cases

```python
# Add controlled disruptions for testing
if config['testing']['data_disruption']['type']:
    disruption_type = config['testing']['data_disruption']['type']
    disruption_level = config['testing']['data_disruption']['level']
    codex_patches.add_disruptions(disruption_type, disruption_level)
```

**Disruption Types**:
- **Gaussian Noise**: Adds random intensity variations
- **Downsampling**: Reduces spatial resolution
- **Artifacts**: Simulates imaging artifacts
- **Intensity Scaling**: Modifies overall brightness

**Configuration Example**:
```yaml
testing:
  data_disruption:
    type: gaussian      # Type of disruption
    level: 2           # Intensity level (1-5)
    save_disrupted_patches: true
```

### B6. Optional Patch Visualization

**Purpose**: Generate visual outputs for quality assessment and debugging

**Process**:
1. **RGB Composition**: Creates false-color images from selected channels
2. **Contrast Optimization**: Applies adaptive contrast enhancement
3. **Grid Layout**: Organizes patches for systematic review
4. **Quality Overlay**: Highlights patches with quality issues

```python
# Generate patch visualizations
if config['visualization']['visualize_patches']:
    save_patches_rgb(codex_patches.extracted_channel_patches, output_dir)
```

**Visualization Options**:
```yaml
visualization:
  visualize_patches: true
  save_all_channel_patches: false  # Save only essential channels
  enhance_contrast: true
```

**Outputs**:
- `patches_rgb/patch-{i}.png`: Individual patch visualizations
- `patch_grid_overview.png`: Complete patch grid overview
- `quality_assessment.png`: Quality control visualization

## Integration with Segmentation Pipeline

The processed patches from Phase B serve as direct inputs to the segmentation pipeline:

1. **Seamless Handoff**: Patch arrays formatted for immediate segmentation
2. **Metadata Preservation**: Quality metrics guide segmentation parameters
3. **Spatial Context**: Coordinate information enables result reconstruction
4. **Batch Processing**: Patches processed in parallel for efficiency

## Performance Considerations

### Memory Management
- **Patch Size Optimization**: Balance between processing efficiency and memory usage
- **Batch Processing**: Process patches in groups to control memory consumption
- **Garbage Collection**: Explicit cleanup of intermediate data structures

### Processing Speed
- **Parallel Extraction**: Multi-core utilization for patch generation
- **Efficient Storage**: Optimized file formats for fast I/O operations
- **Smart Caching**: Avoid redundant computations across patches

### Quality vs. Speed Trade-offs
- **Quality Control Depth**: Balance thoroughness with processing speed
- **Visualization Generation**: Optional outputs that significantly impact processing time
- **Statistical Analysis**: Comprehensive vs. essential metrics

## Next Steps

After completing both processing phases:

1. **Segmentation**: Processed patches ready for cell and nucleus segmentation
2. **Quality Review**: Assess patch quality distribution and adjust parameters if needed
3. **Downstream Analysis**: Proceed to cell profiling and feature extraction
4. **Result Integration**: Combine patch-level results into tissue-level analysis

This comprehensive preprocessing pipeline ensures that multiplex imaging data is optimally prepared for robust downstream analysis while maintaining spatial relationships and biological context.

