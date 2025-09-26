---
sidebar_position: 5
---

# Configuration Reference

This document provides a comprehensive reference for all configuration parameters used in the Main pipeline. Configuration files use YAML format and are typically generated using the [Experiment Configuration](../ExperimentConfiguration/Intro.md) system.

## Configuration File Structure

The main pipeline configuration follows this hierarchical structure:

```yaml
exp_id: experiment_identifier
data: {...}
channels: {...}
patching: {...}
visualization: {...}
patch_qc: {...}
segmentation: {...}
testing: {...}
evaluation: {...}
```

## Data Configuration

Controls input data sources and basic image properties.

```yaml
data:
  file_name: path/to/image.qptiff
  antibodies_file: path/to/antibodies.tsv
  image_mpp: 0.5
  generate_channel_stats: true
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_name` | string | **required** | Path to the main QPTIFF or TIFF file |
| `antibodies_file` | string | **required** | Path to TSV file with antibody channel definitions |
| `image_mpp` | float | 0.5 | Microns per pixel for spatial calculations |
| `generate_channel_stats` | boolean | true | Whether to compute channel-level statistics |

### Example Antibodies File Format

The `antibodies_file` should be a TSV with the following structure:

```tsv
channel_id	antibody_name
0	DAPI
1	Pan-Cytokeratin
2	CD3
3	CD8
4	CD20
```

## Channel Configuration

Specifies which channels to use for nuclear and cell membrane detection.

```yaml
channels:
  nuclear_channel: DAPI
  wholecell_channel: 
    - Pan-Cytokeratin
    - CD31
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nuclear_channel` | string | **required** | Name of nuclear marker channel |
| `wholecell_channel` | string or list | **required** | Cell membrane marker channel(s) |

### Channel Selection Guidelines

- **Nuclear Channel**: Should be a strong, consistent nuclear marker (e.g., DAPI, Hoechst)
- **Wholecell Channel**: Can be single marker or list of markers that will be merged
- **Channel Names**: Must exactly match entries in the antibodies file

## Patching Configuration

Controls how images are divided into manageable patches for processing.

```yaml
patching:
  split_mode: full_image
  split_direction: vertical
  patch_height: 2000
  patch_width: 2000
  overlap: 0.1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split_mode` | string | "full_image" | How to divide the image |
| `split_direction` | string | "vertical" | Direction for splitting |
| `patch_height` | integer | -1 | Height of each patch in pixels |
| `patch_width` | integer | -1 | Width of each patch in pixels |
| `overlap` | float | 0.1 | Fraction of overlap between patches |

### Split Mode Options

- **`full_image`**: Process entire image as single patch
- **`halves`**: Split image into two parts
- **`quarters`**: Split image into four quadrants  
- **`patches`**: Create regular grid of patches with specified dimensions

### Patch Size Guidelines

| Image Size | Recommended Patch Size | Memory Usage |
|------------|----------------------|--------------|
| < 10K x 10K | `full_image` | Low |
| 10K - 20K | 5000 x 5000 | Medium |
| 20K - 40K | 2000 x 2000 | Medium |
| > 40K | 1000 x 1000 | High |

## Visualization Configuration

Controls output visualization generation.

```yaml
visualization:
  visualize_whole_sample: false
  downsample_factor: -1
  enhance_contrast: true
  visualize_patches: false
  save_all_channel_patches: false
  visualize_segmentation: false
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `visualize_whole_sample` | boolean | false | (Deprecated in main pipeline) Whole-sample overview now generated during preprocess |
| `downsample_factor` | integer | -1 | Downsampling for visualization (-1 = auto) |
| `enhance_contrast` | boolean | true | Apply contrast enhancement |
| `visualize_patches` | boolean | false | Save RGB visualizations of patches |
| `save_all_channel_patches` | boolean | false | Save raw multi-channel patches |
| `visualize_segmentation` | boolean | false | Save segmentation mask overlays |

### Performance Impact

| Setting | Processing Time | Storage Space | Quality |
|---------|----------------|---------------|---------|
| All false | Fastest | Minimal | N/A |
| `visualize_whole_sample: true` | n/a | n/a | Use preprocess overview module instead |
| `visualize_patches: true` | +30-50% | +1-5GB | Detailed |
| All true | +50-100% | +5-20GB | Maximum |

## Quality Control Configuration

Sets thresholds for patch-level quality assessment.

```yaml
patch_qc:
  non_zero_perc_threshold: 0.05
  mean_intensity_threshold: 1
  std_intensity_threshold: 1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `non_zero_perc_threshold` | float | 0.05 | Minimum fraction of non-zero pixels |
| `mean_intensity_threshold` | float | 1 | Minimum mean intensity |
| `std_intensity_threshold` | float | 1 | Minimum standard deviation |

### Quality Control Logic

Patches are marked as:
- **Empty**: `non_zero_perc < non_zero_perc_threshold`
- **Noisy**: `mean_intensity < mean_intensity_threshold`
- **Bad**: Empty OR Noisy
- **Informative**: NOT Bad

## Segmentation Configuration

Controls cell segmentation parameters and outputs.

```yaml
segmentation:
  model_path: /path/to/segmentation/model
  save_segmentation_images: true
  save_segmentation_pickle: true
  segmentation_analysis: false
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | **required** | Path to segmentation model |
| `save_segmentation_images` | boolean | true | Save segmentation masks as images |
| `save_segmentation_pickle` | boolean | true | Pickle segmentation results |
| `segmentation_analysis` | boolean | false | Run detailed segmentation analysis |

### Model Path Requirements

The model path should point to a directory containing:
- Model weights/checkpoints
- Model configuration files
- Any required preprocessing parameters

### Segmentation Analysis

When `segmentation_analysis: true`:
- Generates detailed quality metrics
- Creates intensity distribution plots
- Calculates spatial density metrics
- Produces comprehensive analysis reports

## Testing Configuration

Optional parameters for data disruption testing.

```yaml
testing:
  data_disruption:
    type: gaussian
    level: 3
    save_disrupted_patches: true
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | null | Type of disruption to apply |
| `level` | integer | 1 | Intensity level of disruption |
| `save_disrupted_patches` | boolean | false | Save disrupted patches to disk |

### Disruption Types

- **`null`**: No disruption (default)
- **`gaussian`**: Add Gaussian noise
- **`downsampling`**: Reduce image resolution
- **`artifacts`**: Add imaging artifacts

### Disruption Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| 1 | Minimal disruption | Light testing |
| 2-3 | Moderate disruption | Robustness testing |
| 4-5 | Heavy disruption | Stress testing |

## Evaluation Configuration

Controls metric computation for performance assessment.

```yaml
evaluation:
  compute_metrics: false
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compute_metrics` | boolean | false | Whether to compute evaluation metrics |

## Configuration Examples

### Small Image Processing

```yaml
exp_id: small_sample_test
data:
  file_name: small_sample.qptiff
  antibodies_file: antibodies.tsv
  image_mpp: 0.5
  generate_channel_stats: true

channels:
  nuclear_channel: DAPI
  wholecell_channel: Pan-Cytokeratin

patching:
  split_mode: full_image
  overlap: 0.0

visualization:
  visualize_whole_sample: true
  visualize_patches: true
  enhance_contrast: true

segmentation:
  model_path: /path/to/model
  segmentation_analysis: true
```

### Large-Scale Production

```yaml
exp_id: production_batch
data:
  file_name: large_sample.qptiff
  antibodies_file: antibodies.tsv
  image_mpp: 0.5
  generate_channel_stats: false

channels:
  nuclear_channel: DAPI
  wholecell_channel: 
    - Pan-Cytokeratin
    - CD31

patching:
  split_mode: patches
  patch_height: 2000
  patch_width: 2000
  overlap: 0.1

visualization:
  visualize_whole_sample: false
  visualize_patches: false
  save_all_channel_patches: false

patch_qc:
  non_zero_perc_threshold: 0.1
  mean_intensity_threshold: 2

segmentation:
  model_path: /path/to/model
  save_segmentation_images: false
  segmentation_analysis: false
```

### Robustness Testing

```yaml
exp_id: robustness_test
data:
  file_name: test_sample.qptiff
  antibodies_file: antibodies.tsv
  image_mpp: 0.5

channels:
  nuclear_channel: DAPI
  wholecell_channel: Pan-Cytokeratin

patching:
  split_mode: quarters

visualization:
  visualize_whole_sample: true
  visualize_segmentation: true

testing:
  data_disruption:
    type: gaussian
    level: 2
    save_disrupted_patches: true

evaluation:
  compute_metrics: true
```

## Configuration Validation

### Required Parameters

The following parameters must be specified:
- `exp_id`
- `data.file_name`
- `data.antibodies_file`
- `channels.nuclear_channel`
- `channels.wholecell_channel`
- `segmentation.model_path`

### Validation Commands

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test configuration with dry run
python src/main.py --config_file config.yaml --dry_run
```

## Performance Tuning

### Memory Optimization

```yaml
# For limited memory systems
patching:
  patch_height: 1000
  patch_width: 1000

visualization:
  visualize_patches: false
  save_all_channel_patches: false

segmentation:
  save_segmentation_images: false
```

### Speed Optimization

```yaml
# For faster processing
data:
  generate_channel_stats: false

visualization:
  visualize_whole_sample: false
  enhance_contrast: false

segmentation:
  segmentation_analysis: false
```

### Quality Optimization

```yaml
# For best quality results
patch_qc:
  non_zero_perc_threshold: 0.1
  mean_intensity_threshold: 5

visualization:
  enhance_contrast: true

segmentation:
  segmentation_analysis: true
```

## Integration with Experiment Configuration

This configuration system integrates with the automated configuration generation:

1. **Template Base**: Uses `main_template.yaml` as the base template
2. **CSV Override**: Parameters overridden by values in experiment CSV files
3. **Automatic Generation**: Configurations generated via `config_generator.py`
4. **Batch Processing**: Multiple configurations created for batch experiments

See [Experiment Configuration](../ExperimentConfiguration/Intro.md) for details on automated configuration generation.
