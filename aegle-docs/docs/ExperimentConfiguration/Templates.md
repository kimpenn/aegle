---
sidebar_position: 4
---

# Templates and Design Tables

This section covers both YAML configuration templates and CSV experiment design tables, which work together to generate experiment-specific configurations. The templates define the structure and defaults, while the design tables specify experiment-specific parameters.

## Template Types

### Preprocessing Template (`preprocess_template.yaml`)

The preprocessing template handles the initial stages of data processing:

- **Tissue Extraction Parameters**: Configure tissue detection and extraction settings
- **BF Tools Configuration**: Set up BioFormats tools for image reading
- **Basic Data Paths**: Define input data locations and file structures

**Key Sections**:
- `data`: Input file paths and metadata
- `bftools`: BioFormats tools configuration
- `tissue_extraction`: Tissue detection parameters

### Main Template (`main_template.yaml`)

The main analysis template covers the core image processing and analysis steps:

- **Image Segmentation Parameters**: Configure cell segmentation models and settings
- **Patch Extraction Settings**: Define image patching and sampling strategies
- **Visualization Options**: Control output visualizations and quality control images
- **Quality Control Thresholds**: Set filtering criteria for patches and cells

**Key Sections**:
- `data`: Image files and antibody definitions
- `channels`: Nuclear and cell marker channel specifications
- `patching`: Image patch extraction parameters
- `visualization`: Output visualization settings
- `segmentation`: Cell segmentation model configuration
- `patch_qc`: Quality control thresholds
- `evaluation`: Metrics computation settings

### Analysis Template (`analysis_template.yaml`)

The analysis template focuses on downstream statistical analysis and visualization:

- **Downstream Analysis Parameters**: Configure statistical analysis methods
- **Statistical Computation Settings**: Define metrics and statistical tests
- **Output Format Configuration**: Specify result formats and export options

**Key Sections**:
- `analysis`: Statistical analysis parameters
- `visualization`: Advanced plotting and visualization options
- `output`: Result export and format specifications

## Template Structure and Mapping

All templates follow a hierarchical YAML structure that maps directly to the CSV column naming convention using `::` separators.

### Example Template Structure

```yaml
# Top-level configuration
exp_id: default

# Data configuration
data:
  file_name: path/to/image.tiff
  antibodies_file: path/to/antibodies.tsv
  image_mpp: 0.5

# Channel configuration
channels:
  nuclear_channel: DAPI
  wholecell_channel: 
    - Pan-Cytokeratin

# Nested configuration example
patching:
  split_mode: full_image
  patch_height: -1
  patch_width: -1
  overlap: 0.1
```

### CSV to YAML Mapping

The CSV column names map to YAML structure using the `::` separator:

- `data::file_name` → `data.file_name`
- `channels::nuclear_channel` → `channels.nuclear_channel`
- `patching::split_mode` → `patching.split_mode`
- `visualization::visualize_patches` → `visualization.visualize_patches`

## Experiment Design Tables (CSV)

The CSV design tables are exported from the [Google Sheet](https://docs.google.com/spreadsheets/d/1JcabHe3aobjeRa4V0mNSmC2rnh5jJ3h4_v0jnX2vkOE/edit?usp=sharing) and stored in the `exps/csvs/` directory. Each CSV file corresponds to a specific pipeline component and contains experiment-specific parameters.

### Design Table Structure

#### Google Sheet Organization

The experiments are designed in a shared Google Sheet with separate tabs for each pipeline component:

- **Tab Naming Convention**: `{component}_{set_name}`
  - `preprocess_ft` → exports to `preprocess_ft.csv`
  - `main_ft` → exports to `main_ft.csv`
  - `analysis_ft` → exports to `analysis_ft.csv`

#### CSV File Structure

Each CSV file contains:
- **Header Row**: Column names using `::` notation for nested parameters
- **Data Rows**: One row per experiment with specific parameter values
- **Required Column**: `exp_id` - unique identifier for each experiment

### Pipeline-Specific Design Tables

#### Preprocessing Design Table (`preprocess_x.csv`)

**Purpose**: Configure tissue extraction and preprocessing parameters

**Parameters**:
- `exp_id`: Experiment ID
- `data::file_name`: Path to the raw image file (.qptiff)
- `tissue_extraction::manual_mask_json`: Path to manual annotation file. If this file is not provided, the automatic tissue extraction will be used.
- `tissue_extraction::visualize`: Enable/disable visualization output
- `tissue_extraction::n_tissue`: Number of tissue regions to extract for automatic tissue extraction. Set to -1 for manual tissue extraction.
- `tissue_extraction::downscale_factor`: Downscale factor for the image for automatic tissue extraction. Set to -1 for manual tissue extraction.
- `tissue_extraction::min_area`: Minimum area of the tissue region for automatic tissue extraction. Set to -1 for manual tissue extraction.

#### Main Design Table (`main_x.csv`)

**Purpose**: Configure segmentation, patching, and cell profiling parameters

**Parameters**:
- `exp_id`: Experiment ID
- `data::file_name`: Path to processed image file (.ome.tiff or .qptiff). If the raw image has multiple tissues, it should be extracted using the tissue region extraction module in the preprocessing step.
- `data::antibodies_file`: Path to antibody definitions (.tsv). It is generated by the antibody data extraction module in the preprocessing step.
- `data::image_mpp`: Microns per pixel for this image. It is determined by PhenoCycler experiment. Most of the time, it is 0.5.
- `data::generate_channel_stats`: Whether to compute and save channel-level statistics (min, max, 95th percentile, etc.)
- `channels::nuclear_channel`: Nuclear marker channel name. Most of the time, it is DAPI.
- `channels::wholecell_channel`: Comma-separated list of cell marker channels. It is determined by the researcher. If multiple channels are provided, they will be merged into one channel by maximum intensity projection.
- `patching::split_mode`: Image splitting strategy. It can be `full_image`, `halves`, `quarters`, or `patches`. `full_image` means using the entire image as a single patch. `halves` means splitting the image into two halves vertically or horizontally. `quarters` means splitting the image into four quarters. `patches` means using small patches with overlap for segmentation.
- `patching::split_direction`: Image splitting direction for split_mode `halves`. It can be `vertical` or `horizontal`.
- `patching::patch_height`: Height of each patch (in pixels) for split_mode `patches`.
- `patching::patch_width`: Width of each patch (in pixels) for split_mode `patches`.
- `patching::overlap`: Overlap fraction between adjacent patches (0.1 means 10% overlap) for split_mode `patches`.
- `visualization::visualize_whole_sample`: Whether to save an RGB visualization of the entire sample based on the nuclear channel and the whole-cell channel. By default, it is False in the template to save time of processing.
- `visualization::downsample_factor`: How to downsample the visualization image. -1 means not valid. Other than -1, it should be a positive integer.
- `visualization::enhance_contrast`: Whether to apply contrast enhancement to the visualization by adaptive histogram equalization.
- `visualization::visualize_patches`: Whether to save RGB visualizations of all patches
- `visualization::save_all_channel_patches`: Whether to save the raw multi-channel patches to disk. By default, it is False in the template to save space.
- `visualization::visualize_segmentation`: Whether to visualize and save the segmentation mask overlay (after segmentation). This component is disabled in the pipeline to save time.
- `patch_qc::non_zero_perc_threshold`: Minimum fraction of non-zero pixels required for a patch to be considered valid
- `patch_qc::mean_intensity_threshold`: Minimum mean intensity for the patch to be considered informative
- `patch_qc::std_intensity_threshold`: Minimum standard deviation required to avoid marking patches as too "flat"
- `segmentation::model_path`: Path to segmentation model
- `segmentation::save_segmentation_images`: Whether to save segmentation masks as images
- `segmentation::save_segmentation_pickle`: Whether to pickle the entire codex_patches object (containing segmentation results, etc.)
- `segmentation::segmentation_analysis`: Whether to run segmentation analysis
- `testing::data_disruption::type`: Type of data disruption
- `testing::data_disruption::level`: Level of data disruption
- `testing::data_disruption::save_disrupted_patches`: Whether to save disrupted patches
- `evaluation::compute_metrics`: Whether to compute metrics

#### Analysis Design Table (`analysis_x.csv`)

**Purpose**: Configure downstream statistical analysis and visualization

**Parameters**:
- `exp_id`: Experiment ID
- `analysis::data_dir`: Directory containing the CSV or expression files for this analysis
- `analysis::patch_index`: Which patch index to analyze (if you have multiple patches)
- `analysis::skip_viz`: Whether to skip all plotting steps (for faster processing)
- `analysis::clustering_resolution`: Resolution parameter used by Leiden clustering
- `analysis::norm_method`: Normalization method

### Design Table Best Practices

#### Column Naming

1. **Use Descriptive Names**: Column names should clearly indicate their purpose
2. **Follow Hierarchy**: Use `::` to represent nested YAML structure
3. **Consistent Naming**: Maintain consistency across different design tables
4. **Avoid Spaces**: Use underscores instead of spaces in parameter names

#### Data Format Guidelines

1. **Boolean Values**: Use `TRUE/FALSE` (case-insensitive)
2. **Lists**: Use comma-separated values without spaces after commas
3. **File Paths**: Use absolute paths for reliability
4. **Null Values**: Use `None` for null/empty values
5. **Numeric Values**: Use plain numbers without quotes

#### Example Design Table Workflow

1. **Design in Google Sheet**: 
   - Create/edit experiments in the shared Google Sheet
   - Use separate tabs for different pipeline components
   - Validate data formats and file paths

2. **Export to CSV**:
   - Export each tab as a separate CSV file
   - Save to `exps/csvs/` with appropriate filename
   - Ensure proper encoding (UTF-8)

3. **Generate Configurations**:
   - Update `config_generator.py` with correct experiment set name
   - Run generator to create YAML configurations
   - Validate generated files before pipeline execution

## Customization and Extension

### Adding New Parameters

To add new parameters to the system:

1. **Update Template File**: Add the new parameter with default value in the appropriate YAML template
2. **Update Design Table**: Add corresponding column in the Google Sheet and export to CSV
3. **Update Generator Logic**: Add type conversion logic in `config_generator.py` if needed
4. **Test Generation**: Verify the parameter is correctly processed and appears in generated configs

### Creating New Experiment Sets

To create a new experiment set:

1. **Create New Tab**: Add a new tab in the Google Sheet following naming convention `{component}_{set_name}`
2. **Design Experiments**: Add experiment rows with appropriate parameters“
3. **Export CSV**: Save the tab as `{set_name}.csv` in the `exps/csvs/` directory
4. **Update Generator**: Set `experiment_set_name = "{set_name}"` in `config_generator.py`
5. **Generate Configs**: Run the generator to create configuration files

### Parameter Validation

Templates serve as validation references for:
- **Required Parameters**: Ensure all necessary parameters are present
- **Default Values**: Provide fallback values for optional parameters
- **Structure Validation**: Maintain consistent YAML structure across experiments
