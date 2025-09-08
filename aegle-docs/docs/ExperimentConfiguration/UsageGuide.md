---
sidebar_position: 2
---

# Usage Guide

This guide walks you through the process of using the experiment configuration system to generate and manage configurations for your CODEX analysis experiments.

## Quick Start

### 1. Set Script Parameters

Modify the following variables in `config_generator.py`:

```python
# Experiment set name (corresponds to CSV file name)
# It will be the directory name of the generated configuration files.
experiment_set_name = "main_ft"

# Analysis step: "preprocess", "main", "analysis"
# It will select the template file to use.
analysis_step = "main"

# Base path of Experiment Configuration Module
# We have been using absolute path of the `exps` directory in the codebase.
base_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps"
```

### 2. Create Experiment Design Table (CSV)

Create an experiment design table in the `csvs/` directory for each experiment set. Currently, we design the experiments in this [Google Sheet](https://docs.google.com/spreadsheets/d/1JcabHe3aobjeRa4V0mNSmC2rnh5jJ3h4_v0jnX2vkOE/edit?usp=sharing) and export as .csv files.

The format of the .csv file is as follows:

```csv
exp_id,data::file_name,data::antibodies_file,channels::nuclear_channel,channels::wholecell_channel,...
D10_0,/path/to/image.tiff,/path/to/antibodies.tsv,DAPI,"Pan-Cytokeratin,E-cadherin",...
D10_1,/path/to/image2.tiff,/path/to/antibodies2.tsv,DAPI,"Pan-Cytokeratin,E-cadherin",...
```

**Column Name Format Description**:
- Use `::` separator to represent nested configuration levels
- Example: `data::file_name` corresponds to `data.file_name` in YAML
- Example: `channels::wholecell_channel` corresponds to `channels.wholecell_channel`

There are three major components of Aegle Pipeline so there are three different types of Design Tables. They are marked as prefix in the tab name of the Google Sheet as in `preprocess_X`, `main_X`, `analysis_X`. Such as `preprocess_ft`, `main_ft`, `analysis_ft`.

### 3. Check Configuration Templates

Ensure the corresponding template files exist located at `exps/`:
- `preprocess_template.yaml` - Preprocessing step template
- `main_template.yaml` - Main analysis step template  
- `analysis_template.yaml` - Analysis step template

They serve as the default configuration templates for the configuration generator. And also as a reference about the output of the configuration generator.

### 4. Run the Generator
Go to the base path of Experiment Configuration Module and run the generator.
```bash
cd /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps
python config_generator.py
```

## Parameter Types

The configuration generator automatically converts CSV values into appropriate Python types before writing them into YAML configuration files. This ensures consistency between human-readable CSV inputs and machine-readable configuration files.

### Type Conversions

| Parameter Type | CSV Format        | Converted Type | Example |
|----------------|-------------------|----------------|---------|
| List           | Comma-separated   | `List[str]`    | `"DAPI,Pan-Cytokeratin"` → `["DAPI", "Pan-Cytokeratin"]` |
| Integer        | Number            | `int`          | `"512"` → `512` |
| Float          | Decimal           | `float`        | `"0.5"` → `0.5` |
| Boolean        | `TRUE` / `FALSE`  | `bool`         | `"TRUE"` → `true`, `"FALSE"` → `false` |
| Null           | `"None"`          | `null`         | `"None"` → `null` |
| Python Expression | String         | Evaluated type | `"[128, 64]"` → `[128, 64]` (via `ast.literal_eval`) |

If none of the above conversions apply, values remain as strings.

### Conversions Rules
By default, we transform the string from CSV to float Python and write it into YAML.

The following are the rules for some keys that are explicitly handled with custom logic:

- **List Parameters**  
  - `wholecell_channel`: `"A,B,C"` → `["A", "B", "C"]`  
  - `assign_sizes`: `"0.1,0.2"` → `[0.1, 0.2]` (list of floats)  

- **Integer Parameters**  
  - `patch_width`, `patch_height`, `patch_index`, `n_tissue`, `downscale_factor`, `min_area`, `output_dim`  

- **List of Integers**  
  - `hidden_dims`: evaluated with `ast.literal_eval`, e.g. `"[128, 64]"` → `[128, 64]`  

- **Boolean Flags**  
  - `generate_channel_stats`, `visualize_whole_sample`, `visualize_patches`, `save_all_channel_patches`,  
    `visualize_segmentation`, `save_segmentation_images`, `save_segmentation_pickle`, `save_disrupted_patches`,  
    `compute_metrics`, `skip_viz`, `enhance_contrast`, `visualize`, `segmentation_analysis`  


