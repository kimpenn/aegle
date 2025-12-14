---
sidebar_position: 3
---

# Outputs

The configuration generator produces configuration files and also prints an experiment list for downstream batch execution.

## Configuration File Structure

The generated configuration files are organized in the following structure:

```
configs/
└── {analysis_step}/          # Analysis step (preprocess/main/analysis)
    └── {experiment_set_name}/  # Experiment set name
        ├── {exp_id_1}/         # Experiment ID
        │   └── config.yaml
        ├── {exp_id_2}/
        │   └── config.yaml
        └── ...
```

### Example Configuration Directory Structure

For a main analysis experiment set named `main_ft`, the structure would look like:

```
configs/
└── main/
    └── main_ft/
        ├── D10_0/
        │   └── config.yaml
        ├── D10_1/
        │   └── config.yaml
        ├── D10_2/
        │   └── config.yaml
        └── D10_3/
        │   └── config.yaml
```

## Experiment List for Bash Scripts

The configuration generator outputs an experiment list for bash scripts. For example:

```bash
experiments=(
  "D10_0"
  "D10_1"
  "D10_2"
  "D10_3"
)
```

This text can be copied to the bash script to run the downstream experiments. Example bash scripts can be found in the `launcher/` directory:
- `launcher/run_preprocess_ft.sh` - For preprocessing experiments
- `launcher/run_main_ft.sh` - For main analysis experiments
- `launcher/run_analysis_ft.sh` - For downstream analysis experiments

## Generated Configuration Files

Each generated `config.yaml` file contains:

### Complete Parameter Set
- All parameters from the corresponding template file
- Experiment-specific values from the CSV design table
- Proper type conversions (strings, integers, floats, booleans, lists)

### Hierarchical Structure
- Nested YAML structure matching the template
- Parameters organized by functional groups (data, channels, patching, etc.)
- Consistent formatting and indentation

### Example Generated Configuration

```yaml
exp_id: D10_0

data:
  file_name: /path/to/D10_tissue_0.ome.tiff
  antibodies_file: /path/to/antibodies.tsv
  image_mpp: 0.5
  generate_channel_stats: true

channels:
  nuclear_channel: DAPI
  wholecell_channel:
    - Pan-Cytokeratin
    - E-cadherin

patching:
  split_mode: full_image
  patch_height: -1
  patch_width: -1
  overlap: 0.1

visualization:
  visualize_whole_sample: false
  visualize_patches: true
  save_all_channel_patches: false
```

## Output Validation

The generator performs validation to ensure:
- All required parameters are present
- File paths reference existing files (when applicable)
- Parameter types match expected formats
- Experiment IDs are unique within the set
- YAML syntax is valid and properly formatted

<!-- ### Experiment Naming Conventions

- **Preprocessing Experiments**: Typically use sample IDs (e.g., `D10`, `D11`)
- **Main Analysis Experiments**: Use sample ID + split index (e.g., `D10_0`, `D10_1`)
- **Analysis Experiments**: Use descriptive names (e.g., `D18_Scan1_markerset_1`) -->
