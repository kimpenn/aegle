# Experiment Design Directory

This directory contains the experiment configuration generation and management system for the CODEX data analysis pipeline.

## Quick Overview

The configuration generator (`config_generator.py`) enables automated generation of YAML configuration files from CSV experiment design tables, supporting:

- Batch configuration generation from design tables
- Automatic parameter type conversion and validation  
- Nested configuration structures using `::` separators
- Integration with all pipeline components

## Directory Structure

```
exps/
├── README.md                    # This documentation
├── config_generator.py          # Main configuration generator script
├── config_generators/           # Specialized configuration generator modules
├── csvs/                        # Experiment design tables exported from Google Sheet
├── configs/                     # Generated configuration files
├── *_template.yaml              # Configuration template files
└── archive/                     # Historical configurations and experiments
```

## Quick Start

1. **Configure Parameters**: Edit variables in `config_generator.py`
   ```python
   experiment_set_name = "main_ft"  # Experiment set name
   analysis_step = "main"           # Analysis step: preprocess/main/analysis
   ```

2. **Prepare Design Table**: Export CSV from [Google Sheet](https://docs.google.com/spreadsheets/d/1JcabHe3aobjeRa4V0mNSmC2rnh5jJ3h4_v0jnX2vkOE/edit?usp=sharing) to `csvs/` directory

3. **Generate Configurations**:
   ```bash
   cd /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps
   python config_generator.py
   ```

4. **Use Generated Configs**: Configurations will be created in `configs/{analysis_step}/{experiment_set_name}/`

## CSV Format

Use `::` separators for nested configuration:
```csv
exp_id,data::file_name,channels::nuclear_channel,channels::wholecell_channel
D10_0,/path/to/image.tiff,DAPI,"Pan-Cytokeratin,E-cadherin"
```

## Pipeline Integration

Three experiment types correspond to pipeline components:
- **`preprocess_X`**: Tissue extraction and preprocessing  
- **`main_X`**: Image segmentation and cell profiling
- **`analysis_X`**: Downstream statistical analysis

## 📚 Complete Documentation

For detailed usage instructions, parameter handling, troubleshooting, and best practices, see the **[Complete Experiment Configuration Documentation](https://kimpenn.github.io/aegle/docs/ExperimentConfiguration/overview)**.

## Support

- **Issues**: Report problems via GitHub issues
- **Questions**: Consult the [online documentation](https://kimpenn.github.io/aegle/docs/ExperimentConfiguration/overview)
- **Development**: See [best practices guide](https://kimpenn.github.io/aegle/docs/ExperimentConfiguration/best-practices) for extending the system