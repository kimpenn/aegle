# Experiment Design Directory

This directory hosts the experiment configuration workflow for the CODEX data analysis pipeline. YAML configs are generated from design CSVs, validated against schemas, and based on reusable templates.

## Directory Structure

```
exps/
├── README.md                    # This documentation
├── config_generator.py          # Schema-aware config generator (CLI)
├── run_config_generator.sh      # Helper script for day-to-day config generation
├── test_config_generator.py     # Developer smoke tests for the generator
├── csvs/                        # Experiment design tables exported from Google Sheet
├── configs/                     # Generated configuration files (per analysis step/set)
├── templates/                   # YAML templates for preprocess/main/analysis
├── schemas/                     # Validation rules (types, defaults, rules) per analysis step
├── archive/                     # Historical experiments (inactive)
└── config_generators/           # Legacy/specialised generators (kept for reference)
```

## Quick Start

1. **Export a design table** from Google Sheet to `exps/csvs/<experiment_set>.csv`.
2. **Provide the required environment variables** when invoking the helper. No defaults are applied, so set all of the following:
   ```bash
   ANALYSIS_STEP=main \
   EXPERIMENT_SET=main_ft_hb \
   CSV_PATH=/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/csvs/main_ft_hb.csv \
   OUTPUT_DIR=/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/main/main_ft_hb \
     ./run_config_generator.sh
   ```
   Convenience wrappers such as `exps/run_main_config_generator.sh` and `exps/run_preprocess_config_generator.sh` already set these values for common scenarios.

   For the **analysis** stage we follow the same pattern. Production FT assets live at
   `exps/csvs/analysis/main_ft_hb.csv`, `exps/templates/analysis_template.yaml`, and
   `exps/schemas/analysis.yaml`. Invoke the helper with:

   ```bash
   ANALYSIS_STEP=analysis \
   EXPERIMENT_SET=analysis_ft_hb \
   CSV_PATH=/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/csvs/analysis/main_ft_hb.csv \
   OUTPUT_DIR=/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/analysis/analysis_ft_hb \
     ./run_config_generator.sh
   ```

   Convenience wrapper `exps/run_analysis_config_generator.sh` mirrors the preprocess
   and main helpers for day-to-day use.
3. **Consume generated configs** from `exps/configs/<analysis_step>/<experiment_set>/`.

The helper script is a thin wrapper around `config_generator.py`. You can also invoke the Python module directly for ad-hoc runs:

```bash
python exps/config_generator.py \
  --analysis-step main \
  --experiment-set main_ft \
  --base-dir exps \
  --csv exps/csvs/main_ft.csv \
  --template exps/templates/main_template.yaml \
  --schema exps/schemas/main.yaml \
  --output-dir exps/configs/main/main_ft
```

## Validation & Testing

- Schemas (`exps/schemas/*.yaml`) define required columns, data types, defaults, and cross-field rules. The generator will refuse to produce configs if the design table violates these constraints.
- `exps/test_config_generator.py` provides quick smoke tests to verify the generator after schema/template changes. Run it during development:
  ```bash
  cd /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps
  python test_config_generator.py
  ```

- Pytest coverage lives in `tests/exps/test_config_generator.py`; run `pytest tests/exps/test_config_generator.py` to integrate with the main test suite.

## CSV Format Tips

- Use `::` to express nested structure. For example `channels::wholecell_channel` maps to `channels.wholecell_channel` in YAML.
- Boolean-like values accept `TRUE/FALSE`, `yes/no`, `1/0`, etc.
- Absolute paths are required for files referenced by the pipeline (images, antibodies TSVs, model directories, manual masks).

## Pipeline Integration

Experiment sets align with pipeline stages:
- **preprocess** – tissue extraction & antibody metadata prep
- **main** – segmentation, patch extraction, cell profiling
- **analysis** – downstream analytics (differential expression, plots, etc.)

Generated configs in `exps/configs/<analysis_step>/<experiment_set>/` are picked up by the corresponding run scripts (e.g. `run_preprocess_ft.sh`, `run_main_ft.sh`).

## Additional Resources

- Full documentation: [Experiment Configuration Guide](https://kimpenn.github.io/aegle/docs/ExperimentConfiguration/overview)
- Troubleshooting & best practices: [Best Practices](https://kimpenn.github.io/aegle/docs/ExperimentConfiguration/best-practices)
