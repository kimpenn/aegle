---
sidebar_position: 2
---

# Usage Guide

This guide provides comprehensive instructions for running the Main pipeline, from single experiment execution to large-scale batch processing. The template for running the Main pipeline is `run_main_template.sh`. You can make a copy of the template (e.g. `run_main_ft.sh`) and modify it to run the Main pipeline for your experiment set and data. Put the copy in the root directory of your project. Usually you need to change the `EXP_SET_NAME` and the `EXPERIMENTS` array, which are marked with `TODO` in the bash script. Then call `bash run_main_ft.sh` to run with your data.

The runnable script (e.g. `run_main_ft.sh`) calls `scripts/run_main.sh` with the following arguments
- `EXP_SET_NAME`: The name of the experiment set
- `EXP_ID`: The name of the experiment
- `ROOT_DIR`: The root directory of your project
- `DATA_DIR`: The directory of your data
- `CONFIG_DIR`: The directory of your config
- `OUT_DIR`: The directory of your output
- `LOG_LEVEL`: The log level

Then `run_main.sh` runs the `src/main.py` script to launch the Main pipeline.