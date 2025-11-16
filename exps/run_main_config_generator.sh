#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Custom configuration for this run
# ---------------------------------------------------------------------------
ANALYSIS_STEP="main"
# main_ft_hb, main_ovary_hb
EXPERIMENT_SET="main_ovary_hb"

CSV_PATH="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/csvs/${EXPERIMENT_SET}.csv"
OUTPUT_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/${ANALYSIS_STEP}/${EXPERIMENT_SET}"

# ---------------------------------------------------------------------------
# Invoke the generic generator
# ---------------------------------------------------------------------------
ANALYSIS_STEP="${ANALYSIS_STEP}" \
EXPERIMENT_SET="${EXPERIMENT_SET}" \
CSV_PATH="${CSV_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
  /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/run_config_generator.sh
