#!/bin/bash

# [1] Set experiment set name
EXP_SET_NAME="test_analysis"

# [2] Define the base directory (adjust as needed)
ROOT_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"

# [3] Define the path to the inner script that launches run_analysis.py
RUN_FILE="${ROOT_DIR}/scripts/run_analysis.sh"

# [4] Define input data directory
DATA_DIR="${ROOT_DIR}/out/main"


# [5] Directory where your config files for analysis live
#     Example: each experiment has its own subfolder with an analysis config.
CONFIG_DIR="${ROOT_DIR}/exps/configs/analysis/${EXP_SET_NAME}"

# [6] Define the output and log directories for analysis
LOG_DIR="${ROOT_DIR}/logs/analysis/${EXP_SET_NAME}"
OUT_DIR="${ROOT_DIR}/out/analysis/${EXP_SET_NAME}"

# [7] Create them if they don't exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# [8] Define an array of experiment identifiers
#     In your main pipeline, you might have 'D18_Scan1_0', etc.
#     Adjust as needed for whichever analysis you are doing:
declare -a EXPERIMENTS=(
  "D18_Scan1_0"
  "D18_Scan1_0"
)

# [9] Loop over experiments
for EXP_ID in "${EXPERIMENTS[@]}"; do
  # Create a log file for each experiment
  LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
  echo "Initiating analysis for $EXP_ID"
  {
    echo "Current time: $(date)"
    echo "Running analysis for $EXP_ID"
    time bash "${RUN_FILE}" "$EXP_SET_NAME" "$EXP_ID" "$ROOT_DIR" "$DATA_DIR" "$CONFIG_DIR" "$OUT_DIR"
    echo "Experiment $EXP_ID analysis completed."
  } > "$LOG_FILE" 2>&1
done

echo "All analysis experiments completed."