#!/bin/bash

# Define the experiment set name (same logic as in the main script)
EXP_SET_NAME="preprocess/test0206"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"

# Define other variables based on input arguments
RUN_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_extract_tissue.sh"
DATA_DIR="${ROOT_DIR}/data"
CONFIG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/exps/configs/${EXP_SET_NAME}"

# Define the output and log directories
LOG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/logs/${EXP_SET_NAME}_tissue_extraction"
OUT_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/out/${EXP_SET_NAME}_tissue_extraction"
# Create the output and log directories if they do not exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Define an array of experiment names (or ID's)
declare -a EXPERIMENTS=(
  "d18_scan1"
)

# Loop through the experiments and call run_extract_tissue.sh for each
for EXP_ID in "${EXPERIMENTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
  echo "Initiating tissue extraction for experiment $EXP_ID"
  {
    echo "Current time: $(date)"
    echo "Running tissue extraction for $EXP_ID"
    time bash "${RUN_FILE}" \
      "$EXP_SET_NAME" \
      "$EXP_ID" \
      "$ROOT_DIR" \
      "$DATA_DIR" \
      "$CONFIG_DIR" \
      "$OUT_DIR"
    echo "Tissue extraction for $EXP_ID completed."
  } > "$LOG_FILE" 2>&1
done

echo "All tissue extractions completed."