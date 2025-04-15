#!/bin/bash

# Define the experiment set name (same logic as in the main script)
EXP_SET_NAME="preprocess/test0206_preprocess"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"

# Define other variables for the two run scripts
RUN_TISSUE_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_extract_tissue.sh"
RUN_ANTIBODY_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_extract_antibody.sh"

DATA_DIR="${ROOT_DIR}/data"
CONFIG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/exps/configs/${EXP_SET_NAME}"

# Define the output and log directories
LOG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/logs/${EXP_SET_NAME}"
OUT_DIR="${DATA_DIR}"

# Create the output and log directories if they do not exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Define an array of experiment names (or ID's)
declare -a EXPERIMENTS=(
  "d18_scan1"
  # "d18_scan2"
  # "H33_scan1"
  "d11_scan1"
)

# Loop through the experiments and call run_extract_tissue.sh for each
for EXP_ID in "${EXPERIMENTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
  echo "Initiating tissue extraction for experiment $EXP_ID"
  {
    echo "Current time: $(date)"

    # (1) Tissue Extraction
    echo "Running tissue extraction for $EXP_ID"
    time bash "${RUN_TISSUE_FILE}" \
      "$EXP_SET_NAME" \
      "$EXP_ID" \
      "$ROOT_DIR" \
      "$DATA_DIR" \
      "$CONFIG_DIR" \
      "$OUT_DIR"
    echo "Tissue extraction for $EXP_ID completed."

    # (2) Antibody Extraction
    echo "Running antibody extraction for ${EXP_ID}"
    time bash "${RUN_ANTIBODY_FILE}" \
      "${EXP_SET_NAME}" \
      "${EXP_ID}" \
      "${ROOT_DIR}" \
      "${DATA_DIR}" \
      "${CONFIG_DIR}" \
      "${OUT_DIR}"
    echo "Antibody extraction for ${EXP_ID} completed."

  } > "$LOG_FILE" 2>&1
done

echo "All tissue extractions completed."