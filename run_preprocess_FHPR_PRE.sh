#!/bin/bash

# Define the maximum number of concurrent experiments (default: 2)
MAX_CONCURRENT=${1:-2}

# Define the experiment set name (same logic as in the main script)
EXP_SET_NAME="preprocess/preprocess_FHPR_PRE"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"

# Define other variables for the two run scripts
RUN_TISSUE_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_extract_tissue.sh"
RUN_ANTIBODY_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_extract_antibody.sh"
RUN_OVERVIEW_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_generate_overview.sh"

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
  "FHPR_PRE_scan1"
)

echo "Starting processing with maximum $MAX_CONCURRENT concurrent experiments"

# Function to run a single experiment
run_experiment() {
  local EXP_ID=$1
  local LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
  
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

    # (3) Overview Thumbnails
    echo "Generating annotation overviews for ${EXP_ID}"
    time bash "${RUN_OVERVIEW_FILE}" \
      "${EXP_SET_NAME}" \
      "${EXP_ID}" \
      "${ROOT_DIR}" \
      "${DATA_DIR}" \
      "${CONFIG_DIR}" \
      "${OUT_DIR}"
    echo "Overview generation for ${EXP_ID} completed."

  } > "$LOG_FILE" 2>&1
  
  echo "Experiment $EXP_ID finished at $(date)"
}

# Counter for active jobs
declare -a PIDS=()

# Loop through the experiments with concurrency control
for EXP_ID in "${EXPERIMENTS[@]}"; do
  # If we've reached the maximum concurrent jobs, wait for one to finish
  while [ ${#PIDS[@]} -ge $MAX_CONCURRENT ]; do
    # Check which jobs have finished
    for i in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        # Job has finished, remove it from the array
        unset "PIDS[$i]"
      fi
    done
    
    # Rebuild the array to remove gaps
    PIDS=("${PIDS[@]}")
    
    # If still at max, wait a bit before checking again
    if [ ${#PIDS[@]} -ge $MAX_CONCURRENT ]; then
      sleep 1
    fi
  done
  
  # Start the experiment in background
  run_experiment "$EXP_ID" &
  PID=$!
  PIDS+=($PID)
  
  echo "Started experiment $EXP_ID with PID $PID (${#PIDS[@]}/$MAX_CONCURRENT slots used)"
done

# Wait for all remaining jobs to complete
echo "Waiting for all remaining experiments to complete..."
for PID in "${PIDS[@]}"; do
  wait $PID
done

echo "All tissue extractions completed."