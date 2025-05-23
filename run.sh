#!/bin/bash

# Define the experiment set name
EXP_SET_NAME="explore-eval-scores-dev"  # "explore-eval-scores-dev" or "explore-eval-scores"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"

# Define other variables based on input arguments
RUN_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_main.sh"
DATA_DIR="${ROOT_DIR}/data"
CONFIG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/config/${EXP_SET_NAME}"

# Define the output and log directories
LOG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/logs/${EXP_SET_NAME}"
OUT_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/out/${EXP_SET_NAME}"
# Create the output and log directories if they do not exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Define an array of experiment names
declare -a EXPERIMENTS=(
  # "exp-0"
  # "exp-1"
  # "exp-2"
  # "exp-3"
  # "exp-4"
  # "exp-5"
  # "exp-6"
  # "exp-7"
  # "exp-8"
  # "exp-9"
  "exp-10"
  # "exp-11"   
  # "exp-12"   
)

# Loop through the experiments and call run_main.sh for each
for EXP_ID in "${EXPERIMENTS[@]}"; do
  LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
  echo "Initiating experiment $EXP_ID"
  {
  echo "Current time: $(date)"
  echo "Running experiment $EXP_ID"
  time bash ${RUN_FILE} "$EXP_SET_NAME" "$EXP_ID" "$ROOT_DIR" "$DATA_DIR" "$CONFIG_DIR" "$OUT_DIR"
  echo "Experiment $EXP_ID completed."
  } > "$LOG_FILE" 2>&1 # &

  # Call run_main.sh with the current experiment set name, experiment name, and base directory
done

# Wait for all background processes to complete
# wait

# Print the final message
echo "All experiments completed."