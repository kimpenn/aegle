#!/bin/bash

# Define the experiment set name
EXP_SET_NAME="test/oocytes"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"

# Define other variables based on input arguments
RUN_FILE="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/scripts/run_main.sh"
DATA_DIR="${ROOT_DIR}/data"
CONFIG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/exps/configs/main/${EXP_SET_NAME}"

# Define logging level (can be DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# Define the output and log directories
LOG_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/logs/main/${EXP_SET_NAME}"
OUT_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/out/main/${EXP_SET_NAME}"
# Create the output and log directories if they do not exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Define an array of experiment names
declare -a EXPERIMENTS=(
  # "oocyte_0"
  # "oocyte_1"
  # "oocyte_2"
  # "oocyte_0_1"
  # "oocyte_4"
  "oocyte_4_1"
)

# Count total experiments
TOTAL_EXPS=${#EXPERIMENTS[@]}
CURRENT_EXP=1

echo "Starting sequential execution of $TOTAL_EXPS experiments at $(date)"
echo "Experiment set name: $EXP_SET_NAME"
echo "Log directory: $LOG_DIR"
echo "Output directory: $OUT_DIR"
echo "Log level set to: $LOG_LEVEL"
echo "-----------------------------------------------------------"

# Loop through the experiments and call run_main.sh for each
for EXP_ID in "${EXPERIMENTS[@]}"; do
  START_TIME=$(date +%s)
  LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
  echo "Initiating experiment $EXP_ID ($CURRENT_EXP of $TOTAL_EXPS)"
  echo "Start time: $(date)"
  {
    echo "Current time: $(date)"
    echo "Running experiment $EXP_ID ($CURRENT_EXP of $TOTAL_EXPS)"
    echo "Log level: $LOG_LEVEL"
    time bash ${RUN_FILE} "$EXP_SET_NAME" "$EXP_ID" "$ROOT_DIR" "$DATA_DIR" "$CONFIG_DIR" "$OUT_DIR" "$LOG_LEVEL"
    echo "Experiment $EXP_ID completed."
  } > "$LOG_FILE" 2>&1 

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  
  echo "Completed experiment $EXP_ID ($CURRENT_EXP of $TOTAL_EXPS)"
  echo "End time: $(date)"
  echo "Duration: $((DURATION / 60)) minutes and $((DURATION % 60)) seconds"
  echo "-----------------------------------------------------------"
  
  ((CURRENT_EXP++))
done

echo "All $TOTAL_EXPS experiments completed sequentially at $(date)"
echo "-----------------------------------------------------------"