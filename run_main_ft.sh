#!/bin/bash

# Define the experiment set name
EXP_SET_NAME="main_ft"  # "explore-eval-scores-dev" or "explore-eval-scores", "test0206_main"

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
  # "D10_0"
  # "D10_1"
  # "D10_2"
  # "D10_3"
  "D11_0"
  "D11_1"
  "D11_2"
  "D11_3"
  # "D13_0"
  # "D13_1"
  # "D13_2"
  # "D13_3"
  # "D13_4"
  # "D14_0"
  # "D14_1"
  # "D14_2"
  # "D14_3"
  # "D15_0"
  # "D15_1"
  # "D15_2"
  # "D15_3"
  # "D16_0"
  # "D16_1"
  # "D16_2"
  # "D16_3"
  # "D17_0"
  # "D17_1"
  # "D17_2"
  # "D17_3"
  # "D18_0"
  # "D18_1"
  # "D18_2"
  # "D18_3"  
  # "test-1"
  # "D16_Scan1_0"
  # "D18_Scan1_0"
  # "D18_Scan1_1"
  # "D18_Scan1_2"
  # "D18_Scan1_3"
  # "H33_scan1"
  # "D18_Scan1_1_markerset_1"
  # "D18_Scan1_1_markerset_2"
  # "D18_Scan1_1_markerset_3"
  # "D18_Scan1_1_markerset_4"
  # "D18_Scan1_1_markerset_1_patches_1000"
  # "D18_Scan1_1_markerset_3_patches_1000"
  # "D18_Scan1_1_markerset_4_patches_1000"   
  # "D18_Scan1_1_patches_1000"
  # "D18_Scan1_1_markerset_2_patches_1000"
)

# Count total experiments
TOTAL_EXPS=${#EXPERIMENTS[@]}
CURRENT_EXP=1

echo "Starting sequential execution of $TOTAL_EXPS experiments at $(date)"
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