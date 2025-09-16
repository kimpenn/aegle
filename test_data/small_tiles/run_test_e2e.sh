#!/bin/bash

# Define the experiment set name
EXP_SET_NAME="test_tiles"

# Define the base directory
ROOT_DIR="/workspaces/codex-analysis"
PIPELINE_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline"

# Define other variables
RUN_FILE="${PIPELINE_DIR}/scripts/run_main.sh"
DATA_DIR="${PIPELINE_DIR}/test_data/small_tiles"
CONFIG_DIR="${DATA_DIR}"

# Define logging level (can be DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# Define the output and log directories
LOG_DIR="${PIPELINE_DIR}/test_data/small_tiles/logs"
OUT_DIR="${PIPELINE_DIR}/test_data/small_tiles/output"

# Create the output and log directories if they do not exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Define an array of experiment names
declare -a EXPERIMENTS=(
  "tile_0"
  "tile_1"
  "tile_2"
  "tile_3"
)

# Function to run a single experiment
run_experiment() {
  local exp_name=$1
  local config_file="${CONFIG_DIR}/${exp_name}/config.yaml"
  local out_path="${OUT_DIR}/${exp_name}"
  local log_path="${LOG_DIR}/${exp_name}.log"

  echo "=========================================="
  echo "Running experiment: ${exp_name}"
  echo "Config: ${config_file}"
  echo "Output: ${out_path}"
  echo "Log: ${log_path}"
  echo "=========================================="

  # Check if config file exists
  if [ ! -f "${config_file}" ]; then
    echo "ERROR: Config file not found: ${config_file}"
    return 1
  fi

  # Run the experiment
  bash "${RUN_FILE}" \
    --data_dir "${DATA_DIR}" \
    --config_file "${config_file}" \
    --out_dir "${out_path}" \
    --log_level "${LOG_LEVEL}" \
    2>&1 | tee "${log_path}"

  # Check if the experiment was successful
  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Experiment ${exp_name} completed successfully!"
  else
    echo "❌ Experiment ${exp_name} failed!"
    return 1
  fi
}

# Main execution
echo "=========================================="
echo "Starting End-to-End Test Pipeline"
echo "Experiment set: ${EXP_SET_NAME}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "=========================================="

# Track overall success
all_success=true

# Loop through all experiments
for exp_name in "${EXPERIMENTS[@]}"; do
  if ! run_experiment "${exp_name}"; then
    all_success=false
  fi
  echo ""
done

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ "$all_success" = true ]; then
  echo "✅ All experiments completed successfully!"
  exit 0
else
  echo "❌ Some experiments failed. Check the logs for details."
  exit 1
fi
