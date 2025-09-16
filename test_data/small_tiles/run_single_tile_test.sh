#!/bin/bash

# Simple script to test a single tile

# Define paths
ROOT_DIR="/workspaces/codex-analysis"
PIPELINE_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline"
RUN_FILE="${PIPELINE_DIR}/scripts/run_main.sh"
DATA_DIR="${PIPELINE_DIR}/test_data/small_tiles"

# Test with tile_0
EXP_NAME="tile_0"
CONFIG_FILE="${DATA_DIR}/${EXP_NAME}/config.yaml"
OUT_DIR="${DATA_DIR}/output/${EXP_NAME}"
LOG_DIR="${DATA_DIR}/logs"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

# Create directories
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Running single tile test: ${EXP_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="

# Run the pipeline
bash "${RUN_FILE}" \
  --data_dir "${DATA_DIR}" \
  --config_file "${CONFIG_FILE}" \
  --out_dir "${OUT_DIR}" \
  --log_level "INFO" \
  2>&1 | tee "${LOG_FILE}"

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "✅ Test completed successfully!"
  echo ""
  echo "Check results in:"
  echo "  - Output: ${OUT_DIR}"
  echo "  - Log: ${LOG_FILE}"
else
  echo "❌ Test failed! Check log for details: ${LOG_FILE}"
  exit 1
fi
