#!/bin/bash

# Usage check
if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR>"
  exit 1
fi

# Sleep is optional—some people use it to let file systems settle
sleep 3

# [1] Capture arguments
EXP_SET_NAME=$1
EXP_ID=$2
ROOT_DIR=$3
DATA_DIR=$4
CONFIG_DIR=$5
OUT_DIR=$6

# [2] Construct your config file path
ANALYSIS_CONFIG="${CONFIG_DIR}/${EXP_ID}/config.yaml"

# [3] Run the Python script
echo "Running analysis Python script for experiment ${EXP_ID} ..."
python "${ROOT_DIR}/0-phenocycler-penntmc-pipeline/src/run_analysis.py" \
  --data_dir "${DATA_DIR}" \
  --config_file "${ANALYSIS_CONFIG}" \
  --output_dir "${OUT_DIR}/${EXP_ID}"

echo "Analysis for experiment ${EXP_ID} is done."