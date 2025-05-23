#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 6 ] && [ "$#" -ne 7 ]; then
  echo "Usage: $0 <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR> [LOG_LEVEL]"
  exit 1
fi
sleep 3
# Assign input arguments to variables
EXP_SET_NAME=$1
EXP_ID=$2
ROOT_DIR=$3
DATA_DIR=$4
CONFIG_DIR=$5
OUT_DIR=$6
LOG_LEVEL=${7:-INFO}  # Default to INFO if not provided

# Run the Python script with the specified arguments and measure the time
echo "Running the Python script with log level: ${LOG_LEVEL}..."
python "${ROOT_DIR}/0-phenocycler-penntmc-pipeline/src/main.py" \
 --data_dir "${DATA_DIR}" \
 --config_file "${CONFIG_DIR}/${EXP_ID}/config.yaml" \
 --out_dir "${OUT_DIR}/${EXP_ID}" \
 --log_level "${LOG_LEVEL}"
