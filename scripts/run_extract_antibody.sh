#!/bin/bash

# Usage: run_extract_antibody.sh <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR>

if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR>"
  exit 1
fi

EXP_SET_NAME=$1
EXP_ID=$2
ROOT_DIR=$3
DATA_DIR=$4
CONFIG_DIR=$5
OUT_DIR=$6

# Script that does the actual antibody extraction
ANTIBODY_SCRIPT="${ROOT_DIR}/0-phenocycler-penntmc-pipeline/src/extract_antibodies.py"

# e.g. config is stored at: CONFIG_DIR/EXP_ID/config.yaml
CONFIG_FILE="${CONFIG_DIR}/${EXP_ID}/config.yaml"

echo "Running antibody extraction for ${EXP_ID} with config ${CONFIG_FILE}"

# Call the Python module
python "${ANTIBODY_SCRIPT}" \
  --config "${CONFIG_FILE}" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}"

echo "Antibody extraction for ${EXP_ID} completed. Results are in ${OUT_DIR}"