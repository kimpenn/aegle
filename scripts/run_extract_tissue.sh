#!/bin/bash

# Check if the correct number of arguments is provided
# Usage: run_tissue_extraction.sh <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR>
if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR>"
  exit 1
fi

# Sleep briefly (optional), to ensure any ongoing file writes complete
sleep 3

# Assign input arguments to variables
EXP_SET_NAME=$1
EXP_ID=$2
ROOT_DIR=$3
DATA_DIR=$4
CONFIG_DIR=$5
OUT_DIR=$6

# Create an output subdirectory specifically for tissue extraction results
EXTRACT_OUT_DIR="${OUT_DIR}/${EXP_ID}/tissue_extraction"

echo "Running Tissue Region Extraction..."
python "${ROOT_DIR}/0-phenocycler-penntmc-pipeline/src/extract_tissue_regions.py" \
  --config "${CONFIG_DIR}/${EXP_ID}/config.yaml" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}"

echo "Tissue region extraction for ${EXP_ID} completed. Results are in ${OUT_DIR}"