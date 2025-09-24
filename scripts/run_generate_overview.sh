#!/bin/bash

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

CONFIG_FILE="${CONFIG_DIR}/${EXP_ID}/config.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found for ${EXP_ID}: ${CONFIG_FILE}"
  exit 1
fi

# Allow tuning without editing the script
DOWNSCALE=${PREPROCESS_OVERVIEW_DOWNSCALE:-0.5}
QUALITY=${PREPROCESS_OVERVIEW_QUALITY:-85}
CHANNEL=${PREPROCESS_OVERVIEW_CHANNEL:-0}

OVERWRITE_FLAG=""
case "${PREPROCESS_OVERVIEW_OVERWRITE:-0}" in
  1|true|TRUE|yes|YES)
    OVERWRITE_FLAG="--overwrite"
    ;;
  *)
    ;;
esac

python "${ROOT_DIR}/0-phenocycler-penntmc-pipeline/src/generate_overview_thumbnails.py" \
  --config "${CONFIG_FILE}" \
  --root_dir "${ROOT_DIR}" \
  --downscale "${DOWNSCALE}" \
  --quality "${QUALITY}" \
  --channel "${CHANNEL}" \
  ${OVERWRITE_FLAG}
