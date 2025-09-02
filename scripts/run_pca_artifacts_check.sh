#!/bin/bash

# Usage check
if [ "$#" -ne 8 ]; then
  echo "Usage: $0 <INPUT_IMAGE> <COMPONENTS> <SAMPLE_PIXELS> <CHUNK_SIZE> <FULL_SCALE> <OUT_DIR> <LOG_DIR> <RUN_NAME>"
  echo "Example: $0 /path/to/image.ome.tiff 3 128000000 2000000 5000 /path/to/out /path/to/logs artifact_test"
  exit 1
fi

# Capture arguments
INPUT_IMAGE=$1
COMPONENTS=$2
SAMPLE_PIXELS=$3
CHUNK_SIZE=$4
FULL_SCALE=$5
OUT_DIR=$6
LOG_DIR=$7
RUN_NAME=$8

# Create directories if they don't exist
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Construct output paths
OUTPUT_SUBDIR="${OUT_DIR}/${RUN_NAME}_artifact_scale${FULL_SCALE}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_artifact_scale${FULL_SCALE}.log"

# Print configuration
echo "Running PCA artifact check with following parameters:"
echo "  Input image: ${INPUT_IMAGE}"
echo "  Components: ${COMPONENTS}"
echo "  Sample pixels: ${SAMPLE_PIXELS}"
echo "  Chunk size: ${CHUNK_SIZE}"
echo "  Full scale: ${FULL_SCALE}"
echo "  Output directory: ${OUTPUT_SUBDIR}"
echo "  Log file: ${LOG_FILE}"

# Run the PCA artifact check
python /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/src/pca_artifact_check.py \
    "${INPUT_IMAGE}" \
    --components "${COMPONENTS}" \
    --sample_pixels "${SAMPLE_PIXELS}" \
    --chunk_size "${CHUNK_SIZE}" \
    --full_scale "${FULL_SCALE}" \
    --outdir "${OUTPUT_SUBDIR}" \
    > "${LOG_FILE}" 2>&1

echo "PCA artifact check completed. Results saved to ${OUTPUT_SUBDIR}" 