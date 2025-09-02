#!/bin/bash

# Example script showing how to run PCA artifact analysis with different artifact levels

# Set common parameters
# INPUT_IMAGE="/workspaces/codex-analysis/data/uterus/D16/Scan1/D16_Scan1_manual_3.ome.tiff"
RUN_NAME="D16_Scan1_manual_0"
INPUT_IMAGE="/workspaces/codex-analysis/data/uterus/D16/Scan1/D16_Scan1_manual_0.ome.tiff"
COMPONENTS=5
SAMPLE_PIXELS=128000000
CHUNK_SIZE=2000000

WORK_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
OUT_DIR="${WORK_DIR}/out/pca_sweep"
LOG_DIR="${WORK_DIR}/logs"

# Create output directories
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Array of artifact scales to test
SCALES=(0 100 200 500)

# Run PCA analysis for each scale
for SCALE in "${SCALES[@]}"; do
    echo "Running PCA with artifact scale: ${SCALE}"
    ${WORK_DIR}/scripts/run_pca_artifacts_check.sh \
        "${INPUT_IMAGE}" \
        "${COMPONENTS}" \
        "${SAMPLE_PIXELS}" \
        "${CHUNK_SIZE}" \
        "${SCALE}" \
        "${OUT_DIR}" \
        "${LOG_DIR}" \
        "${RUN_NAME}"
done

echo "PCA sweep completed for all artifact scales" 