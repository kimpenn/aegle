#!/bin/bash

# Example script showing how to run PCA artifact analysis with different artifact levels

# Set common parameters
INPUT_IMAGE="/workspaces/codex-analysis/data/FallopianTube/D18/Scan1/D18_Scan1_tissue_0.ome.tiff"
RUN_NAME="D18_Scan1_tissue_0"
COMPONENTS=5
SAMPLE_PIXELS=128000000
CHUNK_SIZE=2000000

WORK_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
OUT_DIR="${WORK_DIR}/out/pca_sweep"
LOG_DIR="${WORK_DIR}/logs"

# Create output directories
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

# Array of artifact scales to test for uint8 images
SCALES=(0 2 4 8 20)

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