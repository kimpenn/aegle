#!/bin/bash
# artifact_classification.sh
# Wrapper script for artifact classification

# Default Variables
CODEX_IMAGE="/AEGLE_CNN/data/img0011.qptiff"
TISSUE_REGIONS="/AEGLE_CNN/data/img0011.qptiff_tissue_region.geojson"
ANTIBODY_NAMES="/AEGLE_CNN/data/img0011_channel.json"
OUTPUT_DIR="/AEGLE_CNN/data/img0011_tiles_results"
MODEL_PATH="/AEGLE_CNN/deepsets_model_250epoch.pt"
CLASS_NAMES="Normal Bubble Debris Fold Smear"

# Override variables via command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --codex_image) CODEX_IMAGE="$2"; shift ;;
        --tissue_regions) TISSUE_REGIONS="$2"; shift ;;
        --antibody_names) ANTIBODY_NAMES="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --class_names) CLASS_NAMES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--codex_image <path>] [--tissue_regions <path>] [--antibody_names <path>] [--output_dir <path>] [--model_path <path>] [--class_names \"Class1 Class2\"]"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Configuration:"
echo "  CODEX_IMAGE: $CODEX_IMAGE"
echo "  TISSUE_REGIONS: $TISSUE_REGIONS"
echo "  ANTIBODY_NAMES: $ANTIBODY_NAMES"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  MODEL_PATH: $MODEL_PATH"

mkdir -p "$OUTPUT_DIR"

# 1. Tiling
# Save to a subdirectory of the codex image file path
CODEX_DIR=$(dirname "$CODEX_IMAGE")
CODEX_BASENAME=$(basename "$CODEX_IMAGE")
CODEX_NAME="${CODEX_BASENAME%.*}"
TILE_DIR="${CODEX_DIR}/${CODEX_NAME}_tiles"

echo "==================================="
echo "1. Tiling Image"
echo "==================================="
# python aegle/artifact_CNN/qptiff_tiler.py \
#     --input "$CODEX_IMAGE" \
#     --output_dir "$TILE_DIR" \
#     --geojson "$TISSUE_REGIONS" \
#     --level 0 \
#     --tile_w 960 --tile_h 720

echo "==================================="
echo "2. Classifying Tiles"
echo "==================================="
# Update PYTHONPATH so that local imports in classify_tiles.py work correctly
PYTHONPATH="aegle/artifact_CNN:$PYTHONPATH" python aegle/artifact_CNN/classify_tiles_cpu.py \
    --data_dir "$TILE_DIR/Unlabeled" \
    --metadata_path "$ANTIBODY_NAMES" \
    --model_path "$MODEL_PATH" \
    --csv_path "${OUTPUT_DIR}/classification_report.csv" \
    --test_ratio 1.0 \
    ${CLASS_NAMES:+--class_names $CLASS_NAMES}

echo "==================================="
echo "Done. Results saved to $OUTPUT_DIR"
echo "==================================="
