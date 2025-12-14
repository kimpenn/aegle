#!/bin/bash
# Patch run to resume cell profiling for failed samples
# Run with: nohup bash /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/run_resume_patch.sh > logs/run_resume_patch_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

# Define paths
ROOT_DIR="/workspaces/codex-analysis"
PIPELINE_DIR="${ROOT_DIR}/0-phenocycler-penntmc-pipeline"
EXP_SET_NAME="main_uterus_hb"
DATA_DIR="${ROOT_DIR}/data"
CONFIG_DIR="${PIPELINE_DIR}/exps/configs/main/${EXP_SET_NAME}"
OUT_DIR="${PIPELINE_DIR}/out/main/${EXP_SET_NAME}"
LOG_DIR="${PIPELINE_DIR}/logs/main/${EXP_SET_NAME}"

echo "=== Patch Run: Resume Cell Profiling for Failed Samples ==="
echo "Start time: $(date)"
echo "-----------------------------------------------------------"

# Sample 1: D14_img0026_3 - COMPLETED (corrupted memmap file - fixed)
# echo ""
# echo "=== Processing D14_img0026_3 ==="
# echo "Removing corrupted memmap file..."
# rm -f "${OUT_DIR}/D14_img0026_3/all_channel_patches.memmap.npy"
#
# echo "Resuming cell profiling..."
# python "${PIPELINE_DIR}/src/main.py" \
#   --config_file "${CONFIG_DIR}/D14_img0026_3/config.yaml" \
#   --data_dir "${DATA_DIR}" \
#   --out_dir "${OUT_DIR}/D14_img0026_3" \
#   --resume_stage cell_profiling \
#   --log_level INFO \
#   2>&1 | tee "${LOG_DIR}/D14_img0026_3_patch.log"
#
# echo "D14_img0026_3 completed at $(date)"
# echo "-----------------------------------------------------------"

# Sample 2: D17_img0038_3 - COMPLETED (disk space issue - fixed)
# echo ""
# echo "=== Processing D17_img0038_3 ==="
# echo "Removing any partial memmap file..."
# rm -f "${OUT_DIR}/D17_img0038_3/all_channel_patches.memmap.npy"
#
# echo "Resuming cell profiling..."
# python "${PIPELINE_DIR}/src/main.py" \
#   --config_file "${CONFIG_DIR}/D17_img0038_3/config.yaml" \
#   --data_dir "${DATA_DIR}" \
#   --out_dir "${OUT_DIR}/D17_img0038_3" \
#   --resume_stage cell_profiling \
#   --log_level INFO \
#   2>&1 | tee "${LOG_DIR}/D17_img0038_3_patch.log"
#
# echo "D17_img0038_3 completed at $(date)"
# echo "-----------------------------------------------------------"

# Sample 3: D16_0 (main_ovary_hb) - OOM during cell profiling
# Segmentation completed successfully, process accumulated 164GB memory
# before cell profiling started. Fresh process should avoid OOM.
echo ""
echo "=== Processing D16_0 (main_ovary_hb) ==="
EXP_SET_NAME_OVARY="main_ovary_hb"
CONFIG_DIR_OVARY="${PIPELINE_DIR}/exps/configs/main/${EXP_SET_NAME_OVARY}"
OUT_DIR_OVARY="${PIPELINE_DIR}/out/main/${EXP_SET_NAME_OVARY}"
LOG_DIR_OVARY="${PIPELINE_DIR}/logs/main/${EXP_SET_NAME_OVARY}"

echo "Resuming cell profiling after OOM..."
python "${PIPELINE_DIR}/src/main.py" \
  --config_file "${CONFIG_DIR_OVARY}/D16_0/config.yaml" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR_OVARY}/D16_0" \
  --resume_stage cell_profiling \
  --log_level INFO \
  2>&1 | tee "${LOG_DIR_OVARY}/D16_0_patch.log"

echo "D16_0 completed at $(date)"
echo "-----------------------------------------------------------"

echo ""
echo "=== Patch Run Complete ==="
echo "End time: $(date)"

# Verify outputs
echo ""
echo "=== Verification ==="
# D16_0 (main_ovary_hb)
if [ -f "${OUT_DIR_OVARY}/D16_0/cell_profiling/cell_by_marker.csv" ]; then
  echo "D16_0 (main_ovary_hb): SUCCESS - cell_by_marker.csv exists"
  wc -l "${OUT_DIR_OVARY}/D16_0/cell_profiling/cell_by_marker.csv"
else
  echo "D16_0 (main_ovary_hb): FAILED - cell_by_marker.csv missing"
fi

# nohup python /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/src/main.py \
#   --config_file /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/main/main_ovary_hb/D14_15_1/config.yaml \
#   --data_dir /workspaces/codex-analysis/data \
#   --out_dir /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ovary_hb/D14_15_1 \
#   --resume_stage cell_profiling \
#   --log_level INFO \
#   > /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/logs/main/main_ovary_hb/D14_15_1_resume_profiling.log 2>&1 &

# nohup python /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/src/main.py \
#         --config_file
#       /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/main/main_ovary_hb/D14_15_1/config.yaml \
#         --data_dir /workspaces/codex-analysis/data \
#         --out_dir /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/main_ovary_hb/D14_15_1 \
#         --resume_stage visualization \
#         --log_level INFO \
#         > /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/logs/main/main_ovary_hb/D14_15_1_resume_visualizati
#       on.log 2>&1 &

