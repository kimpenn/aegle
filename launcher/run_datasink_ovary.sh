#!/bin/bash
# =============================================================================
# Datasink Launcher for Ovary Cohort
# =============================================================================
# Collects HTML reports from main and analysis pipelines and uploads to Box.
#
# Usage:
#   ./launcher/run_datasink_ovary.sh [--dry_run] [--no_upload]
#
# Options:
#   --dry_run     Preview what would be done without copying/uploading
#   --no_upload   Collect reports locally without uploading to Box
#
# =============================================================================
# bash /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/launcher/run_datasink_ovary.sh --no_upload
set -e

# Configuration
ROOT_DIR="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
OUT_ROOT="${ROOT_DIR}/out"
REMOTE="remote:"
REMOTE_PATH="aegle-reports"

# Parse arguments
DRY_RUN=""
UPLOAD="--upload"

for arg in "$@"; do
    case $arg in
        --dry_run)
            DRY_RUN="--dry_run"
            ;;
        --no_upload)
            UPLOAD=""
            ;;
    esac
done

cd "${ROOT_DIR}"

echo "============================================================"
echo "Aegle Datasink - Ovary Cohort"
echo "============================================================"
echo "Remote: ${REMOTE}${REMOTE_PATH}/"
echo "Options: ${DRY_RUN} ${UPLOAD}"
echo "============================================================"

# Collect and upload main pipeline reports
echo ""
echo "[1/2] Collecting main pipeline reports (main_ovary_hb)..."
echo "------------------------------------------------------------"
python src/datasink.py \
    --stage main \
    --exp_set main_ovary_hb \
    --out_root "${OUT_ROOT}" \
    --remote "${REMOTE}" \
    --remote_path "${REMOTE_PATH}" \
    ${UPLOAD} \
    ${DRY_RUN}

# Collect and upload analysis pipeline reports
echo ""
echo "[2/2] Collecting analysis pipeline reports (analysis_ovary_hb)..."
echo "------------------------------------------------------------"
python src/datasink.py \
    --stage analysis \
    --exp_set analysis_ovary_hb \
    --out_root "${OUT_ROOT}" \
    --remote "${REMOTE}" \
    --remote_path "${REMOTE_PATH}" \
    ${UPLOAD} \
    ${DRY_RUN}

echo ""
echo "============================================================"
echo "Datasink completed for Ovary cohort!"
echo "============================================================"
