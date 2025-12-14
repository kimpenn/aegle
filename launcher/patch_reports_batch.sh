#!/bin/bash
# Batch patch HTML reports to add Sample ID and fix Valid Patches metrics
# Creates .bak backups before modifying files
# bash /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/launcher/patch_reports_batch.sh
set -e

REPO_ROOT="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
PATCH_SCRIPT="$REPO_ROOT/scripts/patch_html_reports.py"

# Directories to process
DIRS=(
    "$REPO_ROOT/out/main/main_ft_hb"
    "$REPO_ROOT/out/main/main_ovary_hb"
    "$REPO_ROOT/out/main/main_uterus_hb"
)

echo "=============================================="
echo "Batch HTML Report Patching"
echo "Adds Sample ID and fixes Valid Patches metrics"
echo "=============================================="
echo ""

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Processing: $dir"
        python "$PATCH_SCRIPT" "$dir"
        echo ""
    else
        echo "SKIP: Directory not found: $dir"
        echo ""
    fi
done

echo "=============================================="
echo "Batch patching complete!"
echo "=============================================="
