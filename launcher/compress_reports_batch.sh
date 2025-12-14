#!/bin/bash
# Batch compress HTML reports for multiple output directories
# Renames original to .raw.html and saves compressed as pipeline_report.html
# bash /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/launcher/compress_reports_batch.sh
set -e

REPO_ROOT="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
COMPRESS_SCRIPT="$REPO_ROOT/scripts/compress_html_reports.py"

# Directories to process
DIRS=(
    # Main pipeline reports
    "$REPO_ROOT/out/main/main_ft_hb"
    "$REPO_ROOT/out/main/main_ovary_hb"
    "$REPO_ROOT/out/main/main_uterus_hb"
    # Analysis reports
    "$REPO_ROOT/out/analysis/analysis_ft_hb"
    "$REPO_ROOT/out/analysis/analysis_ovary_hb"
    "$REPO_ROOT/out/analysis/analysis_uterus_hb"
)

echo "=============================================="
echo "Batch HTML Report Compression"
echo "=============================================="
echo ""

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Processing: $dir"
        python "$COMPRESS_SCRIPT" "$dir"
        echo ""
    else
        echo "SKIP: Directory not found: $dir"
        echo ""
    fi
done

echo "=============================================="
echo "Batch compression complete!"
echo "=============================================="
