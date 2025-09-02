#!/bin/bash
SAMPLE_NAME=D16_Scan1_0
python /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle_analysis/analysis_annotator.py \
 --prior /workspaces/codex-analysis/data/uterus/Uterus_OMAP.json \
 --cluster /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/analysis/test_analysis/${SAMPLE_NAME}/differential_expression/top_10_genes_summary.json \
 --output /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/analysis/test_analysis/${SAMPLE_NAME}/differential_expression/cell_annotation_results_v1.txt










