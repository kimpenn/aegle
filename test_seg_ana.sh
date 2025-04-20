export PYTHONPATH="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline:$PYTHONPATH"
python -m aegle.segmentation_analysis.test_segmentation_analysis \
 --skip-cell-matching \
 --log-level DEBUG \
 --output /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle/segmentation_analysis/test_output
