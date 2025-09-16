#!/bin/bash
# Run the pipeline on test tiles

# Set up environment variables
export PYTHONPATH=/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline:$PYTHONPATH

# Run on each tile
for tile in tile_*.ome.tiff; do
    echo "Processing $tile..."
    python /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/scripts/run_main.py \
        --config test_config.yaml \
        --input_dir . \
        --output_dir output_${tile%.ome.tiff} \
        --log_dir logs_${tile%.ome.tiff}
done
