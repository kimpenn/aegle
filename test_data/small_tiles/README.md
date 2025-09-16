# PhenoCycler Pipeline Test Dataset

This directory contains a small test dataset created from the FallopianTube D10 img0003 sample for quick development and testing of the PhenoCycler pipeline.

## Contents

- **tile_000.ome.tiff** - **tile_003.ome.tiff**: Four 512x512 pixel tiles extracted from the original image
- **antibodies.tsv**: Antibody information file (41 channels)
- **tile_info.csv**: Information about each tile (position, intensity, etc.)
- **tile_0/** - **tile_3/**: Individual configuration directories for each tile
- **run_single_tile_test.sh**: Script to test a single tile (tile_0)
- **run_test_e2e.sh**: Script to run all four tiles
- **test_e2e_verification.py**: Python script to verify pipeline outputs

## Quick Start

### 1. Test a Single Tile (Recommended for first test)
```bash
./run_single_tile_test.sh
```

This will process tile_0 and save outputs to `output/tile_0/`. Check the log in `logs/tile_0.log`.

### 2. Verify the Output
```bash
python test_e2e_verification.py output/tile_0
```

### 3. Run All Tiles
```bash
./run_test_e2e.sh
```

## Expected Outputs

After running the pipeline, you should see:

```
output/tile_0/
├── visualization/
│   └── rgb_images/
│       ├── extended_extracted_channel_image.png  # Whole tile visualization
│       └── patch_*.png                           # Individual patches (if split_mode != full_image)
├── segmentation/
│   ├── codex_patches_with_segmentation.pkl      # Segmentation results
│   └── *.tiff                                   # Segmentation masks (optional)
├── cell_profiling/
│   ├── cell_expression_matrix.csv               # Cell x Antibody matrix
│   ├── cell_metadata.csv                        # Cell metadata
│   └── patches_metadata.csv                     # Patch information
└── copied_config.yaml                           # Copy of the configuration used
```

## Troubleshooting

1. **Out of Memory**: The tiles are small (512x512) but with 41 channels. If you encounter memory issues:
   - Process one tile at a time
   - Reduce the number of channels in the config

2. **Segmentation Model Not Found**: Ensure the DeepCell model is available at:
   `/workspaces/codex-analysis/data/deepcell/v7/MultiplexSegmentation`

3. **Missing Dependencies**: Make sure all required packages are installed:
   ```bash
   pip install -r /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/requirements.txt
   ```

## Customization

To adjust the test:

1. **Change tile size**: Edit `create_test_dataset.py` and regenerate tiles
2. **Modify channels**: Edit the `channels` section in `tile_*/config.yaml`
3. **Enable/disable features**: Adjust settings in the config files

## Development Workflow

1. Make changes to the pipeline code
2. Run `./run_single_tile_test.sh` to test quickly
3. Verify outputs with `test_e2e_verification.py`
4. Once working, test all tiles with `./run_test_e2e.sh`
5. Check visualization outputs to ensure quality
