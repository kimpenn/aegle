# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Aegle is a 2D PhenoCycler image processing and analysis pipeline for spatial biology. It handles multi-channel microscopy images (.qptiff files), performs cell segmentation using Mesmer/DeepCell, extracts single-cell features, and provides downstream analysis tools for quality control, clustering, and marker testing.

## Core Architecture

### Data Flow Pipeline
The pipeline operates in distinct stages, each with its own entry point:

1. **Preprocessing** (`src/extract_tissue_regions.py`, `src/extract_antibodies.py`): Extract individual tissues from multi-tissue scans and retrieve antibody metadata
2. **Main Pipeline** (`src/main.py` → `aegle/pipeline.py`): Load image → create patches → segment cells → profile features
3. **Analysis** (`src/run_analysis.py` → `aegle_analysis/analysis_handler.py`): Quality control, clustering, differential marker analysis

### Key Classes

**CodexImage** (`aegle/codex_image.py`): Loads .qptiff/.tiff files, manages channel information from antibodies.tsv, and configures patching parameters based on split_mode (full_image, halves, quarters, patches).

**CodexPatches** (`aegle/codex_patches.py`): Divides the image into processable patches, manages memory-mapped storage for large images, stores segmentation masks and metadata per patch. Can be rehydrated from disk via `load_from_outputs()` to resume downstream stages without re-running segmentation.

**Feature Extraction** (`aegle/extract_features.py`): Implements `extract_features_v2_optimized()` which computes per-cell intensity statistics (mean, std, percentiles) and morphology features from nucleus and cell masks. Supports optional Laplacian variance and coefficient of variation calculations controlled by `profiling.features.compute_laplacian` and `profiling.features.compute_cov` flags.

### Configuration System

All pipeline behavior is controlled through YAML files located in `exps/configs/`. Templates in `exps/templates/` (e.g., `main_template.yaml`) define the full configuration schema with inline documentation. Key sections:

- `data`: file paths, microns-per-pixel, channel stats generation
- `channels`: nuclear_channel (e.g., DAPI) and wholecell_channel (e.g., Pan-Cytokeratin)
- `patching`: split_mode, patch dimensions, overlap
- `segmentation`: model path, output options, compression settings
- `profiling.features`: compute_laplacian, compute_cov, channel_dtype (controls memory/speed tradeoffs)

## Development Commands

### Installation
```bash
# Install aegle package in editable mode
python -m pip install -e .
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_cell_profiling_features.py

# Run specific test class or function
python -m pytest tests/test_extract_features_flags.py::TestExtractFeaturesFlags::test_skip_laplacian_and_cov_drop_columns
```

### Main Pipeline Execution
```bash
# Full pipeline with template config
python src/main.py \
  --config_file exps/templates/main_template.yaml \
  --data_dir /path/to/data \
  --out_dir out/experiment_name \
  --log_level INFO

# Resume from cell profiling stage (after segmentation is complete)
python src/main.py \
  --config_file exps/configs/main/experiment/config.yaml \
  --data_dir /path/to/data \
  --out_dir out/experiment_name \
  --resume_stage cell_profiling

# Using shell wrapper scripts (for specific cohorts)
bash run_main_ft.sh <EXP_SET_NAME> <EXP_ID> <ROOT_DIR> <DATA_DIR> <CONFIG_DIR> <OUT_DIR> [LOG_LEVEL]
```

### Preprocessing
```bash
# Extract tissues from multi-tissue scans
bash scripts/run_extract_tissue.sh

# Extract antibody names from .qptiff metadata
bash scripts/run_extract_antibody.sh
```

### Analysis
```bash
# Run downstream analysis on pipeline outputs
python src/run_analysis.py \
  --config_file exps/templates/analysis_template.yaml \
  --data_dir out/main/experiment_name \
  --output_dir out/analysis/experiment_name

# Using shell wrapper
bash scripts/run_analysis.sh
```

## Development Guidelines

### Git Commit Messages

**IMPORTANT**: Keep commit messages **concise**.

- **First line**: Short summary (50-72 characters max)
- **Body** (optional): Brief explanation of what and why (not how)
- **Keep it simple**: 1-3 sentences total is usually sufficient
- **Don't**: Write multi-paragraph essays explaining every detail

**Good examples:**
```
Add GPU acceleration for cell profiling

Fix typo in extract_features.py logger call

Update .gitignore to exclude .claude/ directory
```

**Bad example:**
```
Add GPU acceleration for cell profiling with 20-50x speedup

Implemented GPU-accelerated feature extraction using CuPy to dramatically
speed up cell profiling for large samples (>1M cells). The optimization
reduces D18_0 processing time from hours (OOM) to 48 minutes.

Phase 1: Configuration & Setup
- Add profiling.features config section with GPU flags (use_gpu, gpu_batch_size)
- Create aegle/gpu_utils.py with GPU detection and memory management
...
[12 more paragraphs]
```

## Project Structure

- `aegle/`: Core pipeline modules (image loading, patching, segmentation, profiling)
- `aegle_analysis/`: Downstream analysis (QC, clustering, differential markers, visualization)
- `src/`: CLI entry points that wrap aegle package functions
- `scripts/`: Shell scripts for common workflows
- `exps/`: Configuration management
  - `templates/`: Reference YAML configs with documentation
  - `configs/`: Experiment-specific configs organized by stage/cohort/scan
- `tests/`: Unit and integration tests
  - `utils/`: Shared test fixtures and synthetic data factory
  - `preprocess/`, `main/`, `analysis/`: Stage-specific tests
- `out/`: Pipeline outputs (git-ignored)
- `logs/`: Execution logs (git-ignored)
- `data/`: Input data directory (git-ignored)

## Key Implementation Details

### Production Deployment: Full Image Mode

**IMPORTANT:** Most production pipelines use `split_mode: full_image` to avoid losing cells at patch boundaries. This architectural decision has critical implications for optimization:

- **Only 1 patch processed per sample** - no parallelization across patches possible
- **All optimization must focus on single-patch efficiency** - cannot distribute work across patches
- **Memory usage proportional to full image size** - entire image must fit in memory
- **GPU acceleration is critical for large samples** - with >1M cells and >40 channels, CPU-only processing can take hours

**Performance Considerations for Full Image Mode:**
- For samples with >1M cells: Enable GPU acceleration (15-20x speedup)
- Use memory-mapped loading with `mmap` to avoid OOM
- Use `float32` instead of `float64` (2x memory reduction)
- Disable expensive features (`compute_laplacian`, `compute_cov`) if not needed
- Typical large sample (1.8M cells, 45 channels): ~20 min with GPU vs ~6-8 hours CPU

### Memory Management
The pipeline uses memory-mapped arrays (`np.memmap`) and optional compression (zstd) to handle large multi-channel images. The `cache_all_channel_patches` flag controls whether decompressed patches are kept in a temporary cache for reuse during visualization.

### Patching Modes
- `full_image`: Process entire image as one patch (**standard for production**)
- `halves`/`quarters`: Split image along specified direction (debugging/development)
- `patches`: Tile image into overlapping patches of specified size (legacy)

### Feature Computation Flags
Recent optimization work (see `tests/test_extract_features_flags.py`) allows disabling expensive features:
- `compute_laplacian: False` skips Laplacian variance (spatial texture measure)
- `compute_cov: False` skips coefficient of variation
- `channel_dtype: float32` reduces memory usage vs. float64

### Resume Capability
Use `--resume_stage cell_profiling` to skip segmentation and recompute only cell profiling. This requires that segmentation outputs (masks and metadata) already exist in the output directory.

## Testing Strategy

The test suite uses a synthetic data factory (`tests/utils/synthetic_data_factory.py`) to generate reproducible test images with known properties. When modifying feature extraction or segmentation code, regenerate preview images to visually verify changes:

```bash
cd tests
python utils/synthetic_data_factory.py  # Regenerates debug_synthetic_previews/
```

## Common Pitfalls

1. **Missing antibodies.tsv**: Ensure the TSV has columns `antibody_name` and matches channel count in the image
2. **Split mode mismatch**: When resuming, the split_mode in config must match the original run
3. **Memory errors**: For large images, use `split_mode: quarters` or `patches`, or reduce `channel_dtype` to `float32`
4. **GPU allocation**: Segmentation requires GPU. Set `TF_GPU_ALLOCATOR=cuda_malloc_async` for better memory handling
5. **Path assumptions**: Scripts expect to be run from the repository root (`/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline`)

## Output Artifacts

Main pipeline (`src/main.py`) produces:
- `patches_metadata.csv`: Patch locations, QC status, cell counts
- `matched_seg_res_batch.pickle.zst`: Compressed segmentation masks
- `segmentation_visualization/`: RGB overlays of masks on images
- `cell_profiling/`: Cell-by-antibody matrices and metadata
  - `cell_overview.csv`: Per-cell morphology and intensity features
  - `markers.csv`: Mean intensity values per cell per channel

Analysis pipeline (`src/run_analysis.py`) produces:
- QC plots (pixel-level, cell-level)
- Clustering results and visualizations
- Differential marker analysis tables
- Summary statistics and reports
