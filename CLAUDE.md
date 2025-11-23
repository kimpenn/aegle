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

## Exploration and Optimization Workflow

**IMPORTANT**: When exploring, profiling, or optimizing the codebase, follow this structured workflow to maintain a clean separation between production code and exploratory work.

### Scratch Folder for Exploration

All exploration, profiling, benchmarking, and analysis work should be placed in the `scratch/` directory to avoid cluttering the production codebase:

- **What goes in scratch/**:
  - Profiling scripts and outputs (`.prof` files, logs)
  - Experimental benchmark scripts
  - Analysis and exploration scripts
  - Investigation reports and findings
  - Demo/prototype code
  - Temporary validation scripts

- **What stays in production**:
  - Core implementation (`aegle/`, `aegle_analysis/`)
  - Production test suite (`tests/test_*.py`)
  - Production benchmarks (`tests/benchmark_*.py`)
  - Test fixtures and utilities (`tests/utils/`)
  - Package configuration (`setup.py`, `pyproject.toml`)

### Planning Before Execution

**REQUIRED**: Before starting any significant optimization or exploration work, create a detailed plan document in `scratch/PLANS_<name>.md`:

1. **File naming**: Use descriptive names (e.g., `PLANS_repair_optimization.md`, `PLANS_gpu_acceleration.md`)

2. **Plan structure**:
   ```markdown
   # Project Name

   ## Context and Background
   - Current state and problem description
   - Performance metrics (before state)
   - Why this work is needed

   ## Goals and Success Criteria
   - Clear, measurable objectives
   - Acceptance criteria
   - Target metrics

   ## Phases and Tasks

   ### Phase 0: Setup
   - [ ] Task 0.1: Description
     - Subtasks with clear deliverables
     - Dependencies: (list task IDs this depends on)

   ### Phase 1: Implementation
   - [ ] Task 1.1: Description
     - Dependencies: 0.1
   - [ ] Task 1.2: Description
     - Dependencies: 0.1
   - [ ] Task 1.3: Description
     - Dependencies: 1.1, 1.2

   ## Parallelization Strategy
   - Tasks 1.1 and 1.2 can run in parallel (both depend only on 0.1)
   - Task 1.3 must wait for 1.1 and 1.2 to complete

   ## Progress Tracking
   [Update this section during execution with discoveries]
   ```

3. **Dependency analysis**: Explicitly identify which tasks can run in parallel vs. sequentially to maximize subagent parallelization

4. **Example**: See `scratch/PLANS_repair_optimization.md` for a complete real-world example

### Progress Tracking

During execution, continuously update the plan document:

1. **Mark completed tasks**: Change `[ ]` to `[x]` as tasks finish
2. **Document discoveries**: Add findings, bottlenecks identified, unexpected issues
3. **Record results**: Include performance numbers, test results, validation outcomes
4. **Update dependencies**: Adjust task lists if new dependencies are discovered
5. **Add new tasks**: If investigation reveals additional work needed

### Organization After Completion

After completing exploration work, organize all artifacts into logical subdirectories within `scratch/`:

- `scratch/phase0-setup/` - Initial setup and baseline work
- `scratch/phase1-implementation/` - Implementation artifacts
  - `profiling/` - Profiling scripts and outputs
  - `benchmarks/` - Performance benchmarks
  - `validation/` - Validation scripts
- `scratch/reports/` - Analysis reports and findings
- `scratch/demos/` - Demo and prototype scripts

Create documentation files:
- `scratch/README.md` - Overview and navigation guide
- `scratch/FILE_INVENTORY.md` - Detailed file listing (optional for large projects)

### Working with Claude Code Subagents

When orchestrating parallel work with subagents:

1. **Reference the plan**: Provide subagents with the relevant section of `PLANS_x.md`
2. **Specify dependencies**: Tell subagents which prior tasks must complete first
3. **Define deliverables**: Clearly state what outputs are expected (scripts, reports, data)
4. **Coordinate outputs**: Ensure subagents write to designated `scratch/` subdirectories

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
- `scratch/`: Exploration, profiling, and optimization work (see Exploration and Optimization Workflow)
  - `PLANS_*.md`: Detailed project plans
  - `phase*-*/`: Phase-organized exploration artifacts
  - `reports/`: Analysis reports and findings
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

### GPU Acceleration

#### Cell Profiling GPU Acceleration

Enable GPU acceleration for cell profiling (intensity feature extraction):

```yaml
profiling:
  features:
    use_gpu: true
    gpu_batch_size: 0  # Auto-size based on GPU memory
```

**Performance:** 15-20x speedup for large samples (>1M cells). A typical large sample (1.8M cells, 45 channels) processes in ~20 minutes with GPU vs ~6-8 hours CPU.

**Requirements:**
- CuPy installed: `pip install cupy-cuda12x`
- CUDA-capable GPU with sufficient VRAM
- Automatically falls back to CPU if GPU unavailable

#### Mask Repair GPU Acceleration

Enable GPU acceleration for mask repair operations (cell-nucleus matching):

```yaml
segmentation:
  repair:
    use_gpu: true
    gpu_batch_size: null  # Auto-size based on GPU memory
    fallback_to_cpu: true
    log_gpu_performance: true
```

**Performance:** 3-6x speedup for overlap computation and morphology operations. Currently, the overall pipeline speedup is ~1x because cell matching (95% of runtime) remains CPU-bound. Future optimization could GPU-accelerate the matching loop for additional gains.

**Requirements:**
- CuPy installed: `pip install cupy-cuda12x`
- CUDA-capable GPU with sufficient VRAM
- Automatically falls back to CPU if GPU unavailable

**Configuration Options:**
- `use_gpu`: Enable/disable GPU acceleration (default: false for backward compatibility)
- `gpu_batch_size`: Number of cells to process per GPU batch (null = auto-detect, recommended)
- `fallback_to_cpu`: Automatically use CPU if GPU fails (default: true, recommended)
- `log_gpu_performance`: Log detailed timing and speedup metrics (default: true)

**Logging:**
When enabled, GPU repair logs include:
- GPU availability and memory status
- Per-patch GPU usage and timing
- Speedup metrics (GPU vs CPU)
- Fallback events and reasons

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
