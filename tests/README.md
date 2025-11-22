# Tests Overview

This directory contains the unit and integration style checks that exercise the
core Aegle pipeline modules.

```
# Get into the pipeline root directory
cd 0-phenocycler-penntmc-pipeline
# Run the all tests
python -m unittest discover -s tests
# Run the tests for a specific module
python -m unittest tests.test_segment
python -m unittest tests.test_cell_profiling_features
```

## test_segment.py
Exercising the segmentation repair stack: we build deterministic masks, run
`repair_masks_batch`, and assert on trimmed nuclei, matched fractions,
unmatched counts, and boundary labelling. The tests rely on
`tests/utils/make_segmentation_result` to assemble repair-ready dictionaries.

## test_cell_profiling_features.py
Uses the synthetic data factory to assert that `extract_features_v2_optimized`
computes whole-cell, nucleus, and cytoplasm statistics as expected. The factory
lives in `tests/utils/synthetic_data_factory.py` and exposes convenient builders
for single cells, nucleus-only cells, and empty patches.

To aid manual inspection, you can render the synthetic patches to disk with:

```bash
cd 0-phenocycler-penntmc-pipeline
python - <<'PY'
from pathlib import Path
from tests.utils import (
    make_empty_patch,
    make_nucleus_only_patch,
    make_single_cell_patch,
    render_patch,
)

out_dir = Path('debug_synthetic_previews')
out_dir.mkdir(exist_ok=True)

render_patch(
    make_single_cell_patch(
        shape=(48, 48),
        channels=("chan0", "chan1"),
        nucleus_intensity={"chan0": 50.0, "chan1": 25.0},
        cytoplasm_intensity={"chan0": 5.0, "chan1": 10.0},
    ),
    out_dir / 'single_cell.png',
)

render_patch(
    make_nucleus_only_patch(
        shape=(40, 40),
        channels=("chan0",),
        nucleus_intensity={"chan0": 17.0},
    ),
    out_dir / 'nucleus_only.png',
)

render_patch(
    make_empty_patch(shape=(32, 32), channels=("chan0", "chan1")),
    out_dir / 'empty_patch.png',
    overlay_masks=False,
)

print('Previews written to', out_dir.resolve())
PY
```

Running the snippet creates PNG previews in
`0-phenocycler-penntmc-pipeline/debug_synthetic_previews/`, which mirrors the
scenarios covered by the tests.

## Environment gates
- `RUN_INTEGRATION=1` enables mocked pipeline integration tests.
- `RUN_ANALYSIS_INT=1` enables the analysis integration stub.
- `RUN_E2E=1` enables the shell smokes under `tests/main/`.
Default runs (without these env vars) execute only the fast unit-style suite.

## Real-assets validator
Use `tests/utils/validate_real_assets.py` to compare real configs/CSV/TIFF
against fixture expectations:
```bash
python tests/utils/validate_real_assets.py --config path/to/config.yaml --data-dir /path/to/data --out-dir /path/to/out
```
## Shell harnesses under `tests/main`
The shell scripts orchestrate end-to-end smoke tests for different experiment
setups (e.g. oocytes, FT patches). They wrap `run_main_test_*.sh` scripts and
are intended for manual or CI-driven pipeline executions.

## Additional utilities
The `tests/utils` package collects helpers used across multiple test modules.
Currently it provides the synthetic data factory and the optional `render_patch`
visualisation entry-point.
