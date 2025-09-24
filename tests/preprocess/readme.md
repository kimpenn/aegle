# Preprocess Test Suite

This directory contains the end-to-end experiments we use to validate the preprocess stage. The tests are orchestrated by `run_preprocess_tests.py`, which will clean previous outputs, call the shell wrappers inside `scripts/`, and perform file-based assertions to ensure each stage completed successfully.

## Structure
- `run_preprocess_test.sh` – legacy bash launcher (kept for reference); the new Python runner invokes the same scripts and adds cleanup + assertions.
- `run_preprocess_tests.py` – main orchestration entrypoint. Discovers experiments under `exps/configs/preprocess/test/`, runs preprocess, and validates the outputs.

## Experiments
Each experiment corresponds to a directory under `exps/configs/preprocess/test/<experiment_name>/`. The YAML config points to the test TIFF and the manual annotations. Currently we have:

- `manual_geojson_combined` – all ROI polygons bundled in a single GeoJSON (`data/test/preprocess/manual_rois/preprocessed_roi/D11_13_Scan1_Oocytes.geojson`).

Additional experiments (e.g. `manual_napari_combined`, `manual_geojson_split`) can be added by providing their config and test data. The Python runner will pick them up automatically.

## Running the tests
```bash
cd /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline
python tests/preprocess/run_preprocess_tests.py
# run single experiment
python tests/preprocess/run_preprocess_tests.py --experiments manual_geojson_split
# run multiple experiments
python tests/preprocess/run_preprocess_tests.py --experiments manual_geojson_combined manual_geojson_split
```

Optional arguments:
- `--experiments manual_geojson_combined` – run a subset of experiments.
- `--skip-cleanup` – keep existing outputs (useful for debugging).

> **Note:** The runner deletes previously generated ROI OME-TIFF/JPEGs, `manual_polygon_overlay.png`, and `extras/` artifacts for the targeted experiments to avoid mixing with stale data.

## Expected Outputs
After each run the script checks for:
- `*_manual_<label>.ome.tiff` ROI crops in `data/`.
- Overview JPEGs produced by `run_generate_overview.sh`.
- `manual_polygon_overlay.png` visualization.
- `extras/antibodies.tsv` produced by the antibody extraction step.
- Completion markers in the per-experiment log under `logs/preprocess/test/`.

If any assertion fails the script exits with a non-zero status and prints the failing experiment.

To give permission to the scripts, run:
```bash
chmod +x scripts/run_extract_tissue.sh scripts/run_extract_antibody.sh scripts/run_generate_overview.sh
```