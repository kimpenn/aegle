# Optional Cell Level QC Filtering

## Overview
During the cell profiling stage of the pipeline, various morphological metrics (such as eccentricity, convex area, area, and solidity) are calculated for both the cell and its nucleus. **In addition, the pipeline calculates a "retained fraction" (e.g., `nucleus_retained_fraction`, `cell_retained_fraction`), representing the proportion of original segmentation pixels kept after the mask repair and matching stage.**

The **Optional Cell Level QC Filter** module (`aegle/qc_filter_cells.py`) utilizes these specific metrics to systematically identify and remove anomalous cells. This helps to clean up segmentation artifacts such as:
* Cells that are unrealistically small or large.
* Highly irregular or fragmented cells (low solidity).
* Obvious artifacts (like perfectly circular false positives).
* **Heavily trimmed nuclei or mismatched cells** that lost too much of their original area during the repair stage (i.e. nucleus leaked outside the cell membrane).

By filtering out these low-quality segmentations, you can significantly improve the reliability of downstream spatial and single-cell expression analyses.

## How it works
The filter acts as a post-processing step. It takes the un-filtered output CSVs from the `cell_profiling` directory (specifically `cell_metadata.csv` and `cell_by_marker.csv`) and applies a set of configurable threshold boundaries to whichever metrics you specify.

Because the filtering logic dynamically reads the columns in `cell_metadata.csv`, you can filter on literally *any* column present in that file (including the newly added `nucleus_retained_fraction`). Cells that meet all criteria are kept, and the module outputs new, leaner CSV files prefixed with `filtered_`.

## Usage

### 1. Standalone Script (Recommended)
Because tuning QC thresholds often requires iterative trial and error, it is highly recommended to run this filter as a standalone script on the outputs of a completed pipeline run. This way, you do not need to re-run the heavy segmentation or feature extraction steps.

```bash
python aegle/qc_filter_cells.py \
  --input_dir out/your_run/cell_profiling \
  --config exps/configs/your_filter_config.yaml
```

* **`--input_dir`**: The directory containing `cell_metadata.csv` and `cell_by_marker.csv`.
* **`--config`**: A YAML file containing your chosen filtering thresholds.

### 2. Integration with `pipeline.py`
If your QC thresholds are entirely finalized and you want the filtering to happen automatically at the end of a big batch-processing job, you can trigger the module directly in Python:

```python
from aegle.qc_filter_cells import apply_morphology_filters

# In pipeline.py, assuming profiling_out_dir is defined
if config.get("qc_filtering"):
    apply_morphology_filters(profiling_out_dir, config["qc_filtering"])
```

## Configuration Reference
Your YAML configuration file must nest its rules under the `qc_filtering` key. For each metric, you can specify a `min` bound, a `max` bound, or both.

**Example `your_filter_config.yaml`:**
```yaml
qc_filtering:
  rules:
    nucleus_area:
      min: 15       # Remove tiny artifacts
      max: 1000     # Remove massive clumps of unresolved nuclei
    nucleus_solidity:
      min: 0.8      # Ensure the nucleus has a regular, non-fragmented shape
    cell_eccentricity:
      max: 0.95     # Remove perfectly linear streaks or artifacts
    nucleus_retained_fraction:
      min: 0.7      # Remove nuclei where more than 30% of pixels were trimmed during repair
    cell_retained_fraction:
      min: 0.8      # Remove cells heavily modified during mask alignment
```

*Note: The keys under `rules` must exactly match the morphology column names found in your `cell_metadata.csv`.*
