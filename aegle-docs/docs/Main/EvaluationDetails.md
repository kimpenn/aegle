---
sidebar_position: 6
---
# Evaluation Details

This step is the automated segmentation quality assessment using the `run_seg_evaluation` function implemented in `evaluation.py`. This evaluation system provides comprehensive metrics to assess the quality of cell segmentation results.

Our function is based on `single_method_eval` from [CellSegmentationEvaluator](https://github.com/murphygroup/CellSegmentationEvaluator/blob/6def33dd172ad9074bd856399535a5deea3e3fd6/SimpleCSE/read_and_eval_seg.py#L55).

## Evaluation Workflow

We use `concurrent.futures.ProcessPoolExecutor` to evaluate the segmentation results of patches in parallel. Each evaluation process (`_process_single_patch`) follows these key steps:

1. **Patch Filtering**: Only evaluates informative patches (marked as `is_informative` in metadata)
2. **Cell Count Validation**: Skips patches with fewer than 20 cells to ensure statistical reliability
3. **Parallel Processing**: Uses 2 worker processes for efficient evaluation across multiple patches
4. **Comprehensive Metrics**: Calculates 14 different quality metrics for each patch
5. **Quality Score Generation**: Combines metrics into a single quality score using PCA-based model

## Key Functions

### `run_seg_evaluation()`
Main orchestration function that:
- Extracts repaired segmentation results and image data for informative patches
- Reshapes image arrays from `(batch, w, h, c)` to `(batch, c, w, h)` format
- Distributes evaluation tasks across parallel workers using `ProcessPoolExecutor`
- Stores evaluation results in `codex_patches.seg_evaluation_metrics`

### `_process_single_patch()`
Worker function that processes individual patches:
- Validates cell count (minimum 20 cells required)
- Prepares image dictionary with proper formatting
- Calls `evaluate_seg_single()` for detailed metric calculation
- Returns quality score and comprehensive metrics or NaN for failed evaluations

### `evaluate_seg_single()`

This function is the core evaluation function that computes detailed quality metrics.
- Takes matched cell masks, nucleus masks, and cytoplasm masks
- Processes original segmentation results for comparison
- Applies image thresholding and foreground/background separation

#### Quality metrics:
1. **Cell Density Metrics**
   - `NumberOfCellsPer100SquareMicrons`: Cell density normalized by area
2. **Coverage Metrics**
   - `FractionOfForegroundOccupiedByCells`: How well cells cover tissue regions
   - `1-FractionOfBackgroundOccupiedByCells`: Background cleanliness
   - `FractionOfCellMaskInForeground`: Mask accuracy in tissue regions
3. **Cell Size Uniformity**
   - `1/(ln(StandardDeviationOfCellSize)+1)`: Consistency of cell sizes
4. **Cell-Nucleus Matching**
   - `FractionOfMatchedCellsAndNuclei`: Success rate of cell-nucleus pairing
5. **Foreground Quality**
   - `1/(AvgCVForegroundOutsideCells+1)`: Uniformity of tissue background
   - `FractionOfFirstPCForegroundOutsideCells`: Principal component analysis of background
6. **Cell Type Clustering Metrics** (for nucleus and cytoplasm compartments)
   - `1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)`: Cell type consistency
   - `AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters`: PCA-based clustering quality
   - `AvgSilhouetteOver2~10NumberOfClusters`: Clustering separation quality

#### Quality Score Calculation
The final quality score is generated using a pre-trained PCA model (`2Dv1.5`) that:
- Standardizes all 14 metrics using pre-computed mean and scale parameters
- Projects metrics onto 2 principal components
- Calculates exponential weighted score: `exp(PC1 × variance_ratio_1 + PC2 × variance_ratio_2)`
- Returns a single quality score representing overall segmentation quality

## Technical Details

### Image Processing
- Uses mean thresholding for foreground/background separation
- Applies morphological operations with disk sizes `(1, 2, 20, 10)` and area sizes `(20000, 1000)`
- Converts pixel sizes from micrometers to nanometers (`config["data"]["image_mpp"] * 1000`)

### Parallel Processing
- Uses `concurrent.futures.ProcessPoolExecutor` with 2 workers
- Maintains result order through indexed futures mapping
- Handles exceptions gracefully with NaN placeholders

### Error Handling
- Patches with insufficient cells (< 20) receive NaN quality scores
- Failed evaluations are logged but don't interrupt the overall process
- Results maintain consistent indexing with input patches

## Output Format

The evaluation results are stored in `codex_patches.seg_evaluation_metrics` as a list of dictionaries, where each dictionary contains:

```python
{
    "Matched Cell": {
        "NumberOfCellsPer100SquareMicrons": float,
        "FractionOfForegroundOccupiedByCells": float,
        "1-FractionOfBackgroundOccupiedByCells": float,
        "FractionOfCellMaskInForeground": float,
        "1/(ln(StandardDeviationOfCellSize)+1)": float,
        "FractionOfMatchedCellsAndNuclei": float,
        "1/(AvgCVForegroundOutsideCells+1)": float,
        "FractionOfFirstPCForegroundOutsideCells": float
    },
    "Nucleus (including nucleus membrane)": {
        "1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)": float,
        "AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters": float,
        "AvgSilhouetteOver2~10NumberOfClusters": float
    },
    "Cell Not Including Nucleus (cell membrane plus cytoplasm)": {
        "1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)": float,
        "AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters": float,
        "AvgSilhouetteOver2~10NumberOfClusters": float
    },
    "QualityScore": float
}
```

This comprehensive evaluation system enables automated quality assessment of segmentation results, helping identify well-segmented patches and potential issues in the segmentation pipeline. Currently, we use pickle to save the evaluation results.

```python
with open(os.path.join(args.out_dir, "seg_evaluation_metrics.pkl.gz"), "wb") as f:
    pickle.dump(codex_patches.seg_evaluation_metrics, f)
```