---
sidebar_position: 5
---

# Segmentation Details

This document provides detailed information about the segmentation process, which is the implmenetation of `run_cell_segmentation` in `segment.py`.

## Segmentation Workflow

The `run_cell_segmentation` function orchestrates the entire segmentation process:

1. **Patch Selection**: Only processes patches marked as informative (not empty or noisy) from Phase B
2. **Model Loading**: Loads the pre-trained DeepCell Mesmer model for multiplex segmentation
3. **Segmentation Processing**: Applies the model to extract cell and nucleus masks
4. **Mask Repair**: Matches cells to nuclei and repairs segmentation artifacts
5. **Result Storage**: Saves segmentation masks and metadata

## Technical Details

The `run_cell_segmentation` function only processes patches marked as informative (not empty or noisy) from Phase B. The informative patches are marked as `is_informative` in `patches_metadata`. We prepare `valid_patches` from this step. The patches can be in memory or on disk based on the `split_mode` in the config. If the patches are on disk, we will only keep the indices of the informative patches since all the patches cannot fit into memory. If the patches are in memory, we will keep the patches in memory and save its reference in `codex_patches.valid_patches`. 

The next step is to apply segmentation model to the patches. If the patches are on disk, we will iterate over the indices of the informative patches and load the patches from disk one by one. If the patches are in memory, we will pass patches to segmentation model in a batch. 

> The iterative calling segmentation model for disked patches might be suboptimal in terms of speed. We could improve this by loading pathces in a batch and call the segmentation. Apply all the patches into Segmentation model may also exhust all the G-RAM. For now, we use memory map to save G-RAM. So in the future, we should consider a better strategy to handle this step. To achieve this, we need to compile a dataset with different size of images to explore different scenarios in terms of number of patches and patch size.

The segmentation is implmented in the `segment` function. It calls the `Mesmer` model to segment the patches. The `Mesmer` model is a pre-trained model that can be found in `deepcell.applications`. The `segment` function is implemented in `segment.py`. The function call is like this:

```python
    segmentation_predictions = model.predict(
        valid_patches, image_mpp=image_mpp, compartment="both"
    )
```

The expected size of `valid_patches` is `(n_patches, height, width, 2)`. The first channel is the nucleus channel and the second channel is the whole cell channel. The `image_mpp` is the microns per pixel of the image. The `compartment` is the compartment to segment. It can be `"whole-cell"`, `"nuclear"`, or `"both"`.

We use `_separate_batch` function to separate the segmentation predictions into cell and nucleus masks. Then use `get_boundary` function, which is based on `skimage.segmentation.find_boundaries` to get the boundaries of the cell and nucleus masks. The output of `segment` function is a list of dictionaries, each containing the cell and nucleus masks and the boundaries as shown below:

```python
    segmentation_output = [
        {
            "cell": cell_mask,
            "nucleus": nucleus_mask,
            "cell_boundary": cell_boundary,
            "nucleus_boundary": nucleus_boundary,
        }
    ]
```

The next part is to repair the masks. We use `repair_masks_batch` function to repair the masks. The function is implemented in `repair_masks.py`. 

The `repair_masks_batch` function performs quality control by matching cells to their corresponding nuclei across all segmentation patches. For each patch, it extracts pixel coordinates for cells, nuclei, and cell membranes, then iteratively finds the best cell-nucleus pairs by minimizing mismatch fraction (nucleus pixels outside the cell interior). The function uses a repair strategy that retains partial matches when the nucleus partially overlaps with the cell, filtering out segmentation artifacts where cells and nuclei don't properly correspond. The output includes matched cell and nucleus masks, cytoplasm masks (cell minus nucleus), and quality metrics including matched fractions and detailed statistics, ensuring downstream analysis only uses properly validated cell-nucleus pairs. 

```python
codex_patches.repaired_seg_res_batch
codex_patches.seg_res_batch
codex_patches.patches_metadata
```
Both `repaired_seg_res_batch` and `seg_res_batch` are saved in `codex_patches.save_seg_res()` as `matched_seg_res_batch.pickle` and `original_seg_res_batch.pickle` respectively. Metadata is also updated in disk in `codex_patches.save_metadata()` as `patches_metadata.csv`.
