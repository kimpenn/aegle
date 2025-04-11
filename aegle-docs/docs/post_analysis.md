---
sidebar_position: 5
---
# Post-Analysis Visualization

Launch napari with the following data

`codex_analysis.h5ad`, `original_seg_res_batch.npy`, `matched_seg_res_batch.npy`, `*.ome.tiff`

Example ipynb code block

```python
import logging
import os
import sys
import anndata
import numpy as np
import pandas as pd
from tifffile import imread

# read anndata from h5ad
file_name = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/out/analysis/test_analysis/exp-1/codex_analysis.h5ad"
adata = anndata.read_h5ad(file_name)
cluster_int = adata.obs["leiden"].astype(int).values


patch_index = 0
# Read the original segmentation results
pkl_file = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/out/main/D18_Scan1_0/original_seg_res_batch.npy"
logging.info(f"[INFO] Loading codex_patches from {pkl_file}")
seg_res_batch = np.load(pkl_file, allow_pickle=True)
seg_data = seg_res_batch[patch_index]
cell_mask = seg_data["cell"]
nuc_mask = seg_data["nucleus"]

# Read the repaired segmentation results
pkl_file = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/out/main/D18_Scan1_0/matched_seg_res_batch.npy"
logging.info(f"[INFO] Loading codex_patches from {pkl_file}")
repaired_seg_res_batch = np.load(pkl_file, allow_pickle=True)
repaired_seg_data = repaired_seg_res_batch[patch_index]
repaired_cell_mask = repaired_seg_data["cell_matched_mask"]
repaired_nuc_mask = repaired_seg_data["nucleus_matched_mask"]

# Load the OME-TIFF image (you may need to provide full path)
file_name = "/Users/kuangda/Developer/1-projects/4-codex-analysis/0-phenocycler-penntmc-pipeline/D18_Scan1_tissue_0.ome.tiff"
ome_image = imread(file_name)  # shape might be (C, Z, Y, X) or (C, Y, X)
# Adjust axis order if necessary (e.g., select 2D image or max projection)
# Example: If image is (C, Y, X), and you want DAPI channel (say channel 0)
image_2d = ome_image[0]  # or np.max(ome_image, axis=0) for Z-projection
image_6 = ome_image[6]
image_7 = ome_image[7]
image_40 = ome_image[40]


import napari
viewer = napari.Viewer()
viewer.add_image(image_2d, name="DAPI / Tissue Image")
viewer.add_image(image_6, name='Pan-Cytokeratin')
viewer.add_image(image_7, name='Collagen IV')
viewer.add_image(image_40, name='FUT4')
# Ensure the mask is integer type
cell_mask_int = cell_mask.astype(np.int32)
viewer.add_labels(cell_mask_int, name="Cell Mask")

# Optional: Add nucleus mask if desired
nuc_mask_int = nuc_mask.astype(np.int32)
viewer.add_labels(nuc_mask_int, name="nucleus Mask")

# Ensure the mask is integer type
repaired_cell_mask_int = repaired_cell_mask.astype(np.int32)
viewer.add_labels(repaired_cell_mask_int, name="Cell Mask Reqaired")

# Optional: Add nucleus mask if desired
repaired_nuc_mask_int = repaired_nuc_mask.astype(np.int32)
viewer.add_labels(repaired_cell_mask_int, name="nucleus Mask Reqaired")

napari.run()
```