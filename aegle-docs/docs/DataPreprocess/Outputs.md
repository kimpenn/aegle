---
sidebar_position: 2
---

# Outputs

After running this preprocessing module, you will have:

## Tissue Region Files
- **OME-TIFF tissue crops**: `{OUT_DIR}/{EXP_ID}/{base_name}_tissue_{i}.ome.tiff`
  - Each crop is a multi-channel image representing a distinct tissue region
  - Images are saved in OME-TIFF format (C,H,W) with LZW compression
  - If `skip_roi_crop` is enabled, a single full image will be saved as `{base_name}_full.ome.tiff`

- **Visualization (optional)**: `{OUT_DIR}/{EXP_ID}/tissue_masks_preview.png`
  - Visual representation of detected tissue regions if `visualize` is enabled

## Antibody Data
- **Antibody mapping**: `{OUT_DIR}/extras/antibodies.tsv`
  - Tab-separated file with columns: `version`, `channel_id`, and `antibody_name`
  - Maps each channel to its corresponding antibody marker

- **OME-XML metadata**: `{data_path}/{base_name}.xml`
  - Extracted OME-XML metadata from the original QPTIFF file
  - Contains detailed information about channels, dimensions, etc.

## Log Files
- Detailed logs for each experiment are saved to `{LOG_DIR}/{EXP_ID}.log`
- Contains timing information, processing steps, and any errors/warnings