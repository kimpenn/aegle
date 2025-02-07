#!/usr/bin/env python

import os
import sys
import argparse
import logging
import yaml

import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt

from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.measure import label, regionprops


##################################
# (A) ARGUMENT / CONFIG HANDLING #
##################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone script to detect tissue regions from a large TIFF/QPTIFF and crop them."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file used by the main pipeline.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspaces/codex-analysis/data",
        help="Directory containing the input image file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output_tissue_regions",
        help="Output directory to save cropped tissue images.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


###########################
# (B) CORE FUNCTIONALITY  #
###########################


def load_image(image_path):
    """
    Load the image from the given path (qptiff, ome-tiff, etc.) using tifffile,
    and return it as a NumPy array with channels last.
    """
    with tiff.TiffFile(image_path) as tif_file:
        page0_tags = tif_file.pages[0].tags
        # Optional: Extract some tag info for debugging
        bits_per_sample = page0_tags.get(258, None).value if 258 in page0_tags else None
        samples_per_pixel = (
            page0_tags.get(277, None).value if 277 in page0_tags else None
        )

    # Actually load the data
    image_ndarray = tiff.imread(image_path)
    # Move channels to the last dimension if needed (shape: H x W x C).
    # (Some qptiff are in shape: C x H x W.)
    if image_ndarray.ndim == 3 and image_ndarray.shape[0] < image_ndarray.shape[-1]:
        image_ndarray = np.transpose(image_ndarray, (1, 2, 0))

    tif_image_details = {
        "DataType": image_ndarray.dtype,
        "Shape": image_ndarray.shape,
        "BitsPerSample": bits_per_sample,
        "SamplesPerPixel": samples_per_pixel,
    }
    logging.info(f"Image loaded. Details: {tif_image_details}")
    return image_ndarray, tif_image_details


def extract_tissue_regions_advanced(
    image, n_tissue, min_area=500, visualize=False, visualization_outpath=None
):
    """
    Identify and extract tissue regions from a (downsampled) grayscale image
    using Sobel + Watershed, optionally with Otsu thresholding.
    Returns up to n_tissue binary masks of shape (H, W).
    If 'visualize=True', saves a figure showing each of the top tissue masks.
    """
    if n_tissue is None:
        raise ValueError("Please specify `n_tissue`.")

    # 1) Sobel-based elevation map
    elevation_map = sobel(image)

    # 2) Markers for background=1, foreground=2
    markers = np.zeros_like(image, dtype=int)

    # Use Otsu threshold if fixed thresholds not provided
    otsu_thresh = threshold_otsu(image)
    markers[image < otsu_thresh] = 1
    markers[image >= otsu_thresh] = 2
    logging.info(f"[TissueDetect] Otsu threshold = {otsu_thresh:.5f}")

    # 3) Watershed
    segmentation = watershed(elevation_map, markers)

    # 4) Convert to binary foreground and fill holes
    binary_foreground = segmentation == 2
    binary_foreground = ndi.binary_fill_holes(binary_foreground)

    # 5) Label connected components
    labeled_tissues, num_labels = ndi.label(binary_foreground)
    logging.info(f"[TissueDetect] Number of raw connected components: {num_labels}")

    # 6) Filter by min_area, keep the largest n_tissue
    tissue_masks = []
    for lbl in range(1, num_labels + 1):
        mask = labeled_tissues == lbl
        area = np.count_nonzero(mask)
        if area >= min_area:
            tissue_masks.append((mask.astype(np.uint8), area))

    # Sort descending by area
    tissue_masks.sort(key=lambda x: x[1], reverse=True)
    top_tissue_masks = [x[0] for x in tissue_masks[:n_tissue]]

    # Optional single-figure visualization
    if visualize and len(top_tissue_masks) > 0:
        fig, axes = plt.subplots(
            1, len(top_tissue_masks), figsize=(4 * len(top_tissue_masks), 4)
        )
        if len(top_tissue_masks) == 1:
            axes = [axes]  # so we can iterate uniformly

        for i, m in enumerate(top_tissue_masks):
            axes[i].imshow(m, cmap="gray")
            axes[i].set_title(f"Tissue Region {i+1}")
            axes[i].axis("off")

        plt.tight_layout()
        if visualization_outpath:
            plt.savefig(visualization_outpath, dpi=150)
            logging.info(
                f"[Visualization] Saved tissue masks preview to {visualization_outpath}"
            )
        else:
            logging.warning("No visualization_outpath provided. Skipping figure save.")
        plt.close(fig)

    return top_tissue_masks


def find_tissue_rois_large_image(
    all_channel_image,
    downscale_factor=64,
    n_tissue=1,
    min_area=500,
    visualize=False,
    visualization_outdir=None,
):
    """
    1) Downsample the image (mean across channels or your method),
    2) Perform advanced segmentation to find tissue,
    3) Return bounding boxes of the largest tissue regions in full-resolution coords.

    If 'visualize=True', saves a subplot figure of each top mask
    in 'visualization_outdir/tissue_masks_preview.png'
    """
    H, W, C = all_channel_image.shape
    small_H, small_W = H // downscale_factor, W // downscale_factor

    # Downsample each channel
    resized_channels = []
    for i in range(C):
        ch_resized = cv2.resize(
            all_channel_image[..., i], (small_W, small_H), interpolation=cv2.INTER_AREA
        )
        resized_channels.append(ch_resized)
    # Combine channels (simple average)
    resized_im = np.stack(resized_channels, axis=-1)
    resized_gray = np.mean(resized_im, axis=-1)

    # Create output path for any visualization
    viz_path = None
    if visualize and visualization_outdir is not None:
        os.makedirs(visualization_outdir, exist_ok=True)
        viz_path = os.path.join(visualization_outdir, "tissue_masks_preview.png")

    # Segment for top n tissue
    tissue_masks = extract_tissue_regions_advanced(
        resized_gray,
        n_tissue=n_tissue,
        min_area=min_area,
        visualize=visualize,
        visualization_outpath=viz_path,
    )
    logging.info(f"[ROI] Found {len(tissue_masks)} tissue mask(s) after filtering.")

    rois = []
    for mask in tissue_masks:
        labeled_mask = label(mask)  # label each mask
        for region in regionprops(labeled_mask):
            minr, minc, maxr, maxc = region.bbox
            # Scale up to original resolution
            rois.append(
                {
                    "min_row": int(minr * downscale_factor),
                    "min_col": int(minc * downscale_factor),
                    "max_row": int(maxr * downscale_factor),
                    "max_col": int(maxc * downscale_factor),
                }
            )

    return rois


def crop_tissue_regions(all_channel_image, rois):
    """
    Crop bounding boxes from the full-resolution image according to `rois`.
    Returns a list of (H, W, C) arrays.
    """
    crops = []
    for i, roi in enumerate(rois):
        minr, minc = roi["min_row"], roi["min_col"]
        maxr, maxc = roi["max_row"], roi["max_col"]
        crop = all_channel_image[minr:maxr, minc:maxc, :]
        crops.append(crop)
    return crops


def save_crops_to_ome_tiff(tissue_crops, tif_image_details, output_dir, base_name):
    """
    Save each tissue crop to an OME‐TIFF with LZW compression + minimal metadata.
    """
    bits_per_sample = tif_image_details["BitsPerSample"]
    dtype_original = tif_image_details["DataType"]
    for i, crop in enumerate(tissue_crops):
        # Ensure crop dtype matches original
        if crop.dtype != dtype_original:
            crop = crop.astype(dtype_original)

        # Reorder dims from (H, W, C) => (C, H, W) for TIFF
        out_path = os.path.join(output_dir, f"{base_name}_tissue_{i}.ome.tiff")
        logging.info(f"Saving tissue crop {i} => {out_path}")
        tiff.imwrite(
            out_path,
            crop.transpose(2, 0, 1),
            compression="lzw",
            ome=True,
            metadata={"axes": "CYX"},
            bigtiff=False,
        )


#########################################
# (C) MAIN ENTRYPOINT: "run_extraction" #
#########################################


def run_extraction(config, args):
    """
    Main function to load the large image, find tissue bounding boxes,
    crop them, and save them to OME-TIFF for the main pipeline.
    """

    # Retrieve relevant fields from config
    file_name = config["data"]["file_name"]
    file_name = os.path.join(args.data_dir, file_name)
    tissue_cfg = config.get("tissue_extraction", {})
    n_tissue = tissue_cfg.get("n_tissue", 4)
    downscale_factor = tissue_cfg.get("downscale_factor", 64)
    min_area = tissue_cfg.get("min_area", 500)
    visualize = tissue_cfg.get("visualize", False)
    logging.info(f"config: {config}")
    logging.info(f"visualize: {visualize}")

    # 1) Load image
    full_image, info = load_image(file_name)

    # 2) Find bounding boxes for top tissue
    rois = find_tissue_rois_large_image(
        full_image,
        downscale_factor=downscale_factor,
        n_tissue=n_tissue,
        min_area=min_area,
        visualize=visualize,
        visualization_outdir=args.out_dir,
    )

    # 3) Crop each ROI from original
    crops = crop_tissue_regions(full_image, rois)

    # 4) Save each crop as an OME‐TIFF
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_name))[0]  # e.g. "D18_Scan1"
    save_crops_to_ome_tiff(crops, info, args.out_dir, base_name)
    logging.info("Tissue‐cropped OME‐TIFF(s) saved successfully.")


##########################
# (D) COMMAND-LINE MAIN  #
##########################


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)

    run_extraction(config, args)
    logging.info("Tissue extraction completed. Exiting.")


if __name__ == "__main__":
    main()
