#!/usr/bin/env python
# coding: utf-8
"""
Memory-efficient tissue region extraction for very large TIFF / QPTIFF.

Author: Da Kuang & ChatGPT (May-2025)
"""

import os
import sys
import argparse
import logging
import yaml
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from skimage.measure import label, regionprops
from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import zarr

###############################################################################
# (A) ARGUMENT & CONFIG
###############################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract tissue ROIs from huge TIFF/QPTIFF")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument(
        "--data_dir",
        default="/workspaces/codex-analysis/data",
        help="base dir for images",
    )
    p.add_argument("--out_dir", default="output_tissue_regions", help="output folder")
    return p.parse_args()


def load_config(cfg_path: str) -> dict:
    if not os.path.exists(cfg_path):
        logging.error(f"Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    logging.info(f"Config:\n{json.dumps(cfg, indent=2)}")
    return cfg


###############################################################################
# (B) LOW-LEVEL IO UTILITIES  (never read full image!)
###############################################################################


def get_tiff_metadata(tiff_path: str) -> Tuple[int, int, int, np.dtype]:
    """Return (H, W, C, dtype) without loading pixels."""
    with tiff.TiffFile(tiff_path) as tf:
        s0 = tf.series[0]
        if s0.ndim == 3:  # (C,Y,X) or (Y,X,C) depending on writer
            if s0.shape[0] < s0.shape[-1]:  # (C,Y,X)
                C, H, W = s0.shape
            else:  # (Y,X,C)
                H, W, C = s0.shape
        else:  # 2-D
            H, W = s0.shape
            C = 1
        dtype = s0.dtype
    return H, W, C, dtype


# def open_as_zarr(tiff_path: str):
#     """Return zarr-like array (no data loaded yet)."""
#     return tiff.TiffFile(tiff_path).series[0].aszarr()

def open_as_zarr(tiff_path: str):
    """
    Return a zarr.Array backed by the first image plane, regardless of
    tifffile version (Array / Store / Group).
    """
    obj = tiff.TiffFile(tiff_path).series[0].aszarr()
    logging.info(f"[open_as_zarr] aszarr() returned type: {type(obj)}")

    # Case 1 ─ already zarr.Array (has ndim/shape)
    if hasattr(obj, "ndim"):
        logging.info(f"[open_as_zarr] Case 1: Already zarr.Array with shape {obj.shape}")
        return obj  # ✅

    # Case 2 ─ Store-like objects (including ZarrTiffStore): try wrap directly
    logging.info(f"[open_as_zarr] Case 2: Trying zarr.open() on {type(obj)}")
    try:
        arr = zarr.open(obj, mode="r")
        logging.info(f"[open_as_zarr] zarr.open() returned type: {type(arr)}")
        if hasattr(arr, "ndim"):
            logging.info(f"[open_as_zarr] Case 2a: zarr.open() returned Array with shape {arr.shape}")
            return arr  # ✅
        # If zarr.open returns a Group, get the first array
        if hasattr(arr, "array_keys"):
            first_key = sorted(arr.array_keys())[0]
            logging.info(f"[open_as_zarr] Case 2b: zarr.open() returned Group, using key '{first_key}'")
            return arr[first_key]  # ✅
    except Exception as e:
        logging.info(f"[open_as_zarr] Case 2 failed: {e}")
        pass  # fall-through to Case 3

    # Case 3 ─ zarr.Group: pick the first array (usually key '0')
    if isinstance(obj, zarr.hierarchy.Group):
        first_key = sorted(obj.array_keys())[0]
        logging.info(f"[open_as_zarr] Case 3: Direct zarr.Group, using key '{first_key}'")
        return obj[first_key]  # ✅

    # Case 4 ─ ZarrTiffStore fallback: try direct zarr.Array construction
    if 'ZarrTiffStore' in str(type(obj)):
        logging.info(f"[open_as_zarr] Case 4: ZarrTiffStore fallback")
        try:
            result = zarr.Array(obj, path='0', read_only=True)
            logging.info(f"[open_as_zarr] Case 4 success: zarr.Array with shape {result.shape}")
            return result
        except Exception as e:
            logging.info(f"[open_as_zarr] Case 4 failed: {e}")
            pass

    # Otherwise unsupported
    logging.error(f"[open_as_zarr] All cases failed for type: {type(obj)}")
    raise TypeError(f"aszarr() produced unsupported type: {type(obj)}")

    # Otherwise unsupported
    raise TypeError(f"aszarr() produced unsupported type: {type(obj)}")


def read_crop_from_tiff(
    zarr_img, bbox: Tuple[int, int, int, int], channels: Optional[List[int]] = None
) -> np.ndarray:
    """
    Random-access read a crop from big TIFF.

    The returned array will be (H, W, C) with channels last.
    If supported (Linux with Python >= 3.8 and successful memfd_create),
    this function returns a memory-mapped NumPy array to conserve RAM.
    Otherwise, it falls back to creating a standard in-memory NumPy array.

    Parameters
    ----------
    zarr_img : zarr-like array (ndim=2 for YX, ndim=3 for CYX)
    bbox     : (minr, maxr, minc, maxc) in original resolution
    channels : list of channel indices for 3D zarr_img; None = all. Ignored for 2D.

    Returns
    -------
    crop : np.ndarray (H, W, C)
           May be an np.memmap instance.
    """
    minr, maxr, minc, maxc = bbox
    H_slice = maxr - minr
    W_slice = maxc - minc

    if H_slice < 0 or W_slice < 0:
        raise ValueError(f"Invalid bbox dimensions: H_slice={H_slice}, W_slice={W_slice}")

    final_dtype = zarr_img.dtype

    source_channel_indices: List[int]
    if zarr_img.ndim == 2:  # (Y,X)
        C_slice = 1
        # No specific source_channel_indices needed, handled as a special case later
    elif zarr_img.ndim == 3:  # (C,Y,X)
        if channels is None:
            C_slice = zarr_img.shape[0]
            source_channel_indices = list(range(C_slice))
        else:
            C_slice = len(channels)
            source_channel_indices = channels
    else:
        raise ValueError(f"zarr_img must be 2D or 3D, got {zarr_img.ndim}D")

    mmap_shape_hwc = (H_slice, W_slice, C_slice)

    use_memfd = hasattr(os, 'memfd_create')
    memmap_array = None
    
    if use_memfd:
        fd = -1
        mmap_file_obj = None
        try:
            fd_name = "crop_memmap_anon"
            flags = 0
            if hasattr(os, 'MFD_CLOEXEC'):
                flags |= os.MFD_CLOEXEC
            
            fd = os.memfd_create(fd_name, flags)

            _size = 1
            for _dim in mmap_shape_hwc: _size *= _dim
            array_nbytes = _size * np.dtype(final_dtype).itemsize
            
            os.ftruncate(fd, array_nbytes)
            
            mmap_file_obj = os.fdopen(fd, "r+b")
            memmap_array = np.memmap(mmap_file_obj, dtype=final_dtype, mode='r+', shape=mmap_shape_hwc)
            
            # fd is now managed by mmap_file_obj; closing mmap_file_obj will close fd.
            # np.memmap keeps mmap_file_obj alive.
            fd = -1 # Indicate fd is now managed by mmap_file_obj

        except Exception as e:
            logging.warning(f"Failed to create anonymous memmap using memfd_create: {e}. Falling back to in-memory array.")
            if mmap_file_obj: # If os.fdopen succeeded but np.memmap or other step failed
                mmap_file_obj.close()
            elif fd != -1: # If memfd_create succeeded but a subsequent step failed before fdopen
                 os.close(fd)
            memmap_array = None # Ensure fallback is triggered

    if memmap_array is None: # Fallback to original method
        logging.debug("Using standard in-memory numpy array for crop reading (memfd not used or failed).")
        if zarr_img.ndim == 2:
            # Original: zarr_img[minr:maxr, minc:maxc][None, ...] gives (1, H, W)
            crop_data_cyx = zarr_img[minr:maxr, minc:maxc][None, ...]
        else: # 3D
            if channels is None:
                crop_data_cyx = zarr_img[:, minr:maxr, minc:maxc]
            else:
                crop_data_cyx = zarr_img[channels, minr:maxr, minc:maxc]
        
        # np.asarray ensures it's an in-memory array, transpose, and ensure dtype
        return np.asarray(crop_data_cyx).transpose(1, 2, 0).astype(final_dtype)

    # Populate the memmap_array (H_slice, W_slice, C_slice)
    if zarr_img.ndim == 2:
        if C_slice == 1: # Should be true
            source_data_plane = zarr_img[minr:maxr, minc:maxc] # (H_slice, W_slice)
            memmap_array[:, :, 0] = source_data_plane
    elif zarr_img.ndim == 3:
        for i_out_c, original_ch_idx in enumerate(source_channel_indices):
            data_plane_from_zarr = zarr_img[original_ch_idx, minr:maxr, minc:maxc] # (H_slice, W_slice)
            memmap_array[:, :, i_out_c] = data_plane_from_zarr
            
    memmap_array.flush()
    return memmap_array


###############################################################################
# (C) VISUALIZATION  (only low-res thumbs)
###############################################################################


def read_lowres_preview(
    zarr_img, down_factor: int = 64, channel: int = 0
) -> np.ndarray:
    """Return a down-sampled single-channel preview."""
    if zarr_img.ndim == 2:
        return zarr_img[::down_factor, ::down_factor]
    return zarr_img[channel, ::down_factor, ::down_factor]


def visualize_polygon_overlay(
    preview: np.ndarray,
    polygons: List[np.ndarray],
    out_path: str,
    down_factor_preview: int,
):
    """Save overlay PNG; preview already downsampled."""
    from matplotlib.patches import Polygon as PatchPolygon

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(preview, cmap="gray")
    for poly in polygons:
        xy_scaled = np.stack(
            [poly[:, 1] / down_factor_preview, poly[:, 0] / down_factor_preview], axis=1
        )
        ax.add_patch(PatchPolygon(xy_scaled, edgecolor="red", fill=False, lw=1.5))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"[Visualization] Polygon overlay saved ➜ {out_path}")


###############################################################################
# (D) STREAM CROP BY POLYGON  (memory safe)
###############################################################################


def stream_crops_by_polygon(
    tiff_path: str,
    polygons: List[np.ndarray],
    out_dir: str,
    base_name: str,
    dtype: np.dtype,
    channels_keep: Optional[List[int]] = None,
):
    """Iterate polygons → read, mask, write; memory peaks < single ROI size."""
    os.makedirs(out_dir, exist_ok=True)
    zarr_img = open_as_zarr(tiff_path)
    H, W, *_ = get_tiff_metadata(tiff_path)
    
    total_polygons = len(polygons)
    print(f"🔄 [Progress] Starting to process {total_polygons} polygons for {base_name}")

    for i, poly in enumerate(polygons):
        print(f"📐 [Progress] Processing polygon {i+1}/{total_polygons} ({poly.shape[0]} vertices)")
        
        # full-size mask shape = (H,W) but we only need bbox
        print(f"   └─ Creating polygon mask...")
        mask_full = polygon2mask((H, W), poly)

        region = regionprops(label(mask_full.astype(np.uint8)))[0]
        minr, minc, maxr, maxc = region.bbox
        bbox = (minr, maxr, minc, maxc)
        crop_size = (maxr-minr, maxc-minc)
        print(f"   └─ Bounding box: ({minr},{minc}) to ({maxr},{maxc}), size: {crop_size}")

        print(f"   └─ Reading crop from TIFF...")
        crop = read_crop_from_tiff(zarr_img, bbox, channels_keep).astype(dtype)

        print(f"   └─ Applying polygon mask...")
        mask_crop = mask_full[minr:maxr, minc:maxc]
        crop[~mask_crop, :] = 0  # broadcast to all channels

        out_path = os.path.join(out_dir, f"{base_name}_manual_{i}.ome.tiff")
        print(f"   └─ Writing to file: {os.path.basename(out_path)}")
        tiff.imwrite(
            out_path,
            crop.transpose(2, 0, 1),
            compression="lzw",
            ome=True,
            metadata={"axes": "CYX"},
            bigtiff=True,
        )
        logging.info(
            f"[Write] Saved #{i}  crop  shape={crop.shape}  --> {out_path}"
        )
        print(f"✅ [Progress] Completed polygon {i+1}/{total_polygons}")

    print(f"🎉 [Progress] All {total_polygons} polygons processed successfully!")
    # 关闭 zarr（防止文件句柄泄漏）
    zarr_img.store.close()


###############################################################################
# (E) AUTOMATIC ROI DETECTION  (uses lowres preview only)
###############################################################################


def automatic_rois_from_preview(
    zarr_img, down_factor: int, n_tissue: int, min_area: int, outdir_vis: str
) -> List[dict]:
    """
    1) Build low-res grayscale preview, watershed segmentation
    2) Return list of bbox dict in full-res coords
    """
    H_full, W_full = zarr_img.shape[-2:]  # works for 2-D or 3-D (C,Y,X)

    # preview （单通道 or mean of few channels）——避免读全部通道
    if zarr_img.ndim == 3:
        preview_gray = np.asarray(zarr_img[0, ::down_factor, ::down_factor])
    else:
        preview_gray = np.asarray(zarr_img[::down_factor, ::down_factor])

    # ----- segmentation -----
    elevation = sobel(preview_gray)
    markers = np.zeros_like(preview_gray, dtype=np.uint8)
    thr = threshold_otsu(preview_gray)
    markers[preview_gray < thr] = 1
    markers[preview_gray >= thr] = 2
    seg = watershed(elevation, markers)
    binary = ndi.binary_fill_holes(seg == 2)
    labeled, num = ndi.label(binary)
    logging.info(f"[AutoROI] components={num}")

    areas_masks = []
    for lbl in range(1, num + 1):
        m = labeled == lbl
        a = np.count_nonzero(m)
        if a >= min_area:
            areas_masks.append((m, a))
    areas_masks.sort(key=lambda x: x[1], reverse=True)
    top_masks = [m for m, _ in areas_masks[:n_tissue]]

    # optional preview
    if outdir_vis:
        os.makedirs(outdir_vis, exist_ok=True)
        vis_path = os.path.join(outdir_vis, "auto_tissue_masks_preview.png")
        fig, axs = plt.subplots(1, len(top_masks), figsize=(4 * len(top_masks), 4))
        if len(top_masks) == 1:
            axs = [axs]
        for i, m in enumerate(top_masks):
            axs[i].imshow(m, cmap="gray")
            axs[i].set_title(f"Tissue {i}")
            axs[i].axis("off")
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150)
        plt.close()
        logging.info(f"[Visualization] auto masks preview ➜ {vis_path}")

    # convert to full-res bbox
    rois = []
    for m in top_masks:
        rprops = regionprops(label(m.astype(np.uint8)))[0]
        pr0, pc0, pr1, pc1 = rprops.bbox
        rois.append(
            {
                "min_row": pr0 * down_factor,
                "min_col": pc0 * down_factor,
                "max_row": pr1 * down_factor,
                "max_col": pc1 * down_factor,
            }
        )
    return rois


def stream_crops_by_bbox(
    tiff_path: str,
    rois: List[dict],
    out_dir: str,
    base_name: str,
    dtype: np.dtype,
    channels_keep: Optional[List[int]] = None,
):
    """Stream read each bbox and write OME-TIFF."""
    os.makedirs(out_dir, exist_ok=True)
    zarr_img = open_as_zarr(tiff_path)
    for i, roi in enumerate(rois):
        bbox = (roi["min_row"], roi["max_row"], roi["min_col"], roi["max_col"])
        crop = read_crop_from_tiff(zarr_img, bbox, channels_keep).astype(dtype)

        out_path = os.path.join(out_dir, f"{base_name}_auto_{i}.ome.tiff")
        tiff.imwrite(
            out_path,
            crop.transpose(2, 0, 1),
            compression="lzw",
            ome=True,
            metadata={"axes": "CYX"},
            bigtiff=True,
        )
        logging.info(f"[Write] Saved auto ROI #{i}  {crop.shape}  -> {out_path}")
    zarr_img.store.close()


###############################################################################
# (F) MAIN PIPELINE
###############################################################################


def parse_napari_json_annotations(json_path: str) -> List[np.ndarray]:
    """
    Parse napari JSON annotation file and return list of polygon coordinates.
    
    Parameters
    ----------
    json_path : str
        Path to the napari JSON annotation file
        
    Returns
    -------
    polygons : List[np.ndarray]
        List of polygon coordinate arrays, each shaped (N, 2) for N points
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    polygons = []
    
    # Handle different possible JSON structures from napari
    if isinstance(data, list):
        # Case 1: Direct list of shapes
        shapes_data = data
    elif isinstance(data, dict):
        # Case 2: Dictionary with shapes under a key (common napari export format)
        if 'shapes' in data:
            shapes_data = data['shapes']
        elif 'data' in data:
            shapes_data = data['data']
        else:
            # Case 3: Assume the dictionary values contain the shape data
            shapes_data = list(data.values())
    else:
        raise ValueError(f"Unsupported JSON structure in {json_path}")
    
    for shape in shapes_data:
        if isinstance(shape, dict):
            # Extract coordinates from shape dictionary
            if 'data' in shape:
                coords = np.array(shape['data'], dtype=float)
            elif 'coordinates' in shape:
                coords = np.array(shape['coordinates'], dtype=float)
            elif 'points' in shape:
                coords = np.array(shape['points'], dtype=float)
            else:
                # Try to find array-like data in the shape
                coord_candidates = [v for v in shape.values() if isinstance(v, (list, np.ndarray))]
                if coord_candidates:
                    coords = np.array(coord_candidates[0], dtype=float)
                else:
                    logging.warning(f"Could not find coordinates in shape: {shape}")
                    continue
        elif isinstance(shape, (list, np.ndarray)):
            # Direct coordinate array
            coords = np.array(shape, dtype=float)
        else:
            logging.warning(f"Unsupported shape format: {type(shape)}")
            continue
        
        # Ensure coordinates are 2D (N, 2)
        coords = np.atleast_2d(coords)
        if coords.shape[1] != 2:
            if coords.shape[0] == 2:
                coords = coords.T  # Transpose if coordinates are (2, N)
            else:
                logging.warning(f"Invalid coordinate shape: {coords.shape}, skipping")
                continue
        
        if len(coords) >= 3:  # Valid polygon needs at least 3 points
            polygons.append(coords)
        else:
            logging.warning(f"Polygon with {len(coords)} points skipped (need >= 3)")
    
    logging.info(f"Parsed {len(polygons)} polygons from {json_path}")
    return polygons


def run_extraction(cfg: dict, args: argparse.Namespace):
    img_rel = cfg["data"]["file_name"]
    tiff_path = os.path.join(args.data_dir, img_rel)
    os.makedirs(args.out_dir, exist_ok=True)

    # metadata
    H, W, C, dtype = get_tiff_metadata(tiff_path)
    logging.info(f"[Meta] H={H}  W={W}  C={C}  dtype={dtype}")

    tissue_cfg = cfg.get("tissue_extraction", {})
    manual_csv = tissue_cfg.get("manual_mask_csv")
    manual_json = tissue_cfg.get("manual_mask_json")
    down_factor = tissue_cfg.get("downscale_factor", 1024)
    n_tissue = tissue_cfg.get("n_tissue", 4)
    min_area = tissue_cfg.get("min_area", 500)
    visualize = tissue_cfg.get("visualize", True)

    base_name = os.path.splitext(os.path.basename(img_rel))[0]  # e.g. D16_Scan1

    # -------------------------------------------------------------------------
    # CASE 1: manual polygons (CSV or JSON format)
    # -------------------------------------------------------------------------
    polygons = None
    annotation_file = None
    
    # Priority: JSON > CSV (if both exist)
    if manual_json and os.path.exists(manual_json):
        logging.info(f"Using manual polygons from JSON: {manual_json}")
        annotation_file = manual_json
        polygons = parse_napari_json_annotations(manual_json)
    elif manual_csv and os.path.exists(manual_csv):
        logging.info(f"Using manual polygons from CSV: {manual_csv}")
        annotation_file = manual_csv
        df = pd.read_csv(manual_csv)
        polygons = [
            g[["axis-0", "axis-1"]].to_numpy(dtype=float)
            for _, g in df.groupby("index")
            if len(g) >= 3
        ]
    
    if polygons:
        logging.info(f"Loaded {len(polygons)} manual polygons from {annotation_file}")
        
        # preview overlay (low-res)
        if visualize:
            zarr_img = open_as_zarr(tiff_path)
            preview = read_lowres_preview(zarr_img, down_factor=down_factor)
            vis_path = os.path.join(args.out_dir, "manual_polygon_overlay.png")
            visualize_polygon_overlay(preview, polygons, vis_path, down_factor)
            zarr_img.store.close()

        stream_crops_by_polygon(
            tiff_path, polygons, args.out_dir, base_name, dtype, channels_keep=None
        )
        return

    # -------------------------------------------------------------------------
    # CASE 2: automatic watershed on low-res preview
    # -------------------------------------------------------------------------
    logging.info("Running automatic ROI detection")
    zarr_img = open_as_zarr(tiff_path)
    rois = automatic_rois_from_preview(
        zarr_img,
        down_factor=down_factor,
        n_tissue=n_tissue,
        min_area=min_area,
        outdir_vis=args.out_dir if visualize else "",
    )
    zarr_img.store.close()

    stream_crops_by_bbox(
        tiff_path, rois, args.out_dir, base_name, dtype, channels_keep=None
    )


###############################################################################
# (G) ENTRY
###############################################################################


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)

    # Make out_dir include sub-folder like uterus/D16/Scan1
    subdir = os.path.dirname(cfg["data"]["file_name"])
    args.out_dir = os.path.join(args.out_dir, subdir)

    run_extraction(cfg, args)
    logging.info("Done ✓")


if __name__ == "__main__":
    main()
