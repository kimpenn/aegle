#!/usr/bin/env python
"""
Standalone script to export segmentation masks to OME-TIFF format.

This script can be used to recover from the mask shape bug where 3D masks (1, H, W)
were not properly handled during OME-TIFF export.

Usage:
    python scripts/export_segmentation_tiff.py --output_dir /path/to/output

The script will:
1. Load matched_seg_res_batch.pickle.zst and original_seg_res_batch.pickle.zst
2. Squeeze 3D masks to 2D
3. Export each mask type as pyramidal OME-TIFF
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile
import zstandard as zstd
from skimage.segmentation import find_boundaries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_pickle_zst(path: str) -> any:
    """Load a zstandard-compressed pickle file."""
    logger.info(f"Loading {path}...")
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = pickle.load(reader)
    logger.info(f"Loaded successfully. Type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
    return data


def build_pyramid(base_level: np.ndarray, min_size: int = 256) -> List[np.ndarray]:
    """Build image pyramid for OME-TIFF."""
    pyramid = [base_level]
    current = base_level
    while (
        current.shape[-2] >= 2 * min_size
        and current.shape[-1] >= 2 * min_size
    ):
        down = current[:, ::2, ::2]
        if down.shape[-2] < min_size or down.shape[-1] < min_size:
            break
        pyramid.append(down)
        current = down
    return pyramid


def save_mask_as_ome_tiff(
    mask: np.ndarray,
    channel_name: str,
    output_path: str,
    image_mpp: Optional[float] = None,
) -> bool:
    """Save a single mask as pyramidal OME-TIFF."""
    if mask is None:
        return False

    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return False

    # Handle 3D masks (1, H, W) -> (H, W)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = np.squeeze(mask_array, axis=0)
        logger.info(f"  Squeezed 3D mask to 2D: {mask_array.shape}")

    mask_array = mask_array.astype(np.uint32, copy=False)

    # Create stack for pyramid (C, Y, X)
    stack = mask_array[None, ...]
    pyramid = build_pyramid(stack)

    metadata = {
        "axes": "CYX",
        "channel_names": [channel_name],
    }
    if image_mpp:
        try:
            metadata["PhysicalSizeX"] = float(image_mpp)
            metadata["PhysicalSizeY"] = float(image_mpp)
        except (TypeError, ValueError):
            pass

    logger.info(f"  Writing {output_path} with {len(pyramid)} pyramid levels...")
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(
            pyramid[0],
            subifds=len(pyramid) - 1,
            dtype=np.uint32,
            compression="zlib",
            photometric="minisblack",
            metadata=metadata,
        )
        for level in pyramid[1:]:
            tif.write(
                level,
                subfiletype=1,
                dtype=np.uint32,
                compression="zlib",
                photometric="minisblack",
            )

    logger.info(f"  ✓ Exported {channel_name} to {output_path}")
    return True


def compute_boundary(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Compute labeled boundary from mask."""
    if mask is None:
        return None
    mask_array = np.asarray(mask)

    # Handle 3D masks
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = np.squeeze(mask_array, axis=0)

    if mask_array.size == 0:
        return None

    boundary_bool = find_boundaries(mask_array, mode="inner")
    boundary = np.zeros_like(mask_array, dtype=np.uint32)
    boundary[boundary_bool] = mask_array[boundary_bool]
    return boundary


def export_segmentation_masks(
    output_dir: str,
    base_name: str = "segmentation",
    image_mpp: Optional[float] = None,
) -> bool:
    """Export all segmentation masks from saved pickle files."""

    # Paths
    matched_pickle = os.path.join(output_dir, "matched_seg_res_batch.pickle.zst")
    original_pickle = os.path.join(output_dir, "original_seg_res_batch.pickle.zst")
    seg_output_dir = os.path.join(output_dir, "segmentations")

    # Check files exist
    if not os.path.exists(matched_pickle):
        logger.error(f"Matched pickle not found: {matched_pickle}")
        return False
    if not os.path.exists(original_pickle):
        logger.error(f"Original pickle not found: {original_pickle}")
        return False

    # Create output directory
    os.makedirs(seg_output_dir, exist_ok=True)

    # Load data
    matched_results = load_pickle_zst(matched_pickle)
    original_results = load_pickle_zst(original_pickle)

    if not matched_results or not original_results:
        logger.error("Empty results loaded")
        return False

    # For full_image mode, we have single patch (index 0)
    matched_seg = matched_results[0] if matched_results else {}
    original_seg = original_results[0] if original_results else {}

    logger.info(f"Matched seg keys: {list(matched_seg.keys())}")
    logger.info(f"Original seg keys: {list(original_seg.keys())}")

    exported_count = 0

    # Export original masks
    original_masks = [
        ("cell", "cell"),
        ("nucleus", "nucleus"),
    ]

    for mask_name, key in original_masks:
        mask = original_seg.get(key)
        if mask is not None:
            logger.info(f"Exporting {mask_name} (shape: {np.asarray(mask).shape})...")
            output_path = os.path.join(seg_output_dir, f"{base_name}.{mask_name}.segmentations.ome.tiff")
            if save_mask_as_ome_tiff(mask, mask_name, output_path, image_mpp):
                exported_count += 1

    # Export original boundaries
    for mask_name, key in original_masks:
        mask = original_seg.get(key)
        if mask is not None:
            boundary = compute_boundary(mask)
            if boundary is not None:
                boundary_name = f"{mask_name}_boundary"
                logger.info(f"Exporting {boundary_name}...")
                output_path = os.path.join(seg_output_dir, f"{base_name}.{boundary_name}.segmentations.ome.tiff")
                if save_mask_as_ome_tiff(boundary, boundary_name, output_path, image_mpp):
                    exported_count += 1

    # Export matched/repaired masks
    matched_masks = [
        ("cell_matched_mask", "cell_matched_mask"),
        ("nucleus_matched_mask", "nucleus_matched_mask"),
        ("cell_outside_nucleus_mask", "cell_outside_nucleus_mask"),
    ]

    for mask_name, key in matched_masks:
        mask = matched_seg.get(key)
        if mask is not None:
            logger.info(f"Exporting {mask_name} (shape: {np.asarray(mask).shape})...")
            output_path = os.path.join(seg_output_dir, f"{base_name}.{mask_name}.segmentations.ome.tiff")
            if save_mask_as_ome_tiff(mask, mask_name, output_path, image_mpp):
                exported_count += 1

    # Export matched boundaries
    for mask_name in ["cell_matched_mask", "nucleus_matched_mask"]:
        mask = matched_seg.get(mask_name)
        if mask is not None:
            boundary = compute_boundary(mask)
            if boundary is not None:
                boundary_name = mask_name.replace("_mask", "_boundary")
                logger.info(f"Exporting {boundary_name}...")
                output_path = os.path.join(seg_output_dir, f"{base_name}.{boundary_name}.segmentations.ome.tiff")
                if save_mask_as_ome_tiff(boundary, boundary_name, output_path, image_mpp):
                    exported_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Export complete: {exported_count} masks exported to {seg_output_dir}")
    logger.info(f"{'='*60}")

    return exported_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Export segmentation masks to OME-TIFF format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to pipeline output directory containing pickle files",
    )
    parser.add_argument(
        "--base_name",
        type=str,
        default="segmentation",
        help="Base name for output files (default: segmentation)",
    )
    parser.add_argument(
        "--image_mpp",
        type=float,
        default=0.5,
        help="Microns per pixel (default: 0.5)",
    )

    args = parser.parse_args()

    success = export_segmentation_masks(
        output_dir=args.output_dir,
        base_name=args.base_name,
        image_mpp=args.image_mpp,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
