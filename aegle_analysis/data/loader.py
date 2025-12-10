"""
Data loading functions for CODEX analysis.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
import io
import gzip

try:  # Optional dependency for zstd-compressed artifacts
    import zstandard as zstd
except ImportError:  # pragma: no cover - zstandard may be absent in some envs
    zstd = None


def load_metadata(data_dir: str, patch_index: int = None) -> pd.DataFrame:
    """
    Load cell metadata. Checks for merged file first, then falls back to patch-specific file.

    Args:
        data_dir: Directory containing the data
        patch_index: Index of the patch to load (optional for merged files)

    Returns:
        DataFrame containing cell metadata
    """
    cell_profiling_dir = os.path.join(data_dir, "cell_profiling")
    
    # First, try to load merged file (from halves/quarters/full_image modes)
    merged_file = os.path.join(cell_profiling_dir, "cell_metadata.csv")
    if os.path.exists(merged_file):
        meta_df = pd.read_csv(merged_file)
        logging.info(f"Merged metadata loaded from {merged_file}. Shape: {meta_df.shape}")
        
        # If patch_index is specified, filter to that patch
        if patch_index is not None and "patch_id" in meta_df.columns:
            patch_meta_df = meta_df[meta_df["patch_id"] == patch_index]
            logging.info(f"Filtered to patch {patch_index}. Shape: {patch_meta_df.shape}")
            return patch_meta_df
        
        return meta_df
    
    # Fall back to patch-specific file (from patches mode)
    if patch_index is None:
        raise ValueError("patch_index must be specified when merged file is not available")
        
    patch_file = os.path.join(cell_profiling_dir, f"patch-{patch_index}-cell_metadata.csv")
    if os.path.exists(patch_file):
        meta_df = pd.read_csv(patch_file)
        logging.info(f"Patch-specific metadata loaded from {patch_file}. Shape: {meta_df.shape}")
        return meta_df
    
    # Neither file exists
    raise FileNotFoundError(f"No metadata file found. Checked: {merged_file} and {patch_file}")


def load_expression(data_dir: str, patch_index: int = None) -> pd.DataFrame:
    """
    Load marker expression data. Checks for merged file first, then falls back to patch-specific file.

    Args:
        data_dir: Directory containing the data
        patch_index: Index of the patch to load (optional for merged files)

    Returns:
        DataFrame containing marker expression data
    """
    cell_profiling_dir = os.path.join(data_dir, "cell_profiling")
    
    # First, try to load merged file (from halves/quarters/full_image modes)
    merged_file = os.path.join(cell_profiling_dir, "cell_by_marker.csv")
    if os.path.exists(merged_file):
        exp_df = pd.read_csv(merged_file)
        logging.info(f"Merged expression data loaded from {merged_file}. Shape: {exp_df.shape}")
        
        # If patch_index is specified, filter to that patch
        if patch_index is not None and "patch_id" in exp_df.columns:
            patch_exp_df = exp_df[exp_df["patch_id"] == patch_index]
            logging.info(f"Filtered to patch {patch_index}. Shape: {patch_exp_df.shape}")
            return patch_exp_df
        
        return exp_df
    
    # Fall back to patch-specific file (from patches mode)
    if patch_index is None:
        raise ValueError("patch_index must be specified when merged file is not available")
        
    patch_file = os.path.join(cell_profiling_dir, f"patch-{patch_index}-cell_by_marker.csv")
    if os.path.exists(patch_file):
        exp_df = pd.read_csv(patch_file)
        logging.info(f"Patch-specific expression data loaded from {patch_file}. Shape: {exp_df.shape}")
        return exp_df
    
    # Neither file exists
    raise FileNotFoundError(f"No expression file found. Checked: {merged_file} and {patch_file}")


def check_data_format(data_dir: str) -> dict:
    """
    Check whether the data directory contains merged files or patch-specific files.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        Dictionary with information about available data format
    """
    cell_profiling_dir = os.path.join(data_dir, "cell_profiling")
    
    # Check for merged files
    merged_metadata = os.path.join(cell_profiling_dir, "cell_metadata.csv")
    merged_expression = os.path.join(cell_profiling_dir, "cell_by_marker.csv")
    has_merged = os.path.exists(merged_metadata) and os.path.exists(merged_expression)
    
    # Check for patch-specific files (look for patch-0 files as indicator)
    patch_metadata = os.path.join(cell_profiling_dir, "patch-0-cell_metadata.csv")
    patch_expression = os.path.join(cell_profiling_dir, "patch-0-cell_by_marker.csv")
    has_patches = os.path.exists(patch_metadata) and os.path.exists(patch_expression)
    
    # Count available patch files if they exist
    patch_count = 0
    if has_patches:
        patch_files = [f for f in os.listdir(cell_profiling_dir) if f.startswith("patch-") and f.endswith("-cell_metadata.csv")]
        patch_count = len(patch_files)
    
    result = {
        "has_merged_files": has_merged,
        "has_patch_files": has_patches,
        "patch_count": patch_count,
        "data_format": "merged" if has_merged else ("patches" if has_patches else "unknown"),
        "split_mode_inferred": "halves/quarters/full_image" if has_merged else ("patches" if has_patches else "unknown")
    }
    
    logging.info(f"Data format check: {result}")
    return result


def load_all_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all available data (both metadata and expression) regardless of format.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        Tuple of (metadata_df, expression_df)
    """
    format_info = check_data_format(data_dir)
    
    if format_info["has_merged_files"]:
        # Load merged files directly
        metadata_df = load_metadata(data_dir)
        expression_df = load_expression(data_dir)
        logging.info(f"Loaded merged data: {metadata_df.shape[0]} cells, {expression_df.shape[1]} markers")
    elif format_info["has_patch_files"]:
        # Load and concatenate all patch files
        all_metadata = []
        all_expression = []
        
        for patch_idx in range(format_info["patch_count"]):
            try:
                patch_meta = load_metadata(data_dir, patch_idx)
                patch_exp = load_expression(data_dir, patch_idx)
                all_metadata.append(patch_meta)
                all_expression.append(patch_exp)
                logging.info(f"Loaded patch {patch_idx}: {patch_meta.shape[0]} cells")
            except FileNotFoundError:
                logging.warning(f"Patch {patch_idx} files not found, skipping...")
                continue
        
        if all_metadata:
            metadata_df = pd.concat(all_metadata, ignore_index=True)
            expression_df = pd.concat(all_expression, ignore_index=True)
            logging.info(f"Concatenated {len(all_metadata)} patches: {metadata_df.shape[0]} cells total")
        else:
            raise FileNotFoundError("No valid patch files found")
    else:
        raise FileNotFoundError(f"No valid data files found in {data_dir}")
    
    return metadata_df, expression_df


def process_dapi(
    meta_df: pd.DataFrame, exp_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Move DAPI column from expression data to metadata if present.

    Args:
        meta_df: Metadata DataFrame
        exp_df: Expression DataFrame

    Returns:
        Tuple of (updated metadata DataFrame, updated expression DataFrame)
    """
    if "DAPI" in exp_df.columns:
        dapi = exp_df["DAPI"].values
        meta_df = meta_df.copy()  # To avoid modifying the input DataFrame
        meta_df["DAPI"] = dapi
        exp_df = exp_df.drop(columns=["DAPI"])
        logging.info("Moved DAPI column to metadata.")

    return meta_df, exp_df


def load_segmentation_data(
    seg_path: Optional[str],
    segmentation_format: str = "pickle",
    patch_index: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Load segmentation data from an on-disk artifact if it exists.

    Args:
        seg_path: Path to the segmentation artifact
        segmentation_format: Serialization/compression format identifier
        patch_index: Index of the patch to load

    Returns:
        Dictionary containing segmentation data or None if the artifact is missing or unsupported.
    """

    if not seg_path:
        logging.info("Segmentation path not provided; skipping spatial overlays.")
        return None

    if not os.path.exists(seg_path):
        logging.warning(f"Segmentation artifact not found at {seg_path}.")
        return None

    logging.info(
        "[INFO] Loading segmentation artifact from %s (format=%s)",
        seg_path,
        segmentation_format,
    )

    loader = (segmentation_format or "pickle").lower()

    if loader == "pickle":
        with open(seg_path, "rb") as handle:
            repaired_seg_res_batch = np.load(handle, allow_pickle=True)
    elif loader == "pickle.gz":
        with gzip.open(seg_path, "rb") as handle:
            repaired_seg_res_batch = np.load(handle, allow_pickle=True)
    elif loader == "pickle.zst":
        if zstd is None:
            raise RuntimeError(
                "zstandard package is required to read .pickle.zst segmentation artifacts"
            )
        with open(seg_path, "rb") as handle:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(handle) as reader:
                decompressed = reader.read()
        repaired_seg_res_batch = np.load(io.BytesIO(decompressed), allow_pickle=True)
    elif loader in {"none", ""}:
        logging.info("Segmentation format set to 'none'; skipping spatial overlays.")
        return None
    else:
        logging.error("Unsupported segmentation format '%s'", segmentation_format)
        return None

    patch_data = repaired_seg_res_batch[patch_index]
    return patch_data
