"""
Common helper functions for the AEGLE analysis pipeline.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def ensure_dir(directory: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Path to directory

    Returns:
        The directory path (for chaining)
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def get_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Return the boundary mask for each labeled cell in a segmented mask.
    The boundary is set to the original integer cell ID in boundary pixels.

    Args:
        mask: Segmentation mask where each cell has a unique integer ID

    Returns:
        Boundary mask with cell IDs at boundary pixels
    """
    from skimage.segmentation import find_boundaries

    boundary = find_boundaries(mask, mode="inner").astype(np.uint8)
    # For each boundary pixel, set it to the cell ID in the original `mask`.
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def save_dataframe(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
    """
    Save a DataFrame to CSV with proper directory creation.

    Args:
        df: DataFrame to save
        output_path: Path to save the CSV file
        index: Whether to include the index in the CSV (default: False)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=index)
    print(f"[INFO] Saved DataFrame to {output_path}")
