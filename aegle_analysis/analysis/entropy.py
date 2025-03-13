"""
Entropy calculation functions for CODEX analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def calculate_channel_entropy(df: pd.DataFrame, base: int = 2) -> pd.Series:
    """
    Calculate Shannon entropy for each cell (row) across all markers.

    Args:
        df: DataFrame with cells as rows and markers as columns
        base: Base of the logarithm used for entropy calculation (default: 2 for bits)

    Returns:
        Series containing entropy value for each cell
    """
    # Normalize each row to sum to 1 (probability distribution)
    row_sums = df.sum(axis=1)
    normalized_df = df.div(row_sums, axis=0)

    # Calculate entropy for each row
    entropies = normalized_df.apply(lambda row: entropy(row.values, base=base), axis=1)
    return entropies


def add_entropy_to_metadata(
    meta_df: pd.DataFrame, exp_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate entropy across channels and add it to metadata if not present.

    Args:
        meta_df: Metadata DataFrame
        exp_df: Expression DataFrame with markers as columns

    Returns:
        Updated metadata DataFrame with entropy column added
    """
    if "cell_entropy" not in meta_df.columns:
        print("[INFO] Calculating cell entropy...")
        # Calculate entropy across all markers for each cell
        meta_df = meta_df.copy()  # To avoid modifying the input DataFrame
        meta_df["cell_entropy"] = calculate_channel_entropy(exp_df)
        print("[INFO] Added cell entropy to metadata.")

    return meta_df


def calculate_entropy_bounds(n_channels: int) -> tuple:
    """
    Calculate theoretical minimum and maximum entropy for a given number of channels.

    Args:
        n_channels: Number of channels/markers

    Returns:
        Tuple of (min_entropy, max_entropy)
    """
    # Minimum entropy: all intensity in one channel
    pk_min = [0] * n_channels
    pk_min[0] = 1
    min_entropy = entropy(pk_min, base=2)

    # Maximum entropy: uniform distribution across all channels
    pk_max = [1 / n_channels] * n_channels
    max_entropy = entropy(pk_max, base=2)

    return min_entropy, max_entropy
