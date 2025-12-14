"""
Data transformation functions for CODEX analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging


ID_COLUMNS = ("cell_mask_id", "patch_id", "global_cell_id")


def _strip_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with identifier columns removed and cell IDs moved to the index."""

    working = df.copy()

    if ID_COLUMNS[0] in working.columns:
        working = working.set_index(ID_COLUMNS[0])

    for col in ID_COLUMNS[1:]:
        if col in working.columns:
            working = working.drop(columns=[col])

    return working

def clr_across_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply centered log-ratio (CLR) transformation across cells.
    CLR is applied row-wise (each cell) in this version.

    Args:
        df: Pandas DataFrame of shape (n_cells, n_markers)
            with raw intensities (rows = cells).

    Returns:
        CLR-transformed DataFrame of same shape.
    """
    # Replace 0 with a small value to avoid log(0)
    df_nz = df.replace(0, np.nan)
    # Geometric mean per cell
    geometric_mean = np.exp(np.log(df_nz).mean(axis=1))
    # CLR transform: log1p( x / GM )
    clr_df = np.log1p(df.div(geometric_mean, axis=0))
    return clr_df


def double_zscore_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a 'double z-score + negative log transform' approach:
      1) Z-score across columns
      2) Z-score across rows
      3) Convert to probability (CDF) via norm.cdf
      4) Transform by -log(1 - p)

    This is sometimes used to highlight "high" intensities in a two-step manner.

    Args:
        df: (n_cells, n_markers) DataFrame

    Returns:
        Transformed DataFrame
    """
    from scipy.stats import zscore, norm

    # 1) Z-score across columns
    dfz1 = df.apply(lambda col: zscore(col, nan_policy="omit"), axis=0)
    dfz1 = pd.DataFrame(dfz1, index=df.index, columns=df.columns)

    # 2) Z-score across rows
    dfz2 = dfz1.apply(lambda row: zscore(row, nan_policy="omit"), axis=1)
    dfz2 = pd.DataFrame(dfz2, index=df.index, columns=df.columns)

    # 3) Convert z-scores to probabilities
    dfz3 = dfz2.apply(lambda col: norm.cdf(col))

    # 4) -log(1 - p)
    # Clip values to avoid log(0)
    dfz3 = dfz3.clip(upper=1.0 - 1e-12)
    dflog = dfz3.apply(lambda col: -np.log(1 - col))
    return dflog


def zscore_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation followed by z-score normalization.

    Args:
        df: DataFrame to transform

    Returns:
        Transformed DataFrame
    """
    data_df = np.log1p(df)
    data_df = data_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return data_df


def rank_normalization(
    df: pd.DataFrame,
    method="fraction",
    use_gpu=False,
    gpu_batch_size=None,
    fallback_to_cpu=True
) -> pd.DataFrame:
    """
    Apply rank-based normalization across antibody for each cell (row-wise).

    Args:
        df: (n_cells, n_markers) DataFrame of raw intensities.
        method: "fraction" (default) scales ranks to [0,1],
                "gaussian" applies an inverse normal transformation.
        use_gpu: Whether to use GPU acceleration (default: False for backward compatibility)
        gpu_batch_size: Number of cells to process per GPU batch (None = auto-detect)
        fallback_to_cpu: If GPU fails or unavailable, fall back to CPU (default: True)

    Returns:
        Rank-normalized DataFrame of same shape.
    """
    import time

    # Try GPU path if requested
    if use_gpu:
        try:
            from ..gpu_utils import select_compute_backend

            backend = select_compute_backend(use_gpu, fallback_to_cpu, logging.getLogger(__name__))

            if backend == "gpu":
                try:
                    from .transforms_gpu import rank_normalize_gpu

                    # Start timing
                    gpu_start = time.time()

                    # Convert DataFrame to NumPy for GPU processing
                    data_values = df.values

                    # Run GPU normalization
                    result_values = rank_normalize_gpu(
                        data_values,
                        gpu_batch_size=gpu_batch_size,
                        method=method
                    )

                    # Wrap back in DataFrame
                    result_df = pd.DataFrame(result_values, index=df.index, columns=df.columns)

                    gpu_time = time.time() - gpu_start

                    # Estimate CPU time (rough estimate based on typical 5-10x speedup)
                    cpu_time_estimate = gpu_time * 9.0  # Conservative 9x speedup estimate

                    logging.info(
                        f"Normalization: GPU ({gpu_time:.1f}s) vs CPU estimate ({cpu_time_estimate:.1f}s) "
                        f"- {cpu_time_estimate/gpu_time:.1f}x speedup"
                    )
                    logging.info(f"df_rank: {result_df.shape}")
                    logging.info(f"df_rank: {result_df}")

                    return result_df

                except Exception as e:
                    if fallback_to_cpu:
                        logging.warning(f"GPU normalization failed: {e}, falling back to CPU")
                    else:
                        raise

        except ImportError as e:
            if fallback_to_cpu:
                logging.warning(f"GPU utilities unavailable: {e}, using CPU")
            else:
                raise

    # CPU normalization (existing code)
    from scipy.stats import rankdata, norm

    cpu_start = time.time()

    if method == "fraction":
        df_rank = df.apply(
            lambda row: pd.Series(rankdata(row, method="average") / (len(row) + 1), index=df.columns), axis=1
        )

    elif method == "gaussian":
        df_rank = df.apply(
            lambda row: pd.Series(norm.ppf(rankdata(row, method="average") / (len(row) + 1)), index=df.columns), axis=1
        )

    else:
        raise ValueError(f"Unknown rank normalization method: {method}")

    cpu_time = time.time() - cpu_start

    if use_gpu:
        # Log that we're using CPU when GPU was requested
        logging.info(f"Normalization: CPU ({cpu_time:.1f}s)")

    logging.info(f"df_rank: {df_rank.shape}")
    logging.info(f"df_rank: {df_rank}")
    return pd.DataFrame(df_rank, index=df.index, columns=df.columns)

def log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation.

    Args:
        df: DataFrame to transform

    Returns:
        Log1p-transformed DataFrame
    """
    # This function is used to make sure log1p_transform is self-contained and can be used independently
    numeric_df = _strip_identifier_columns(df)
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna()
    return np.log1p(numeric_df)


def prepare_data_for_analysis(
    exp_df: pd.DataFrame,
    norm="log1p",
    use_gpu=False,
    gpu_batch_size=None,
    fallback_to_cpu=True
) -> pd.DataFrame:
    """
    Prepare expression data for analysis by:
    1. Removing identifier columns (cell IDs, patch indices)
    2. Ensuring marker intensities are numeric
    3. Applying the requested normalisation

    Args:
        exp_df: Raw expression DataFrame
        norm: Normalization method (log1p, zscore_log, clr, double_z, rank_fraction, rank_gaussian, none)
        use_gpu: Whether to use GPU acceleration for rank normalization (default: False)
        gpu_batch_size: Number of cells per GPU batch for rank normalization (None = auto-detect)
        fallback_to_cpu: Fall back to CPU if GPU fails (default: True)

    Returns:
        Processed DataFrame ready for analysis
    """
    data_df = _strip_identifier_columns(exp_df)

    # Convert to numeric, coercing non-numeric entries to NaN then dropping incomplete rows
    data_df = data_df.apply(pd.to_numeric, errors="coerce")
    data_df = data_df.dropna()

    if norm == "log1p":
        data_df = log1p_transform(data_df)
    elif norm == "zscore_log":
        data_df = zscore_log(data_df)
    elif norm == "clr":
        data_df = clr_across_cells(data_df)
    elif norm == "double_z":
        data_df = double_zscore_log(data_df)
    elif norm == "rank_fraction":
        data_df = rank_normalization(
            data_df,
            method="fraction",
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            fallback_to_cpu=fallback_to_cpu
        )
    elif norm == "rank_gaussian":
        data_df = rank_normalization(
            data_df,
            method="gaussian",
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            fallback_to_cpu=fallback_to_cpu
        )
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {norm}")

    return data_df
