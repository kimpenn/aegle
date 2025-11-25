"""Validation utilities for E2E testing."""

import os
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr
from typing import List, Tuple, Optional


def compare_cell_profiling(
    baseline_path: str,
    test_path: str,
    min_correlation: float = 0.9999,
) -> Tuple[bool, List[Tuple[str, float]]]:
    """Compare cell profiling outputs.

    Args:
        baseline_path: Path to baseline cell_overview.csv
        test_path: Path to test cell_overview.csv
        min_correlation: Minimum correlation threshold

    Returns:
        (passed, failed_markers) - List of (marker_name, correlation) for failures
    """
    baseline = pd.read_csv(baseline_path)
    test = pd.read_csv(test_path)

    metadata_cols = {'cell_mask_id', 'y', 'x', 'area'}
    marker_cols = [c for c in baseline.columns if c not in metadata_cols]

    failed = []
    for col in marker_cols:
        if col in test.columns:
            corr, _ = pearsonr(baseline[col], test[col])
            if corr < min_correlation:
                failed.append((col, corr))

    return len(failed) == 0, failed


def compare_segmentation_tiffs(
    baseline_dir: str,
    test_dir: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Tuple[bool, List[Tuple[str, str]]]:
    """Compare segmentation TIFF files.

    Args:
        baseline_dir: Path to baseline segmentations directory
        test_dir: Path to test segmentations directory
        rtol: Relative tolerance for array comparison
        atol: Absolute tolerance for array comparison

    Returns:
        (passed, errors) - List of (filename, error_message) for failures
    """
    errors = []

    if not os.path.isdir(baseline_dir):
        return False, [("", f"Baseline dir not found: {baseline_dir}")]
    if not os.path.isdir(test_dir):
        return False, [("", f"Test dir not found: {test_dir}")]

    baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith(".ome.tiff")]

    for filename in baseline_files:
        baseline_path = os.path.join(baseline_dir, filename)
        test_path = os.path.join(test_dir, filename)

        if not os.path.exists(test_path):
            errors.append((filename, "File not found in test directory"))
            continue

        try:
            with tifffile.TiffFile(baseline_path) as tif:
                baseline_data = tif.asarray()
            with tifffile.TiffFile(test_path) as tif:
                test_data = tif.asarray()

            if baseline_data.shape != test_data.shape:
                errors.append((filename, f"Shape mismatch: {baseline_data.shape} vs {test_data.shape}"))
                continue

            if not np.allclose(baseline_data, test_data, rtol=rtol, atol=atol):
                max_diff = np.abs(baseline_data - test_data).max()
                errors.append((filename, f"Array mismatch: max diff = {max_diff}"))
        except Exception as e:
            errors.append((filename, f"Error reading files: {str(e)}"))

    return len(errors) == 0, errors


def find_tiff_file(directory: str, pattern: str) -> Optional[str]:
    """Find TIFF file matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Substring pattern to match

    Returns:
        Full path to matching file, or None if not found
    """
    if not os.path.isdir(directory):
        return None

    for f in os.listdir(directory):
        if pattern in f:
            return os.path.join(directory, f)
    return None


def validate_cell_counts(
    baseline_path: str,
    test_path: str,
    expected_count: Optional[int] = None,
) -> Tuple[bool, str]:
    """Validate cell counts match between baseline and test.

    Args:
        baseline_path: Path to baseline cell_overview.csv
        test_path: Path to test cell_overview.csv
        expected_count: Optional expected cell count

    Returns:
        (passed, message)
    """
    baseline = pd.read_csv(baseline_path)
    test = pd.read_csv(test_path)

    if len(baseline) != len(test):
        return False, f"Cell count mismatch: baseline={len(baseline)}, test={len(test)}"

    if expected_count is not None and len(baseline) != expected_count:
        return False, f"Cell count {len(baseline)} != expected {expected_count}"

    return True, f"Cell count matches: {len(baseline)}"
