import logging
import time
from typing import Dict, List, Optional, Tuple

from cv2 import log
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

# Configure module-level logger
logger = logging.getLogger(__name__)

def calculate_count_density_metrics(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_matched_mask: np.ndarray,
    nucleus_matched_mask: np.ndarray,
    nucleus_unmatched_mask: np.ndarray,
    image_mpp: float = 0.5,
) -> Dict:
    """
    Compute all density metrics for a single patch.

    Args:
        cell_mask: Original cell mask
        nucleus_mask: Original nucleus mask
        cell_matched_mask: Repaired cell mask
        nucleus_matched_mask: Repaired nucleus mask
        nucleus_unmatched_mask: Mask of nuclei present in original but not repaired segmentation
        image_mpp: Microns per pixel

    Returns:
        Dictionary with all cell density metrics
    """
    logger.info("Starting cell density metrics calculation")
    start_time = time.time()
    window_size_microns = 200
    # Global density metrics
    logger.info("Calculating global cell density metrics for cell mask...")
    cell_metrics = compute_mask_density(cell_mask, image_mpp)
    logger.info("Calculating global cell density metrics for nucleus mask...")
    nucleus_metrics = compute_mask_density(nucleus_mask, image_mpp)
    logger.info("Calculating global cell density metrics for repaired cell mask...")
    repaired_cell_metrics = compute_mask_density(cell_matched_mask, image_mpp)
    logger.info("Calculating global cell density metrics for repaired nucleus mask...")
    repaired_nucleus_metrics = compute_mask_density(nucleus_matched_mask, image_mpp)
    logger.info("Calculating global cell density metrics for unmatched nucleus mask...")
    unmatched_nucleus_metrics = compute_mask_density(nucleus_unmatched_mask, image_mpp)

    # Local density metrics
    logger.info("Calculating local cell density metrics for cell mask...")
    cell_local_densities = compute_local_densities(cell_mask, image_mpp=image_mpp, window_size_microns=window_size_microns)
    logger.info("Calculating local cell density metrics for nucleus mask...")
    nucleus_local_densities = compute_local_densities(nucleus_mask, image_mpp=image_mpp, window_size_microns=window_size_microns)
    logger.info("Calculating local cell density metrics for repaired cell mask...")
    repaired_cell_local_densities = compute_local_densities(cell_matched_mask, image_mpp=image_mpp, window_size_microns=window_size_microns)
    logger.info("Calculating local cell density metrics for repaired nucleus mask...")
    repaired_nucleus_local_densities = compute_local_densities(nucleus_matched_mask, image_mpp=image_mpp, window_size_microns=window_size_microns)
    logger.info("Calculating local cell density metrics for unmatched nucleus mask...")
    unmatched_nucleus_local_densities = compute_local_densities(nucleus_unmatched_mask, image_mpp=image_mpp, window_size_microns=window_size_microns)

    # Local density stats
    logger.info("Calculating local cell density stats for cell mask...")
    cell_local_stats = compute_local_density_stats(cell_local_densities)
    logger.info("Calculating local cell density stats for nucleus mask...")
    nucleus_local_stats = compute_local_density_stats(nucleus_local_densities)
    logger.info("Calculating local cell density stats for repaired cell mask...")
    repaired_cell_local_stats = compute_local_density_stats(repaired_cell_local_densities)
    logger.info("Calculating local cell density stats for repaired nucleus mask...")
    repaired_nucleus_local_stats = compute_local_density_stats(repaired_nucleus_local_densities)
    logger.info("Calculating local cell density stats for unmatched nucleus mask...")
    unmatched_nucleus_local_stats = compute_local_density_stats(unmatched_nucleus_local_densities)

    metrics = {
        "global": {
            "cell": cell_metrics,
            "nucleus": nucleus_metrics,
            "repaired_cell": repaired_cell_metrics,
            "repaired_nucleus": repaired_nucleus_metrics,
            "unmatched_nucleus": unmatched_nucleus_metrics,
        },
        "local": {
            "cell": cell_local_stats,
            "nucleus": nucleus_local_stats,
            "repaired_cell": repaired_cell_local_stats,
            "repaired_nucleus": repaired_nucleus_local_stats,
            "unmatched_nucleus": unmatched_nucleus_local_stats,
        },
        "local_density_list": {
            "cell": cell_local_densities,
            "nucleus": nucleus_local_densities,
            "repaired_cell": repaired_cell_local_densities,
            "repaired_nucleus": repaired_nucleus_local_densities,
            "unmatched_nucleus": unmatched_nucleus_local_densities,
        },
        "window_size_microns": window_size_microns,
    }
    logger.info(f"Total number of windows: {len(cell_local_densities)}")
    elapsed_time = time.time() - start_time
    logger.info(f"Density metrics calculation completed in {elapsed_time:.2f} seconds")
    return metrics


def compute_mask_density(mask: np.ndarray, image_mpp: float = 0.5) -> Dict:
    """
    Computes cell density from a labeled mask.

    Args:
        mask (np.ndarray): Labeled mask of shape (H, W) or (1, H, W). Each unique non-zero label is a cell/nucleus.
        image_mpp (float): Microns per pixel.

    Returns:
        Dict: Dictionary containing:
            - n_objects: Number of unique objects
            - area_mm2: Area in square millimeters
            - density: Density in objects per mm²
    """
    logger.info("Starting mask density computation")
    
    if mask.ndim == 3:
        mask = np.squeeze(mask)  # shape becomes (H, W)

    # Convert boolean mask to labeled mask if needed
    if mask.dtype == bool:
        mask = ndimage.label(mask)[0]

    n_objects = len(np.unique(mask)) - (1 if 0 in mask else 0)  # exclude background
    tissue_pixels = np.count_nonzero(mask)  # actual tissue area in pixels
    area_mm2 = tissue_pixels * (image_mpp**2) / 1e6

    if area_mm2 == 0:
        logger.warning("Zero area detected in mask density computation")
        return {
            "n_objects": 0,
            "area_mm2": 0,
            "density": 0.0,
        }

    density = n_objects / area_mm2
    logger.info(f"Computed density: {density:.2f} objects/mm² for {n_objects} objects")
    return {
        "n_objects": n_objects,
        "area_mm2": area_mm2,
        "density": density,
    }


def compute_local_densities(
    mask: np.ndarray,
    image_mpp: float = 0.5,
    window_size_microns: float = 200,
    step_microns: Optional[float] = None,
) -> List[float]:
    """
    Subdivides a labeled mask into smaller windows and computes
    cell density within each window. Returns a list of local densities.

    Args:
        mask: Labeled mask (H, W) with int labels.
        image_mpp: Microns per pixel.
        window_size_microns: Side length of each sub-window, in microns.
        step_microns: Step size for sliding the window; defaults to window_size_microns (non-overlapping).

    Returns:
        List[float]: A list of local densities (cells / mm^2).
    """
    logger.info("Starting local densities computation")
    start_time = time.time()

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    # Convert boolean mask to labeled mask if needed
    if mask.dtype == bool:
        mask = ndimage.label(mask)[0]

    if step_microns is None:
        step_microns = window_size_microns

    # Convert from microns to pixels
    window_size_pixels = int(window_size_microns / image_mpp)
    step_pixels = int(step_microns / image_mpp)

    height, width = mask.shape

    # regionprops to get centroids
    props = regionprops(mask)
    centroids = [prop.centroid for prop in props]  # (row, col)

    local_densities = []

    for top in range(0, height, step_pixels):
        for left in range(0, width, step_pixels):
            bottom = min(top + window_size_pixels, height)
            right = min(left + window_size_pixels, width)

            # Count how many cells fall into this sub-window (centroid in bounding box)
            cell_count = 0
            for cy, cx in centroids:
                if (cy >= top) and (cy < bottom) and (cx >= left) and (cx < right):
                    cell_count += 1

            sub_height = bottom - top
            sub_width = right - left
            # area in mm^2
            sub_area_mm2 = sub_height * sub_width * (image_mpp**2) / 1e6

            if sub_area_mm2 > 0:
                local_density = cell_count / sub_area_mm2
            else:
                local_density = 0.0

            local_densities.append(local_density)

    elapsed_time = time.time() - start_time
    logger.info(f"Local densities computation completed in {elapsed_time:.2f} seconds")
    return local_densities


def compute_local_density_stats(local_density_list: List[float]) -> Dict[str, float]:
    """
    Compute summary stats (mean, median, std, quantiles) for a list of local densities.
    
    Args:
        local_density_list: List of local densities
        
    Returns:
        Dict containing:
            - mean: Mean local density
            - median: Median local density
            - std: Standard deviation of local densities
            - quantile_25: 25th percentile
            - quantile_75: 75th percentile
    """
    logger.info("Starting local density stats computation")
    
    if len(local_density_list) == 0:
        logger.warning("Empty local density list provided")
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "quantile_25": 0.0,
            "quantile_75": 0.0,
        }
    
    stats = {
        "mean": float(np.mean(local_density_list)),
        "median": float(np.median(local_density_list)),
        "std": float(np.std(local_density_list)),
        "quantile_25": float(np.quantile(local_density_list, 0.25)),
        "quantile_75": float(np.quantile(local_density_list, 0.75)),
    }
    
    logger.info(f"Computed local density stats: mean={stats['mean']:.2f}, median={stats['median']:.2f}, std={stats['std']:.2f}")
    return stats


def update_metadata_with_density_metrics(
    patches_metadata_df,
    informative_idx,
    metrics_list: List[Dict],
) -> None:
    """
    Update metadata dataframe with density metrics.

    Args:
        patches_metadata_df: Metadata dataframe
        informative_idx: Boolean mask for the rows to update
        metrics_list: List of dictionaries with metrics for each patch
    """
    logger.info("Starting metadata update with density metrics")
    start_time = time.time()

    # Initialize metric lists
    global_metrics = {
        "cell_mask_nobj": [],
        "cell_mask_areamm2": [],
        "cell_mask_density": [],
        "nucleus_mask_nobj": [],
        "nucleus_mask_areamm2": [],
        "nucleus_mask_density": [],
        "repaired_cell_mask_nobj": [],
        "repaired_cell_mask_areamm2": [],
        "repaired_cell_mask_density": [],
        "repaired_nucleus_mask_nobj": [],
        "repaired_nucleus_mask_areamm2": [],
        "repaired_nucleus_mask_density": [],
        "unmatched_nucleus_mask_nobj": [],
        "unmatched_nucleus_mask_areamm2": [],
        "unmatched_nucleus_mask_density": [],
    }

    local_metrics = {
        "cell_mask_local_density_mean": [],
        "cell_mask_local_density_median": [],
        "cell_mask_local_density_std": [],
        "cell_mask_local_density_quantile_25": [],
        "cell_mask_local_density_quantile_75": [],
        "nucleus_mask_local_density_mean": [],
        "nucleus_mask_local_density_median": [],
        "nucleus_mask_local_density_std": [],
        "nucleus_mask_local_density_quantile_25": [],
        "nucleus_mask_local_density_quantile_75": [],
        "repaired_cell_mask_local_density_mean": [],
        "repaired_cell_mask_local_density_median": [],
        "repaired_cell_mask_local_density_std": [],
        "repaired_cell_mask_local_density_quantile_25": [],
        "repaired_cell_mask_local_density_quantile_75": [],
        "repaired_nucleus_mask_local_density_mean": [],
        "repaired_nucleus_mask_local_density_median": [],
        "repaired_nucleus_mask_local_density_std": [],
        "repaired_nucleus_mask_local_density_quantile_25": [],
        "repaired_nucleus_mask_local_density_quantile_75": [],
        "unmatched_nucleus_mask_local_density_mean": [],
        "unmatched_nucleus_mask_local_density_median": [],
        "unmatched_nucleus_mask_local_density_std": [],
        "unmatched_nucleus_mask_local_density_quantile_25": [],
        "unmatched_nucleus_mask_local_density_quantile_75": [],
    }

    # Extract metrics for each patch
    for metrics in metrics_list:
        # Global metrics
        global_metrics["cell_mask_nobj"].append(metrics["global"]["cell"]["n_objects"])
        global_metrics["cell_mask_areamm2"].append(metrics["global"]["cell"]["area_mm2"])
        global_metrics["cell_mask_density"].append(metrics["global"]["cell"]["density"])
        global_metrics["nucleus_mask_nobj"].append(metrics["global"]["nucleus"]["n_objects"])
        global_metrics["nucleus_mask_areamm2"].append(metrics["global"]["nucleus"]["area_mm2"])
        global_metrics["nucleus_mask_density"].append(metrics["global"]["nucleus"]["density"])
        global_metrics["repaired_cell_mask_nobj"].append(metrics["global"]["repaired_cell"]["n_objects"])
        global_metrics["repaired_cell_mask_areamm2"].append(metrics["global"]["repaired_cell"]["area_mm2"])
        global_metrics["repaired_cell_mask_density"].append(metrics["global"]["repaired_cell"]["density"])
        global_metrics["repaired_nucleus_mask_nobj"].append(metrics["global"]["repaired_nucleus"]["n_objects"])
        global_metrics["repaired_nucleus_mask_areamm2"].append(metrics["global"]["repaired_nucleus"]["area_mm2"])
        global_metrics["repaired_nucleus_mask_density"].append(metrics["global"]["repaired_nucleus"]["density"])
        global_metrics["unmatched_nucleus_mask_nobj"].append(metrics["global"]["unmatched_nucleus"]["n_objects"])
        global_metrics["unmatched_nucleus_mask_areamm2"].append(metrics["global"]["unmatched_nucleus"]["area_mm2"])
        global_metrics["unmatched_nucleus_mask_density"].append(metrics["global"]["unmatched_nucleus"]["density"])

        # Local metrics
        local_metrics["cell_mask_local_density_mean"].append(metrics["local"]["cell"]["mean"])
        local_metrics["cell_mask_local_density_median"].append(metrics["local"]["cell"]["median"])
        local_metrics["cell_mask_local_density_std"].append(metrics["local"]["cell"]["std"])
        local_metrics["cell_mask_local_density_quantile_25"].append(metrics["local"]["cell"]["quantile_25"])
        local_metrics["cell_mask_local_density_quantile_75"].append(metrics["local"]["cell"]["quantile_75"])

        local_metrics["nucleus_mask_local_density_mean"].append(metrics["local"]["nucleus"]["mean"])
        local_metrics["nucleus_mask_local_density_median"].append(metrics["local"]["nucleus"]["median"])
        local_metrics["nucleus_mask_local_density_std"].append(metrics["local"]["nucleus"]["std"])
        local_metrics["nucleus_mask_local_density_quantile_25"].append(metrics["local"]["nucleus"]["quantile_25"])
        local_metrics["nucleus_mask_local_density_quantile_75"].append(metrics["local"]["nucleus"]["quantile_75"])

        local_metrics["repaired_cell_mask_local_density_mean"].append(metrics["local"]["repaired_cell"]["mean"])
        local_metrics["repaired_cell_mask_local_density_median"].append(metrics["local"]["repaired_cell"]["median"])
        local_metrics["repaired_cell_mask_local_density_std"].append(metrics["local"]["repaired_cell"]["std"])
        local_metrics["repaired_cell_mask_local_density_quantile_25"].append(metrics["local"]["repaired_cell"]["quantile_25"])
        local_metrics["repaired_cell_mask_local_density_quantile_75"].append(metrics["local"]["repaired_cell"]["quantile_75"])

        local_metrics["repaired_nucleus_mask_local_density_mean"].append(metrics["local"]["repaired_nucleus"]["mean"])
        local_metrics["repaired_nucleus_mask_local_density_median"].append(metrics["local"]["repaired_nucleus"]["median"])
        local_metrics["repaired_nucleus_mask_local_density_std"].append(metrics["local"]["repaired_nucleus"]["std"])
        local_metrics["repaired_nucleus_mask_local_density_quantile_25"].append(metrics["local"]["repaired_nucleus"]["quantile_25"])
        local_metrics["repaired_nucleus_mask_local_density_quantile_75"].append(metrics["local"]["repaired_nucleus"]["quantile_75"])

        local_metrics["unmatched_nucleus_mask_local_density_mean"].append(metrics["local"]["unmatched_nucleus"]["mean"])
        local_metrics["unmatched_nucleus_mask_local_density_median"].append(metrics["local"]["unmatched_nucleus"]["median"])
        local_metrics["unmatched_nucleus_mask_local_density_std"].append(metrics["local"]["unmatched_nucleus"]["std"])
        local_metrics["unmatched_nucleus_mask_local_density_quantile_25"].append(metrics["local"]["unmatched_nucleus"]["quantile_25"])
        local_metrics["unmatched_nucleus_mask_local_density_quantile_75"].append(metrics["local"]["unmatched_nucleus"]["quantile_75"])

    # Update dataframe
    for metric, values in global_metrics.items():
        patches_metadata_df.loc[informative_idx, metric] = values

    for metric, values in local_metrics.items():
        patches_metadata_df.loc[informative_idx, metric] = values

    elapsed_time = time.time() - start_time
    logger.info(f"Metadata update completed in {elapsed_time:.2f} seconds")
    return patches_metadata_df 