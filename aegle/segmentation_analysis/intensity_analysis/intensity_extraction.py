import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Configure logging
logger = logging.getLogger(__name__)

def calculate_mean_intensity_per_object(
    channel_img: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate mean intensity per object in a mask.
    
    Args:
        channel_img: Channel image data
        mask: Object mask where each object has a unique integer label
        
    Returns:
        Array of mean intensities per object
    """
    # Get unique object labels (excluding background)
    object_labels = np.unique(mask)
    object_labels = object_labels[object_labels > 0]
    
    # Create a filtered mask that only includes valid objects
    labels_matrix = np.where(np.isin(mask, object_labels), mask, 0).astype(np.int64)
    
    # Calculate sum of intensities per object using bincount
    sum_per_label = np.bincount(labels_matrix.ravel(), weights=channel_img.ravel())[object_labels]
    
    # Calculate count of pixels per object
    count_per_label = np.bincount(labels_matrix.ravel())[object_labels]
    
    # Calculate mean intensity per object
    mean_intensities = sum_per_label / count_per_label
    
    return mean_intensities

def _process_mask(channel_img, mask, mask_name):
    """Helper function to calculate intensities for a mask in parallel"""
    logging.info(f"--- Calculating mean intensity per object for {mask_name} mask")
    return calculate_mean_intensity_per_object(channel_img, mask)

def extract_channel_intensities(
    channel_img: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_matched_mask: np.ndarray,
    nucleus_matched_mask: np.ndarray,
    nucleus_unmatched_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract intensity data for a single channel, calculating mean intensity per cell/nucleus.
    
    Args:
        channel_img: Channel image data
        cell_mask: Original cell mask
        nucleus_mask: Original nucleus mask
        cell_matched_mask: Repaired cell mask
        nucleus_matched_mask: Repaired nucleus mask
        nucleus_unmatched_mask: Mask for unmatched nuclei (optional)
        
    Returns:
        Dictionary containing mean intensities per object for each mask type
    """
    # Calculate mean intensity per object for each mask type in parallel
    logging.info(f"--- Calculating mean intensity per object for each mask type")
    
    # Define masks to process
    masks = {
        "cell": cell_mask,
        "nucleus": nucleus_mask,
        "cell_matched": cell_matched_mask,
        "nucleus_matched": nucleus_matched_mask,
    }
    
    # Add unmatched nuclei mask if provided
    if nucleus_unmatched_mask is not None:
        masks["nucleus_unmatched"] = nucleus_unmatched_mask
    
    # Process masks in parallel
    result = {}
    with ProcessPoolExecutor() as executor:
        # Create a partial function with the channel image already bound
        process_fn = partial(_process_mask, channel_img)
        
        # Submit all tasks
        futures = {
            mask_name: executor.submit(process_fn, mask, mask_name) 
            for mask_name, mask in masks.items()
        }
        
        # Collect results
        for mask_name, future in futures.items():
            result[mask_name] = future.result()
    
    return result

def calculate_intensity_statistics(intensities: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for an array of intensity values.
    
    Args:
        intensities: Array of intensity values
        
    Returns:
        Dictionary containing statistics (mean, median, std)
    """
    if len(intensities) > 0:
        return {
            "mean": float(np.mean(intensities)),
            "median": float(np.median(intensities)),
            "std": float(np.std(intensities))
        }
    else:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0
        }

def calculate_intensity_bias_metrics(
    original_stats: Dict[str, float],
    repaired_stats: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate bias metrics comparing original and repaired statistics.
    
    Args:
        original_stats: Dictionary of statistics for original mask
        repaired_stats: Dictionary of statistics for repaired mask
        
    Returns:
        Dictionary containing bias metrics
    """
    # Calculate percent change in mean intensity
    original_mean = original_stats["mean"]
    repaired_mean = repaired_stats["mean"]
    
    if original_mean > 0:
        percent_change = ((repaired_mean - original_mean) / original_mean) * 100
    else:
        percent_change = 0.0
    
    return {
        "percent_change": percent_change,
        "absolute_change": repaired_mean - original_mean
    }

def extract_intensity_data_across_channels(
    image_ndarray: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_matched_mask: np.ndarray,
    nucleus_matched_mask: np.ndarray,
    nucleus_unmatched_mask: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Extract intensity data across all channels, including cell-level intensities and statistics.
    
    Args:
        image_ndarray: Image data array with multiple channels
        cell_mask: Original cell mask
        nucleus_mask: Original nucleus mask
        cell_matched_mask: Repaired cell mask
        nucleus_matched_mask: Repaired nucleus mask
        nucleus_unmatched_mask: Mask for unmatched nuclei (optional)
        channel_names: Optional list of channel names
        
    Returns:
        Dictionary containing intensity data for each channel
    """
    logger.info("Extracting intensity data across channels")
    
    num_channels = image_ndarray.shape[-1]
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(num_channels)]
    
    intensity_data = {}
    
    for channel_idx in range(num_channels):
        channel_name = channel_names[channel_idx]
        logger.info(f"- Processing channel {channel_idx+1}/{num_channels}: {channel_name}")
        
        # Extract channel data
        channel_img = image_ndarray[..., channel_idx]
        
        # Extract intensities for this channel
        channel_intensities = extract_channel_intensities(
            channel_img,
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
            nucleus_unmatched_mask,
        )
        
        # Calculate statistics for each mask type
        stats = {
            "cell": calculate_intensity_statistics(channel_intensities["cell"]),
            "nucleus": calculate_intensity_statistics(channel_intensities["nucleus"]),
            "cell_matched": calculate_intensity_statistics(channel_intensities["cell_matched"]),
            "nucleus_matched": calculate_intensity_statistics(channel_intensities["nucleus_matched"]),
        }
        
        # Add statistics for unmatched nuclei if available
        if nucleus_unmatched_mask is not None:
            stats["nucleus_unmatched"] = calculate_intensity_statistics(channel_intensities["nucleus_unmatched"])
        
        # Calculate bias metrics
        bias_metrics = {
            "cell": calculate_intensity_bias_metrics(stats["cell"], stats["cell_matched"]),
            "nucleus": calculate_intensity_bias_metrics(stats["nucleus"], stats["nucleus_matched"]),
        }
        
        # Store all results for this channel
        intensity_data[channel_name] = {
            "intensities": channel_intensities,
            "statistics": stats,
            "bias_metrics": bias_metrics,
        }
    
    logger.info("Completed intensity data extraction")
    return intensity_data 