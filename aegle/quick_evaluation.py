# quick_evaluation.py
import os
import sys
import time
import pickle
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import ndimage
from skimage.filters import threshold_mean  # , threshold_otsu
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
from skimage.measure import regionprops

src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
sys.path.append(src_path)
from aegle.codex_patches import CodexPatches
from aegle.evaluation_func import *


def run_quick_evaluation(
    codex_patches: CodexPatches,
    config,
    args=None,
):
    """Run a streamlined, *per‑patch* evaluation of repaired segmentations.

    Args:
        codex_patches: CodexPatches object containing patches and segmentation data
        config: Configuration Dictionary for evaluation options
        args: Optional additional arguments
    """
    # # save codex_patches into pickle file for debug
    # file_name = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/D18_Scan1_1_markerset_1_codex_patches.pkl"
    # with open(file_name, "wb") as f:
    #     pickle.dump(codex_patches, f)
    # logging.info(f"Saved CodexPatches object to {file_name}")

    # Extract segmentation data and metadata
    original_seg_res_batch = codex_patches.original_seg_res_batch
    repaired_seg_res_batch = codex_patches.repaired_seg_res_batch
    patches_metadata_df = codex_patches.get_patches_metadata()
    antibody_df = codex_patches.antibody_df
    logging.info(f"antibody_df:\n{antibody_df}")
    antibody_list = antibody_df["antibody_name"].to_list()

    # Filter for informative patches
    informative_idx = patches_metadata_df["is_informative"] == True
    logging.info(f"Number of informative patches: {informative_idx.sum()}")
    image_ndarray = codex_patches.all_channel_patches[informative_idx]
    logging.info(f"image_ndarray.shape: {image_ndarray.shape}")

    output_dir = config.get("output_dir", "./output")

    # --- Matched fraction ------------------------------------------
    # This is precalculated after segmentation in segmentation.py
    matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
    patches_metadata_df.loc[informative_idx, "matched_fraction"] = matched_fraction_list

    # Get microns per pixel from config
    image_mpp = config.get("data", {}).get("image_mpp", 0.5)

    # List to store all density metrics
    all_density_metrics = []

    # Process each patch
    res_list = []
    for idx, repaired_seg_res in enumerate(repaired_seg_res_batch):
        logging.info(f"Processing patch {idx+1}/{len(repaired_seg_res_batch)}")
        if repaired_seg_res is None:
            logging.warning(f"Repaired segmentation result for patch {idx} is None.")
            continue

        # Visualize results if specified in config
        patch_output_dir = f"{output_dir}/patch_{idx}"
        # ----------
        # Extract masks from original and repaired segmentation results
        # ----------
        original_seg_res = original_seg_res_batch[idx]
        repaired_seg_res = repaired_seg_res_batch[idx]

        # Get masks from original segmentation results
        cell_mask = original_seg_res.get("cell")
        nucleus_mask = original_seg_res.get("nucleus")
        logging.info(f"cell_mask.shape: {cell_mask.shape}")
        logging.info(f"nucleus_mask.shape: {nucleus_mask.shape}")

        # Get masks from repaired segmentation results
        cell_matched_mask = repaired_seg_res["cell_matched_mask"]
        nucleus_matched_mask = repaired_seg_res["nucleus_matched_mask"]
        logging.info(f"cell_matched_mask.shape: {cell_matched_mask.shape}")
        logging.info(f"nucleus_matched_mask.shape: {nucleus_matched_mask.shape}")

        # ----------
        # Compute density metrics
        # ----------
        density_metrics = update_patch_metrics(cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask, image_mpp)
        all_density_metrics.append(density_metrics)

        # ----------
        # Check the gain of intensity from repaired segmentation by barplot
        # ----------
        # Analyze repair bias across channels
        bias_results = analyze_intensity_bias_across_channels(
            image_ndarray[idx],
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
        )

        visualize_channel_bias(
            bias_results,
            output_dir=os.path.join(patch_output_dir, "repairedment_gain"),
            channels_per_figure=config.get("channels_per_figure", 10),
        )

        # ----------
        # Check the shift of intensity between matched and unmatched (density plots)
        # ----------

        # Create density plots for this patch
        density_results = visualize_intensity_distributions(
            image_ndarray[idx],
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
            output_dir=os.path.join(patch_output_dir, "density_shits"),
            use_log_scale=True,
            channel_names=antibody_list,
        )

        # Store results
        evaluation_result = {
            "patch_idx": idx,
            "bias_analysis": bias_results,
            "density_analysis": density_results,
            "density_metrics": density_metrics,
            # Add other metrics as needed
        }
        res_list.append(evaluation_result)

    # Update metadata with density metrics
    patches_metadata_df = update_metadata_with_density_metrics(
        patches_metadata_df, informative_idx, all_density_metrics
    )
    codex_patches.seg_evaluation_metrics = res_list
    codex_patches.set_metadata(patches_metadata_df)


def analyze_repair_bias_across_channels(
    image_ndarray, cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask
):
    """
    Analyze if mask repair introduces any bias across different channels of the CODEX image.
    Optimized for speed with vectorized operations.

    Args:
        image_ndarray: narray of shape (height, width, num_channels)
        cell_mask: Original cell mask (2D array)
        nucleus_mask: Original nucleus mask (2D array)
        cell_matched_mask: Repaired cell mask (2D array)
        nucleus_matched_mask: Repaired nucleus mask (2D array)
    Returns:
        Dictionary with channel-wise analysis
    """
    # Validate required masks
    if cell_mask is None or nucleus_mask is None:
        logging.error("Missing original masks in repaired result")
        return {}
    # Validate matched masks
    if cell_matched_mask is None or nucleus_matched_mask is None:
        logging.error("Missing matched masks in repaired result")
        return {}

    # Determine number of channels
    logging.info(f"image_ndarray.shape: {image_ndarray.shape}")
    num_channels = image_ndarray.shape[2]

    # Initialize results
    channel_results = {}

    # Pre-compute mask stats once (instead of for each channel)
    start_time = time.time()
    logging.info("Pre-computing mask statistics...")

    # Extract unique object IDs for each mask (excluding background)
    # TODO: I would like to check if orig and matched shared the same IDs for the same cells
    masks = {
        "orig_cell": {
            "mask": cell_mask,
            "ids": np.unique(cell_mask)[np.unique(cell_mask) > 0],
        },
        "orig_nucleus": {
            "mask": nucleus_mask,
            "ids": np.unique(nucleus_mask)[np.unique(nucleus_mask) > 0],
        },
        "matched_cell": {
            "mask": cell_matched_mask,
            "ids": np.unique(cell_matched_mask)[np.unique(cell_matched_mask) > 0],
        },
        "matched_nucleus": {
            "mask": nucleus_matched_mask,
            "ids": np.unique(nucleus_matched_mask)[np.unique(nucleus_matched_mask) > 0],
        },
    }

    # Store counts
    counts = {
        mask_name: len(mask_data["ids"]) for mask_name, mask_data in masks.items()
    }

    logging.info(f"Masks prepared in {time.time() - start_time:.2f}s")
    logging.info(
        f"Object counts - Original: {counts['orig_cell']} cells, {counts['orig_nucleus']} nuclei | "
        + f"Matched: {counts['matched_cell']} cells, {counts['matched_nucleus']} nuclei"
    )

    # Process each channel
    for c in range(num_channels):
        start_channel = time.time()
        logging.info(f"Analyzing channel {c+1}/{num_channels}")
        channel_image = image_ndarray[:, :, c]

        # Calculate stats for all mask types using optimized function
        stats = {}
        for mask_name, mask_data in masks.items():
            stats[mask_name] = get_mask_intensity_stats(
                channel_image, mask_data["mask"], mask_data["ids"]
            )
            stats[mask_name]["count"] = counts[mask_name]

        # Calculate bias metrics (percent change in mean intensity)
        cell_intensity_change = calc_percent_change(
            stats["matched_cell"]["mean"], stats["orig_cell"]["mean"]
        )

        nucleus_intensity_change = calc_percent_change(
            stats["matched_nucleus"]["mean"], stats["orig_nucleus"]["mean"]
        )

        # Calculate bias metrics (percent change in mean intensity)
        cell_intensity_change = (
            (
                (stats["matched_cell"]["mean"] - stats["orig_cell"]["mean"])
                / stats["orig_cell"]["mean"]
                * 100
            )
            if stats["orig_cell"]["mean"] > 0
            else 0
        )

        nucleus_intensity_change = (
            (
                (stats["matched_nucleus"]["mean"] - stats["orig_nucleus"]["mean"])
                / stats["orig_nucleus"]["mean"]
                * 100
            )
            if stats["orig_nucleus"]["mean"] > 0
            else 0
        )

        # Store results
        channel_results[f"Channel_{c}"] = {
            "original": {"cell": stats["orig_cell"], "nucleus": stats["orig_nucleus"]},
            "repaired": {
                "cell": stats["matched_cell"],
                "nucleus": stats["matched_nucleus"],
            },
            "bias_metrics": {
                "cell_intensity_percent_change": cell_intensity_change,
                "nucleus_intensity_percent_change": nucleus_intensity_change,
            },
        }

        print(f"Channel {c+1} processed in {time.time() - start_channel:.2f}s")

    return channel_results


def calc_percent_change(new_val, old_val):
    """Calculate percent change between values, handling division by zero"""
    if old_val == 0 or np.isclose(old_val, 0):
        return 0
    return (new_val - old_val) / old_val * 100


def get_mask_intensity_stats(image, mask, object_ids=None):
    """
    Calculate intensity statistics for objects in a given mask.

    Args:
        image: 2D image
        mask: 2D segmentation mask
        object_ids: Optional precomputed unique object IDs

    Returns:
        Dictionary with intensity statistics
    """
    import numpy as np
    from scipy import ndimage

    # If object_ids not provided, compute them
    if object_ids is None:
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids > 0]  # Exclude background

    if len(object_ids) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "count": 0}

    # Calculate mean intensity for each object using ndimage.labeled_statistics
    mean_intensities = ndimage.mean(image, labels=mask, index=object_ids)

    # Compute statistics on the mean intensities
    return {
        "mean": np.mean(mean_intensities),
        "median": np.median(mean_intensities),
        "std": np.std(mean_intensities),
        "min": np.min(mean_intensities),
        "max": np.max(mean_intensities),
        "count": len(mean_intensities),
    }


def visualize_channel_bias(channel_results, output_dir=None, channels_per_figure=10):
    """
    Visualize the bias across channels, creating summary and detailed plots.

    Args:
        channel_results: Results from analyze_repair_bias_across_channels
        output_dir: Directory to save figures. (None to skip saving)
        channels_per_figure: Number of channels to include in each figure
    """

    channels = list(channel_results.keys())

    # If we have many channels, group them into multiple figures
    channel_groups = [
        channels[i : i + channels_per_figure]
        for i in range(0, len(channels), channels_per_figure)
    ]

    # Create summary figure for all channels
    create_summary_plot(channel_results, channels, output_dir)

    # Create detailed figures for channel groups
    for group_idx, channel_group in enumerate(channel_groups):
        create_group_plot(channel_results, channel_group, group_idx, output_dir)

    # Create individual channel plots
    for channel in channels:
        create_channel_plot(channel_results, channel, output_dir)

    # Generate summary of channels with significant bias
    report_significant_bias(channel_results, channels, output_dir)


def create_summary_plot(channel_results, channels, output_dir=None):
    """Create summary plot for all channels"""
    plt.figure(figsize=(16, 8))
    x = np.arange(len(channels))
    cell_bias = [
        channel_results[c]["bias_metrics"]["cell_intensity_percent_change"]
        for c in channels
    ]
    nucleus_bias = [
        channel_results[c]["bias_metrics"]["nucleus_intensity_percent_change"]
        for c in channels
    ]

    plt.bar(x - 0.2, cell_bias, 0.4, label="Cell Bias")
    plt.bar(x + 0.2, nucleus_bias, 0.4, label="Nucleus Bias")
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Channel Index")
    plt.ylabel("Intensity % Change (Repaired vs Original)")
    plt.title("Repair Bias Summary Across All Channels")
    plt.xticks(x, [c.split("_")[1] for c in channels], rotation=90)
    plt.legend()
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "all_channels_bias_summary.png"), dpi=300)
        plt.close()
    else:
        plt.show()


def create_group_plot(channel_results, channel_group, group_idx, output_dir=None):
    """Create plot for a group of channels"""
    plt.figure(figsize=(max(12, len(channel_group) * 1.2), 6))

    x = np.arange(len(channel_group))
    cell_bias = [
        channel_results[c]["bias_metrics"]["cell_intensity_percent_change"]
        for c in channel_group
    ]
    nucleus_bias = [
        channel_results[c]["bias_metrics"]["nucleus_intensity_percent_change"]
        for c in channel_group
    ]

    plt.bar(x - 0.2, cell_bias, 0.4, label="Cell Bias")
    plt.bar(x + 0.2, nucleus_bias, 0.4, label="Nucleus Bias")

    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Channels")
    plt.ylabel("Intensity % Change (Repaired vs Original)")
    plt.title(f"Repair Bias Across Channels (Group {group_idx+1})")
    plt.xticks(x, [c.split("_")[1] for c in channel_group])
    plt.legend()
    plt.tight_layout()

    if output_dir:
        plt.savefig(
            os.path.join(output_dir, f"channel_bias_group_{group_idx+1}.png"), dpi=300
        )
        plt.close()
    else:
        plt.show()


def create_channel_plot(channel_results, channel, output_dir=None):
    """Create detailed plot for individual channel"""
    plt.figure(figsize=(10, 6))

    # Extract stats
    orig_cell = channel_results[channel]["original"]["cell"]
    orig_nuc = channel_results[channel]["original"]["nucleus"]
    rep_cell = channel_results[channel]["repaired"]["cell"]
    rep_nuc = channel_results[channel]["repaired"]["nucleus"]

    # Create bar plot for this specific channel
    metrics = ["mean", "median", "std"]
    x = np.arange(len(metrics))
    width = 0.2

    # Plot cell metrics
    plt.bar(
        x - 0.3,
        [orig_cell[m] for m in metrics],
        width,
        label="Original Cell",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        x - 0.1,
        [rep_cell[m] for m in metrics],
        width,
        label="Repaired Cell",
        color="lightblue",
        alpha=0.7,
    )

    # Plot nucleus metrics
    plt.bar(
        x + 0.1,
        [orig_nuc[m] for m in metrics],
        width,
        label="Original Nucleus",
        color="red",
        alpha=0.7,
    )
    plt.bar(
        x + 0.3,
        [rep_nuc[m] for m in metrics],
        width,
        label="Repaired Nucleus",
        color="salmon",
        alpha=0.7,
    )

    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title(f"Channel {channel}: Original vs Repaired Statistics")
    plt.xticks(x, metrics)
    plt.legend()

    # Add counts as text
    plt.figtext(
        0.5,
        0.01,
        f"Counts - Original: {orig_cell['count']} cells, {orig_nuc['count']} nuclei | "
        + f"Repaired: {rep_cell['count']} cells, {rep_nuc['count']} nuclei",
        ha="center",
    )

    plt.tight_layout()

    if output_dir:
        plt.savefig(
            os.path.join(output_dir, f"channel_{channel}_detailed.png"), dpi=300
        )
        plt.close()
    else:
        plt.show()


def report_significant_bias(channel_results, channels, output_dir=None, threshold=5.0):
    """Report channels with significant bias above threshold"""
    significant_bias = []
    for c in channels:
        cell_change = channel_results[c]["bias_metrics"][
            "cell_intensity_percent_change"
        ]
        nuc_change = channel_results[c]["bias_metrics"][
            "nucleus_intensity_percent_change"
        ]

        if abs(cell_change) > threshold or abs(nuc_change) > threshold:
            significant_bias.append((c, cell_change, nuc_change))

    # Write summary to file if output directory provided
    if output_dir:
        with open(os.path.join(output_dir, "significant_bias_summary.txt"), "w") as f:
            if significant_bias:
                f.write(f"Channels with significant bias (>{threshold}% change):\n")
                for c, cell_ch, nuc_ch in significant_bias:
                    f.write(f"{c}: Cell: {cell_ch:.2f}%, Nucleus: {nuc_ch:.2f}%\n")
            else:
                f.write(f"No channels show significant bias (>{threshold}% change)\n")

    # Also print to console
    if significant_bias:
        logging.info(f"Channels with significant bias (>{threshold}% change):")
        for c, cell_ch, nuc_ch in significant_bias:
            logging.info(f"{c}: Cell: {cell_ch:.2f}%, Nucleus: {nuc_ch:.2f}%")
    else:
        logging.info(f"No channels show significant bias (>{threshold}% change)")


def create_comparison_density_plots(
    image_array,
    cell_mask,
    nucleus_mask,
    cell_matched_mask,
    nucleus_matched_mask,
    output_dir=None,
    use_log_scale=True,
    channel_names=None,
):
    """
    Create density plots comparing intensity distributions between:
    1. Matched vs Unmatched nuclei
    2. Matched vs Unmatched cells
    3. All nuclei vs Unselected regions

    Args:
        image_array: 3D array of shape (height, width, channels)
        cell_mask: Original cell mask
        nucleus_mask: Original nucleus mask
        cell_matched_mask: Repaired cell mask
        nucleus_matched_mask: Repaired nucleus mask
        output_dir: Directory to save plots (if None, will display them)
        use_log_scale: Whether to use log scaling for x-axis
        channel_names: Dictionary mapping channel indices to names (optional)

    Returns:
        Dictionary with intensity data for each channel
    """
    logging.info("Creating comparison density plots...")
    start_time = time.time()

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get number of channels
    num_channels = image_array.shape[2]

    # If no channel names provided, use defaults
    if channel_names is None:
        channel_names = {i: f"Channel {i}" for i in range(num_channels)}

    channels_to_plot = list(range(num_channels))
    logging.info(f"Processing {len(channels_to_plot)}channels")

    # Pre-compute masks and IDs once for all channels
    mask_data = precompute_mask_data(
        nucleus_mask, nucleus_matched_mask, cell_mask, cell_matched_mask
    )

    # Initialize data containers
    channel_data = defaultdict(
        lambda: {
            "matched_nuclei": [],
            "unmatched_nuclei": [],
            "matched_cells": [],
            "unmatched_cells": [],
            "all_nuclei": [],
            "unselected": [],
        }
    )

    # Process each requested channel
    for c in channels_to_plot:
        channel_start = time.time()

        # Get channel name
        channel_name = channel_names.get(c, f"Channel {c}")
        logging.info(f"Processing {channel_name} (channel {c+1}/{num_channels})")

        # Extract current channel image
        channel_img = image_array[:, :, c]

        # Process intensities for this channel
        channel_data[c] = process_channel_intensities(channel_img, mask_data)

        # Create and save/show plots
        create_channel_plots(
            channel_data[c], channel_name, c, output_dir, use_log_scale
        )

        logging.info(
            f"Channel {c} ({channel_name}) processed in {time.time() - channel_start:.2f}s"
        )

    logging.info(f"All density plots completed in {time.time() - start_time:.2f}s")
    return dict(channel_data)


def precompute_mask_data(
    nucleus_mask, nucleus_matched_mask, cell_mask, cell_matched_mask
):
    """Helper function to precompute all mask data needed for analysis"""
    import numpy as np
    import logging

    # Get unique IDs for nuclei
    all_nucleus_ids = np.unique(nucleus_mask)
    all_nucleus_ids = all_nucleus_ids[all_nucleus_ids > 0]  # Remove background

    matched_nucleus_ids = np.unique(nucleus_matched_mask)
    matched_nucleus_ids = matched_nucleus_ids[matched_nucleus_ids > 0]

    # Create set for faster lookup
    matched_nucleus_set = set(matched_nucleus_ids)

    # Find unmatched nuclei
    unmatched_nucleus_ids = np.array(
        [nid for nid in all_nucleus_ids if nid not in matched_nucleus_set]
    )

    # Get unique IDs for cells
    all_cell_ids = np.unique(cell_mask)
    all_cell_ids = all_cell_ids[all_cell_ids > 0]

    matched_cell_ids = np.unique(cell_matched_mask)
    matched_cell_ids = matched_cell_ids[matched_cell_ids > 0]

    # Create set for faster lookup
    matched_cell_set = set(matched_cell_ids)

    # Find unmatched cells
    unmatched_cell_ids = np.array(
        [cid for cid in all_cell_ids if cid not in matched_cell_set]
    )

    # Create unselected mask (regions with no nuclei)
    unselected_mask = nucleus_mask == 0

    logging.info(
        f"Found {len(all_nucleus_ids)} total nuclei, {len(matched_nucleus_ids)} matched, {len(unmatched_nucleus_ids)} unmatched"
    )
    logging.info(
        f"Found {len(all_cell_ids)} total cells, {len(matched_cell_ids)} matched, {len(unmatched_cell_ids)} unmatched"
    )

    return {
        "nucleus_mask": nucleus_mask,
        "cell_mask": cell_mask,
        "matched_nucleus_ids": matched_nucleus_ids,
        "unmatched_nucleus_ids": unmatched_nucleus_ids,
        "matched_cell_ids": matched_cell_ids,
        "unmatched_cell_ids": unmatched_cell_ids,
        "unselected_mask": unselected_mask,
    }


def process_channel_intensities(channel_img, mask_data):
    """Process intensity data for a single channel"""
    import numpy as np
    from scipy import ndimage

    result = {
        "matched_nuclei": np.array([]),
        "unmatched_nuclei": np.array([]),
        "matched_cells": np.array([]),
        "unmatched_cells": np.array([]),
        "all_nuclei": np.array([]),
        "unselected": np.array([]),
    }

    # Process nuclei
    if len(mask_data["matched_nucleus_ids"]) > 0:
        result["matched_nuclei"] = ndimage.mean(
            channel_img,
            labels=mask_data["nucleus_mask"],
            index=mask_data["matched_nucleus_ids"],
        )

    if len(mask_data["unmatched_nucleus_ids"]) > 0:
        result["unmatched_nuclei"] = ndimage.mean(
            channel_img,
            labels=mask_data["nucleus_mask"],
            index=mask_data["unmatched_nucleus_ids"],
        )

    # Process cells
    if len(mask_data["matched_cell_ids"]) > 0:
        result["matched_cells"] = ndimage.mean(
            channel_img,
            labels=mask_data["cell_mask"],
            index=mask_data["matched_cell_ids"],
        )

    if len(mask_data["unmatched_cell_ids"]) > 0:
        result["unmatched_cells"] = ndimage.mean(
            channel_img,
            labels=mask_data["cell_mask"],
            index=mask_data["unmatched_cell_ids"],
        )

    # All nuclei (combine matched and unmatched)
    if len(result["matched_nuclei"]) > 0 or len(result["unmatched_nuclei"]) > 0:
        result["all_nuclei"] = (
            np.concatenate([result["matched_nuclei"], result["unmatched_nuclei"]])
            if len(result["matched_nuclei"]) > 0 and len(result["unmatched_nuclei"]) > 0
            else (
                result["matched_nuclei"]
                if len(result["matched_nuclei"]) > 0
                else result["unmatched_nuclei"]
            )
        )

    # Sample unselected regions
    if np.any(mask_data["unselected_mask"]):
        unselected_indices = np.where(mask_data["unselected_mask"])
        # Limit sample size for performance
        sample_size = min(10000, len(unselected_indices[0]))

        if sample_size > 0:
            # Sample random points from unselected regions
            sample_idx = np.random.choice(
                len(unselected_indices[0]), sample_size, replace=False
            )

            result["unselected"] = channel_img[
                unselected_indices[0][sample_idx], unselected_indices[1][sample_idx]
            ]

    return result


def create_channel_plots(
    channel_data, channel_name, channel_idx, output_dir, use_log_scale
):
    """Create and save/show plots for a channel"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Matched vs Unmatched nuclei
    create_kde_plot(
        ax1,
        [
            (
                channel_data["matched_nuclei"],
                f'Matched Nuclei (n={len(channel_data["matched_nuclei"])})',
                0.5,
            ),
            (
                channel_data["unmatched_nuclei"],
                f'Unmatched Nuclei (n={len(channel_data["unmatched_nuclei"])})',
                0.5,
            ),
        ],
        f"{channel_name}: Matched vs Unmatched Nuclei",
        "Mean Intensity",
        "Density",
        use_log_scale,
    )

    # Plot 2: Matched vs Unmatched cells
    create_kde_plot(
        ax2,
        [
            (
                channel_data["matched_cells"],
                f'Matched Cells (n={len(channel_data["matched_cells"])})',
                0.5,
            ),
            (
                channel_data["unmatched_cells"],
                f'Unmatched Cells (n={len(channel_data["unmatched_cells"])})',
                0.5,
            ),
        ],
        f"{channel_name}: Matched vs Unmatched Cells",
        "Mean Intensity",
        "Density",
        use_log_scale,
    )

    # Plot 3: All Nuclei vs Unselected
    create_kde_plot(
        ax3,
        [
            (
                channel_data["all_nuclei"],
                f'All Nuclei (n={len(channel_data["all_nuclei"])})',
                0.5,
            ),
            (
                channel_data["unselected"],
                f'Unselected Regions (n={len(channel_data["unselected"])})',
                0.5,
            ),
        ],
        f"{channel_name}: All Nuclei vs Unselected Regions",
        "Mean Intensity",
        "Density",
        use_log_scale,
    )

    plt.tight_layout()

    # Handle saving or displaying
    if output_dir:
        # Use channel name in filename (replace spaces and slashes with underscores for safety)
        safe_name = str(channel_name).replace(" ", "_").replace("/", "_")
        plt.savefig(
            os.path.join(output_dir, f"ch{channel_idx}-{safe_name}_density_plots.png"),
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


def create_kde_plot(ax, data_items, title, xlabel, ylabel, use_log_scale):
    """Helper function to create a KDE plot on the given axis"""
    import seaborn as sns
    import numpy as np

    # Plot each data item with KDE
    for data, label, alpha in data_items:
        if len(data) > 0:
            sns.kdeplot(data, ax=ax, label=label, fill=True, alpha=alpha)

    # Set plot labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # Handle log scale if requested
    if use_log_scale:
        try:
            # Set log scale
            ax.set_xscale("log")
            ax.set_xlabel(f"{xlabel} (log scale)")

            # Fix log(0) issues by setting proper xlim
            lines = ax.get_lines()
            min_vals = []
            for line in lines:
                data = line.get_xdata()
                positive_data = data[data > 0]
                if len(positive_data) > 0:
                    min_vals.append(np.min(positive_data))

            if min_vals:
                min_val = max(0.1, min(min_vals) * 0.9)
                ax.set_xlim(left=min_val)
        except Exception:
            # Fallback if there's an issue
            ax.set_xlim(left=0.1)


def compute_global_density_metrics(mask, image_mpp=0.5):
    """
    Compute global density metrics for a mask.

    Args:
        mask (np.ndarray): Labeled mask
        image_mpp (float): Microns per pixel

    Returns:
        dict: Dictionary with global density metrics
    """
    return compute_mask_density(mask, image_mpp)


def compute_local_density_metrics(mask, image_mpp=0.5, window_size_microns=200):
    """
    Compute local density metrics for a mask.

    Args:
        mask (np.ndarray): Labeled mask
        image_mpp (float): Microns per pixel
        window_size_microns (float): Size of window for local density calculation

    Returns:
        dict: Dictionary with local density metrics
    """
    local_densities = compute_local_densities(
        mask, image_mpp=image_mpp, window_size_microns=window_size_microns
    )
    return compute_local_density_stats(local_densities)


def update_patch_metrics(cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask, image_mpp=0.5):
    """
    Compute all density metrics for a single patch.

    Args:
        original_seg_res (dict): Original segmentation results
        repaired_seg_res (dict): Repaired segmentation results
        image_mpp (float): Microns per pixel

    Returns:
        dict: Dictionary with all density metrics
    """
    # Global density metrics
    metrics = {
        "global": {
            "cell": compute_global_density_metrics(cell_mask, image_mpp),
            "nucleus": compute_global_density_metrics(nucleus_mask, image_mpp),
            "repaired_cell": compute_global_density_metrics(
                cell_matched_mask, image_mpp
            ),
            "repaired_nucleus": compute_global_density_metrics(
                nucleus_matched_mask, image_mpp
            ),
        },
        "local": {
            "cell": compute_local_density_metrics(cell_mask, image_mpp),
            "nucleus": compute_local_density_metrics(nucleus_mask, image_mpp),
            "repaired_cell": compute_local_density_metrics(
                cell_matched_mask, image_mpp
            ),
            "repaired_nucleus": compute_local_density_metrics(
                nucleus_matched_mask, image_mpp
            ),
        },
    }

    return metrics


def update_metadata_with_density_metrics(
    patches_metadata_df, informative_idx, metrics_list
):
    """
    Update metadata dataframe with density metrics.

    Args:
        patches_metadata_df (pd.DataFrame): Metadata dataframe
        informative_idx (pd.Series): Boolean mask for the rows to update
        metrics_list (list): List of dictionaries with metrics for each patch
    """
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
    }

    local_metrics = {
        "cell_mask_local_density_mean": [],
        "cell_mask_local_density_median": [],
        "cell_mask_local_density_std": [],
        "cell_mask_local_density_qunatile_25": [],
        "cell_mask_local_density_qunatile_75": [],
        "nucleus_mask_local_density_mean": [],
        "nucleus_mask_local_density_median": [],
        "nucleus_mask_local_density_std": [],
        "nucleus_mask_local_density_qunatile_25": [],
        "nucleus_mask_local_density_qunatile_75": [],
        "repaired_cell_mask_local_density_mean": [],
        "repaired_cell_mask_local_density_median": [],
        "repaired_cell_mask_local_density_std": [],
        "repaired_cell_mask_local_density_qunatile_25": [],
        "repaired_cell_mask_local_density_qunatile_75": [],
        "repaired_nucleus_mask_local_density_mean": [],
        "repaired_nucleus_mask_local_density_median": [],
        "repaired_nucleus_mask_local_density_std": [],
        "repaired_nucleus_mask_local_density_qunatile_25": [],
        "repaired_nucleus_mask_local_density_qunatile_75": [],
    }

    # Extract metrics for each patch
    for metrics in metrics_list:
        # Global metrics
        global_metrics["cell_mask_nobj"].append(metrics["global"]["cell"]["n_objects"])
        global_metrics["cell_mask_areamm2"].append(
            metrics["global"]["cell"]["area_mm2"]
        )
        global_metrics["cell_mask_density"].append(metrics["global"]["cell"]["density"])
        global_metrics["nucleus_mask_nobj"].append(
            metrics["global"]["nucleus"]["n_objects"]
        )
        global_metrics["nucleus_mask_areamm2"].append(
            metrics["global"]["nucleus"]["area_mm2"]
        )
        global_metrics["nucleus_mask_density"].append(
            metrics["global"]["nucleus"]["density"]
        )
        global_metrics["repaired_cell_mask_nobj"].append(
            metrics["global"]["repaired_cell"]["n_objects"]
        )
        global_metrics["repaired_cell_mask_areamm2"].append(
            metrics["global"]["repaired_cell"]["area_mm2"]
        )
        global_metrics["repaired_cell_mask_density"].append(
            metrics["global"]["repaired_cell"]["density"]
        )
        global_metrics["repaired_nucleus_mask_nobj"].append(
            metrics["global"]["repaired_nucleus"]["n_objects"]
        )
        global_metrics["repaired_nucleus_mask_areamm2"].append(
            metrics["global"]["repaired_nucleus"]["area_mm2"]
        )
        global_metrics["repaired_nucleus_mask_density"].append(
            metrics["global"]["repaired_nucleus"]["density"]
        )

        # Local metrics
        local_metrics["cell_mask_local_density_mean"].append(
            metrics["local"]["cell"]["mean"]
        )
        local_metrics["cell_mask_local_density_median"].append(
            metrics["local"]["cell"]["median"]
        )
        local_metrics["cell_mask_local_density_std"].append(
            metrics["local"]["cell"]["std"]
        )
        local_metrics["cell_mask_local_density_qunatile_25"].append(
            metrics["local"]["cell"]["qunatile_25"]
        )
        local_metrics["cell_mask_local_density_qunatile_75"].append(
            metrics["local"]["cell"]["qunatile_75"]
        )

        local_metrics["nucleus_mask_local_density_mean"].append(
            metrics["local"]["nucleus"]["mean"]
        )
        local_metrics["nucleus_mask_local_density_median"].append(
            metrics["local"]["nucleus"]["median"]
        )
        local_metrics["nucleus_mask_local_density_std"].append(
            metrics["local"]["nucleus"]["std"]
        )
        local_metrics["nucleus_mask_local_density_qunatile_25"].append(
            metrics["local"]["nucleus"]["qunatile_25"]
        )
        local_metrics["nucleus_mask_local_density_qunatile_75"].append(
            metrics["local"]["nucleus"]["qunatile_75"]
        )

        local_metrics["repaired_cell_mask_local_density_mean"].append(
            metrics["local"]["repaired_cell"]["mean"]
        )
        local_metrics["repaired_cell_mask_local_density_median"].append(
            metrics["local"]["repaired_cell"]["median"]
        )
        local_metrics["repaired_cell_mask_local_density_std"].append(
            metrics["local"]["repaired_cell"]["std"]
        )
        local_metrics["repaired_cell_mask_local_density_qunatile_25"].append(
            metrics["local"]["repaired_cell"]["qunatile_25"]
        )
        local_metrics["repaired_cell_mask_local_density_qunatile_75"].append(
            metrics["local"]["repaired_cell"]["qunatile_75"]
        )

        local_metrics["repaired_nucleus_mask_local_density_mean"].append(
            metrics["local"]["repaired_nucleus"]["mean"]
        )
        local_metrics["repaired_nucleus_mask_local_density_median"].append(
            metrics["local"]["repaired_nucleus"]["median"]
        )
        local_metrics["repaired_nucleus_mask_local_density_std"].append(
            metrics["local"]["repaired_nucleus"]["std"]
        )
        local_metrics["repaired_nucleus_mask_local_density_qunatile_25"].append(
            metrics["local"]["repaired_nucleus"]["qunatile_25"]
        )
        local_metrics["repaired_nucleus_mask_local_density_qunatile_75"].append(
            metrics["local"]["repaired_nucleus"]["qunatile_75"]
        )

    # Update dataframe
    for metric, values in global_metrics.items():
        patches_metadata_df.loc[informative_idx, metric] = values

    for metric, values in local_metrics.items():
        patches_metadata_df.loc[informative_idx, metric] = values

    return patches_metadata_df


def compute_local_density_stats(local_density_list: List[float]) -> Dict[str, float]:
    """
    Compute summary stats (mean, median, std) for a list of local densities.
    """
    if len(local_density_list) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "qunatile_25": 0.0,
            "qunatile_75": 0.0,
        }
    return {
        "mean": float(np.mean(local_density_list)),
        "median": float(np.median(local_density_list)),
        "std": float(np.std(local_density_list)),
        "qunatile_25": float(np.quantile(local_density_list, 0.25)),
        "qunatile_75": float(np.quantile(local_density_list, 0.75)),
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
        mask (np.ndarray): Labeled mask (H, W) with int labels.
        image_mpp (float): Microns per pixel.
        window_size_microns (float): Side length of each sub-window, in microns.
        step_microns (float): Step size for sliding the window; defaults to window_size_microns (non-overlapping).

    Returns:
        List[float]: A list of local densities (cells / mm^2).
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask)

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

    return local_densities

def compute_mask_density(mask, image_mpp: float = 0.5) -> float:
    """
    Computes cell density from a labeled mask.

    Args:
        mask (np.ndarray): Labeled mask of shape (H, W) or (1, H, W). Each unique non-zero label is a cell/nucleus.
        image_mpp (float): Microns per pixel.

    Returns:
        float: Cell/nucleus density in cells per mm².
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask)  # shape becomes (H, W)

    n_objects = len(np.unique(mask)) - (1 if 0 in mask else 0)  # exclude background
    tissue_pixels = np.count_nonzero(mask)  # actual tissue area in pixels
    area_mm2 = tissue_pixels * (image_mpp**2) / 1e6

    if area_mm2 == 0:
        return 0.0  # avoid division by zero

    density = n_objects / area_mm2
    return {
        "n_objects": n_objects,
        "area_mm2": area_mm2,
        "density": density,
    }

def main():
    """
    Main function to test run_quick_evaluation standalone using a saved CodexPatches object.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Path to the saved CodexPatches object
    pickle_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/D18_Scan1_1_markerset_1_codex_patches.pkl"

    if not os.path.exists(pickle_path):
        logging.error(f"CodexPatches pickle file not found at: {pickle_path}")
        sys.exit(1)

    try:
        # Load the CodexPatches object
        logging.info(f"Loading CodexPatches from: {pickle_path}")
        with open(pickle_path, "rb") as f:
            codex_patches = pickle.load(f)

        # Define a configuration for testing
        config = {
            "save_bias_visualizations": True,
            "output_dir": "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle/output_bias_analysis",
            "channels_per_figure": 10,
        }

        # Create output directory if it doesn't exist
        os.makedirs(config["output_dir"], exist_ok=True)

        # Run the quick evaluation
        logging.info("Running quick evaluation...")
        run_quick_evaluation(codex_patches, config)

        # Print results summary
        if hasattr(codex_patches, "seg_evaluation_metrics"):
            logging.info(
                f"Evaluation completed successfully, analyzed {len(codex_patches.seg_evaluation_metrics)} patches"
            )

            # Optionally save the results to a file
            results_path = os.path.join(config["output_dir"], "evaluation_results.pkl")
            with open(results_path, "wb") as f:
                pickle.dump(codex_patches.seg_evaluation_metrics, f)
            logging.info(f"Results saved to: {results_path}")
        else:
            logging.warning("No evaluation metrics were generated")

    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        import traceback

        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
