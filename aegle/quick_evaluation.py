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

src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
sys.path.append(src_path)
from aegle.codex_patches import CodexPatches
from aegle.evaluation_func import *


def run_quick_evaluation(
    codex_patches: CodexPatches,
    config,
    args,
):
    """
    Run a customized evaluation on CODEX patches to assess segmentation repair quality.

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

    # Filter for informative patches
    informative_idx = patches_metadata_df["is_infomative"] == True
    logging.info(f"Number of informative patches: {informative_idx.sum()}")
    image_ndarray = codex_patches.all_channel_patches[informative_idx]
    logging.info(f"image_ndarray.shape: {image_ndarray.shape}")

    output_dir = config.get("output_dir", "./output")

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
        # Check the gain of intensity from repaired segmentation by barplot
        # ----------
        # Analyze repair bias across channels
        bias_results = analyze_repair_bias_across_channels(
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
        density_results = create_comparison_density_plots(
            image_ndarray[idx],
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
            output_dir=os.path.join(patch_output_dir, "density_shits"),
            channels_to_plot=None,
            use_log_scale=True,
            channel_names=None,
        )
        # ----------
        # Average local cell density in 100x100 micrometer^2 window
        # ----------
        # TODO: Implement local cell density calculation

        # Store results
        evaluation_result = {
            "patch_idx": idx,
            "bias_analysis": bias_results,
            "density_analysis": density_results,
            # Add other metrics as needed
        }
        res_list.append(evaluation_result)

    codex_patches.seg_evaluation_metrics = res_list


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
    channels_to_plot=None,
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
        channels_to_plot: List of specific channel indices to plot (if None, will plot all)
        use_log_scale: Whether to use log scaling for x-axis
        channel_names: Dictionary mapping channel indices to names (optional)

    Returns:
        Dictionary with intensity data for each channel
    """

    start_time = time.time()

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get number of channels
    num_channels = image_array.shape[2]

    # If no channel names provided, use defaults
    if channel_names is None:
        channel_names = {i: f"Channel {i}" for i in range(num_channels)}

    # Determine which channels to process
    if channels_to_plot is None:
        channels_to_plot = list(range(num_channels))
    else:
        # Make sure all indices are valid
        channels_to_plot = [c for c in channels_to_plot if 0 <= c < num_channels]

    logging.info(f"Processing {len(channels_to_plot)} of {num_channels} channels")

    # Get unique IDs for nuclei
    all_nucleus_ids = np.unique(nucleus_mask)
    all_nucleus_ids = all_nucleus_ids[all_nucleus_ids > 0]

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

    logging.info(
        f"Found {len(all_nucleus_ids)} total nuclei, {len(matched_nucleus_ids)} matched, {len(unmatched_nucleus_ids)} unmatched"
    )
    logging.info(
        f"Found {len(all_cell_ids)} total cells, {len(matched_cell_ids)} matched, {len(unmatched_cell_ids)} unmatched"
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

        channel_img = image_array[:, :, c]

        # Process nuclei
        if len(matched_nucleus_ids) > 0:
            matched_means = ndimage.mean(
                channel_img, labels=nucleus_mask, index=matched_nucleus_ids
            )
            channel_data[c]["matched_nuclei"] = matched_means

        if len(unmatched_nucleus_ids) > 0:
            unmatched_means = ndimage.mean(
                channel_img, labels=nucleus_mask, index=unmatched_nucleus_ids
            )
            channel_data[c]["unmatched_nuclei"] = unmatched_means

        # Process cells
        if len(matched_cell_ids) > 0:
            matched_cell_means = ndimage.mean(
                channel_img, labels=cell_mask, index=matched_cell_ids
            )
            channel_data[c]["matched_cells"] = matched_cell_means

        if len(unmatched_cell_ids) > 0:
            unmatched_cell_means = ndimage.mean(
                channel_img, labels=cell_mask, index=unmatched_cell_ids
            )
            channel_data[c]["unmatched_cells"] = unmatched_cell_means

        # All nuclei (combine the two lists)
        channel_data[c]["all_nuclei"] = (
            np.concatenate(
                [channel_data[c]["matched_nuclei"], channel_data[c]["unmatched_nuclei"]]
            )
            if len(channel_data[c]["matched_nuclei"]) > 0
            and len(channel_data[c]["unmatched_nuclei"]) > 0
            else np.array([])
        )

        # For unselected regions (sample a subset of pixels)
        unselected_mask = nucleus_mask == 0
        if np.any(unselected_mask):
            unselected_indices = np.where(unselected_mask)
            sample_size = min(10000, len(unselected_indices[0]))
            sample_idx = np.random.choice(
                len(unselected_indices[0]), sample_size, replace=False
            )

            sampled_intensities = channel_img[
                unselected_indices[0][sample_idx], unselected_indices[1][sample_idx]
            ]
            channel_data[c]["unselected"] = sampled_intensities

        # Create and save/show plots (3 plots now)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot 1: Matched vs Unmatched nuclei
        if len(channel_data[c]["matched_nuclei"]) > 0:
            sns.kdeplot(
                channel_data[c]["matched_nuclei"],
                ax=ax1,
                label=f'Matched Nuclei (n={len(channel_data[c]["matched_nuclei"])})',
                fill=True,
                alpha=0.5,
            )

        if len(channel_data[c]["unmatched_nuclei"]) > 0:
            sns.kdeplot(
                channel_data[c]["unmatched_nuclei"],
                ax=ax1,
                label=f'Unmatched Nuclei (n={len(channel_data[c]["unmatched_nuclei"])})',
                fill=True,
                alpha=0.5,
            )

        ax1.set_title(f"{channel_name}: Matched vs Unmatched Nuclei")
        ax1.set_xlabel("Mean Intensity")
        ax1.set_ylabel("Density")
        ax1.legend()

        if use_log_scale:
            ax1.set_xscale("log")
            ax1.set_xlabel("Mean Intensity (log scale)")

        # Plot 2: Matched vs Unmatched cells
        if len(channel_data[c]["matched_cells"]) > 0:
            sns.kdeplot(
                channel_data[c]["matched_cells"],
                ax=ax2,
                label=f'Matched Cells (n={len(channel_data[c]["matched_cells"])})',
                fill=True,
                alpha=0.5,
            )

        if len(channel_data[c]["unmatched_cells"]) > 0:
            sns.kdeplot(
                channel_data[c]["unmatched_cells"],
                ax=ax2,
                label=f'Unmatched Cells (n={len(channel_data[c]["unmatched_cells"])})',
                fill=True,
                alpha=0.5,
            )

        ax2.set_title(f"{channel_name}: Matched vs Unmatched Cells")
        ax2.set_xlabel("Mean Intensity")
        ax2.set_ylabel("Density")
        ax2.legend()

        if use_log_scale:
            ax2.set_xscale("log")
            ax2.set_xlabel("Mean Intensity (log scale)")

        # Plot 3: All Nuclei vs Unselected
        if len(channel_data[c]["all_nuclei"]) > 0:
            sns.kdeplot(
                channel_data[c]["all_nuclei"],
                ax=ax3,
                label=f'All Nuclei (n={len(channel_data[c]["all_nuclei"])})',
                fill=True,
                alpha=0.5,
            )

        if len(channel_data[c]["unselected"]) > 0:
            sns.kdeplot(
                channel_data[c]["unselected"],
                ax=ax3,
                label=f'Unselected Regions (n={len(channel_data[c]["unselected"])})',
                fill=True,
                alpha=0.5,
            )

        ax3.set_title(f"{channel_name}: All Nuclei vs Unselected Regions")
        ax3.set_xlabel("Mean Intensity")
        ax3.set_ylabel("Density")
        ax3.legend()

        if use_log_scale:
            ax3.set_xscale("log")
            ax3.set_xlabel("Mean Intensity (log scale)")

        plt.tight_layout()

        # Handle potential log(0) issues by setting xlim
        if use_log_scale:
            for ax in [ax1, ax2, ax3]:
                try:
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
                except:
                    # Fallback if there's an issue
                    ax.set_xlim(left=0.1)

        if output_dir:
            # Use channel name in filename (replace spaces and slashes with underscores for safety)
            safe_name = str(channel_name).replace(" ", "_").replace("/", "_")
            plt.savefig(
                os.path.join(output_dir, f"ch{c}-{safe_name}_density_plots.png"),
                dpi=300,
            )
            plt.close()
        else:
            plt.show()

        logging.info(
            f"Channel {c} ({channel_name}) processed in {time.time() - channel_start:.2f}s"
        )

    logging.info(f"All density plots completed in {time.time() - start_time:.2f}s")
    return channel_data


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
