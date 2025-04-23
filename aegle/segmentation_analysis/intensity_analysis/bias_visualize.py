import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename by replacing unsafe characters with underscores.
    
    Args:
        filename: The string to sanitize
        
    Returns:
        A sanitized string safe for use as a filename
    """
    # Replace any character that's not alphanumeric, underscore, or hyphen with underscore
    return re.sub(r'[^\w\-]', '_', filename)

def visualize_channel_intensity_bias(
    intensity_data: Dict,
    output_dir: Optional[str] = None,
    channels_per_figure: int = 10,
    channel_names: Optional[List[str]] = None,
) -> None:
    """
    Visualize the intensity bias analysis results across channels.
    
    Args:
        intensity_data: Dictionary containing intensity analysis results from extract_intensity_data_across_channels
        output_dir: Directory to save plots (if None, plots will be displayed)
        channels_per_figure: Number of channels to plot per figure
        channel_names: Optional list of channel names to visualize. If None, all channels will be used.
    """
    logger.debug("Starting channel intensity bias visualization")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")
        
        # Create statistics subfolder
        stats_dir = os.path.join(output_dir, "channel_statistics")
        os.makedirs(stats_dir, exist_ok=True)
        logger.debug(f"Created statistics directory: {stats_dir}")

    # Get channel names
    if channel_names is None:
        channel_names = sorted(intensity_data.keys())
        logger.debug(f"Using all available channels: {len(channel_names)} channels")
    else:
        # Validate that all requested channels exist in results
        missing_channels = [ch for ch in channel_names if ch not in intensity_data]
        if missing_channels:
            logger.warning(f"Some requested channels not found in results: {missing_channels}")
            channel_names = [ch for ch in channel_names if ch in intensity_data]
            if not channel_names:
                logger.error("No valid channels found in results")
                return
        logger.debug(f"Using specified channels: {len(channel_names)} channels: {channel_names}")

    num_channels = len(channel_names)
    if num_channels == 0:
        logger.error("No channels to visualize")
        return
        
    # Save statistics for all channels as CSV
    if output_dir:
        save_channel_statistics_as_csv(intensity_data, channel_names, output_dir)

    # Calculate number of figures needed
    num_figures = (num_channels + channels_per_figure - 1) // channels_per_figure
    logger.debug(f"Will create {num_figures} figures with {channels_per_figure} channels per figure")

    for fig_idx in range(num_figures):
        start_idx = fig_idx * channels_per_figure
        end_idx = min((fig_idx + 1) * channels_per_figure, num_channels)
        current_channels = channel_names[start_idx:end_idx]
        logger.debug(f"Creating figure {fig_idx + 1}/{num_figures} for channels {start_idx + 1}-{end_idx}")
        
        # Create figure with only 2 subfigures (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Channel Intensity Bias Analysis (Channels {start_idx+1}-{end_idx})', fontsize=16)

        # Extract data for current channels
        cell_bias = [
            intensity_data[ch]["bias_metrics"]["cell"]["percent_change"]
            for ch in current_channels
        ]
        nucleus_bias = [
            intensity_data[ch]["bias_metrics"]["nucleus"]["percent_change"]
            for ch in current_channels
        ]

        # Plot cell bias
        logger.debug(f"Plotting cell bias for channels {current_channels}")
        sns.barplot(x=current_channels, y=cell_bias, ax=axs[0])
        axs[0].set_title('Cell Bias')
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')
        axs[0].set_ylabel('Bias Value (%)')

        # Plot nucleus bias
        logger.debug(f"Plotting nucleus bias for channels {current_channels}")
        sns.barplot(x=current_channels, y=nucleus_bias, ax=axs[1])
        axs[1].set_title('Nucleus Bias')
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
        axs[1].set_ylabel('Bias Value (%)')

        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, f'bias_analysis_channels_{start_idx+1}_{end_idx}.png')
            plt.savefig(output_path, dpi=300)
            logger.debug(f"Saved figure to {output_path}")
            plt.close()
        else:
            plt.show()
            
    # Plot individual statistics for each channel
    if output_dir:
        logger.debug("Creating individual statistics plots for each channel")
        plot_channel_statistics(intensity_data, channel_names, stats_dir)

    # Create summary plot for all channels
    logger.debug("Creating summary plot for all channels")
    plt.figure(figsize=(16, 8))
    x = np.arange(num_channels)
    cell_bias = [
        intensity_data[ch]["bias_metrics"]["cell"]["percent_change"]
        for ch in channel_names
    ]
    nucleus_bias = [
        intensity_data[ch]["bias_metrics"]["nucleus"]["percent_change"]
        for ch in channel_names
    ]

    plt.bar(x - 0.2, cell_bias, 0.4, label="Cell Bias")
    plt.bar(x + 0.2, nucleus_bias, 0.4, label="Nucleus Bias")
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Channel")
    plt.ylabel("Intensity % Change (Repaired vs Original)")
    plt.title("Repair Bias Summary Across All Channels")
    plt.xticks(x, channel_names, rotation=90)
    plt.legend()
    plt.tight_layout()

    if output_dir:
        output_path = os.path.join(output_dir, "all_channels_bias_summary.png")
        plt.savefig(output_path, dpi=300)
        logger.debug(f"Saved summary plot to {output_path}")
        plt.close()
    else:
        plt.show()

    # Report significant bias
    threshold = 5.0
    significant_bias = []
    for ch in channel_names:
        cell_change = intensity_data[ch]["bias_metrics"]["cell"]["percent_change"]
        nuc_change = intensity_data[ch]["bias_metrics"]["nucleus"]["percent_change"]

        if abs(cell_change) > threshold or abs(nuc_change) > threshold:
            significant_bias.append((ch, cell_change, nuc_change))

    if output_dir:
        summary_path = os.path.join(output_dir, "significant_bias_summary.txt")
        with open(summary_path, "w") as f:
            if significant_bias:
                f.write(f"Channels with significant bias (>{threshold}% change):\n")
                for ch, cell_ch, nuc_ch in significant_bias:
                    f.write(f"{ch}: Cell: {cell_ch:.2f}%, Nucleus: {nuc_ch:.2f}%\n")
            else:
                f.write(f"No channels show significant bias (>{threshold}% change)\n")
        logger.debug(f"Saved significant bias summary to {summary_path}")

    if significant_bias:
        logger.debug(f"Channels with significant bias (>{threshold}% change):")
        for ch, cell_ch, nuc_ch in significant_bias:
            logger.debug(f"{ch}: Cell: {cell_ch:.2f}%, Nucleus: {nuc_ch:.2f}%")
    else:
        logger.debug(f"No channels show significant bias (>{threshold}% change)")
    
    logger.debug("Channel intensity bias visualization completed")
    
def plot_channel_statistics(intensity_data, channel_names, output_dir):
    """
    Create individual statistics plots for each channel and save to the output directory.
    
    Args:
        intensity_data: Dictionary containing intensity analysis results
        channel_names: List of channel names to visualize
        output_dir: Directory to save plots
    """
    metrics = ["mean", "median", "std"]
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, channel in enumerate(channel_names):
        logger.debug(f"Creating statistics plot for channel {channel} ({i+1}/{len(channel_names)})")
        
        # Check if unmatched nuclei statistics are available
        has_unmatched_nuclei = "nucleus_unmatched" in intensity_data[channel]["statistics"]
        
        # Always create a figure with 2 subplots (cell and nucleus)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Statistics for Channel: {channel}', fontsize=16)
        
        # Get cell statistics
        orig_cell = intensity_data[channel]["statistics"]["cell"]
        rep_cell = intensity_data[channel]["statistics"]["cell_matched"]
        
        # Plot cell statistics
        axs[0].bar(x - 0.2, [orig_cell[m] for m in metrics], width, label='Original', color='blue', alpha=0.7)
        axs[0].bar(x + 0.2, [rep_cell[m] for m in metrics], width, label='Repaired', color='lightblue', alpha=0.7)
        axs[0].set_title(f'Cell Statistics')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(metrics)
        axs[0].legend()
        
        # Get nucleus statistics
        orig_nuc = intensity_data[channel]["statistics"]["nucleus"]
        rep_nuc = intensity_data[channel]["statistics"]["nucleus_matched"]
        
        # Adjust bar positions based on whether unmatched nuclei data is available
        if has_unmatched_nuclei:
            # Plot nucleus statistics with 3 groups
            axs[1].bar(x - width, [orig_nuc[m] for m in metrics], width, label='Original', color='red', alpha=0.7)
            axs[1].bar(x, [rep_nuc[m] for m in metrics], width, label='Repaired', color='salmon', alpha=0.7)
            
            # Add unmatched nuclei to the same subplot
            unmatched_nuc = intensity_data[channel]["statistics"]["nucleus_unmatched"]
            axs[1].bar(x + width, [unmatched_nuc[m] for m in metrics], width, label='Unmatched', color='purple', alpha=0.7)
        else:
            # Plot nucleus statistics with 2 groups
            axs[1].bar(x - 0.2, [orig_nuc[m] for m in metrics], width, label='Original', color='red', alpha=0.7)
            axs[1].bar(x + 0.2, [rep_nuc[m] for m in metrics], width, label='Repaired', color='salmon', alpha=0.7)
            
        axs[1].set_title(f'Nucleus Statistics')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(metrics)
        axs[1].legend()
        
        plt.tight_layout()
        
        # Properly sanitize the filename
        safe_channel_name = sanitize_filename(channel)
        output_path = os.path.join(output_dir, f'statistics_{i+1:03d}_{safe_channel_name}.png')
        plt.savefig(output_path, dpi=300)
        logger.debug(f"Saved statistics plot to {output_path}")
        plt.close()

def save_channel_statistics_as_csv(intensity_data, channel_names, output_dir):
    """
    Save all channel statistics to a CSV file.
    
    Args:
        intensity_data: Dictionary containing intensity analysis results
        channel_names: List of channel names to include
        output_dir: Directory to save the CSV file
    """
    logger.debug("Saving channel statistics to CSV")
    
    # Create a DataFrame to store the statistics
    stats_data = []
    
    for channel in channel_names:
        # Cell statistics
        cell_stats = intensity_data[channel]["statistics"]["cell"]
        cell_matched_stats = intensity_data[channel]["statistics"]["cell_matched"]
        cell_percent_change = intensity_data[channel]["bias_metrics"]["cell"]["percent_change"]
        
        # Nucleus statistics
        nuc_stats = intensity_data[channel]["statistics"]["nucleus"]
        nuc_matched_stats = intensity_data[channel]["statistics"]["nucleus_matched"]
        nuc_percent_change = intensity_data[channel]["bias_metrics"]["nucleus"]["percent_change"]
        
        # Check if unmatched nuclei statistics are available
        has_unmatched_nuclei = "nucleus_unmatched" in intensity_data[channel]["statistics"]
        
        # Create basic data dictionary
        channel_data = {
            "channel": channel,
            "cell_mean": cell_stats["mean"],
            "cell_median": cell_stats["median"],
            "cell_std": cell_stats["std"],
            "cell_percent_change": cell_percent_change,
            "nucleus_mean": nuc_stats["mean"],
            "nucleus_median": nuc_stats["median"],
            "nucleus_std": nuc_stats["std"],
            "nucleus_percent_change": nuc_percent_change
        }
        
        # Add unmatched nuclei statistics if available
        if has_unmatched_nuclei:
            nuc_unmatched_stats = intensity_data[channel]["statistics"]["nucleus_unmatched"]
            channel_data.update({
                "nucleus_unmatched_mean": nuc_unmatched_stats["mean"],
                "nucleus_unmatched_median": nuc_unmatched_stats["median"],
                "nucleus_unmatched_std": nuc_unmatched_stats["std"]
            })
        
        # Add to data
        stats_data.append(channel_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats_data)
    csv_path = os.path.join(output_dir, "channel_statistics.csv")
    df.to_csv(csv_path, index=False)
    logger.debug(f"Saved channel statistics to {csv_path}") 