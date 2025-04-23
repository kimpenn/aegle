import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def visualize_density_distributions(
    density_metrics: Dict,
    output_dir: Optional[str] = None,
    save_plots: bool = True,
) -> None:
    """
    Visualize density distributions for cells and nuclei, comparing original and repaired masks.
    
    Creates multiple plots:
    1. Histogram comparisons for different mask combinations
    2. Density plot comparisons for different mask combinations
    3. Individual histograms with mean/median
    4. Individual density plots with mean/median
    5. Combined density plot of all distributions
    
    Args:
        density_metrics: Dictionary containing density metrics for different mask types
        output_dir: Directory to save plots (if None, plots will be displayed)
        save_plots: Whether to save the plots to files
    """
    logger.debug("Starting density distribution visualization")
    window_size_microns = density_metrics["window_size_microns"]
    
    # Extract density lists from metrics
    density_list_cell = density_metrics["local_density_list"]["cell"]
    density_list_nucleus = density_metrics["local_density_list"]["nucleus"]
    density_list_repaired_cell = density_metrics["local_density_list"]["repaired_cell"]
    density_list_repaired_nucleus = density_metrics["local_density_list"]["repaired_nucleus"]
    density_list_unmatched_nucleus = density_metrics["local_density_list"]["unmatched_nucleus"]

    logger.debug(f"Extracted density lists with shapes: cell={len(density_list_cell)}, nucleus={len(density_list_nucleus)}, "
                f"repaired_cell={len(density_list_repaired_cell)}, repaired_nucleus={len(density_list_repaired_nucleus)}, "
                f"unmatched_nucleus={len(density_list_unmatched_nucleus)}")

    # Data and labels
    data_lists = [
        density_list_cell,
        density_list_nucleus, 
        density_list_repaired_cell, 
        density_list_repaired_nucleus,
        density_list_unmatched_nucleus
    ]
    labels = ['Cell', 'Nucleus', 'Repaired Cell', 'Repaired Nucleus', 'Unmatched Nucleus']
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Create output directory if needed
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")

    # 1. Create comparison histograms
    logger.debug("Creating histogram comparison plots")
    
    # Pairs to compare in plots
    plot_comparisons = [
        # Original vs original
        (density_list_cell, density_list_nucleus, 'Cell', 'Nucleus', 'blue', 'green'),
        # Repaired vs repaired
        (density_list_repaired_cell, density_list_repaired_nucleus, 'Repaired Cell', 'Repaired Nucleus', 'red', 'purple'),
        # Cell vs repaired cell
        (density_list_cell, density_list_repaired_cell, 'Cell', 'Repaired Cell', 'blue', 'red'),
        # Nucleus vs repaired nucleus
        (density_list_nucleus, density_list_repaired_nucleus, 'Nucleus', 'Repaired Nucleus', 'green', 'purple'),
        # Nucleus vs unmatched nucleus
        (density_list_nucleus, density_list_unmatched_nucleus, 'Nucleus', 'Unmatched Nucleus', 'green', 'orange'),
        # Repaired nucleus vs unmatched nucleus
        (density_list_repaired_nucleus, density_list_unmatched_nucleus, 'Repaired Nucleus', 'Unmatched Nucleus', 'purple', 'orange')
    ]
    
    # Create histograms in a 3x2 grid
    fig1, axs1 = plt.subplots(3, 2, figsize=(16, 15))
    fig1.suptitle('Comparison Histograms of Mask Densities', fontsize=16)
    
    for i, (data1, data2, label1, label2, color1, color2) in enumerate(plot_comparisons):
        row, col = i // 2, i % 2
        sns.histplot(data1, ax=axs1[row, col], color=color1, alpha=0.5, label=label1)
        sns.histplot(data2, ax=axs1[row, col], color=color2, alpha=0.5, label=label2)
        axs1[row, col].set_title(f'{label1} vs {label2}')
        axs1[row, col].set_xlabel(f'# of cells per ({window_size_microns} µm)^2 window')
        axs1[row, col].set_ylabel('Frequency')
        axs1[row, col].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_plots and output_dir:
        output_path = os.path.join(output_dir, 'density_comparison_histograms.png')
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved histogram comparison plot to: {output_path}")
        plt.close()
    else:
        plt.show()

    # 2. Create comparison density plots
    logger.debug("Creating density plot comparisons")
    
    # Create density plots in a 3x2 grid
    fig2, axs2 = plt.subplots(3, 2, figsize=(16, 15))
    fig2.suptitle('Comparison Density Plots of Mask Densities', fontsize=16)
    
    for i, (data1, data2, label1, label2, color1, color2) in enumerate(plot_comparisons):
        row, col = i // 2, i % 2
        sns.kdeplot(data1, ax=axs2[row, col], color=color1, fill=True, alpha=0.4, label=label1)
        sns.kdeplot(data2, ax=axs2[row, col], color=color2, fill=True, alpha=0.4, label=label2)
        axs2[row, col].set_title(f'{label1} vs {label2}')
        axs2[row, col].set_xlabel(f'# of cells per ({window_size_microns} µm)^2 window')
        axs2[row, col].set_ylabel('Probability Density')
        axs2[row, col].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_plots and output_dir:
        output_path = os.path.join(output_dir, 'density_comparison_kde.png')
        plt.savefig(output_path, dpi=300)
        logger.debug(f"Saved density plot comparison to: {output_path}")
        plt.close()
    else:
        plt.show()

    # 3. Create individual histograms with mean and median
    logger.debug("Creating individual histograms with mean/median")
    num_plots = len(data_lists)
    rows = (num_plots + 1) // 2  # Round up division
    fig3, axs3 = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    fig3.suptitle('Individual Histograms with Mean and Median', fontsize=16)
    
    # Handle single row case
    if rows == 1:
        axs3 = [axs3]
        
    # Create plot positions
    plot_positions = [(i // 2, i % 2) for i in range(num_plots)]
    
    # If odd number, hide the last unused plot
    if num_plots % 2 == 1:
        if rows > 1:
            axs3[-1][-1].axis('off')
    
    # Plot histograms with mean and median markers
    for (data, label, color, pos) in zip(data_lists, labels, colors, plot_positions):
        row, col = pos
        sns.histplot(data, kde=False, ax=axs3[row][col], color=color, alpha=0.7)
        axs3[row][col].set_title(f'Histogram of {label}')
        axs3[row][col].set_xlabel(f'# of cells per ({window_size_microns} µm)^2 window')
        axs3[row][col].set_ylabel('Frequency')
        
        # Add mean and median lines
        mean_val = np.mean(data)
        median_val = np.median(data)
        axs3[row][col].axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        axs3[row][col].axvline(median_val, color='black', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
        axs3[row][col].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_plots and output_dir:
        output_path = os.path.join(output_dir, 'individual_histograms.png')
        plt.savefig(output_path, dpi=300)
        logger.debug(f"Saved individual histograms to: {output_path}")
        plt.close()
    else:
        plt.show()

    # 4. Create individual density plots with mean and median
    logger.debug("Creating individual density plots with mean/median")
    fig4, axs4 = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    fig4.suptitle('Individual Density Plots with Mean and Median', fontsize=16)

    # Handle single row case
    if rows == 1:
        axs4 = [axs4]
        
    # If odd number, hide the last unused plot
    if num_plots % 2 == 1:
        if rows > 1:
            axs4[-1][-1].axis('off')
    
    # Plot density plots with mean and median markers
    for (data, label, color, pos) in zip(data_lists, labels, colors, plot_positions):
        row, col = pos
        sns.kdeplot(data, ax=axs4[row][col], color=color, fill=True, alpha=0.5)
        axs4[row][col].set_title(f'Density Plot of {label}')
        axs4[row][col].set_xlabel(f'# of cells per ({window_size_microns} µm)^2 window')
        axs4[row][col].set_ylabel('Probability Density')
        
        # Add mean and median lines
        mean_val = np.mean(data)
        median_val = np.median(data)
        axs4[row][col].axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        axs4[row][col].axvline(median_val, color='black', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
        axs4[row][col].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_plots and output_dir:
        output_path = os.path.join(output_dir, 'individual_density_plots.png')
        plt.savefig(output_path, dpi=300)
        logger.debug(f"Saved individual density plots to: {output_path}")
        plt.close()
    else:
        plt.show()

    # 5. Create combined density plot
    logger.debug("Creating combined density plot")
    plt.figure(figsize=(12, 7))

    # Create distinct line styles and patterns for better visual distinction
    linestyles = ['-', '--', '-.', ':', '-']
    alphas = [0.4, 0.3, 0.3, 0.3, 0.25]

    # Plot all distributions with different styles
    for i, (data, label, color, ls, alpha) in enumerate(zip(data_lists, labels, colors, linestyles, alphas)):
        sns.kdeplot(data, color=color, label=label, linestyle=ls, linewidth=2)
        # Add a filled version with low alpha for additional visual distinction
        if i % 2 == 0:  # Add hatching to even-indexed plots
            sns.kdeplot(data, color=color, alpha=alpha, fill=True)
        else:
            sns.kdeplot(data, color=color, alpha=alpha, fill=True)

    plt.title('Comparison of All Density Distributions', fontsize=16)
    plt.xlabel(f'# of cells per ({window_size_microns} µm)^2 window', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add statistical summary as a text box
    stats_text = "Statistical Summary:\n"
    for label, data in zip(labels, data_lists):
        stats_text += f"\n{label}:\n"
        stats_text += f"Mean: {np.mean(data):.2f}, Median: {np.median(data):.2f}\n"
        stats_text += f"Std Dev: {np.std(data):.2f}, Range: [{np.min(data):.2f}, {np.max(data):.2f}]"

    plt.tight_layout()

    if save_plots and output_dir:
        output_path = os.path.join(output_dir, f'combined_density_plot_{window_size_microns}um.png')
        plt.savefig(output_path, dpi=300)
        logger.debug(f"Saved combined density plot to: {output_path}")
        plt.close()
    else:
        plt.show()
    
    logger.debug("Completed density distribution visualization") 