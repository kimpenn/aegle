"""
Marker distribution visualization functions for CODEX data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_marker_distributions(
    df: pd.DataFrame,
    log_transform: bool = False,
    markers_per_plot: int = 11,
    output_dir: Optional[str] = None,
    plot_type: str = "both",
) -> None:
    """
    Create violin and box plots for all markers, grouping them into subplots.

    Args:
        df: DataFrame containing marker expression data (columns are markers)
        log_transform: Whether to apply log1p transformation
        markers_per_plot: Number of markers to show per subplot
        output_dir: Directory to save plots (if None, plots are not saved)
        plot_type: Type of plots to create ("violin", "box", or "both")
    """
    # Ensure all values are numeric and drop problematic rows
    df_numeric = df.apply(pd.to_numeric, errors="coerce").dropna()

    # Apply log transformation if requested
    if log_transform:
        df_numeric = np.log1p(df_numeric)
        transform_text = "Log-transformed"
    else:
        transform_text = "Raw"

    num_markers = len(df_numeric.columns)
    num_subplots = int(np.ceil(num_markers / markers_per_plot))

    # Create violin plots
    if plot_type in ["violin", "both"]:
        fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 5 * num_subplots))
        axes = np.array(axes).flatten() if num_subplots > 1 else np.array([axes])

        for i, ax in enumerate(axes):
            start_idx = i * markers_per_plot
            end_idx = min(start_idx + markers_per_plot, num_markers)
            selected_markers = df_numeric.columns[start_idx:end_idx]

            if len(selected_markers) == 0:
                continue  # Skip empty plots

            # Violin plot for selected markers
            sns.violinplot(
                data=df_numeric[selected_markers],
                density_norm="width",
                inner="quartile",
                ax=ax,
            )
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{transform_text} Marker Expression - Violin Plot {i+1}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, f"{transform_text.lower()}_violin_plots.png"),
                dpi=300,
            )

        plt.show()

    # Create box plots
    if plot_type in ["box", "both"]:
        fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 5 * num_subplots))
        axes = np.array(axes).flatten() if num_subplots > 1 else np.array([axes])

        for i, ax in enumerate(axes):
            start_idx = i * markers_per_plot
            end_idx = min(start_idx + markers_per_plot, num_markers)
            selected_markers = df_numeric.columns[start_idx:end_idx]

            if len(selected_markers) == 0:
                continue  # Skip empty plots

            # Box plot for selected markers
            sns.boxplot(data=df_numeric[selected_markers], ax=ax)
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{transform_text} Marker Expression - Box Plot {i+1}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, f"{transform_text.lower()}_box_plots.png"),
                dpi=300,
            )

        plt.show()


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str = "Cluster Heatmap",
    center: float = 0,
    cmap: str = "RdBu_r",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
) -> None:
    """
    Create a clustered heatmap from a data matrix.

    Args:
        matrix: DataFrame with rows and columns to display as a heatmap
        title: Title for the heatmap
        center: Center value for the colormap
        cmap: Colormap name
        figsize: Figure size as (width, height)
        output_path: Path to save the figure (if None, figure is not saved)
    """
    # Set up figure
    sns.set(font_scale=1.0)

    # Create clustermap
    g = sns.clustermap(
        matrix.T,
        cmap=cmap,
        vmin=-1, vmax=1, # for log2 fold change
        center=center,
        robust=True,
        figsize=figsize,
        cbar_kws={"label": title.split(":")[-1].strip() if ":" in title else "Value"},
        method="average",  # for hierarchical clustering
        metric="euclidean",
    )

    g.fig.suptitle(title)

    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_marker_correlations(
    df: pd.DataFrame,
    method: str = "spearman",
    title: str = "Marker Correlation Heatmap",
    output_path: Optional[str] = None,
) -> None:
    """
    Create a correlation heatmap for markers.

    Args:
        df: DataFrame with marker expression data
        method: Correlation method ("pearson", "spearman", or "kendall")
        title: Title for the heatmap
        output_path: Path to save the figure (if None, figure is not saved)
    """
    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)

    # Create heatmap
    plt.figure(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
    )

    plt.title(title)
    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
