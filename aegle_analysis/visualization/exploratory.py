"""
Exploratory visualization functions for CODEX data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


def plot_cell_metrics(
    meta_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Generate exploratory plots for cell metrics (area, DAPI, entropy).

    Args:
        meta_df: DataFrame containing cell metadata
        output_dir: Directory to save plots (if None, plots are not saved)
        metrics: List of metrics to plot. If None, uses ["area", "DAPI", "cell_entropy"] if available
    """
    if metrics is None:
        metrics = []
        for metric in ["area", "DAPI", "cell_entropy"]:
            if metric in meta_df.columns:
                metrics.append(metric)

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Plot relationships between metrics
    for i, metric1 in enumerate(metrics):
        for metric2 in metrics[i + 1 :]:
            plot_metric_relationship(
                meta_df,
                metric1,
                metric2,
                output_path=(
                    os.path.join(output_dir, f"{metric1}_vs_{metric2}_scatter.png")
                    if output_dir
                    else None
                ),
            )

    # Plot distributions for each metric
    for metric in metrics:
        plot_metric_distribution(
            meta_df,
            metric,
            output_path=(
                os.path.join(output_dir, f"{metric}_distribution.png")
                if output_dir
                else None
            ),
        )


def plot_metric_relationship(
    meta_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Create a scatter plot showing the relationship between two cell metrics.

    Args:
        meta_df: DataFrame containing cell metadata
        x_metric: Name of the column to plot on x-axis
        y_metric: Name of the column to plot on y-axis
        output_path: Path to save the figure (if None, figure is not saved)
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(meta_df[x_metric], meta_df[y_metric], alpha=0.5, s=1)
    plt.xlabel(f"{x_metric.replace('_', ' ').title()}")
    plt.ylabel(f"{y_metric.replace('_', ' ').title()}")
    plt.title(
        f"Scatter Plot: {y_metric.replace('_', ' ').title()} vs. {x_metric.replace('_', ' ').title()}"
    )
    plt.grid(True, linestyle="--", alpha=0.7)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()


def plot_metric_distribution(
    meta_df: pd.DataFrame,
    metric: str,
    bins: int = 50,
    output_path: Optional[str] = None,
) -> None:
    """
    Create a histogram and boxplot showing the distribution of a cell metric.

    Args:
        meta_df: DataFrame containing cell metadata
        metric: Name of the column to plot
        bins: Number of bins for histogram
        output_path: Path to save the figure (if None, figure is not saved)
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Histogram
    axes[0].hist(meta_df[metric], bins=bins, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel(f"{metric.replace('_', ' ').title()}")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Histogram: {metric.replace('_', ' ').title()}")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Boxplot
    axes[1].boxplot(meta_df[metric], vert=False)
    axes[1].set_xlabel(f"{metric.replace('_', ' ').title()}")
    axes[1].set_title(f"Boxplot: {metric.replace('_', ' ').title()}")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()


def plot_pca(
    data_df: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> None:
    """
    Perform PCA and plot the results.

    Args:
        data_df: DataFrame with cells as rows and features as columns
        n_components: Number of PCA components to compute
        random_state: Random seed for reproducibility
        output_path: Path to save the figure (if None, figure is not saved)
    """
    from sklearn.decomposition import PCA

    # Perform PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(data_df)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot first two components
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=3)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title("PCA of Cell Expression Data")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add total explained variance
    total_var = pca.explained_variance_ratio_[:n_components].sum()
    plt.annotate(
        f"Total explained variance: {total_var:.2%}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()

    # Plot scree plot if more than 2 components
    if n_components > 2:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot")
        plt.xticks(range(1, n_components + 1))
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        if output_path:
            output_path_scree = output_path.replace(".png", "_scree.png")
            plt.savefig(output_path_scree, dpi=300)

        plt.show()
