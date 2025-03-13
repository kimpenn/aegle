"""
Clustering and dimensionality reduction functions for CODEX data.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional


def create_anndata(
    data_df: pd.DataFrame, meta_df: pd.DataFrame, var_df: Optional[pd.DataFrame] = None
) -> anndata.AnnData:
    """
    Create an AnnData object from data and metadata.

    Args:
        data_df: Data matrix with cells as rows and features as columns
        meta_df: Metadata DataFrame with cells as rows
        var_df: Optional DataFrame with variable annotations

    Returns:
        AnnData object
    """
    if var_df is None:
        var_df = pd.DataFrame(index=data_df.columns)

    adata = anndata.AnnData(X=data_df.values, obs=meta_df, var=var_df)
    return adata


def run_clustering(
    adata: anndata.AnnData,
    n_neighbors: int = 10,
    resolution: float = 0.2,
    random_state: int = 42,
    use_rep: str = "X",
    one_indexed: bool = True,
) -> anndata.AnnData:
    """
    Run standard clustering pipeline with neighbors, leiden, and UMAP.

    Args:
        adata: AnnData object with expression data
        n_neighbors: Number of neighbors for neighborhood graph
        resolution: Resolution parameter for leiden clustering
        random_state: Random seed for reproducibility
        use_rep: Representation to use for computing neighbors
        one_indexed: Whether to use 1-indexed cluster labels (True) or 0-indexed (False)

    Returns:
        AnnData object with computed neighbors, leiden clusters, and UMAP coordinates
    """
    # Compute neighbors
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state
    )

    # Run leiden clustering
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)

    # Optionally shift cluster labels from "0,1,2..." to "1,2,3..."
    if one_indexed:
        adata.obs["leiden"] = adata.obs["leiden"].astype(int) + 1

    # Convert cluster labels to strings
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)

    # Compute UMAP embeddings
    sc.tl.umap(adata, random_state=random_state)

    return adata


def map_clusters_to_colors(
    cell_mask: np.ndarray, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    Map cluster labels to colors for visualization of segmentation masks.

    Args:
        cell_mask: Cell segmentation mask with integer cell IDs
        cluster_labels: Cluster labels for each cell ID

    Returns:
        Color-coded mask for visualization
    """
    import seaborn as sns

    # Get the maximum cell ID in the mask
    max_cell_id = int(cell_mask.max())

    # Ensure cluster_labels is a numpy array
    cluster_labels = np.array(cluster_labels)

    # Build a map from "cell ID" -> "cluster"
    id_to_cluster = np.zeros(max_cell_id + 1, dtype=int)

    # The naive approach assumes row i => cell ID i+1
    id_to_cluster[1 : len(cluster_labels) + 1] = cluster_labels

    # Create a color palette
    unique_labels = np.unique(cluster_labels)
    palette = np.array(sns.color_palette("tab20", len(unique_labels)))

    # For quick indexing, cluster i -> color palette[i-1] if cluster is 1-based
    cluster_to_color = np.zeros((max(unique_labels) + 1, 3))
    for i, lbl in enumerate(unique_labels):
        cluster_to_color[lbl] = palette[i]

    # Map each cell ID to its cluster color
    color_array = cluster_to_color[id_to_cluster]
    color_mask = color_array[cell_mask]

    return color_mask
