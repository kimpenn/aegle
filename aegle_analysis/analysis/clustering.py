"""
Clustering and dimensionality reduction functions for CODEX data.

Supports GPU-accelerated k-NN graph construction via FAISS-GPU for
significant speedup (10-50x) on large datasets.
"""

import logging
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


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
    use_gpu: bool = False,
    gpu_id: int = 0,
    fallback_to_cpu: bool = True,
) -> anndata.AnnData:
    """
    Run standard clustering pipeline with neighbors, leiden, and UMAP.

    Supports GPU-accelerated k-NN graph construction via FAISS-GPU for
    10-50x speedup on large datasets (>10K cells).

    Args:
        adata: AnnData object with expression data
        n_neighbors: Number of neighbors for neighborhood graph
        resolution: Resolution parameter for leiden clustering
        random_state: Random seed for reproducibility
        use_rep: Representation to use for computing neighbors
        one_indexed: Whether to use 1-indexed cluster labels (True) or 0-indexed (False)
        use_gpu: Whether to use GPU acceleration for k-NN graph construction.
                 Default is False for backward compatibility.
        gpu_id: GPU device ID to use (default: 0)
        fallback_to_cpu: If GPU fails, fall back to CPU (default: True)

    Returns:
        AnnData object with computed neighbors, leiden clusters, and UMAP coordinates
    """
    n_cells = adata.n_obs
    total_start = time.time()

    # Compute neighbors (GPU or CPU)
    neighbors_start = time.time()

    if use_gpu:
        gpu_success = False
        try:
            from .clustering_gpu import is_faiss_gpu_available, compute_neighbors_gpu

            if is_faiss_gpu_available():
                logger.info(f"Computing neighbors with FAISS-GPU (n={n_cells:,}, k={n_neighbors})")
                compute_neighbors_gpu(
                    adata,
                    n_neighbors=n_neighbors,
                    use_rep=use_rep,
                    metric="euclidean",
                    gpu_id=gpu_id,
                    random_state=random_state,
                )
                gpu_success = True
                neighbors_time = time.time() - neighbors_start
                logger.info(f"GPU neighbors: {neighbors_time:.2f}s")
            else:
                logger.warning("FAISS-GPU not available")

        except Exception as e:
            logger.warning(f"GPU neighbors failed: {e}")
            if not fallback_to_cpu:
                raise

        if not gpu_success:
            if fallback_to_cpu:
                logger.info("Falling back to CPU for neighbors computation")
                sc.pp.neighbors(
                    adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state
                )
                neighbors_time = time.time() - neighbors_start
                logger.info(f"CPU neighbors: {neighbors_time:.2f}s")
            else:
                raise RuntimeError("GPU neighbors failed and fallback_to_cpu=False")
    else:
        # CPU path (original behavior)
        sc.pp.neighbors(
            adata, n_neighbors=n_neighbors, use_rep=use_rep, random_state=random_state
        )
        neighbors_time = time.time() - neighbors_start
        if n_cells > 10000:
            logger.info(f"CPU neighbors: {neighbors_time:.2f}s (consider use_gpu=True for speedup)")

    # Run leiden clustering (always CPU - fast on precomputed graph)
    leiden_start = time.time()
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    leiden_time = time.time() - leiden_start

    # Optionally shift cluster labels from "0,1,2..." to "1,2,3..."
    if one_indexed:
        adata.obs["leiden"] = adata.obs["leiden"].astype(int) + 1

    # Convert cluster labels to strings
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)

    # Compute UMAP embeddings (uses precomputed neighbors)
    umap_start = time.time()
    sc.tl.umap(adata, random_state=random_state)
    umap_time = time.time() - umap_start

    total_time = time.time() - total_start

    # Log performance summary for large datasets
    if n_cells > 5000:
        logger.info(
            f"Clustering complete ({n_cells:,} cells): "
            f"neighbors={neighbors_time:.1f}s, leiden={leiden_time:.1f}s, "
            f"umap={umap_time:.1f}s, total={total_time:.1f}s"
        )

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
