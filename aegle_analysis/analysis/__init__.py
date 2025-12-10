"""
Analysis modules for CODEX data.

Includes GPU-accelerated clustering via FAISS-GPU for k-NN graph construction.
"""

from .clustering import (
    create_anndata,
    run_clustering,
    map_clusters_to_colors,
)

from .clustering_gpu import (
    is_faiss_gpu_available,
    build_knn_graph_faiss_gpu,
    compute_neighbors_gpu,
)

from .differential import (
    one_vs_rest_wilcoxon,
    build_log_fold_change_matrix,
    build_fold_change_matrix,
    export_de_results,
)

from .entropy import (
    calculate_channel_entropy,
    add_entropy_to_metadata,
    calculate_entropy_bounds,
)

__all__ = [
    # Clustering (CPU and GPU)
    "create_anndata",
    "run_clustering",
    "map_clusters_to_colors",
    "is_faiss_gpu_available",
    "build_knn_graph_faiss_gpu",
    "compute_neighbors_gpu",
    # Differential expression
    "one_vs_rest_wilcoxon",
    "build_log_fold_change_matrix",
    "export_de_results",
    "build_fold_change_matrix",
    # Entropy
    "calculate_channel_entropy",
    "add_entropy_to_metadata",
    "calculate_entropy_bounds",
]
