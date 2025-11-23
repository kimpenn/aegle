"""
Baseline CPU tests for clustering and dimensionality reduction.

These tests establish a baseline for the current CPU-only clustering functionality
and will be used to ensure GPU versions produce equivalent results.
"""

import numpy as np
import pytest

from aegle_analysis.analysis.clustering import run_clustering, create_anndata
from tests.utils.synthetic_analysis_data import get_small_dataset, get_medium_dataset


class TestClustering:
    """Test suite for clustering functions."""

    def test_run_clustering_basic(self):
        """Test that clustering produces expected outputs on small dataset."""
        # Get synthetic data
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering
        adata_clustered = run_clustering(
            adata, n_neighbors=10, resolution=0.2, random_state=42
        )

        # Validate outputs
        assert "leiden" in adata_clustered.obs.columns, "Leiden labels not added to obs"
        assert "X_umap" in adata_clustered.obsm, "UMAP embedding not computed"

        # Check UMAP dimensions (should be n_cells x 2)
        assert adata_clustered.obsm["X_umap"].shape == (
            adata_clustered.n_obs,
            2,
        ), "UMAP embedding has incorrect shape"

        # Check that cluster labels are strings
        assert adata_clustered.obs["leiden"].dtype == object, "Cluster labels should be strings"

        # Check that we have a reasonable number of clusters (not 1, not n_cells)
        n_clusters = adata_clustered.obs["leiden"].nunique()
        assert 2 <= n_clusters <= adata_clustered.n_obs // 10, (
            f"Unexpected number of clusters: {n_clusters}"
        )

    def test_run_clustering_reproducibility(self):
        """Test that clustering is reproducible with same random seed."""
        synth_data = get_small_dataset(random_seed=42)
        adata1 = synth_data.adata.copy()
        adata2 = synth_data.adata.copy()

        # Run clustering twice with same seed
        adata1 = run_clustering(adata1, random_state=42)
        adata2 = run_clustering(adata2, random_state=42)

        # Check that cluster assignments are identical
        assert np.array_equal(
            adata1.obs["leiden"].values, adata2.obs["leiden"].values
        ), "Cluster assignments not reproducible"

        # Check that UMAP embeddings are identical
        assert np.allclose(
            adata1.obsm["X_umap"], adata2.obsm["X_umap"]
        ), "UMAP embeddings not reproducible"

    def test_run_clustering_different_seeds(self):
        """Test that different random seeds produce different results."""
        synth_data = get_small_dataset(random_seed=42)
        adata1 = synth_data.adata.copy()
        adata2 = synth_data.adata.copy()

        # Run clustering with different seeds
        adata1 = run_clustering(adata1, random_state=42)
        adata2 = run_clustering(adata2, random_state=123)

        # Results should differ (at least UMAP coordinates)
        assert not np.allclose(
            adata1.obsm["X_umap"], adata2.obsm["X_umap"]
        ), "UMAP should differ with different random seeds"

    def test_run_clustering_resolution_parameter(self):
        """Test that resolution parameter affects number of clusters."""
        synth_data = get_small_dataset(random_seed=42)
        adata_low = synth_data.adata.copy()
        adata_high = synth_data.adata.copy()

        # Run clustering with different resolutions
        adata_low = run_clustering(adata_low, resolution=0.1, random_state=42)
        adata_high = run_clustering(adata_high, resolution=1.0, random_state=42)

        n_clusters_low = adata_low.obs["leiden"].nunique()
        n_clusters_high = adata_high.obs["leiden"].nunique()

        # Higher resolution should generally produce more clusters
        # (not guaranteed for all datasets, but should hold for synthetic data)
        assert n_clusters_high >= n_clusters_low, (
            f"Higher resolution should produce more clusters: "
            f"low={n_clusters_low}, high={n_clusters_high}"
        )

    def test_run_clustering_one_indexed(self):
        """Test that cluster labels are 1-indexed when one_indexed=True."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering with one_indexed=True (default)
        adata = run_clustering(adata, random_state=42, one_indexed=True)

        # Convert string labels back to integers to check
        cluster_labels = adata.obs["leiden"].astype(int).values
        min_label = cluster_labels.min()

        assert min_label >= 1, f"Cluster labels should be 1-indexed, got min={min_label}"

    def test_run_clustering_zero_indexed(self):
        """Test that cluster labels are 0-indexed when one_indexed=False."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering with one_indexed=False
        adata = run_clustering(adata, random_state=42, one_indexed=False)

        # Convert string labels back to integers to check
        cluster_labels = adata.obs["leiden"].astype(int).values
        min_label = cluster_labels.min()

        assert min_label == 0, f"Cluster labels should be 0-indexed, got min={min_label}"

    def test_run_clustering_medium_dataset(self):
        """Test clustering on medium dataset (integration test)."""
        synth_data = get_medium_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering
        adata = run_clustering(
            adata, n_neighbors=15, resolution=0.3, random_state=42
        )

        # Basic validation
        assert "leiden" in adata.obs.columns
        assert "X_umap" in adata.obsm

        # With 10K cells and 5 true clusters, we should detect several clusters
        n_clusters = adata.obs["leiden"].nunique()
        assert 3 <= n_clusters <= 20, (
            f"Expected 3-20 clusters for medium dataset, got {n_clusters}"
        )

    def test_run_clustering_n_neighbors_parameter(self):
        """Test that n_neighbors parameter is respected."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering with different n_neighbors
        # Both should succeed without error
        adata1 = run_clustering(adata.copy(), n_neighbors=5, random_state=42)
        adata2 = run_clustering(adata.copy(), n_neighbors=30, random_state=42)

        # Check that both produced valid results
        assert "leiden" in adata1.obs.columns
        assert "leiden" in adata2.obs.columns
        assert "X_umap" in adata1.obsm
        assert "X_umap" in adata2.obsm

    def test_create_anndata(self):
        """Test create_anndata function creates valid AnnData object."""
        import pandas as pd
        import anndata

        # Create simple test data
        data_df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0],
            "marker2": [4.0, 5.0, 6.0],
        })
        meta_df = pd.DataFrame({
            "cell_id": [1, 2, 3],
            "some_metadata": ["A", "B", "C"],
        })

        # Create AnnData
        adata = create_anndata(data_df, meta_df)

        # Validate structure
        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs == 3
        assert adata.n_vars == 2
        assert np.array_equal(adata.X, data_df.values)
        assert "cell_id" in adata.obs.columns
        assert "some_metadata" in adata.obs.columns

    def test_clustering_preserves_original_data(self):
        """Test that clustering doesn't modify original expression data."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Store original data
        original_X = adata.X.copy()

        # Run clustering
        adata = run_clustering(adata, random_state=42)

        # Check that X is unchanged
        assert np.array_equal(adata.X, original_X), (
            "Clustering should not modify expression matrix"
        )
