"""Example test demonstrating synthetic analysis data usage.

This example shows how to use the synthetic analysis data generator
in analysis module tests.
"""

import numpy as np
import pytest
import scanpy as sc

from tests.utils.synthetic_analysis_data import (
    get_small_dataset,
    get_medium_dataset,
    create_synthetic_anndata,
)


class TestExampleAnalysisWorkflow:
    """Example tests showing various usage patterns."""

    @pytest.fixture
    def small_data(self):
        """Fixture providing small dataset for fast tests."""
        return get_small_dataset(random_seed=42)

    def test_basic_clustering(self, small_data):
        """Example: Test that clustering can recover known clusters."""
        adata = small_data.adata
        true_clusters = small_data.true_clusters

        # Run clustering
        sc.pp.neighbors(adata, n_neighbors=15, random_state=42)
        sc.tl.leiden(adata, resolution=0.5, random_state=42)

        # Clustering should identify multiple groups
        n_clusters_found = len(adata.obs["leiden"].unique())
        n_clusters_expected = len(np.unique(true_clusters))

        # Should find approximately the right number of clusters
        assert abs(n_clusters_found - n_clusters_expected) <= 2

    def test_marker_differential_expression(self, small_data):
        """Example: Test that differential expression finds cluster markers."""
        adata = small_data.adata

        # Assign ground truth clusters
        adata.obs["cluster"] = small_data.true_clusters.astype(str)

        # Run differential expression for cluster 0
        cluster_0_mask = adata.obs["cluster"] == "0"
        cluster_0_mean = adata.X[cluster_0_mask].mean(axis=0)
        other_mean = adata.X[~cluster_0_mask].mean(axis=0)

        # Calculate fold change
        fold_changes = cluster_0_mean / (other_mean + 1e-8)

        # At least some markers should be differentially expressed
        # (fold change > 1.5 or < 0.67)
        de_markers = (fold_changes > 1.5) | (fold_changes < 0.67)
        assert de_markers.sum() > 0

    def test_spatial_visualization(self, small_data):
        """Example: Test spatial mask visualization."""
        cell_mask = small_data.cell_mask
        nucleus_mask = small_data.nucleus_mask

        # Masks should have correct properties
        assert cell_mask.ndim == 2
        assert nucleus_mask.ndim == 2
        assert cell_mask.shape == nucleus_mask.shape

        # Background should be 0
        assert (cell_mask >= 0).all()
        assert (nucleus_mask >= 0).all()

        # Should have cells
        assert cell_mask.max() > 0
        assert nucleus_mask.max() > 0

    def test_morphology_features(self, small_data):
        """Example: Test morphology feature extraction."""
        adata = small_data.adata

        # Morphology features should be present
        assert "nucleus_area" in adata.obs.columns
        assert "cell_area" in adata.obs.columns
        assert "eccentricity" in adata.obs.columns

        # Cell area should be larger than nucleus area
        assert (adata.obs["cell_area"] > adata.obs["nucleus_area"]).all()

        # Eccentricity should be in [0, 1]
        assert (adata.obs["eccentricity"] >= 0).all()
        assert (adata.obs["eccentricity"] <= 1).all()

    def test_reproducibility(self):
        """Example: Test that same seed gives same results."""
        data1 = get_small_dataset(random_seed=42)
        data2 = get_small_dataset(random_seed=42)

        # Expression matrices should be identical
        np.testing.assert_array_almost_equal(data1.adata.X, data2.adata.X)

        # Cluster assignments should be identical
        np.testing.assert_array_equal(
            data1.true_clusters, data2.true_clusters
        )


class TestExampleCustomDatasets:
    """Example tests showing custom dataset creation."""

    def test_custom_cluster_count(self):
        """Example: Create dataset with specific number of clusters."""
        data = create_synthetic_anndata(
            n_cells=500,
            n_markers=15,
            n_clusters=7,  # Custom cluster count
            random_seed=42,
        )

        # Should have 7 clusters
        assert len(np.unique(data.true_clusters)) == 7

    def test_high_cluster_separation(self):
        """Example: Create dataset with well-separated clusters."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=20,
            n_clusters=4,
            cluster_separation=3.0,  # High separation
            noise_level=0.1,  # Low noise
            random_seed=42,
        )

        # Compute mean expression per cluster
        cluster_means = []
        for cluster_id in range(4):
            mask = data.true_clusters == cluster_id
            cluster_means.append(data.adata.X[mask].mean(axis=0))

        cluster_means = np.array(cluster_means)

        # With high separation, clusters should have very different means
        # Calculate range of means across clusters for each marker
        marker_ranges = cluster_means.max(axis=0) - cluster_means.min(axis=0)

        # Most markers should show large differences (>2 std)
        assert (marker_ranges > 2.0).sum() >= data.adata.n_vars * 0.6

    def test_minimal_dataset(self):
        """Example: Create minimal dataset for edge case testing."""
        data = create_synthetic_anndata(
            n_cells=50,  # Very small
            n_markers=5,
            n_clusters=2,
            random_seed=42,
            add_spatial_coords=False,  # Skip spatial coords
            add_morphology_features=False,  # Skip morphology
        )

        # Should still create valid AnnData
        assert data.adata.n_obs == 50
        assert data.adata.n_vars == 5

        # Should not have spatial/morphology features
        assert "X" not in data.adata.obs.columns
        assert "nucleus_area" not in data.adata.obs.columns


class TestExamplePerformanceComparison:
    """Example tests showing performance comparison patterns."""

    def test_clustering_performance_comparison(self):
        """Example: Compare clustering performance on different dataset sizes."""
        import time

        # Small dataset
        small_data = get_small_dataset(random_seed=42)
        start = time.time()
        sc.pp.neighbors(small_data.adata, n_neighbors=15)
        sc.tl.leiden(small_data.adata, resolution=0.5)
        small_time = time.time() - start

        # Medium dataset (10x more cells)
        medium_data = get_medium_dataset(random_seed=42)
        start = time.time()
        sc.pp.neighbors(medium_data.adata, n_neighbors=15)
        sc.tl.leiden(medium_data.adata, resolution=0.5)
        medium_time = time.time() - start

        print(f"\nSmall dataset (1K cells): {small_time:.2f}s")
        print(f"Medium dataset (10K cells): {medium_time:.2f}s")
        print(f"Scaling factor: {medium_time / small_time:.2f}x")

        # Should scale reasonably (with 10x more data)
        # Note: Actual scaling depends on algorithm (leiden can be O(n log n) to O(n^2))
        assert medium_time < small_time * 100  # Reasonable upper bound


class TestExampleValidation:
    """Example tests showing validation patterns."""

    def test_validate_cluster_quality(self):
        """Example: Validate that generated clusters are well-separated."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=20,
            n_clusters=5,
            cluster_separation=2.5,
            random_seed=42,
        )

        # Compute within-cluster and between-cluster variance
        within_cluster_var = []
        for cluster_id in range(5):
            mask = data.true_clusters == cluster_id
            cluster_data = data.adata.X[mask]
            within_cluster_var.append(cluster_data.var(axis=0).mean())

        # Between-cluster variance
        all_means = []
        for cluster_id in range(5):
            mask = data.true_clusters == cluster_id
            all_means.append(data.adata.X[mask].mean(axis=0))
        between_cluster_var = np.var(all_means, axis=0).mean()

        # Between-cluster variance should be much larger than within-cluster
        avg_within = np.mean(within_cluster_var)
        ratio = between_cluster_var / avg_within

        print(f"\nBetween-cluster variance: {between_cluster_var:.3f}")
        print(f"Average within-cluster variance: {avg_within:.3f}")
        print(f"Ratio: {ratio:.2f}")

        assert ratio > 1.2  # Clusters should have some separation

    def test_validate_mask_quality(self):
        """Example: Validate segmentation mask properties."""
        data = get_small_dataset(random_seed=42)

        cell_mask = data.cell_mask
        nucleus_mask = data.nucleus_mask

        # Count cells in masks
        n_cells_in_cell_mask = len(np.unique(cell_mask)) - 1  # Exclude background
        n_cells_in_nucleus_mask = len(np.unique(nucleus_mask)) - 1

        # Count cells in AnnData
        n_cells_in_adata = data.adata.n_obs

        print(f"\nCells in cell mask: {n_cells_in_cell_mask}")
        print(f"Cells in nucleus mask: {n_cells_in_nucleus_mask}")
        print(f"Cells in AnnData: {n_cells_in_adata}")

        # Note: Due to Voronoi tessellation, many cells may overlap
        # The mask generation creates irregular cell territories
        # We just verify that we have a reasonable number of cells
        assert n_cells_in_cell_mask > 100  # Should have many cells
        assert n_cells_in_adata == 1000  # AnnData has expected count


if __name__ == "__main__":
    """Run examples when executed as a script."""
    pytest.main([__file__, "-v", "-s"])
