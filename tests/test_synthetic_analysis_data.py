"""Tests for synthetic analysis data generator.

This test suite validates the synthetic AnnData factory used across all
analysis module tests.
"""

import numpy as np
import pytest

from tests.utils.synthetic_analysis_data import (
    create_synthetic_anndata,
    create_synthetic_cell_mask,
    get_small_dataset,
    get_medium_dataset,
    get_large_dataset,
    get_stress_dataset,
)


class TestSyntheticAnnDataCreation:
    """Test core synthetic AnnData generation."""

    def test_basic_creation(self):
        """Test basic synthetic data creation."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )

        assert data.adata.n_obs == 100
        assert data.adata.n_vars == 10
        assert len(data.true_clusters) == 100
        assert data.true_clusters.max() < 3  # 0-indexed
        assert data.cell_mask.shape == (512, 512)
        assert data.nucleus_mask.shape == (512, 512)

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        data1 = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )
        data2 = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )

        np.testing.assert_array_equal(data1.true_clusters, data2.true_clusters)
        np.testing.assert_array_almost_equal(data1.adata.X, data2.adata.X)

    def test_different_seeds(self):
        """Test that different seeds produce different data."""
        data1 = create_synthetic_anndata(
            n_cells=100, n_markers=10, n_clusters=3, random_seed=42
        )
        data2 = create_synthetic_anndata(
            n_cells=100, n_markers=10, n_clusters=3, random_seed=99
        )

        # Should not be identical
        assert not np.array_equal(data1.true_clusters, data2.true_clusters)

    def test_cluster_distribution(self):
        """Test that clusters are reasonably balanced."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=10,
            n_clusters=5,
            random_seed=42,
        )

        unique, counts = np.unique(data.true_clusters, return_counts=True)
        assert len(unique) == 5

        # Check that clusters are somewhat balanced (not too skewed)
        # Each cluster should have at least 10% of cells
        min_cluster_size = 0.1 * data.adata.n_obs
        assert all(counts >= min_cluster_size)

    def test_expression_matrix_properties(self):
        """Test that expression matrix has expected properties."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=20,
            n_clusters=5,
            random_seed=42,
        )

        # Should be normalized (mean ~ 0, std ~ 1 per marker)
        means = data.adata.X.mean(axis=0)
        stds = data.adata.X.std(axis=0)

        # Check normalization (within tolerance)
        np.testing.assert_allclose(means, 0, atol=0.1)
        np.testing.assert_allclose(stds, 1, atol=0.2)


class TestSyntheticAnnDataMetadata:
    """Test metadata fields in synthetic AnnData."""

    def test_obs_structure(self):
        """Test that obs (cell metadata) has expected columns."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )

        obs = data.adata.obs

        # Required columns
        assert "cell_id" in obs.columns
        assert "true_cluster" in obs.columns
        assert "X" in obs.columns
        assert "Y" in obs.columns
        assert "nucleus_area" in obs.columns
        assert "cell_area" in obs.columns
        assert "eccentricity" in obs.columns

    def test_spatial_coordinates(self):
        """Test that spatial coordinates are valid."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
            image_shape=(512, 512),
        )

        # Coordinates should be within image bounds
        assert data.adata.obs["X"].min() >= 0
        assert data.adata.obs["X"].max() < 512
        assert data.adata.obs["Y"].min() >= 0
        assert data.adata.obs["Y"].max() < 512

    def test_morphology_features(self):
        """Test that morphology features have realistic values."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )

        obs = data.adata.obs

        # Nucleus area should be positive
        assert (obs["nucleus_area"] > 0).all()

        # Cell area should be larger than nucleus area
        assert (obs["cell_area"] > obs["nucleus_area"]).all()

        # Eccentricity should be in [0, 1]
        assert (obs["eccentricity"] >= 0).all()
        assert (obs["eccentricity"] <= 1).all()

    def test_var_structure(self):
        """Test that var (marker metadata) has expected structure."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )

        var = data.adata.var

        assert "marker_name" in var.columns
        assert "n_clusters_specific" in var.columns
        assert len(var) == 10

    def test_uns_metadata(self):
        """Test that uns (unstructured metadata) is populated."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
        )

        uns = data.adata.uns

        assert "n_clusters" in uns
        assert "random_seed" in uns
        assert "cluster_separation" in uns
        assert "noise_level" in uns

        assert uns["n_clusters"] == 3
        assert uns["random_seed"] == 42


class TestSyntheticCellMask:
    """Test synthetic cell mask generation."""

    def test_basic_mask_creation(self):
        """Test basic mask creation."""
        cell_mask, nucleus_mask = create_synthetic_cell_mask(
            n_cells=50,
            image_shape=(256, 256),
            random_seed=42,
        )

        assert cell_mask.shape == (256, 256)
        assert nucleus_mask.shape == (256, 256)
        assert cell_mask.dtype == np.uint32
        assert nucleus_mask.dtype == np.uint32

    def test_mask_labels(self):
        """Test that mask labels are 1-indexed."""
        cell_mask, nucleus_mask = create_synthetic_cell_mask(
            n_cells=50,
            image_shape=(256, 256),
            random_seed=42,
        )

        # Background should be 0
        assert 0 in cell_mask
        assert 0 in nucleus_mask

        # Cell IDs should start from 1
        cell_ids = np.unique(cell_mask)
        cell_ids = cell_ids[cell_ids > 0]  # Exclude background

        assert cell_ids.min() == 1

    def test_nucleus_inside_cell(self):
        """Test that nucleus is contained within cell for most cells."""
        cell_mask, nucleus_mask = create_synthetic_cell_mask(
            n_cells=50,
            image_shape=(256, 256),
            random_seed=42,
        )

        # For each nucleus label, it should also exist in cell mask
        nucleus_ids = np.unique(nucleus_mask)
        nucleus_ids = nucleus_ids[nucleus_ids > 0]

        # Check that on average, nuclei have reasonable overlap with cells
        # (Note: Some cells may be partially occluded by later cells in the generation)
        overlap_ratios = []
        for nuc_id in nucleus_ids:
            nuc_pixels = nucleus_mask == nuc_id
            # Most nucleus pixels should also be in the corresponding cell
            cell_overlap = (cell_mask == nuc_id) & nuc_pixels
            overlap_ratio = cell_overlap.sum() / nuc_pixels.sum()
            overlap_ratios.append(overlap_ratio)

        # Average overlap should be reasonable (some cells may be partially occluded)
        mean_overlap = np.mean(overlap_ratios)
        assert mean_overlap > 0.3  # At least 30% average overlap

    def test_spatial_coords_integration(self):
        """Test mask creation with provided spatial coordinates."""
        rng = np.random.RandomState(42)
        coords = rng.uniform(50, 200, size=(20, 2))

        cell_mask, nucleus_mask = create_synthetic_cell_mask(
            n_cells=20,
            image_shape=(256, 256),
            random_seed=42,
            spatial_coords=coords,
        )

        # Should have created masks
        assert cell_mask.max() > 0
        assert nucleus_mask.max() > 0


class TestPredefinedDatasets:
    """Test predefined dataset generators."""

    def test_small_dataset(self):
        """Test small dataset generator."""
        data = get_small_dataset(random_seed=42)

        assert data.adata.n_obs == 1000
        assert data.adata.n_vars == 10
        assert data.adata.uns["n_clusters"] == 3
        assert data.cell_mask.shape == (256, 256)

    def test_medium_dataset(self):
        """Test medium dataset generator."""
        data = get_medium_dataset(random_seed=42)

        assert data.adata.n_obs == 10000
        assert data.adata.n_vars == 25
        assert data.adata.uns["n_clusters"] == 5
        assert data.cell_mask.shape == (512, 512)

    def test_large_dataset(self):
        """Test large dataset generator."""
        data = get_large_dataset(random_seed=42)

        assert data.adata.n_obs == 100000
        assert data.adata.n_vars == 50
        assert data.adata.uns["n_clusters"] == 8
        assert data.cell_mask.shape == (1024, 1024)

    @pytest.mark.slow
    def test_stress_dataset(self):
        """Test stress dataset generator (slow test)."""
        data = get_stress_dataset(random_seed=42)

        assert data.adata.n_obs == 500000
        assert data.adata.n_vars == 50
        assert data.adata.uns["n_clusters"] == 10
        assert data.cell_mask.shape == (2048, 2048)


class TestClusterSpecificity:
    """Test cluster-specific expression patterns."""

    def test_cluster_separation(self):
        """Test that clusters have distinct expression patterns."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=20,
            n_clusters=4,
            random_seed=42,
            cluster_separation=3.0,
        )

        # Compute mean expression per cluster
        cluster_means = []
        for cluster_id in range(4):
            mask = data.true_clusters == cluster_id
            cluster_means.append(data.adata.X[mask].mean(axis=0))

        cluster_means = np.array(cluster_means)

        # Clusters should have different mean expression patterns
        # Check that at least some markers differ significantly between clusters
        marker_ranges = cluster_means.max(axis=0) - cluster_means.min(axis=0)

        # At least 50% of markers should show cluster differences > 1 std
        assert (marker_ranges > 1.0).sum() >= data.adata.n_vars * 0.5

    def test_marker_cluster_specificity(self):
        """Test that marker cluster specificity is tracked."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=20,
            n_clusters=4,
            random_seed=42,
        )

        # Some markers should be cluster-specific
        assert len(data.marker_cluster_specificity) > 0

        # Each cluster-specific marker should map to at least one cluster
        for marker, clusters in data.marker_cluster_specificity.items():
            assert len(clusters) > 0
            assert all(0 <= c < 4 for c in clusters)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_cluster(self):
        """Test creation with single cluster."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=1,
            random_seed=42,
        )

        assert data.adata.n_obs == 100
        assert len(np.unique(data.true_clusters)) == 1

    def test_many_clusters(self):
        """Test creation with many clusters."""
        data = create_synthetic_anndata(
            n_cells=1000,
            n_markers=50,
            n_clusters=20,
            random_seed=42,
        )

        assert data.adata.n_obs == 1000
        # Should have all clusters represented
        assert len(np.unique(data.true_clusters)) >= 15  # Allow some variability

    def test_small_image_shape(self):
        """Test with very small image shape."""
        data = create_synthetic_anndata(
            n_cells=10,
            n_markers=5,
            n_clusters=2,
            random_seed=42,
            image_shape=(64, 64),
        )

        assert data.cell_mask.shape == (64, 64)
        assert data.nucleus_mask.shape == (64, 64)

    def test_no_spatial_coords(self):
        """Test creation without spatial coordinates."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
            add_spatial_coords=False,
        )

        # Should not have X, Y columns
        assert "X" not in data.adata.obs.columns
        assert "Y" not in data.adata.obs.columns

    def test_no_morphology_features(self):
        """Test creation without morphology features."""
        data = create_synthetic_anndata(
            n_cells=100,
            n_markers=10,
            n_clusters=3,
            random_seed=42,
            add_morphology_features=False,
        )

        # Should not have morphology columns
        assert "nucleus_area" not in data.adata.obs.columns
        assert "cell_area" not in data.adata.obs.columns
        assert "eccentricity" not in data.adata.obs.columns
