"""Utilities for generating synthetic AnnData objects for analysis testing.

This module provides factory functions to create realistic single-cell expression
data with known cluster structure, mimicking the output of the Aegle pipeline for
use in downstream analysis testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import anndata
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticAnalysisData:
    """Container holding synthetic analysis data and ground truth."""

    adata: anndata.AnnData
    cell_mask: np.ndarray
    nucleus_mask: np.ndarray
    true_clusters: np.ndarray
    marker_cluster_specificity: dict


def create_synthetic_anndata(
    n_cells: int = 1000,
    n_markers: int = 20,
    n_clusters: int = 5,
    random_seed: int = 42,
    *,
    image_shape: Tuple[int, int] = (512, 512),
    cluster_separation: float = 2.0,
    noise_level: float = 0.3,
    add_spatial_coords: bool = True,
    add_morphology_features: bool = True,
) -> SyntheticAnalysisData:
    """Generate synthetic single-cell data with known cluster structure.

    This function creates a realistic AnnData object mimicking Aegle pipeline output,
    including marker expression patterns, cell metadata, spatial coordinates, and
    segmentation masks. The data has known ground truth cluster assignments.

    Parameters
    ----------
    n_cells : int
        Number of cells to generate
    n_markers : int
        Number of protein markers (channels)
    n_clusters : int
        Number of cell type clusters with distinct expression patterns
    random_seed : int
        Random seed for reproducibility
    image_shape : Tuple[int, int]
        Height and width of synthetic image for spatial mask generation
    cluster_separation : float
        How separated clusters should be in expression space (higher = more distinct)
    noise_level : float
        Amount of random noise to add to marker expression (0-1 scale)
    add_spatial_coords : bool
        Whether to add X, Y spatial coordinates to obs
    add_morphology_features : bool
        Whether to add nucleus_area, cell_area, eccentricity to obs

    Returns
    -------
    SyntheticAnalysisData
        Container with adata, masks, and ground truth cluster assignments

    Notes
    -----
    The generated data includes:
    - adata.X: Normalized marker expression (mean-centered, unit variance per marker)
    - adata.obs: Cell metadata (true_cluster, spatial coords, morphology)
    - adata.var: Marker metadata (marker_name, cluster_specificity)
    - Cell and nucleus segmentation masks with IDs matching cell indices
    """
    rng = np.random.RandomState(random_seed)

    # Generate marker names (similar to antibody panel)
    marker_names = [
        f"Marker{i+1:02d}" if i >= 10 else f"CD{i*4+10}"
        for i in range(n_markers)
    ]

    # Assign ground truth cluster labels (balanced distribution)
    true_clusters = rng.choice(n_clusters, size=n_cells, replace=True)

    # Create cluster-specific expression patterns
    # Each cluster has 2-4 highly expressed markers, some shared across clusters
    marker_cluster_specificity = {}
    expression_matrix = np.zeros((n_cells, n_markers), dtype=np.float32)

    # Assign markers to clusters
    markers_per_cluster = max(2, n_markers // n_clusters)
    for cluster_id in range(n_clusters):
        # Primary markers (cluster-specific)
        n_primary = max(2, markers_per_cluster // 2)
        primary_markers = rng.choice(
            n_markers, size=n_primary, replace=False
        )

        # Secondary markers (shared with other clusters)
        n_secondary = markers_per_cluster - n_primary
        secondary_markers = rng.choice(
            n_markers, size=n_secondary, replace=False
        )

        cluster_mask = true_clusters == cluster_id
        n_cells_in_cluster = cluster_mask.sum()

        # Set primary marker expression (high)
        for marker_idx in primary_markers:
            base_expression = rng.normal(
                loc=cluster_separation * 2.0,
                scale=0.5,
                size=n_cells_in_cluster,
            )
            expression_matrix[cluster_mask, marker_idx] = np.maximum(
                base_expression, 0
            )

            # Track which markers are cluster-specific
            marker_name = marker_names[marker_idx]
            if marker_name not in marker_cluster_specificity:
                marker_cluster_specificity[marker_name] = []
            marker_cluster_specificity[marker_name].append(cluster_id)

        # Set secondary marker expression (moderate)
        for marker_idx in secondary_markers:
            base_expression = rng.normal(
                loc=cluster_separation,
                scale=0.5,
                size=n_cells_in_cluster,
            )
            # Add to existing expression if marker is shared
            expression_matrix[cluster_mask, marker_idx] += np.maximum(
                base_expression, 0
            )

    # Add background noise to all markers
    noise = rng.normal(
        loc=0.5, scale=noise_level, size=(n_cells, n_markers)
    )
    expression_matrix += np.maximum(noise, 0)

    # Normalize: mean-center and scale to unit variance per marker
    expression_matrix = (
        expression_matrix - expression_matrix.mean(axis=0)
    ) / (expression_matrix.std(axis=0) + 1e-8)

    # Create spatial coordinates (clustered in space)
    if add_spatial_coords:
        spatial_coords = _generate_spatial_coordinates(
            true_clusters, n_clusters, image_shape, rng
        )
    else:
        spatial_coords = None

    # Create morphology features (realistic ranges from real data)
    if add_morphology_features:
        morphology_features = _generate_morphology_features(
            n_cells, true_clusters, rng
        )
    else:
        morphology_features = {}

    # Build observation (cell) metadata
    obs_data = {
        "cell_id": np.arange(1, n_cells + 1),
        "true_cluster": true_clusters.astype(str),
    }

    if spatial_coords is not None:
        obs_data["X"] = spatial_coords[:, 1]
        obs_data["Y"] = spatial_coords[:, 0]

    obs_data.update(morphology_features)

    obs = pd.DataFrame(obs_data, index=[f"cell_{i+1}" for i in range(n_cells)])

    # Build variable (marker) metadata
    var = pd.DataFrame(
        {"marker_name": marker_names},
        index=marker_names,
    )

    # Add cluster specificity information
    var["n_clusters_specific"] = [
        len(marker_cluster_specificity.get(m, [])) for m in marker_names
    ]

    # Create AnnData object
    adata = anndata.AnnData(X=expression_matrix, obs=obs, var=var)

    # Add unstructured metadata
    adata.uns["n_clusters"] = n_clusters
    adata.uns["random_seed"] = random_seed
    adata.uns["cluster_separation"] = cluster_separation
    adata.uns["noise_level"] = noise_level

    # Generate segmentation masks
    cell_mask, nucleus_mask = create_synthetic_cell_mask(
        n_cells, image_shape, random_seed, spatial_coords
    )

    return SyntheticAnalysisData(
        adata=adata,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        true_clusters=true_clusters,
        marker_cluster_specificity=marker_cluster_specificity,
    )


def _generate_spatial_coordinates(
    cluster_labels: np.ndarray,
    n_clusters: int,
    image_shape: Tuple[int, int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate spatially clustered coordinates for cells.

    Each cluster is centered in a different region of the image with some overlap.
    """
    n_cells = len(cluster_labels)
    height, width = image_shape
    coords = np.zeros((n_cells, 2), dtype=np.float32)

    # Create cluster centers in a grid pattern
    grid_size = int(np.ceil(np.sqrt(n_clusters)))
    cluster_centers = []

    for i in range(n_clusters):
        grid_y = (i // grid_size) + 0.5
        grid_x = (i % grid_size) + 0.5
        center_y = (grid_y / grid_size) * height
        center_x = (grid_x / grid_size) * width
        cluster_centers.append((center_y, center_x))

    # Assign cells to spatial locations around their cluster center
    spread = min(height, width) / (grid_size * 2.5)

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        n_cluster_cells = cluster_mask.sum()

        center_y, center_x = cluster_centers[cluster_id]

        # Generate positions with Gaussian spread around center
        cell_y = rng.normal(loc=center_y, scale=spread, size=n_cluster_cells)
        cell_x = rng.normal(loc=center_x, scale=spread, size=n_cluster_cells)

        # Clip to image bounds
        cell_y = np.clip(cell_y, 10, height - 10)
        cell_x = np.clip(cell_x, 10, width - 10)

        coords[cluster_mask, 0] = cell_y
        coords[cluster_mask, 1] = cell_x

    return coords


def _generate_morphology_features(
    n_cells: int, cluster_labels: np.ndarray, rng: np.random.RandomState
) -> dict:
    """Generate realistic morphology features (area, eccentricity).

    Different cell types (clusters) have slightly different size distributions.
    """
    # Realistic ranges based on actual PhenoCycler data
    base_nucleus_area = 300  # pixels
    base_cell_area = 800  # pixels

    nucleus_areas = np.zeros(n_cells)
    cell_areas = np.zeros(n_cells)
    eccentricities = np.zeros(n_cells)

    n_clusters = cluster_labels.max() + 1

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        n_cluster_cells = cluster_mask.sum()

        # Slight size variation per cluster (some cell types are larger)
        size_factor = 0.8 + (cluster_id / n_clusters) * 0.6

        # Nucleus area with log-normal distribution
        nucleus_areas[cluster_mask] = rng.lognormal(
            mean=np.log(base_nucleus_area * size_factor),
            sigma=0.3,
            size=n_cluster_cells,
        )

        # Cell area (always larger than nucleus)
        cell_areas[cluster_mask] = (
            nucleus_areas[cluster_mask] * rng.uniform(2.0, 4.0, n_cluster_cells)
        )

        # Eccentricity (0 = circle, 1 = line)
        eccentricities[cluster_mask] = rng.beta(2, 5, size=n_cluster_cells)

    return {
        "nucleus_area": nucleus_areas,
        "cell_area": cell_areas,
        "eccentricity": eccentricities,
    }


def create_synthetic_cell_mask(
    n_cells: int,
    image_shape: Tuple[int, int] = (512, 512),
    random_seed: int = 42,
    spatial_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic segmentation masks for spatial plots.

    Parameters
    ----------
    n_cells : int
        Number of cells to generate
    image_shape : Tuple[int, int]
        Height and width of output mask
    random_seed : int
        Random seed for reproducibility
    spatial_coords : Optional[np.ndarray]
        Pre-computed spatial coordinates (n_cells, 2) in (y, x) format.
        If None, random positions are generated.

    Returns
    -------
    cell_mask : np.ndarray
        Cell segmentation mask (background=0, cells=1,2,3...n_cells)
    nucleus_mask : np.ndarray
        Nucleus segmentation mask (background=0, nuclei=1,2,3...n_cells)

    Notes
    -----
    Cell IDs are 1-indexed and match AnnData cell indices (cell_1 = label 1).
    """
    from scipy.ndimage import distance_transform_edt

    rng = np.random.RandomState(random_seed)
    height, width = image_shape

    cell_mask = np.zeros(image_shape, dtype=np.uint32)
    nucleus_mask = np.zeros(image_shape, dtype=np.uint32)

    # Generate cell centers
    if spatial_coords is not None:
        # Use provided coordinates
        centers = spatial_coords.astype(int)
    else:
        # Random placement with minimum distance
        centers = []
        min_distance = 15

        attempts = 0
        max_attempts = n_cells * 100

        while len(centers) < n_cells and attempts < max_attempts:
            y = rng.randint(20, height - 20)
            x = rng.randint(20, width - 20)

            # Check minimum distance to existing centers
            if len(centers) == 0 or all(
                np.sqrt((y - cy) ** 2 + (x - cx) ** 2) >= min_distance
                for cy, cx in centers
            ):
                centers.append((y, x))

            attempts += 1

        if len(centers) < n_cells:
            # Fallback: just place remaining cells randomly
            for _ in range(n_cells - len(centers)):
                y = rng.randint(20, height - 20)
                x = rng.randint(20, width - 20)
                centers.append((y, x))

        centers = np.array(centers)

    # Create Voronoi-like cell territories
    # For each pixel, assign it to the nearest cell center
    yy, xx = np.mgrid[0:height, 0:width]

    for cell_id in range(1, n_cells + 1):
        cy, cx = centers[cell_id - 1]

        # Create circular-ish cell shape with some irregularity
        cell_radius = rng.uniform(8, 15)
        nucleus_radius = cell_radius * rng.uniform(0.4, 0.6)

        # Distance from center
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        # Add some shape irregularity using random deformation
        angle = np.arctan2(yy - cy, xx - cx)
        deformation = rng.normal(0, 0.15, size=8)
        angle_bins = np.linspace(-np.pi, np.pi, 9)

        irregular_factor = np.ones_like(dist)
        for i in range(8):
            angle_mask = (angle >= angle_bins[i]) & (angle < angle_bins[i + 1])
            irregular_factor[angle_mask] = 1.0 + deformation[i]

        # Create cell and nucleus masks
        cell_area = dist <= (cell_radius * irregular_factor)
        nucleus_area = dist <= (nucleus_radius * irregular_factor * 0.8)

        # Only paint if not already occupied (later cells don't override)
        cell_mask[cell_area & (cell_mask == 0)] = cell_id
        nucleus_mask[nucleus_area & (nucleus_mask == 0)] = cell_id

    return cell_mask, nucleus_mask


# =============================================================================
# Predefined Dataset Generators
# =============================================================================


def get_small_dataset(random_seed: int = 42) -> SyntheticAnalysisData:
    """Get small synthetic dataset for fast unit tests.

    Returns
    -------
    SyntheticAnalysisData
        1K cells, 10 markers, 3 clusters
    """
    return create_synthetic_anndata(
        n_cells=1000,
        n_markers=10,
        n_clusters=3,
        random_seed=random_seed,
        image_shape=(256, 256),
    )


def get_medium_dataset(random_seed: int = 42) -> SyntheticAnalysisData:
    """Get medium synthetic dataset for integration tests.

    Returns
    -------
    SyntheticAnalysisData
        10K cells, 25 markers, 5 clusters
    """
    return create_synthetic_anndata(
        n_cells=10000,
        n_markers=25,
        n_clusters=5,
        random_seed=random_seed,
        image_shape=(512, 512),
    )


def get_large_dataset(random_seed: int = 42) -> SyntheticAnalysisData:
    """Get large synthetic dataset for performance tests.

    Returns
    -------
    SyntheticAnalysisData
        100K cells, 50 markers, 8 clusters
    """
    return create_synthetic_anndata(
        n_cells=100000,
        n_markers=50,
        n_clusters=8,
        random_seed=random_seed,
        image_shape=(1024, 1024),
        cluster_separation=2.5,
    )


def get_stress_dataset(random_seed: int = 42) -> SyntheticAnalysisData:
    """Get stress test dataset for maximum load tests.

    Returns
    -------
    SyntheticAnalysisData
        500K cells, 50 markers, 10 clusters
    """
    return create_synthetic_anndata(
        n_cells=500000,
        n_markers=50,
        n_clusters=10,
        random_seed=random_seed,
        image_shape=(2048, 2048),
        cluster_separation=3.0,
    )


# =============================================================================
# Validation and Visualization
# =============================================================================


def validate_synthetic_data(
    output_dir: Path = Path("tests/debug_synthetic_analysis_previews"),
    random_seed: int = 42,
) -> None:
    """Generate validation visualizations for synthetic data quality.

    Creates preview images showing:
    - UMAP of cluster structure
    - Heatmap of marker expression patterns
    - Spatial mask visualizations
    - Distribution plots

    Parameters
    ----------
    output_dir : Path
        Directory to save preview images
    random_seed : int
        Random seed for reproducibility
    """
    import matplotlib.pyplot as plt
    import scanpy as sc
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Validating Synthetic Analysis Data Generator")
    print("=" * 80)

    # Test each dataset size
    datasets = {
        "small": get_small_dataset(random_seed),
        "medium": get_medium_dataset(random_seed),
        "large": get_large_dataset(random_seed),
    }

    for dataset_name, synth_data in datasets.items():
        print(f"\n[{dataset_name.upper()} Dataset]")
        adata = synth_data.adata

        print(f"  Cells: {adata.n_obs:,}")
        print(f"  Markers: {adata.n_vars}")
        print(f"  Clusters: {adata.uns['n_clusters']}")
        print(f"  Cell mask shape: {synth_data.cell_mask.shape}")
        print(f"  Unique cell IDs in mask: {len(np.unique(synth_data.cell_mask)) - 1}")

        # Verify data structure
        assert adata.obs["true_cluster"].nunique() == adata.uns["n_clusters"]
        assert "X" in adata.obs.columns and "Y" in adata.obs.columns
        assert "nucleus_area" in adata.obs.columns
        assert "cell_area" in adata.obs.columns
        assert "eccentricity" in adata.obs.columns

        # Compute UMAP for visualization
        print(f"  Computing UMAP...")
        sc.pp.neighbors(adata, n_neighbors=15, random_state=random_seed)
        sc.tl.umap(adata, random_state=random_seed)

        # Plot 1: UMAP colored by cluster
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cluster_colors = plt.cm.tab10(synth_data.true_clusters / adata.uns["n_clusters"])

        for cluster_id in range(adata.uns["n_clusters"]):
            mask = synth_data.true_clusters == cluster_id
            ax.scatter(
                adata.obsm["X_umap"][mask, 0],
                adata.obsm["X_umap"][mask, 1],
                c=[cluster_colors[mask][0]],
                label=f"Cluster {cluster_id}",
                alpha=0.6,
                s=10,
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP - {dataset_name.capitalize()} Dataset\n{adata.n_obs:,} cells, {adata.uns['n_clusters']} clusters")
        ax.legend(markerscale=2, frameon=True, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_umap.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved UMAP plot")

        # Plot 2: Heatmap of marker expression by cluster
        # Compute mean expression per cluster
        cluster_means = []
        for cluster_id in range(adata.uns["n_clusters"]):
            mask = synth_data.true_clusters == cluster_id
            cluster_means.append(adata.X[mask].mean(axis=0))

        cluster_means = np.array(cluster_means)
        cluster_df = pd.DataFrame(
            cluster_means,
            index=[f"Cluster {i}" for i in range(adata.uns["n_clusters"])],
            columns=adata.var_names,
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.heatmap(
            cluster_df,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Mean Expression (z-score)"},
            ax=ax,
        )
        ax.set_title(f"Marker Expression Heatmap - {dataset_name.capitalize()} Dataset")
        ax.set_xlabel("Markers")
        ax.set_ylabel("Clusters")
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved heatmap plot")

        # Plot 3: Spatial mask visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Cell mask
        axes[0].imshow(synth_data.cell_mask, cmap="nipy_spectral", interpolation="nearest")
        axes[0].set_title(f"Cell Mask - {dataset_name.capitalize()}")
        axes[0].axis("off")

        # Nucleus mask
        axes[1].imshow(synth_data.nucleus_mask, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Nucleus Mask - {dataset_name.capitalize()}")
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_masks.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved mask visualizations")

        # Plot 4: Morphology distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(adata.obs["nucleus_area"], bins=50, alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("Nucleus Area (pixels)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Nucleus Area Distribution")

        axes[1].hist(adata.obs["cell_area"], bins=50, alpha=0.7, edgecolor="black", color="orange")
        axes[1].set_xlabel("Cell Area (pixels)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Cell Area Distribution")

        axes[2].hist(adata.obs["eccentricity"], bins=50, alpha=0.7, edgecolor="black", color="green")
        axes[2].set_xlabel("Eccentricity")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Eccentricity Distribution")

        fig.suptitle(f"Morphology Features - {dataset_name.capitalize()} Dataset")
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_morphology.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved morphology distributions")

    # Summary statistics
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("\n✓ All datasets generated successfully")
    print(f"✓ Preview images saved to: {output_dir.absolute()}")
    print(f"✓ Total files created: {len(list(output_dir.glob('*.png')))}")

    print("\nDataset Sizes:")
    print(f"  - Small:  {datasets['small'].adata.n_obs:>7,} cells")
    print(f"  - Medium: {datasets['medium'].adata.n_obs:>7,} cells")
    print(f"  - Large:  {datasets['large'].adata.n_obs:>7,} cells")

    print("\nValidation complete!")
    print("=" * 80)


if __name__ == "__main__":
    """Run validation when executed as a script."""
    validate_synthetic_data()
