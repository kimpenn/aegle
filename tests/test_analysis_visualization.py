"""
Baseline CPU tests for visualization functions.

These tests verify that visualization functions run without errors on the current
CPU-only implementation. They do not validate plot appearance, only execution.
"""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend to avoid display issues
matplotlib.use("Agg")

from aegle_analysis.visualization.marker_plots import (
    plot_marker_distributions,
    plot_heatmap,
    plot_marker_correlations,
)
from aegle_analysis.visualization.spatial_plots import (
    plot_segmentation_masks,
    plot_clustering_on_mask,
    plot_marker_expression_on_mask,
    plot_umap,
)
from aegle_analysis.analysis.clustering import run_clustering
from tests.utils.synthetic_analysis_data import get_small_dataset


class TestMarkerPlots:
    """Test suite for marker distribution plotting functions."""

    def test_plot_marker_distributions_violin(self):
        """Test that violin plots run without error."""
        synth_data = get_small_dataset(random_seed=42)

        # Extract expression data
        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        # Plot (no saving, just test execution)
        try:
            plot_marker_distributions(
                df, log_transform=False, markers_per_plot=5, plot_type="violin"
            )
            plt.close("all")  # Clean up
        except Exception as e:
            pytest.fail(f"plot_marker_distributions (violin) raised exception: {e}")

    def test_plot_marker_distributions_box(self):
        """Test that box plots run without error."""
        synth_data = get_small_dataset(random_seed=42)

        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        try:
            plot_marker_distributions(
                df, log_transform=False, markers_per_plot=5, plot_type="box"
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_marker_distributions (box) raised exception: {e}")

    def test_plot_marker_distributions_both(self):
        """Test that both violin and box plots run without error."""
        synth_data = get_small_dataset(random_seed=42)

        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        try:
            plot_marker_distributions(
                df, log_transform=False, markers_per_plot=5, plot_type="both"
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_marker_distributions (both) raised exception: {e}")

    def test_plot_marker_distributions_log_transform(self):
        """Test marker distributions with log transformation."""
        synth_data = get_small_dataset(random_seed=42)

        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        try:
            plot_marker_distributions(
                df, log_transform=True, markers_per_plot=5, plot_type="violin"
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_marker_distributions with log_transform raised exception: {e}")

    def test_plot_marker_distributions_save(self):
        """Test that plots can be saved to file."""
        synth_data = get_small_dataset(random_seed=42)

        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                plot_marker_distributions(
                    df,
                    log_transform=False,
                    markers_per_plot=5,
                    plot_type="violin",
                    output_dir=tmpdir,
                )
                plt.close("all")

                # Check that file was created
                expected_file = os.path.join(tmpdir, "raw_violin_plots.png")
                assert os.path.exists(expected_file), f"Plot file not created: {expected_file}"
            except Exception as e:
                pytest.fail(f"plot_marker_distributions save raised exception: {e}")

    def test_plot_heatmap_basic(self):
        """Test that heatmap plotting runs without error."""
        # Create simple test matrix
        matrix = pd.DataFrame({
            "marker1": [1.0, -0.5, 0.2],
            "marker2": [-0.3, 0.8, -0.1],
            "marker3": [0.5, -0.2, 0.9],
        }, index=["cluster1", "cluster2", "cluster3"])

        try:
            plot_heatmap(matrix, title="Test Heatmap")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_heatmap raised exception: {e}")

    def test_plot_heatmap_save(self):
        """Test that heatmap can be saved to file."""
        matrix = pd.DataFrame({
            "marker1": [1.0, -0.5],
            "marker2": [-0.3, 0.8],
        }, index=["cluster1", "cluster2"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_heatmap.png")

            try:
                plot_heatmap(matrix, output_path=output_path)
                plt.close("all")

                assert os.path.exists(output_path), "Heatmap file not created"
            except Exception as e:
                pytest.fail(f"plot_heatmap save raised exception: {e}")

    def test_plot_marker_correlations_basic(self):
        """Test that correlation heatmap runs without error."""
        synth_data = get_small_dataset(random_seed=42)

        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        try:
            plot_marker_correlations(df, method="spearman")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_marker_correlations raised exception: {e}")

    def test_plot_marker_correlations_methods(self):
        """Test correlation heatmap with different methods."""
        synth_data = get_small_dataset(random_seed=42)

        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        methods = ["pearson", "spearman", "kendall"]

        for method in methods:
            try:
                plot_marker_correlations(df, method=method)
                plt.close("all")
            except Exception as e:
                pytest.fail(f"plot_marker_correlations ({method}) raised exception: {e}")


class TestSpatialPlots:
    """Test suite for spatial visualization functions."""

    def test_plot_segmentation_masks_cell_only(self):
        """Test plotting cell mask only."""
        synth_data = get_small_dataset(random_seed=42)
        cell_mask = synth_data.cell_mask

        try:
            plot_segmentation_masks(cell_mask, nuc_mask=None)
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_segmentation_masks (cell only) raised exception: {e}")

    def test_plot_segmentation_masks_both(self):
        """Test plotting both cell and nucleus masks."""
        synth_data = get_small_dataset(random_seed=42)
        cell_mask = synth_data.cell_mask
        nucleus_mask = synth_data.nucleus_mask

        try:
            plot_segmentation_masks(cell_mask, nuc_mask=nucleus_mask)
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_segmentation_masks (both) raised exception: {e}")

    def test_plot_segmentation_masks_save(self):
        """Test saving segmentation mask plot."""
        synth_data = get_small_dataset(random_seed=42)
        cell_mask = synth_data.cell_mask

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "masks.png")

            try:
                plot_segmentation_masks(cell_mask, output_path=output_path)
                plt.close("all")

                assert os.path.exists(output_path), "Mask plot not created"
            except Exception as e:
                pytest.fail(f"plot_segmentation_masks save raised exception: {e}")

    def test_plot_clustering_on_mask_basic(self):
        """Test plotting clusters on segmentation mask."""
        synth_data = get_small_dataset(random_seed=42)
        cell_mask = synth_data.cell_mask

        # Convert true_cluster to integer labels
        cluster_labels = synth_data.adata.obs["true_cluster"].astype(int).values

        # The synthetic mask may not have all cell IDs (some cells lost in mask generation)
        # So we need to handle the case where mask has fewer cells than adata
        max_cell_id = int(cell_mask.max())

        # Only use cluster labels for cells that exist in the mask
        cluster_labels_trimmed = cluster_labels[:max_cell_id]

        try:
            plot_clustering_on_mask(cell_mask, cluster_labels_trimmed)
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_clustering_on_mask raised exception: {e}")

    def test_plot_marker_expression_on_mask_basic(self):
        """Test plotting marker expression on mask."""
        synth_data = get_small_dataset(random_seed=42)
        cell_mask = synth_data.cell_mask

        # Get expression values for first marker
        expression_values = synth_data.adata.X[:, 0]

        # The synthetic mask may not have all cell IDs (some cells lost in mask generation)
        # So we need to handle the case where mask has fewer cells than adata
        max_cell_id = int(cell_mask.max())

        # Only use expression values for cells that exist in the mask
        expression_values_trimmed = expression_values[:max_cell_id]

        try:
            plot_marker_expression_on_mask(
                cell_mask, expression_values_trimmed, marker_name="Marker1"
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_marker_expression_on_mask raised exception: {e}")

    def test_plot_umap_basic(self):
        """Test UMAP plotting runs without error."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering to generate UMAP
        adata = run_clustering(adata, random_state=42)

        try:
            plot_umap(adata, color_by=["leiden"])
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_umap raised exception: {e}")

    def test_plot_umap_multiple_colors(self):
        """Test UMAP plotting with multiple color variables."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering
        adata = run_clustering(adata, random_state=42)

        try:
            plot_umap(adata, color_by=["leiden", "true_cluster"])
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_umap (multiple colors) raised exception: {e}")

    def test_plot_umap_continuous_variable(self):
        """Test UMAP plotting with continuous variable."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering
        adata = run_clustering(adata, random_state=42)

        try:
            # Color by a continuous variable (cell area)
            plot_umap(adata, color_by=["cell_area"])
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot_umap (continuous variable) raised exception: {e}")

    def test_plot_umap_save(self):
        """Test UMAP plot saving."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Run clustering
        adata = run_clustering(adata, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                plot_umap(adata, color_by=["leiden"], output_dir=tmpdir)
                plt.close("all")

                # Scanpy saves with "umap_" prefix and color variable name
                expected_file = os.path.join(tmpdir, "umap_leiden.png")
                assert os.path.exists(expected_file), f"UMAP plot not created: {expected_file}"
            except Exception as e:
                pytest.fail(f"plot_umap save raised exception: {e}")


class TestVisualizationRobustness:
    """Test visualization robustness with edge cases."""

    def test_plot_with_small_data(self):
        """Test that plots work with very small datasets."""
        # Create minimal dataset
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0],
            "marker2": [4.0, 5.0, 6.0],
        })

        try:
            plot_marker_distributions(df, markers_per_plot=5, plot_type="violin")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"Plotting small dataset raised exception: {e}")

    def test_plot_with_single_marker(self):
        """Test plotting with only one marker."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        try:
            plot_marker_distributions(df, plot_type="violin")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"Plotting single marker raised exception: {e}")
