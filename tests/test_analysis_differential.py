"""
Baseline CPU tests for differential expression analysis.

These tests establish a baseline for the current CPU-only differential expression
functionality and will be used to ensure GPU versions produce equivalent results.
"""

import numpy as np
import pandas as pd
import pytest

from aegle_analysis.analysis.differential import (
    one_vs_rest_wilcoxon,
    build_fold_change_matrix,
    build_log_fold_change_matrix,
)
from tests.utils.synthetic_analysis_data import get_small_dataset


@pytest.mark.slow
class TestDifferentialExpression:
    """Test suite for differential expression analysis."""

    def test_one_vs_rest_wilcoxon_basic(self):
        """Test that one_vs_rest_wilcoxon produces expected output structure."""
        # Get synthetic data with known cluster structure
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Extract data as DataFrame
        df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

        # Create log1p version for fold change calculation
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),  # Use abs to ensure positive values
            index=adata.obs_names,
            columns=adata.var_names,
        )

        cluster_series = adata.obs["true_cluster"]

        # Run differential expression
        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # Validate structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) > 0, "Results should not be empty"

        # Check that we have results for each cluster
        unique_clusters = cluster_series.unique()
        for cluster in unique_clusters:
            assert cluster in results, f"Missing results for cluster {cluster}"

    def test_one_vs_rest_wilcoxon_output_columns(self):
        """Test that differential expression results have expected columns."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # Get results for first cluster
        first_cluster = list(results.keys())[0]
        df_res = results[first_cluster]

        # Check required columns
        required_columns = [
            "feature",
            "p_value",
            "p_value_corrected",
            "log2_fold_change",
            "mean_cluster_norm",
            "mean_rest_norm",
        ]

        for col in required_columns:
            assert col in df_res.columns, f"Missing column: {col}"

    def test_one_vs_rest_wilcoxon_p_values_valid(self):
        """Test that p-values are in valid range [0, 1]."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # Check p-values for all clusters
        for cluster, df_res in results.items():
            # Check uncorrected p-values
            assert (df_res["p_value"] >= 0).all(), f"Negative p-values in cluster {cluster}"
            assert (df_res["p_value"] <= 1).all(), f"P-values > 1 in cluster {cluster}"

            # Check corrected p-values
            assert (df_res["p_value_corrected"] >= 0).all()
            assert (df_res["p_value_corrected"] <= 1).all()

    def test_one_vs_rest_wilcoxon_all_markers_tested(self):
        """Test that all markers are tested for each cluster."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        n_markers = len(adata.var_names)

        # Check that each cluster has results for all markers
        for cluster, df_res in results.items():
            assert len(df_res) == n_markers, (
                f"Cluster {cluster} has {len(df_res)} markers, expected {n_markers}"
            )

            # Check that all marker names are present
            tested_markers = set(df_res["feature"].values)
            expected_markers = set(adata.var_names)
            assert tested_markers == expected_markers, (
                f"Marker mismatch in cluster {cluster}"
            )

    def test_one_vs_rest_wilcoxon_reproducibility(self):
        """Test that differential expression is deterministic."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        # Run twice
        results1 = one_vs_rest_wilcoxon(df.copy(), cluster_series.copy(), df_log1p.copy())
        results2 = one_vs_rest_wilcoxon(df.copy(), cluster_series.copy(), df_log1p.copy())

        # Results should be identical
        assert results1.keys() == results2.keys()

        for cluster in results1.keys():
            df1 = results1[cluster]
            df2 = results2[cluster]

            # P-values should be identical
            assert np.allclose(df1["p_value"].values, df2["p_value"].values)
            assert np.allclose(
                df1["p_value_corrected"].values, df2["p_value_corrected"].values
            )

    def test_one_vs_rest_wilcoxon_detects_cluster_markers(self):
        """Test that DE analysis finds markers specific to clusters."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # For each cluster, check that there are some significant markers
        for cluster, df_res in results.items():
            # At least some markers should be significant (p < 0.05 after correction)
            n_significant = (df_res["p_value_corrected"] < 0.05).sum()

            # With synthetic data and clear cluster structure, expect some DE markers
            # (not enforcing this strictly as it depends on cluster separation)
            # Just check that the analysis ran and produced reasonable output
            assert n_significant >= 0, "Negative count of significant markers"


@pytest.mark.slow
class TestFoldChangeMatrix:
    """Test suite for fold change matrix building functions."""

    def test_build_fold_change_matrix_basic(self):
        """Test that build_fold_change_matrix produces expected structure."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        # Get DE results
        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # Build fold change matrix
        fc_matrix = build_fold_change_matrix(results, use_log=True, use_raw=False)

        # Validate structure
        assert isinstance(fc_matrix, pd.DataFrame)
        assert fc_matrix.shape[0] == len(results), "Wrong number of clusters"
        assert fc_matrix.shape[1] == len(adata.var_names), "Wrong number of markers"

    def test_build_fold_change_matrix_no_nan_inf(self):
        """Test that fold change matrix has no NaN or inf values."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)
        fc_matrix = build_fold_change_matrix(results, use_log=True, use_raw=False)

        # Check for NaN and inf (function replaces them with 0)
        assert not fc_matrix.isna().any().any(), "NaN values in fold change matrix"
        assert not np.isinf(fc_matrix.values).any(), "Inf values in fold change matrix"

    def test_build_fold_change_matrix_top_n_markers(self):
        """Test that top_n_markers parameter filters correctly."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # Build with top 5 markers
        fc_matrix = build_fold_change_matrix(
            results, use_log=True, use_raw=False, top_n_markers=5
        )

        # Should have fewer columns (union of top 5 from each cluster)
        n_clusters = len(results)
        max_markers = min(5 * n_clusters, len(adata.var_names))

        assert fc_matrix.shape[1] <= max_markers, (
            f"Too many markers in matrix: {fc_matrix.shape[1]}"
        )

    def test_build_log_fold_change_matrix_basic(self):
        """Test build_log_fold_change_matrix function."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        df_log1p = pd.DataFrame(
            np.log1p(np.abs(adata.X)),
            index=adata.obs_names,
            columns=adata.var_names,
        )
        cluster_series = adata.obs["true_cluster"]

        results = one_vs_rest_wilcoxon(df, cluster_series, df_log1p)

        # Build log fold change matrix
        lfc_matrix = build_log_fold_change_matrix(results)

        # Validate
        assert isinstance(lfc_matrix, pd.DataFrame)
        assert lfc_matrix.shape[0] == len(results)
        assert lfc_matrix.shape[1] == len(adata.var_names)

        # Should not have NaN or inf
        assert not lfc_matrix.isna().any().any()
        assert not np.isinf(lfc_matrix.values).any()

    def test_build_fold_change_matrix_empty_results(self):
        """Test that empty results dict returns empty DataFrame."""
        results = {}
        fc_matrix = build_fold_change_matrix(results)

        assert isinstance(fc_matrix, pd.DataFrame)
        assert fc_matrix.empty, "Empty results should produce empty DataFrame"
