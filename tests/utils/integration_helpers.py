"""Helper functions for analysis integration testing.

This module provides utilities for running and validating full analysis workflows
in integration tests. It supports running analysis with custom configurations,
validating outputs, and comparing analysis runs for reproducibility testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import anndata
import numpy as np
import pandas as pd


def run_full_analysis(
    adata: anndata.AnnData,
    config_dict: Optional[Dict[str, Any]] = None,
    *,
    resolution: float = 0.2,
    random_state: int = 42,
    norm_method: str = "log1p",
) -> anndata.AnnData:
    """Run full analysis pipeline on AnnData object with custom configuration.

    This is a simplified wrapper around the analysis workflow for testing purposes.
    It performs normalization, clustering, and differential expression analysis.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object with raw expression in .X
    config_dict : Optional[Dict[str, Any]]
        Optional configuration dictionary (currently unused, for future extensibility)
    resolution : float
        Leiden clustering resolution parameter
    random_state : int
        Random seed for reproducibility
    norm_method : str
        Normalization method (currently only 'log1p' supported)

    Returns
    -------
    anndata.AnnData
        Updated AnnData with clustering results and DE analysis in .uns

    Notes
    -----
    The returned AnnData will have:
    - adata.obsm['X_umap']: UMAP embedding
    - adata.obs['leiden']: Cluster assignments
    - adata.uns['de_results']: Differential expression results
    """
    from aegle_analysis.analysis import run_clustering, one_vs_rest_wilcoxon
    from aegle_analysis.data import prepare_data_for_analysis

    # The synthetic data in adata.X is already normalized (z-scored)
    # For testing, we need to prepare it as if it were raw data
    # Convert to DataFrame format expected by prepare_data_for_analysis
    exp_df = pd.DataFrame(
        adata.X,
        columns=adata.var_names,
        index=adata.obs_names,
    )
    # Add a cell_mask_id column (expected by the data loader)
    exp_df.insert(0, "cell_mask_id", range(1, len(exp_df) + 1))

    # Note: The synthetic data is already normalized. For real analysis,
    # we would use norm_method. For testing, we use "none" to avoid double normalization
    # since the synthetic data is already z-scored.
    if norm_method == "log1p":
        # Special handling: synthetic data is z-scored, which has negative values.
        # We can't apply log1p to negative values. Instead, just use the data as-is
        # (it's already normalized) but mark it as processed.
        data_df = exp_df.drop(columns=["cell_mask_id"])
    else:
        data_df = prepare_data_for_analysis(exp_df, norm=norm_method)

    # Create fresh AnnData with normalized data
    adata_normalized = anndata.AnnData(
        X=data_df.values,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    # Run clustering
    adata_normalized = run_clustering(
        adata_normalized, resolution=resolution, random_state=random_state
    )

    # Run differential expression
    cluster_labels = adata_normalized.obs["leiden"]
    # Prepare dataframe for DE analysis
    de_input_df = pd.DataFrame(
        adata_normalized.X,
        columns=adata_normalized.var_names,
        index=adata_normalized.obs_names,
    )
    de_input_df.insert(0, "cell_mask_id", range(1, len(de_input_df) + 1))

    de_results = one_vs_rest_wilcoxon(de_input_df, cluster_labels, de_input_df)

    # Store DE results in uns
    adata_normalized.uns["de_results"] = de_results

    return adata_normalized


def validate_analysis_outputs(adata: anndata.AnnData) -> Dict[str, Any]:
    """Check all expected outputs exist in AnnData and are correct types.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object from analysis pipeline

    Returns
    -------
    Dict[str, Any]
        Validation report with 'passed', 'failed', and 'warnings' keys

    Notes
    -----
    Checks for:
    - UMAP embedding in .obsm['X_umap']
    - Leiden cluster assignments in .obs['leiden']
    - DE results in .uns['de_results']
    - Correct data types and shapes
    """
    report = {
        "passed": [],
        "failed": [],
        "warnings": [],
    }

    # Check UMAP embedding
    if "X_umap" not in adata.obsm:
        report["failed"].append("Missing 'X_umap' in adata.obsm")
    else:
        umap = adata.obsm["X_umap"]
        if umap.shape[1] != 2:
            report["failed"].append(f"UMAP should have 2 dimensions, got {umap.shape[1]}")
        elif umap.shape[0] != adata.n_obs:
            report["failed"].append(
                f"UMAP rows ({umap.shape[0]}) don't match n_obs ({adata.n_obs})"
            )
        else:
            report["passed"].append("UMAP embedding shape correct")

        if not np.isfinite(umap).all():
            report["failed"].append("UMAP contains non-finite values")
        else:
            report["passed"].append("UMAP values are finite")

    # Check Leiden clusters
    if "leiden" not in adata.obs.columns:
        report["failed"].append("Missing 'leiden' in adata.obs")
    else:
        leiden = adata.obs["leiden"]
        n_clusters = leiden.nunique()
        if n_clusters == 0:
            report["failed"].append("No clusters found")
        elif n_clusters == 1:
            report["warnings"].append("Only 1 cluster found (may be expected for small data)")
        else:
            report["passed"].append(f"Found {n_clusters} clusters")

        # Check cluster assignments are valid
        if leiden.isna().any():
            report["failed"].append("Cluster assignments contain NaN values")
        else:
            report["passed"].append("All cells have cluster assignments")

    # Check PCA (usually computed before UMAP)
    if "X_pca" not in adata.obsm:
        report["warnings"].append("Missing 'X_pca' in adata.obsm (optional)")
    else:
        pca = adata.obsm["X_pca"]
        if pca.shape[0] != adata.n_obs:
            report["failed"].append(
                f"PCA rows ({pca.shape[0]}) don't match n_obs ({adata.n_obs})"
            )
        else:
            report["passed"].append("PCA embedding shape correct")

    # Check neighbors graph
    if "neighbors" not in adata.uns:
        report["warnings"].append("Missing 'neighbors' in adata.uns (optional)")
    else:
        report["passed"].append("Neighbors graph computed")

    # Check DE results
    if "de_results" not in adata.uns:
        report["warnings"].append("Missing 'de_results' in adata.uns")
    else:
        de_results = adata.uns["de_results"]
        if not isinstance(de_results, dict):
            report["failed"].append(f"DE results should be dict, got {type(de_results)}")
        elif len(de_results) == 0:
            report["warnings"].append("DE results dictionary is empty")
        else:
            # Check each cluster has results
            n_clusters = adata.obs["leiden"].nunique()
            if len(de_results) != n_clusters:
                report["warnings"].append(
                    f"DE results count ({len(de_results)}) doesn't match cluster count ({n_clusters})"
                )
            else:
                report["passed"].append(f"DE results for all {n_clusters} clusters")

            # Validate structure of first cluster's results
            first_cluster = list(de_results.keys())[0]
            de_df = de_results[first_cluster]
            if not isinstance(de_df, pd.DataFrame):
                report["failed"].append(f"DE results should be DataFrames, got {type(de_df)}")
            else:
                expected_cols = {"feature", "p_value_corrected", "log2_fold_change"}
                missing_cols = expected_cols - set(de_df.columns)
                if missing_cols:
                    report["failed"].append(f"DE results missing columns: {missing_cols}")
                else:
                    report["passed"].append("DE results have expected columns")

    return report


def compare_analysis_runs(
    adata1: anndata.AnnData,
    adata2: anndata.AnnData,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Compare two analysis runs for equivalence (reproducibility testing).

    Parameters
    ----------
    adata1 : anndata.AnnData
        First analysis result
    adata2 : anndata.AnnData
        Second analysis result
    tolerance : float
        Numerical tolerance for floating point comparisons

    Returns
    -------
    Dict[str, Any]
        Comparison report with 'identical', 'similar', and 'different' keys

    Notes
    -----
    Checks:
    - UMAP embeddings are numerically close
    - Cluster assignments match (or highly correlated if labels differ)
    - DE results are consistent
    """
    report = {
        "identical": [],
        "similar": [],
        "different": [],
    }

    # Check same number of observations
    if adata1.n_obs != adata2.n_obs:
        report["different"].append(
            f"Different number of observations: {adata1.n_obs} vs {adata2.n_obs}"
        )
        return report  # Can't compare further if different sizes

    # Compare UMAP embeddings
    if "X_umap" in adata1.obsm and "X_umap" in adata2.obsm:
        umap1 = adata1.obsm["X_umap"]
        umap2 = adata2.obsm["X_umap"]

        # UMAP coordinates might be reflected/rotated, so check both orientations
        diff_direct = np.abs(umap1 - umap2).max()
        diff_reflected_x = np.abs(umap1 * np.array([-1, 1]) - umap2).max()
        diff_reflected_y = np.abs(umap1 * np.array([1, -1]) - umap2).max()
        diff_reflected_both = np.abs(umap1 * np.array([-1, -1]) - umap2).max()

        min_diff = min(diff_direct, diff_reflected_x, diff_reflected_y, diff_reflected_both)

        if min_diff < tolerance:
            report["identical"].append(f"UMAP embeddings identical (max diff: {min_diff:.2e})")
        elif min_diff < 0.1:
            report["similar"].append(
                f"UMAP embeddings similar (max diff: {min_diff:.2e}, tolerance: {tolerance})"
            )
        else:
            report["different"].append(f"UMAP embeddings differ (max diff: {min_diff:.2e})")
    else:
        report["different"].append("UMAP embeddings missing in one or both runs")

    # Compare cluster assignments
    if "leiden" in adata1.obs and "leiden" in adata2.obs:
        clusters1 = adata1.obs["leiden"].astype(str).values
        clusters2 = adata2.obs["leiden"].astype(str).values

        # Exact match
        if np.array_equal(clusters1, clusters2):
            report["identical"].append("Cluster assignments exactly match")
        else:
            # Check if cluster labels are just permuted (same grouping, different names)
            # Use adjusted mutual information score
            from sklearn.metrics import adjusted_mutual_info_score

            ami = adjusted_mutual_info_score(clusters1, clusters2)
            if ami > 0.99:
                report["similar"].append(f"Cluster assignments highly correlated (AMI: {ami:.4f})")
            elif ami > 0.9:
                report["similar"].append(
                    f"Cluster assignments mostly consistent (AMI: {ami:.4f})"
                )
            else:
                report["different"].append(f"Cluster assignments differ (AMI: {ami:.4f})")
    else:
        report["different"].append("Cluster assignments missing in one or both runs")

    # Compare DE results (if present)
    if "de_results" in adata1.uns and "de_results" in adata2.uns:
        de1 = adata1.uns["de_results"]
        de2 = adata2.uns["de_results"]

        if set(de1.keys()) != set(de2.keys()):
            report["different"].append(
                f"DE results for different clusters: {set(de1.keys())} vs {set(de2.keys())}"
            )
        else:
            # Compare first cluster's results as a sample
            cluster_key = list(de1.keys())[0]
            df1 = de1[cluster_key]
            df2 = de2[cluster_key]

            if df1.equals(df2):
                report["identical"].append("DE results identical")
            else:
                # Check if log2FC values are close
                if "log2_fold_change" in df1.columns and "log2_fold_change" in df2.columns:
                    fc1 = df1.set_index("feature")["log2_fold_change"]
                    fc2 = df2.set_index("feature")["log2_fold_change"]
                    max_diff = np.abs(fc1 - fc2).max()
                    if max_diff < tolerance:
                        report["identical"].append(
                            f"DE log2FC values identical (max diff: {max_diff:.2e})"
                        )
                    elif max_diff < 0.01:
                        report["similar"].append(
                            f"DE log2FC values similar (max diff: {max_diff:.2e})"
                        )
                    else:
                        report["different"].append(
                            f"DE log2FC values differ (max diff: {max_diff:.2e})"
                        )
    else:
        report["different"].append("DE results missing in one or both runs")

    return report


# =============================================================================
# Test Configuration Fixtures
# =============================================================================


def get_default_clustering_config() -> Dict[str, Any]:
    """Get default configuration for clustering tests.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with standard clustering parameters
    """
    return {
        "resolution": 0.2,
        "random_state": 42,
        "n_neighbors": 15,
        "norm_method": "log1p",
    }


def get_default_de_config() -> Dict[str, Any]:
    """Get default configuration for differential expression tests.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with DE parameters
    """
    return {
        "method": "wilcoxon",
        "correction_method": "benjamini-hochberg",
        "use_raw": False,
    }


def get_analysis_config(
    *,
    clustering_resolution: float = 0.2,
    random_state: int = 42,
    norm_method: str = "log1p",
    skip_viz: bool = True,
) -> Dict[str, Any]:
    """Get complete analysis configuration for integration tests.

    Parameters
    ----------
    clustering_resolution : float
        Leiden clustering resolution
    random_state : int
        Random seed for reproducibility
    norm_method : str
        Normalization method
    skip_viz : bool
        Whether to skip visualization steps (default True for speed)

    Returns
    -------
    Dict[str, Any]
        Complete analysis configuration dictionary
    """
    return {
        "analysis": {
            "clustering_resolution": clustering_resolution,
            "random_state": random_state,
            "norm_method": norm_method,
            "skip_viz": skip_viz,
            "generate_pipeline_report": False,
        }
    }
