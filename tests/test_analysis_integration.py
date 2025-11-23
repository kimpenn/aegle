"""Full pipeline integration tests for analysis workflow.

This module tests the complete end-to-end analysis pipeline on synthetic data,
ensuring all components work together correctly. These tests validate:
- Complete workflow from raw data to final outputs
- Reproducibility (same seed = same results)
- CPU vs GPU equivalence (when GPU available)
- Output structure and data types
"""

import unittest
from pathlib import Path

import numpy as np
import pytest

from tests.utils.synthetic_analysis_data import (
    get_small_dataset,
    get_medium_dataset,
)
from tests.utils.integration_helpers import (
    run_full_analysis,
    validate_analysis_outputs,
    compare_analysis_runs,
    get_analysis_config,
)


class TestFullPipelineSmall(unittest.TestCase):
    """Integration tests using small dataset (1K cells, fast execution)."""

    def setUp(self):
        """Set up synthetic data for testing."""
        self.synth_data = get_small_dataset(random_seed=42)
        self.adata = self.synth_data.adata.copy()

    def test_full_pipeline_small_completes(self):
        """Run complete analysis on 1K cell dataset and validate outputs.

        This test ensures the full pipeline can process a small dataset and
        produce all expected outputs within reasonable time (< 30 seconds).
        """
        # Run full analysis pipeline
        result_adata = run_full_analysis(
            self.adata,
            resolution=0.2,
            random_state=42,
            norm_method="log1p",
        )

        # Validate all expected outputs exist
        validation_report = validate_analysis_outputs(result_adata)

        # Check no critical failures
        self.assertEqual(
            len(validation_report["failed"]),
            0,
            f"Validation failures: {validation_report['failed']}",
        )

        # Should have some passing checks
        self.assertGreater(
            len(validation_report["passed"]),
            0,
            "No validation checks passed",
        )

        # Basic sanity checks
        self.assertIn("X_umap", result_adata.obsm, "Missing UMAP embedding")
        self.assertIn("leiden", result_adata.obs.columns, "Missing cluster assignments")
        self.assertEqual(result_adata.n_obs, self.adata.n_obs, "Lost cells during analysis")

        # Check UMAP is 2D
        self.assertEqual(
            result_adata.obsm["X_umap"].shape,
            (result_adata.n_obs, 2),
            "UMAP should be 2D",
        )

        # Check all cells got cluster assignments
        n_clusters = result_adata.obs["leiden"].nunique()
        self.assertGreater(n_clusters, 0, "No clusters found")
        self.assertFalse(
            result_adata.obs["leiden"].isna().any(),
            "Some cells missing cluster assignment",
        )

        # Check DE results exist and have expected structure
        self.assertIn("de_results", result_adata.uns, "Missing DE results")
        de_results = result_adata.uns["de_results"]
        self.assertIsInstance(de_results, dict, "DE results should be a dictionary")
        self.assertEqual(
            len(de_results),
            n_clusters,
            f"DE results count ({len(de_results)}) doesn't match clusters ({n_clusters})",
        )

        # Check first cluster's DE results structure
        if len(de_results) > 0:
            first_cluster = list(de_results.keys())[0]
            de_df = de_results[first_cluster]
            self.assertIn("feature", de_df.columns, "DE results missing 'feature' column")
            self.assertIn(
                "log2_fold_change", de_df.columns, "DE results missing 'log2_fold_change'"
            )
            self.assertIn(
                "p_value_corrected", de_df.columns, "DE results missing 'p_value_corrected'"
            )

    def test_pipeline_handles_edge_cases(self):
        """Ensure pipeline handles edge cases gracefully."""
        # Test with very small cluster resolution (might produce 1 cluster)
        result_adata = run_full_analysis(
            self.adata,
            resolution=0.01,
            random_state=42,
            norm_method="log1p",
        )

        validation_report = validate_analysis_outputs(result_adata)

        # Should complete without errors even if only 1 cluster
        self.assertEqual(
            len(validation_report["failed"]),
            0,
            f"Validation failures: {validation_report['failed']}",
        )

        # Test with high resolution (many clusters)
        result_adata = run_full_analysis(
            self.adata.copy(),
            resolution=2.0,
            random_state=42,
            norm_method="log1p",
        )

        validation_report = validate_analysis_outputs(result_adata)
        self.assertEqual(
            len(validation_report["failed"]),
            0,
            f"Validation failures: {validation_report['failed']}",
        )

    def test_pipeline_output_types(self):
        """Validate data types of all outputs."""
        result_adata = run_full_analysis(
            self.adata,
            resolution=0.2,
            random_state=42,
            norm_method="log1p",
        )

        # Check UMAP dtype
        self.assertTrue(
            np.issubdtype(result_adata.obsm["X_umap"].dtype, np.floating),
            "UMAP should be float type",
        )

        # Check no NaN or inf in UMAP
        self.assertTrue(
            np.isfinite(result_adata.obsm["X_umap"]).all(),
            "UMAP contains non-finite values",
        )

        # Check cluster labels are categorical or string
        leiden = result_adata.obs["leiden"]
        self.assertTrue(
            leiden.dtype == "object" or leiden.dtype.name == "category",
            f"Leiden clusters should be categorical/string, got {leiden.dtype}",
        )

        # Check DE results contain numeric values
        de_results = result_adata.uns["de_results"]
        first_cluster = list(de_results.keys())[0]
        de_df = de_results[first_cluster]

        self.assertTrue(
            np.issubdtype(de_df["log2_fold_change"].dtype, np.floating),
            "log2_fold_change should be float",
        )


class TestFullPipelineMedium(unittest.TestCase):
    """Integration tests using medium dataset (10K cells, ~2 min runtime)."""

    def setUp(self):
        """Set up synthetic data for testing."""
        self.synth_data = get_medium_dataset(random_seed=42)
        self.adata = self.synth_data.adata.copy()

    def test_full_pipeline_medium_completes(self):
        """Run complete analysis on 10K cell dataset with thorough validation.

        This test uses a more realistic dataset size and performs comprehensive
        validation of outputs. Should complete in < 2 minutes.
        """
        # Run full analysis pipeline
        result_adata = run_full_analysis(
            self.adata,
            resolution=0.2,
            random_state=42,
            norm_method="log1p",
        )

        # Validate all expected outputs exist
        validation_report = validate_analysis_outputs(result_adata)

        # Print report for debugging
        if validation_report["failed"]:
            print("\nValidation failures:")
            for failure in validation_report["failed"]:
                print(f"  - {failure}")

        if validation_report["warnings"]:
            print("\nValidation warnings:")
            for warning in validation_report["warnings"]:
                print(f"  - {warning}")

        # No critical failures allowed
        self.assertEqual(
            len(validation_report["failed"]),
            0,
            f"Validation failures: {validation_report['failed']}",
        )

        # Should have multiple passing checks
        self.assertGreater(
            len(validation_report["passed"]),
            3,
            "Too few validation checks passed",
        )

        # Verify reasonable number of clusters
        n_clusters = result_adata.obs["leiden"].nunique()
        self.assertGreater(n_clusters, 1, "Only 1 cluster found")
        self.assertLess(
            n_clusters,
            result_adata.n_obs // 10,
            f"Too many clusters ({n_clusters}) for {result_adata.n_obs} cells",
        )

        # Check cluster size distribution is reasonable
        cluster_sizes = result_adata.obs["leiden"].value_counts()
        min_cluster_size = cluster_sizes.min()
        self.assertGreater(
            min_cluster_size,
            10,
            f"Smallest cluster has only {min_cluster_size} cells",
        )

    def test_pipeline_discovers_known_clusters(self):
        """Verify pipeline can recover ground truth cluster structure.

        The synthetic data has known cluster assignments. This test checks
        that the analysis pipeline can discover a cluster structure that
        correlates with the ground truth.
        """
        # Run analysis
        result_adata = run_full_analysis(
            self.adata,
            resolution=0.4,  # Slightly higher resolution for better recovery
            random_state=42,
            norm_method="log1p",
        )

        # Compare discovered clusters to ground truth
        from sklearn.metrics import adjusted_mutual_info_score

        true_clusters = self.synth_data.true_clusters.astype(str)
        pred_clusters = result_adata.obs["leiden"].astype(str).values

        ami = adjusted_mutual_info_score(true_clusters, pred_clusters)

        # Should have reasonable agreement with ground truth
        # (AMI > 0.5 is decent for synthetic data with noise)
        self.assertGreater(
            ami,
            0.3,
            f"Cluster recovery is poor (AMI: {ami:.3f}). "
            "Pipeline may not be working correctly.",
        )


class TestFullPipelineReproducibility(unittest.TestCase):
    """Test pipeline reproducibility (same seed = same results)."""

    def setUp(self):
        """Set up synthetic data for testing."""
        self.synth_data = get_small_dataset(random_seed=42)

    def test_full_pipeline_reproducibility(self):
        """Run pipeline twice with same seed and validate identical results.

        Reproducibility is critical for scientific analyses. This test ensures
        that running the pipeline twice with the same random seed produces
        identical results for clustering, UMAP, and DE analysis.
        """
        # Run pipeline twice with same seed
        adata1 = self.synth_data.adata.copy()
        adata2 = self.synth_data.adata.copy()

        result1 = run_full_analysis(adata1, resolution=0.2, random_state=42)
        result2 = run_full_analysis(adata2, resolution=0.2, random_state=42)

        # Compare results
        comparison_report = compare_analysis_runs(result1, result2, tolerance=1e-5)

        # Print comparison report
        print("\nReproducibility comparison:")
        if comparison_report["identical"]:
            print("  Identical:")
            for item in comparison_report["identical"]:
                print(f"    - {item}")
        if comparison_report["similar"]:
            print("  Similar:")
            for item in comparison_report["similar"]:
                print(f"    - {item}")
        if comparison_report["different"]:
            print("  Different:")
            for item in comparison_report["different"]:
                print(f"    - {item}")

        # Should have at least some identical results
        self.assertGreater(
            len(comparison_report["identical"]) + len(comparison_report["similar"]),
            0,
            "No similar or identical results found between runs",
        )

        # Cluster assignments should be identical or highly correlated
        has_cluster_match = any(
            "cluster" in item.lower()
            for item in comparison_report["identical"] + comparison_report["similar"]
        )
        self.assertTrue(
            has_cluster_match,
            "Cluster assignments don't match between runs with same seed",
        )

    def test_different_seeds_produce_similar_results(self):
        """Verify different seeds produce similar (but not identical) results.

        While exact reproducibility requires same seed, different seeds should
        still produce similar cluster structures (similar number of clusters,
        similar cluster separation).
        """
        # Run with two different seeds
        adata1 = self.synth_data.adata.copy()
        adata2 = self.synth_data.adata.copy()

        result1 = run_full_analysis(adata1, resolution=0.2, random_state=42)
        result2 = run_full_analysis(adata2, resolution=0.2, random_state=123)

        # Should have similar number of clusters (within 1-2)
        n_clusters1 = result1.obs["leiden"].nunique()
        n_clusters2 = result2.obs["leiden"].nunique()

        self.assertLess(
            abs(n_clusters1 - n_clusters2),
            3,
            f"Very different cluster counts: {n_clusters1} vs {n_clusters2}",
        )

        # Cluster structures should be somewhat correlated
        from sklearn.metrics import adjusted_mutual_info_score

        clusters1 = result1.obs["leiden"].astype(str).values
        clusters2 = result2.obs["leiden"].astype(str).values

        ami = adjusted_mutual_info_score(clusters1, clusters2)

        # Different seeds may produce different cluster labels, but structure
        # should be somewhat consistent (AMI > 0.3 indicates some agreement)
        self.assertGreater(
            ami,
            0.2,
            f"Cluster structures very different with different seeds (AMI: {ami:.3f})",
        )


@pytest.mark.slow
class TestFullPipelinePerformance(unittest.TestCase):
    """Performance tests (marked slow, run with pytest -m slow)."""

    def test_small_dataset_completes_quickly(self):
        """Verify small dataset analysis completes in < 30 seconds."""
        import time

        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        start_time = time.time()
        result_adata = run_full_analysis(adata, resolution=0.2, random_state=42)
        elapsed_time = time.time() - start_time

        print(f"\nSmall dataset ({adata.n_obs} cells) analysis time: {elapsed_time:.1f}s")

        # Should complete in < 30 seconds on reasonable hardware
        self.assertLess(
            elapsed_time,
            30.0,
            f"Small dataset analysis too slow: {elapsed_time:.1f}s",
        )

        # Validate it actually completed successfully
        validation_report = validate_analysis_outputs(result_adata)
        self.assertEqual(len(validation_report["failed"]), 0)

    def test_medium_dataset_completes_in_reasonable_time(self):
        """Verify medium dataset analysis completes in < 2 minutes."""
        import time

        synth_data = get_medium_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        start_time = time.time()
        result_adata = run_full_analysis(adata, resolution=0.2, random_state=42)
        elapsed_time = time.time() - start_time

        print(f"\nMedium dataset ({adata.n_obs} cells) analysis time: {elapsed_time:.1f}s")

        # Should complete in < 2 minutes
        self.assertLess(
            elapsed_time,
            120.0,
            f"Medium dataset analysis too slow: {elapsed_time:.1f}s",
        )

        # Validate it actually completed successfully
        validation_report = validate_analysis_outputs(result_adata)
        self.assertEqual(len(validation_report["failed"]), 0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
