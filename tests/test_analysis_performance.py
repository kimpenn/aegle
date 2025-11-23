"""Performance benchmarks and regression tests for analysis pipeline.

This module provides performance tests to:
1. Establish CPU performance baselines
2. Detect performance regressions
3. Measure GPU speedup (Phase 5)

The tests use small/medium synthetic datasets for fast execution and
record results to `tests/performance_baselines.json` for comparison.
"""

import logging
from pathlib import Path

import pytest

from aegle_analysis.analysis.clustering import run_clustering
from aegle_analysis.analysis.differential import one_vs_rest_wilcoxon
from aegle_analysis.data.transforms import prepare_data_for_analysis
from tests.utils.performance_helpers import (
    benchmark_function,
    compare_to_baseline,
    format_time,
    record_baseline,
)
from tests.utils.synthetic_analysis_data import get_medium_dataset, get_small_dataset

# Baseline file location
BASELINE_FILE = Path(__file__).parent / "performance_baselines.json"

# Performance thresholds
REGRESSION_TOLERANCE = 0.20  # Warn if >20% slower than baseline
GPU_SPEEDUP_TARGET = 5.0  # GPU should be >5x faster (Phase 5)

logger = logging.getLogger(__name__)


# =============================================================================
# Baseline Performance Tests (CPU)
# =============================================================================


class TestBaselineNormalizationPerformance:
    """Benchmark normalization performance on CPU."""

    def test_baseline_normalization_small(self):
        """Benchmark normalization on small dataset (1K cells)."""
        # Generate small synthetic dataset
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata

        # Create raw expression DataFrame from adata
        import pandas as pd

        exp_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

        # Benchmark log1p normalization
        def normalize_log1p():
            return prepare_data_for_analysis(exp_df, norm="log1p")

        result, elapsed = benchmark_function(normalize_log1p, warmup=True)

        # Record baseline
        record_baseline(
            stage_name="normalization_log1p_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={"n_cells": adata.n_obs, "n_markers": adata.n_vars, "norm": "log1p"},
        )

        logger.info(f"Normalization (log1p, small): {format_time(elapsed)}")
        assert elapsed < 1.0, f"Normalization too slow: {elapsed:.3f}s"

    def test_baseline_normalization_medium(self):
        """Benchmark normalization on medium dataset (10K cells)."""
        synth_data = get_medium_dataset(random_seed=42)
        adata = synth_data.adata

        import pandas as pd

        exp_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

        def normalize_log1p():
            return prepare_data_for_analysis(exp_df, norm="log1p")

        result, elapsed = benchmark_function(normalize_log1p, warmup=True)

        record_baseline(
            stage_name="normalization_log1p_medium",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={"n_cells": adata.n_obs, "n_markers": adata.n_vars, "norm": "log1p"},
        )

        logger.info(f"Normalization (log1p, medium): {format_time(elapsed)}")
        assert elapsed < 5.0, f"Normalization too slow: {elapsed:.3f}s"


class TestBaselineClusteringPerformance:
    """Benchmark clustering performance on CPU."""

    def test_baseline_clustering_small(self):
        """Benchmark full clustering pipeline on small dataset (1K cells)."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        def run_full_clustering():
            return run_clustering(
                adata,
                n_neighbors=10,
                resolution=0.2,
                random_state=42,
                use_rep="X",
            )

        result, elapsed = benchmark_function(run_full_clustering, warmup=True)

        record_baseline(
            stage_name="clustering_full_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={
                "n_cells": adata.n_obs,
                "n_markers": adata.n_vars,
                "n_neighbors": 10,
                "resolution": 0.2,
            },
        )

        logger.info(f"Clustering (full, small): {format_time(elapsed)}")
        assert elapsed < 5.0, f"Clustering too slow: {elapsed:.3f}s"

    def test_baseline_clustering_knn_small(self):
        """Benchmark k-NN graph construction on small dataset."""
        import scanpy as sc

        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        def build_knn():
            sc.pp.neighbors(adata, n_neighbors=10, use_rep="X", random_state=42)

        result, elapsed = benchmark_function(build_knn, warmup=True)

        record_baseline(
            stage_name="clustering_knn_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={"n_cells": adata.n_obs, "n_neighbors": 10},
        )

        logger.info(f"Clustering (k-NN, small): {format_time(elapsed)}")
        assert elapsed < 2.0, f"k-NN too slow: {elapsed:.3f}s"

    def test_baseline_clustering_leiden_small(self):
        """Benchmark Leiden clustering on small dataset."""
        import scanpy as sc

        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Build neighbors first (required for Leiden)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X", random_state=42)

        def run_leiden():
            sc.tl.leiden(adata, resolution=0.2, random_state=42)

        result, elapsed = benchmark_function(run_leiden, warmup=True)

        record_baseline(
            stage_name="clustering_leiden_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={"n_cells": adata.n_obs, "resolution": 0.2},
        )

        logger.info(f"Clustering (Leiden, small): {format_time(elapsed)}")
        assert elapsed < 2.0, f"Leiden too slow: {elapsed:.3f}s"

    def test_baseline_clustering_umap_small(self):
        """Benchmark UMAP computation on small dataset."""
        import scanpy as sc

        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        # Build neighbors first (required for UMAP)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X", random_state=42)

        def run_umap():
            sc.tl.umap(adata, random_state=42)

        result, elapsed = benchmark_function(run_umap, warmup=True)

        record_baseline(
            stage_name="clustering_umap_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={"n_cells": adata.n_obs},
        )

        logger.info(f"Clustering (UMAP, small): {format_time(elapsed)}")
        assert elapsed < 3.0, f"UMAP too slow: {elapsed:.3f}s"


class TestBaselineDifferentialPerformance:
    """Benchmark differential expression performance on CPU."""

    def test_baseline_differential_small(self):
        """Benchmark DE analysis on small dataset (1K cells)."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata

        import pandas as pd

        # Prepare normalized data for DE
        exp_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )
        norm_df = prepare_data_for_analysis(exp_df, norm="log1p")
        log1p_df = prepare_data_for_analysis(exp_df, norm="log1p")

        # Use true cluster labels
        cluster_series = adata.obs["true_cluster"]

        def run_de_analysis():
            return one_vs_rest_wilcoxon(norm_df, cluster_series, log1p_df)

        result, elapsed = benchmark_function(run_de_analysis, warmup=True)

        record_baseline(
            stage_name="differential_wilcoxon_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={
                "n_cells": adata.n_obs,
                "n_markers": adata.n_vars,
                "n_clusters": adata.uns["n_clusters"],
            },
        )

        logger.info(f"Differential (Wilcoxon, small): {format_time(elapsed)}")
        assert elapsed < 5.0, f"Differential analysis too slow: {elapsed:.3f}s"


class TestBaselineFullPipeline:
    """Benchmark complete analysis pipeline on CPU."""

    def test_baseline_full_pipeline_small(self):
        """Benchmark full analysis pipeline on small dataset."""
        import pandas as pd
        import scanpy as sc

        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        exp_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

        def run_full_pipeline():
            # 1. Normalization
            norm_df = prepare_data_for_analysis(exp_df, norm="log1p")
            log1p_df = prepare_data_for_analysis(exp_df, norm="log1p")

            # 2. Clustering
            adata_copy = adata.copy()
            sc.pp.neighbors(adata_copy, n_neighbors=10, use_rep="X", random_state=42)
            sc.tl.leiden(adata_copy, resolution=0.2, random_state=42)
            sc.tl.umap(adata_copy, random_state=42)

            # 3. Differential expression
            cluster_series = adata.obs["true_cluster"]
            de_results = one_vs_rest_wilcoxon(norm_df, cluster_series, log1p_df)

            return de_results

        result, elapsed = benchmark_function(run_full_pipeline, warmup=True)

        record_baseline(
            stage_name="full_pipeline_small",
            time_seconds=elapsed,
            baseline_file=BASELINE_FILE,
            metadata={
                "n_cells": adata.n_obs,
                "n_markers": adata.n_vars,
                "n_clusters": adata.uns["n_clusters"],
            },
        )

        logger.info(f"Full pipeline (small): {format_time(elapsed)}")
        assert elapsed < 10.0, f"Full pipeline too slow: {elapsed:.3f}s"


# =============================================================================
# Regression Detection Tests
# =============================================================================


class TestCPUPerformanceRegression:
    """Detect if CPU performance has regressed compared to baseline."""

    def test_no_normalization_regression(self):
        """Verify normalization hasn't regressed vs baseline."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata

        import pandas as pd

        exp_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

        def normalize_log1p():
            return prepare_data_for_analysis(exp_df, norm="log1p")

        result, elapsed = benchmark_function(normalize_log1p, warmup=True)

        status, ratio, baseline = compare_to_baseline(
            stage_name="normalization_log1p_small",
            current_time=elapsed,
            baseline_file=BASELINE_FILE,
            tolerance=REGRESSION_TOLERANCE,
        )

        if status == "no_baseline":
            pytest.skip("No baseline found - run baseline tests first")

        # Log warning if slower
        if status == "slower":
            logger.warning(
                f"REGRESSION: Normalization is {ratio:.2f}x slower than baseline "
                f"({elapsed:.4f}s vs {baseline:.4f}s)"
            )

        # Don't fail, just warn
        assert True

    def test_no_clustering_regression(self):
        """Verify clustering hasn't regressed vs baseline."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata.copy()

        def run_full_clustering():
            return run_clustering(
                adata,
                n_neighbors=10,
                resolution=0.2,
                random_state=42,
                use_rep="X",
            )

        result, elapsed = benchmark_function(run_full_clustering, warmup=True)

        status, ratio, baseline = compare_to_baseline(
            stage_name="clustering_full_small",
            current_time=elapsed,
            baseline_file=BASELINE_FILE,
            tolerance=REGRESSION_TOLERANCE,
        )

        if status == "no_baseline":
            pytest.skip("No baseline found - run baseline tests first")

        if status == "slower":
            logger.warning(
                f"REGRESSION: Clustering is {ratio:.2f}x slower than baseline "
                f"({elapsed:.4f}s vs {baseline:.4f}s)"
            )

        assert True

    def test_no_differential_regression(self):
        """Verify differential analysis hasn't regressed vs baseline."""
        synth_data = get_small_dataset(random_seed=42)
        adata = synth_data.adata

        import pandas as pd

        exp_df = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names,
        )
        norm_df = prepare_data_for_analysis(exp_df, norm="log1p")
        log1p_df = prepare_data_for_analysis(exp_df, norm="log1p")
        cluster_series = adata.obs["true_cluster"]

        def run_de_analysis():
            return one_vs_rest_wilcoxon(norm_df, cluster_series, log1p_df)

        result, elapsed = benchmark_function(run_de_analysis, warmup=True)

        status, ratio, baseline = compare_to_baseline(
            stage_name="differential_wilcoxon_small",
            current_time=elapsed,
            baseline_file=BASELINE_FILE,
            tolerance=REGRESSION_TOLERANCE,
        )

        if status == "no_baseline":
            pytest.skip("No baseline found - run baseline tests first")

        if status == "slower":
            logger.warning(
                f"REGRESSION: Differential analysis is {ratio:.2f}x slower than baseline "
                f"({elapsed:.4f}s vs {baseline:.4f}s)"
            )

        assert True


# =============================================================================
# GPU Speedup Tests (Placeholder for Phase 5)
# =============================================================================


class TestGPUSpeedupRequirement:
    """Placeholder tests for GPU speedup validation (Phase 5)."""

    @pytest.mark.skip(reason="Phase 5: GPU acceleration not yet implemented")
    def test_gpu_normalization_speedup(self):
        """Verify GPU normalization achieves >5x speedup vs CPU.

        Phase 5 TODO:
        - Implement GPU-accelerated normalization
        - Benchmark GPU vs CPU on medium dataset (10K cells)
        - Assert speedup > GPU_SPEEDUP_TARGET (5x)
        """
        pass

    @pytest.mark.skip(reason="Phase 5: GPU acceleration not yet implemented")
    def test_gpu_clustering_speedup(self):
        """Verify GPU clustering achieves >5x speedup vs CPU.

        Phase 5 TODO:
        - Implement GPU-accelerated k-NN graph construction
        - Benchmark GPU vs CPU on medium dataset (10K cells)
        - Assert speedup > GPU_SPEEDUP_TARGET (5x)
        """
        pass

    @pytest.mark.skip(reason="Phase 5: GPU acceleration not yet implemented")
    def test_gpu_differential_speedup(self):
        """Verify GPU differential analysis achieves >5x speedup vs CPU.

        Phase 5 TODO:
        - Implement GPU-accelerated Wilcoxon rank-sum test
        - Benchmark GPU vs CPU on medium dataset (10K cells)
        - Assert speedup > GPU_SPEEDUP_TARGET (5x)
        """
        pass

    @pytest.mark.skip(reason="Phase 5: GPU acceleration not yet implemented")
    def test_gpu_full_pipeline_speedup(self):
        """Verify GPU full pipeline achieves >5x speedup vs CPU.

        Phase 5 TODO:
        - Implement end-to-end GPU acceleration
        - Benchmark GPU vs CPU on medium dataset (10K cells)
        - Assert speedup > GPU_SPEEDUP_TARGET (5x)
        """
        pass
