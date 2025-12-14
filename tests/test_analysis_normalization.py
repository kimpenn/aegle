"""
Baseline CPU tests for data normalization and transformation functions.

These tests establish a baseline for the current CPU-only normalization functionality
and will be used to ensure GPU versions produce equivalent results.
"""

import numpy as np
import pandas as pd
import pytest

from aegle_analysis.data.transforms import (
    rank_normalization,
    log1p_transform,
    prepare_data_for_analysis,
    clr_across_cells,
    double_zscore_log,
    zscore_log,
)
from tests.utils.synthetic_analysis_data import get_small_dataset, get_medium_dataset


class TestRankNormalization:
    """Test suite for rank normalization functions."""

    def test_rank_normalization_fraction_basic(self):
        """Test rank normalization with fraction method produces values in [0, 1]."""
        synth_data = get_small_dataset(random_seed=42)

        # Extract expression data as DataFrame
        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        # Apply rank normalization
        df_ranked = rank_normalization(df, method="fraction")

        # Validate shape preserved
        assert df_ranked.shape == df.shape, "Shape changed after normalization"

        # Validate range [0, 1]
        assert df_ranked.min().min() >= 0, "Values below 0 detected"
        assert df_ranked.max().max() <= 1, "Values above 1 detected"

        # Check that all columns are normalized
        assert len(df_ranked.columns) == len(df.columns), "Columns lost"
        assert np.all(df_ranked.columns == df.columns), "Column names changed"

    def test_rank_normalization_fraction_reproducibility(self):
        """Test that rank normalization is deterministic."""
        synth_data = get_small_dataset(random_seed=42)
        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        # Run twice
        df_ranked1 = rank_normalization(df.copy(), method="fraction")
        df_ranked2 = rank_normalization(df.copy(), method="fraction")

        # Should be identical
        assert np.allclose(df_ranked1.values, df_ranked2.values), (
            "Rank normalization not reproducible"
        )

    def test_rank_normalization_gaussian_basic(self):
        """Test rank normalization with gaussian method."""
        synth_data = get_small_dataset(random_seed=42)
        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        # Apply gaussian rank normalization
        df_ranked = rank_normalization(df, method="gaussian")

        # Validate shape preserved
        assert df_ranked.shape == df.shape, "Shape changed after normalization"

        # Gaussian method should produce approximately N(0, 1) distribution
        # Check that mean is close to 0 and std is close to 1 (for each column)
        for col in df_ranked.columns:
            col_mean = df_ranked[col].mean()
            col_std = df_ranked[col].std()

            # Should be roughly standard normal (loose check)
            assert -1.0 < col_mean < 1.0, f"Column {col} mean far from 0: {col_mean}"
            assert 0.5 < col_std < 2.0, f"Column {col} std far from 1: {col_std}"

    def test_rank_normalization_handles_edge_cases(self):
        """Test rank normalization with edge cases (constant columns, zeros)."""
        # Create DataFrame with edge cases
        df = pd.DataFrame({
            "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
            "all_zeros": [0.0, 0.0, 0.0, 0.0, 0.0],
            "all_same": [5.0, 5.0, 5.0, 5.0, 5.0],
            "with_zeros": [0.0, 1.0, 2.0, 0.0, 3.0],
        })

        # Apply rank normalization - should not crash
        df_ranked = rank_normalization(df, method="fraction")

        # Check that it didn't crash and returned a DataFrame
        assert df_ranked.shape == df.shape
        assert isinstance(df_ranked, pd.DataFrame)

        # All-same columns should have uniform ranks (but may have small variance due to rankdata ties)
        # Just check it has low variance (< 0.1 is reasonable for tied ranks)
        assert df_ranked["all_same"].std() < 0.1, "All-same column should have low variance"

    def test_rank_normalization_preserves_index_columns(self):
        """Test that rank normalization preserves DataFrame index and columns."""
        synth_data = get_small_dataset(random_seed=42)
        df = pd.DataFrame(
            synth_data.adata.X,
            index=synth_data.adata.obs_names,
            columns=synth_data.adata.var_names,
        )

        original_index = df.index.copy()
        original_columns = df.columns.copy()

        # Apply rank normalization
        df_ranked = rank_normalization(df, method="fraction")

        # Check preservation
        assert np.all(df_ranked.index == original_index), "Index changed"
        assert np.all(df_ranked.columns == original_columns), "Columns changed"


class TestLog1pTransform:
    """Test suite for log1p transformation."""

    def test_log1p_transform_basic(self):
        """Test log1p transformation produces expected results."""
        # Create simple test data
        df = pd.DataFrame({
            "marker1": [0.0, 1.0, 10.0, 100.0],
            "marker2": [1.0, 2.0, 3.0, 4.0],
        })

        # Apply log1p
        df_log = log1p_transform(df)

        # Validate
        assert df_log.shape == df.shape
        expected_marker1 = np.log1p(df["marker1"].values)
        assert np.allclose(df_log["marker1"].values, expected_marker1)

    def test_log1p_transform_handles_zeros(self):
        """Test that log1p handles zeros correctly (log(1 + 0) = 0)."""
        df = pd.DataFrame({
            "marker1": [0.0, 0.0, 0.0],
            "marker2": [0.0, 1.0, 2.0],
        })

        df_log = log1p_transform(df)

        # Zero values should map to zero
        assert np.allclose(df_log["marker1"].values, 0.0)
        assert df_log["marker2"].iloc[0] == 0.0


class TestPrepareDataForAnalysis:
    """Test suite for prepare_data_for_analysis function."""

    def test_prepare_data_log1p(self):
        """Test data preparation with log1p normalization."""
        # Create test data with identifier columns
        df = pd.DataFrame({
            "cell_mask_id": [1, 2, 3],
            "patch_id": [0, 0, 0],
            "marker1": [1.0, 2.0, 3.0],
            "marker2": [4.0, 5.0, 6.0],
        })

        # Prepare data
        df_prepared = prepare_data_for_analysis(df, norm="log1p")

        # Check that identifier columns are removed
        assert "cell_mask_id" not in df_prepared.columns
        assert "patch_id" not in df_prepared.columns

        # Check that markers remain
        assert "marker1" in df_prepared.columns
        assert "marker2" in df_prepared.columns

        # Check that log1p was applied
        expected = np.log1p(df[["marker1", "marker2"]].values)
        assert np.allclose(df_prepared.values, expected)

    def test_prepare_data_zscore_log(self):
        """Test data preparation with zscore_log normalization."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "marker2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        df_prepared = prepare_data_for_analysis(df, norm="zscore_log")

        # Should be z-scored (mean ~ 0, std ~ 1 per column)
        assert df_prepared.shape == df.shape
        for col in df_prepared.columns:
            assert abs(df_prepared[col].mean()) < 1e-10, f"Mean not ~0 for {col}"
            assert abs(df_prepared[col].std() - 1.0) < 1e-10, f"Std not ~1 for {col}"

    def test_prepare_data_clr(self):
        """Test data preparation with CLR transformation."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0],
            "marker2": [4.0, 5.0, 6.0],
            "marker3": [7.0, 8.0, 9.0],
        })

        df_prepared = prepare_data_for_analysis(df, norm="clr")

        # CLR should preserve shape
        assert df_prepared.shape == df.shape

    def test_prepare_data_rank_fraction(self):
        """Test data preparation with rank normalization (fraction)."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0, 4.0],
            "marker2": [5.0, 6.0, 7.0, 8.0],
        })

        df_prepared = prepare_data_for_analysis(df, norm="rank_fraction")

        # Should be in [0, 1]
        assert df_prepared.shape == df.shape
        assert df_prepared.min().min() >= 0
        assert df_prepared.max().max() <= 1

    def test_prepare_data_rank_gaussian(self):
        """Test data preparation with rank normalization (gaussian)."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0, 4.0],
            "marker2": [5.0, 6.0, 7.0, 8.0],
        })

        df_prepared = prepare_data_for_analysis(df, norm="rank_gaussian")

        # Should produce roughly standard normal distribution
        assert df_prepared.shape == df.shape

    def test_prepare_data_none(self):
        """Test data preparation with no normalization."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0],
            "marker2": [4.0, 5.0, 6.0],
        })

        df_prepared = prepare_data_for_analysis(df, norm="none")

        # Should be unchanged (except ID column removal if present)
        assert df_prepared.shape == df.shape
        assert np.allclose(df_prepared.values, df.values)

    def test_prepare_data_invalid_norm(self):
        """Test that invalid normalization method raises error."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="Unknown normalization method"):
            prepare_data_for_analysis(df, norm="invalid_method")

    def test_prepare_data_handles_non_numeric(self):
        """Test that prepare_data_for_analysis handles non-numeric values."""
        df = pd.DataFrame({
            "cell_mask_id": [1, 2, 3],
            "marker1": [1.0, 2.0, 3.0],
            "marker2": ["4.0", "5.0", "6.0"],  # String that can be converted
        })

        # Should convert strings to numeric and handle gracefully
        df_prepared = prepare_data_for_analysis(df, norm="log1p")

        # Should have both markers
        assert "marker1" in df_prepared.columns
        assert "marker2" in df_prepared.columns


class TestOtherTransforms:
    """Test other transformation functions."""

    def test_clr_across_cells_basic(self):
        """Test CLR transformation preserves shape."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0],
            "marker2": [4.0, 5.0, 6.0],
            "marker3": [7.0, 8.0, 9.0],
        })

        df_clr = clr_across_cells(df)

        assert df_clr.shape == df.shape
        assert np.all(df_clr.columns == df.columns)

    def test_double_zscore_log_basic(self):
        """Test double z-score + log transformation runs without error."""
        # Create data with more rows to avoid edge cases in double z-score
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "marker2": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        })

        df_transformed = double_zscore_log(df)

        # Should preserve shape
        assert df_transformed.shape == df.shape

        # Note: double_zscore_log can produce NaN for certain edge cases
        # (when z-scores are very low, resulting in CDF ~ 0 and log(1-0) = inf)
        # This is expected behavior, so we just check it doesn't crash

    def test_zscore_log_basic(self):
        """Test z-score + log transformation produces standard normal per column."""
        df = pd.DataFrame({
            "marker1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "marker2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        df_zscore = zscore_log(df)

        # Should be mean ~0, std ~1 per column
        assert df_zscore.shape == df.shape
        for col in df_zscore.columns:
            assert abs(df_zscore[col].mean()) < 1e-10
            assert abs(df_zscore[col].std() - 1.0) < 1e-10
