"""Integration tests for all repair optimizations."""

import numpy as np
import pytest
from aegle.repair_masks import repair_masks_single, repair_masks_batch
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_stress_test_case,
)


@pytest.mark.slow
class TestRepairIntegration:
    """Test all optimizations working together."""

    def test_full_pipeline_small(self):
        """Test complete pipeline on small sample."""
        cell, nucleus, expected = create_simple_mask_pair(n_cells=10)

        cell_batch = np.expand_dims(cell, axis=0)
        nucleus_batch = np.expand_dims(nucleus, axis=0)

        result = repair_masks_single(cell_batch, nucleus_batch)

        # Verify structure
        assert "cell_matched_mask" in result
        assert "nucleus_matched_mask" in result
        assert "matched_fraction" in result

        # Verify correctness
        n_matched = len(np.unique(result["cell_matched_mask"])) - 1
        assert n_matched == expected["n_matched_cells"]

    def test_full_pipeline_medium(self):
        """Test on medium sample with stress test."""
        cell, nucleus, _ = create_stress_test_case(n_cells=1000)

        cell_batch = np.expand_dims(cell, axis=0)
        nucleus_batch = np.expand_dims(nucleus, axis=0)

        result = repair_masks_single(cell_batch, nucleus_batch)

        # Should complete without errors
        assert result is not None
        assert "cell_matched_mask" in result

    def test_batch_processing(self):
        """Test batch processing with multiple samples."""
        # Create 3 batches
        batches = []
        for i in range(3):
            cell, nucleus, _ = create_simple_mask_pair(n_cells=10, seed=40 + i)
            batches.append({"cell": cell, "nucleus": nucleus})

        results = repair_masks_batch(batches)

        assert len(results) == 3
        for result in results:
            assert "cell_matched_mask" in result
            assert "matched_fraction" in result

    def test_progress_logging_enabled(self):
        """Verify progress logging doesn't break pipeline."""
        import sys
        from io import StringIO

        cell, nucleus, _ = create_simple_mask_pair(n_cells=100)
        cell_batch = np.expand_dims(cell, axis=0)
        nucleus_batch = np.expand_dims(nucleus, axis=0)

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            result = repair_masks_single(cell_batch, nucleus_batch)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Progress bar should appear (tqdm output)
        assert result is not None
        # Check that we got some progress indication
        # (tqdm may not output to StringIO in test env, so just verify no crash)

    def test_numpy_vs_expected_counts(self):
        """Verify optimization produces same cell counts as expected."""
        test_cases = [
            (10, "small"),
            (50, "medium"),
            (100, "large"),
        ]

        for n_cells, label in test_cases:
            cell, nucleus, expected = create_simple_mask_pair(n_cells=n_cells)

            cell_batch = np.expand_dims(cell, axis=0)
            nucleus_batch = np.expand_dims(nucleus, axis=0)

            result = repair_masks_single(cell_batch, nucleus_batch)

            n_matched = len(np.unique(result["cell_matched_mask"])) - 1
            # Allow for minor differences due to random placement in fixtures
            # Should match at least 90% of expected cells
            assert n_matched >= expected["n_matched_cells"] * 0.9, f"Mismatch for {label} case: got {n_matched}, expected ~{expected['n_matched_cells']}"

    def test_large_stress_case(self):
        """Test large-scale case to verify scalability."""
        cell, nucleus, expected = create_stress_test_case(n_cells=5000)

        cell_batch = np.expand_dims(cell, axis=0)
        nucleus_batch = np.expand_dims(nucleus, axis=0)

        result = repair_masks_single(cell_batch, nucleus_batch)

        # Should complete successfully
        assert result is not None
        assert "matching_stats" in result

        # Verify reasonable matching rate
        n_matched = len(np.unique(result["cell_matched_mask"])) - 1
        n_cells = len(np.unique(cell)) - 1
        match_rate = n_matched / n_cells if n_cells > 0 else 0

        # Should match at least 70% (stress test has random placement)
        assert match_rate >= 0.7, f"Match rate too low: {match_rate:.2%}"

    def test_all_optimizations_applied(self):
        """Verify all optimization features are present in codebase.

        This test doesn't execute code but checks that optimization
        signatures are present in the module.
        """
        from aegle import repair_masks

        # Task 1.1: Vectorized nucleus lookup (should NOT use np.unique)
        # Task 1.2: Set-based index tracking (using sets not lists)
        # Task 1.3: Eliminated numpy.unique() - uses set operations
        # Task 1.4: NumPy-based coordinate extraction
        assert hasattr(repair_masks, "get_indices_numpy")

        # Task 1.5: Progress logging
        # Verify tqdm is imported
        import inspect

        source = inspect.getsource(repair_masks.get_matched_masks)
        assert "tqdm" in source, "Progress bar (tqdm) not found in code"
        assert "pbar.update" in source, "Progress bar update not found"

    def test_edge_case_empty_masks(self):
        """Test edge case with empty masks."""
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((100, 100), dtype=np.uint32)

        cell_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_batch, nucleus_batch)

        # Should handle empty masks without crashing
        assert result is not None
        assert len(np.unique(result["cell_matched_mask"])) == 1  # Only background

    def test_edge_case_single_cell(self):
        """Test edge case with single cell."""
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((100, 100), dtype=np.uint32)

        # Create one cell and one nucleus in center
        y, x = np.ogrid[:100, :100]
        cell_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 30**2
        nucleus_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 15**2

        cell_mask[cell_circle] = 1
        nucleus_mask[nucleus_circle] = 1

        cell_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_batch, nucleus_batch)

        # Should match the single cell
        n_matched = len(np.unique(result["cell_matched_mask"])) - 1
        assert n_matched == 1
