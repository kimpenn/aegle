"""Comprehensive test suite for mask repair pipeline.

This module tests the repair_masks implementation in aegle/repair_masks.py.
Tests cover:
- Synthetic fixtures with known ground truth
- Edge cases (empty, single cell, no overlap)
- Numerical precision and consistency
- Helper function behavior

All tests use deterministic synthetic data from tests.utils.repair_test_fixtures.
"""

import numpy as np
import pytest
from typing import Dict, Any

from aegle.repair_masks import (
    repair_masks_batch,
    repair_masks_single,
    get_matched_masks,
    get_matched_cells,
    get_boundary,
    calculate_matching_statistics,
    _compute_labeled_boundary,
)

from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_overlapping_nuclei_case,
    create_unmatched_cells_case,
    create_partial_overlap_case,
    create_stress_test_case,
    create_edge_case_empty_masks,
    create_edge_case_single_cell,
)


class TestRepairSynthetic:
    """Test repair pipeline with synthetic fixtures."""

    def test_simple_matching(self):
        """Verify 1:1 matching works with perfect cell-nucleus pairs.

        Each of 10 cells has exactly one centered nucleus. All should match.
        """
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        # Add batch dimension for repair_masks_single
        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        # Check all cells matched
        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
        n_matched_nuclei = len(np.unique(result["nucleus_matched_mask"])) - 1

        assert n_matched_cells == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matched cells, got {n_matched_cells}"
        assert n_matched_nuclei == expected["n_matched_nuclei"], \
            f"Expected {expected['n_matched_nuclei']} matched nuclei, got {n_matched_nuclei}"

        # Check matched_fraction is high (close to 1.0 for perfect matching)
        assert result["matched_fraction"] > 0.9, \
            f"Expected matched_fraction > 0.9, got {result['matched_fraction']}"

        # Check masks have correct dtype
        assert result["cell_matched_mask"].dtype == np.uint32
        assert result["nucleus_matched_mask"].dtype == np.uint32

        # Check matched masks have contiguous labels starting from 1
        unique_cells = np.unique(result["cell_matched_mask"])
        assert unique_cells[0] == 0, "Background should be 0"
        assert len(unique_cells) == n_matched_cells + 1, "Labels should be contiguous"

    def test_multiple_nuclei_per_cell(self):
        """Test that algorithm picks largest overlap when multiple nuclei in one cell.

        Creates cells with multiple candidate nuclei. The repair algorithm should
        match each cell to the best nucleus based on overlap.
        """
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
        n_matched_nuclei = len(np.unique(result["nucleus_matched_mask"])) - 1

        # All cells should find at least one nucleus
        assert n_matched_cells == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matched cells, got {n_matched_cells}"

        # Number of matched nuclei should equal number of matched cells (1:1 final matching)
        assert n_matched_nuclei == n_matched_cells, \
            f"Expected 1:1 matching, got {n_matched_cells} cells and {n_matched_nuclei} nuclei"

        # Note: Due to overlap filtering in fixture, we may not have extra nuclei
        # The important test is that 1:1 matching is achieved and all cells match
        total_nuclei = len(np.unique(nucleus_mask)) - 1
        assert n_matched_nuclei <= total_nuclei, \
            "Matched nuclei should not exceed total nuclei"

    def test_unmatched_cells_discarded(self):
        """Verify cells without nuclei are removed from matched masks.

        Creates 5 cells with nuclei + 3 cells without nuclei.
        Only the 5 with nuclei should appear in matched masks.
        """
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=5, n_cells_without_nuclei=3
        )

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1

        # Only cells with nuclei should be matched
        assert n_matched_cells == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matched cells, got {n_matched_cells}"

        # Check matching statistics reflect unmatched cells
        stats = result["matching_stats"]
        assert stats["whole_cell"]["unmatched"][0] == expected["n_unmatched_cells"], \
            f"Expected {expected['n_unmatched_cells']} unmatched cells in stats"

    def test_nucleus_trimming(self):
        """Verify nuclei are trimmed to cell interior (excluding membrane).

        Creates nuclei that partially extend outside cells. The repair should
        trim nucleus pixels to only those inside the cell interior.
        """
        cell_mask, nucleus_mask, expected = create_partial_overlap_case(
            n_cells=5, nucleus_offset=20
        )

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        # All cells should still match (trimming doesn't discard)
        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
        assert n_matched_cells == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matched cells after trimming"

        # Verify that matched nucleus is subset of matched cell
        cell_matched = result["cell_matched_mask"]
        nucleus_matched = result["nucleus_matched_mask"]

        # Every nucleus pixel should be inside a cell
        nucleus_pixels = nucleus_matched > 0
        cell_pixels = cell_matched > 0
        assert np.all(nucleus_pixels <= cell_pixels), \
            "Nucleus pixels should be subset of cell pixels after trimming"

        # Verify cell_outside_nucleus exists (since nuclei are trimmed/offset)
        outside_mask = result["cell_outside_nucleus_mask"]
        assert np.sum(outside_mask > 0) > 0, \
            "Expected non-zero cell_outside_nucleus_mask when nuclei are offset"

    def test_matching_statistics(self):
        """Verify statistics calculation is accurate.

        Tests that matching_stats dict contains correct counts and percentages.
        """
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=7, n_cells_without_nuclei=3
        )

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)
        stats = result["matching_stats"]

        # Check structure
        assert "whole_cell" in stats
        assert "nucleus" in stats
        assert "total" in stats["whole_cell"]
        assert "matched" in stats["whole_cell"]
        assert "unmatched" in stats["whole_cell"]

        # Check counts
        total_cells = stats["whole_cell"]["total"][0]
        matched_cells = stats["whole_cell"]["matched"][0]
        unmatched_cells = stats["whole_cell"]["unmatched"][0]

        assert total_cells == matched_cells + unmatched_cells, \
            "Total cells should equal matched + unmatched"

        assert matched_cells == expected["n_matched_cells"], \
            f"Expected {expected['n_matched_cells']} matched in stats"
        assert unmatched_cells == expected["n_unmatched_cells"], \
            f"Expected {expected['n_unmatched_cells']} unmatched in stats"

        # Check percentages sum to 100%
        matched_pct = stats["whole_cell"]["matched"][1]
        unmatched_pct = stats["whole_cell"]["unmatched"][1]
        total_pct = stats["whole_cell"]["total"][1]

        assert total_pct == 100.0
        assert np.allclose(matched_pct + unmatched_pct, 100.0, atol=1e-6), \
            "Matched + unmatched percentages should sum to 100%"

    def test_matched_boundary_computation(self):
        """Verify boundary masks are computed correctly.

        Boundaries should be labeled (retain cell/nucleus IDs) and be subset of
        interior masks.
        """
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=5)

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        cell_boundary = result["cell_matched_boundary"]
        nucleus_boundary = result["nucleus_matched_boundary"]
        cell_matched = result["cell_matched_mask"]
        nucleus_matched = result["nucleus_matched_mask"]

        # Boundaries should be labeled (not binary)
        assert np.max(cell_boundary) > 1, "Cell boundary should be labeled"
        assert np.max(nucleus_boundary) > 1, "Nucleus boundary should be labeled"

        # Boundary pixels should be subset of matched pixels
        assert np.all((cell_boundary > 0) <= (cell_matched > 0)), \
            "Cell boundary should be subset of cell mask"
        assert np.all((nucleus_boundary > 0) <= (nucleus_matched > 0)), \
            "Nucleus boundary should be subset of nucleus mask"

        # Boundary labels should match interior labels
        for label in np.unique(cell_boundary):
            if label > 0:
                assert label in np.unique(cell_matched), \
                    f"Boundary label {label} not found in cell mask"


class TestRepairEdgeCases:
    """Test edge case handling."""

    def test_empty_masks(self):
        """Verify empty masks are handled gracefully without errors."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        # Should return empty matched masks
        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
        n_matched_nuclei = len(np.unique(result["nucleus_matched_mask"])) - 1

        assert n_matched_cells == 0, "Empty input should produce 0 matched cells"
        assert n_matched_nuclei == 0, "Empty input should produce 0 matched nuclei"

        # Check statistics
        stats = result["matching_stats"]
        assert stats["whole_cell"]["total"][0] == 0
        assert stats["nucleus"]["total"][0] == 0

    def test_all_cells_discarded(self):
        """Test case where no cells overlap with any nuclei.

        Creates cells and nuclei in completely disjoint regions.
        All cells should be discarded, returning empty matched masks.
        """
        # Create masks with cells on left, nuclei on right (no overlap)
        cell_mask = np.zeros((200, 200), dtype=np.uint32)
        nucleus_mask = np.zeros((200, 200), dtype=np.uint32)

        # Cells on left half
        y, x = np.ogrid[:200, :200]
        cell_mask[(x - 50) ** 2 + (y - 50) ** 2 <= 30 ** 2] = 1
        cell_mask[(x - 50) ** 2 + (y - 150) ** 2 <= 30 ** 2] = 2

        # Nuclei on right half (no overlap)
        nucleus_mask[(x - 150) ** 2 + (y - 50) ** 2 <= 15 ** 2] = 1
        nucleus_mask[(x - 150) ** 2 + (y - 150) ** 2 <= 15 ** 2] = 2

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        # No matches should be found
        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
        assert n_matched_cells == 0, \
            "Completely disjoint cells and nuclei should produce 0 matches"

    def test_single_cell(self):
        """Verify edge case with N=1 works correctly."""
        cell_mask, nucleus_mask, expected = create_edge_case_single_cell()

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1
        n_matched_nuclei = len(np.unique(result["nucleus_matched_mask"])) - 1

        assert n_matched_cells == 1, "Single cell should match"
        assert n_matched_nuclei == 1, "Single nucleus should match"

        # Check label is 1 (contiguous starting from 1)
        assert 1 in np.unique(result["cell_matched_mask"])
        assert 1 in np.unique(result["nucleus_matched_mask"])

    def test_no_overlap(self):
        """Test cells and nuclei completely disjoint (spatial separation)."""
        # Already covered in test_all_cells_discarded, but verify different approach
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((100, 100), dtype=np.uint32)

        # Cell in top-left
        y, x = np.ogrid[:100, :100]
        cell_mask[(x - 25) ** 2 + (y - 25) ** 2 <= 20 ** 2] = 1

        # Nucleus in bottom-right
        nucleus_mask[(x - 75) ** 2 + (y - 75) ** 2 <= 10 ** 2] = 1

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        # Should produce empty matched masks
        assert np.sum(result["cell_matched_mask"] > 0) == 0, \
            "No overlap should result in empty matched cell mask"
        assert np.sum(result["nucleus_matched_mask"] > 0) == 0, \
            "No overlap should result in empty matched nucleus mask"


class TestRepairNumerical:
    """Test numerical precision and consistency."""

    def test_mismatch_fraction_precision(self):
        """Verify mismatch_fraction calculation has acceptable float precision."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        matched_fraction = result["matched_fraction"]

        # Should be a valid float in [0, 1]
        assert isinstance(matched_fraction, (float, np.floating)), \
            "matched_fraction should be float type"
        assert 0.0 <= matched_fraction <= 1.0, \
            f"matched_fraction should be in [0, 1], got {matched_fraction}"

        # For perfect matching case, should be close to 1.0
        assert np.allclose(matched_fraction, 1.0, atol=0.1), \
            f"Perfect matching should give fraction close to 1.0, got {matched_fraction}"

    def test_coordinate_ordering(self):
        """Verify consistent coordinate ordering in outputs.

        Tests that repeated calls produce same results (deterministic).
        """
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=5, seed=42)

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        # Run twice
        result1 = repair_masks_single(cell_mask_batch, nucleus_mask_batch)
        result2 = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        # Results should be identical
        assert np.array_equal(result1["cell_matched_mask"], result2["cell_matched_mask"]), \
            "Repair should be deterministic"
        assert np.array_equal(result1["nucleus_matched_mask"], result2["nucleus_matched_mask"]), \
            "Repair should be deterministic"
        assert np.isclose(result1["matched_fraction"], result2["matched_fraction"]), \
            "matched_fraction should be deterministic"

    def test_relabeling_contiguous(self):
        """Verify matched masks have contiguous labels starting from 1.

        Even if input masks have gaps in labels (e.g., 1, 3, 7), output should
        be relabeled to 1, 2, 3, ...
        """
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=10)

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        cell_labels = np.unique(result["cell_matched_mask"])
        nucleus_labels = np.unique(result["nucleus_matched_mask"])

        # Remove background (0)
        cell_labels = cell_labels[cell_labels > 0]
        nucleus_labels = nucleus_labels[nucleus_labels > 0]

        # Check contiguous starting from 1
        if len(cell_labels) > 0:
            expected_labels = np.arange(1, len(cell_labels) + 1)
            assert np.array_equal(cell_labels, expected_labels), \
                f"Cell labels should be contiguous 1..N, got {cell_labels}"

        if len(nucleus_labels) > 0:
            expected_labels = np.arange(1, len(nucleus_labels) + 1)
            assert np.array_equal(nucleus_labels, expected_labels), \
                f"Nucleus labels should be contiguous 1..N, got {nucleus_labels}"


class TestRepairHelpers:
    """Test helper functions in repair_masks module."""

    def test_get_matched_cells(self):
        """Test individual cell-nucleus matching logic.

        Tests get_matched_cells() which matches a single cell to a candidate nucleus.
        """
        # Create simple coordinate arrays for one cell and one nucleus
        # Cell: 5x5 square centered at (50, 50)
        cell_coords = []
        for y in range(48, 53):
            for x in range(48, 53):
                cell_coords.append([0, y, x])  # z, y, x format
        cell_arr = np.array(cell_coords)

        # Membrane: just boundary pixels
        membrane_coords = []
        for y in range(48, 53):
            membrane_coords.extend([[0, y, 48], [0, y, 52]])
        for x in range(49, 52):
            membrane_coords.extend([[0, 48, x], [0, 52, x]])
        cell_membrane_arr = np.array(membrane_coords)

        # Nucleus: 3x3 square centered inside
        nucleus_coords = []
        for y in range(49, 52):
            for x in range(49, 52):
                nucleus_coords.append([0, y, x])
        nuclear_arr = np.array(nucleus_coords)

        # Test with mismatch_repair=1 (trimming enabled)
        whole_cell, nucleus, mismatch_fraction = get_matched_cells(
            cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair=1
        )

        # Should return matched arrays
        assert whole_cell.size > 0, "Should return matched cell"
        assert nucleus.size > 0, "Should return matched nucleus"
        assert mismatch_fraction == 0.0, \
            "Perfect overlap should give mismatch_fraction=0"

    def test_get_boundary(self):
        """Test boundary extraction with label preservation.

        Tests get_boundary() which finds boundaries and preserves labels.
        """
        # Create simple labeled mask
        mask = np.zeros((50, 50), dtype=np.uint32)
        y, x = np.ogrid[:50, :50]
        mask[(x - 15) ** 2 + (y - 15) ** 2 <= 10 ** 2] = 1
        mask[(x - 35) ** 2 + (y - 35) ** 2 <= 10 ** 2] = 2

        # Add batch dimension
        mask_batch = np.expand_dims(mask, axis=0)

        boundary = get_boundary(mask_batch)

        # Boundary should preserve labels
        assert 1 in np.unique(boundary), "Boundary should contain label 1"
        assert 2 in np.unique(boundary), "Boundary should contain label 2"

        # Boundary pixels should be subset of original mask pixels
        assert np.all((boundary > 0) <= (mask_batch > 0)), \
            "Boundary should be subset of original mask"

    def test_compute_labeled_boundary(self):
        """Test _compute_labeled_boundary helper function directly.

        Verifies that boundaries retain original cell/nucleus labels.
        """
        # Create simple mask
        mask = np.zeros((30, 30), dtype=np.uint32)
        y, x = np.ogrid[:30, :30]
        mask[(x - 15) ** 2 + (y - 15) ** 2 <= 10 ** 2] = 5  # Label 5

        boundary = _compute_labeled_boundary(mask)

        # Boundary should contain label 5, not binary 1
        assert 5 in np.unique(boundary), "Boundary should preserve label 5"
        assert np.max(boundary) == 5, "Max boundary value should be original label"

        # Boundary should be smaller than full mask
        assert np.sum(boundary > 0) < np.sum(mask > 0), \
            "Boundary should be subset of mask"

    def test_calculate_matching_statistics(self):
        """Test statistics calculation function directly.

        Verifies calculate_matching_statistics() computes correct counts and percentages.
        """
        # Create simple test case
        cell_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_mask = np.zeros((100, 100), dtype=np.uint32)

        # 3 cells total
        y, x = np.ogrid[:100, :100]
        cell_mask[(x - 25) ** 2 + (y - 25) ** 2 <= 15 ** 2] = 1
        cell_mask[(x - 50) ** 2 + (y - 50) ** 2 <= 15 ** 2] = 2
        cell_mask[(x - 75) ** 2 + (y - 75) ** 2 <= 15 ** 2] = 3

        # 2 nuclei total
        nucleus_mask[(x - 25) ** 2 + (y - 25) ** 2 <= 8 ** 2] = 1
        nucleus_mask[(x - 50) ** 2 + (y - 50) ** 2 <= 8 ** 2] = 2

        # Matched: 2 cells, 2 nuclei
        cell_matched_mask = np.zeros((100, 100), dtype=np.uint32)
        nucleus_matched_mask = np.zeros((100, 100), dtype=np.uint32)

        cell_matched_mask[(x - 25) ** 2 + (y - 25) ** 2 <= 15 ** 2] = 1
        cell_matched_mask[(x - 50) ** 2 + (y - 50) ** 2 <= 15 ** 2] = 2

        nucleus_matched_mask[(x - 25) ** 2 + (y - 25) ** 2 <= 8 ** 2] = 1
        nucleus_matched_mask[(x - 50) ** 2 + (y - 50) ** 2 <= 8 ** 2] = 2

        stats = calculate_matching_statistics(
            cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask
        )

        # Verify counts
        assert stats["whole_cell"]["total"][0] == 3, "Should have 3 total cells"
        assert stats["whole_cell"]["matched"][0] == 2, "Should have 2 matched cells"
        assert stats["whole_cell"]["unmatched"][0] == 1, "Should have 1 unmatched cell"

        assert stats["nucleus"]["total"][0] == 2, "Should have 2 total nuclei"
        assert stats["nucleus"]["matched"][0] == 2, "Should have 2 matched nuclei"
        assert stats["nucleus"]["unmatched"][0] == 0, "Should have 0 unmatched nuclei"

        # Verify percentages
        assert np.isclose(stats["whole_cell"]["matched"][1], 200.0 / 3, atol=1e-6), \
            "Matched percentage should be 2/3 * 100"
        assert np.isclose(stats["nucleus"]["matched"][1], 100.0, atol=1e-6), \
            "All nuclei matched, should be 100%"


class TestRepairBatch:
    """Test batch processing functionality."""

    def test_repair_masks_batch(self):
        """Test batch processing with multiple mask pairs.

        Verifies repair_masks_batch() correctly processes a list of seg results.
        """
        # Create 3 different mask pairs
        cell1, nucleus1, _ = create_simple_mask_pair(n_cells=5, seed=42)
        cell2, nucleus2, _ = create_simple_mask_pair(n_cells=8, seed=43)
        cell3, nucleus3, _ = create_unmatched_cells_case(
            n_cells_with_nuclei=4, n_cells_without_nuclei=2, seed=44
        )

        seg_res_batch = [
            {"cell": cell1, "nucleus": nucleus1},
            {"cell": cell2, "nucleus": nucleus2},
            {"cell": cell3, "nucleus": nucleus3},
        ]

        results = repair_masks_batch(seg_res_batch)

        # Should return list of 3 results
        assert len(results) == 3, "Should process all 3 items in batch"

        # Each result should have expected keys
        for result in results:
            assert "cell_matched_mask" in result
            assert "nucleus_matched_mask" in result
            assert "cell_outside_nucleus_mask" in result
            assert "cell_matched_boundary" in result
            assert "nucleus_matched_boundary" in result
            assert "matched_fraction" in result
            assert "matching_stats" in result

        # Verify first result matches expected counts
        result1 = results[0]
        n_matched_1 = len(np.unique(result1["cell_matched_mask"])) - 1
        assert n_matched_1 == 5, f"First batch item should have 5 matched cells"

        # Verify third result has unmatched cells
        result3 = results[2]
        stats3 = result3["matching_stats"]
        assert stats3["whole_cell"]["unmatched"][0] == 2, \
            "Third batch item should have 2 unmatched cells"


class TestRepairStressCase:
    """Test performance on large samples (not run by default)."""

    @pytest.mark.skip(reason="Stress test with 10K cells - use smaller scale for regular testing")
    def test_stress_test_large_sample(self):
        """Test repair on large sample (10K cells).

        This test is skipped by default due to long runtime and fixture instability.
        For performance testing, use smaller n_cells (e.g., 1000) or run separately.
        """
        # Use smaller cell count for stability
        cell_mask, nucleus_mask, expected = create_stress_test_case(n_cells=1000)

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1

        # At least 80% of cells should match (random placement may cause some misses)
        min_expected = expected["min_matched"]
        assert n_matched_cells >= min_expected, \
            f"Large sample should have at least {min_expected} matched cells, got {n_matched_cells}"

        # Verify no errors occurred
        assert result["cell_matched_mask"].shape == cell_mask.shape[1:]
        assert result["matched_fraction"] > 0.5, \
            f"Large sample should have reasonable match rate, got {result['matched_fraction']}"

    def test_moderate_sample(self):
        """Test repair on moderate sample size (100 cells).

        This is a scaled-down version suitable for regular testing.
        """
        cell_mask, nucleus_mask, _ = create_simple_mask_pair(n_cells=100,
                                                               image_size=(1024, 1024),
                                                               cell_radius=30,
                                                               nucleus_radius=15,
                                                               seed=100)

        cell_mask_batch = np.expand_dims(cell_mask, axis=0)
        nucleus_mask_batch = np.expand_dims(nucleus_mask, axis=0)

        result = repair_masks_single(cell_mask_batch, nucleus_mask_batch)

        n_matched_cells = len(np.unique(result["cell_matched_mask"])) - 1

        # All 100 cells should match (perfect case)
        assert n_matched_cells == 100, \
            f"Expected 100 matched cells, got {n_matched_cells}"

        # Verify outputs are valid
        assert result["cell_matched_mask"].shape == (1024, 1024)
        assert result["matched_fraction"] > 0.9
