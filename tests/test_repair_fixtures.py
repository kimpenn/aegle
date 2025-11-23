"""Quick verification tests for repair test fixtures."""

import numpy as np
import pytest
from tests.utils.repair_test_fixtures import (
    create_simple_mask_pair,
    create_overlapping_nuclei_case,
    create_unmatched_cells_case,
    create_partial_overlap_case,
    create_edge_case_empty_masks,
    create_edge_case_single_cell,
)


class TestFixtureGeneration:
    """Verify that test fixtures generate valid masks."""

    def test_simple_mask_pair(self):
        """Test simple 1:1 matching case."""
        cell_mask, nucleus_mask, expected = create_simple_mask_pair(n_cells=10)

        assert cell_mask.shape == (512, 512)
        assert nucleus_mask.shape == (512, 512)
        assert cell_mask.dtype == np.uint32
        assert nucleus_mask.dtype == np.uint32

        # Check number of cells and nuclei
        n_cells = len(np.unique(cell_mask)) - 1
        n_nuclei = len(np.unique(nucleus_mask)) - 1

        assert n_cells == 10, f"Expected 10 cells, got {n_cells}"
        assert n_nuclei == 10, f"Expected 10 nuclei, got {n_nuclei}"

        # Verify all nuclei are inside cells
        for nucleus_id in range(1, n_nuclei + 1):
            nucleus_pixels = nucleus_mask == nucleus_id
            cells_overlapping = np.unique(cell_mask[nucleus_pixels])
            cells_overlapping = cells_overlapping[cells_overlapping > 0]
            assert len(cells_overlapping) > 0, f"Nucleus {nucleus_id} not inside any cell"

    def test_overlapping_nuclei_case(self):
        """Test case with multiple nuclei per cell."""
        cell_mask, nucleus_mask, expected = create_overlapping_nuclei_case(
            n_cells=5, n_nuclei=10
        )

        n_cells = len(np.unique(cell_mask)) - 1
        n_nuclei = len(np.unique(nucleus_mask)) - 1

        assert n_cells == 5
        assert n_nuclei >= 5, f"Should have at least 5 nuclei, got {n_nuclei}"

    def test_unmatched_cells_case(self):
        """Test case with cells without nuclei."""
        cell_mask, nucleus_mask, expected = create_unmatched_cells_case(
            n_cells_with_nuclei=5, n_cells_without_nuclei=3
        )

        n_cells = len(np.unique(cell_mask)) - 1
        n_nuclei = len(np.unique(nucleus_mask)) - 1

        assert n_cells == 8, f"Expected 8 total cells, got {n_cells}"
        assert n_nuclei == 5, f"Expected 5 nuclei, got {n_nuclei}"

        # Verify that some cells have no overlapping nucleus
        cells_without_nucleus = 0
        for cell_id in range(1, n_cells + 1):
            cell_pixels = cell_mask == cell_id
            nuclei_in_cell = np.unique(nucleus_mask[cell_pixels])
            nuclei_in_cell = nuclei_in_cell[nuclei_in_cell > 0]
            if len(nuclei_in_cell) == 0:
                cells_without_nucleus += 1

        assert cells_without_nucleus == 3, f"Expected 3 cells without nucleus, got {cells_without_nucleus}"

    def test_partial_overlap_case(self):
        """Test case where nuclei extend outside cells."""
        cell_mask, nucleus_mask, expected = create_partial_overlap_case(n_cells=5)

        n_cells = len(np.unique(cell_mask)) - 1
        n_nuclei = len(np.unique(nucleus_mask)) - 1

        assert n_cells == 5
        assert n_nuclei == 5

        # Verify some nuclei pixels are outside corresponding cells
        found_partial_overlap = False
        for i in range(1, n_nuclei + 1):
            nucleus_pixels = nucleus_mask == i
            cell_at_nucleus = cell_mask[nucleus_pixels]
            # If nucleus has pixels outside its matched cell
            if len(np.unique(cell_at_nucleus[cell_at_nucleus > 0])) > 1 or \
               np.sum(cell_at_nucleus == 0) > 0:
                found_partial_overlap = True
                break

        assert found_partial_overlap, "Should have at least one nucleus with partial overlap"

    def test_edge_case_empty_masks(self):
        """Test empty mask edge case."""
        cell_mask, nucleus_mask, expected = create_edge_case_empty_masks()

        assert cell_mask.shape == (100, 100)
        assert nucleus_mask.shape == (100, 100)
        assert np.sum(cell_mask) == 0
        assert np.sum(nucleus_mask) == 0

    def test_edge_case_single_cell(self):
        """Test single cell edge case."""
        cell_mask, nucleus_mask, expected = create_edge_case_single_cell()

        n_cells = len(np.unique(cell_mask)) - 1
        n_nuclei = len(np.unique(nucleus_mask)) - 1

        assert n_cells == 1
        assert n_nuclei == 1

    def test_reproducibility(self):
        """Test that same seed produces identical masks."""
        cell1, nucleus1, _ = create_simple_mask_pair(seed=42)
        cell2, nucleus2, _ = create_simple_mask_pair(seed=42)

        assert np.array_equal(cell1, cell2)
        assert np.array_equal(nucleus1, nucleus2)

    def test_different_seeds(self):
        """Test that different seeds produce different masks."""
        cell1, nucleus1, _ = create_simple_mask_pair(seed=42)
        cell2, nucleus2, _ = create_simple_mask_pair(seed=100)

        assert not np.array_equal(cell1, cell2)
