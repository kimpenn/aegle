import unittest

import numpy as np
from typing import Tuple
from skimage.segmentation import find_boundaries

from aegle.repair_masks import (
    calculate_matching_statistics,
    get_boundary,
    get_matched_fraction,
    repair_masks_batch,
)
from tests.utils import make_segmentation_result


def expected_fraction(mask: np.ndarray, matched_mask: np.ndarray, nucleus_mask: np.ndarray) -> float:
    """Replicate the helper logic for computing matched fraction in tests."""
    matched_cell_num = len(np.unique(matched_mask))
    total_cell_num = len(np.unique(mask))
    total_nuclei_num = len(np.unique(nucleus_mask))
    denominator = (
        (total_cell_num - matched_cell_num)
        + (total_nuclei_num - matched_cell_num)
        + matched_cell_num
    )
    if denominator == 0:
        return 0.0
    return matched_cell_num / denominator


def build_two_cell_masks() -> Tuple[np.ndarray, np.ndarray]:
    """Return cell/nucleus masks with two labelled cells for testing."""
    cell = np.zeros((15, 15), dtype=np.uint32)
    # Cell 1 occupies a 5x5 square in the upper-left quadrant.
    cell[2:7, 2:7] = 1
    # Cell 2 occupies a 5x5 square in the lower-right quadrant.
    cell[8:13, 8:13] = 2

    nucleus = np.zeros_like(cell)
    # Nuclei sit well inside their respective cells.
    nucleus[3:6, 3:6] = 1
    nucleus[9:12, 9:12] = 2

    return cell, nucleus


def repair_single(cell_mask: np.ndarray, nucleus_mask: np.ndarray):
    """Run mask repair on a single segmentation result and return the output dict."""
    seg_res = make_segmentation_result(cell_mask, nucleus_mask)
    repaired_batch = repair_masks_batch([seg_res])
    assert len(repaired_batch) == 1
    return repaired_batch[0]


class TestRepairMasksBatch(unittest.TestCase):
    def test_perfect_overlap_returns_all_labels(self):
        """Matched masks should mirror the originals when nuclei align perfectly."""
        cell_mask, nucleus_mask = build_two_cell_masks()

        repaired = repair_single(cell_mask, nucleus_mask)

        np.testing.assert_array_equal(repaired["cell_matched_mask"] > 0, cell_mask > 0)
        np.testing.assert_array_equal(
            repaired["nucleus_matched_mask"] > 0, nucleus_mask > 0
        )
        self.assertAlmostEqual(repaired["matched_fraction"], 1.0)

        stats = repaired["matching_stats"]
        self.assertEqual(stats["whole_cell"]["unmatched"][0], 0)
        self.assertEqual(stats["nucleus"]["unmatched"][0], 0)

    def test_missing_nucleus_marks_cell_unmatched(self):
        """Dropping a nucleus should flag the corresponding cell as unmatched."""
        cell_mask, nucleus_mask = build_two_cell_masks()
        nucleus_mask[nucleus_mask == 2] = 0

        repaired = repair_single(cell_mask, nucleus_mask)

        self.assertListEqual(sorted(np.unique(repaired["cell_matched_mask"])), [0, 1])
        self.assertListEqual(sorted(np.unique(repaired["nucleus_matched_mask"])), [0, 1])

        stats = repaired["matching_stats"]
        self.assertEqual(stats["whole_cell"]["matched"][0], 1)
        self.assertEqual(stats["whole_cell"]["unmatched"][0], 1)
        self.assertEqual(stats["nucleus"]["unmatched"][0], 0)

        expected = expected_fraction(cell_mask, repaired["cell_matched_mask"], nucleus_mask)
        self.assertAlmostEqual(repaired["matched_fraction"], expected)

    def test_partial_overlap_trims_nucleus_outside_cell(self):
        """Nucleus pixels outside the cell interior should be removed during repair."""
        cell_mask, nucleus_mask = build_two_cell_masks()
        nucleus_with_spill = nucleus_mask.copy()
        nucleus_with_spill[7, 4] = 1

        repaired = repair_single(cell_mask, nucleus_with_spill)

        cell_boundary = find_boundaries(cell_mask, mode="inner")
        cell_interior = np.where(cell_boundary, 0, cell_mask)
        expected_trimmed = np.where(cell_interior > 0, nucleus_mask, 0).astype(np.uint32)
        np.testing.assert_array_equal(repaired["nucleus_matched_mask"], expected_trimmed)

        self.assertAlmostEqual(repaired["matched_fraction"], 1.0)


class TestMatchingStatistics(unittest.TestCase):
    def test_statistics_report_matched_and_unmatched_counts(self):
        """Statistics helper should report totals and unmatched cells/nuclei."""
        cell_mask, nucleus_mask = build_two_cell_masks()
        nucleus_mask[nucleus_mask == 2] = 0
        repaired = repair_single(cell_mask, nucleus_mask)

        stats = calculate_matching_statistics(
            cell_mask,
            nucleus_mask,
            repaired["cell_matched_mask"],
            repaired["nucleus_matched_mask"],
        )

        self.assertEqual(stats["whole_cell"]["total"][0], 2)
        self.assertEqual(stats["whole_cell"]["matched"][0], 1)
        self.assertEqual(stats["whole_cell"]["unmatched"][0], 1)
        self.assertEqual(stats["nucleus"]["total"][0], 1)
        self.assertEqual(stats["nucleus"]["matched"][0], 1)
        self.assertEqual(stats["nucleus"]["unmatched"][0], 0)


class TestMatchedFraction(unittest.TestCase):
    def test_fraction_matches_manual_calculation(self):
        """`get_matched_fraction` should align with manual matched fraction computation."""
        cell_mask, nucleus_mask = build_two_cell_masks()
        nucleus_mask[nucleus_mask == 2] = 0
        repaired = repair_single(cell_mask, nucleus_mask)

        fraction = get_matched_fraction(
            "nonrepaired_matched_mask",
            cell_mask,
            repaired["cell_matched_mask"],
            nucleus_mask,
        )
        expected = expected_fraction(cell_mask, repaired["cell_matched_mask"], nucleus_mask)
        self.assertAlmostEqual(fraction, expected)

    def test_repaired_mask_variant_returns_one(self):
        """The repaired mask variant should always return 1.0 matched fraction."""
        cell_mask, nucleus_mask = build_two_cell_masks()
        repaired = repair_single(cell_mask, nucleus_mask)

        fraction = get_matched_fraction(
            "repaired_matched_mask",
            cell_mask,
            repaired["cell_matched_mask"],
            nucleus_mask,
        )
        self.assertAlmostEqual(fraction, 1.0)


class TestBoundaryHelpers(unittest.TestCase):
    def test_get_boundary_preserves_labels_on_object_edges(self):
        """`get_boundary` should keep object labels along boundary pixels."""
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[1:5, 1:5] = 3

        expected = np.where(find_boundaries(mask, mode="inner"), mask, 0).astype(np.uint32)
        np.testing.assert_array_equal(get_boundary(mask), expected)


if __name__ == "__main__":
    unittest.main()
