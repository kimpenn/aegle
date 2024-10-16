import unittest
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries
from aegle.segment import (
    _match_cells_to_nuclei,
    get_matched_cells,
    _get_mask_coordinates,
    get_mask,
    get_boundary,
    get_matched_masks,
    _get_indexed_mask,
)


class TestGetMaskCoordinates(unittest.TestCase):
    def test_get_mask_coordinates(self):
        # Create a simple mask with three objects
        mask = np.zeros((5, 5), dtype=int)
        mask[1, 1] = 1
        mask[2, 2] = 2
        mask[3, 3] = 3
        # Visualize the mask
        # [
        #     [0 0 0 0 0]
        #     [0 1 0 0 0]
        #     [0 0 2 0 0]
        #     [0 0 0 3 0]
        #     [0 0 0 0 0]
        # ]

        # Expected coordinates
        expected_coords = [
            np.array([[1, 1]]),
            np.array([[2, 2]]),
            np.array([[3, 3]]),
        ]

        # Get coordinates using the function
        coords_list = _get_mask_coordinates(mask)

        # Check the number of objects
        self.assertEqual(len(coords_list), 3)

        # Check coordinates for each object
        for coords, expected in zip(coords_list, expected_coords):
            np.testing.assert_array_equal(coords, expected)


class TestGetMask(unittest.TestCase):
    def test_get_mask(self):
        # Create sample coordinate lists for four objects
        coords_list = [
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),  # Object 1 (top-left corner)
            np.array([[2, 2], [2, 3], [3, 2], [3, 3]]),  # Object 2 (center)
            np.array([[4, 4]]),  # Object 3 (single pixel, bottom-right corner)
            np.array([[0, 4], [1, 4], [0, 3], [1, 3]]),  # Object 4 (top-right corner)
        ]

        # Expected mask
        expected_mask = np.array(
            [
                [1, 1, 0, 4, 4],
                [1, 1, 0, 4, 4],
                [0, 0, 2, 2, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 0, 3],
            ]
        )

        # Generate mask using the function
        mask = get_mask(coords_list, shape=(5, 5))

        # Check if the generated mask matches the expected mask
        np.testing.assert_array_equal(mask, expected_mask)

    def test_get_mask_with_overlap(self):
        # Create sample coordinate lists with overlapping objects
        coords_list = [
            np.array([[1, 1], [1, 2], [2, 1]]),  # Object 1
            np.array([[1, 2], [2, 2], [2, 1]]),  # Object 2 overlaps with Object 1
        ]

        # Expected mask
        expected_mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        # Generate mask using the function
        mask = get_mask(coords_list, shape=(5, 5))
        # Check if the generated mask matches the expected mask, considering overlaps
        np.testing.assert_array_equal(mask, expected_mask)

    def test_get_mask_with_out_of_bounds_coords(self):
        # Create sample coordinate lists with out-of-bounds coordinates
        coords_list = [
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),  # Valid coordinates
            np.array([[5, 5], [6, 6]]),  # Out-of-bounds coordinates
        ]

        # Expected mask
        expected_mask = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        # Generate mask using the function
        # Since the out-of-bounds coordinates should be ignored, the mask should be as expected
        mask = get_mask(coords_list, shape=(5, 5))

        # Check if the generated mask matches the expected mask
        np.testing.assert_array_equal(mask, expected_mask)


class TestGetBoundary(unittest.TestCase):
    def test_get_boundary(self):
        # Create a more complex mask of size (10, 10) with multiple objects
        mask = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 0
                [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],  # Row 1
                [0, 1, 1, 1, 0, 0, 2, 0, 2, 0],  # Row 2
                [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],  # Row 3
                [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],  # Row 4
                [0, 0, 0, 0, 3, 3, 3, 0, 4, 0],  # Row 5
                [0, 0, 0, 0, 3, 3, 3, 0, 4, 0],  # Row 6
                [0, 0, 5, 5, 5, 0, 0, 0, 4, 0],  # Row 7
                [0, 0, 5, 0, 5, 0, 0, 0, 4, 0],  # Row 8
                [0, 0, 5, 5, 5, 0, 0, 0, 0, 0],  # Row 9
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 10
            ]
        )

        # Expected boundary mask
        expected_boundary = np.zeros_like(mask)

        # Object 1 boundary (label 1)
        expected_boundary[1, 1] = 1
        expected_boundary[1, 2] = 1
        expected_boundary[1, 3] = 1
        expected_boundary[2, 1] = 1
        expected_boundary[2, 3] = 1
        expected_boundary[3, 1] = 1
        expected_boundary[3, 2] = 1
        expected_boundary[3, 3] = 1

        # Object 2 boundary (label 2)
        expected_boundary[1, 6] = 2
        expected_boundary[1, 7] = 2
        expected_boundary[1, 8] = 2
        expected_boundary[2, 6] = 2
        expected_boundary[2, 8] = 2
        expected_boundary[3, 6] = 2
        expected_boundary[3, 7] = 2
        expected_boundary[3, 8] = 2

        # Object 3 boundary (label 3)
        expected_boundary[4, 4] = 3
        expected_boundary[4, 5] = 3
        expected_boundary[4, 6] = 3
        expected_boundary[5, 4] = 3
        expected_boundary[5, 6] = 3
        expected_boundary[6, 4] = 3
        expected_boundary[6, 5] = 3
        expected_boundary[6, 6] = 3

        # Object 4 boundary (label 4)
        expected_boundary[5, 8] = 4
        expected_boundary[6, 8] = 4
        expected_boundary[7, 8] = 4
        expected_boundary[8, 8] = 4

        # Object 5 boundary (label 5)
        expected_boundary[7, 2] = 5
        expected_boundary[7, 3] = 5
        expected_boundary[7, 4] = 5
        expected_boundary[8, 2] = 5
        expected_boundary[8, 4] = 5
        expected_boundary[9, 2] = 5
        expected_boundary[9, 3] = 5
        expected_boundary[9, 4] = 5

        # Generate boundary using the function
        boundaries = get_boundary([mask])
        print(boundaries[0])
        print("---")
        # print(expected_boundary)
        # There should be one boundary mask
        self.assertEqual(len(boundaries), 1)

        # Check if the boundary matches the expected boundary
        np.testing.assert_array_equal(boundaries[0], expected_boundary)


class TestGetMatchedCells(unittest.TestCase):
    def test_get_matched_cells_no_mismatch(self):
        # Create cell and nucleus coordinates with perfect overlap
        cell_coords = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        cell_membrane_coords = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        nucleus_coords = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

        # Expected mismatch fraction is 0
        expected_mismatch_fraction = 0.0

        # Get matched cells without mismatch repair
        whole_cell, nucleus, mismatch_fraction = get_matched_cells(
            cell_coords, cell_membrane_coords, nucleus_coords, mismatch_repair=False
        )

        # Check if the mismatch fraction is correct
        self.assertEqual(mismatch_fraction, expected_mismatch_fraction)

        # Check if the returned coordinates match the input
        np.testing.assert_array_equal(whole_cell, cell_coords)
        np.testing.assert_array_equal(nucleus, nucleus_coords)

    def test_get_matched_cells_with_mismatch(self):
        # Create cell and nucleus coordinates with partial overlap
        cell_coords = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
        cell_membrane_coords = np.array([[1, 1], [1, 2]])
        nucleus_coords = np.array([[2, 1], [2, 2], [3, 3]])

        # Expected mismatch fraction
        expected_mismatch_fraction = 1 / 3  # One mismatched pixel out of three

        # Get matched cells with mismatch repair
        whole_cell, nucleus, mismatch_fraction = get_matched_cells(
            cell_coords, cell_membrane_coords, nucleus_coords, mismatch_repair=True
        )

        # Check if the mismatch fraction is correct
        self.assertAlmostEqual(mismatch_fraction, expected_mismatch_fraction)

        # Check if the matched nucleus coordinates are correct
        expected_matched_nucleus = np.array([[2, 1], [2, 2]])
        np.testing.assert_array_equal(nucleus, expected_matched_nucleus)


class TestMatchCellsToNuclei(unittest.TestCase):
    def test_match_cells_to_nuclei(self):
        # Create sample masks
        cell_mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        nucleus_mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ]
        )

        cell_membrane_mask = find_boundaries(cell_mask, mode="inner").astype(int)

        # Extract coordinates
        cell_coords = _get_mask_coordinates(cell_mask)
        nucleus_coords = _get_mask_coordinates(nucleus_mask)
        cell_membrane_coords = _get_mask_coordinates(cell_membrane_mask)

        print(cell_coords)
        # Perform matching
        cell_matched_list, nucleus_matched_list = _match_cells_to_nuclei(
            cell_coords,
            nucleus_coords,
            cell_membrane_coords,
            nucleus_mask,
            do_mismatch_repair=False,
        )

        # Expected results
        expected_cell_matched = [cell_coords[0]]
        expected_nucleus_matched = [nucleus_coords[0]]

        # Check if the matched lists have one entry
        self.assertEqual(len(cell_matched_list), 1)
        self.assertEqual(len(nucleus_matched_list), 1)

        # Check if the matched coordinates match expected
        np.testing.assert_array_equal(cell_matched_list[0], expected_cell_matched[0])
        np.testing.assert_array_equal(
            nucleus_matched_list[0], expected_nucleus_matched[0]
        )


class TestGetMatchedMasks(unittest.TestCase):
    def test_get_matched_masks(self):
        # Create sample segmentation output
        segmentation_output = [
            {
                "cell": np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                "nucleus": np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 2, 2, 0],
                        [0, 2, 2, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                "cell_boundary": np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                "nucleus_boundary": np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 2, 2, 0],
                        [0, 2, 2, 0],
                        [0, 0, 0, 0],
                    ]
                ),
            }
        ]

        # Get matched masks
        matched_output, fraction_matched_cells = get_matched_masks(
            segmentation_output, do_mismatch_repair=False
        )

        # Expected fraction of matched cells
        expected_fraction = 1.0

        # Check if the fraction matched is as expected
        self.assertEqual(fraction_matched_cells, expected_fraction)

        # Check if the matched_output has the correct keys
        self.assertEqual(len(matched_output), 1)
        self.assertIn("cell", matched_output[0])
        self.assertIn("nucleus", matched_output[0])
        self.assertIn("cell_boundary", matched_output[0])
        self.assertIn("nucleus_boundary", matched_output[0])

        # Additional checks can be added here


class TestGetIndexedMask(unittest.TestCase):
    def test_get_indexed_mask(self):
        # Create a sample mask and boundary
        mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        boundary = find_boundaries(mask, mode="inner").astype(int)

        # Expected indexed boundary mask
        expected_boundary_mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        )

        # Get the indexed mask
        indexed_boundary = _get_indexed_mask(mask, boundary)

        # Check if the indexed boundary matches the expected
        np.testing.assert_array_equal(indexed_boundary, expected_boundary_mask)
