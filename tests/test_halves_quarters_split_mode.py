"""Tests for halves and quarters split_mode in CodexPatches.

This module tests the complete data flow for split_mode='halves' and 'quarters':
1. Patch generation with correct coordinates
2. Mask merging with proper label reassignment
3. Integration with cell profiling
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aegle.codex_image import CodexImage
from aegle.codex_patches import CodexPatches
from tests.utils.fixtures import write_antibodies_tsv, write_tiny_ome_tiff


def _make_config(split_mode: str, split_direction: str = "vertical"):
    """Create a config dict for halves/quarters testing."""
    return {
        "data": {
            "file_name": "img.ome.tiff",
            "antibodies_file": "antibodies.tsv",
            "image_mpp": 0.5,
            "generate_channel_stats": False,
        },
        "channels": {"nucleus_channel": "ch0", "wholecell_channel": ["ch1"]},
        "patching": {
            "split_mode": split_mode,
            "split_direction": split_direction,
        },
        "visualization": {},
        "patch_qc": {},
        "segmentation": {"segmentation_pickle_compression_threads": 0},
        "evaluation": {"compute_metrics": False},
        "report": {"generate_report": False, "report_format": "html"},
    }


class TestHalvesPatchGeneration(unittest.TestCase):
    """Tests for _generate_halves_patches() coordinate calculation."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = self.tmpdir.name
        self.out_dir = os.path.join(self.tmpdir.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def _write_inputs(self, shape):
        """Write test image and antibodies file."""
        write_tiny_ome_tiff(os.path.join(self.data_dir, "img.ome.tiff"), shape=shape)
        write_antibodies_tsv(
            os.path.join(self.data_dir, "antibodies.tsv"),
            [f"ch{i}" for i in range(shape[2])],
        )

    # Task 1.1: Halves vertical split tests
    def test_halves_vertical_generates_two_patches(self):
        """Vertical halves split should generate exactly 2 patches."""
        # Create image with width > height (e.g., 10x20 where height=10, width=20)
        self._write_inputs(shape=(10, 20, 3))
        config = _make_config("halves", split_direction="vertical")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()
        codex_patches = CodexPatches(codex_image, config, args)

        # Verify exactly 2 patches created
        metadata = codex_patches.get_patches_metadata()
        self.assertEqual(len(metadata), 2, "Vertical halves should generate exactly 2 patches")

        # Check patches saved to disk (halves mode uses disk-based storage)
        self.assertTrue(
            codex_patches.is_using_disk_based_patches(),
            "Halves mode should use disk-based patches"
        )

    def test_halves_vertical_coordinates_correct(self):
        """Verify correct coordinates for vertical split patches."""
        # For 10x20 image:
        # Patch 0 (left): x_start=0, y_start=0, width=10, height=10
        # Patch 1 (right): x_start=10, y_start=0, width=10, height=10
        self._write_inputs(shape=(10, 20, 3))
        config = _make_config("halves", split_direction="vertical")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()
        codex_patches = CodexPatches(codex_image, config, args)

        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # Verify patch 0 (left half)
        patch0 = metadata_df.iloc[0]
        self.assertEqual(patch0["x_start"], 0, "Left patch should start at x=0")
        self.assertEqual(patch0["y_start"], 0, "Left patch should start at y=0")
        self.assertEqual(patch0["patch_width"], 10, "Left patch width should be 10")
        self.assertEqual(patch0["patch_height"], 10, "Left patch height should be 10")

        # Verify patch 1 (right half)
        patch1 = metadata_df.iloc[1]
        self.assertEqual(patch1["x_start"], 10, "Right patch should start at x=10")
        self.assertEqual(patch1["y_start"], 0, "Right patch should start at y=0")
        self.assertEqual(patch1["patch_width"], 10, "Right patch width should be 10")
        self.assertEqual(patch1["patch_height"], 10, "Right patch height should be 10")

    # Task 1.2: Halves horizontal split tests
    def test_halves_horizontal_generates_two_patches(self):
        """Horizontal halves split should generate exactly 2 patches."""
        # Create image with height > width (e.g., 20x10)
        self._write_inputs(shape=(20, 10, 3))
        config = _make_config("halves", split_direction="horizontal")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()
        codex_patches = CodexPatches(codex_image, config, args)

        # Verify exactly 2 patches created
        metadata = codex_patches.get_patches_metadata()
        self.assertEqual(len(metadata), 2, "Horizontal halves should generate exactly 2 patches")

        # Check patches saved to disk (halves mode uses disk-based storage)
        self.assertTrue(
            codex_patches.is_using_disk_based_patches(),
            "Halves mode should use disk-based patches"
        )

    def test_halves_horizontal_coordinates_correct(self):
        """Verify correct coordinates for horizontal split patches."""
        # For 20x10 image (height=20, width=10):
        # Patch 0 (top): x_start=0, y_start=0, width=10, height=10
        # Patch 1 (bottom): x_start=0, y_start=10, width=10, height=10
        self._write_inputs(shape=(20, 10, 3))
        config = _make_config("halves", split_direction="horizontal")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()
        codex_patches = CodexPatches(codex_image, config, args)

        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # Verify patch 0 (top half)
        patch0 = metadata_df.iloc[0]
        self.assertEqual(patch0["x_start"], 0, "Top patch should start at x=0")
        self.assertEqual(patch0["y_start"], 0, "Top patch should start at y=0")
        self.assertEqual(patch0["patch_width"], 10, "Top patch width should be 10")
        self.assertEqual(patch0["patch_height"], 10, "Top patch height should be 10")

        # Verify patch 1 (bottom half)
        patch1 = metadata_df.iloc[1]
        self.assertEqual(patch1["x_start"], 0, "Bottom patch should start at x=0")
        self.assertEqual(patch1["y_start"], 10, "Bottom patch should start at y=10")
        self.assertEqual(patch1["patch_width"], 10, "Bottom patch width should be 10")
        self.assertEqual(patch1["patch_height"], 10, "Bottom patch height should be 10")

    # Task 1.3: Halves odd dimensions tests
    def test_halves_vertical_odd_width_no_pixel_loss(self):
        """Vertical split on odd width should not lose pixels."""
        # Create 10x21 image (height=10, width=21, channels=2)
        self._write_inputs(shape=(10, 21, 2))
        config = _make_config("halves", "vertical")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()

        codex_patches = CodexPatches(codex_image, config, args)
        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # Should have 2 patches
        self.assertEqual(len(metadata_df), 2)

        # Left half: width=10, Right half: width=11
        # Verify no pixels lost
        total_width = metadata_df.iloc[0]["patch_width"] + metadata_df.iloc[1]["patch_width"]
        self.assertEqual(total_width, 21, "Total width should equal original image width")

        # Left patch starts at x=0
        self.assertEqual(metadata_df.iloc[0]["x_start"], 0)
        # Right patch starts where left ends
        self.assertEqual(metadata_df.iloc[1]["x_start"], metadata_df.iloc[0]["patch_width"])

        # Both patches should span full height
        self.assertEqual(metadata_df.iloc[0]["patch_height"], 10)
        self.assertEqual(metadata_df.iloc[1]["patch_height"], 10)

    def test_halves_horizontal_odd_height_no_pixel_loss(self):
        """Horizontal split on odd height should not lose pixels."""
        # Create 21x10 image (height=21, width=10, channels=2)
        self._write_inputs(shape=(21, 10, 2))
        config = _make_config("halves", "horizontal")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()

        codex_patches = CodexPatches(codex_image, config, args)
        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # Should have 2 patches
        self.assertEqual(len(metadata_df), 2)

        # Top half: height=10, Bottom half: height=11
        # Verify no pixels lost
        total_height = metadata_df.iloc[0]["patch_height"] + metadata_df.iloc[1]["patch_height"]
        self.assertEqual(total_height, 21, "Total height should equal original image height")

        # Top patch starts at y=0
        self.assertEqual(metadata_df.iloc[0]["y_start"], 0)
        # Bottom patch starts where top ends
        self.assertEqual(metadata_df.iloc[1]["y_start"], metadata_df.iloc[0]["patch_height"])

        # Both patches should span full width
        self.assertEqual(metadata_df.iloc[0]["patch_width"], 10)
        self.assertEqual(metadata_df.iloc[1]["patch_width"], 10)


class TestQuartersPatchGeneration(unittest.TestCase):
    """Tests for _generate_quarters_patches() coordinate calculation."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = self.tmpdir.name
        self.out_dir = os.path.join(self.tmpdir.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def _write_inputs(self, shape):
        """Write test image and antibodies file."""
        write_tiny_ome_tiff(os.path.join(self.data_dir, "img.ome.tiff"), shape=shape)
        write_antibodies_tsv(
            os.path.join(self.data_dir, "antibodies.tsv"),
            [f"ch{i}" for i in range(shape[2])],
        )

    # Task 1.4: Quarters patch generation tests
    def test_quarters_generates_four_patches(self):
        """Quarters split should generate exactly 4 patches."""
        # Create 20x20 image (height=20, width=20, channels=2)
        self._write_inputs(shape=(20, 20, 2))
        config = _make_config("quarters")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()

        codex_patches = CodexPatches(codex_image, config, args)
        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # Should have exactly 4 patches
        self.assertEqual(len(metadata_df), 4, "Quarters split should generate 4 patches")

    def test_quarters_coordinates_correct(self):
        """Verify correct coordinates for quarters split."""
        # Create 20x20 image (height=20, width=20, channels=2)
        self._write_inputs(shape=(20, 20, 2))
        config = _make_config("quarters")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()

        codex_patches = CodexPatches(codex_image, config, args)
        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # For 20x20 image, expect 2x2 grid of 10x10 patches
        # Patch order: Top-Left, Top-Right, Bottom-Left, Bottom-Right

        # Patch 0 (Top-Left): x_start=0, y_start=0, width=10, height=10
        self.assertEqual(metadata_df.iloc[0]["x_start"], 0)
        self.assertEqual(metadata_df.iloc[0]["y_start"], 0)
        self.assertEqual(metadata_df.iloc[0]["patch_width"], 10)
        self.assertEqual(metadata_df.iloc[0]["patch_height"], 10)

        # Patch 1 (Top-Right): x_start=10, y_start=0, width=10, height=10
        self.assertEqual(metadata_df.iloc[1]["x_start"], 10)
        self.assertEqual(metadata_df.iloc[1]["y_start"], 0)
        self.assertEqual(metadata_df.iloc[1]["patch_width"], 10)
        self.assertEqual(metadata_df.iloc[1]["patch_height"], 10)

        # Patch 2 (Bottom-Left): x_start=0, y_start=10, width=10, height=10
        self.assertEqual(metadata_df.iloc[2]["x_start"], 0)
        self.assertEqual(metadata_df.iloc[2]["y_start"], 10)
        self.assertEqual(metadata_df.iloc[2]["patch_width"], 10)
        self.assertEqual(metadata_df.iloc[2]["patch_height"], 10)

        # Patch 3 (Bottom-Right): x_start=10, y_start=10, width=10, height=10
        self.assertEqual(metadata_df.iloc[3]["x_start"], 10)
        self.assertEqual(metadata_df.iloc[3]["y_start"], 10)
        self.assertEqual(metadata_df.iloc[3]["patch_width"], 10)
        self.assertEqual(metadata_df.iloc[3]["patch_height"], 10)

    def test_quarters_odd_dimensions_no_pixel_loss(self):
        """Quarters split on odd dimensions should not lose pixels."""
        # Create 21x21 image (height=21, width=21, channels=2)
        self._write_inputs(shape=(21, 21, 2))
        config = _make_config("quarters")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()

        codex_patches = CodexPatches(codex_image, config, args)
        metadata_df = pd.DataFrame(codex_patches.get_patches_metadata())

        # Should have 4 patches
        self.assertEqual(len(metadata_df), 4)

        # Verify sum of widths covers full width
        # First two patches are top row
        top_row_width = metadata_df.iloc[0]["patch_width"] + metadata_df.iloc[1]["patch_width"]
        self.assertEqual(top_row_width, 21, "Top row width should equal original image width")

        # Last two patches are bottom row
        bottom_row_width = metadata_df.iloc[2]["patch_width"] + metadata_df.iloc[3]["patch_width"]
        self.assertEqual(
            bottom_row_width, 21, "Bottom row width should equal original image width"
        )

        # Verify sum of heights covers full height
        # Patches 0 and 2 are left column
        left_col_height = metadata_df.iloc[0]["patch_height"] + metadata_df.iloc[2]["patch_height"]
        self.assertEqual(
            left_col_height, 21, "Left column height should equal original image height"
        )

        # Patches 1 and 3 are right column
        right_col_height = metadata_df.iloc[1]["patch_height"] + metadata_df.iloc[3]["patch_height"]
        self.assertEqual(
            right_col_height, 21, "Right column height should equal original image height"
        )

        # Verify coordinates are correct
        # Top-left should start at (0, 0)
        self.assertEqual(metadata_df.iloc[0]["x_start"], 0)
        self.assertEqual(metadata_df.iloc[0]["y_start"], 0)

        # Top-right should start at (first_width, 0)
        self.assertEqual(metadata_df.iloc[1]["x_start"], metadata_df.iloc[0]["patch_width"])
        self.assertEqual(metadata_df.iloc[1]["y_start"], 0)

        # Bottom-left should start at (0, first_height)
        self.assertEqual(metadata_df.iloc[2]["x_start"], 0)
        self.assertEqual(metadata_df.iloc[2]["y_start"], metadata_df.iloc[0]["patch_height"])

        # Bottom-right should start at (first_width, first_height)
        self.assertEqual(metadata_df.iloc[3]["x_start"], metadata_df.iloc[0]["patch_width"])
        self.assertEqual(metadata_df.iloc[3]["y_start"], metadata_df.iloc[0]["patch_height"])


class TestMaskMerging(unittest.TestCase):
    """Tests for _merge_segmentation_masks() coordinate placement and label handling."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.out_dir = self.tmpdir.name

    def _create_minimal_patches_object(self):
        """Create minimal CodexPatches-like object to call _merge_segmentation_masks."""
        # Create a stub object that has the method
        patches = SimpleNamespace()
        # Bind the method from CodexPatches to our stub
        patches._merge_segmentation_masks = CodexPatches._merge_segmentation_masks.__get__(
            patches, type(patches)
        )
        return patches

    # Task 2.1: Two-patch merge coordinate tests

    def test_merge_two_patches_places_masks_at_correct_coordinates(self):
        """Masks should be placed at x_start, y_start positions in merged output."""
        patches = self._create_minimal_patches_object()

        # Create two small masks with known cell positions
        # Patch 0: 10x10 mask with cell at local position (5, 5)
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[5, 5] = 1  # Cell label 1 at position (5, 5)

        # Patch 1: 10x10 mask at offset (0, 10) with cell at local position (3, 3)
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[3, 3] = 1  # Cell label 1 at position (3, 3) - will become (3, 13) after merge

        # Create metadata - patches placed side by side horizontally
        metadata_df = pd.DataFrame(
            [
                {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
                {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
            ]
        )

        seg_results = [
            {"cell_matched_mask": mask0},
            {"cell_matched_mask": mask1},
        ]

        informative_indices = [0, 1]
        result_key = "cell_matched_mask"
        image_shape = (10, 20)  # 10 rows, 20 columns

        merged = patches._merge_segmentation_masks(
            metadata_df, informative_indices, seg_results, result_key, image_shape
        )

        # Verify shape
        self.assertEqual(merged.shape, (10, 20))

        # Patch 0 cell should be at global position (5, 5)
        self.assertGreater(merged[5, 5], 0, "Cell from patch 0 should exist at (5, 5)")

        # Patch 1 cell should be at global position (3, 13) = (3, 3+10)
        self.assertGreater(merged[3, 13], 0, "Cell from patch 1 should exist at (3, 13)")

        # All other positions should be 0 (background)
        cells_found = np.count_nonzero(merged)
        self.assertEqual(cells_found, 2, "Should have exactly 2 cells in merged mask")

    def test_merge_horizontal_patches_coordinate_placement(self):
        """Test merge with vertically stacked patches (horizontal split)."""
        patches = self._create_minimal_patches_object()

        # Patch 0: at (0, 0), cell at local (2, 2)
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[2, 2] = 1

        # Patch 1: at (10, 0) - below patch 0, cell at local (4, 4)
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[4, 4] = 1

        # Create metadata - patches stacked vertically
        metadata_df = pd.DataFrame(
            [
                {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
                {"x_start": 0, "y_start": 10, "patch_width": 10, "patch_height": 10},
            ]
        )

        seg_results = [
            {"cell_matched_mask": mask0},
            {"cell_matched_mask": mask1},
        ]

        informative_indices = [0, 1]
        result_key = "cell_matched_mask"
        image_shape = (20, 10)  # 20 rows, 10 columns

        merged = patches._merge_segmentation_masks(
            metadata_df, informative_indices, seg_results, result_key, image_shape
        )

        # Verify shape
        self.assertEqual(merged.shape, (20, 10))

        # Patch 0 cell should be at global position (2, 2)
        self.assertGreater(merged[2, 2], 0, "Cell from patch 0 should exist at (2, 2)")

        # Patch 1 cell should be at global position (14, 4) = (4+10, 4)
        self.assertGreater(merged[14, 4], 0, "Cell from patch 1 should exist at (14, 4)")

        # Should have exactly 2 cells
        cells_found = np.count_nonzero(merged)
        self.assertEqual(cells_found, 2, "Should have exactly 2 cells in merged mask")

    # Task 2.2: Label reassignment tests

    def test_merge_two_patches_reassigns_labels_uniquely(self):
        """Labels should be reassigned to avoid conflicts after merge."""
        patches = self._create_minimal_patches_object()

        # Patch 0: cells with labels 1, 2, 3
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[1, 1] = 1
        mask0[2, 2] = 2
        mask0[3, 3] = 3

        # Patch 1: cells with labels 1, 2 (same local labels!)
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[4, 4] = 1
        mask1[5, 5] = 2

        metadata_df = pd.DataFrame(
            [
                {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
                {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
            ]
        )

        seg_results = [
            {"cell_matched_mask": mask0},
            {"cell_matched_mask": mask1},
        ]

        informative_indices = [0, 1]
        result_key = "cell_matched_mask"
        image_shape = (10, 20)

        merged = patches._merge_segmentation_masks(
            metadata_df, informative_indices, seg_results, result_key, image_shape
        )

        # After merge: should have 5 unique labels (1, 2, 3, 4, 5)
        unique_labels = np.unique(merged)
        # Remove background (0)
        unique_labels = unique_labels[unique_labels > 0]

        self.assertEqual(
            len(unique_labels), 5, "Should have 5 unique non-zero labels after merge"
        )

        # Verify labels are consecutive starting from 1
        expected_labels = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        np.testing.assert_array_equal(
            unique_labels, expected_labels, "Labels should be 1, 2, 3, 4, 5"
        )

    def test_merge_preserves_relative_label_order_within_patch(self):
        """Relative order of labels within a patch should be preserved."""
        patches = self._create_minimal_patches_object()

        # Patch 0: labels 1, 2, 3
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[1, 1] = 1
        mask0[2, 2] = 2
        mask0[3, 3] = 3

        # Patch 1: labels 1, 2 -> should become 4, 5 (offset by max of patch 0)
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[4, 4] = 1
        mask1[5, 5] = 2

        metadata_df = pd.DataFrame(
            [
                {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
                {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
            ]
        )

        seg_results = [
            {"cell_matched_mask": mask0},
            {"cell_matched_mask": mask1},
        ]

        informative_indices = [0, 1]
        result_key = "cell_matched_mask"
        image_shape = (10, 20)

        merged = patches._merge_segmentation_masks(
            metadata_df, informative_indices, seg_results, result_key, image_shape
        )

        # Patch 0 cells should keep labels 1, 2, 3
        self.assertEqual(merged[1, 1], 1, "Patch 0, cell 1 should have label 1")
        self.assertEqual(merged[2, 2], 2, "Patch 0, cell 2 should have label 2")
        self.assertEqual(merged[3, 3], 3, "Patch 0, cell 3 should have label 3")

        # Patch 1 cells should become labels 4, 5
        self.assertEqual(merged[4, 14], 4, "Patch 1, cell 1 should have label 4")
        self.assertEqual(merged[5, 15], 5, "Patch 1, cell 2 should have label 5")

    def test_merge_no_label_conflicts_verified(self):
        """Verify no duplicate labels exist in merged mask."""
        patches = self._create_minimal_patches_object()

        # Create multiple patches with overlapping local labels
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[1, 1] = 1
        mask0[2, 2] = 2

        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[1, 1] = 1  # Same label as patch 0
        mask1[2, 2] = 2

        mask2 = np.zeros((10, 10), dtype=np.uint32)
        mask2[1, 1] = 1  # Same label again
        mask2[2, 2] = 2

        metadata_df = pd.DataFrame(
            [
                {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
                {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
                {"x_start": 20, "y_start": 0, "patch_width": 10, "patch_height": 10},
            ]
        )

        seg_results = [
            {"cell_matched_mask": mask0},
            {"cell_matched_mask": mask1},
            {"cell_matched_mask": mask2},
        ]

        informative_indices = [0, 1, 2]
        result_key = "cell_matched_mask"
        image_shape = (10, 30)

        merged = patches._merge_segmentation_masks(
            metadata_df, informative_indices, seg_results, result_key, image_shape
        )

        # Count total cells (non-zero pixels)
        total_cells = np.count_nonzero(merged)
        self.assertEqual(total_cells, 6, "Should have 6 cells total")

        # Count unique labels (excluding 0)
        unique_labels = np.unique(merged)
        unique_labels = unique_labels[unique_labels > 0]
        self.assertEqual(
            len(unique_labels), 6, "Should have 6 unique labels (no conflicts)"
        )

        # Verify labels are consecutive 1-6
        expected_labels = np.arange(1, 7, dtype=np.uint32)
        np.testing.assert_array_equal(
            unique_labels, expected_labels, "Labels should be consecutive 1-6"
        )

    # Task 2.3: Four-patch merge tests

    def test_merge_four_patches_coordinate_placement(self):
        """Test merge with 4 patches (quarters mode)."""
        # Create 4 patches arranged in 2x2 grid
        # Each 10x10, total image 20x20
        #
        # Patch layout:
        # [0: TL] [1: TR]
        # [2: BL] [3: BR]
        #
        # Place one cell in each patch at local (5, 5)
        # Expected global positions:
        # - Patch 0: (5, 5)
        # - Patch 1: (5, 15)    # x offset by 10
        # - Patch 2: (15, 5)    # y offset by 10
        # - Patch 3: (15, 15)   # both offsets

        metadata_df = pd.DataFrame([
            {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},   # TL
            {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},  # TR
            {"x_start": 0, "y_start": 10, "patch_width": 10, "patch_height": 10},  # BL
            {"x_start": 10, "y_start": 10, "patch_width": 10, "patch_height": 10}, # BR
        ])

        # Create 4 masks, each with one cell at local position (5, 5)
        seg_results = []
        for i in range(4):
            mask = np.zeros((10, 10), dtype=np.uint32)
            mask[5, 5] = 1  # One cell per patch
            seg_results.append({"cell_matched_mask": mask})

        # Create minimal patches object to access _merge_segmentation_masks method
        patches = CodexPatches.__new__(CodexPatches)
        patches.codex_image = SimpleNamespace(image_shape=(20, 20))

        merged = patches._merge_segmentation_masks(
            metadata_df=metadata_df,
            informative_indices=[0, 1, 2, 3],
            seg_results=seg_results,
            result_key="cell_matched_mask",
            image_shape=(20, 20),
        )

        # Verify shape
        self.assertEqual(merged.shape, (20, 20))

        # Verify we have 4 unique cells (labels 1-4)
        unique_labels = np.unique(merged)
        self.assertEqual(len(unique_labels) - 1, 4)  # Exclude background 0
        self.assertTrue(np.array_equal(unique_labels, [0, 1, 2, 3, 4]))

        # Verify each cell is at expected global position
        expected_positions = [
            (5, 5),    # Patch 0: TL
            (5, 15),   # Patch 1: TR
            (15, 5),   # Patch 2: BL
            (15, 15),  # Patch 3: BR
        ]

        for label, expected_pos in enumerate(expected_positions, start=1):
            y_coords, x_coords = np.where(merged == label)
            self.assertEqual(len(y_coords), 1, f"Label {label} should appear exactly once")
            actual_pos = (y_coords[0], x_coords[0])
            self.assertEqual(
                actual_pos,
                expected_pos,
                f"Label {label} at {actual_pos}, expected {expected_pos}",
            )

    def test_merge_four_patches_all_quadrants_populated(self):
        """Verify all 4 quadrants of merged mask contain expected cells."""
        # Create 4 patches with different cell counts
        metadata_df = pd.DataFrame([
            {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},   # TL
            {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},  # TR
            {"x_start": 0, "y_start": 10, "patch_width": 10, "patch_height": 10},  # BL
            {"x_start": 10, "y_start": 10, "patch_width": 10, "patch_height": 10}, # BR
        ])

        seg_results = []

        # Patch 0 (TL): 2 cells
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[2, 2] = 1
        mask0[7, 7] = 2
        seg_results.append({"cell_matched_mask": mask0})

        # Patch 1 (TR): 1 cell
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[3, 3] = 1
        seg_results.append({"cell_matched_mask": mask1})

        # Patch 2 (BL): 3 cells
        mask2 = np.zeros((10, 10), dtype=np.uint32)
        mask2[1, 1] = 1
        mask2[4, 4] = 2
        mask2[8, 8] = 3
        seg_results.append({"cell_matched_mask": mask2})

        # Patch 3 (BR): 2 cells
        mask3 = np.zeros((10, 10), dtype=np.uint32)
        mask3[5, 5] = 1
        mask3[9, 9] = 2
        seg_results.append({"cell_matched_mask": mask3})

        # Create minimal patches object to access _merge_segmentation_masks method
        patches = CodexPatches.__new__(CodexPatches)
        patches.codex_image = SimpleNamespace(image_shape=(20, 20))

        merged = patches._merge_segmentation_masks(
            metadata_df=metadata_df,
            informative_indices=[0, 1, 2, 3],
            seg_results=seg_results,
            result_key="cell_matched_mask",
            image_shape=(20, 20),
        )

        # Verify total cell count: 2 + 1 + 3 + 2 = 8 cells
        unique_labels = np.unique(merged)
        self.assertEqual(len(unique_labels) - 1, 8)  # Exclude background 0

        # Verify each quadrant has expected number of cells
        # TL quadrant (0:10, 0:10): 2 cells
        tl_quadrant = merged[0:10, 0:10]
        tl_cells = len(np.unique(tl_quadrant)) - 1
        self.assertEqual(tl_cells, 2, "TL quadrant should have 2 cells")

        # TR quadrant (0:10, 10:20): 1 cell
        tr_quadrant = merged[0:10, 10:20]
        tr_cells = len(np.unique(tr_quadrant)) - 1
        self.assertEqual(tr_cells, 1, "TR quadrant should have 1 cell")

        # BL quadrant (10:20, 0:10): 3 cells
        bl_quadrant = merged[10:20, 0:10]
        bl_cells = len(np.unique(bl_quadrant)) - 1
        self.assertEqual(bl_cells, 3, "BL quadrant should have 3 cells")

        # BR quadrant (10:20, 10:20): 2 cells
        br_quadrant = merged[10:20, 10:20]
        br_cells = len(np.unique(br_quadrant)) - 1
        self.assertEqual(br_cells, 2, "BR quadrant should have 2 cells")

    def test_merge_four_patches_labels_unique(self):
        """All labels should be unique after merging 4 patches."""
        # Each patch has cells labeled 1, 2
        # After merge: should have 8 unique labels (1-8)
        metadata_df = pd.DataFrame([
            {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
            {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
            {"x_start": 0, "y_start": 10, "patch_width": 10, "patch_height": 10},
            {"x_start": 10, "y_start": 10, "patch_width": 10, "patch_height": 10},
        ])

        seg_results = []
        for i in range(4):
            mask = np.zeros((10, 10), dtype=np.uint32)
            mask[3, 3] = 1
            mask[6, 6] = 2
            seg_results.append({"cell_matched_mask": mask})

        # Create minimal patches object to access _merge_segmentation_masks method
        patches = CodexPatches.__new__(CodexPatches)
        patches.codex_image = SimpleNamespace(image_shape=(20, 20))

        merged = patches._merge_segmentation_masks(
            metadata_df=metadata_df,
            informative_indices=[0, 1, 2, 3],
            seg_results=seg_results,
            result_key="cell_matched_mask",
            image_shape=(20, 20),
        )

        # Verify all labels are unique
        unique_labels = np.unique(merged)
        expected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 0 = background, 1-8 = cells
        self.assertTrue(
            np.array_equal(unique_labels, expected_labels),
            f"Expected labels {expected_labels}, got {unique_labels.tolist()}",
        )

        # Verify each label appears exactly at one position
        for label in range(1, 9):
            count = np.sum(merged == label)
            self.assertEqual(count, 1, f"Label {label} should appear exactly once")

    # Task 2.4: GPU 3D mask handling tests

    def test_merge_handles_gpu_3d_masks(self):
        """GPU repair outputs (1, H, W) masks that should be squeezed."""
        # Create masks with shape (1, 10, 10) instead of (10, 10)
        metadata_df = pd.DataFrame([
            {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
            {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
        ])

        # Create 3D GPU-style masks
        mask0_2d = np.zeros((10, 10), dtype=np.uint32)
        mask0_2d[5, 5] = 1
        mask0_3d = mask0_2d[np.newaxis, ...]  # Shape: (1, 10, 10)

        mask1_2d = np.zeros((10, 10), dtype=np.uint32)
        mask1_2d[3, 3] = 1
        mask1_3d = mask1_2d[np.newaxis, ...]  # Shape: (1, 10, 10)

        seg_results = [
            {"cell_matched_mask": mask0_3d},
            {"cell_matched_mask": mask1_3d},
        ]

        # Create minimal patches object to access _merge_segmentation_masks method
        patches = CodexPatches.__new__(CodexPatches)
        patches.codex_image = SimpleNamespace(image_shape=(10, 20))

        merged = patches._merge_segmentation_masks(
            metadata_df=metadata_df,
            informative_indices=[0, 1],
            seg_results=seg_results,
            result_key="cell_matched_mask",
            image_shape=(10, 20),
        )

        # Verify merge completed successfully
        self.assertIsNotNone(merged)
        self.assertEqual(merged.shape, (10, 20))

        # Verify we have 2 unique cells
        unique_labels = np.unique(merged)
        self.assertEqual(len(unique_labels) - 1, 2)  # Exclude background 0

        # Verify cells are at expected positions
        # Patch 0 cell at (5, 5)
        y0, x0 = np.where(merged == 1)
        self.assertEqual(len(y0), 1)
        self.assertEqual((y0[0], x0[0]), (5, 5))

        # Patch 1 cell at (3, 13) (local 3,3 + x_offset 10)
        y1, x1 = np.where(merged == 2)
        self.assertEqual(len(y1), 1)
        self.assertEqual((y1[0], x1[0]), (3, 13))

    def test_merge_handles_mixed_2d_and_3d_masks(self):
        """Merge should handle mix of 2D and 3D (GPU) masks."""
        # Patch 0: 2D mask (H, W)
        # Patch 1: 3D mask (1, H, W) from GPU
        metadata_df = pd.DataFrame([
            {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
            {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
        ])

        # Patch 0: Standard 2D mask
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[4, 4] = 1

        # Patch 1: GPU-style 3D mask
        mask1_2d = np.zeros((10, 10), dtype=np.uint32)
        mask1_2d[6, 6] = 1
        mask1_3d = mask1_2d[np.newaxis, ...]

        seg_results = [
            {"cell_matched_mask": mask0},
            {"cell_matched_mask": mask1_3d},
        ]

        # Create minimal patches object to access _merge_segmentation_masks method
        patches = CodexPatches.__new__(CodexPatches)
        patches.codex_image = SimpleNamespace(image_shape=(10, 20))

        merged = patches._merge_segmentation_masks(
            metadata_df=metadata_df,
            informative_indices=[0, 1],
            seg_results=seg_results,
            result_key="cell_matched_mask",
            image_shape=(10, 20),
        )

        # Verify both cells were merged correctly
        self.assertIsNotNone(merged)
        unique_labels = np.unique(merged)
        self.assertEqual(len(unique_labels) - 1, 2)

        # Verify cell positions
        y0, x0 = np.where(merged == 1)
        self.assertEqual((y0[0], x0[0]), (4, 4), "2D mask cell at wrong position")

        y1, x1 = np.where(merged == 2)
        self.assertEqual((y1[0], x1[0]), (6, 16), "3D mask cell at wrong position")

    def test_merge_handles_empty_gpu_mask(self):
        """Empty GPU mask (1, H, W) with all zeros should not cause issues."""
        # Create (1, 10, 10) mask with all zeros
        metadata_df = pd.DataFrame([
            {"x_start": 0, "y_start": 0, "patch_width": 10, "patch_height": 10},
            {"x_start": 10, "y_start": 0, "patch_width": 10, "patch_height": 10},
        ])

        # Patch 0: Empty 3D GPU mask
        empty_mask_3d = np.zeros((1, 10, 10), dtype=np.uint32)

        # Patch 1: Normal mask with one cell
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[5, 5] = 1

        seg_results = [
            {"cell_matched_mask": empty_mask_3d},
            {"cell_matched_mask": mask1},
        ]

        # Create minimal patches object to access _merge_segmentation_masks method
        patches = CodexPatches.__new__(CodexPatches)
        patches.codex_image = SimpleNamespace(image_shape=(10, 20))

        merged = patches._merge_segmentation_masks(
            metadata_df=metadata_df,
            informative_indices=[0, 1],
            seg_results=seg_results,
            result_key="cell_matched_mask",
            image_shape=(10, 20),
        )

        # Verify merge completed without error
        self.assertIsNotNone(merged)

        # Verify only the second patch's cell is present
        unique_labels = np.unique(merged)
        self.assertEqual(len(unique_labels) - 1, 1)  # Only 1 cell

        # Verify cell position (patch 1 at local 5,5 with x_offset 10)
        y, x = np.where(merged == 1)
        self.assertEqual(len(y), 1)
        self.assertEqual((y[0], x[0]), (5, 15))


class TestHalvesIntegration(unittest.TestCase):
    """Integration tests for halves mode with cell profiling."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = self.tmpdir.name
        self.out_dir = os.path.join(self.tmpdir.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def _write_inputs(self, shape):
        """Write test image and antibodies file."""
        write_tiny_ome_tiff(os.path.join(self.data_dir, "img.ome.tiff"), shape=shape)
        write_antibodies_tsv(
            os.path.join(self.data_dir, "antibodies.tsv"),
            [f"ch{i}" for i in range(shape[2])],
        )

    # Task 3.1: Halves cell profiling integration test
    def test_halves_cell_profiling_global_coordinates(self):
        """Test that cell profiling with halves mode produces correct global coordinates.

        This test verifies the complete flow:
        1. Image is split into 2 halves
        2. Each patch has cells with known positions
        3. Cell profiling merges results with coordinate transformation
        4. Output CSV has correct global coordinates
        """
        from aegle.cell_profiling import run_cell_profiling

        # Create a FakeCodexPatches for cell profiling testing
        class FakeCodexPatches:
            """Minimal stub to drive run_cell_profiling without heavy IO."""

            def __init__(self, antibody_df, patches_metadata, seg_results, channel_patches):
                self.antibody_df = antibody_df
                self._patches_metadata = patches_metadata
                self.repaired_seg_res_batch = seg_results
                self._channel_patches = channel_patches

            def get_patches_metadata(self):
                return self._patches_metadata

            def get_all_channel_patch(self, patch_index: int):
                return self._channel_patches[patch_index]

            def is_using_disk_based_patches(self):
                return False

        # Create test data
        antibody_df = pd.DataFrame({"antibody_name": ["marker0", "marker1"]})

        # Two patches: left half (x_start=0) and right half (x_start=10)
        patches_metadata = pd.DataFrame([
            {
                "patch_id": 0, "patch_index": 0,
                "x_start": 0, "y_start": 0,
                "patch_width": 10, "patch_height": 10,
                "is_informative": True
            },
            {
                "patch_id": 1, "patch_index": 1,
                "x_start": 10, "y_start": 0,
                "patch_width": 10, "patch_height": 10,
                "is_informative": True
            },
        ])

        # Create masks with cells at known LOCAL positions
        # Patch 0: cell at local (3, 4) - should become global (3, 4)
        mask0 = np.zeros((10, 10), dtype=np.uint32)
        mask0[3, 4] = 1

        # Patch 1: cell at local (5, 6) - should become global (5, 16) due to x_start=10
        mask1 = np.zeros((10, 10), dtype=np.uint32)
        mask1[5, 6] = 1

        seg_results = [
            {
                "cell_matched_mask": mask0,
                "nucleus_matched_mask": mask0,
            },
            {
                "cell_matched_mask": mask1,
                "nucleus_matched_mask": mask1,
            },
        ]

        # Create channel patches (10x10 with 2 channels)
        channel_patches = [
            np.ones((10, 10, 2), dtype=np.float32) * 100,
            np.ones((10, 10, 2), dtype=np.float32) * 200,
        ]

        codex = FakeCodexPatches(antibody_df, patches_metadata, seg_results, channel_patches)

        args = SimpleNamespace(out_dir=self.out_dir, data_dir=".")
        config = {"patching": {"split_mode": "halves"}}

        run_cell_profiling(codex, config, args)

        # Verify output files exist
        profiling_dir = os.path.join(self.out_dir, "cell_profiling")
        meta_path = os.path.join(profiling_dir, "cell_metadata.csv")
        self.assertTrue(os.path.exists(meta_path), "cell_metadata.csv should exist")

        # Load and verify coordinates
        meta = pd.read_csv(meta_path)
        self.assertEqual(len(meta), 2, "Should have 2 cells total")

        # Cell from patch 0 should have global coordinates (3, 4)
        # Note: In the metadata, 'y' is the row (first index), 'x' is the column (second index)
        # Cell centroid calculation may vary, so we check the patch-level coordinate transformation
        patch0_cells = meta[meta["patch_id"] == 0]
        self.assertEqual(len(patch0_cells), 1)

        patch1_cells = meta[meta["patch_id"] == 1]
        self.assertEqual(len(patch1_cells), 1)

        # Verify patch 1 cells have x-coordinates offset by 10
        # The x coordinate should be local_x + x_start = 6 + 10 = 16 (for centroid around column 6)
        # Check that x coordinate is in the right half (>= 10)
        self.assertTrue(
            (patch1_cells["x"] >= 10).all(),
            "Patch 1 cells should have x >= 10 after coordinate transformation"
        )

        # Verify unique cell IDs across patches
        self.assertEqual(
            len(meta["cell_mask_id"].unique()), 2,
            "Should have 2 unique cell_mask_id values"
        )


if __name__ == "__main__":
    unittest.main()
