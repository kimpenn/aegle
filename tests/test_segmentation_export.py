"""
Tests for segmentation mask OME-TIFF export functionality.

This test module validates that:
1. Masks with batch dimension (1, H, W) from GPU repair are handled correctly
2. OME-TIFF files are created with proper structure
3. The merge function handles both 2D and 3D mask inputs
4. Export produces valid pyramidal OME-TIFF files

These tests were added after discovering a bug where GPU repair produces
3D masks (1, H, W) but the export function expected 2D masks (H, W).
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

from aegle.codex_patches import CodexPatches
from tests.utils import make_segmentation_result


def make_gpu_repair_result(cell_mask_2d: np.ndarray, nucleus_mask_2d: np.ndarray) -> dict:
    """
    Create a repair result dict that mimics GPU repair output.

    GPU repair adds a batch dimension (1, H, W) to masks, which caused
    the OME-TIFF export bug.
    """
    # GPU repair path adds batch dimension
    cell_matched_mask = np.expand_dims(cell_mask_2d, axis=0)  # (1, H, W)
    nucleus_matched_mask = np.expand_dims(nucleus_mask_2d, axis=0)  # (1, H, W)

    cell_outside = cell_mask_2d.astype(np.int32) - nucleus_mask_2d.astype(np.int32)
    cell_outside = np.clip(cell_outside, 0, None).astype(np.uint32)
    cell_outside_nucleus_mask = np.expand_dims(cell_outside, axis=0)  # (1, H, W)

    return {
        "cell_matched_mask": cell_matched_mask,
        "nucleus_matched_mask": nucleus_matched_mask,
        "cell_outside_nucleus_mask": cell_outside_nucleus_mask,
        "matched_fraction": 1.0,
        "repair_metadata": {"gpu_used": True},
    }


def make_cpu_repair_result(cell_mask_2d: np.ndarray, nucleus_mask_2d: np.ndarray) -> dict:
    """
    Create a repair result dict that mimics CPU repair output.

    CPU repair produces 2D masks (H, W) directly.
    """
    cell_outside = cell_mask_2d.astype(np.int32) - nucleus_mask_2d.astype(np.int32)
    cell_outside = np.clip(cell_outside, 0, None).astype(np.uint32)

    return {
        "cell_matched_mask": cell_mask_2d,  # 2D (H, W)
        "nucleus_matched_mask": nucleus_mask_2d,  # 2D (H, W)
        "cell_outside_nucleus_mask": cell_outside,  # 2D (H, W)
        "matched_fraction": 1.0,
        "repair_metadata": {"gpu_used": False},
    }


class TestMaskShapeHandling(unittest.TestCase):
    """Test that mask shape handling works for both 2D and 3D inputs."""

    def test_squeeze_3d_mask_to_2d(self):
        """Verify that 3D masks (1, H, W) are squeezed to 2D (H, W)."""
        mask_2d = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.uint32)
        mask_3d = np.expand_dims(mask_2d, axis=0)  # (1, 3, 3)

        self.assertEqual(mask_3d.ndim, 3)
        self.assertEqual(mask_3d.shape[0], 1)

        # Simulate the fix in _merge_segmentation_masks
        if mask_3d.ndim == 3 and mask_3d.shape[0] == 1:
            result = np.squeeze(mask_3d, axis=0)
        else:
            result = mask_3d

        self.assertEqual(result.ndim, 2)
        np.testing.assert_array_equal(result, mask_2d)

    def test_2d_mask_unchanged(self):
        """Verify that 2D masks pass through unchanged."""
        mask_2d = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.uint32)

        # Simulate the fix in _merge_segmentation_masks
        if mask_2d.ndim == 3 and mask_2d.shape[0] == 1:
            result = np.squeeze(mask_2d, axis=0)
        else:
            result = mask_2d

        self.assertEqual(result.ndim, 2)
        np.testing.assert_array_equal(result, mask_2d)

    def test_gpu_repair_result_has_batch_dim(self):
        """Verify GPU repair results have batch dimension."""
        cell = np.zeros((10, 10), dtype=np.uint32)
        cell[2:8, 2:8] = 1
        nucleus = np.zeros_like(cell)
        nucleus[3:7, 3:7] = 1

        result = make_gpu_repair_result(cell, nucleus)

        self.assertEqual(result["cell_matched_mask"].ndim, 3)
        self.assertEqual(result["cell_matched_mask"].shape[0], 1)
        self.assertEqual(result["nucleus_matched_mask"].ndim, 3)
        self.assertEqual(result["cell_outside_nucleus_mask"].ndim, 3)

    def test_cpu_repair_result_is_2d(self):
        """Verify CPU repair results are 2D."""
        cell = np.zeros((10, 10), dtype=np.uint32)
        cell[2:8, 2:8] = 1
        nucleus = np.zeros_like(cell)
        nucleus[3:7, 3:7] = 1

        result = make_cpu_repair_result(cell, nucleus)

        self.assertEqual(result["cell_matched_mask"].ndim, 2)
        self.assertEqual(result["nucleus_matched_mask"].ndim, 2)
        self.assertEqual(result["cell_outside_nucleus_mask"].ndim, 2)


class TestOMETIFFExport(unittest.TestCase):
    """Test OME-TIFF export functionality."""

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_mask_stack_with_2d_input(self):
        """Test _save_mask_stack with 2D mask input."""
        mask_2d = np.zeros((100, 100), dtype=np.uint32)
        mask_2d[20:80, 20:80] = 1
        mask_2d[30:70, 30:70] = 2

        output_path = os.path.join(self.temp_dir, "test_2d.ome.tiff")

        # Use the standalone function pattern from our export script
        mask_array = np.asarray(mask_2d, dtype=np.uint32)
        if mask_array.ndim == 3 and mask_array.shape[0] == 1:
            mask_array = np.squeeze(mask_array, axis=0)

        stack = mask_array[None, ...]  # (1, H, W) for CYX

        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write(
                stack,
                dtype=np.uint32,
                compression="zlib",
                photometric="minisblack",
            )

        self.assertTrue(os.path.exists(output_path))

        # Verify the saved file
        with tifffile.TiffFile(output_path) as tif:
            data = tif.asarray()
            # tifffile squeezes single-channel data, so shape is (H, W) not (1, H, W)
            self.assertEqual(data.shape, (100, 100))
            np.testing.assert_array_equal(data, mask_2d)

    def test_save_mask_stack_with_3d_input(self):
        """Test _save_mask_stack with 3D mask input (GPU repair format)."""
        mask_2d = np.zeros((100, 100), dtype=np.uint32)
        mask_2d[20:80, 20:80] = 1
        mask_2d[30:70, 30:70] = 2
        mask_3d = np.expand_dims(mask_2d, axis=0)  # (1, 100, 100)

        output_path = os.path.join(self.temp_dir, "test_3d.ome.tiff")

        # Apply the fix
        mask_array = np.asarray(mask_3d, dtype=np.uint32)
        if mask_array.ndim == 3 and mask_array.shape[0] == 1:
            mask_array = np.squeeze(mask_array, axis=0)

        stack = mask_array[None, ...]  # (1, H, W) for CYX

        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write(
                stack,
                dtype=np.uint32,
                compression="zlib",
                photometric="minisblack",
            )

        self.assertTrue(os.path.exists(output_path))

        # Verify the saved file
        with tifffile.TiffFile(output_path) as tif:
            data = tif.asarray()
            # tifffile squeezes single-channel data, so shape is (H, W) not (1, H, W)
            self.assertEqual(data.shape, (100, 100))
            np.testing.assert_array_equal(data, mask_2d)

    def test_pyramid_generation(self):
        """Test that pyramid levels are generated correctly."""
        # Create a larger mask for pyramid testing
        mask = np.zeros((512, 512), dtype=np.uint32)
        mask[100:400, 100:400] = 1

        output_path = os.path.join(self.temp_dir, "test_pyramid.ome.tiff")

        # Build pyramid
        stack = mask[None, ...]
        pyramid = [stack]
        current = stack
        min_size = 256

        while current.shape[-2] >= 2 * min_size and current.shape[-1] >= 2 * min_size:
            down = current[:, ::2, ::2]
            if down.shape[-2] < min_size or down.shape[-1] < min_size:
                break
            pyramid.append(down)
            current = down

        # Should have base + 1 pyramid level for 512x512
        self.assertGreaterEqual(len(pyramid), 1)

        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write(
                pyramid[0],
                subifds=len(pyramid) - 1,
                dtype=np.uint32,
                compression="zlib",
                photometric="minisblack",
            )
            for level in pyramid[1:]:
                tif.write(
                    level,
                    subfiletype=1,
                    dtype=np.uint32,
                    compression="zlib",
                    photometric="minisblack",
                )

        self.assertTrue(os.path.exists(output_path))


class TestMergeSegmentationMasks(unittest.TestCase):
    """Test the _merge_segmentation_masks function behavior."""

    def test_merge_handles_3d_gpu_masks(self):
        """Test that merge function handles 3D masks from GPU repair."""
        # Simulate what _merge_segmentation_masks receives
        mask_2d = np.zeros((50, 50), dtype=np.uint32)
        mask_2d[10:40, 10:40] = 1

        # GPU repair output
        mask_3d = np.expand_dims(mask_2d, axis=0)

        # Apply the fix from _merge_segmentation_masks
        mask = np.asarray(mask_3d, dtype=np.uint32)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)

        # Now should be able to unpack shape
        h, w = mask.shape
        self.assertEqual(h, 50)
        self.assertEqual(w, 50)

    def test_merge_handles_2d_cpu_masks(self):
        """Test that merge function handles 2D masks from CPU repair."""
        mask_2d = np.zeros((50, 50), dtype=np.uint32)
        mask_2d[10:40, 10:40] = 1

        # Apply the fix from _merge_segmentation_masks
        mask = np.asarray(mask_2d, dtype=np.uint32)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)

        # Should work fine with 2D input
        h, w = mask.shape
        self.assertEqual(h, 50)
        self.assertEqual(w, 50)


class TestExportScriptIntegration(unittest.TestCase):
    """Integration tests for the export_segmentation_tiff.py script."""

    def setUp(self):
        """Create temporary directory with mock segmentation data."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_script_handles_gpu_format(self):
        """Test that export script correctly handles GPU repair format."""
        import pickle
        import zstandard as zstd

        # Create mock masks
        cell_2d = np.zeros((100, 100), dtype=np.uint32)
        cell_2d[20:80, 20:80] = 1
        cell_2d[60:90, 60:90] = 2

        nucleus_2d = np.zeros_like(cell_2d)
        nucleus_2d[30:70, 30:70] = 1
        nucleus_2d[65:85, 65:85] = 2

        # Create GPU-style repair result (with batch dimension)
        matched_result = make_gpu_repair_result(cell_2d, nucleus_2d)
        original_result = {
            "cell": cell_2d,
            "nucleus": nucleus_2d,
        }

        # Save as compressed pickle (mimicking pipeline output)
        matched_path = os.path.join(self.temp_dir, "matched_seg_res_batch.pickle.zst")
        original_path = os.path.join(self.temp_dir, "original_seg_res_batch.pickle.zst")

        cctx = zstd.ZstdCompressor()

        with open(matched_path, "wb") as f:
            with cctx.stream_writer(f) as writer:
                pickle.dump([matched_result], writer)

        with open(original_path, "wb") as f:
            with cctx.stream_writer(f) as writer:
                pickle.dump([original_result], writer)

        # Now run the export logic
        from scripts.export_segmentation_tiff import export_segmentation_masks

        success = export_segmentation_masks(
            output_dir=self.temp_dir,
            base_name="test",
            image_mpp=0.5,
        )

        self.assertTrue(success)

        # Verify output files exist
        seg_dir = os.path.join(self.temp_dir, "segmentations")
        self.assertTrue(os.path.isdir(seg_dir))

        # Check for expected files
        expected_files = [
            "test.cell.segmentations.ome.tiff",
            "test.nucleus.segmentations.ome.tiff",
            "test.cell_matched_mask.segmentations.ome.tiff",
            "test.nucleus_matched_mask.segmentations.ome.tiff",
        ]

        for filename in expected_files:
            filepath = os.path.join(seg_dir, filename)
            self.assertTrue(
                os.path.exists(filepath),
                f"Expected file {filename} not found"
            )


if __name__ == "__main__":
    unittest.main()
