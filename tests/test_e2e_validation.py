"""
E2E validation tests for D11_0 sample.

Validates:
1. Bug fix for 3D mask shape (GPU repair path)
2. Cell profiling correctness against production baseline
3. Segmentation TIFF correctness against production baseline
4. Patch script equivalence to pipeline output

Run with: RUN_E2E=1 python -m pytest tests/test_e2e_validation.py -v
"""

import os
import unittest
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr

# Constants
BASELINE_DIR = "out/main/main_ft_hb/D11_0"
OPTIMIZED_DIR = "out/test_optimized/main_ft_hb/D11_0"
RTOL = 1e-5
ATOL = 1e-6
MIN_CORRELATION = 0.9999


@unittest.skipUnless(os.environ.get("RUN_E2E"), "Set RUN_E2E=1")
class TestD11_0Validation(unittest.TestCase):
    """E2E validation tests using D11_0 sample."""

    @classmethod
    def setUpClass(cls):
        """Verify test data exists."""
        cls.baseline_dir = BASELINE_DIR
        cls.optimized_dir = OPTIMIZED_DIR

        # Verify baseline exists
        cls.baseline_cell_overview = os.path.join(
            cls.baseline_dir, "cell_profiling", "cell_overview.csv"
        )
        cls.optimized_cell_overview = os.path.join(
            cls.optimized_dir, "cell_profiling", "cell_overview.csv"
        )

        if not os.path.exists(cls.baseline_cell_overview):
            raise unittest.SkipTest(f"Baseline not found: {cls.baseline_cell_overview}")
        if not os.path.exists(cls.optimized_cell_overview):
            raise unittest.SkipTest(f"Optimized not found: {cls.optimized_cell_overview}")

    def test_01_cell_count_matches(self):
        """Cell counts should match exactly."""
        baseline = pd.read_csv(self.baseline_cell_overview)
        optimized = pd.read_csv(self.optimized_cell_overview)

        self.assertEqual(len(baseline), len(optimized),
            f"Cell count mismatch: baseline={len(baseline)}, optimized={len(optimized)}")
        self.assertEqual(len(baseline), 79347, "Expected 79,347 cells")

    def test_02_cell_ids_match(self):
        """Cell IDs should match exactly."""
        baseline = pd.read_csv(self.baseline_cell_overview)
        optimized = pd.read_csv(self.optimized_cell_overview)

        np.testing.assert_array_equal(
            baseline['cell_mask_id'].values,
            optimized['cell_mask_id'].values,
            "Cell IDs do not match"
        )

    def test_03_morphology_correlation(self):
        """Morphology features should have correlation > 0.9999."""
        baseline = pd.read_csv(self.baseline_cell_overview)
        optimized = pd.read_csv(self.optimized_cell_overview)

        morph_cols = ['area']
        for col in morph_cols:
            if col in baseline.columns and col in optimized.columns:
                corr, _ = pearsonr(baseline[col], optimized[col])
                self.assertGreater(corr, MIN_CORRELATION,
                    f"Morphology '{col}' correlation {corr:.10f} < {MIN_CORRELATION}")

    def test_04_marker_intensities_correlation(self):
        """All marker intensities should have correlation > 0.9999."""
        baseline = pd.read_csv(self.baseline_cell_overview)
        optimized = pd.read_csv(self.optimized_cell_overview)

        # Identify marker columns (exclude metadata)
        metadata_cols = {'cell_mask_id', 'y', 'x', 'area'}
        marker_cols = [c for c in baseline.columns if c not in metadata_cols]

        failed_markers = []
        for col in marker_cols:
            if col in optimized.columns:
                corr, _ = pearsonr(baseline[col], optimized[col])
                if corr < MIN_CORRELATION:
                    failed_markers.append((col, corr))

        self.assertEqual(len(failed_markers), 0,
            f"Markers with correlation < {MIN_CORRELATION}: {failed_markers}")

    def test_05_segmentation_tiffs_exist(self):
        """Segmentation TIFFs should exist in both directories."""
        baseline_seg_dir = os.path.join(self.baseline_dir, "segmentations")
        optimized_seg_dir = os.path.join(self.optimized_dir, "segmentations")

        self.assertTrue(os.path.isdir(baseline_seg_dir),
            f"Baseline segmentations dir not found: {baseline_seg_dir}")
        self.assertTrue(os.path.isdir(optimized_seg_dir),
            f"Optimized segmentations dir not found: {optimized_seg_dir}")

        baseline_files = os.listdir(baseline_seg_dir)
        optimized_files = os.listdir(optimized_seg_dir)

        self.assertGreater(len(baseline_files), 0, "No baseline TIFF files")
        self.assertGreater(len(optimized_files), 0, "No optimized TIFF files")

    def test_06_segmentation_tiffs_match(self):
        """Segmentation TIFF arrays should match between baseline and optimized.

        Note: Due to GPU vs CPU numerical differences, we allow small pixel-level
        differences as long as the overall segmentation statistics match.
        """
        baseline_seg_dir = os.path.join(self.baseline_dir, "segmentations")
        optimized_seg_dir = os.path.join(self.optimized_dir, "segmentations")

        # Key masks to compare
        mask_names = [
            "cell.segmentations.ome.tiff",
            "nucleus.segmentations.ome.tiff",
            "cell_matched_mask.segmentations.ome.tiff",
            "nucleus_matched_mask.segmentations.ome.tiff",
        ]

        for mask_name in mask_names:
            with self.subTest(mask=mask_name):
                # Find files matching pattern
                baseline_file = self._find_tiff_file(baseline_seg_dir, mask_name)
                optimized_file = self._find_tiff_file(optimized_seg_dir, mask_name)

                if baseline_file is None:
                    self.skipTest(f"Baseline TIFF not found for {mask_name}")
                if optimized_file is None:
                    self.skipTest(f"Optimized TIFF not found for {mask_name}")

                with tifffile.TiffFile(baseline_file) as tif:
                    baseline_data = tif.asarray()
                with tifffile.TiffFile(optimized_file) as tif:
                    optimized_data = tif.asarray()

                # Compare shapes
                self.assertEqual(baseline_data.shape, optimized_data.shape,
                    f"Shape mismatch for {mask_name}: {baseline_data.shape} vs {optimized_data.shape}")

                # Compare unique cell counts (should be identical)
                baseline_unique = np.unique(baseline_data)
                optimized_unique = np.unique(optimized_data)
                self.assertEqual(len(baseline_unique), len(optimized_unique),
                    f"Unique cell count mismatch for {mask_name}: "
                    f"{len(baseline_unique)} vs {len(optimized_unique)}")

                # Check pixel-level agreement (allow small differences)
                total_pixels = baseline_data.size
                diff_pixels = np.sum(baseline_data != optimized_data)
                diff_pct = diff_pixels / total_pixels * 100

                # Allow up to 0.01% pixel differences (GPU vs CPU numerical variance)
                # The D11_0 validation showed ~0.003% differences are normal
                max_diff_pct = 0.01
                self.assertLess(diff_pct, max_diff_pct,
                    f"Too many pixel differences for {mask_name}: "
                    f"{diff_pixels:,} / {total_pixels:,} ({diff_pct:.6f}%)")

    def _find_tiff_file(self, directory, pattern):
        """Find TIFF file matching pattern in directory."""
        for f in os.listdir(directory):
            if pattern in f:
                return os.path.join(directory, f)
        return None


@unittest.skipUnless(os.environ.get("RUN_E2E"), "Set RUN_E2E=1")
class TestPatchScriptEquivalence(unittest.TestCase):
    """Test that patch script produces identical results to pipeline."""

    def test_patch_script_output_matches_baseline(self):
        """Patch script should produce TIFFs identical to pipeline baseline."""
        import tempfile
        import shutil
        from scripts.export_segmentation_tiff import export_segmentation_masks

        baseline_dir = BASELINE_DIR
        baseline_seg_dir = os.path.join(baseline_dir, "segmentations")

        if not os.path.exists(os.path.join(baseline_dir, "matched_seg_res_batch.pickle.zst")):
            self.skipTest("Baseline pickle files not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy pickle files to temp directory
            for pkl_file in ["matched_seg_res_batch.pickle.zst", "original_seg_res_batch.pickle.zst"]:
                src = os.path.join(baseline_dir, pkl_file)
                dst = os.path.join(temp_dir, pkl_file)
                if os.path.exists(src):
                    shutil.copy(src, dst)

            # Run patch script
            success = export_segmentation_masks(
                output_dir=temp_dir,
                base_name="D11_0",
                image_mpp=0.5,
            )
            self.assertTrue(success, "Patch script failed")

            # Compare outputs with baseline
            temp_seg_dir = os.path.join(temp_dir, "segmentations")
            self.assertTrue(os.path.isdir(temp_seg_dir), "Patch script didn't create segmentations/")

            # Compare each TIFF file
            for baseline_file in os.listdir(baseline_seg_dir):
                if not baseline_file.endswith(".ome.tiff"):
                    continue

                with self.subTest(file=baseline_file):
                    baseline_path = os.path.join(baseline_seg_dir, baseline_file)
                    temp_path = os.path.join(temp_seg_dir, baseline_file)

                    self.assertTrue(os.path.exists(temp_path),
                        f"Patch script missing file: {baseline_file}")

                    with tifffile.TiffFile(baseline_path) as tif:
                        baseline_data = tif.asarray()
                    with tifffile.TiffFile(temp_path) as tif:
                        temp_data = tif.asarray()

                    np.testing.assert_array_equal(
                        baseline_data, temp_data,
                        err_msg=f"Patch script output differs from baseline: {baseline_file}"
                    )


if __name__ == "__main__":
    unittest.main()
