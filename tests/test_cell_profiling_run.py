import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aegle.cell_profiling import run_cell_profiling
from aegle.gpu_utils import is_cupy_available
from tests.utils.gpu_test_utils import requires_gpu


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


class TestRunCellProfiling(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.out_dir = self.tmpdir.name

    def _make_seg_result(self, cell_mask):
        return {
            "cell_matched_mask": cell_mask.astype(np.uint32),
            "nucleus_matched_mask": cell_mask.astype(np.uint32),
        }

    def test_non_informative_patches_skip_outputs(self):
        """When no informative patches, profiling should not emit CSVs."""
        antibody_df = pd.DataFrame({"antibody_name": ["chan0"]})
        patches_metadata = pd.DataFrame(
            [
                {"patch_id": 0, "patch_index": 0, "x_start": 0, "y_start": 0, "is_informative": False},
            ]
        )
        seg_results = [self._make_seg_result(np.zeros((2, 2), dtype=np.uint32))]
        channel_patches = [np.zeros((2, 2, 1), dtype=np.float32)]
        codex = FakeCodexPatches(antibody_df, patches_metadata, seg_results, channel_patches)

        args = SimpleNamespace(out_dir=self.out_dir, data_dir=".")
        config = {"patching": {"split_mode": "full_image"}}
        run_cell_profiling(codex, config, args)

        profiling_dir = os.path.join(self.out_dir, "cell_profiling")
        self.assertFalse(os.path.exists(os.path.join(profiling_dir, "cell_by_marker.csv")))
        self.assertFalse(os.path.exists(os.path.join(profiling_dir, "cell_metadata.csv")))

    def test_zero_label_masks_produce_no_rows(self):
        """Zero-labeled masks should yield empty profiling outputs."""
        antibody_df = pd.DataFrame({"antibody_name": ["chan0"]})
        patches_metadata = pd.DataFrame(
            [{"patch_id": 0, "patch_index": 0, "x_start": 0, "y_start": 0, "is_informative": True}]
        )
        zero_mask = np.zeros((2, 2), dtype=np.uint32)
        seg_results = [self._make_seg_result(zero_mask)]
        channel_patches = [np.zeros((2, 2, 1), dtype=np.float32)]
        codex = FakeCodexPatches(antibody_df, patches_metadata, seg_results, channel_patches)

        args = SimpleNamespace(out_dir=self.out_dir, data_dir=".")
        config = {"patching": {"split_mode": "full_image"}}
        run_cell_profiling(codex, config, args)

        profiling_dir = os.path.join(self.out_dir, "cell_profiling")
        self.assertFalse(os.path.exists(os.path.join(profiling_dir, "cell_metadata.csv")))
        self.assertFalse(os.path.exists(os.path.join(profiling_dir, "cell_by_marker.csv")))

    def test_global_cell_ids_increment_across_patches(self):
        """cell_mask_id should offset by previous max label and coords should shift."""
        antibody_df = pd.DataFrame({"antibody_name": ["chan0", "chan1"]})
        patches_metadata = pd.DataFrame(
            [
                {"patch_id": 0, "patch_index": 0, "x_start": 0, "y_start": 0, "is_informative": True},
                {"patch_id": 1, "patch_index": 1, "x_start": 5, "y_start": 7, "is_informative": True},
            ]
        )

        seg_results = [
            self._make_seg_result(np.array([[1, 0], [0, 2]], dtype=np.uint32)),
            self._make_seg_result(np.array([[1, 0], [0, 0]], dtype=np.uint32)),
        ]
        channel_patches = [
            np.ones((2, 2, 2), dtype=np.float32),
            np.ones((2, 2, 2), dtype=np.float32) * 2,
        ]
        codex = FakeCodexPatches(antibody_df, patches_metadata, seg_results, channel_patches)

        args = SimpleNamespace(out_dir=self.out_dir, data_dir=".")
        config = {"patching": {"split_mode": "halves"}}

        run_cell_profiling(codex, config, args)

        profiling_dir = os.path.join(self.out_dir, "cell_profiling")
        meta_path = os.path.join(profiling_dir, "cell_metadata.csv")
        exp_path = os.path.join(profiling_dir, "cell_by_marker.csv")

        self.assertTrue(os.path.exists(meta_path))
        self.assertTrue(os.path.exists(exp_path))

        meta = pd.read_csv(meta_path)
        self.assertSetEqual(set(meta["cell_mask_id"]), {1, 2, 3})

        # Coordinates for the second patch should reflect offsets.
        patch1_rows = meta[meta["patch_id"] == 1]
        self.assertTrue((patch1_rows["x"] >= 5).all())
        self.assertTrue((patch1_rows["y"] >= 7).all())


class TestRunCellProfilingWithGPU(unittest.TestCase):
    """Test cell profiling pipeline with GPU enabled."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.out_dir = self.tmpdir.name

    def _make_seg_result(self, cell_mask):
        return {
            "cell_matched_mask": cell_mask.astype(np.uint32),
            "nucleus_matched_mask": cell_mask.astype(np.uint32),
        }

    @requires_gpu()
    def test_cell_profiling_with_gpu_enabled(self):
        """Cell profiling should work with GPU enabled."""
        antibody_df = pd.DataFrame({"antibody_name": ["CD3", "CD8", "CD45"]})
        patches_metadata = pd.DataFrame(
            [{"patch_id": 0, "patch_index": 0, "x_start": 0, "y_start": 0, "is_informative": True}]
        )

        # Create mask with 3 cells
        mask = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 0, 0],
            [3, 3, 0, 0, 0],
        ], dtype=np.uint32)

        seg_results = [self._make_seg_result(mask)]

        # Create channel data with different intensities per cell
        channel_patch = np.zeros((5, 5, 3), dtype=np.float32)
        # Cell 1: CD3=100, CD8=50, CD45=200
        channel_patch[mask == 1, 0] = 100
        channel_patch[mask == 1, 1] = 50
        channel_patch[mask == 1, 2] = 200
        # Cell 2: CD3=80, CD8=120, CD45=150
        channel_patch[mask == 2, 0] = 80
        channel_patch[mask == 2, 1] = 120
        channel_patch[mask == 2, 2] = 150
        # Cell 3: CD3=90, CD8=70, CD45=180
        channel_patch[mask == 3, 0] = 90
        channel_patch[mask == 3, 1] = 70
        channel_patch[mask == 3, 2] = 180

        channel_patches = [channel_patch]
        codex = FakeCodexPatches(antibody_df, patches_metadata, seg_results, channel_patches)

        # Config with GPU enabled
        config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "use_gpu": True,
                    "gpu_batch_size": 0,  # Auto-detect
                    "compute_laplacian": False,
                    "compute_cov": False,
                }
            }
        }

        args = SimpleNamespace(out_dir=self.out_dir, data_dir=".")
        run_cell_profiling(codex, config, args)

        # Check outputs exist
        profiling_dir = os.path.join(self.out_dir, "cell_profiling")
        markers_path = os.path.join(profiling_dir, "cell_by_marker.csv")
        overview_path = os.path.join(profiling_dir, "cell_overview.csv")

        self.assertTrue(os.path.exists(markers_path))
        self.assertTrue(os.path.exists(overview_path))

        # Load and validate results
        markers = pd.read_csv(markers_path)
        self.assertEqual(len(markers), 3)  # 3 cells

        # Check channel columns exist
        for channel in ["CD3", "CD8", "CD45"]:
            self.assertIn(channel, markers.columns)

    @requires_gpu()
    def test_gpu_vs_cpu_profiling_equivalence(self):
        """GPU and CPU profiling should produce equivalent results."""
        antibody_df = pd.DataFrame({"antibody_name": ["chan0", "chan1"]})
        patches_metadata = pd.DataFrame(
            [{"patch_id": 0, "patch_index": 0, "x_start": 0, "y_start": 0, "is_informative": True}]
        )

        mask = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.uint32)
        seg_results = [self._make_seg_result(mask)]

        channel_patch = np.ones((3, 3, 2), dtype=np.float32)
        channel_patch[:, :, 0] *= 100  # chan0
        channel_patch[:, :, 1] *= 50   # chan1
        channel_patches = [channel_patch]

        codex = FakeCodexPatches(antibody_df, patches_metadata, seg_results, channel_patches)

        base_config = {
            "patching": {"split_mode": "full_image"},
            "profiling": {
                "features": {
                    "compute_laplacian": False,
                    "compute_cov": False,
                }
            }
        }

        # Run with CPU
        cpu_config = base_config.copy()
        cpu_config["profiling"]["features"]["use_gpu"] = False

        cpu_tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(cpu_tmpdir.cleanup)
        args_cpu = SimpleNamespace(out_dir=cpu_tmpdir.name, data_dir=".")
        run_cell_profiling(codex, cpu_config, args_cpu)

        cpu_markers = pd.read_csv(os.path.join(cpu_tmpdir.name, "cell_profiling", "cell_by_marker.csv"))

        # Run with GPU
        gpu_config = base_config.copy()
        gpu_config["profiling"]["features"]["use_gpu"] = True

        gpu_tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(gpu_tmpdir.cleanup)
        args_gpu = SimpleNamespace(out_dir=gpu_tmpdir.name, data_dir=".")
        run_cell_profiling(codex, gpu_config, args_gpu)

        gpu_markers = pd.read_csv(os.path.join(gpu_tmpdir.name, "cell_profiling", "cell_by_marker.csv"))

        # Compare numerical columns
        for channel in ["chan0", "chan1"]:
            np.testing.assert_allclose(
                gpu_markers[channel].values,
                cpu_markers[channel].values,
                rtol=1e-5, atol=1e-6,
                err_msg=f"Channel {channel} differs between GPU and CPU"
            )


if __name__ == "__main__":
    unittest.main()
