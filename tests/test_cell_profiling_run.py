import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aegle.cell_profiling import run_cell_profiling


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


if __name__ == "__main__":
    unittest.main()
