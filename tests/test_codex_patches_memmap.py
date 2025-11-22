import logging
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from aegle.codex_patches import CodexPatches
from tests.utils.fixtures import write_zstd_npy


class TestCodexPatchesMemmap(unittest.TestCase):
    def test_prepare_all_channel_memmap_decompresses_once(self):
        """Memmap prep should decompress zstd once and reuse the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir, exist_ok=True)

            arr = np.arange(16, dtype=np.uint16).reshape(1, 4, 4, 1)
            zstd_path = os.path.join(tmpdir, "all_channel_patches.npy.zst")
            write_zstd_npy(arr, zstd_path)

            dummy = CodexPatches.__new__(CodexPatches)
            dummy.all_channel_patches_path = zstd_path
            dummy.all_channel_memmap_path = None
            dummy.args = SimpleNamespace(out_dir=out_dir)
            dummy.logger = logging.getLogger("codex_patches_test")

            memmap_path = dummy._prepare_all_channel_memmap()
            self.assertTrue(os.path.exists(memmap_path))
            first = np.load(memmap_path, mmap_mode="r", allow_pickle=False)
            np.testing.assert_array_equal(first, arr)

            # Second call should reuse the same path without re-decompressing.
            second_path = dummy._prepare_all_channel_memmap()
            self.assertEqual(memmap_path, second_path)
            second = np.load(second_path, mmap_mode="r", allow_pickle=False)
            np.testing.assert_array_equal(second, arr)

            # Clean up memmap file to simulate teardown.
            os.remove(memmap_path)
            dummy.all_channel_memmap_path = None
            memmap_path2 = dummy._prepare_all_channel_memmap()
            self.assertTrue(os.path.exists(memmap_path2))


if __name__ == "__main__":
    unittest.main()
