import logging
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from aegle.codex_patches import CodexPatches


class TestCodexPatchesDiskMode(unittest.TestCase):
    def test_ensure_all_channel_patches_loaded_raises_for_disk_mode(self):
        dummy = CodexPatches.__new__(CodexPatches)
        dummy.patch_files = [{"all": "path"}]
        dummy.logger = None
        with self.assertRaises(ValueError):
            dummy._ensure_all_channel_patches_loaded()

    def test_get_all_channel_patch_loads_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_path = os.path.join(tmpdir, "patch_all.npy")
            arr = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
            np.save(patch_path, arr)

            dummy = CodexPatches.__new__(CodexPatches)
            dummy.patch_files = [{"all": patch_path, "extracted": patch_path}]
            dummy.logger = logging.getLogger("codex_patches_disk_test")
            dummy.cache_all_channel_patches = False

            loaded = dummy.get_all_channel_patch(0)
            np.testing.assert_array_equal(loaded, arr)


if __name__ == "__main__":
    unittest.main()
