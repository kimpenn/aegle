import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from aegle.codex_patches import CodexPatches
from tests.utils.fixtures import write_antibodies_tsv, write_tiny_ome_tiff


class TestCodexPatchesLoadFromOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def _write_minimum_artifacts(self):
        pd.DataFrame(
            [
                {"patch_id": 0, "patch_index": 0, "x_start": 0, "y_start": 0, "patch_width": 4, "patch_height": 4}
            ]
        ).to_csv(os.path.join(self.root, "patches_metadata.csv"), index=False)

    def test_missing_metadata_raises(self):
        config = {"data": {"antibodies_file": "antibodies.tsv"}}
        args = SimpleNamespace(out_dir=self.root, data_dir=self.root)
        with self.assertRaises(FileNotFoundError):
            CodexPatches.load_from_outputs(config, args)

    def test_missing_patches_array_raises(self):
        self._write_minimum_artifacts()
        config = {"data": {"antibodies_file": "antibodies.tsv"}}
        # write antibodies so load works
        write_antibodies_tsv(os.path.join(self.root, "antibodies.tsv"), ["ch0"])
        args = SimpleNamespace(out_dir=self.root, data_dir=self.root)
        # missing all_channel_patches -> FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            CodexPatches.load_from_outputs(config, args)


if __name__ == "__main__":
    unittest.main()
