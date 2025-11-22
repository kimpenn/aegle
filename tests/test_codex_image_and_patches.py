import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from aegle.codex_image import CodexImage
from aegle.codex_patches import CodexPatches
from tests.utils.fixtures import write_antibodies_tsv, write_tiny_ome_tiff


def _make_config(split_mode: str, *, patch_height=None, patch_width=None, overlap=None):
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
            "patch_height": patch_height,
            "patch_width": patch_width,
            "overlap": overlap,
        },
        "visualization": {},
        "patch_qc": {},
        "segmentation": {"segmentation_pickle_compression_threads": 0},
        "evaluation": {"compute_metrics": False},
        "report": {"generate_report": False, "report_format": "html"},
    }


class TestCodexImageAndPatches(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data_dir = self.tmpdir.name
        self.out_dir = os.path.join(self.tmpdir.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def _write_inputs(self, shape=(16, 16, 4)):
        write_tiny_ome_tiff(os.path.join(self.data_dir, "img.ome.tiff"), shape=shape)
        write_antibodies_tsv(
            os.path.join(self.data_dir, "antibodies.tsv"),
            [f"ch{i}" for i in range(shape[2])],
        )

    def test_full_image_split_uses_whole_dimensions(self):
        """full_image mode should set patch dims to image size and keep shapes."""
        self._write_inputs(shape=(10, 12, 4))
        config = _make_config("full_image")
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()

        self.assertEqual(codex_image.patch_height, 10)
        self.assertEqual(codex_image.patch_width, 12)
        self.assertEqual(codex_image.split_mode, "full_image")
        self.assertEqual(
            codex_image.extended_all_channel_image.shape, (10, 12, 4)
        )
        # Extracted channels should contain nucleus + one wholecell channel.
        self.assertEqual(
            codex_image.extended_extracted_channel_image.shape, (10, 12, 2)
        )

    def test_patches_split_generates_expected_patch_count(self):
        """patches mode should crop into the correct number of patches."""
        self._write_inputs(shape=(16, 16, 3))
        config = _make_config("patches", patch_height=8, patch_width=8, overlap=0.25)
        args = SimpleNamespace(data_dir=self.data_dir, out_dir=self.out_dir)

        codex_image = CodexImage(config, args)
        codex_image.extract_target_channels()
        codex_image.extend_image()
        codex_patches = CodexPatches(codex_image, config, args)

        # Extended to 22x22 for full coverage -> positions at 0, 6, 12 => 9 patches.
        self.assertEqual(codex_patches.extracted_channel_patches.shape[0], 9)
        self.assertEqual(codex_patches.extracted_channel_patches.shape[1:3], (8, 8))
        self.assertEqual(len(codex_patches.get_patches_metadata()), 9)


if __name__ == "__main__":
    unittest.main()
