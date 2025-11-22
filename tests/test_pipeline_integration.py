import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
import zstandard as zstd

from aegle.pipeline import run_pipeline
from tests.utils.fixtures import (
    write_antibodies_tsv,
    write_tiny_ome_tiff,
    write_zstd_npy,
)


def _write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return path


def _make_base_config(split_mode: str = "full_image") -> dict:
    return {
        "exp_id": "test_exp",
        "sample_id": "sample",
        "data": {
            "file_name": "img.ome.tiff",
            "antibodies_file": "antibodies.tsv",
            "image_mpp": 0.5,
            "generate_channel_stats": False,
        },
        "channels": {"nucleus_channel": "ch0", "wholecell_channel": ["ch1"]},
        "patching": {
            "split_mode": split_mode,
            "patch_height": None,
            "patch_width": None,
            "overlap": None,
        },
        "visualization": {
            "visualize_patches": False,
            "cache_all_channel_patches": True,
            "save_all_channel_patches": True,
            "visualize_segmentation": False,
        },
        "patch_qc": {},
        "segmentation": {
            "save_segmentation_images": False,
            "save_segmentation_pickle": False,
            "segmentation_pickle_compression_threads": 0,
            "segmentation_analysis": False,
        },
        "evaluation": {"compute_metrics": False},
        "report": {"generate_report": False, "report_format": "html"},
    }


def _fake_segmentation(codex_patches, config, args):
    """Stub segmentation to attach tiny masks without invoking models."""
    patch_shape = codex_patches.extracted_channel_patches[0].shape[:2]
    mask = np.zeros(patch_shape, dtype=np.uint32)
    mask[:2, :2] = 1
    seg_res = {
        "cell_matched_mask": mask,
        "nucleus_matched_mask": mask,
    }
    codex_patches.repaired_seg_res_batch = [seg_res]
    codex_patches.original_seg_res_batch = [seg_res]
    codex_patches.valid_patches = [0]


@unittest.skipUnless(os.environ.get("RUN_INTEGRATION"), "Set RUN_INTEGRATION=1 to run integration tests")
class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_dir = self.root
        self.out_dir = self.root / "out"

    def _write_inputs(self, shape=(8, 8, 3)):
        write_tiny_ome_tiff(self.root / "img.ome.tiff", shape=shape)
        write_antibodies_tsv(self.root / "antibodies.tsv", [f"ch{i}" for i in range(shape[2])])

    @patch("aegle.pipeline.generate_pipeline_report", lambda *args, **kwargs: None)
    @patch("aegle.pipeline.run_segmentation_analysis", lambda *args, **kwargs: None)
    @patch("aegle.pipeline.run_cell_segmentation", side_effect=_fake_segmentation)
    def test_run_pipeline_with_mocked_segmentation(self, _mock_seg):
        """Pipeline should produce profiling outputs when segmentation is stubbed."""
        self._write_inputs(shape=(8, 8, 3))
        config = _make_base_config("full_image")
        config_path = _write_config(self.root / "config.yaml", config)
        args = SimpleNamespace(
            config_file=str(config_path),
            data_dir=str(self.data_dir),
            out_dir=str(self.out_dir),
            resume_stage=None,
            log_level="INFO",
        )

        run_pipeline(config, args)

        profiling_dir = self.out_dir / "cell_profiling"
        self.assertTrue((profiling_dir / "cell_by_marker.csv").exists())
        self.assertTrue((profiling_dir / "cell_metadata.csv").exists())
        meta = pd.read_csv(profiling_dir / "cell_metadata.csv")
        # Expect at least the one stubbed cell.
        self.assertGreaterEqual(len(meta), 1)

    def test_resume_pipeline_with_precomputed_outputs(self):
        """Resume mode should load existing segmentation artifacts and run profiling."""
        self._write_inputs(shape=(6, 6, 3))
        # Prepare out_dir with required artifacts.
        out_dir = self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Patches metadata for a single informative patch.
        patches_meta = pd.DataFrame(
            [
                {
                    "patch_id": 0,
                    "patch_index": 0,
                    "x_start": 0,
                    "y_start": 0,
                    "patch_width": 6,
                    "patch_height": 6,
                    "is_informative": True,
                }
            ]
        )
        patches_meta.to_csv(out_dir / "patches_metadata.csv", index=False)

        # Segmentation result pickle (compressed).
        mask = np.zeros((6, 6), dtype=np.uint32)
        mask[:2, :2] = 1
        seg_res = {"cell_matched_mask": mask, "nucleus_matched_mask": mask}
        comp = zstd.ZstdCompressor()
        with open(out_dir / "matched_seg_res_batch.pickle.zst", "wb") as fh:
            with comp.stream_writer(fh) as writer:
                pickle.dump([seg_res], writer, protocol=pickle.HIGHEST_PROTOCOL)

        # All-channel patches artifact.
        all_channels = np.ones((1, 6, 6, 3), dtype=np.uint16)
        write_zstd_npy(all_channels, out_dir / "all_channel_patches.npy.zst")

        config = _make_base_config("full_image")
        config_path = _write_config(self.root / "config.yaml", config)
        args = SimpleNamespace(
            config_file=str(config_path),
            data_dir=str(self.data_dir),
            out_dir=str(out_dir),
            resume_stage="cell_profiling",
            log_level="INFO",
        )

        run_pipeline(config, args)

        profiling_dir = out_dir / "cell_profiling"
        self.assertTrue((profiling_dir / "cell_by_marker.csv").exists())
        meta = pd.read_csv(profiling_dir / "cell_metadata.csv")
        self.assertGreaterEqual(len(meta), 1)

    @patch("aegle.pipeline.generate_pipeline_report", lambda *args, **kwargs: None)
    @patch("aegle.pipeline.run_segmentation_analysis", lambda *args, **kwargs: None)
    @patch("aegle.pipeline.run_cell_segmentation")
    def test_pipeline_patches_mode_mocked_segmentation(self, mock_seg):
        """Patches split mode should handle multiple patches and profile them."""
        self._write_inputs(shape=(8, 8, 3))
        config = _make_base_config("patches")
        config["patching"]["patch_height"] = 4
        config["patching"]["patch_width"] = 4
        config["patching"]["overlap"] = 0.0
        config_path = _write_config(self.root / "config_patches.yaml", config)
        args = SimpleNamespace(
            config_file=str(config_path),
            data_dir=str(self.data_dir),
            out_dir=str(self.out_dir),
            resume_stage=None,
            log_level="INFO",
        )

        def _seg_stub(codex_patches, *_args, **_kwargs):
            segs = []
            for patch in codex_patches.extracted_channel_patches:
                mask = np.zeros(patch.shape[:2], dtype=np.uint32)
                mask[:1, :1] = 1
                segs.append({"cell_matched_mask": mask, "nucleus_matched_mask": mask})
            codex_patches.repaired_seg_res_batch = segs
            codex_patches.original_seg_res_batch = segs
            codex_patches.valid_patches = list(range(len(segs)))

        mock_seg.side_effect = _seg_stub

        run_pipeline(config, args)

        profiling_dir = self.out_dir / "cell_profiling"
        metas = []
        for idx in range(4):
            meta_path = profiling_dir / f"patch-{idx}-cell_metadata.csv"
            self.assertTrue(meta_path.exists())
            metas.append(pd.read_csv(meta_path))
        total_cells = sum(len(df) for df in metas)
        self.assertEqual(total_cells, 4)


if __name__ == "__main__":
    unittest.main()
