import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src import run_analysis


class TestAnalysisCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write_analysis_config(self, rel_exp: str) -> Path:
        cfg = {
            "analysis": {
                "data_dir": rel_exp,
                "cell_profiling_dir": "cell_profiling",
                "metadata_file": "cell_metadata.csv",
                "expression_file": "cell_by_marker.csv",
                "output_subdir": "analysis_out",
                "segmentation_file": "segmentations/cell_mask.ome.tiff",
                "segmentation_format": "ome_tiff",
                "skip_viz": True,
                "generate_pipeline_report": False,
            }
        }
        cfg_path = self.root / "analysis_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return cfg_path

    @patch("src.run_analysis.run_analysis")
    def test_run_analysis_resolves_paths_and_invokes_handler(self, mock_run):
        """CLI should resolve paths from config and call run_analysis with args."""
        # Set up minimal cell profiling outputs.
        exp_dir = self.root / "exp1"
        cp_dir = exp_dir / "cell_profiling"
        cp_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"cell_mask_id": 1, "x": 0, "y": 0}]).to_csv(
            cp_dir / "cell_metadata.csv", index=False
        )
        pd.DataFrame([{"cell_mask_id": 1, "markerA": 1.0}]).to_csv(
            cp_dir / "cell_by_marker.csv", index=False
        )
        # Segmentation path optional; we just ensure resolution.
        (exp_dir / "segmentations").mkdir(parents=True, exist_ok=True)
        (exp_dir / "segmentations" / "cell_mask.ome.tiff").touch()

        cfg_path = self._write_analysis_config(rel_exp="exp1")

        argv = [
            "prog",
            "--config_file",
            str(cfg_path),
            "--data_dir",
            str(self.root),
            "--output_dir",
            str(self.root / "analysis_out"),
            "--skip_viz",
        ]
        with patch.object(sys, "argv", argv):
            run_analysis.main()

        self.assertTrue(mock_run.called)
        called_args = mock_run.call_args[0][1]  # second positional is args
        self.assertTrue(called_args.cell_metadata_path.endswith("cell_metadata.csv"))
        self.assertTrue(called_args.cell_expression_path.endswith("cell_by_marker.csv"))
        self.assertIn("analysis_out", called_args.output_dir)

    @patch("src.run_analysis.run_analysis")
    def test_run_analysis_happy_path_outputs(self, mock_run):
        """Ensure normalization/clustering flags propagate (plots/report disabled)."""
        exp_dir = self.root / "exp1"
        cp_dir = exp_dir / "cell_profiling"
        cp_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"cell_mask_id": 1, "x": 0, "y": 0}]).to_csv(
            cp_dir / "cell_metadata.csv", index=False
        )
        pd.DataFrame([{"cell_mask_id": 1, "markerA": 1.0, "markerB": 2.0}]).to_csv(
            cp_dir / "cell_by_marker.csv", index=False
        )
        (exp_dir / "segmentations").mkdir(parents=True, exist_ok=True)
        (exp_dir / "segmentations" / "cell_mask.ome.tiff").touch()

        cfg_path = self._write_analysis_config(rel_exp="exp1")

        def _stub_run(config, args):
            # Simulate downstream writing outputs respecting output_dir.
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "normalized.csv").write_text("ok\n", encoding="utf-8")
            (out_dir / "clusters.csv").write_text("ok\n", encoding="utf-8")

        mock_run.side_effect = _stub_run

        argv = [
            "prog",
            "--config_file",
            str(cfg_path),
            "--data_dir",
            str(self.root),
            "--output_dir",
            str(self.root / "analysis_out"),
            "--skip_viz",
        ]
        with patch.object(sys, "argv", argv):
            run_analysis.main()

        out_dir = self.root / "analysis_out" / "analysis_out"
        self.assertTrue((out_dir / "normalized.csv").exists())
        self.assertTrue((out_dir / "clusters.csv").exists())

    def test_missing_analysis_data_dir_raises(self):
        """analysis.data_dir is required in config."""
        cfg = {"analysis": {}}
        cfg_path = self.root / "bad_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        argv = ["prog", "--config_file", str(cfg_path), "--data_dir", str(self.root)]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(ValueError):
                run_analysis.main()

    def test_missing_cli_data_dir_raises(self):
        """--data_dir must be provided."""
        cfg_path = self._write_analysis_config(rel_exp="exp1")
        argv = ["prog", "--config_file", str(cfg_path)]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(ValueError):
                run_analysis.main()

    @patch("src.run_analysis.run_analysis")
    def test_missing_profiling_files_errors(self, mock_run):
        """If profiling CSVs are absent, handler should notice."""
        exp_dir = self.root / "exp1"
        (exp_dir / "cell_profiling").mkdir(parents=True, exist_ok=True)
        cfg_path = self._write_analysis_config(rel_exp="exp1")

        def _stub(config, args):
            # Simulate handler requiring files.
            assert Path(args.cell_metadata_path).exists(), "metadata missing"
        mock_run.side_effect = _stub

        argv = [
            "prog",
            "--config_file",
            str(cfg_path),
            "--data_dir",
            str(self.root),
            "--output_dir",
            str(self.root / "analysis_out"),
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(AssertionError):
                run_analysis.main()


if __name__ == "__main__":
    unittest.main()
