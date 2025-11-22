import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import run_analysis


@unittest.skipUnless(os.environ.get("RUN_ANALYSIS_INT"), "Set RUN_ANALYSIS_INT=1 to run analysis integration stub")
class TestAnalysisIntegrationStub(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exp_dir = self.root / "exp1"
        self.cp_dir = self.exp_dir / "cell_profiling"
        self.out_dir = self.root / "analysis_out"
        self.cp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "segmentations").mkdir(parents=True, exist_ok=True)

        pd.DataFrame([{"cell_mask_id": 1, "x": 0, "y": 0}]).to_csv(
            self.cp_dir / "cell_metadata.csv", index=False
        )
        pd.DataFrame(
            [
                {"cell_mask_id": 1, "markerA": 1.0, "markerB": 2.0},
            ]
        ).to_csv(self.cp_dir / "cell_by_marker.csv", index=False)
        (self.exp_dir / "segmentations" / "cell_mask.ome.tiff").touch()

        self.cfg_path = self.root / "analysis_config.yaml"
        self.cfg_path.write_text(
            """
exp_id: exp1
analysis:
  data_dir: exp1
  cell_profiling_dir: cell_profiling
  metadata_file: cell_metadata.csv
  expression_file: cell_by_marker.csv
  segmentation_file: segmentations/cell_mask.ome.tiff
  segmentation_format: ome_tiff
  skip_viz: true
  output_subdir: integration_out
  generate_pipeline_report: false
""",
            encoding="utf-8",
        )

    @patch("src.run_analysis.run_analysis")
    def test_analysis_stub_writes_expected_outputs(self, mock_run):
        """Run analysis main with stubbed handler and assert outputs written."""

        def _stub(config, args):
            out_dir = Path(args.output_dir)
            (out_dir / "plots").mkdir(parents=True, exist_ok=True)
            (out_dir / "plots" / "umap.png").write_text("ok", encoding="utf-8")
            (out_dir / "clusters.csv").write_text("ok", encoding="utf-8")
            (out_dir / "normalized.csv").write_text("ok", encoding="utf-8")

        mock_run.side_effect = _stub

        argv = [
            "prog",
            "--config_file",
            str(self.cfg_path),
            "--data_dir",
            str(self.root),
            "--output_dir",
            str(self.out_dir),
        ]
        with patch.object(os.sys, "argv", argv):
            run_analysis.main()

        out_dir = self.out_dir / "integration_out"
        self.assertTrue((out_dir / "plots" / "umap.png").exists())
        self.assertTrue((out_dir / "clusters.csv").exists())
        self.assertTrue((out_dir / "normalized.csv").exists())


if __name__ == "__main__":
    unittest.main()
