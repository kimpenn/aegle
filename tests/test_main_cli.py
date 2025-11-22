import sys
import unittest
from unittest.mock import patch

from src.main import parse_args


class TestMainCLI(unittest.TestCase):
    def test_parse_args_with_resume(self):
        argv = [
            "prog",
            "--config_file",
            "cfg.yaml",
            "--data_dir",
            "/tmp/data",
            "--out_dir",
            "/tmp/out",
            "--log_level",
            "DEBUG",
            "--resume_stage",
            "cell_profiling",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.config_file, "cfg.yaml")
        self.assertEqual(args.data_dir, "/tmp/data")
        self.assertEqual(args.out_dir, "/tmp/out")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.resume_stage, "cell_profiling")

    def test_parse_args_defaults(self):
        argv = ["prog", "--config_file", "cfg.yaml"]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.log_level, "INFO")
        self.assertIsNone(args.resume_stage)


if __name__ == "__main__":
    unittest.main()
