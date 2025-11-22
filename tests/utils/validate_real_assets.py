#!/usr/bin/env python3
"""Lightweight validator to compare real assets with test fixture assumptions.

Usage:
  python tests/utils/validate_real_assets.py --config path/to/config.yaml --data-dir DATA --out-dir OUT

This does not modify any files; it reports mismatches and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import tifffile
import yaml


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")


def _expect_columns(path: Path, required: Iterable[str]) -> None:
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise AssertionError(f"{path} missing columns: {missing}")


def _check_tiff(path: Path) -> None:
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    if arr.ndim != 3:
        raise AssertionError(f"{path} expected 3D (C,H,W), got shape {arr.shape}")
    if arr.dtype != "uint16":
        raise AssertionError(f"{path} expected uint16, got {arr.dtype}")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate(config_path: Path | None, data_dir: Path, out_dir: Path) -> None:
    if config_path:
        cfg = _load_yaml(config_path)
        data_cfg = cfg.get("data", {})
        channels_cfg = cfg.get("channels", {})
        required_data_keys = ["file_name", "antibodies_file"]
        for key in required_data_keys:
            if key not in data_cfg:
                raise AssertionError(f"Config missing data.{key}")
        if "nucleus_channel" not in channels_cfg or "wholecell_channel" not in channels_cfg:
            raise AssertionError("Config missing channels.nucleus_channel or channels.wholecell_channel")
        img_path = data_dir / data_cfg["file_name"]
        ab_path = data_dir / data_cfg["antibodies_file"]
        _check_exists(img_path, "image")
        _check_exists(ab_path, "antibodies.tsv")
        _check_tiff(img_path)
    else:
        cfg = {}

    profiling_dir = out_dir / "cell_profiling"
    if profiling_dir.exists():
        meta_path = profiling_dir / "cell_metadata.csv"
        exp_path = profiling_dir / "cell_by_marker.csv"
        _check_exists(meta_path, "cell_metadata.csv")
        _check_exists(exp_path, "cell_by_marker.csv")
        _expect_columns(meta_path, ["cell_mask_id", "x", "y"])
        # Require at least one marker column besides cell_mask_id.
        exp_df = pd.read_csv(exp_path, nrows=1)
        if "cell_mask_id" not in exp_df.columns or len(exp_df.columns) <= 1:
            raise AssertionError(f"{exp_path} should contain marker columns")

    print("Validation completed successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real assets against fixture expectations.")
    parser.add_argument("--config", type=Path, help="Path to main pipeline config YAML", default=None)
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory containing images/antibodies.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Pipeline output directory containing cell_profiling/.")
    args = parser.parse_args(argv)
    validate(args.config, args.data_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
