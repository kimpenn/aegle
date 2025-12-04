#!/usr/bin/env python
"""Generate downsampled overview JPEGs from manual OME-TIFF annotations.

Uses VIPS for efficient tile-based processing of large images, avoiding
Bio-Formats' 2GB memory limit for single-plane extraction.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--root_dir", help="Repository root directory (deprecated, no longer required)")
    parser.add_argument("--downscale", type=float, default=0.5, help="Resize factor applied when generating overview JPEG")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality percentage (0-100)")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to export from the OME-TIFF")
    parser.add_argument("--overwrite", action="store_true", help="Recreate JPEGs even if they already exist")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_paths(config: dict) -> Tuple[Path, Path]:
    data_section = config.get("data", {})
    file_name = data_section.get("file_name")
    if not file_name:
        raise ValueError("Missing 'data.file_name' in config; cannot locate manual annotations")
    data_path = Path(file_name)
    if not data_path.exists():
        raise FileNotFoundError(f"Image file referenced in config is missing: {data_path}")

    tissue_cfg = config.get("tissue_extraction", {})
    output_dir_cfg = tissue_cfg.get("output_dir")
    if output_dir_cfg:
        output_dir = Path(output_dir_cfg)
    else:
        raise ValueError("Missing 'tissue_extraction.output_dir' in config; cannot locate manual annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    return data_path, output_dir


def discover_manual_ome(output_dir: Path, base_name: str) -> list[Path]:
    candidates = sorted(output_dir.glob(f"{base_name}_manual_*.ome.tiff"))
    return [p for p in candidates if p.is_file()]


def run_command(cmd: list[str], env: Optional[Dict[str, str]] = None) -> None:
    logging.debug("Executing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def convert_single_vips(
    ome_path: Path, channel: int, downscale: float, quality: int, overwrite: bool
) -> None:
    """Convert OME-TIFF channel to downsampled JPEG using VIPS.

    VIPS processes images tile-by-tile in streaming mode, avoiding the 2GB
    memory limit that affects Bio-Formats bfconvert for large images.

    In OME-TIFF files, each channel is stored as a separate page (IFD).
    """
    stem = ome_path.with_suffix("")
    jpg_path = Path(f"{stem}_ch{channel}.jpg")

    if jpg_path.exists() and not overwrite:
        logging.info("Skipping existing JPEG: %s", jpg_path)
        return

    logging.info(
        "Converting channel %d from %s -> %s (scale=%.2f, Q=%d)",
        channel,
        ome_path.name,
        jpg_path.name,
        downscale,
        quality,
    )

    # VIPS pipeline: load specific page (channel) -> resize -> save as JPEG
    # Using sequential access for efficient streaming of large images
    run_command([
        "vips",
        "resize",
        f"{ome_path}[page={channel},access=sequential]",
        f"{jpg_path}[Q={quality},strip]",
        str(downscale),
    ])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    config_path = Path(args.config)

    config = load_config(config_path)
    data_path, output_dir = build_paths(config)
    base_name = data_path.stem
    ome_files = discover_manual_ome(output_dir, base_name)

    if not ome_files:
        logging.warning("No manual OME-TIFF files found in %s", output_dir)
        return

    logging.info("Found %d manual OME-TIFF files", len(ome_files))

    for ome_path in ome_files:
        try:
            convert_single_vips(
                ome_path=ome_path,
                channel=args.channel,
                downscale=args.downscale,
                quality=args.quality,
                overwrite=args.overwrite,
            )
        except subprocess.CalledProcessError as exc:
            logging.error("Command failed for %s: %s", ome_path.name, exc)
            raise

    logging.info("Overview JPEG generation completed.")


if __name__ == "__main__":
    main()
