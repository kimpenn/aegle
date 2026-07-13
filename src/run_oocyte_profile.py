#!/usr/bin/env python3
"""Profile reviewed oocyte labels against every registered raw channel."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.oocyte.profiling import profile_oocyte_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one raw within-mask marker-intensity row per final oocyte label."
        )
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--antibodies", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float, required=True)
    parser.add_argument("--max-region-height-px", type=int, default=512)
    parser.add_argument("--merge-gap-px", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    result = profile_oocyte_labels(
        sample_id=args.sample_id,
        image_path=args.image,
        antibodies_path=args.antibodies,
        label_path=args.labels,
        mapping_path=args.mapping,
        candidates_path=args.candidates,
        out_dir=args.out_dir,
        pixel_size_um=args.pixel_size_um,
        max_region_height_px=args.max_region_height_px,
        merge_gap_px=args.merge_gap_px,
    )
    logging.info("Marker matrix: %s", result.artifact_paths["markers"])
    logging.info("Metadata: %s", result.artifact_paths["metadata"])
    logging.info("Manifest: %s", result.artifact_paths["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
