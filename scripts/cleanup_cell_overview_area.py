#!/usr/bin/env python3
"""Backfill cell_overview.csv area column using cell_area from metadata outputs."""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_ROOT = Path("out/main/main_ft_hb")
DEFAULT_GLOB = "D*_*/cell_profiling/cell_overview.csv"
METADATA_FILENAME = "cell_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the area column in cell_overview.csv using cell_area from the matching "
            "cell_metadata.csv file."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Root directory searched when --files is not provided (default: "
            f"{DEFAULT_ROOT})"
        ),
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob pattern (relative to --root) for locating overview files.",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        help="Explicit list of overview files to process instead of using --root/--glob.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be updated without modifying them.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding (default: utf-8)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def discover_files(root: Path, pattern: str) -> List[Path]:
    root = root.resolve()
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    logging.debug(
        "Discovered %d file(s) under %s using pattern %s", len(files), root, pattern
    )
    return files


def build_area_series(overview_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Optional[pd.Series]:
    if "cell_area" not in metadata_df.columns:
        return None

    for key in ("cell_mask_id", "global_cell_id"):
        if key in overview_df.columns and key in metadata_df.columns:
            mapping = metadata_df[[key, "cell_area"]].dropna(subset=["cell_area"])
            mapping = mapping.drop_duplicates(subset=[key]).set_index(key)["cell_area"]
            return overview_df[key].map(mapping)

    if len(overview_df) == len(metadata_df):
        return metadata_df["cell_area"].reset_index(drop=True)

    return None


def process_file(path: Path, encoding: str, dry_run: bool) -> bool:
    overview_df = pd.read_csv(path, encoding=encoding)
    if "area" not in overview_df.columns:
        logging.info("Skipping %s (no area column)", path)
        return False

    metadata_path = path.with_name(METADATA_FILENAME)
    if not metadata_path.exists():
        logging.warning("Skipping %s (missing %s)", path, metadata_path.name)
        return False

    metadata_df = pd.read_csv(metadata_path, encoding=encoding)
    new_area = build_area_series(overview_df, metadata_df)

    if new_area is None:
        logging.warning("Could not derive cell_area for %s", path)
        return False

    new_area = new_area.astype(float)
    current_area = overview_df["area"].astype(float)

    missing_mask = new_area.isna()
    if missing_mask.any():
        logging.debug(
            "Retaining existing area for %d rows without matching cell_area in %s",
            missing_mask.sum(),
            path,
        )
        new_area = new_area.where(~missing_mask, current_area)

    if new_area.equals(current_area):
        logging.info("No updates required for %s", path)
        return False

    logging.info("Updating area values in %s", path)
    if not dry_run:
        overview_df["area"] = new_area
        overview_df.to_csv(path, index=False, encoding=encoding)

    return True


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    files = [p.resolve() for p in args.files] if args.files else discover_files(args.root, args.glob)
    if not files:
        logging.warning("No files found to process.")
        return 1

    processed = 0
    updated = 0

    for path in files:
        if not path.exists():
            logging.warning("Skipping missing file %s", path)
            continue

        processed += 1
        if process_file(path, args.encoding, args.dry_run):
            updated += 1

    logging.info(
        "Processed %d file(s); %d updated%s.",
        processed,
        updated,
        " (dry run)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
