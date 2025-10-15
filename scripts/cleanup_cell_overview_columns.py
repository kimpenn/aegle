#!/usr/bin/env python3
"""Utility script to remove patch-specific identifiers from cell overview CSV outputs."""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

TARGET_COLUMNS = ("patch_id", "global_cell_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop patch_id and global_cell_id columns from existing cell_overview.csv files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out/main/main_ft_hb"),
        help="Root directory searched when --files is not provided (default: out/main/main_ft_hb)",
    )
    parser.add_argument(
        "--glob",
        default="D*_*/cell_profiling/cell_overview.csv",
        help="Glob pattern (relative to --root) for locating overview files.",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        help="Explicit list of files to process instead of using --root/--glob.",
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
    logging.debug("Discovered %d file(s) under %s using pattern %s", len(files), root, pattern)
    return files


def process_file(path: Path, encoding: str, dry_run: bool) -> Tuple[bool, List[str]]:
    df = pd.read_csv(path, encoding=encoding)
    missing_columns = [c for c in TARGET_COLUMNS if c not in df.columns]

    if len(missing_columns) == len(TARGET_COLUMNS):
        logging.info("Skipping %s (no target columns present)", path)
        return False, missing_columns

    columns_to_drop = [c for c in TARGET_COLUMNS if c in df.columns]
    logging.info("Dropping %s in %s", ", ".join(columns_to_drop), path)

    if not dry_run:
        updated = df.drop(columns=columns_to_drop)
        updated.to_csv(path, index=False, encoding=encoding)

    return True, missing_columns


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.files:
        files = [p.resolve() for p in args.files]
    else:
        files = discover_files(args.root, args.glob)

    if not files:
        logging.warning("No files found to process.")
        return 1

    updated_count = 0
    processed_count = 0

    for path in files:
        if not path.exists():
            logging.warning("Skipping missing file %s", path)
            continue

        processed_count += 1
        changed, missing = process_file(path, args.encoding, args.dry_run)
        if changed:
            updated_count += 1
        else:
            logging.debug("No changes applied to %s; missing columns: %s", path, missing)

    logging.info(
        "Processed %d file(s); %d had columns removed%s.",
        processed_count,
        updated_count,
        " (dry run)" if args.dry_run else "",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
