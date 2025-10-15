#!/usr/bin/env python3
"""Remove the area column from existing cell_metadata.csv outputs."""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

TARGET_COLUMN = "area"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop the area column from cell_metadata.csv files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("out/main/main_ft_hb"),
        help="Root directory searched when --files is not provided (default: out/main/main_ft_hb)",
    )
    parser.add_argument(
        "--glob",
        default="D*_*/cell_profiling/cell_metadata.csv",
        help="Glob pattern (relative to --root) for locating metadata files.",
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


def process_file(path: Path, encoding: str, dry_run: bool) -> Tuple[bool, bool]:
    df = pd.read_csv(path, encoding=encoding)
    if TARGET_COLUMN not in df.columns:
        logging.info("Skipping %s (no %s column)", path, TARGET_COLUMN)
        return False, False

    logging.info("Dropping %s in %s", TARGET_COLUMN, path)
    if not dry_run:
        df.drop(columns=[TARGET_COLUMN]).to_csv(path, index=False, encoding=encoding)
    return True, True


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
        changed, _ = process_file(path, args.encoding, args.dry_run)
        if changed:
            updated += 1

    logging.info(
        "Processed %d file(s); %d had columns removed%s.",
        processed,
        updated,
        " (dry run)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
