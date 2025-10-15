#!/usr/bin/env python3
"""Generate nucleus_overview.csv files from existing cell profiling outputs."""

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

DEFAULT_ROOT = Path("out/main/main_ft_hb")
DEFAULT_GLOB = "D*_*/cell_profiling/cell_overview.csv"
NUCLEUS_SUFFIX = "_nucleus_mean"
NON_MARKER_COLUMNS = {
    "cell_mask_id",
    "y",
    "x",
    "area",
    "patch_id",
    "global_cell_id",
    "patch_x",
    "patch_y",
    "patch_centroid_x",
    "patch_centroid_y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or refresh nucleus_overview.csv files, using nucleus means from "
            "cell_metadata.csv to mirror the structure of cell_overview.csv."
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
        help="Glob pattern (relative to --root) for locating cell_overview.csv files.",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        help="Explicit list of cell_overview.csv files to process instead of using --root/--glob.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be updated without writing changes.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding for reads/writes (default: utf-8)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def discover_files(root: Path, pattern: str) -> List[Path]:
    root = root.resolve()
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    logging.debug(
        "Discovered %d file(s) under %s using pattern %s", len(files), root, pattern
    )
    return files


def build_metadata_lookup(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if "cell_mask_id" not in metadata_df.columns:
        raise ValueError("metadata file does not contain a cell_mask_id column")

    deduped = metadata_df.drop_duplicates(subset=["cell_mask_id"], keep="last")
    lookup = deduped.set_index("cell_mask_id")
    logging.debug("Prepared metadata lookup with %d unique cell IDs", lookup.shape[0])
    return lookup


def extract_markers_from_metadata(metadata_df: pd.DataFrame) -> dict:
    marker_map = {}
    for col in metadata_df.columns:
        if col.endswith(NUCLEUS_SUFFIX):
            marker = col[: -len(NUCLEUS_SUFFIX)]
            marker_map[marker] = col
    logging.debug("Found %d nucleus marker columns in metadata", len(marker_map))
    return marker_map


def extract_overview_markers(columns: Iterable[str]) -> List[str]:
    markers = [
        col
        for col in columns
        if col not in NON_MARKER_COLUMNS and not col.endswith(NUCLEUS_SUFFIX)
    ]
    logging.debug("cell_overview contains %d marker columns", len(markers))
    return markers


def build_nucleus_overview(
    cell_overview_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    marker_map = extract_markers_from_metadata(metadata_df)
    overview_markers = extract_overview_markers(cell_overview_df.columns)

    missing_markers = [m for m in overview_markers if m not in marker_map]
    if missing_markers:
        logging.warning(
            "Missing nucleus mean columns for markers: %s",
            ", ".join(sorted(missing_markers)),
        )
    extra_markers = [m for m in marker_map.keys() if m not in overview_markers]
    if extra_markers:
        logging.debug(
            "Metadata contains nucleus markers absent from cell_overview: %s",
            ", ".join(sorted(extra_markers)),
        )

    lookup = build_metadata_lookup(metadata_df)
    cell_ids = cell_overview_df.get("cell_mask_id")
    if cell_ids is None:
        raise ValueError("cell_overview.csv is missing cell_mask_id column")

    missing_ids = cell_ids[~cell_ids.isin(lookup.index)]
    if not missing_ids.empty:
        logging.warning(
            "%d cell_mask_id values are missing from metadata; filling from cell_overview",
            missing_ids.nunique(),
        )

    reindexed_metadata = lookup.reindex(cell_ids)

    result = pd.DataFrame(
        {"cell_mask_id": cell_ids.values}, index=cell_overview_df.index
    )

    def assign_column(
        target: str,
        metadata_col: Optional[str],
        fallback_col: Optional[str] = None,
    ) -> None:
        series = None
        if metadata_col and metadata_col in reindexed_metadata.columns:
            series = reindexed_metadata[metadata_col]
        if series is None and fallback_col and fallback_col in cell_overview_df.columns:
            logging.debug(
                "Using fallback column %s for %s due to missing metadata", fallback_col, target
            )
            series = cell_overview_df[fallback_col]
        if series is None:
            series = pd.Series([pd.NA] * len(result), index=result.index)
        result[target] = series.to_numpy()

    assign_column("y", "y", "y")
    assign_column("x", "x", "x")
    if "nucleus_area" in reindexed_metadata.columns:
        assign_column("area", "nucleus_area")
    elif "area" in reindexed_metadata.columns:
        logging.warning("nucleus_area missing; using metadata area column instead")
        assign_column("area", "area")
    else:
        logging.warning("nucleus_area and area missing; falling back to cell_overview area")
        assign_column("area", None, "area")

    for marker in overview_markers:
        nucleus_col = marker_map.get(marker)
        if nucleus_col and nucleus_col in reindexed_metadata.columns:
            result[marker] = reindexed_metadata[nucleus_col].to_numpy()
        else:
            logging.warning(
                "Falling back to cell_overview values for marker %s", marker
            )
            fallback = cell_overview_df.get(marker)
            if fallback is not None:
                fallback_series = fallback.reindex(result.index)
            else:
                fallback_series = pd.Series([pd.NA] * len(result), index=result.index)
            result[marker] = fallback_series.to_numpy()

    output_columns = ["cell_mask_id", "y", "x", "area"] + overview_markers
    return result[output_columns]


def process_file(path: Path, *, encoding: str, dry_run: bool) -> bool:
    overview_path = path
    metadata_path = overview_path.with_name("cell_metadata.csv")
    nucleus_path = overview_path.with_name("nucleus_overview.csv")

    if not metadata_path.exists():
        logging.warning("Skipping %s (missing %s)", overview_path, metadata_path.name)
        return False

    logging.info("Processing %s", overview_path)

    cell_overview_df = pd.read_csv(overview_path, encoding=encoding)
    metadata_df = pd.read_csv(metadata_path, encoding=encoding)

    nucleus_df = build_nucleus_overview(cell_overview_df, metadata_df)

    if dry_run:
        logging.info("Dry run: would write %s", nucleus_path)
        return False

    if nucleus_path.exists():
        logging.warning("Overwriting existing %s", nucleus_path)

    nucleus_df.to_csv(nucleus_path, index=False, encoding=encoding)
    return True


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
        logging.warning("No cell_overview.csv files found to process.")
        return 1

    processed = 0
    updated = 0

    for path in files:
        if not path.exists():
            logging.warning("Skipping missing file %s", path)
            continue
        processed += 1
        try:
            changed = process_file(path, encoding=args.encoding, dry_run=args.dry_run)
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to process %s: %s", path, exc)
            continue
        if changed:
            updated += 1

    logging.info(
        "Processed %d file(s); %d nucleus_overview.csv file(s) written%s.",
        processed,
        updated,
        " (dry run)" if args.dry_run else "",
    )

    return 0 if updated or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
