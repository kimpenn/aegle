#!/usr/bin/env python3
"""Generate algorithm and per-sample biological review HTML."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.oocyte.report import generate_html_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--rescue-delta-dir", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--patch-radius", type=int, default=180)
    parser.add_argument("--no-label-export", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = generate_html_reports(
        args.batch_dir,
        rescue_delta_dir=args.rescue_delta_dir,
        references_path=args.references,
        patch_radius_px=args.patch_radius,
        export_combined_labels=not args.no_label_export,
    )
    logging.info("Algorithm document: %s", result.algorithm_document)
    logging.info("Sample pages: %s", len(result.sample_pages))
    logging.info("Batch index: %s", result.batch_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
