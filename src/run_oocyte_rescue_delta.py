#!/usr/bin/env python3
"""Generate a reviewable secondary-rescue delta from completed v6 outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.oocyte.delta import generate_rescue_delta_batch
from aegle.oocyte.review import generate_review_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the crowded-field rescue pass on completed v6 samples."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default="donor13_v6_rescue_v1",
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        help="Sample ID to process; repeat the option or omit it for all samples.",
    )
    parser.add_argument("--montage-columns", type=int, default=3)
    parser.add_argument("--montage-rows", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    result = generate_rescue_delta_batch(
        args.baseline_dir,
        out_dir=args.out_dir,
        profile_name=args.profile,
        sample_ids=args.samples,
    )
    review = generate_review_pack(
        result.out_dir,
        columns=args.montage_columns,
        rows=args.montage_rows,
        neighbor_overlays=True,
    )
    logging.info("Rescue candidates: %s", review.accepted_count)
    logging.info("Delta output: %s", result.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
