#!/usr/bin/env python3
"""Command-line entry point for standalone raw-UCHL1 oocyte detection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.oocyte.batch import detect_oocyte_batch
from aegle.oocyte.report import generate_html_reports
from aegle.oocyte.review import generate_review_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and segment oocytes directly from raw UCHL1 OME-TIFFs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", type=int)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling samples after the first failure.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute samples even when matching complete outputs already exist.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    settings = {}
    if args.config is not None:
        with args.config.open() as handle:
            settings = yaml.safe_load(handle) or {}
    experimental = settings.get("experimental", {}) or {}
    if bool(experimental.get("border_rescue_enabled", False)):
        raise ValueError("donor13_v6 does not support experimental border rescue")
    jobs = args.jobs if args.jobs is not None else int(settings.get("jobs", 1))
    continue_on_error = bool(settings.get("continue_on_error", True))
    if args.fail_fast:
        continue_on_error = False
    result = detect_oocyte_batch(
        args.manifest,
        out_dir=args.out_dir,
        jobs=jobs,
        continue_on_error=continue_on_error,
        resume_completed=not args.no_resume
        and bool(settings.get("resume_completed", True)),
        default_profile=str(settings.get("detector_profile", "donor13_v6")),
        default_channel_name=str(settings.get("channel_name", "UCHL1")),
    )
    references_value = settings.get("comparison_references")
    references_path = None
    if references_value:
        references_path = Path(str(references_value)).expanduser()
        if not references_path.is_absolute():
            config_dir = (
                args.config.resolve().parent if args.config is not None else Path.cwd()
            )
            references_path = (config_dir / references_path).resolve()
    review = settings.get("review", {}) or {}
    if bool(review.get("enabled", True)) and bool(
        (result.summary["status"] == "complete").any()
    ):
        review_result = generate_review_pack(
            result.out_dir,
            references_path=references_path,
            reference_min_final_score=float(
                review.get("reference_min_final_score", 0.35)
            ),
            match_radius_px=float(review.get("match_radius_px", 100.0)),
            columns=int(review.get("montage_columns", 3)),
            rows=int(review.get("montage_rows", 4)),
            neighbor_overlays=bool(review.get("neighbor_overlays", True)),
            max_neighbor_overlays=int(review.get("max_neighbor_overlays", 8)),
        )
        logging.info(
            "Review counts: accepted=%s novel=%s missed=%s",
            review_result.accepted_count,
            review_result.novel_count,
            review_result.missed_reference_count,
        )
    html_report = settings.get("html_report", {}) or {}
    if bool(html_report.get("enabled", True)) and bool(
        (result.summary["status"] == "complete").any()
    ):
        rescue_delta_value = html_report.get("rescue_delta_dir")
        rescue_delta_dir = None
        if rescue_delta_value:
            rescue_delta_dir = Path(str(rescue_delta_value)).expanduser().resolve()
        html_result = generate_html_reports(
            result.out_dir,
            rescue_delta_dir=rescue_delta_dir,
            references_path=references_path,
            patch_radius_px=int(html_report.get("patch_radius_px", 180)),
            export_combined_labels=bool(
                html_report.get("export_combined_labels", True)
            ),
        )
        logging.info("HTML review index: %s", html_result.batch_index)
    counts = result.summary["status"].value_counts().to_dict()
    logging.info("Batch status counts: %s", counts)
    logging.info("Batch summary: %s", result.artifact_paths["batch_summary_csv"])
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
