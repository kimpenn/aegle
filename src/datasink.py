#!/usr/bin/env python3
"""
CLI entry point for the Aegle datasink module.

Collects HTML reports from pipeline outputs and optionally uploads to remote storage.

Example usage:
    # Collect main pipeline reports
    python src/datasink.py --stage main --exp_set main_ft_hb

    # Collect and upload analysis reports to Box
    python src/datasink.py --stage analysis --exp_set analysis_ft_hb --upload --remote box:

    # Dry run to preview what would be collected
    python src/datasink.py --stage main --exp_set main_ft_hb --dry_run

    # Collect specific experiments
    python src/datasink.py --stage main --exp_set main_ft_hb --exp_ids D10_0 D10_1 D10_2
"""

import argparse
import logging
import sys
from pathlib import Path

# Add source directory to path
src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
sys.path.insert(0, src_path)

from aegle.datasink import run_datasink, format_size


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging with proper levels.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.handlers = []
    root_logger.addHandler(console_handler)

    # Set specific log levels
    logging.getLogger("aegle.datasink").setLevel(getattr(logging, log_level))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect HTML reports from Aegle pipeline outputs and optionally upload to remote storage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect main pipeline reports (dry run)
  python src/datasink.py --stage main --exp_set main_ft_hb --dry_run

  # Collect and upload analysis reports to Box
  python src/datasink.py --stage analysis --exp_set analysis_ft_hb --upload --remote box:

  # Collect specific experiments only
  python src/datasink.py --stage main --exp_set main_ft_hb --exp_ids D10_0 D10_1

  # Exclude problematic experiments
  python src/datasink.py --stage main --exp_set main_ft_hb --exclude D11_0_bad
        """,
    )

    # Required arguments
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["main", "analysis"],
        help="Pipeline stage to collect reports from.",
    )
    parser.add_argument(
        "--exp_set",
        type=str,
        required=True,
        help="Experiment set name (e.g., 'main_ft_hb', 'analysis_ft_hb').",
    )

    # Path arguments
    parser.add_argument(
        "--out_root",
        type=str,
        default="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out",
        help="Root directory containing pipeline outputs. Default: out/",
    )
    parser.add_argument(
        "--sink_dir",
        type=str,
        default=None,
        help="Output directory for collected reports. Default: out/datasink/<exp_set>_<timestamp>",
    )

    # Filter arguments
    parser.add_argument(
        "--exp_ids",
        type=str,
        nargs="*",
        default=None,
        help="Specific experiment IDs to include. If not specified, all experiments are included.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Experiment IDs to exclude.",
    )

    # Upload arguments
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload collected reports to remote storage via rclone.",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="box:",
        help="rclone remote name (e.g., 'box:', 'gdrive:'). Default: 'box:'",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        default="aegle-reports",
        help="Base path on remote storage. Default: 'aegle-reports'",
    )

    # Operation modes
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually copying or uploading files.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level. Default: INFO",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Aegle Datasink")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied or uploaded")

    logger.info(f"Stage: {args.stage}")
    logger.info(f"Experiment set: {args.exp_set}")
    logger.info(f"Output root: {args.out_root}")

    if args.exp_ids:
        logger.info(f"Include experiments: {args.exp_ids}")
    if args.exclude:
        logger.info(f"Exclude experiments: {args.exclude}")
    if args.upload:
        logger.info(f"Upload enabled: {args.remote}{args.remote_path}/")

    # Run datasink
    try:
        sink_path = run_datasink(
            stage=args.stage,
            exp_set=args.exp_set,
            out_root=args.out_root,
            sink_dir=args.sink_dir,
            exp_ids=args.exp_ids,
            exclude=args.exclude,
            upload=args.upload,
            remote=args.remote,
            remote_path=args.remote_path,
            dry_run=args.dry_run,
        )

        if sink_path:
            logger.info("=" * 60)
            logger.info("Datasink completed successfully!")
            logger.info(f"Output directory: {sink_path}")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("Datasink failed - no output generated")
            return 1

    except Exception as e:
        logger.exception(f"Datasink failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
