#!/usr/bin/env python3
"""
Patch existing HTML pipeline reports to add Sample ID and fix Valid Patches metrics.

This script processes existing pipeline_report.html files to:
1. Add Sample ID metric after Experiment ID in the executive summary
2. Fix Valid Patches showing "0/0" by reading actual values from patches_metadata.csv

Usage:
    # Single file
    python scripts/patch_html_reports.py /path/to/pipeline_report.html

    # Directory (batch) - default targets out/main/
    python scripts/patch_html_reports.py /path/to/out/main/ --recursive

    # Options
    --dry-run           Show what would be changed without modifying files
    --no-backup         Overwrite files without creating .bak backup
    --config-dir DIR    Override config directory (default: exps/configs/main/)
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "exps" / "configs" / "main"


def find_config_yaml(report_path: Path, config_dir: Path) -> Optional[Path]:
    """
    Find the config.yaml associated with a report based on directory structure.

    Expects report path like: out/main/{exp_set}/{exp_id}/pipeline_report.html
    Config path will be: exps/configs/main/{exp_set}/{exp_id}/config.yaml
    """
    # Extract exp_set and exp_id from path
    # e.g., /path/to/out/main/main_ft_hb/D10_0/pipeline_report.html
    parts = report_path.parent.parts

    # Find 'main' in path and get exp_set and exp_id after it
    try:
        main_idx = parts.index('main')
        if main_idx + 2 < len(parts):
            exp_set = parts[main_idx + 1]
            exp_id = parts[main_idx + 2]
            config_path = config_dir / exp_set / exp_id / "config.yaml"
            if config_path.exists():
                return config_path
    except (ValueError, IndexError):
        pass

    return None


def get_sample_id(config_path: Optional[Path]) -> str:
    """Extract sample_id from config.yaml file."""
    if config_path is None or not config_path.exists():
        return "Unknown"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('sample_id', 'Unknown')
    except Exception as e:
        logger.warning(f"Failed to read config {config_path}: {e}")
        return "Unknown"


def get_patch_metrics(report_dir: Path) -> Tuple[int, int]:
    """
    Read patches_metadata.csv and return (informative_patches, total_patches).

    Checks both cell_profiling/patches_metadata.csv and patches_metadata.csv in report dir.
    """
    # Try cell_profiling subdirectory first (preferred location)
    csv_paths = [
        report_dir / "cell_profiling" / "patches_metadata.csv",
        report_dir / "patches_metadata.csv",
    ]

    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                total_patches = len(df)

                if 'is_informative' in df.columns:
                    # Handle both boolean and string values
                    informative = df['is_informative'].apply(
                        lambda x: x if isinstance(x, bool) else str(x).lower() == 'true'
                    ).sum()
                else:
                    # If no is_informative column, assume all patches are informative
                    informative = total_patches

                return int(informative), total_patches
            except Exception as e:
                logger.warning(f"Failed to read {csv_path}: {e}")

    return 0, 0


def patch_html_content(
    html_content: str,
    sample_id: str,
    informative_patches: int,
    total_patches: int
) -> Tuple[str, bool, bool]:
    """
    Patch the HTML content to add Sample ID and fix Valid Patches.

    Returns:
        Tuple of (patched_content, sample_id_added, patches_fixed)
    """
    sample_id_added = False
    patches_fixed = False

    # Pattern to find Experiment ID metric block
    # We want to insert Sample ID after the closing </div> of Experiment ID metric
    exp_id_pattern = re.compile(
        r'(<div class="metric">\s*'
        r'<div class="metric-value">[^<]*</div>\s*'
        r'<div class="metric-label">Experiment ID</div>\s*'
        r'</div>)',
        re.DOTALL
    )

    # Check if Sample ID already exists
    if '<div class="metric-label">Sample ID</div>' not in html_content:
        # Insert Sample ID metric after Experiment ID
        sample_id_metric = f'''
            <div class="metric">
                <div class="metric-value">{sample_id}</div>
                <div class="metric-label">Sample ID</div>
            </div>'''

        def insert_sample_id(match):
            return match.group(1) + sample_id_metric

        new_content = exp_id_pattern.sub(insert_sample_id, html_content)
        if new_content != html_content:
            html_content = new_content
            sample_id_added = True

    # Fix Valid Patches 0/0 pattern
    # Pattern: <div class="metric-value">0/0</div>\s*<div class="metric-label">Valid Patches</div>
    patches_pattern = re.compile(
        r'(<div class="metric-value">)0/0(</div>\s*'
        r'<div class="metric-label">Valid Patches</div>)',
        re.DOTALL
    )

    replacement = f'\\g<1>{informative_patches}/{total_patches}\\2'
    new_content = patches_pattern.sub(replacement, html_content)
    if new_content != html_content:
        html_content = new_content
        patches_fixed = True

    return html_content, sample_id_added, patches_fixed


def process_report(
    report_path: Path,
    config_dir: Path,
    dry_run: bool = False,
    create_backup: bool = True
) -> Tuple[bool, str]:
    """
    Process a single pipeline_report.html file.

    Returns:
        Tuple of (success, message)
    """
    if not report_path.exists():
        return False, f"File not found: {report_path}"

    # Read HTML content
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        return False, f"Failed to read {report_path}: {e}"

    # Get sample_id from config
    config_path = find_config_yaml(report_path, config_dir)
    sample_id = get_sample_id(config_path)

    # Get patch metrics
    report_dir = report_path.parent
    informative_patches, total_patches = get_patch_metrics(report_dir)

    # Patch the content
    patched_content, sample_id_added, patches_fixed = patch_html_content(
        html_content, sample_id, informative_patches, total_patches
    )

    if not sample_id_added and not patches_fixed:
        return True, "No changes needed"

    changes = []
    if sample_id_added:
        changes.append(f"Sample ID: {sample_id}")
    if patches_fixed:
        changes.append(f"Valid Patches: {informative_patches}/{total_patches}")

    if dry_run:
        return True, f"Would update: {', '.join(changes)}"

    # Create backup if requested
    if create_backup:
        backup_path = report_path.with_suffix('.html.bak')
        try:
            shutil.copy2(report_path, backup_path)
        except Exception as e:
            return False, f"Failed to create backup: {e}"

    # Write patched content
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)
    except Exception as e:
        return False, f"Failed to write {report_path}: {e}"

    return True, f"Updated: {', '.join(changes)}"


def find_reports(path: Path, recursive: bool = True) -> list:
    """Find all pipeline_report.html files in a directory."""
    if path.is_file():
        if path.name == 'pipeline_report.html':
            return [path]
        return []

    pattern = '**/pipeline_report.html' if recursive else 'pipeline_report.html'
    return list(path.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description='Patch HTML pipeline reports to add Sample ID and fix Valid Patches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Path to pipeline_report.html or directory containing reports'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Recursively search for reports in subdirectories (default: True)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .bak backup files'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f'Config directory (default: {DEFAULT_CONFIG_DIR})'
    )

    args = parser.parse_args()

    # Find reports
    reports = find_reports(args.path, args.recursive)

    if not reports:
        logger.error(f"No pipeline_report.html files found in {args.path}")
        sys.exit(1)

    logger.info(f"Found {len(reports)} report(s) to process")

    if args.dry_run:
        logger.info("DRY RUN - no files will be modified")

    # Process each report
    success_count = 0
    error_count = 0
    skip_count = 0

    for report_path in sorted(reports):
        success, message = process_report(
            report_path,
            args.config_dir,
            dry_run=args.dry_run,
            create_backup=not args.no_backup
        )

        if success:
            if "No changes" in message:
                skip_count += 1
                logger.debug(f"{report_path}: {message}")
            else:
                success_count += 1
                logger.info(f"{report_path}: {message}")
        else:
            error_count += 1
            logger.error(f"{report_path}: {message}")

    # Summary
    logger.info(f"\nSummary: {success_count} updated, {skip_count} skipped, {error_count} errors")

    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
