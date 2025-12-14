"""
Datasink module for collecting HTML reports from Aegle pipeline outputs.

This module provides functionality to:
1. Discover and collect HTML reports from pipeline outputs
2. Generate an index.html with metadata for easy navigation
3. Optionally upload collected reports to remote storage via rclone
"""

import datetime
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)

# Report types mapping by pipeline stage
REPORT_TYPES = {
    "main": [
        ("pipeline_report.html", "pipeline_report"),
    ],
    "analysis": [
        ("analysis_highlights.html", "analysis_highlights"),
        ("pipeline_report_with_analysis.html", "pipeline_report_with_analysis"),
    ],
}


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_number(n: Optional[int]) -> str:
    """Format number with comma separators."""
    if n is None:
        return "-"
    return f"{n:,}"


@dataclass
class ReportInfo:
    """Information about a collected report file."""

    exp_id: str
    report_type: str  # "pipeline_report", "analysis_highlights", etc.
    source_path: Path
    dest_filename: str  # e.g., "D10_0_pipeline_report.html"
    file_size_bytes: int
    file_mtime: datetime.datetime
    cell_count: Optional[int] = None
    stage: str = "main"

    @property
    def file_size_human(self) -> str:
        """Human-readable file size."""
        return format_size(self.file_size_bytes)

    @property
    def file_mtime_str(self) -> str:
        """Formatted modification time."""
        return self.file_mtime.strftime("%Y-%m-%d %H:%M")


class ReportCollector:
    """Collects HTML reports from pipeline outputs."""

    def __init__(
        self,
        out_root: Path,
        stage: str,
        exp_set: str,
        exp_ids: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        """
        Initialize report collector.

        Args:
            out_root: Root directory containing pipeline outputs (e.g., out/)
            stage: Pipeline stage ('main' or 'analysis')
            exp_set: Experiment set name (e.g., 'main_ft_hb')
            exp_ids: Optional list of specific experiment IDs to include
            exclude: Optional list of experiment IDs to exclude
        """
        self.out_root = Path(out_root)
        self.stage = stage
        self.exp_set = exp_set
        self.exp_ids = exp_ids
        self.exclude = exclude or []

        # Determine source directory based on stage
        self.source_dir = self.out_root / stage / exp_set

    def discover_experiments(self) -> List[str]:
        """
        Discover all experiment directories in the exp_set.

        Returns:
            List of experiment IDs found in the source directory
        """
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return []

        experiments = []
        for item in sorted(self.source_dir.iterdir()):
            if item.is_dir():
                exp_id = item.name
                # Skip if not in explicit list (when provided)
                if self.exp_ids and exp_id not in self.exp_ids:
                    continue
                # Skip if in exclude list
                if exp_id in self.exclude:
                    logger.info(f"Excluding experiment: {exp_id}")
                    continue
                experiments.append(exp_id)

        logger.info(f"Discovered {len(experiments)} experiments in {self.source_dir}")
        return experiments

    def collect_reports(self) -> List[ReportInfo]:
        """
        Collect all HTML reports from discovered experiments.

        Returns:
            List of ReportInfo objects for all found reports
        """
        reports = []
        experiments = self.discover_experiments()

        for exp_id in experiments:
            exp_dir = self.source_dir / exp_id

            for filename, report_type in REPORT_TYPES.get(self.stage, []):
                report_path = exp_dir / filename

                if not report_path.exists():
                    logger.debug(f"Report not found: {report_path}")
                    continue

                try:
                    stat = report_path.stat()
                    cell_count = self.get_cell_count(exp_dir)

                    reports.append(
                        ReportInfo(
                            exp_id=exp_id,
                            report_type=report_type,
                            source_path=report_path,
                            dest_filename=f"{exp_id}_{report_type}.html",
                            file_size_bytes=stat.st_size,
                            file_mtime=datetime.datetime.fromtimestamp(stat.st_mtime),
                            cell_count=cell_count,
                            stage=self.stage,
                        )
                    )
                    logger.debug(f"Collected: {report_path}")

                except Exception as e:
                    logger.error(f"Error processing {report_path}: {e}")
                    continue

        logger.info(f"Collected {len(reports)} reports from {len(experiments)} experiments")
        return reports

    def get_cell_count(self, exp_dir: Path) -> Optional[int]:
        """
        Extract cell count from cell profiling data.

        Args:
            exp_dir: Experiment output directory

        Returns:
            Number of cells, or None if not available
        """
        # Try cell_overview.csv first (main pipeline)
        cell_overview = exp_dir / "cell_profiling" / "cell_overview.csv"
        if cell_overview.exists():
            try:
                # Only read first column for speed
                df = pd.read_csv(cell_overview, usecols=[0], nrows=None)
                return len(df)
            except Exception as e:
                logger.debug(f"Failed to read cell count from {cell_overview}: {e}")

        # Try patches_metadata.csv (alternative source)
        patches_meta = exp_dir / "cell_profiling" / "patches_metadata.csv"
        if patches_meta.exists():
            try:
                df = pd.read_csv(patches_meta)
                if "n_cells" in df.columns:
                    return int(df["n_cells"].sum())
            except Exception as e:
                logger.debug(f"Failed to read cell count from {patches_meta}: {e}")

        return None


class DataSink:
    """Main datasink coordinator for collecting and organizing reports."""

    def __init__(
        self,
        collector: ReportCollector,
        sink_dir: Path,
        dry_run: bool = False,
    ):
        """
        Initialize datasink.

        Args:
            collector: ReportCollector instance
            sink_dir: Output directory for collected reports
            dry_run: If True, don't actually copy files
        """
        self.collector = collector
        self.sink_dir = Path(sink_dir)
        self.dry_run = dry_run
        self.reports: List[ReportInfo] = []

    def run(self) -> Path:
        """
        Execute the datasink operation.

        Returns:
            Path to the sink directory
        """
        # Collect reports
        self.reports = self.collector.collect_reports()

        if not self.reports:
            logger.warning("No reports found to collect")
            return self.sink_dir

        # Create sink directory
        if not self.dry_run:
            self.sink_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created sink directory: {self.sink_dir}")
        else:
            logger.info(f"[DRY RUN] Would create sink directory: {self.sink_dir}")

        # Copy reports
        self.copy_reports()

        # Generate index
        index_html = self.generate_index()
        if not self.dry_run:
            index_path = self.sink_dir / "index.html"
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(index_html)
            logger.info(f"Generated index at: {index_path}")
        else:
            logger.info("[DRY RUN] Would generate index.html")

        return self.sink_dir

    def copy_reports(self) -> None:
        """Copy reports to sink directory with prefixed naming."""
        for report in self.reports:
            dest_path = self.sink_dir / report.dest_filename

            if self.dry_run:
                logger.info(f"[DRY RUN] Would copy: {report.source_path} -> {dest_path}")
            else:
                shutil.copy2(report.source_path, dest_path)
                logger.debug(f"Copied: {report.source_path} -> {dest_path}")

        if not self.dry_run:
            logger.info(f"Copied {len(self.reports)} reports to {self.sink_dir}")

    def generate_index(self) -> str:
        """
        Generate index.html with metadata.

        Returns:
            HTML content for index page
        """
        # Calculate summary statistics
        total_reports = len(self.reports)
        total_experiments = len(set(r.exp_id for r in self.reports))
        total_size_bytes = sum(r.file_size_bytes for r in self.reports)
        total_cells = sum(r.cell_count or 0 for r in self.reports)

        # Sort reports by exp_id and report_type
        sorted_reports = sorted(self.reports, key=lambda r: (r.exp_id, r.report_type))

        template_data = {
            "exp_set": self.collector.exp_set,
            "stage": self.collector.stage,
            "total_experiments": total_experiments,
            "total_reports": total_reports,
            "total_cells": format_number(total_cells) if total_cells > 0 else "-",
            "total_size_human": format_size(total_size_bytes),
            "reports": sorted_reports,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return self._render_index_template(template_data)

    def _render_index_template(self, data: Dict[str, Any]) -> str:
        """Render the index HTML template."""
        template = Template(INDEX_HTML_TEMPLATE)
        return template.render(**data, format_number=format_number)


class RemoteUploader:
    """Upload collected reports to remote storage via rclone."""

    def __init__(
        self,
        remote: str,
        remote_path: str,
        dry_run: bool = False,
    ):
        """
        Initialize remote uploader.

        Args:
            remote: rclone remote name (e.g., 'box:')
            remote_path: Base path on remote storage
            dry_run: If True, don't actually upload
        """
        self.remote = remote.rstrip(":") + ":"  # Ensure trailing colon
        self.remote_path = remote_path
        self.dry_run = dry_run

    def check_rclone(self) -> bool:
        """
        Check if rclone is available and the remote is configured.

        Returns:
            True if rclone is ready, False otherwise
        """
        try:
            # Check if rclone exists
            result = subprocess.run(
                ["rclone", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.error("rclone is not installed or not in PATH")
                return False

            # Check if remote is configured
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if self.remote not in result.stdout:
                logger.error(f"Remote '{self.remote}' is not configured in rclone")
                logger.info(f"Available remotes: {result.stdout.strip()}")
                return False

            logger.info(f"rclone is configured with remote: {self.remote}")
            return True

        except FileNotFoundError:
            logger.error("rclone is not installed")
            return False
        except subprocess.TimeoutExpired:
            logger.error("rclone command timed out")
            return False

    def upload(self, local_dir: Path, folder_name: str) -> bool:
        """
        Upload directory to remote storage.

        Args:
            local_dir: Local directory to upload
            folder_name: Name for the remote folder

        Returns:
            True if upload succeeded, False otherwise
        """
        remote_dest = f"{self.remote}{self.remote_path}/{folder_name}"

        cmd = [
            "rclone",
            "copy",
            str(local_dir),
            remote_dest,
            "--progress",
            "--transfers",
            "4",  # Parallel transfers
        ]

        if self.dry_run:
            cmd.append("--dry-run")
            logger.info(f"[DRY RUN] Would upload to: {remote_dest}")

        logger.info(f"Uploading to: {remote_dest}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Show progress
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"rclone upload failed with return code: {result.returncode}")
                return False

            logger.info(f"Successfully uploaded to {remote_dest}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Upload timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False


def validate_and_summarize(reports: List[ReportInfo]) -> Dict[str, Any]:
    """
    Validate collected reports and generate summary.

    Args:
        reports: List of collected ReportInfo objects

    Returns:
        Summary dictionary with statistics
    """
    summary = {
        "total_reports": len(reports),
        "total_experiments": len(set(r.exp_id for r in reports)),
        "total_size_bytes": sum(r.file_size_bytes for r in reports),
        "total_cells": sum(r.cell_count or 0 for r in reports),
        "missing_cell_counts": sum(1 for r in reports if r.cell_count is None),
        "by_type": {},
    }

    for report in reports:
        if report.report_type not in summary["by_type"]:
            summary["by_type"][report.report_type] = 0
        summary["by_type"][report.report_type] += 1

    return summary


def run_datasink(
    stage: str,
    exp_set: str,
    out_root: str = "out",
    sink_dir: Optional[str] = None,
    exp_ids: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    upload: bool = False,
    remote: str = "box:",
    remote_path: str = "aegle-reports",
    dry_run: bool = False,
) -> Optional[Path]:
    """
    Main datasink workflow.

    Args:
        stage: Pipeline stage ('main' or 'analysis')
        exp_set: Experiment set name
        out_root: Root directory containing pipeline outputs
        sink_dir: Output directory for collected reports
        exp_ids: Optional list of specific experiment IDs
        exclude: Optional list of experiment IDs to exclude
        upload: Whether to upload to remote storage
        remote: rclone remote name
        remote_path: Base path on remote storage
        dry_run: If True, don't actually copy/upload files

    Returns:
        Path to sink directory, or None if failed
    """
    logger.info(f"Starting datasink: stage={stage}, exp_set={exp_set}")

    # Create collector
    collector = ReportCollector(
        out_root=Path(out_root),
        stage=stage,
        exp_set=exp_set,
        exp_ids=exp_ids,
        exclude=exclude,
    )

    # Verify source directory exists
    if not collector.source_dir.exists():
        logger.error(f"Source directory does not exist: {collector.source_dir}")
        return None

    # Discover and collect reports
    reports = collector.collect_reports()

    if not reports:
        logger.error("No reports found. Check stage and exp_set arguments.")
        return None

    # Generate summary
    summary = validate_and_summarize(reports)
    logger.info(
        f"Found {summary['total_reports']} reports from {summary['total_experiments']} experiments"
    )
    logger.info(f"Total size: {format_size(summary['total_size_bytes'])}")
    if summary["missing_cell_counts"] > 0:
        logger.warning(
            f"{summary['missing_cell_counts']} reports missing cell count information"
        )

    # Create sink directory path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{exp_set}_{timestamp}"

    if sink_dir:
        sink_path = Path(sink_dir)
    else:
        sink_path = Path(out_root) / "datasink" / folder_name

    # Create datasink and run
    datasink = DataSink(
        collector=collector,
        sink_dir=sink_path,
        dry_run=dry_run,
    )
    datasink.reports = reports
    datasink.run()

    # Upload if requested
    if upload:
        uploader = RemoteUploader(
            remote=remote,
            remote_path=remote_path,
            dry_run=dry_run,
        )

        if not uploader.check_rclone():
            logger.error("rclone check failed. Skipping upload.")
        else:
            success = uploader.upload(sink_path, folder_name)
            if success:
                logger.info("Upload completed successfully")
            else:
                logger.error("Upload failed")

    logger.info(f"Datasink completed. Output: {sink_path}")
    return sink_path


# HTML template for index page
INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aegle Reports Index - {{ exp_set }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .summary-box {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .metric {
            text-align: center;
            min-width: 120px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #e8f5e9;
        }
        a {
            color: #2196F3;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .file-size {
            color: #666;
            font-size: 0.9em;
        }
        .cell-count {
            font-weight: bold;
        }
        .report-type {
            text-transform: capitalize;
        }
        .footer {
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #888;
            font-size: 12px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Aegle Reports Index</h1>

        <div class="summary-box">
            <div class="metric">
                <div class="metric-value">{{ exp_set }}</div>
                <div class="metric-label">Experiment Set</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ stage }}</div>
                <div class="metric-label">Stage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ total_experiments }}</div>
                <div class="metric-label">Experiments</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ total_reports }}</div>
                <div class="metric-label">Reports</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ total_cells }}</div>
                <div class="metric-label">Total Cells</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ total_size_human }}</div>
                <div class="metric-label">Total Size</div>
            </div>
        </div>

        <h2>Reports</h2>
        <table>
            <thead>
                <tr>
                    <th>Experiment ID</th>
                    <th>Report Type</th>
                    <th>Cell Count</th>
                    <th>File Size</th>
                    <th>Last Modified</th>
                    <th>Link</th>
                </tr>
            </thead>
            <tbody>
                {% for report in reports %}
                <tr>
                    <td>{{ report.exp_id }}</td>
                    <td class="report-type">{{ report.report_type | replace("_", " ") }}</td>
                    <td class="cell-count">{{ format_number(report.cell_count) }}</td>
                    <td class="file-size">{{ report.file_size_human }}</td>
                    <td>{{ report.file_mtime_str }}</td>
                    <td><a href="{{ report.dest_filename }}" target="_blank">View Report</a></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="footer">
            Generated: {{ generated_at }}<br>
            Aegle Pipeline Datasink v0.1
        </div>
    </div>
</body>
</html>
"""
