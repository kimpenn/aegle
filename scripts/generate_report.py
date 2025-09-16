#!/usr/bin/env python3
"""
Generate analysis report for completed pipeline runs.
Can be run after pipeline completion to create comprehensive reports.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add pipeline directory to path
sys.path.insert(0, '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline')

from aegle.report_generator import generate_pipeline_report


def load_config(output_dir):
    """Load configuration from copied config file."""
    import yaml
    
    config_path = Path(output_dir) / "copied_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logging.warning(f"Config file not found: {config_path}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Generate pipeline analysis report')
    parser.add_argument('output_dir', help='Pipeline output directory')
    parser.add_argument('--format', choices=['html', 'pdf'], default='html',
                       help='Report format (default: html)')
    parser.add_argument('--output', '-o', help='Custom output path for report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check output directory exists
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logging.error(f"Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(output_dir)
    
    # Generate report
    try:
        report_path = generate_pipeline_report(str(output_dir), config)
        print(f"\n✅ Report generated successfully: {report_path}")
        
        # Open in browser if HTML
        if args.format == 'html' and not args.output:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            
    except Exception as e:
        logging.error(f"Failed to generate report: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
