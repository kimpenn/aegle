#!/usr/bin/env python3
# /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/src/run_analysis.py

import argparse
import yaml
import logging
import os
import sys

# Adjust this path if needed to reach the correct Python package
src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
sys.path.append(src_path)

from aegle_analysis.analysis_handler import run_analysis


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the CODEX/PhenoCycler downstream analysis pipeline."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    # Default arguments – you can override them in YAML or command line
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/test-ft/exp-1",
        help="Directory containing the data.",
    )
    parser.add_argument(
        "--patch_index",
        type=int,
        default=0,
        help="Index of the patch to load for boundary plotting.",
    )
    parser.add_argument(
        "--skip_viz",
        action="store_true",
        help="Skip visualization steps to speed up processing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output files and figures.",
    )
    parser.add_argument(
        "--clustering_resolution",
        type=float,
        default=0.2,
        help="Resolution parameter for Leiden clustering.",
    )
    return parser.parse_args()


def load_config(config_file):
    logging.info(f"Loading configuration from {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")
    return config


def main():
    setup_logging()
    logging.info("Starting analysis main function.")
    args = parse_args()

    # Load configuration from the YAML file
    config = load_config(args.config_file)

    # Run the analysis pipeline
    logging.info("Running the CODEX/PhenoCycler downstream analysis pipeline.")
    run_analysis(config, args)
    logging.info("Analysis pipeline execution completed.")


if __name__ == "__main__":
    main()
