#!/usr/bin/env python3

import argparse
import yaml  # For reading YAML configuration files
import logging
import os
import time

src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
# Add the source directory to the Python path
import sys

sys.path.append(src_path)

# Import custom modules from the installed 'aegle' package
from aegle.pipeline import run_pipeline

# from aegle.logging_config import setup_logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess images into patches for CODEX image analysis pipeline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspaces/codex-analysis/data",
        help="Directory containing the data.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/output_dev",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML configuration file containing parameters.",
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
    logging.info("Starting main function.")
    args = parse_args()

    # Load configuration from the YAML file
    config = load_config(args.config_file)
    logging.info(f"Config: {config}")
    # Run the pipeline
    logging.info("Running the CODEX image analysis pipeline.")
    run_pipeline(config, args)
    logging.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()
