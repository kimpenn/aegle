#!/usr/bin/env python3

import argparse
import yaml  # For reading YAML configuration files

# Import custom modules from the installed 'aegle' package
from aegle.pipeline import run_pipeline
from aegle.logging_config import setup_logging


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
        "--output_dir",
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
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    setup_logging()

    # Load configuration from the YAML file
    config = load_config(args.config_file)

    # Run the pipeline
    run_pipeline(config, args)


if __name__ == "__main__":
    main()
