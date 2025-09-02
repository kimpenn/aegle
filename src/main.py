#!/usr/bin/env python3

import argparse
import yaml  # For reading YAML configuration files
import logging
import os
import time
import traceback
import sys

src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
# Add the source directory to the Python path
import sys

sys.path.append(src_path)

# Import custom modules from the installed 'aegle' package
from aegle.pipeline import run_pipeline

# from aegle.logging_config import setup_logging


class StackTraceHandler(logging.Handler):
    """Custom handler that prints stack trace when target message is logged."""
    
    def __init__(self, target_message="Converting image dtype to float"):
        super().__init__()
        self.target_message = target_message
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    def emit(self, record):
        try:
            message = record.getMessage()
            if self.target_message.lower() in message.lower():
                formatted = self.format(record)
                print("\n" + "="*80)
                print(f"FOUND TARGET MESSAGE: {formatted}")
                print(f"Logger: {record.name}")
                print(f"File: {record.pathname}:{record.lineno}")
                print(f"Function: {record.funcName}")
                print("\nFull Stack Trace:")
                traceback.print_stack()
                print("="*80 + "\n")
        except Exception as e:
            print(f"[StackTraceHandler] emit failed: {e}", file=sys.stderr)

def setup_logging():
    """
    Configure logging with proper levels for different modules.
    
    Logs are sent to stdout, which is redirected to a file by the bash script.
    """
    # Define formatter for consistent log format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create the custom stack trace handler
    trace_handler = StackTraceHandler()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicate logs
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    root_logger.addHandler(trace_handler)
    
    # Set specific log levels for different modules
    # Main pipeline components use INFO level
    logging.getLogger("aegle.pipeline").setLevel(logging.INFO)
    logging.getLogger("aegle.codex_image").setLevel(logging.INFO)
    logging.getLogger("aegle.codex_patches").setLevel(logging.INFO)
    
    # Lower log levels for repetitive operations
    logging.getLogger("aegle.segmentation_analysis.spatial_analysis.density_metrics").setLevel(logging.WARNING)
    # Only log significant events for visualization modules
    logging.getLogger("aegle.segmentation_analysis.spatial_analysis.density_visualization").setLevel(logging.INFO)
    
    # Configure matplotlib and other third-party libs to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Add trace handler to specific loggers that might be involved
    loggers_to_trace = [
        'deepcell',
        'deepcell_toolbox', 
        'tensorflow',
        'Mesmer',
        'aegle'
    ]
    
    for logger_name in loggers_to_trace:
        logger = logging.getLogger(logger_name)
        logger.addHandler(trace_handler)
        # Ensure we catch all messages for tracing
        if logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)
    
    print("Stack trace logging enabled for 'Converting image dtype to float'")


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
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level.",
    )
    return parser.parse_args()


def load_config(config_file):
    logging.info(f"Loading configuration from {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"Full configuration loaded: {config}")
    return config


def main():
    args = parse_args()
    
    # Initialize logging
    setup_logging()
    
    # Set log level based on command line argument
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logging.info(f"Starting pipeline with log level: {args.log_level}")
    
    # Load configuration from the YAML file
    config = load_config(args.config_file)
    
    # Record key configuration parameters at INFO level
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Output directory: {args.out_dir}")
    
    # Start pipeline execution with timing
    start_time = time.time()
    logging.info("Running the CODEX image analysis pipeline.")
    run_pipeline(config, args)
    
    # Log completion with timing information
    elapsed_time = time.time() - start_time
    logging.info(f"Pipeline execution completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
