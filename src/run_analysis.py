#!/usr/bin/env python3
# /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/src/run_analysis.py

import argparse
import yaml
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Adjust this path if needed to reach the correct Python package
src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
sys.path.append(src_path)

from aegle_analysis.analysis_handler import run_analysis
from aegle_analysis.analysis_annotator import DEFAULT_TISSUE_DESCRIPTOR
from aegle_analysis.analysis_annotator import DEFAULT_MODEL as DEFAULT_LLM_MODEL


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
        default="",
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


def _resolve_analysis_paths(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Augment CLI args with paths/settings derived from the analysis config."""

    analysis_cfg: Dict[str, Any] = config.get("analysis", {}) or {}
    config_dir = Path(args.config_file).resolve().parent

    if not args.data_dir:
        raise ValueError(
            "--data_dir must point to the main pipeline output root (e.g. out/main/main_ft_hb)."
        )

    rel_exp_dir = analysis_cfg.get("data_dir")
    if not rel_exp_dir:
        raise ValueError("analysis.data_dir is required in the configuration")

    experiment_dir = os.path.join(args.data_dir, rel_exp_dir)
    args.experiment_dir = experiment_dir

    cell_profiling_dir = analysis_cfg.get("cell_profiling_dir", "cell_profiling")
    args.cell_profiling_dir = os.path.join(experiment_dir, cell_profiling_dir)

    metadata_file = analysis_cfg.get("metadata_file", "cell_metadata.csv")
    expression_file = analysis_cfg.get("expression_file", "cell_by_marker.csv")
    args.cell_metadata_path = os.path.join(args.cell_profiling_dir, metadata_file)
    args.cell_expression_path = os.path.join(args.cell_profiling_dir, expression_file)

    segmentation_file = analysis_cfg.get("segmentation_file")
    args.segmentation_path = (
        os.path.join(experiment_dir, segmentation_file)
        if segmentation_file
        else None
    )
    args.segmentation_format = analysis_cfg.get("segmentation_format", "none")

    output_subdir = analysis_cfg.get("output_subdir")
    if output_subdir:
        args.output_dir = os.path.join(args.output_dir, output_subdir)

    args.plots_subdir = analysis_cfg.get("plots_subdir", "plots")
    args.de_subdir = analysis_cfg.get("de_subdir", "differential_expression")

    # Optional LLM-assisted annotation settings
    args.annotate_cell_types = bool(analysis_cfg.get("annotate_cell_types", False))

    prior_path = analysis_cfg.get("llm_prior_path")
    if prior_path:
        prior_path = Path(prior_path).expanduser()
        if not prior_path.is_absolute():
            prior_path = (config_dir / prior_path).resolve()
        args.llm_prior_path = str(prior_path)
    else:
        args.llm_prior_path = None

    args.llm_model = analysis_cfg.get("llm_model", DEFAULT_LLM_MODEL)
    args.llm_temperature = analysis_cfg.get("llm_temperature", 0.1)
    args.llm_max_tokens = int(analysis_cfg.get("llm_max_tokens", 4000))
    args.llm_system_prompt = analysis_cfg.get("llm_system_prompt")
    args.llm_output_file = analysis_cfg.get("llm_output_file")
    summarize_default = analysis_cfg.get("annotate_cell_types", False)
    args.summarize_annotation = bool(
        analysis_cfg.get("summarize_annotation", summarize_default)
    )
    args.llm_summary_system_prompt = analysis_cfg.get("llm_summary_system_prompt")
    args.llm_summary_output_file = analysis_cfg.get("llm_summary_output_file")
    args.tissue_descriptor = analysis_cfg.get(
        "tissue_descriptor", DEFAULT_TISSUE_DESCRIPTOR
    )
    args.generate_pipeline_report = bool(
        analysis_cfg.get("generate_pipeline_report", True)
    )

    # Apply analysis-level overrides for booleans/numerics
    args.skip_viz = args.skip_viz or bool(analysis_cfg.get("skip_viz", False))

    if "patch_index" in analysis_cfg:
        args.patch_index = analysis_cfg["patch_index"]

    if "clustering_resolution" in analysis_cfg:
        args.clustering_resolution = analysis_cfg["clustering_resolution"]

    args.norm_method = analysis_cfg.get("norm_method")

    logging.info("Resolved experiment directory: %s", args.experiment_dir)
    logging.info("Cell metadata path: %s", args.cell_metadata_path)
    logging.info("Cell expression path: %s", args.cell_expression_path)
    if args.segmentation_path:
        logging.info(
            "Segmentation artifact: %s (format=%s)",
            args.segmentation_path,
            args.segmentation_format,
        )
    else:
        logging.info("Segmentation artifact not specified; spatial plots may be skipped")

    logging.info("Analysis outputs will be written to: %s", args.output_dir)


def main():
    setup_logging()
    logging.info("Starting analysis main function.")
    args = parse_args()

    # Load configuration from the YAML file
    config = load_config(args.config_file)
    logging.info(f"Config: {config}")
    _resolve_analysis_paths(config, args)

    # Run the analysis pipeline
    logging.info("Running the CODEX/PhenoCycler downstream analysis pipeline.")
    run_analysis(config, args)
    logging.info("Analysis pipeline execution completed.")


if __name__ == "__main__":
    main()
