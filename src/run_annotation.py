#!/usr/bin/env python3
"""Standalone entry point to run LLM-based cluster annotation on existing analysis outputs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import anndata as ad

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from aegle_analysis.analysis_annotator import (
    annotate_clusters,
    summarize_annotation,
    load_json_file,
    save_results,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
)


def resolve_path(base_dir: Path, value: str | None) -> str | None:
    """Resolve configuration paths relative to the config directory."""
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM annotation on existing analysis outputs")
    parser.add_argument("--config_file", required=True, help="Path to analysis config YAML")
    parser.add_argument("--analysis_out_dir", required=True, help="Existing analysis output directory for the experiment")
    parser.add_argument("--cluster_payload", help="Optional override for cluster payload JSON (defaults to differential_expression/llm_cluster_payload.json)")
    parser.add_argument("--prior_path", help="Optional override for prior knowledge JSON (defaults to config value)")
    parser.add_argument("--model", help="Override LLM model (defaults to config value)")
    parser.add_argument("--system_prompt", help="Override system prompt text")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    parser.add_argument("--max_tokens", type=int, help="Override max response tokens")
    parser.add_argument("--output_file", help="Override output file name for annotation text")
    parser.add_argument("--log_level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config_path = Path(args.config_file).resolve()
    if not config_path.is_file():
        raise SystemExit(f"config_file not found: {config_path}")

    config_dir = config_path.parent
    config_data = load_config(config_path)
    logging.info(f"Config: {config_data}")
    analysis_cfg = config_data.get("analysis", {}) or {}

    output_dir = Path(args.analysis_out_dir).resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"analysis_out_dir not found: {output_dir}")

    de_subdir = analysis_cfg.get("de_subdir", "differential_expression")
    de_dir = output_dir / de_subdir
    if not de_dir.is_dir():
        raise SystemExit(f"Differential expression directory missing: {de_dir}")

    cluster_payload_path = args.cluster_payload or de_dir / "llm_cluster_payload.json"
    cluster_payload_path = Path(cluster_payload_path).resolve()
    if not cluster_payload_path.is_file():
        raise SystemExit(f"Cluster payload JSON not found: {cluster_payload_path}")

    cluster_payload = load_json_file(cluster_payload_path)
    if not cluster_payload:
        raise SystemExit(f"Cluster payload is empty or invalid: {cluster_payload_path}")

    # Resolve prior knowledge path
    prior_path = args.prior_path or analysis_cfg.get("llm_prior_path")
    prior_path = resolve_path(config_dir, prior_path)
    if not prior_path:
        raise SystemExit("No prior knowledge path provided (config.llm_prior_path or --prior_path required)")

    prior_data = load_json_file(prior_path)
    if not prior_data:
        raise SystemExit(f"Prior knowledge JSON empty or invalid: {prior_path}")

    model = args.model or analysis_cfg.get("llm_model", "gpt-5-2025-08-07")
    temperature = args.temperature if args.temperature is not None else float(analysis_cfg.get("llm_temperature", 1))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(analysis_cfg.get("llm_max_tokens", 4000))
    system_prompt = (
        args.system_prompt
        if args.system_prompt is not None
        else analysis_cfg.get("llm_system_prompt")
    ) or DEFAULT_SYSTEM_PROMPT
    summarize = bool(analysis_cfg.get("summarize_annotation", analysis_cfg.get("annotate_cell_types", False)))
    summary_system_prompt = analysis_cfg.get("llm_summary_system_prompt") or DEFAULT_SUMMARY_SYSTEM_PROMPT
    summary_output_filename = analysis_cfg.get("llm_summary_output_file")

    logging.info("Running LLM annotation...")
    annotation_text = annotate_clusters(
        prior_data,
        cluster_payload,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    if not annotation_text:
        raise SystemExit("LLM returned empty annotation result")

    annotation_filename = (
        args.output_file
        or analysis_cfg.get("llm_output_file")
        or "llm_annotation.txt"
    )
    annotation_path = de_dir / annotation_filename
    save_results(annotation_text, annotation_path)

    summary_text = ""
    summary_path: Path | None = None
    if summarize:
        logging.info("Running LLM annotation summary...")
        try:
            summary_text = summarize_annotation(
                prior_data,
                cluster_payload,
                annotation_text,
                model=model,
                system_prompt=summary_system_prompt,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        except Exception as exc:
            logging.error("LLM annotation summary failed: %s", exc)
            summary_text = ""
        if summary_text:
            summary_filename = summary_output_filename or "llm_annotation_summary.txt"
            summary_path = de_dir / summary_filename
            save_results(summary_text, summary_path)
        else:
            logging.warning("LLM returned empty summary response; skipping summary save.")

    # Persist annotation metadata to the existing h5ad file
    h5ad_path = output_dir / "codex_analysis.h5ad"
    if h5ad_path.is_file():
        logging.info("Updating %s with annotation metadata", h5ad_path)
        adata = ad.read_h5ad(h5ad_path)
        adata.uns.pop("llm_cluster_payload", None)
        adata.uns["llm_annotation"] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prior_path": prior_path,
            "cluster_payload_path": str(cluster_payload_path),
            "result_path": str(annotation_path),
            "result_text": annotation_text,
        }
        if summarize and summary_text and summary_path is not None:
            adata.uns["llm_annotation"].update(
                {
                    "summary_path": str(summary_path),
                    "summary_text": summary_text,
                }
            )
        adata.write(h5ad_path, compression="gzip")
    else:
        logging.warning("h5ad file not found at %s; skipping uns metadata update", h5ad_path)

    logging.info("Annotation complete. Result saved to %s", annotation_path)
    if summarize and summary_text and summary_path is not None:
        logging.info("Annotation summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
