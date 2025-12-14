# /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle_analysis/analysis.py

import base64
import logging
import os
import json
import html
import re
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import shutil
from typing import List, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import modules from the package
from aegle_analysis.data import (
    load_metadata,
    load_expression,
    process_dapi,
    load_segmentation_data,
    prepare_data_for_analysis,
    log1p_transform,
)

from aegle_analysis.analysis import (
    create_anndata,
    run_clustering,
    one_vs_rest_wilcoxon,
    build_fold_change_matrix,
    export_de_results,
)

from aegle_analysis.visualization import (
    plot_marker_distributions,
    plot_cell_metrics,
    plot_heatmap,
    plot_segmentation_masks,
    plot_clustering_on_mask,
    plot_umap,
)

from aegle_analysis.utils import ensure_dir
from aegle_analysis.analysis_annotator import (
    annotate_clusters,
    summarize_annotation,
    load_json_file,
    save_results,
    DEFAULT_MODEL,
    DEFAULT_TISSUE_DESCRIPTOR,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
)


LOGGER = logging.getLogger(__name__)

HIGHLIGHTS_STYLE = """
<style>
.analysis-highlights {
    background-color: #ffffff;
    border: 1px solid #dfe3e8;
    border-radius: 8px;
    padding: 24px;
    margin: 30px 0;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}
.analysis-highlights h2 {
    margin-top: 0;
}
.analysis-card {
    margin-top: 24px;
}
.analysis-card h3 {
    margin-bottom: 12px;
}
.analysis-pre {
    background-color: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow: auto;
}
.analysis-figure-grid {
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.analysis-figure {
    text-align: center;
    margin: 0;
}
.analysis-figure img {
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 6px;
    margin: 0 auto;
}
.analysis-figure figcaption {
    margin-top: 8px;
    font-size: 14px;
    color: #555;
}
.analysis-details summary {
    font-weight: 600;
    cursor: pointer;
}
.analysis-metric-table td,
.analysis-metric-table th {
    text-align: left;
}
</style>
"""


def _safe_read_text(path: Optional[Path], descriptor: str) -> Optional[str]:
    """Return file content or None while logging useful diagnostics."""

    if path is None:
        return None

    if not path.is_file():
        LOGGER.warning("Expected %s at %s but file is missing.", descriptor, path)
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to read %s from %s: %s", descriptor, path, exc)
        return None

    if not content:
        LOGGER.warning("%s at %s is empty.", descriptor, path)
        return None

    return content


def _extract_style_block(html_text: str) -> str:
    """Return the first <style> block from the provided HTML text."""

    match = re.search(r"<style[\s\S]*?</style>", html_text, re.IGNORECASE)
    return match.group(0) if match else ""


def _load_and_compress_image(
    image_path: Path,
    max_width: int = 1200,
    jpeg_quality: int = 85,
) -> Optional[str]:
    """
    Load an image file, compress it, and return as base64 data URI.

    Args:
        image_path: Path to the image file
        max_width: Maximum width in pixels (maintains aspect ratio)
        jpeg_quality: JPEG compression quality (1-100)

    Returns:
        Base64 data URI string (e.g., "data:image/jpeg;base64,...") or None if failed
    """
    if not PIL_AVAILABLE:
        LOGGER.warning("PIL not available; cannot embed image as base64: %s", image_path)
        return None

    if not image_path.is_file():
        LOGGER.warning("Image file not found: %s", image_path)
        return None

    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed (JPEG doesn't support alpha)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if wider than max_width
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            # Save as JPEG to buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
            buffer.seek(0)

            # Encode as base64
            base64_data = base64.b64encode(buffer.read()).decode('ascii')
            return f"data:image/jpeg;base64,{base64_data}"

    except Exception as exc:
        LOGGER.error("Failed to load and compress image %s: %s", image_path, exc)
        return None


def _build_analysis_highlights_section(
    output_dir: Path,
    de_dir: Path,
    plots_dir: Path,
    cluster_sizes,
    annotation_path: Optional[Path],
    summary_path: Optional[Path],
    include_style: bool = True,
):
    """Assemble the HTML fragment summarizing downstream analysis outputs."""

    section_parts: List[str] = []
    if include_style:
        section_parts.append(HIGHLIGHTS_STYLE)

    section_parts.append('<section id="analysis-highlights" class="analysis-highlights">')
    section_parts.append('<h2>Analysis Highlights</h2>')

    # Cluster size overview
    if cluster_sizes is not None and len(cluster_sizes) > 0:
        size_rows = []
        try:
            cluster_items = list(cluster_sizes.items())
        except AttributeError:
            cluster_items = list(cluster_sizes.to_dict().items()) if cluster_sizes is not None else []

        def _cluster_sort_key(item):
            key, _ = item
            try:
                return (0, int(key))
            except (ValueError, TypeError):
                return (1, str(key))

        for cluster_id, count in sorted(cluster_items, key=_cluster_sort_key):
            size_rows.append(
                f"<tr><td>{html.escape(str(cluster_id))}</td><td>{count:,}</td></tr>"
            )
        section_parts.append('<div class="analysis-card">')
        section_parts.append('<h3>Cluster Sizes</h3>')
        section_parts.append('<table class="analysis-metric-table">')
        section_parts.append('<thead><tr><th>Cluster</th><th>Cells</th></tr></thead>')
        section_parts.append('<tbody>')
        section_parts.extend(size_rows)
        section_parts.append('</tbody></table></div>')
    else:
        section_parts.append(
            '<div class="analysis-card"><h3>Cluster Sizes</h3><p>No cluster size metadata was produced.</p></div>'
        )

    # Top marker summary
    top_markers_path = de_dir / "top_10_genes_summary.json"
    top_marker_rows = []
    if top_markers_path.is_file():
        try:
            with top_markers_path.open("r", encoding="utf-8") as handle:
                top_markers_data = json.load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Unable to parse top marker summary at %s: %s", top_markers_path, exc)
            top_markers_data = {}
        for cluster_id, markers in sorted(top_markers_data.items(), key=lambda item: str(item[0])):
            marker_text = ", ".join(markers) if markers else "None"
            top_marker_rows.append(
                f"<tr><td>{html.escape(str(cluster_id))}</td><td>{html.escape(marker_text)}</td></tr>"
            )
    else:
        LOGGER.warning("Top marker summary not found at %s", top_markers_path)

    if top_marker_rows:
        section_parts.append('<div class="analysis-card">')
        section_parts.append('<h3>Top Markers Per Cluster</h3>')
        section_parts.append('<table class="analysis-metric-table">')
        section_parts.append('<thead><tr><th>Cluster</th><th>Markers</th></tr></thead>')
        section_parts.append('<tbody>')
        section_parts.extend(top_marker_rows)
        section_parts.append('</tbody></table></div>')
    else:
        section_parts.append(
            '<div class="analysis-card"><h3>Top Markers Per Cluster</h3><p>No differential expression summary could be rendered.</p></div>'
        )

    # LLM summary and annotation
    summary_text = _safe_read_text(summary_path, "LLM annotation summary")
    annotation_text = _safe_read_text(annotation_path, "LLM annotation detail")

    if summary_text:
        section_parts.append('<div class="analysis-card">')
        section_parts.append('<h3>LLM Summary</h3>')
        section_parts.append(f'<pre class="analysis-pre">{html.escape(summary_text)}</pre>')
        section_parts.append('</div>')
    else:
        section_parts.append(
            '<div class="analysis-card"><h3>LLM Summary</h3><p>No summary text is available.</p></div>'
        )

    if annotation_text:
        section_parts.append('<div class="analysis-card">')
        section_parts.append('<details class="analysis-details">')
        section_parts.append('<summary>Expand full LLM annotation</summary>')
        section_parts.append(f'<pre class="analysis-pre">{html.escape(annotation_text)}</pre>')
        section_parts.append('</details></div>')
    else:
        section_parts.append(
            '<div class="analysis-card"><h3>LLM Annotation</h3><p>No annotation text is available.</p></div>'
        )

    # Key figures - embed as base64 for self-contained HTML
    figure_specs = [
        ("Cluster Heatmap (log2 Fold Change Top 5)", plots_dir / "cluster_heatmap_log2fc_top5.png"),
        ("Spatial Clustering Overlay", plots_dir / "spatial" / "clustering_on_mask.png"),
    ]
    figure_blocks: List[str] = []
    for title, figure_path in figure_specs:
        if figure_path.is_file():
            # Embed image as compressed base64 for self-contained HTML
            base64_src = _load_and_compress_image(figure_path, max_width=1200, jpeg_quality=85)
            if base64_src:
                figure_blocks.append(
                    f'<figure class="analysis-figure"><img src="{base64_src}" alt="{html.escape(title)}" />'
                    f'<figcaption>{html.escape(title)}</figcaption></figure>'
                )
            else:
                # Fallback to relative path if compression failed
                rel_path = os.path.relpath(figure_path, output_dir).replace(os.sep, "/")
                figure_blocks.append(
                    f'<figure class="analysis-figure"><img src="{rel_path}" alt="{html.escape(title)}" />'
                    f'<figcaption>{html.escape(title)}</figcaption></figure>'
                )
                LOGGER.warning("Failed to embed image as base64, using relative path: %s", figure_path)
        else:
            LOGGER.info("Key figure not found: %s", figure_path)

    if figure_blocks:
        section_parts.append('<div class="analysis-card">')
        section_parts.append('<h3>Key Visualisations</h3>')
        section_parts.append('<div class="analysis-figure-grid">')
        section_parts.extend(figure_blocks)
        section_parts.append('</div></div>')

    section_parts.append('</section>')

    return "".join(section_parts)


def _maybe_generate_pipeline_report(
    args,
    output_dir: str,
    de_dir: str,
    plots_dir: str,
    cluster_sizes,
    annotation_path: Optional[Path],
    summary_path: Optional[Path],
):
    """Copy the main pipeline report and append analysis highlights when requested."""

    if not getattr(args, "generate_pipeline_report", True):
        LOGGER.info("Skipping pipeline report augmentation because generate_pipeline_report is False.")
        return

    experiment_dir = getattr(args, "experiment_dir", None)
    if not experiment_dir:
        LOGGER.warning("Experiment directory is not available on args; cannot locate pipeline report.")
        return

    source_report = Path(experiment_dir) / "pipeline_report.html"
    if not source_report.is_file():
        LOGGER.warning("Pipeline report not found at %s; skipping copy.", source_report)
        return

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    target_report = output_dir_path / "pipeline_report_with_analysis.html"
    try:
        shutil.copyfile(source_report, target_report)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to copy pipeline report from %s to %s: %s", source_report, target_report, exc)
        return

    LOGGER.info("Copied pipeline report to %s", target_report)

    highlight_html = _build_analysis_highlights_section(
        output_dir=output_dir_path,
        de_dir=Path(de_dir),
        plots_dir=Path(plots_dir),
        cluster_sizes=cluster_sizes,
        annotation_path=annotation_path,
        summary_path=summary_path,
        include_style=True,
    )

    try:
        report_html = target_report.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Unable to read copied pipeline report at %s: %s", target_report, exc)
        return

    insertion_order = [
        '<div class="success"',
        '</div>\n    <script',
        '</body>'
    ]
    updated_html = None
    for anchor in insertion_order:
        idx = report_html.find(anchor)
        if idx != -1:
            updated_html = report_html[:idx] + highlight_html + "\n" + report_html[idx:]
            LOGGER.info("Inserted analysis highlights before anchor %s", anchor)
            break

    if updated_html is None:
        LOGGER.warning("Failed to find an insertion point for analysis highlights in %s", target_report)
        updated_html = report_html + "\n" + highlight_html

    try:
        target_report.write_text(updated_html, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Unable to write enhanced pipeline report at %s: %s", target_report, exc)
        return

    LOGGER.info("Enhanced pipeline report written to %s", target_report)

    style_block = _extract_style_block(report_html)
    highlights_only = _build_analysis_highlights_section(
        output_dir=output_dir_path,
        de_dir=Path(de_dir),
        plots_dir=Path(plots_dir),
        cluster_sizes=cluster_sizes,
        annotation_path=annotation_path,
        summary_path=summary_path,
        include_style=False,
    )

    standalone_html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8" />',
        '<title>Analysis Highlights</title>',
    ]
    if style_block:
        standalone_html.append(style_block)
    standalone_html.append(HIGHLIGHTS_STYLE)
    standalone_html.extend([
        "</head>",
        "<body>",
        '<div class="container">',
        highlights_only,
        "</div>",
        "</body>",
        "</html>",
    ])

    standalone_path = output_dir_path / "analysis_highlights.html"
    try:
        standalone_path.write_text("\n".join(standalone_html), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Unable to write standalone highlights report at %s: %s", standalone_path, exc)
        return

    LOGGER.info("Standalone highlights report written to %s", standalone_path)

def run_analysis(config, args):
    """
    CODEX/PhenoCycler downstream analysis pipeline.

    Args:
        config (dict): Configuration parameters loaded from a YAML file.
        args (Namespace): Command-line arguments parsed by argparse.
    """
    logging.info(
        "----- Running analysis pipeline with provided configuration and arguments."
    )

    # Extract relevant parameters from config and args
    data_dir = os.path.join(
        args.data_dir, 
        config.get("analysis", {}).get("data_dir", "")
        )
    logging.info(f"Resolved data_dir: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    patch_index = config.get("analysis", {}).get("patch_index", args.patch_index)
    skip_viz = config.get("analysis", {}).get("skip_viz", args.skip_viz)
    output_dir = config.get("analysis", {}).get("output_dir", args.output_dir)
    clustering_resolution = config.get("analysis", {}).get(
        "clustering_resolution", args.clustering_resolution
    )
    norm_method = config.get("analysis", {}).get("norm_method", "log1p")

    # GPU acceleration settings
    use_gpu = config.get("analysis", {}).get("use_gpu", False)
    gpu_id = config.get("analysis", {}).get("gpu_id", 0)

    # Create output directory
    output_dir = args.output_dir
    logging.info(f"Resolved output_dir: {output_dir}")
    output_dir = ensure_dir(output_dir)
    copied_config_path = os.path.join(args.output_dir, "copied_config.yaml")
    shutil.copy(args.config_file, copied_config_path)    
    
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))
    marker_plots_dir = ensure_dir(os.path.join(plots_dir, "markers"))
    metric_plots_dir = ensure_dir(os.path.join(plots_dir, "metrics"))
    spatial_plots_dir = ensure_dir(os.path.join(plots_dir, "spatial"))
    umap_plots_dir = ensure_dir(os.path.join(plots_dir, "umaps"))
    de_dir = ensure_dir(os.path.join(output_dir, "differential_expression"))

    logging.info(f"Output will be saved to: {output_dir}")

    # ================= 1. DATA LOADING =================
    meta_df = load_metadata(data_dir, patch_index)
    exp_df = load_expression(data_dir, patch_index)

    # Move DAPI from expression to metadata if present
    meta_df, exp_df = process_dapi(meta_df, exp_df)

    logging.info(
        f"Loaded {exp_df.shape[0]} cells and {exp_df.shape[1]} markers from patch {patch_index}."
    )
    logging.info(f"meta_df: {meta_df}")
    logging.info(f"exp_df: {exp_df}")

    # ================= 2. EXPLORATORY VISUALIZATION =================
    if not skip_viz:
        logging.info("Generating exploratory visualizations...")

        # Plot marker distributions
        logging.info("Plotting marker distributions...")
        marker_df = exp_df.iloc[:, 1:]  # Skips the first column if it's cell IDs
        plot_marker_distributions(
            marker_df,
            log_transform=False,
            output_dir=marker_plots_dir,
            plot_type="both",
        )
        plot_marker_distributions(
            marker_df, log_transform=True, output_dir=marker_plots_dir, plot_type="both"
        )

        # Plot cell metric distributions
        logging.info("Plotting cell metric distributions...")
        plot_cell_metrics(meta_df, output_dir=metric_plots_dir)

    # ================= 3. DATA NORMALIZATION =================
    logging.info(f"Normalizing data with {norm_method}...")
    data_df = prepare_data_for_analysis(exp_df, norm=norm_method)
    logging.info(f"data_df: {data_df}")
    
    # ================= 3.5. PREPARE LOG1P-TRANSFORMED RAW DATA =================
    logging.info("Preparing log1p-transformed raw data for calculating raw means...")
    log1p_data_df = log1p_transform(exp_df)
    logging.info(f"log1p_data_df: {log1p_data_df}")

    # ================= 4. CLUSTERING =================
    logging.info(f"Running clustering analysis (use_gpu={use_gpu})...")
    adata = create_anndata(data_df, meta_df)
    adata = run_clustering(
        adata,
        resolution=clustering_resolution,
        random_state=42,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        fallback_to_cpu=True,
    )

    # ================= 4. CLUSTER LEVEL STATISTICS =================
    logging.info("Calculating cluster-level statistics...")

    # ================= 5. DIFFERENTIAL EXPRESSION ANALYSIS =================
    logging.info("Performing differential expression analysis...")
    cluster_labels = adata.obs["leiden"]
    logging.info(f"Cluster labels: {cluster_labels}")
    # Pass both normalized and log1p-transformed data to DE analysis
    de_results = one_vs_rest_wilcoxon(data_df, cluster_labels, log1p_data_df)

    # Export DE results
    export_de_results(de_results, de_dir)

    # Prepare payload for optional LLM annotation and persist for transparency
    cluster_sizes = adata.obs["leiden"].value_counts()
    cluster_payload = {}
    for cluster_key, df_res in de_results.items():
        cluster_id = str(cluster_key)
        df_sorted = df_res.copy()
        df_sorted["abs_log2_fc"] = df_sorted["log2_fold_change"].abs()
        df_sorted = df_sorted.sort_values("abs_log2_fc", ascending=False)

        top_entries = []
        for _, row in df_sorted.head(5).iterrows():
            entry = {
                "marker": str(row["feature"]),
                "log2_fold_change": None if pd.isna(row["log2_fold_change"]) else float(row["log2_fold_change"]),
                "fold_change": None if pd.isna(row["fold_change"]) else float(row["fold_change"]),
                "p_value_corrected": None if pd.isna(row["p_value_corrected"]) else float(row["p_value_corrected"]),
                "mean_cluster_norm": None if pd.isna(row["mean_cluster_norm"]) else float(row["mean_cluster_norm"]),
                "mean_rest_norm": None if pd.isna(row["mean_rest_norm"]) else float(row["mean_rest_norm"]),
            }
            top_entries.append(entry)

        cluster_payload[cluster_id] = {
            "cluster_size": int(
                cluster_sizes.get(cluster_id, cluster_sizes.get(int(cluster_key), 0))
            )
            if not cluster_sizes.empty
            else 0,
            "top_markers": top_entries,
        }

    cluster_payload_path = os.path.join(de_dir, "llm_cluster_payload.json")
    with open(cluster_payload_path, "w", encoding="utf-8") as handle:
        json.dump(cluster_payload, handle, indent=2)
    logging.info("Saved cluster payload for annotation: %s", cluster_payload_path)

    # Build and save both fold change matrices
    logging.info("Building fold change matrices...")
    lfc_matrix = build_fold_change_matrix(de_results, use_log=True, use_raw=False)
    fc_matrix = build_fold_change_matrix(de_results, use_log=False, use_raw=False)
    lfc_raw_matrix = build_fold_change_matrix(de_results, use_log=True, use_raw=True)
    fc_raw_matrix = build_fold_change_matrix(de_results, use_log=False, use_raw=True)
    
    # Build top 5 markers matrices
    logging.info("Building top 5 markers fold change matrices...")
    lfc_matrix_top5 = build_fold_change_matrix(de_results, use_log=True, use_raw=False, top_n_markers=5)
    fc_matrix_top5 = build_fold_change_matrix(de_results, use_log=False, use_raw=False, top_n_markers=5)
    lfc_raw_matrix_top5 = build_fold_change_matrix(de_results, use_log=True, use_raw=True, top_n_markers=5)
    fc_raw_matrix_top5 = build_fold_change_matrix(de_results, use_log=False, use_raw=True, top_n_markers=5)
    
    # Save all matrices
    lfc_matrix.to_csv(os.path.join(de_dir, "log_fold_change_matrix_norm.csv"))
    fc_matrix.to_csv(os.path.join(de_dir, "fold_change_matrix_norm.csv"))
    lfc_raw_matrix.to_csv(os.path.join(de_dir, "log_fold_change_matrix_raw.csv"))
    fc_raw_matrix.to_csv(os.path.join(de_dir, "fold_change_matrix_raw.csv"))
    
    # Save top 5 matrices
    lfc_matrix_top5.to_csv(os.path.join(de_dir, "log_fold_change_matrix_norm_top5.csv"))
    fc_matrix_top5.to_csv(os.path.join(de_dir, "fold_change_matrix_norm_top5.csv"))
    lfc_raw_matrix_top5.to_csv(os.path.join(de_dir, "log_fold_change_matrix_raw_top5.csv"))
    fc_raw_matrix_top5.to_csv(os.path.join(de_dir, "fold_change_matrix_raw_top5.csv"))
    
    # save log1p_data_df to csv
    log1p_data_df.to_csv(os.path.join(de_dir, "log1p_data_df.csv"))

    # ================= 6. HEATMAP VISUALIZATION =================
    if not skip_viz:
        logging.info("Generating cluster heatmaps...")
        
        # Log2 fold change heatmap
        plot_heatmap(
            lfc_matrix,
            title="Cluster vs. Rest: log2 Fold Change",
            center=0,
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap_log2fc.png"),
        )
        
        # Regular fold change heatmap
        plot_heatmap(
            fc_matrix,
            title="Cluster vs. Rest: Fold Change",
            center=1,  # Center at 1 for fold change (no change)
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap_fc.png"),
        )

        # Log2 fold change heatmap
        plot_heatmap(
            lfc_raw_matrix,
            title="Cluster vs. Rest: log2 Fold Change (raw)",
            center=0,
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap_log2fc_raw.png"),
        )

        # Regular fold change heatmap
        plot_heatmap(
            fc_raw_matrix,
            title="Cluster vs. Rest: Fold Change (raw)",
            center=1,
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap_fc_raw.png"),
        )
        
        # Top 5 markers heatmaps
        logging.info("Generating top 5 markers heatmaps...")
        
        plot_heatmap(
            lfc_matrix_top5,
            title="Cluster vs. Rest: log2 Fold Change (Top 5 Markers)",
            center=0,
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap_log2fc_top5.png"),
        )
        
        plot_heatmap(
            lfc_raw_matrix_top5,
            title="Cluster vs. Rest: log2 Fold Change Raw (Top 5 Markers)",
            center=0,
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap_log2fc_raw_top5.png"),
        )

    # ================= 7. OPTIONAL LLM ANNOTATION =================
    annotation_output_path: Optional[Path] = None
    summary_output_path: Optional[Path] = None
    if getattr(args, "annotate_cell_types", False):
        prior_path = getattr(args, "llm_prior_path", None)
        if not prior_path:
            logging.warning("LLM annotation enabled but no prior knowledge path provided; skipping." )
        else:
            prior_data = load_json_file(prior_path)
            if not prior_data:
                logging.warning("Unable to load prior knowledge from %s; skipping LLM annotation.", prior_path)
            else:
                tissue_descriptor = getattr(args, "tissue_descriptor", DEFAULT_TISSUE_DESCRIPTOR)
                try:
                    annotation_text = annotate_clusters(
                        prior_data,
                        cluster_payload,
                        model=getattr(args, "llm_model", DEFAULT_MODEL),
                        system_prompt=getattr(args, "llm_system_prompt", None) or DEFAULT_SYSTEM_PROMPT,
                        temperature=float(getattr(args, "llm_temperature", 0.1)),
                        max_completion_tokens=int(getattr(args, "llm_max_tokens", 4000)),
                        tissue_descriptor=tissue_descriptor,
                    )
                except Exception as exc:
                    logging.error("LLM annotation failed: %s", exc)
                else:
                    if annotation_text:
                        annotation_filename = getattr(args, "llm_output_file", None) or "llm_annotation.txt"
                        annotation_path = os.path.join(de_dir, annotation_filename)
                        save_results(annotation_text, annotation_path)
                        annotation_output_path = Path(annotation_path)
                        adata.uns["llm_annotation"] = {
                            "model": getattr(args, "llm_model", DEFAULT_MODEL),
                            "temperature": float(getattr(args, "llm_temperature", 0.1)),
                            "max_tokens": int(getattr(args, "llm_max_tokens", 4000)),
                            "tissue_descriptor": tissue_descriptor,
                            "prior_path": str(prior_path),
                            "cluster_payload_path": cluster_payload_path,
                            "result_path": annotation_path,
                            "result_text": annotation_text,
                        }
                        logging.info("LLM annotation written to %s", annotation_path)

                        if getattr(args, "summarize_annotation", False):
                            try:
                                summary_text = summarize_annotation(
                                    prior_data,
                                    cluster_payload,
                                    annotation_text,
                                    model=getattr(args, "llm_model", DEFAULT_MODEL),
                                    system_prompt=getattr(args, "llm_summary_system_prompt", None)
                                    or DEFAULT_SUMMARY_SYSTEM_PROMPT,
                                    temperature=float(getattr(args, "llm_temperature", 0.1)),
                                    max_completion_tokens=int(getattr(args, "llm_max_tokens", 4000)),
                                    tissue_descriptor=tissue_descriptor,
                                )
                            except Exception as exc:
                                logging.error("LLM annotation summary failed: %s", exc)
                            else:
                                if summary_text:
                                    summary_filename = (
                                        getattr(args, "llm_summary_output_file", None)
                                        or "llm_annotation_summary.txt"
                                    )
                                    summary_path = os.path.join(de_dir, summary_filename)
                                    save_results(summary_text, summary_path)
                                    summary_output_path = Path(summary_path)
                                    adata.uns["llm_annotation"].update(
                                        {
                                            "summary_path": summary_path,
                                            "summary_text": summary_text,
                                        }
                                    )
                                    logging.info(
                                        "LLM annotation summary written to %s", summary_path
                                    )
                                else:
                                    logging.warning(
                                        "LLM returned empty summary response; skipping save."
                                    )
                        else:
                            logging.info(
                                "LLM annotation summary skipped because summarize_annotation is False."
                            )
                    else:
                        logging.warning("LLM returned empty annotation response; skipping save.")

    # ================= 8. SPATIAL VISUALIZATION =================
    cluster_colors = None
    seg_data = load_segmentation_data(
        getattr(args, "segmentation_path", None),
        getattr(args, "segmentation_format", "pickle"),
        patch_index,
    )
    if seg_data is None:
        logging.info("Skipping spatial overlays: segmentation artifact unavailable or disabled.")
    elif skip_viz:
        logging.info("Skipping spatial overlays: skip_viz flag is set.")
    else:
        logging.info("Generating spatial visualizations...")
        cell_mask = seg_data["cell_matched_mask"]
        nuc_mask = seg_data["nucleus_matched_mask"]

        # Plot segmentation masks
        plot_segmentation_masks(
            cell_mask,
            nuc_mask,
            title="Cell and nucleus Segmentation",
            output_path=os.path.join(spatial_plots_dir, "segmentation_masks.png"),
        )

        # Plot clustering on segmentation mask
        cluster_int = adata.obs["leiden"].astype(int).values
        cluster_colors = plot_clustering_on_mask(
            cell_mask,
            cluster_int,
            title=f"Cell clustering (resolution={clustering_resolution})",
            output_path=os.path.join(spatial_plots_dir, "clustering_on_mask.png"),
        )

    # ================= 9. DIMENSIONALITY REDUCTION VISUALIZATION =================
    if not skip_viz:
        logging.info("Plotting UMAP...")
        # Plot UMAP with clusters
        plot_umap(
            adata, color_by=["leiden"], 
            output_dir=umap_plots_dir,
            cluster_colors=cluster_colors
        )

        # Plot UMAP colored by metadata columns if present
        for column in ["cell_entropy", "area", "DAPI"]:
            if column in meta_df.columns:
                plot_umap(adata, color_by=[column], output_dir=umap_plots_dir)

    # ================= 10. SAVE RESULTS =================
    adata.write(
        os.path.join(output_dir, "codex_analysis.h5ad"),
        compression="gzip",
    )

    _maybe_generate_pipeline_report(
        args=args,
        output_dir=output_dir,
        de_dir=de_dir,
        plots_dir=plots_dir,
        cluster_sizes=cluster_sizes,
        annotation_path=annotation_output_path,
        summary_path=summary_output_path,
    )

    logging.info("Analysis complete. Results saved to: %s", output_dir)
