# /workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle_analysis/analysis.py

import logging
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import shutil

# Import modules from the package
from aegle_analysis.data import (
    load_metadata,
    load_expression,
    process_dapi,
    load_segmentation_data,
    prepare_data_for_analysis,
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

    # ================= 4. CLUSTERING =================
    logging.info("Running clustering analysis...")
    adata = create_anndata(data_df, meta_df)
    adata = run_clustering(adata, resolution=clustering_resolution, random_state=42)

    # ================= 4. CLUSTER LEVEL STATISTICS =================
    # Save anndata object
    adata.write(os.path.join(output_dir, "codex_analysis.h5ad"))
    logging.info("Anndata object saved.")
    logging.info("Calculating cluster-level statistics...")

    # ================= 5. DIFFERENTIAL EXPRESSION ANALYSIS =================
    logging.info("Performing differential expression analysis...")
    cluster_labels = adata.obs["leiden"]
    logging.info(f"Cluster labels: {cluster_labels}")
    de_results = one_vs_rest_wilcoxon(data_df, cluster_labels)

    # Export DE results
    export_de_results(de_results, de_dir)

    # Build and save log fold change matrix
    lfc_matrix = build_fold_change_matrix(de_results)
    lfc_matrix.to_csv(os.path.join(de_dir, "log_fold_change_matrix.csv"))

    # ================= 6. HEATMAP VISUALIZATION =================
    if not skip_viz:
        logging.info("Generating cluster heatmap...")
        plot_heatmap(
            lfc_matrix,
            title="Cluster vs. Rest: log2 Fold Change",
            center=0,
            cmap="RdBu_r",
            output_path=os.path.join(plots_dir, "cluster_heatmap.png"),
        )

    # ================= 7. SPATIAL VISUALIZATION =================
    seg_data = load_segmentation_data(data_dir, 0)
    if seg_data and not skip_viz:
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

    # ================= 8. DIMENSIONALITY REDUCTION VISUALIZATION =================
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

    # ================= 9. SAVE RESULTS =================
    adata.write(os.path.join(output_dir, "codex_analysis.h5ad"))
    logging.info("Analysis complete. Results saved to: %s", output_dir)
