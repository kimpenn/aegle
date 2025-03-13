"""
Differential expression analysis functions for CODEX data.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import numpy as np

def one_vs_rest_wilcoxon(
    df: pd.DataFrame,
    cluster_series: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """
    Performs one-vs-rest Wilcoxon rank-sum tests for each cluster.

    Args:
        df: (n_cells, n_markers) DataFrame of intensities (already transformed if needed)
        cluster_series: Series of cluster labels for each row in df
    Returns:
        Dictionary of cluster -> DataFrame with columns:
            [feature, p_value, p_value_corrected, log2_fold_change]
    """
    logging.info("Starting differential expression analysis...")

    # Debugging index mismatch
    logging.info(f"df index: {df.index}")
    logging.info(f"cluster_series index: {cluster_series.index}")

    logging.info(f"df: {df}")
    logging.info(f"cluster_series: {cluster_series}")

    if not df.index.equals(cluster_series.index):
        logging.warning(
            "Index mismatch detected between df and cluster_series. Reindexing cluster_series."
        )
        try:
            cluster_series.index = cluster_series.index.astype(
                int
            )  # Convert index to integer
        except ValueError as e:
            logging.error(f"Error converting cluster_series index to int: {e}")

        cluster_series = cluster_series.reindex(df.index)

    logging.info(f"updated cluster_series: {cluster_series}")

    features = df.columns
    unique_clusters = sorted(cluster_series.unique())
    logging.info(f"Unique clusters: {unique_clusters}")

    # Check for NaN values in cluster_series
    if cluster_series.isna().any():
        logging.warning("NaN values detected in cluster_series. These will be ignored.")
        cluster_series = cluster_series.dropna()

    unique_clusters = sorted(cluster_series.unique())
    logging.info(f"Unique clusters after dropping NaNs: {unique_clusters}")

    results = {}

    for cluster in unique_clusters:
        logging.info(f"Processing cluster: {cluster}")

        cluster_mask = cluster_series == cluster
        logging.info(f"cluster mask index: {cluster_mask.index}")

        cluster_samples = df.loc[cluster_mask]
        rest_samples = df.loc[~cluster_mask]

        p_values = []
        fc_list = []
        log2_fc_list = []
        mean_cluster_list = []
        mean_rest_list = []
        A_value_list = []
        
        for feature in features:
            # Wilcoxon rank-sum
            stat, p = ranksums(cluster_samples[feature], rest_samples[feature])
            p_values.append(p)

            # Compute fold change
            mean_clust = cluster_samples[feature].mean()
            mean_rest = rest_samples[feature].mean()
            
            # Calculate FC as effect size 
            # TODO: Consider other effect size succh as Cliff’s Delta or AUC
            
            # Fold Change (log2( (mean_clust + ε) / (mean_rest + ε) ))
            fc_value = (
                (mean_clust + 1e-9) / (mean_rest + 1e-9)
                if mean_rest > 0
                else np.nan
            )
            log2_fc_value = np.log2(fc_value) if fc_value > 0 else np.nan
            # For MA-plot: A = 0.5 * (log2(mean_clust + 1) + log2(mean_rest + 1))
            A_value = 0.5 * (
                np.log2(mean_clust + 1e-9) + np.log2(mean_rest + 1e-9)
            )


            fc_list.append(fc_value)
            log2_fc_list.append(log2_fc_value)
            mean_cluster_list.append(mean_clust)
            mean_rest_list.append(mean_rest)
            A_value_list.append(A_value)
            logging.info(f"Feature: {feature}, p-value: {p}, log2_fc: {log2_fc_value}, fc: {fc_value}")
        
        corrected_p = multipletests(p_values, method="fdr_bh")[1]
        cluster_res = pd.DataFrame(
            {
                "feature": features,
                "p_value": p_values,
                "p_value_corrected": corrected_p,
                "fold_change": fc_list,
                "log2_fold_change": log2_fc_list,
                "mean_cluster": mean_cluster_list,
                "mean_rest": mean_rest_list,
                "A_value": A_value_list,
            }
        )
        # Filter by significance
        # cluster_res = cluster_res[cluster_res["p_value_corrected"] < alpha]
        # Sort by fold change
        # cluster_res = cluster_res.sort_values("fold_change", ascending=False)
        # Add cluster label
        results[cluster] = cluster_res

    logging.info("Differential expression analysis completed.")
    
    return results


def build_fold_change_matrix(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a matrix of fold changes (either log2 or normal) from differential expression results.

    Args:
        results: Dictionary mapping cluster labels to differential expression results

    Returns:
        DataFrame with clusters as rows and features as columns, containing fold changes.
    """
    if not results:
        logging.error("Empty results dictionary. Returning empty DataFrame.")
        return pd.DataFrame()

    # Determine if the results contain log2 fold change or regular fold change
    first_cluster = next(iter(results))
    if "log2_fold_change" in results[first_cluster].columns:
        fc_col = "log2_fold_change"
    elif "fold_change" in results[first_cluster].columns:
        fc_col = "fold_change"
    else:
        logging.error(
            "No valid fold change column found in results. Returning empty DataFrame."
        )
        return pd.DataFrame()

    features = results[first_cluster]["feature"].values
    unique_clusters = sorted(results.keys())

    # Initialize matrix
    fc_matrix = pd.DataFrame(index=unique_clusters, columns=features, dtype=float)

    # Fill matrix with fold changes
    for cluster, df_res in results.items():
        for _, row in df_res.iterrows():
            feat = row["feature"]
            fc_matrix.loc[cluster, feat] = row[fc_col]

    # Replace any NaNs or inf
    fc_matrix = fc_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

    return fc_matrix


def build_log_fold_change_matrix(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a matrix of log fold changes from differential expression results.

    Args:
        results: Dictionary mapping cluster labels to differential expression results

    Returns:
        DataFrame with clusters as rows and features as columns, containing log fold changes
    """
    # Extract all features and clusters
    first_cluster = next(iter(results))
    features = results[first_cluster]["feature"].values
    unique_clusters = sorted(results.keys())

    # Initialize matrix
    lfc_matrix = pd.DataFrame(index=unique_clusters, columns=features, dtype=float)

    # Fill matrix with log fold changes
    for cluster, df_res in results.items():
        for _, row in df_res.iterrows():
            feat = row["feature"]
            lfc_matrix.loc[cluster, feat] = row["log2_fold_change"]

    # Replace any NaNs or inf
    lfc_matrix = lfc_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

    return lfc_matrix


def export_de_results(results: Dict[str, pd.DataFrame], output_dir: str, alpha=0.05, lfc_thresh=0.75) -> None:
    """
    Export differential expression results to CSV files.

    Args:
        results: Dictionary mapping cluster labels to differential expression results
        output_dir: Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "top_10_genes_summary.json")
    summary_data = {}

    for cluster, df_res_full in results.items():
        # 1) Plot before filtering so we show all points in the plots
        # --- Plot Volcano: no labels ---
        plot_volcano(
            df_res_full,
            cluster_label=cluster,
            output_dir=output_dir,
            alpha=alpha,
            lfc_thresh=lfc_thresh,
            label_points=False,
            suffix="_nolabels"
        )
        # --- Plot Volcano: with labels ---
        plot_volcano(
            df_res_full,
            cluster_label=cluster,
            output_dir=output_dir,
            alpha=alpha,
            lfc_thresh=lfc_thresh,
            label_points=True,
            suffix="_labeled"
        )

        # --- Plot MA: no labels ---
        plot_ma(
            df_res_full,
            cluster_label=cluster,
            output_dir=output_dir,
            alpha=alpha,
            lfc_thresh=lfc_thresh,
            label_points=False,
            suffix="_nolabels"
        )
        # --- Plot MA: with labels ---
        plot_ma(
            df_res_full,
            cluster_label=cluster,
            output_dir=output_dir,
            alpha=alpha,
            lfc_thresh=lfc_thresh,
            label_points=True,
            suffix="_labeled"
        )
        
        # 2) Create a filtered version for saving. (e.g., p_value_corrected < alpha)
        df_res = df_res_full[df_res_full["p_value_corrected"] < alpha].copy()

        # 3) Sort by fold_change descending, etc.
        df_res.sort_values(by="fold_change", ascending=False, inplace=True)

        # 4) Save the full DE list for each cluster (filtered by p-value, but not by fold change)
        output_path = os.path.join(output_dir, f"de_markers_cluster_{cluster}.csv")
        df_res.to_csv(output_path, index=False)
        logging.info(f"Saved DE results (filtered) for cluster {cluster} to {output_path}")
        
        # 5) Optionally, split into positive/negative fold_change
        df_res_pos = df_res[df_res["fold_change"] > 1]  # e.g. fold change > 1
        df_res_neg = df_res[df_res["fold_change"] < 1]  # e.g. fold change < 1

        df_res_pos.to_csv(os.path.join(output_dir, f"de_markers_cluster_{cluster}_pos.csv"), index=False)
        df_res_neg.to_csv(os.path.join(output_dir, f"de_markers_cluster_{cluster}_neg.csv"), index=False)

        # 6) Extract top 10 genes
        top_genes = df_res.iloc[:10]["feature"].tolist() if not df_res.empty else []
        summary_data[cluster] = top_genes

        # # Save the full DE list for each cluster
        # output_path = os.path.join(output_dir, f"de_markers_cluster_{cluster}.csv")
        # df_res.to_csv(output_path, index=False)
        # logging.info(
        #     f"Saved differential expression results for cluster {cluster} to {output_path}"
        # )
        
        # # Save the positive DE results
        # df_res_pos = df_res[df_res["fold_change"] > 0]
        # output_path_pos = os.path.join(output_dir, f"de_markers_cluster_{cluster}_pos.csv")
        # df_res_pos.to_csv(output_path_pos, index=False)
        # logging.info(
        #     f"Saved positive differential expression results for cluster {cluster} to {output_path_pos}"
        # )
        # # Save the negative DE results
        # df_res_neg = df_res[df_res["fold_change"] < 0]
        # output_path_neg = os.path.join(output_dir, f"de_markers_cluster_{cluster}_neg.csv")
        # df_res_neg.to_csv(output_path_neg, index=False)
        # logging.info(
        #     f"Saved negative differential expression results for cluster {cluster} to {output_path_neg}"
        # )
        # # TODO:MA plot
        # # plot_ma(df_res, cluster, output_dir)
        # # TODO: Volcano plot 
        # # plot_volcano(df_res, cluster, output_dir)
        
        # # Extract the top 10 genes (assume the first column contains gene names)
        # top_genes = df_res.iloc[:10, 0].tolist() if not df_res.empty else []
        # summary_data[cluster] = top_genes

        
    # Save the summary as JSON
    with open(summary_file, "w") as json_f:
        json.dump(summary_data, json_f, indent=4)

    logging.info(f"Saved top 10 genes summary to {summary_file}")

def plot_volcano(
    df: pd.DataFrame,
    cluster_label,
    output_dir: str,
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    label_points: bool = False,
    suffix: str = ""    
):
    """
    Volcano plot for DE results, with optional feature labels.

    Args:
        df: DataFrame containing:
            - feature
            - log2_fold_change
            - p_value_corrected
        cluster_label: cluster identifier
        output_dir: path to output directory
        alpha: FDR threshold
        lfc_thresh: log2 fold-change threshold for significance lines
        label_points: whether to label points with their feature names
        suffix: additional string to append to output filename
    """
    # Make sure we don’t get NaNs messing with plotting
    df = df.dropna(subset=["log2_fold_change", "p_value_corrected"]).copy()
    df["neg_log10_p"] = -np.log10(df["p_value_corrected"] + 1e-300)

    # Determine significance: p < alpha and abs(log2FC) >= lfc_thresh
    df["significant"] = (
        (df["p_value_corrected"] < alpha) &
        (df["log2_fold_change"].abs() >= lfc_thresh)
    )

    fig, ax = plt.subplots(figsize=(6,5))

    # Plot non-significant points in one color
    nonsig = df[~df["significant"]]
    ax.scatter(
        nonsig["log2_fold_change"],
        nonsig["neg_log10_p"],
        c="gray", alpha=0.5, s=20, label="Not significant"
    )

    # Plot significant points in another color
    sig = df[df["significant"]]
    ax.scatter(
        sig["log2_fold_change"],
        sig["neg_log10_p"],
        c="red", alpha=0.7, s=20, label="Significant"
    )

    # Optionally add text labels
    if label_points:
        for _, row in df.iterrows():
            ax.text(
                row["log2_fold_change"],
                row["neg_log10_p"],
                row["feature"],
                fontsize=8,
                ha="center",
                va="bottom"
            )
            
    # Add threshold lines (vertical for fold-change, horizontal for alpha)
    ax.axvline(x=-lfc_thresh, color='blue', linestyle='--', linewidth=1)
    ax.axvline(x=lfc_thresh, color='blue', linestyle='--', linewidth=1)
    ax.axhline(y=-np.log10(alpha), color='blue', linestyle='--', linewidth=1)

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(adj. p-value)")
    ax.set_title(f"Volcano Plot (Cluster {cluster_label})")
    ax.legend()
    
    plt.tight_layout()
    
    filename = f"volcano_cluster_{cluster_label}{suffix}.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved volcano plot for cluster {cluster_label} to {plot_path}")
    
def plot_ma(
    df: pd.DataFrame,
    cluster_label,
    output_dir: str,
    alpha: float = 0.05,
    lfc_thresh: float = 1.0,
    label_points: bool = False,
    suffix: str = ""    
):
    """
    MA plot for DE results, with optional feature labels.

    Args:
        df: DataFrame containing:
            - feature
            - A_value (average abundance)
            - log2_fold_change
            - p_value_corrected
        cluster_label: cluster identifier
        output_dir: path to output directory
        alpha: FDR threshold
        lfc_thresh: log2 fold-change threshold for significance lines
        label_points: whether to label points with their feature names
        suffix: additional string to append to output filename
    """
    # Drop rows that lack needed columns
    df = df.dropna(subset=["A_value", "log2_fold_change", "p_value_corrected"]).copy()
    
    df["significant"] = (
        (df["p_value_corrected"] < alpha) &
        (df["log2_fold_change"].abs() >= lfc_thresh)
    )

    fig, ax = plt.subplots(figsize=(6,5))

    nonsig = df[~df["significant"]]
    ax.scatter(
        nonsig["A_value"], nonsig["log2_fold_change"],
        c="gray", alpha=0.5, s=20, label="Not significant"
    )

    sig = df[df["significant"]]
    ax.scatter(
        sig["A_value"], sig["log2_fold_change"],
        c="red", alpha=0.7, s=20, label="Significant"
    )

    # Optionally add text labels
    if label_points:
        for _, row in df.iterrows():
            ax.text(
                row["A_value"],
                row["log2_fold_change"],
                row["feature"],
                fontsize=8,
                ha="center",
                va="bottom"
            )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=-lfc_thresh, color='blue', linestyle='--', linewidth=1)
    ax.axhline(y=lfc_thresh, color='blue', linestyle='--', linewidth=1)

    ax.set_xlabel("A (average abundance)")
    ax.set_ylabel("M (log2 Fold Change)")
    ax.set_title(f"MA Plot (Cluster {cluster_label})")
    ax.legend()
    
    plt.tight_layout()

    # Save the plot, with suffix to differentiate
    filename = f"MA_cluster_{cluster_label}{suffix}.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved MA plot for cluster {cluster_label} to {plot_path}")