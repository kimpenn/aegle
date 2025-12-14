"""
Spatial visualization functions for CODEX data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple, List
import anndata
import logging
import scanpy as sc

sc.settings.verbosity = "error"  # Suppress Scanpy warnings


def plot_segmentation_masks(
    cell_mask: np.ndarray,
    nuc_mask: Optional[np.ndarray] = None,
    title: str = "Cell Segmentation",
    figsize: Tuple[int, int] = (10, 10),
    output_path: Optional[str] = None,
) -> None:
    """
    Plot cell and nuclear segmentation masks.

    Args:
        cell_mask: Cell segmentation mask with integer cell IDs
        nuc_mask: Optional nuclear segmentation mask with integer cell IDs
        title: Title for the plot
        figsize: Figure size as (width, height)
        output_path: Path to save the figure (if None, figure is not saved)
    """
    # Squeeze extra dimensions (e.g., (1, H, W) -> (H, W))
    cell_mask = np.squeeze(cell_mask)
    if nuc_mask is not None:
        nuc_mask = np.squeeze(nuc_mask)

    if nuc_mask is not None:
        # Plot both cell and nuclear masks
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot cell mask
        axes[0].imshow(cell_mask > 0, cmap="viridis")
        axes[0].set_title("Cell Boundaries")
        axes[0].axis("off")

        # Plot nuclear mask
        axes[1].imshow(nuc_mask > 0, cmap="plasma")
        axes[1].set_title("Nuclear Boundaries")
        axes[1].axis("off")

        plt.suptitle(title)
    else:
        # Plot only cell mask
        plt.figure(figsize=figsize)
        plt.imshow(cell_mask > 0, cmap="viridis")
        plt.title(title)
        plt.axis("off")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()

def get_cluster_colors(cluster_labels, palette_name="tab20"):
    """
    Generate a consistent color mapping for clusters.
    
    Args:
        cluster_labels: Array of cluster labels
        palette_name: Name of the color palette to use
    
    Returns:
        Dictionary mapping cluster IDs to RGB colors
    """
    unique_labels = np.unique(cluster_labels)
    n_colors = len(unique_labels)
    
    # Get color palette as RGB values
    if palette_name == "tab20" and n_colors > 20:
        # If more than 20 clusters, use a palette with more colors
        palette_name = "tab20" if n_colors <= 20 else "husl"
    
    palette = np.array(sns.color_palette(palette_name, n_colors))
    
    # Create mapping from cluster ID to color
    cluster_to_color = {}
    for i, label in enumerate(unique_labels):
        cluster_to_color[label] = palette[i]
    
    return cluster_to_color

def plot_clustering_on_mask(
    cell_mask: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Segmentation colored by cluster",
    figsize: Tuple[int, int] = (10, 10),
    output_path: Optional[str] = None,
    color_dict: Optional[Dict[int, Tuple[float, float, float]]] = None,
    palette_name: str = "tab20",    
) -> None:
    """
    Visualize clustering results on segmentation mask.

    Args:
        cell_mask: Cell segmentation mask with integer cell IDs
        cluster_labels: Cluster labels for each cell
        title: Title for the plot
        figsize: Figure size as (width, height)
        output_path: Path to save the figure (if None, figure is not saved)
        color_dict: Optional dictionary mapping cluster IDs to RGB colors
        palette_name: Name of the color palette to use if color_dict is None      

    Returns:
        Dictionary mapping cluster IDs to RGB colors          
    """
    logging.info("Starting clustering visualization...")

    # Squeeze extra dimensions (e.g., (1, H, W) -> (H, W))
    cell_mask = np.squeeze(cell_mask)

    # Ensure cell_mask is integer type
    if not np.issubdtype(cell_mask.dtype, np.integer):
        logging.warning(
            f"cell_mask dtype {cell_mask.dtype} is not integer. Converting..."
        )
        cell_mask = cell_mask.astype(int)

    # Get the maximum cell ID in the mask
    max_cell_id = int(cell_mask.max())
    logging.info(f"Max cell ID in mask: {max_cell_id}")

    # Check if the number of cluster labels matches expected IDs
    logging.info(
        f"Expected {max_cell_id} cluster labels, received {len(cluster_labels)}"
    )
    unique_labels = np.unique(cluster_labels)
    logging.info(f"Unique cluster labels: {unique_labels}")
    logging.info(f"Max cluster label: {max(unique_labels)}")
    logging.info(f"Number of cluster labels: {len(unique_labels)}")

    # Get consistent color mapping
    if color_dict is None:
        color_dict = get_cluster_colors(cluster_labels, palette_name)
    
    # Build a map from "cell ID" -> "cluster"
    id_to_cluster = np.zeros(max_cell_id + 1, dtype=int)

    if len(cluster_labels) < max_cell_id:
        logging.error(
            "Mismatch: more cell IDs in the mask than provided cluster labels!"
        )
        return color_dict

    # Assign cluster labels correctly
    id_to_cluster[1 : len(cluster_labels) + 1] = cluster_labels

    # Check for invalid indices in cell_mask
    if (cell_mask < 0).any() or (cell_mask > max_cell_id).any():
        logging.error("Invalid values in cell_mask detected!")
        return color_dict

    # Create a color array for each cell ID
    # Map each cluster ID to its color
    cluster_to_color_array = np.zeros((max(unique_labels) + 1, 3))
    for cluster_id, color in color_dict.items():
        if cluster_id < len(cluster_to_color_array):
            cluster_to_color_array[cluster_id] = color

    # Map each cell ID to its cluster color
    color_array = cluster_to_color_array[id_to_cluster]

    try:
        color_mask = color_array[cell_mask]
    except IndexError as e:
        logging.error(f"IndexError when mapping colors: {e}")
        return color_dict

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(color_mask)
    plt.title(title)
    plt.axis("off")

    # Add legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color_dict[lbl]) for lbl in unique_labels
    ]
    plt.legend(
        legend_patches,
        [f"Cluster {lbl}" for lbl in unique_labels],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    logging.info("Clustering visualization completed.")

    return color_dict


def plot_marker_expression_on_mask(
    cell_mask: np.ndarray,
    expression_values: np.ndarray,
    marker_name: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 10),
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize marker expression on segmentation mask.

    Args:
        cell_mask: Cell segmentation mask with integer cell IDs
        expression_values: Expression values for each cell
        marker_name: Name of the marker
        cmap: Colormap name
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        figsize: Figure size as (width, height)
        output_path: Path to save the figure (if None, figure is not saved)
    """
    # Squeeze extra dimensions (e.g., (1, H, W) -> (H, W))
    cell_mask = np.squeeze(cell_mask)

    # Get the maximum cell ID in the mask
    max_cell_id = int(cell_mask.max())

    # Build a map from "cell ID" -> "expression value"
    id_to_expression = np.zeros(max_cell_id + 1)

    # The naive approach assumes row i => cell ID i+1
    id_to_expression[1 : len(expression_values) + 1] = expression_values

    # Map each cell ID to its expression value
    expression_mask = id_to_expression[cell_mask]

    # Plot
    plt.figure(figsize=figsize)
    im = plt.imshow(expression_mask, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f"{marker_name} Expression")
    plt.axis("off")

    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(f"{marker_name} Expression")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()


def plot_umap(
    adata: anndata.AnnData,
    color_by: List[str] = ["leiden"],
    output_dir: Optional[str] = None,
    cluster_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,    
    cmap: str = "viridis",
    save_format: str = "png",
    dpi: int = 300,
) -> None:
    """
    Plot UMAP embeddings from an AnnData object.

    Args:
        adata: AnnData object with computed UMAP
        color_by: List of variables to color points by
        output_dir: Directory to save plots (if None, plots are not saved)
        cluster_colors: Dictionary mapping cluster IDs to RGB colors for categorical variables        
        cmap: Colormap for continuous variables
        save_format: Format to save plots (png, pdf, svg, etc.)
        dpi: Resolution for rasterized formats
    """
    # Set Scanpy figure parameters
    sc.set_figure_params(figsize=(10, 10), dpi_save=dpi)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sc.settings.figdir = output_dir  # Set Scanpy's figure directory
    
    for color in color_by:
        filename_suffix = f"_{color}.{save_format}" if output_dir else None
        
        # If color is a categorical variable and we have cluster colors
        if (
            color in adata.obs
            and adata.obs[color].dtype.name == "category"
            and cluster_colors is not None
        ):
            # Get the categories
            categories = adata.obs[color].cat.categories
            
            # Check if the categories can be converted to integers
            try:
                # Map category names to integers if possible
                cat_to_int = {cat: int(cat) for cat in categories}
                
                # Create a palette dictionary that scanpy can use
                # Map each category name to its color
                palette = {
                    cat: cluster_colors.get(cat_to_int[cat], (0.7, 0.7, 0.7))
                    for cat in categories
                    if cat_to_int[cat] in cluster_colors
                }
                
                # Use the custom palette in the UMAP plot
                logging.info(f"Using custom color palette for {color}")
                sc.pl.umap(
                    adata,
                    color=color,
                    title=f"UMAP colored by {color}",
                    palette=palette,
                    show=output_dir is None,
                    save=filename_suffix,
                )
                continue
            except (ValueError, TypeError):
                logging.info(f"Could not convert {color} categories to integers")
        
        # Default case: use standard color mapping
        logging.info(f"Using default color mapping for {color}")
        sc.pl.umap(
            adata,
            color=color,
            title=f"UMAP colored by {color}",
            cmap=cmap,
            show=output_dir is None,
            save=filename_suffix,
        )

    # for color in color_by:
    #     filename_suffix = f"_{color}.{save_format}" if output_dir else None
    #     logging.info(
    #         f"Scanpy saving UMAP plot to {output_dir if output_dir else 'figures/'}"
    #     )

    #     sc.pl.umap(
    #         adata,
    #         color=color,
    #         title=f"UMAP colored by {color}",
    #         cmap=cmap,
    #         show=output_dir is None,  # Only show if not saving
    #         save=filename_suffix,  # Provide suffix instead of full path
    #     )

    # # Set figure parameters
    # sc.set_figure_params(figsize=(10, 10))

    # for color in color_by:
    #     # Create file name if saving
    #     filename = None
    #     if output_dir:
    #         os.makedirs(output_dir, exist_ok=True)
    #         filename = os.path.join(output_dir, f"umap_{color}.{save_format}")

    #     logging.info(f"Scanpy saved UMAP plot to {filename}")
    #     # Plot UMAP
    #     sc.pl.umap(
    #         adata,
    #         color=color,
    #         title=f"UMAP colored by {color}",
    #         cmap=cmap,
    #         show=filename is None,  # Only show if not saving
    #         save=filename is not None,  # Save if filename is provided
    #     )

    #     # If saving, handle the file rename since scanpy adds "umap_" prefix
    #     if filename:
    #         # Get the file that scanpy created
    #         scanpy_file = os.path.join(output_dir, f"umap_{color}.{save_format}")
    #         # Rename it to our desired filename
    #         if os.path.exists(scanpy_file):
    #             os.rename(scanpy_file, filename)

# def plot_clustering_on_mask(
#     cell_mask: np.ndarray,
#     cluster_labels: np.ndarray,
#     title: str = "Segmentation colored by cluster",
#     figsize: Tuple[int, int] = (10, 10),
#     output_path: Optional[str] = None,
# ) -> None:
#     """
#     Visualize clustering results on segmentation mask.

#     Args:
#         cell_mask: Cell segmentation mask with integer cell IDs
#         cluster_labels: Cluster labels for each cell
#         title: Title for the plot
#         figsize: Figure size as (width, height)
#         output_path: Path to save the figure (if None, figure is not saved)
#     """
#     # Get the maximum cell ID in the mask
#     max_cell_id = int(cell_mask.max())

#     # Build a map from "cell ID" -> "cluster"
#     id_to_cluster = np.zeros(max_cell_id + 1, dtype=int)

#     # The naive approach assumes row i => cell ID i+1
#     id_to_cluster[1 : len(cluster_labels) + 1] = cluster_labels

#     # Create a color palette
#     unique_labels = np.unique(cluster_labels)
#     palette = np.array(sns.color_palette("tab20", len(unique_labels)))

#     # For quick indexing, cluster i -> color palette[i-1] if cluster is 1-based
#     cluster_to_color = np.zeros((max(unique_labels) + 1, 3))
#     for i, lbl in enumerate(unique_labels):
#         cluster_to_color[lbl] = palette[i]

#     # Map each cell ID to its cluster color
#     color_array = cluster_to_color[id_to_cluster]
#     color_mask = color_array[cell_mask]

#     # Plot
#     plt.figure(figsize=figsize)
#     plt.imshow(color_mask)
#     plt.title(title)
#     plt.axis("off")

#     # Add legend
#     legend_patches = [
#         plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(unique_labels))
#     ]
#     plt.legend(
#         legend_patches,
#         [f"Cluster {lbl}" for lbl in unique_labels],
#         loc="center left",
#         bbox_to_anchor=(1, 0.5),
#     )

#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path, dpi=300, bbox_inches="tight")

#     plt.show()
