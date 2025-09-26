"""
Segmentation visualization module for PhenoCycler pipeline.
Provides various visualization functions for segmentation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
from skimage import morphology, measure
from skimage.segmentation import find_boundaries
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import os
from matplotlib.patches import Patch


logger = logging.getLogger(__name__)


def _count_unique_labels(mask: Optional[np.ndarray]) -> int:
    """Count unique non-zero labels in a segmentation mask."""
    if mask is None:
        return 0
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return 0
    labels = np.unique(mask_array)
    if labels.size <= 1:
        return 0
    return int(np.count_nonzero(labels))


def _normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """Convert image to float RGB and normalize to [0, 1] for visualization."""
    if image.ndim == 2:
        display_img = np.stack([image] * 3, axis=-1)
    else:
        display_img = image.copy()

    display_img = display_img.astype(np.float32)
    img_min = display_img.min()
    img_max = display_img.max()
    if img_max > img_min:
        display_img = (display_img - img_min) / (img_max - img_min)
    else:
        display_img = np.zeros_like(display_img)
    return display_img


def _build_mask_overlays(
    mask: Optional[np.ndarray],
    cmap_name: str,
    alpha: float,
    boundary_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    boundary_mask: Optional[np.ndarray] = None,
    require_centroids: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[colors.Colormap], Optional[colors.Normalize], List]:
    if mask is None:
        return None, None, None, None, []

    if mask.size == 0:
        empty_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        return empty_overlay, empty_overlay.copy(), plt.cm.get_cmap(cmap_name), None, []

    flat_mask = mask.ravel()
    counts = np.bincount(flat_mask)
    if counts.shape[0] <= 1:
        empty_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        return empty_overlay, empty_overlay.copy(), plt.cm.get_cmap(cmap_name), None, []

    labels = np.nonzero(counts)[0]
    labels = labels[labels > 0]
    if labels.size == 0:
        empty_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        return empty_overlay, empty_overlay.copy(), plt.cm.get_cmap(cmap_name), None, []

    areas = counts[labels].astype(np.float32)
    vmin = float(areas.min())
    vmax = float(areas.max())
    if vmin == vmax:
        vmin = max(vmin - 0.5, 0.0)
        vmax = vmax + 0.5

    cmap = plt.cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    color_lut = np.zeros((counts.shape[0], 4), dtype=np.float32)
    mapped_rgba = cmap(norm(areas))
    color_lut[labels, :3] = mapped_rgba[:, :3]
    color_lut[labels, 3] = alpha
    colored_mask = color_lut[flat_mask].reshape((*mask.shape, 4))

    if boundary_mask is None:
        boundary_mask = find_boundaries(mask, mode="outer")
    boundary_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    boundary_overlay[boundary_mask > 0] = boundary_color

    props: List = []
    if require_centroids:
        from skimage import measure

        props = measure.regionprops(mask)

    return colored_mask, boundary_overlay, cmap, norm, props


def create_segmentation_overlay(
    image: np.ndarray,
    nucleus_mask: Optional[np.ndarray],
    cell_mask: Optional[np.ndarray],
    matched_pairs: Optional[Dict[int, int]] = None,
    alpha: float = 0.5,
    show_ids: bool = False,
    figsize: Tuple[int, int] = (12, 12),
    reference_nucleus_mask: Optional[np.ndarray] = None,
    reference_cell_mask: Optional[np.ndarray] = None,
    show_cell_overlay: bool = True,
    show_nucleus_overlay: bool = True,
    custom_title: Optional[str] = None,
    show_reference_highlights: bool = True,
    fill_cell_mask: bool = True,
    fill_nucleus_mask: bool = True,
    cell_boundary_mask: Optional[np.ndarray] = None,
    nucleus_boundary_mask: Optional[np.ndarray] = None,
    cell_count: Optional[int] = None,
    nucleus_count: Optional[int] = None,
) -> plt.Figure:
    """Create a styled overlay showing cell and nucleus masks on the original image."""

    fig, ax = plt.subplots(figsize=figsize)

    display_img = _normalize_image_for_display(image)
    ax.imshow(display_img, cmap='gray' if image.ndim == 2 else None)

    mask_alpha = max(min(alpha, 0.85), 0.2)
    cell_overlay = cell_boundary = cell_cmap = cell_norm = None
    nucleus_overlay = nucleus_boundary = nucleus_cmap = nucleus_norm = None
    cell_props: List = []
    nucleus_props: List = []

    if cell_mask is not None:
        (
            cell_overlay,
            cell_boundary,
            cell_cmap,
            cell_norm,
            cell_props,
        ) = _build_mask_overlays(
            cell_mask,
            'viridis',
            max(mask_alpha * 0.9, 0.25),
            boundary_color=(0.0, 0.75, 0.3, 1.0),
            boundary_mask=cell_boundary_mask,
            require_centroids=show_ids,
        )
    if nucleus_mask is not None:
        (
            nucleus_overlay,
            nucleus_boundary,
            nucleus_cmap,
            nucleus_norm,
            nucleus_props,
        ) = _build_mask_overlays(
            nucleus_mask,
            'magma',
            max(mask_alpha, 0.3),
            boundary_color=(1.0, 0.35, 0.35, 1.0),
            boundary_mask=nucleus_boundary_mask,
            require_centroids=show_ids,
        )

    if not show_cell_overlay:
        cell_overlay = None
        cell_boundary = None
        cell_cmap = None
        cell_norm = None
    elif not fill_cell_mask:
        cell_overlay = None
        cell_cmap = None
        cell_norm = None
    if not show_nucleus_overlay:
        nucleus_overlay = None
        nucleus_boundary = None
        nucleus_cmap = None
        nucleus_norm = None
    elif not fill_nucleus_mask:
        nucleus_overlay = None
        nucleus_cmap = None
        nucleus_norm = None

    if cell_overlay is not None:
        ax.imshow(cell_overlay)
    if nucleus_overlay is not None:
        ax.imshow(nucleus_overlay)

    if cell_boundary is not None:
        ax.imshow(cell_boundary)
    if nucleus_boundary is not None:
        ax.imshow(nucleus_boundary)

    if show_ids:
        for prop in cell_props:
            y, x = prop.centroid
            ax.text(
                x,
                y,
                str(prop.label),
                color='white',
                fontsize=7,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
            )
        for prop in nucleus_props:
            y, x = prop.centroid
            ax.text(
                x,
                y,
                str(prop.label),
                color='yellow',
                fontsize=7,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
            )

    unmatched_handles: List[Patch] = []

    if show_reference_highlights and reference_nucleus_mask is not None:
        ref_mask = reference_nucleus_mask
        unmatched_pixels = ref_mask > 0
        if nucleus_mask is not None:
            unmatched_pixels &= nucleus_mask == 0
        highlight = np.zeros((*ref_mask.shape, 4), dtype=np.float32)
        highlight[unmatched_pixels] = [1.0, 1.0, 0.0, 0.45]
        if np.any(highlight[..., 3] > 0):
            ax.imshow(highlight)
            unmatched_labels = np.unique(ref_mask[unmatched_pixels])
            unmatched_count = len(unmatched_labels[unmatched_labels > 0])
            if unmatched_count:
                unmatched_handles.append(
                    Patch(facecolor=(1.0, 1.0, 0.0, 0.45), edgecolor='white', label=f'Unmatched nuclei: {unmatched_count}')
                )

    if show_reference_highlights and reference_cell_mask is not None:
        ref_mask = reference_cell_mask
        unmatched_pixels = ref_mask > 0
        if cell_mask is not None:
            unmatched_pixels &= cell_mask == 0
        highlight = np.zeros((*ref_mask.shape, 4), dtype=np.float32)
        highlight[unmatched_pixels] = [0.0, 0.8, 1.0, 0.45]
        if np.any(highlight[..., 3] > 0):
            ax.imshow(highlight)
            unmatched_labels = np.unique(ref_mask[unmatched_pixels])
            unmatched_count = len(unmatched_labels[unmatched_labels > 0])
            if unmatched_count:
                unmatched_handles.append(
                    Patch(facecolor=(0.0, 0.8, 1.0, 0.45), edgecolor='white', label=f'Unmatched cells: {unmatched_count}')
                )

    if matched_pairs:
        logger.debug('matched_pairs argument is deprecated; provide reference masks for highlighting.')

    colorbar_pad = 0.04
    if show_cell_overlay and cell_norm is not None and cell_cmap is not None:
        sm_cell = plt.cm.ScalarMappable(cmap=cell_cmap, norm=cell_norm)
        sm_cell.set_array([])
        cbar_cell = fig.colorbar(sm_cell, ax=ax, fraction=0.04, pad=colorbar_pad)
        cbar_cell.set_label('Cell area (pixels)', rotation=270, labelpad=15)
        colorbar_pad += 0.08
    if show_nucleus_overlay and nucleus_norm is not None and nucleus_cmap is not None:
        sm_nuc = plt.cm.ScalarMappable(cmap=nucleus_cmap, norm=nucleus_norm)
        sm_nuc.set_array([])
        cbar_nuc = fig.colorbar(sm_nuc, ax=ax, fraction=0.04, pad=colorbar_pad)
        cbar_nuc.set_label('Nucleus area (pixels)', rotation=270, labelpad=15)

    if unmatched_handles:
        legend = ax.legend(handles=unmatched_handles, loc='upper right', fontsize=8, frameon=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#cccccc')

    title_lines = [custom_title or 'Segmentation Overlay']
    if show_cell_overlay:
        if cell_count is not None:
            title_lines.append(f'Cells detected (Green): {cell_count}')
        elif cell_props:
            title_lines.append(f'Cells detected (Green): {len(cell_props)}')
    if show_nucleus_overlay:
        if nucleus_count is not None:
            title_lines.append(f'Nuclei detected (Red): {nucleus_count}')
        elif nucleus_props:
            title_lines.append(f'Nuclei detected (Red): {len(nucleus_props)}')
    ax.set_title('\n'.join(title_lines))
    ax.axis('off')

    return fig


def create_quality_heatmaps(
    segmentation_results: List[Dict],
    patches_metadata: pd.DataFrame,
    output_dir: str,
    metrics: List[str] = ['cell_density', 'mean_cell_area', 'matched_fraction']
) -> Dict[str, plt.Figure]:
    """
    Create spatial heatmaps of segmentation quality metrics.
    
    Args:
        segmentation_results: List of segmentation results per patch
        patches_metadata: DataFrame with patch spatial information
        output_dir: Directory to save figures
        metrics: List of metrics to visualize
        
    Returns:
        Dict of metric names to figures
    """
    figures = {}
    
    # Calculate metrics for each patch
    metric_data = {}
    for metric in metrics:
        metric_data[metric] = []
    
    for i, seg_result in enumerate(segmentation_results):
        if seg_result is None:
            for metric in metrics:
                metric_data[metric].append(np.nan)
            continue
            
        # Calculate cell density
        if 'cell_density' in metrics:
            cell_mask = seg_result.get('cell_matched_mask', seg_result.get('cell'))
            n_cells = len(np.unique(cell_mask)) - 1  # Subtract background
            area = cell_mask.shape[0] * cell_mask.shape[1]
            density = n_cells / area * 1000000  # cells per mm²
            metric_data['cell_density'].append(density)
        
        # Calculate mean cell area
        if 'mean_cell_area' in metrics:
            cell_mask = seg_result.get('cell_matched_mask', seg_result.get('cell'))
            areas = [r.area for r in measure.regionprops(cell_mask)]
            mean_area = np.mean(areas) if areas else 0
            metric_data['mean_cell_area'].append(mean_area)
        
        # Get matched fraction
        if 'matched_fraction' in metrics:
            matched_frac = seg_result.get('matched_fraction', np.nan)
            metric_data['matched_fraction'].append(matched_frac)
    
    # Create heatmaps
    for metric, values in metric_data.items():
        if not any(~np.isnan(values)):
            continue
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create 2D grid from patch positions
        if 'x_start' in patches_metadata.columns and 'y_start' in patches_metadata.columns:
            # Create scatter plot with color coding
            scatter = ax.scatter(
                patches_metadata['x_start'],
                patches_metadata['y_start'],
                c=values,
                s=200,
                cmap='viridis',
                edgecolors='black',
                linewidth=1
            )
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.invert_yaxis()  # Match image coordinates
        else:
            # Simple bar plot if no spatial info
            ax.bar(range(len(values)), values)
            ax.set_xlabel('Patch Index')
            ax.set_ylabel(metric)
        
        if 'x_start' in patches_metadata.columns:
            fig.colorbar(scatter, ax=ax)
        ax.set_title(f'Spatial Distribution of {metric.replace("_", " ").title()}')
        
        figures[metric] = fig
        
        # Save figure
        fig.savefig(os.path.join(output_dir, f'quality_heatmap_{metric}.png'), 
                   dpi=150, bbox_inches='tight')
    
    return figures


def plot_cell_morphology_stats(
    segmentation_results: List[Dict],
    output_dir: str,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create comprehensive morphology statistics plots.
    
    Args:
        segmentation_results: List of segmentation results
        output_dir: Directory to save figure
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    # Collect all morphology data
    all_areas = []
    all_eccentricities = []
    all_solidities = []
    all_nuc_cell_ratios = []
    
    for seg_result in segmentation_results:
        if seg_result is None:
            continue
            
        cell_mask = seg_result.get('cell_matched_mask', seg_result.get('cell'))
        nucleus_mask = seg_result.get('nucleus_matched_mask', seg_result.get('nucleus'))
        
        # Get matched pairs
        cell_props = {r.label: r for r in measure.regionprops(cell_mask)}
        nuc_props = {r.label: r for r in measure.regionprops(nucleus_mask)}
        
        for cell_id, cell_region in cell_props.items():
            if cell_id == 0:
                continue
                
            all_areas.append(cell_region.area)
            all_eccentricities.append(cell_region.eccentricity)
            all_solidities.append(cell_region.solidity)
            
            # Find corresponding nucleus
            nucleus_overlap = nucleus_mask[cell_mask == cell_id]
            nucleus_ids = np.unique(nucleus_overlap[nucleus_overlap > 0])
            
            if len(nucleus_ids) == 1:
                nuc_id = nucleus_ids[0]
                if nuc_id in nuc_props:
                    ratio = nuc_props[nuc_id].area / cell_region.area
                    all_nuc_cell_ratios.append(ratio)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Cell area distribution
    axes[0, 0].hist(all_areas, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Cell Area (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Cell Area Distribution')
    axes[0, 0].axvline(np.median(all_areas), color='red', linestyle='--', 
                       label=f'Median: {np.median(all_areas):.1f}')
    axes[0, 0].legend()
    
    # Eccentricity distribution
    axes[0, 1].hist(all_eccentricities, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Eccentricity (0=circle, 1=line)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Cell Eccentricity Distribution')
    
    # Solidity distribution
    axes[1, 0].hist(all_solidities, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Solidity (0=irregular, 1=convex)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Cell Solidity Distribution')
    
    # Nucleus/Cell ratio
    axes[1, 1].hist(all_nuc_cell_ratios, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Nucleus/Cell Area Ratio')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Nucleus to Cell Ratio Distribution')
    axes[1, 1].axvline(np.median(all_nuc_cell_ratios), color='red', linestyle='--',
                       label=f'Median: {np.median(all_nuc_cell_ratios):.2f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(output_dir, 'cell_morphology_stats.png'), 
               dpi=150, bbox_inches='tight')
    
    # Log statistics
    logger.info(f"Total cells analyzed: {len(all_areas)}")
    logger.info(f"Mean cell area: {np.mean(all_areas):.1f} ± {np.std(all_areas):.1f} pixels²")
    logger.info(f"Mean nucleus/cell ratio: {np.mean(all_nuc_cell_ratios):.3f} ± {np.std(all_nuc_cell_ratios):.3f}")
    
    return fig


def visualize_segmentation_errors(
    image: np.ndarray,
    segmentation_result: Dict,
    error_types: List[str] = ['oversized', 'undersized', 'high_eccentricity', 'unmatched'],
    thresholds: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Highlight potential segmentation errors.
    
    Args:
        image: Original image
        segmentation_result: Segmentation masks and metadata
        error_types: Types of errors to highlight
        thresholds: Custom thresholds for error detection
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    if thresholds is None:
        thresholds = {
            'oversized': 2000,      # pixels²
            'undersized': 50,       # pixels²
            'high_eccentricity': 0.95,
            'low_solidity': 0.7
        }
    
    cell_mask = segmentation_result.get('cell_matched_mask', segmentation_result.get('cell'))
    nucleus_mask = segmentation_result.get('nucleus_matched_mask', segmentation_result.get('nucleus'))
    
    # Prepare subplots
    n_errors = len(error_types)
    n_cols = min(3, n_errors)
    n_rows = (n_errors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_errors == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, error_type in enumerate(error_types):
        ax = axes[idx]
        
        # Show base image
        ax.imshow(image, cmap='gray')
        
        # Create error mask
        error_mask = np.zeros_like(cell_mask, dtype=bool)
        
        if error_type == 'oversized':
            for region in measure.regionprops(cell_mask):
                if region.area > thresholds['oversized']:
                    error_mask[cell_mask == region.label] = True
                    
        elif error_type == 'undersized':
            for region in measure.regionprops(cell_mask):
                if region.area < thresholds['undersized'] and region.label > 0:
                    error_mask[cell_mask == region.label] = True
                    
        elif error_type == 'high_eccentricity':
            for region in measure.regionprops(cell_mask):
                if region.eccentricity > thresholds['high_eccentricity']:
                    error_mask[cell_mask == region.label] = True
                    
        elif error_type == 'unmatched':
            # Find unmatched cells/nuclei
            matched_cells = set()
            matched_nuclei = set()
            
            for region in measure.regionprops(cell_mask):
                if region.label == 0:
                    continue
                nucleus_overlap = nucleus_mask[cell_mask == region.label]
                nucleus_ids = np.unique(nucleus_overlap[nucleus_overlap > 0])
                if len(nucleus_ids) == 1:
                    matched_cells.add(region.label)
                    matched_nuclei.update(nucleus_ids)
            
            # Highlight unmatched
            for cell_id in np.unique(cell_mask):
                if cell_id > 0 and cell_id not in matched_cells:
                    error_mask[cell_mask == cell_id] = True
        
        # Overlay error mask
        overlay = np.zeros((*image.shape[:2], 4))
        overlay[error_mask] = [1, 0, 0, 0.5]  # Red with transparency
        ax.imshow(overlay)
        
        # Count errors
        n_errors_found = len(np.unique(cell_mask[error_mask])) - (1 if 0 in cell_mask[error_mask] else 0)
        ax.set_title(f'{error_type.replace("_", " ").title()}\n({n_errors_found} cells)')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(error_types), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def create_segmentation_comparison(
    image: np.ndarray,
    segmentation_results: Dict[str, Dict],
    figsize: Tuple[int, int] = (20, 5)
) -> plt.Figure:
    """
    Compare different segmentation results side by side.
    
    Args:
        image: Original image
        segmentation_results: Dict mapping method names to segmentation masks
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    n_methods = len(segmentation_results)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)
    
    # Show original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show each segmentation method
    for idx, (method_name, seg_result) in enumerate(segmentation_results.items()):
        ax = axes[idx + 1]
        
        # Create colored overlay
        cell_mask = seg_result.get('cell', seg_result.get('cell_matched_mask'))
        nucleus_mask = seg_result.get('nucleus', seg_result.get('nucleus_matched_mask'))
        
        # Create RGB image
        rgb = np.zeros((*image.shape[:2], 3))
        rgb[:, :, 0] = (cell_mask > 0).astype(float)
        rgb[:, :, 1] = (nucleus_mask > 0).astype(float)
        
        ax.imshow(image, cmap='gray', alpha=0.5)
        ax.imshow(rgb, alpha=0.5)
        
        # Add statistics
        n_cells = len(np.unique(cell_mask)) - 1
        n_nuclei = len(np.unique(nucleus_mask)) - 1
        matched_frac = seg_result.get('matched_fraction', 0)
        
        ax.set_title(f'{method_name}\nCells: {n_cells}, Nuclei: {n_nuclei}\nMatched: {matched_frac:.1%}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_nucleus_mask_visualization(
    image: np.ndarray,
    nucleus_mask: np.ndarray,
    show_ids: bool = False,
    figsize: Tuple[int, int] = (12, 12)
) -> plt.Figure:
    """
    Create a visualization of all nucleus masks with a morphology-aware colormap.

    Args:
        image: Original image (can be single channel or RGB)
        nucleus_mask: Nucleus segmentation mask
        show_ids: Whether to show nucleus IDs
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    display_img = _normalize_image_for_display(image)
    ax.imshow(display_img, cmap='gray' if image.ndim == 2 else None, alpha=0.5)

    colored_mask, boundary_overlay, cmap, norm, props = _build_mask_overlays(
        nucleus_mask, 'magma', 0.65
    )

    if colored_mask is not None:
        ax.imshow(colored_mask)
    if boundary_overlay is not None:
        ax.imshow(boundary_overlay)

    if show_ids:
        for prop in props:
            y, x = prop.centroid
            ax.text(
                x,
                y,
                str(prop.label),
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
            )

    if cmap is not None and norm is not None and props:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Nucleus area (pixels)', rotation=270, labelpad=15)

    nucleus_count = _count_unique_labels(nucleus_mask)
    ax.set_title(
        f'Nucleus Mask Visualization\n({nucleus_count} nuclei detected, color = area)'
    )
    ax.axis('off')

    return fig

def create_wholecell_mask_visualization(
    image: np.ndarray,
    cell_mask: np.ndarray,
    show_ids: bool = False,
    figsize: Tuple[int, int] = (12, 12)
) -> plt.Figure:
    """
    Create a visualization of all whole cell masks with a morphology-aware colormap.

    Args:
        image: Original image (can be single channel or RGB)
        cell_mask: Whole cell segmentation mask
        show_ids: Whether to show cell IDs
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    display_img = _normalize_image_for_display(image)
    ax.imshow(display_img, cmap='gray' if image.ndim == 2 else None, alpha=0.5)

    colored_mask, boundary_overlay, cmap, norm, props = _build_mask_overlays(
        cell_mask, 'viridis', 0.6
    )

    if colored_mask is not None:
        ax.imshow(colored_mask)
    if boundary_overlay is not None:
        ax.imshow(boundary_overlay)

    if show_ids:
        for prop in props:
            y, x = prop.centroid
            ax.text(
                x,
                y,
                str(prop.label),
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
            )

    if cmap is not None and norm is not None and props:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Cell area (pixels)', rotation=270, labelpad=15)

    cell_count = _count_unique_labels(cell_mask)
    ax.set_title(
        f'Whole-cell Mask Visualization\n({cell_count} cells detected, color = area)'
    )
    ax.axis('off')

    return fig
