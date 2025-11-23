"""Test fixtures for mask repair testing.

This module provides synthetic mask pairs (cell_mask, nucleus_mask) with known
ground truth for testing the repair pipeline. Each fixture returns:
    - cell_mask: Labeled cell segmentation mask
    - nucleus_mask: Labeled nucleus segmentation mask
    - expected_results: Dictionary with expected repair outputs

All fixtures use deterministic random seeds for reproducibility.
"""

import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging


def create_simple_mask_pair(
    n_cells: int = 10,
    image_size: Tuple[int, int] = (512, 512),
    cell_radius: int = 40,
    nucleus_radius: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create simple case with perfect 1:1 cell-nucleus matching.

    Each cell has exactly one nucleus centered inside it. No overlaps,
    no missing nuclei, no extra nuclei.

    Args:
        n_cells: Number of cells to create
        image_size: (height, width) of output masks
        cell_radius: Radius of each cell (pixels)
        nucleus_radius: Radius of each nucleus (pixels)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
        - cell_mask: (H, W) labeled mask
        - nucleus_mask: (H, W) labeled mask
        - expected_results: Dict with expected outputs after repair
    """
    np.random.seed(seed)
    H, W = image_size

    cell_mask = np.zeros((H, W), dtype=np.uint32)
    nucleus_mask = np.zeros((H, W), dtype=np.uint32)

    # Create grid of non-overlapping cells
    spacing = int(np.sqrt(H * W / n_cells))
    positions = []

    for i in range(n_cells):
        # Grid placement to avoid overlaps
        row = (i // int(np.sqrt(n_cells))) * spacing + spacing // 2
        col = (i % int(np.sqrt(n_cells))) * spacing + spacing // 2

        # Add small random jitter
        row += np.random.randint(-spacing // 4, spacing // 4)
        col += np.random.randint(-spacing // 4, spacing // 4)

        # Ensure within bounds
        row = np.clip(row, cell_radius, H - cell_radius)
        col = np.clip(col, cell_radius, W - cell_radius)

        positions.append((row, col))

        # Create circular cell
        y, x = np.ogrid[:H, :W]
        cell_circle = (x - col) ** 2 + (y - row) ** 2 <= cell_radius ** 2
        cell_mask[cell_circle] = i + 1

        # Create centered nucleus
        nucleus_circle = (x - col) ** 2 + (y - row) ** 2 <= nucleus_radius ** 2
        nucleus_mask[nucleus_circle] = i + 1

    # Expected results: all cells should match perfectly
    expected_results = {
        "n_matched_cells": n_cells,
        "n_matched_nuclei": n_cells,
        "n_unmatched_cells": 0,
        "n_unmatched_nuclei": 0,
        "matched_cell_ids": set(range(1, n_cells + 1)),
        "matched_nucleus_ids": set(range(1, n_cells + 1)),
        "mismatch_fractions": {i + 1: 0.0 for i in range(n_cells)},  # Perfect matches
    }

    return cell_mask, nucleus_mask, expected_results


def create_overlapping_nuclei_case(
    n_cells: int = 5,
    n_nuclei: int = 10,
    image_size: Tuple[int, int] = (512, 512),
    cell_radius: int = 50,
    nucleus_radius: int = 15,
    seed: int = 43,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create case where multiple nuclei overlap same cell.

    Some cells have multiple candidate nuclei. The repair algorithm should
    match each cell to the nucleus with the largest overlap.

    Args:
        n_cells: Number of cells
        n_nuclei: Number of nuclei (should be > n_cells)
        image_size: (height, width) of output masks
        cell_radius: Radius of each cell
        nucleus_radius: Radius of each nucleus
        seed: Random seed

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
    """
    np.random.seed(seed)
    H, W = image_size

    cell_mask = np.zeros((H, W), dtype=np.uint32)
    nucleus_mask = np.zeros((H, W), dtype=np.uint32)

    # Create cells in grid
    spacing = int(np.sqrt(H * W / n_cells))
    cell_positions = []

    for i in range(n_cells):
        row = (i // int(np.sqrt(n_cells))) * spacing + spacing // 2
        col = (i % int(np.sqrt(n_cells))) * spacing + spacing // 2

        row = np.clip(row, cell_radius, H - cell_radius)
        col = np.clip(col, cell_radius, W - cell_radius)

        cell_positions.append((row, col))

        y, x = np.ogrid[:H, :W]
        cell_circle = (x - col) ** 2 + (y - row) ** 2 <= cell_radius ** 2
        cell_mask[cell_circle] = i + 1

    # Create nuclei - some cells get multiple nuclei nearby
    nucleus_id = 1
    matched_pairs = {}  # cell_id -> best_nucleus_id (based on position)

    for i in range(n_nuclei):
        # Assign nucleus to a cell (some cells get multiple)
        cell_idx = i % n_cells
        row, col = cell_positions[cell_idx]

        # Add offset so multiple nuclei don't perfectly overlap
        if i >= n_cells:
            offset = nucleus_radius * 2
            angle = 2 * np.pi * (i - n_cells) / max(1, n_nuclei - n_cells)
            row += int(offset * np.sin(angle))
            col += int(offset * np.cos(angle))

        row = np.clip(row, nucleus_radius, H - nucleus_radius)
        col = np.clip(col, nucleus_radius, W - nucleus_radius)

        y, x = np.ogrid[:H, :W]
        nucleus_circle = (x - col) ** 2 + (y - row) ** 2 <= nucleus_radius ** 2

        # Only add if not overlapping existing nucleus
        if np.sum((nucleus_mask > 0) & nucleus_circle) == 0:
            nucleus_mask[nucleus_circle] = nucleus_id

            # Track which nucleus is best for each cell (first one = centered)
            if (cell_idx + 1) not in matched_pairs:
                matched_pairs[cell_idx + 1] = nucleus_id

            nucleus_id += 1

    expected_results = {
        "n_matched_cells": n_cells,
        "n_matched_nuclei": n_cells,
        "n_unmatched_cells": 0,
        "n_unmatched_nuclei": nucleus_id - 1 - n_cells,  # Extra nuclei
        "matched_cell_ids": set(range(1, n_cells + 1)),
        "note": "Multiple nuclei per cell - algorithm picks best overlap",
    }

    return cell_mask, nucleus_mask, expected_results


def create_unmatched_cells_case(
    n_cells_with_nuclei: int = 5,
    n_cells_without_nuclei: int = 3,
    image_size: Tuple[int, int] = (512, 512),
    cell_radius: int = 40,
    nucleus_radius: int = 20,
    seed: int = 44,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create case with cells that have no overlapping nucleus.

    Some cells should be discarded because they don't have a nucleus.

    Args:
        n_cells_with_nuclei: Number of cells with nuclei
        n_cells_without_nuclei: Number of cells without nuclei (should be discarded)
        image_size: (height, width)
        cell_radius: Radius of cells
        nucleus_radius: Radius of nuclei
        seed: Random seed

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
    """
    np.random.seed(seed)
    H, W = image_size

    n_total = n_cells_with_nuclei + n_cells_without_nuclei

    cell_mask = np.zeros((H, W), dtype=np.uint32)
    nucleus_mask = np.zeros((H, W), dtype=np.uint32)

    spacing = int(np.sqrt(H * W / n_total))

    cells_with_nucleus_ids = set()

    for i in range(n_total):
        row = (i // int(np.sqrt(n_total))) * spacing + spacing // 2
        col = (i % int(np.sqrt(n_total))) * spacing + spacing // 2

        row = np.clip(row, cell_radius, H - cell_radius)
        col = np.clip(col, cell_radius, W - cell_radius)

        y, x = np.ogrid[:H, :W]
        cell_circle = (x - col) ** 2 + (y - row) ** 2 <= cell_radius ** 2
        cell_mask[cell_circle] = i + 1

        # Only create nucleus for first n_cells_with_nuclei cells
        if i < n_cells_with_nuclei:
            nucleus_circle = (x - col) ** 2 + (y - row) ** 2 <= nucleus_radius ** 2
            nucleus_mask[nucleus_circle] = i + 1
            cells_with_nucleus_ids.add(i + 1)

    expected_results = {
        "n_matched_cells": n_cells_with_nuclei,
        "n_matched_nuclei": n_cells_with_nuclei,
        "n_unmatched_cells": n_cells_without_nuclei,
        "n_unmatched_nuclei": 0,
        "matched_cell_ids": cells_with_nucleus_ids,
        "matched_nucleus_ids": cells_with_nucleus_ids,
    }

    return cell_mask, nucleus_mask, expected_results


def create_partial_overlap_case(
    n_cells: int = 5,
    image_size: Tuple[int, int] = (512, 512),
    cell_radius: int = 40,
    nucleus_radius: int = 25,
    nucleus_offset: int = 20,
    seed: int = 45,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create case where nuclei partially extend outside cell membrane.

    Nuclei should be trimmed to only include pixels inside cell interior
    (excluding membrane).

    Args:
        n_cells: Number of cells
        image_size: (height, width)
        cell_radius: Radius of cells
        nucleus_radius: Radius of nuclei (larger than usual)
        nucleus_offset: Offset of nucleus center from cell center
        seed: Random seed

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
    """
    np.random.seed(seed)
    H, W = image_size

    cell_mask = np.zeros((H, W), dtype=np.uint32)
    nucleus_mask = np.zeros((H, W), dtype=np.uint32)

    spacing = int(np.sqrt(H * W / n_cells))

    for i in range(n_cells):
        row = (i // int(np.sqrt(n_cells))) * spacing + spacing // 2
        col = (i % int(np.sqrt(n_cells))) * spacing + spacing // 2

        row = np.clip(row, cell_radius, H - cell_radius)
        col = np.clip(col, cell_radius, W - cell_radius)

        y, x = np.ogrid[:H, :W]

        # Create cell
        cell_circle = (x - col) ** 2 + (y - row) ** 2 <= cell_radius ** 2
        cell_mask[cell_circle] = i + 1

        # Create offset nucleus (partially outside)
        nucleus_row = row + nucleus_offset
        nucleus_col = col + nucleus_offset
        nucleus_circle = (
            (x - nucleus_col) ** 2 + (y - nucleus_row) ** 2 <= nucleus_radius ** 2
        )
        nucleus_mask[nucleus_circle] = i + 1

    expected_results = {
        "n_matched_cells": n_cells,
        "n_matched_nuclei": n_cells,
        "n_unmatched_cells": 0,
        "n_unmatched_nuclei": 0,
        "note": "Nuclei should be trimmed to cell interior (excluding membrane)",
        "expect_nonzero_mismatch": True,  # mismatch_fraction > 0 due to trimming
    }

    return cell_mask, nucleus_mask, expected_results


def create_stress_test_case(
    n_cells: int = 10000,
    image_size: Tuple[int, int] = (2000, 2000),
    seed: int = 46,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create large-scale stress test with many cells.

    Tests performance on realistic sample sizes. Uses Voronoi-like tessellation
    for more realistic cell shapes.

    Args:
        n_cells: Number of cells (default 10K)
        image_size: (height, width) - should be large enough
        seed: Random seed

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
    """
    np.random.seed(seed)
    H, W = image_size

    # Generate random cell centers
    centers = np.random.randint(0, min(H, W), size=(n_cells, 2))

    # Create Voronoi-like cell assignment
    cell_mask = np.zeros((H, W), dtype=np.uint32)
    nucleus_mask = np.zeros((H, W), dtype=np.uint32)

    # For performance, use simple random assignment instead of true Voronoi
    # Randomly assign each pixel to nearest center (sampled)
    logging.info(f"Creating stress test with {n_cells} cells on {H}x{W} image...")

    # Create sparse random cells to avoid full Voronoi computation
    cell_radius_range = (20, 50)
    nucleus_radius_range = (10, 25)

    for i, (cy, cx) in enumerate(centers):
        cell_r = np.random.randint(*cell_radius_range)
        nucleus_r = np.random.randint(*nucleus_radius_range)

        # Create circular cell
        y, x = np.ogrid[
            max(0, cy - cell_r) : min(H, cy + cell_r),
            max(0, cx - cell_r) : min(W, cx + cell_r),
        ]
        cell_circle = (x - cx) ** 2 + (y - cy) ** 2 <= cell_r ** 2

        y_start = max(0, cy - cell_r)
        x_start = max(0, cx - cell_r)

        # Only paint where not already assigned (avoid overlaps)
        mask_region = cell_mask[y_start : y_start + cell_circle.shape[0],
                                 x_start : x_start + cell_circle.shape[1]]
        cell_circle_masked = cell_circle & (mask_region == 0)
        mask_region[cell_circle_masked] = i + 1

        # Create nucleus
        y, x = np.ogrid[
            max(0, cy - nucleus_r) : min(H, cy + nucleus_r),
            max(0, cx - nucleus_r) : min(W, cx + nucleus_r),
        ]
        nucleus_circle = (x - cx) ** 2 + (y - cy) ** 2 <= nucleus_r ** 2

        y_start = max(0, cy - nucleus_r)
        x_start = max(0, cx - nucleus_r)

        nucleus_region = nucleus_mask[y_start : y_start + nucleus_circle.shape[0],
                                       x_start : x_start + nucleus_circle.shape[1]]
        nucleus_circle_masked = nucleus_circle & (nucleus_region == 0)
        nucleus_region[nucleus_circle_masked] = i + 1

    n_cells_created = len(np.unique(cell_mask)) - 1
    n_nuclei_created = len(np.unique(nucleus_mask)) - 1

    expected_results = {
        "n_cells_created": n_cells_created,
        "n_nuclei_created": n_nuclei_created,
        "note": f"Large stress test with {n_cells} target cells",
        "min_matched": min(n_cells_created, n_nuclei_created) * 0.8,  # At least 80% should match
    }

    return cell_mask, nucleus_mask, expected_results


def plot_mask_pair(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    title: str = "Cell and Nucleus Masks",
    save_path: str = None,
):
    """Visualize cell and nucleus masks for debugging.

    Args:
        cell_mask: (H, W) labeled cell mask
        nucleus_mask: (H, W) labeled nucleus mask
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Cell mask
    axes[0].imshow(cell_mask, cmap="tab20", interpolation="nearest")
    axes[0].set_title(f"Cell Mask\n{len(np.unique(cell_mask)) - 1} cells")
    axes[0].axis("off")

    # Nucleus mask
    axes[1].imshow(nucleus_mask, cmap="tab20", interpolation="nearest")
    axes[1].set_title(f"Nucleus Mask\n{len(np.unique(nucleus_mask)) - 1} nuclei")
    axes[1].axis("off")

    # Overlay
    overlay = np.zeros((*cell_mask.shape, 3), dtype=np.float32)
    overlay[cell_mask > 0] = [1, 0, 0]  # Red cells
    overlay[nucleus_mask > 0] = [0, 1, 0]  # Green nuclei
    overlay[(cell_mask > 0) & (nucleus_mask > 0)] = [1, 1, 0]  # Yellow overlap

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay\n(Red=cell, Green=nucleus, Yellow=overlap)")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def create_edge_case_empty_masks() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create edge case with empty masks (no cells, no nuclei).

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
    """
    cell_mask = np.zeros((100, 100), dtype=np.uint32)
    nucleus_mask = np.zeros((100, 100), dtype=np.uint32)

    expected_results = {
        "n_matched_cells": 0,
        "n_matched_nuclei": 0,
        "n_unmatched_cells": 0,
        "n_unmatched_nuclei": 0,
        "note": "Empty masks should return empty results without errors",
    }

    return cell_mask, nucleus_mask, expected_results


def create_edge_case_single_cell() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create edge case with single cell and nucleus.

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
    """
    cell_mask = np.zeros((100, 100), dtype=np.uint32)
    nucleus_mask = np.zeros((100, 100), dtype=np.uint32)

    # Create one cell and one nucleus in center
    y, x = np.ogrid[:100, :100]
    cell_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 30 ** 2
    nucleus_circle = (x - 50) ** 2 + (y - 50) ** 2 <= 15 ** 2

    cell_mask[cell_circle] = 1
    nucleus_mask[nucleus_circle] = 1

    expected_results = {
        "n_matched_cells": 1,
        "n_matched_nuclei": 1,
        "n_unmatched_cells": 0,
        "n_unmatched_nuclei": 0,
        "note": "Single cell-nucleus pair should match correctly",
    }

    return cell_mask, nucleus_mask, expected_results


def create_noncontiguous_labels_case(
    label_gaps: str = "nucleus",
    nucleus_labels: list = [1, 5, 7, 23],
    cell_labels: list = [1, 2, 3, 4],
    image_size: Tuple[int, int] = (256, 256),
    cell_radius: int = 30,
    nucleus_radius: int = 15,
    seed: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create test case with non-contiguous label IDs to test label→index mapping.

    This tests the fix for the bug where nucleus_coords[j - 1] assumed contiguous
    labels (1, 2, 3, ...). With non-contiguous labels (e.g., 1, 5, 7, 23), the
    code must use a label→index mapping instead of assuming label-1 = index.

    Args:
        label_gaps: Which mask has non-contiguous labels ("nucleus", "cell", or "both")
        nucleus_labels: List of nucleus label IDs to use (can have gaps)
        cell_labels: List of cell label IDs to use (can have gaps)
        image_size: (height, width) of output masks
        cell_radius: Radius of each cell (pixels)
        nucleus_radius: Radius of each nucleus (pixels)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (cell_mask, nucleus_mask, expected_results)
        - cell_mask: (H, W) labeled mask with specified labels
        - nucleus_mask: (H, W) labeled mask with specified labels
        - expected_results: Dict with expected outputs after repair

    Example:
        # Test non-contiguous nucleus labels (most common case)
        cell, nucleus, expected = create_noncontiguous_labels_case(
            label_gaps="nucleus",
            nucleus_labels=[1, 5, 7, 23],  # Gaps: missing 2,3,4,6,8-22
            cell_labels=[1, 2, 3, 4]       # Contiguous
        )
    """
    np.random.seed(seed)
    H, W = image_size

    # Determine which labels to use based on label_gaps parameter
    if label_gaps == "nucleus":
        n_labels_nucleus = nucleus_labels
        n_labels_cell = cell_labels
    elif label_gaps == "cell":
        n_labels_nucleus = nucleus_labels if nucleus_labels != [1, 5, 7, 23] else [1, 2, 3, 4]
        n_labels_cell = cell_labels if cell_labels != [1, 2, 3, 4] else [2, 10, 15, 30]
    elif label_gaps == "both":
        n_labels_nucleus = nucleus_labels
        n_labels_cell = cell_labels if cell_labels != [1, 2, 3, 4] else [2, 10, 15, 30]
    else:
        raise ValueError(f"label_gaps must be 'nucleus', 'cell', or 'both', got: {label_gaps}")

    n_cells = len(n_labels_cell)

    cell_mask = np.zeros((H, W), dtype=np.uint32)
    nucleus_mask = np.zeros((H, W), dtype=np.uint32)

    # Create grid of non-overlapping cells with specified label IDs
    grid_size = int(np.ceil(np.sqrt(n_cells)))
    spacing = min(H, W) // (grid_size + 1)

    positions = []
    for idx in range(n_cells):
        row = idx // grid_size
        col = idx % grid_size
        y = spacing * (row + 1)
        x = spacing * (col + 1)
        positions.append((y, x))

    # Create cells with specified label IDs
    for idx, (cy, cx) in enumerate(positions):
        label_id = n_labels_cell[idx]
        y_grid, x_grid = np.ogrid[:H, :W]
        cell_circle = (y_grid - cy)**2 + (x_grid - cx)**2 <= cell_radius**2
        cell_mask[cell_circle] = label_id

    # Create nuclei with specified label IDs (one per cell, centered)
    for idx, (cy, cx) in enumerate(positions):
        if idx < len(n_labels_nucleus):
            label_id = n_labels_nucleus[idx]
            y_grid, x_grid = np.ogrid[:H, :W]
            nucleus_circle = (y_grid - cy)**2 + (x_grid - cx)**2 <= nucleus_radius**2
            nucleus_mask[nucleus_circle] = label_id

    # Expected results: All cells should match (perfect overlap)
    n_matched = min(len(n_labels_cell), len(n_labels_nucleus))

    expected_results = {
        "n_matched_cells": n_matched,
        "n_matched_nuclei": n_matched,
        "n_unmatched_cells": len(n_labels_cell) - n_matched,
        "n_unmatched_nuclei": len(n_labels_nucleus) - n_matched,
        "note": f"Non-contiguous labels test (label_gaps={label_gaps})",
        "cell_labels": n_labels_cell,
        "nucleus_labels": n_labels_nucleus,
        "first_cell_label": n_labels_cell[0],
        "first_nucleus_label": n_labels_nucleus[0],
    }

    return cell_mask, nucleus_mask, expected_results


# Export all fixture functions
__all__ = [
    "create_simple_mask_pair",
    "create_overlapping_nuclei_case",
    "create_unmatched_cells_case",
    "create_partial_overlap_case",
    "create_stress_test_case",
    "create_edge_case_empty_masks",
    "create_edge_case_single_cell",
    "create_noncontiguous_labels_case",
    "plot_mask_pair",
]
