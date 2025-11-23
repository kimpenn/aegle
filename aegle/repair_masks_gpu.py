"""GPU-accelerated integrated repair pipeline.

This module provides the complete GPU-accelerated mask repair pipeline,
integrating morphological operations, overlap computation, and cell matching.
"""

import logging
import numpy as np
import time
from typing import Tuple, Dict, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def repair_masks_gpu(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    config: Optional[Dict] = None,
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """GPU-accelerated mask repair pipeline.

    This is the main entry point for GPU-accelerated repair. It integrates:
    - GPU morphological operations (boundary detection)
    - GPU overlap matrix computation
    - Cell-nucleus matching logic
    - CPU fallback on errors

    Args:
        cell_mask: Cell segmentation mask (H, W) uint32
        nucleus_mask: Nucleus segmentation mask (H, W) uint32
        config: Optional config dict with GPU settings
        use_gpu: Whether to attempt GPU acceleration
        batch_size: Optional batch size override for overlap computation

    Returns:
        Tuple of:
            - repaired_cell_mask: Repaired cell mask (H, W) uint32
            - repaired_nucleus_mask: Repaired nucleus mask (H, W) uint32
            - repair_metadata: Dict with stats and timing

    Example:
        >>> cell_mask = np.zeros((1000, 1000), dtype=np.uint32)
        >>> nucleus_mask = np.zeros((1000, 1000), dtype=np.uint32)
        >>> cell_repaired, nucleus_repaired, metadata = repair_masks_gpu(
        ...     cell_mask, nucleus_mask, use_gpu=True
        ... )
    """
    from aegle.gpu_utils import is_cupy_available, log_gpu_memory, clear_gpu_memory

    # Ensure masks are 2D
    cell_mask = np.asarray(cell_mask).squeeze()
    nucleus_mask = np.asarray(nucleus_mask).squeeze()

    # Initialize metadata
    metadata = {
        "use_gpu": use_gpu,
        "gpu_available": is_cupy_available(),
        "gpu_used": False,
        "fallback_to_cpu": False,
        "total_time": 0.0,
        "n_cells_input": len(np.unique(cell_mask)) - 1,
        "n_nuclei_input": len(np.unique(nucleus_mask)) - 1,
    }

    start_time = time.time()

    # Check GPU availability
    if not use_gpu:
        logger.info("GPU disabled by config, using CPU version")
        metadata["fallback_to_cpu"] = True
        return _repair_masks_cpu_fallback(cell_mask, nucleus_mask, metadata)

    if not is_cupy_available():
        logger.warning("GPU requested but not available, falling back to CPU")
        metadata["fallback_to_cpu"] = True
        return _repair_masks_cpu_fallback(cell_mask, nucleus_mask, metadata)

    try:
        # Attempt GPU repair
        logger.info("=" * 80)
        logger.info("STARTING GPU-ACCELERATED MASK REPAIR (Phase 3)")
        logger.info("=" * 80)
        log_gpu_memory("GPU memory before repair")

        cell_repaired, nucleus_repaired, gpu_metadata = _repair_masks_gpu_impl(
            cell_mask, nucleus_mask, batch_size
        )

        # Merge GPU metadata
        metadata.update(gpu_metadata)
        metadata["gpu_used"] = True
        metadata["total_time"] = time.time() - start_time

        log_gpu_memory("GPU memory after repair")
        clear_gpu_memory()

        logger.info(
            f"GPU repair completed in {metadata['total_time']:.2f}s "
            f"({metadata['n_matched_cells']}/{metadata['n_cells_input']} cells matched)"
        )

        return cell_repaired, nucleus_repaired, metadata

    except Exception as e:
        logger.error(f"GPU repair failed: {e}")
        logger.warning("Falling back to CPU version")
        metadata["fallback_to_cpu"] = True
        metadata["fallback_reason"] = str(e)
        clear_gpu_memory()
        return _repair_masks_cpu_fallback(cell_mask, nucleus_mask, metadata)


def _repair_masks_gpu_impl(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Internal GPU implementation (no fallback logic).

    Args:
        cell_mask: Cell mask (H, W)
        nucleus_mask: Nucleus mask (H, W)
        batch_size: Batch size for overlap computation

    Returns:
        Tuple of (cell_repaired, nucleus_repaired, metadata)
    """
    from aegle.repair_masks_gpu_morphology import compute_labeled_boundary_gpu
    from aegle.repair_masks_gpu_overlap import compute_overlap_matrix_gpu
    from aegle.repair_masks_gpu_mismatch import compute_mismatch_matrix_gpu
    from aegle.repair_masks import get_indices_numpy

    metadata = {}
    stage_timings = {}

    # Stage 1: Compute labeled boundaries (GPU)
    logger.info("Stage 1/5: Computing labeled cell boundaries on GPU...")
    t0 = time.time()
    cell_membrane_mask = compute_labeled_boundary_gpu(cell_mask)
    stage_timings["boundary_computation"] = time.time() - t0
    logger.info(f"  ✓ Boundary computation completed in {stage_timings['boundary_computation']:.3f}s")

    # Stage 2: Extract coordinates (CPU - already fast, keep as-is)
    logger.info("Stage 2/5: Extracting cell/nucleus coordinates (CPU)...")
    t0 = time.time()
    cell_coords_dict = get_indices_numpy(cell_mask)
    nucleus_coords_dict = get_indices_numpy(nucleus_mask)
    stage_timings["coordinate_extraction"] = time.time() - t0
    logger.info(f"  ✓ Coordinate extraction completed in {stage_timings['coordinate_extraction']:.3f}s")

    # Remove background and get labels
    cell_labels = sorted([label for label in cell_coords_dict.keys() if label > 0])
    nucleus_labels = sorted([label for label in nucleus_coords_dict.keys() if label > 0])

    if not cell_labels or not nucleus_labels:
        logger.warning("No cells or nuclei found, returning empty masks")
        empty_mask = np.zeros_like(cell_mask, dtype=np.uint32)
        metadata.update(stage_timings)
        metadata.update({
            "n_matched_cells": 0,
            "n_matched_nuclei": 0,
            "n_repaired": 0,
        })
        return empty_mask, empty_mask, metadata

    # Convert to list format and create label→index mapping
    cell_coords = [np.array(cell_coords_dict[label]).T for label in cell_labels]
    nucleus_coords = [np.array(nucleus_coords_dict[label]).T for label in nucleus_labels]
    nucleus_label_to_idx = {label: idx for idx, label in enumerate(nucleus_labels)}

    # Extract membrane coordinates using sparse method
    from aegle.repair_masks import get_indices_sparse
    cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]
    cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))

    logger.info(f"  Found {len(cell_coords)} cells, {len(nucleus_coords)} nuclei")

    # Stage 3: Compute overlap matrix (GPU)
    logger.info("Stage 3/5: Computing cell-nucleus overlap matrix on GPU...")
    t0 = time.time()
    overlap_matrix, cell_labels_arr, nucleus_labels_arr = compute_overlap_matrix_gpu(
        cell_mask, nucleus_mask, batch_size=batch_size or 0
    )
    stage_timings["overlap_computation"] = time.time() - t0
    logger.info(
        f"  ✓ Overlap computation completed in {stage_timings['overlap_computation']:.3f}s "
        f"({overlap_matrix.sum()} overlapping pairs)"
    )

    # Stage 3.5: Compute mismatch matrix (GPU) - Phase 3 optimization
    logger.info("Stage 4/5: Computing mismatch fractions on GPU (Phase 3)...")
    t0 = time.time()
    mismatch_matrix_sparse = compute_mismatch_matrix_gpu(
        cell_mask,
        nucleus_mask,
        cell_membrane_mask,
        overlap_matrix,
        cell_labels_arr,
        nucleus_labels_arr,
        batch_size=batch_size or 10000,
    )
    stage_timings["mismatch_computation"] = time.time() - t0
    n_pairs = mismatch_matrix_sparse.nnz
    sparse_mb = mismatch_matrix_sparse.data.nbytes / 1e6
    pairs_per_sec = n_pairs / stage_timings["mismatch_computation"] if stage_timings["mismatch_computation"] > 0 else 0
    logger.info(
        f"  ✓ Mismatch computation completed in {stage_timings['mismatch_computation']:.3f}s "
        f"({n_pairs} pairs at {pairs_per_sec:.0f} pairs/sec, {sparse_mb:.2f} MB sparse)"
    )

    # Stage 4: Match cells to nuclei (CPU with GPU-precomputed mismatch fractions)
    logger.info("Stage 5/5: Matching cells to nuclei (CPU greedy algorithm with GPU mismatch)...")
    t0 = time.time()
    cell_matched_list, nucleus_matched_list, n_repaired = _match_cells_to_nuclei(
        cell_coords,
        nucleus_coords,
        cell_membrane_coords,
        nucleus_mask,
        overlap_matrix,
        cell_labels,
        nucleus_labels,
        nucleus_labels_arr,  # Pass the array from overlap computation
        nucleus_label_to_idx,
        mismatch_matrix_sparse,  # NEW: Pass precomputed mismatch matrix
    )
    stage_timings["cell_matching"] = time.time() - t0
    cells_per_sec = len(cell_coords) / stage_timings["cell_matching"] if stage_timings["cell_matching"] > 0 else 0
    logger.info(
        f"  ✓ Cell matching completed in {stage_timings['cell_matching']:.3f}s "
        f"({len(cell_matched_list)}/{len(cell_coords)} cells matched at {cells_per_sec:.0f} cells/sec, "
        f"{n_repaired} repaired)"
    )

    # Stage 5: Assemble repaired masks (CPU)
    logger.info("Assembling final repaired masks...")
    t0 = time.time()
    from aegle.repair_masks import get_mask
    cell_matched_mask = get_mask(cell_matched_list, cell_mask.shape)
    nucleus_matched_mask = get_mask(nucleus_matched_list, nucleus_mask.shape)
    stage_timings["mask_assembly"] = time.time() - t0
    logger.info(f"  ✓ Mask assembly completed in {stage_timings['mask_assembly']:.3f}s")

    # Update metadata
    metadata.update(stage_timings)
    metadata.update({
        "n_matched_cells": len(cell_matched_list),
        "n_matched_nuclei": len(nucleus_matched_list),
        "n_repaired": n_repaired,
        "overlap_density": overlap_matrix.sum() / (len(cell_labels) * len(nucleus_labels))
        if len(cell_labels) * len(nucleus_labels) > 0 else 0,
    })

    # Log summary
    total_time = sum(stage_timings.values())
    logger.info("=" * 80)
    logger.info("GPU REPAIR PIPELINE COMPLETED (Phase 3)")
    logger.info("=" * 80)
    logger.info(f"Stage Timing Breakdown:")
    logger.info(f"  Boundary (GPU):      {stage_timings['boundary_computation']:7.2f}s ({stage_timings['boundary_computation']/total_time*100:5.1f}%)")
    logger.info(f"  Coordinates (CPU):   {stage_timings['coordinate_extraction']:7.2f}s ({stage_timings['coordinate_extraction']/total_time*100:5.1f}%)")
    logger.info(f"  Overlap (GPU):       {stage_timings['overlap_computation']:7.2f}s ({stage_timings['overlap_computation']/total_time*100:5.1f}%)")
    logger.info(f"  Mismatch (GPU):      {stage_timings['mismatch_computation']:7.2f}s ({stage_timings['mismatch_computation']/total_time*100:5.1f}%)")
    logger.info(f"  Matching (CPU+GPU):  {stage_timings['cell_matching']:7.2f}s ({stage_timings['cell_matching']/total_time*100:5.1f}%)")
    logger.info(f"  Assembly (CPU):      {stage_timings['mask_assembly']:7.2f}s ({stage_timings['mask_assembly']/total_time*100:5.1f}%)")
    logger.info(f"  {'─' * 30}")
    logger.info(f"  TOTAL:               {total_time:7.2f}s")
    logger.info(f"Results: {len(cell_matched_list)}/{len(cell_coords)} cells matched, {n_repaired} repaired")
    logger.info("=" * 80)

    return cell_matched_mask.astype(np.uint32), nucleus_matched_mask.astype(np.uint32), metadata


def _match_cells_to_nuclei(
    cell_coords,
    nucleus_coords,
    cell_membrane_coords,
    nucleus_mask,
    overlap_matrix,
    cell_labels,
    nucleus_labels,
    nucleus_labels_arr,
    nucleus_label_to_idx,
    mismatch_matrix_sparse=None,
):
    """Match cells to nuclei using precomputed mismatch matrix.

    This is the CPU-based matching logic from the original repair_masks.py,
    but accelerated by using the GPU-precomputed mismatch matrix to avoid
    expensive Python set operations in the inner loop.

    Args:
        cell_coords: List of cell coordinate arrays
        nucleus_coords: List of nucleus coordinate arrays
        cell_membrane_coords: List of membrane coordinate arrays
        nucleus_mask: Nucleus mask for looking up nucleus IDs
        overlap_matrix: (n_cells, n_nuclei) boolean overlap matrix from GPU
        cell_labels: List of cell label IDs
        nucleus_labels: List of nucleus label IDs
        nucleus_labels_arr: NumPy array of nucleus labels (for indexing)
        nucleus_label_to_idx: Dict mapping nucleus label → index
        mismatch_matrix_sparse: (n_cells, n_nuclei) sparse matrix of precomputed
                               mismatch fractions (optional, for GPU acceleration)

    Returns:
        Tuple of (cell_matched_list, nucleus_matched_list, n_repaired)
    """
    from aegle.repair_masks import get_matched_cells

    cell_matched_indices = set()
    nucleus_matched_indices = set()
    cell_matched_list = []
    nucleus_matched_list = []
    repaired_num = 0
    total_cells = len(cell_coords)

    # Progress bar for cell matching
    with tqdm(
        total=total_cells,
        desc="Matching cells to nuclei",
        unit="cell",
        mininterval=1.0,
        dynamic_ncols=True,
    ) as pbar:
        for i in range(total_cells):
            if len(cell_coords[i]) == 0:
                pbar.update(1)
                continue

            # Use overlap matrix to find candidate nuclei (GPU precomputed, sparse CSR)
            # overlap_matrix.getrow(i) gives sparse row, .indices gives column indices of nonzeros
            overlap_row = overlap_matrix.getrow(i)  # Sparse CSR row (O(1) access)
            overlapping_nucleus_indices = overlap_row.indices  # Column indices of nonzeros

            if len(overlapping_nucleus_indices) == 0:
                # No overlapping nuclei - skip this cell
                pbar.update(1)
                continue

            # Convert nucleus indices back to label IDs
            nucleus_search_num = nucleus_labels_arr[overlapping_nucleus_indices]

            best_mismatch_fraction = 1
            whole_cell_best = []

            # Phase 3 optimization: Use precomputed mismatch matrix if available
            if mismatch_matrix_sparse is not None:
                # GPU-accelerated path: Use sparse matrix lookups
                # Get mismatch fractions for all overlapping nuclei at once
                mismatch_row = mismatch_matrix_sparse.getrow(i)

                if mismatch_row.nnz > 0:
                    # Filter to unmatched nuclei only
                    available_mask = np.array(
                        [idx not in nucleus_matched_indices for idx in mismatch_row.indices]
                    )

                    if np.any(available_mask) and i not in cell_matched_indices:
                        # Find best nucleus (minimum mismatch among unmatched)
                        available_indices = mismatch_row.indices[available_mask]
                        available_mismatches = mismatch_row.data[available_mask]
                        best_local_idx = np.argmin(available_mismatches)
                        j_idx = available_indices[best_local_idx]
                        best_mismatch_fraction = available_mismatches[best_local_idx]

                        # Still need to call get_matched_cells once for trimming
                        whole_cell_best, nucleus_best, _ = get_matched_cells(
                            cell_coords[i],
                            cell_membrane_coords[i],
                            nucleus_coords[j_idx],
                            mismatch_repair=1,
                        )

                        if whole_cell_best.size > 0 and nucleus_best.size > 0:
                            i_ind = i
                            j_ind = j_idx
                        else:
                            whole_cell_best = []
            else:
                # CPU fallback path: Original loop with get_matched_cells
                for j in nucleus_search_num:
                    # Use label→index mapping
                    if j != 0 and j in nucleus_label_to_idx:
                        j_idx = nucleus_label_to_idx[j]

                        if (j_idx not in nucleus_matched_indices) and (
                            i not in cell_matched_indices
                        ):
                            whole_cell, nucleus, mismatch_fraction = get_matched_cells(
                                cell_coords[i],
                                cell_membrane_coords[i],
                                nucleus_coords[j_idx],
                                mismatch_repair=1,
                            )

                            if whole_cell.size > 0 and nucleus.size > 0:
                                if mismatch_fraction < best_mismatch_fraction:
                                    best_mismatch_fraction = mismatch_fraction
                                    whole_cell_best = whole_cell
                                    nucleus_best = nucleus
                                    i_ind = i
                                    j_ind = j_idx

            if best_mismatch_fraction < 1 and best_mismatch_fraction > 0:
                repaired_num += 1

            if len(whole_cell_best) > 0:
                cell_matched_list.append(whole_cell_best)
                nucleus_matched_list.append(nucleus_best)
                cell_matched_indices.add(i_ind)
                nucleus_matched_indices.add(j_ind)

            # Update progress bar
            pbar.update(1)

            # Update progress bar stats every 1000 cells
            if i % 1000 == 0 and i > 0:
                pbar.set_postfix({
                    'matched': len(cell_matched_indices),
                    'repaired': repaired_num,
                })

    if repaired_num > 0:
        logger.info(f"{repaired_num} cells repaired out of {total_cells} cells")

    return cell_matched_list, nucleus_matched_list, repaired_num


def _repair_masks_cpu_fallback(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    metadata: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """CPU fallback using the original repair_masks implementation.

    Args:
        cell_mask: Cell mask (H, W)
        nucleus_mask: Nucleus mask (H, W)
        metadata: Metadata dict to update

    Returns:
        Tuple of (cell_repaired, nucleus_repaired, metadata)
    """
    from aegle.repair_masks import get_matched_masks, get_mask

    logger.info("Using CPU repair implementation...")
    t0 = time.time()

    # Add first dim to match original implementation
    cell_mask_3d = np.expand_dims(cell_mask, axis=0)
    nucleus_mask_3d = np.expand_dims(nucleus_mask, axis=0)

    # Run CPU repair
    cell_matched_mask, nucleus_matched_mask, cell_outside_nucleus_mask = (
        get_matched_masks(cell_mask_3d, nucleus_mask_3d)
    )

    # Squeeze back to 2D
    cell_matched_mask = np.squeeze(cell_matched_mask)
    nucleus_matched_mask = np.squeeze(nucleus_matched_mask)

    # Update metadata
    metadata["total_time"] = time.time() - t0
    metadata["n_matched_cells"] = len(np.unique(cell_matched_mask)) - 1
    metadata["n_matched_nuclei"] = len(np.unique(nucleus_matched_mask)) - 1
    metadata["cpu_time"] = metadata["total_time"]

    logger.info(
        f"CPU repair completed in {metadata['total_time']:.2f}s "
        f"({metadata['n_matched_cells']}/{metadata['n_cells_input']} cells matched)"
    )

    return cell_matched_mask.astype(np.uint32), nucleus_matched_mask.astype(np.uint32), metadata


def get_matched_masks_gpu(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    use_gpu: bool = True,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated version of get_matched_masks from repair_masks.py.

    This is a drop-in replacement for the CPU version with the same interface.

    Args:
        cell_mask: Cell segmentation mask (with batch dim: (1, H, W) or 2D: (H, W))
        nucleus_mask: Nucleus segmentation mask (with batch dim: (1, H, W) or 2D: (H, W))
        use_gpu: Whether to use GPU acceleration
        batch_size: Optional batch size for overlap computation

    Returns:
        Tuple of:
            - cell_matched_mask: Matched cell mask (same shape as input)
            - nucleus_matched_mask: Matched nucleus mask (same shape as input)
            - cell_outside_nucleus_mask: Difference mask (same shape as input)
    """
    # Store original shape
    original_shape = cell_mask.shape

    # Ensure 2D for GPU processing
    cell_mask_2d = np.asarray(cell_mask).squeeze()
    nucleus_mask_2d = np.asarray(nucleus_mask).squeeze()

    # Run GPU repair
    cell_matched_2d, nucleus_matched_2d, metadata = repair_masks_gpu(
        cell_mask_2d,
        nucleus_mask_2d,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    # Compute outside mask
    cell_outside_nucleus_2d = cell_matched_2d.astype(np.int32) - nucleus_matched_2d.astype(np.int32)
    cell_outside_nucleus_2d = np.clip(cell_outside_nucleus_2d, 0, None).astype(np.uint32)

    # Restore original shape if needed
    if len(original_shape) == 3:
        cell_matched_mask = np.expand_dims(cell_matched_2d, axis=0)
        nucleus_matched_mask = np.expand_dims(nucleus_matched_2d, axis=0)
        cell_outside_nucleus_mask = np.expand_dims(cell_outside_nucleus_2d, axis=0)
    else:
        cell_matched_mask = cell_matched_2d
        nucleus_matched_mask = nucleus_matched_2d
        cell_outside_nucleus_mask = cell_outside_nucleus_2d

    return cell_matched_mask, nucleus_matched_mask, cell_outside_nucleus_mask
