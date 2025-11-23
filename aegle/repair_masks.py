# repair_mask.py
import gc
from deepcell.applications import Mesmer
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from scipy.sparse import csr_matrix
import time
from tqdm import tqdm

from aegle.codex_patches import CodexPatches
from aegle.visualization import make_outline_overlay


def get_indices_numpy(data: np.ndarray) -> Dict[int, Tuple]:
    """Extract coordinates for each label using pure NumPy.

    This function replaces the pandas-based implementation (get_indices_pandas)
    to eliminate DataFrame overhead and improve performance. Returns the same
    data structure: a dictionary mapping label → coordinate tuples.

    Args:
        data: Labeled mask array (can have batch dimension or be 2D/3D)

    Returns:
        Dictionary mapping label → tuple of coordinate arrays
        - Keys: integer labels found in the mask
        - Values: tuples of coordinate arrays (one array per dimension)
          matching the format returned by np.unravel_index()

    Example:
        For a 2D mask, returns {1: (array([y1, y2, ...]), array([x1, x2, ...])), ...}
        For a 3D mask, returns {1: (array([z1, z2, ...]), array([y1, y2, ...]), array([x1, x2, ...])), ...}
    """
    arr = np.asarray(data)
    if arr.size == 0:
        return {}

    # Flatten and sort indices by label
    flat = arr.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]

    # Get unique labels and split points
    labels, counts = np.unique(sorted_labels, return_counts=True)
    split_points = np.cumsum(counts[:-1])

    # Split the ordered indices into groups by label
    groups = np.split(order, split_points) if split_points.size else [order]

    # Convert flat indices to coordinate tuples for each label
    coords_dict = {}
    for label, group in zip(labels, groups):
        coords_dict[int(label)] = np.unravel_index(group, arr.shape)

    return coords_dict


def _compute_labeled_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a boundary mask where pixels retain their original labels."""
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.uint32, copy=False)
    boundary_bool = find_boundaries(mask, mode="inner")
    boundary_mask = np.zeros_like(mask, dtype=np.uint32)
    boundary_mask[boundary_bool] = mask[boundary_bool]
    return boundary_mask


def repair_masks_batch(
    seg_res_batch,
    use_gpu=False,
    gpu_batch_size=None,
    use_bincount_overlap=True,
    fallback_to_cpu=True,
):
    """Repair masks for a batch of segmentation results with progress tracking.

    Args:
        seg_res_batch: List of segmentation results, each containing 'cell' and 'nucleus' masks
        use_gpu: Whether to use GPU acceleration (default: False for backward compatibility)
        gpu_batch_size: Batch size for GPU overlap computation (None = auto-detect)
        use_bincount_overlap: Use Phase 5c bincount approach (default: True, 400-540x speedup)
        fallback_to_cpu: Automatically fallback to CPU on GPU errors (default: True)

    Returns:
        List of repaired mask dictionaries, each containing repair_metadata if GPU was used
    """
    res_list = []
    total_batches = len(seg_res_batch)

    for idx, seg_res in enumerate(seg_res_batch):
        logging.info("=" * 60)
        logging.info(f"Processing batch {idx + 1}/{total_batches}")
        logging.info("=" * 60)

        cell_mask = seg_res["cell"]
        nucleus_mask = seg_res["nucleus"]

        # Get cell and nucleus counts for logging
        n_cells = len(np.unique(cell_mask)) - 1  # Exclude background (0)
        n_nuclei = len(np.unique(nucleus_mask)) - 1

        logging.info(f"Batch {idx} - Cells: {n_cells:,}, Nuclei: {n_nuclei:,}")

        # Route to GPU or CPU implementation
        batch_start = time.time()

        if use_gpu:
            # Use GPU implementation
            from aegle.repair_masks_gpu import repair_masks_gpu

            # Ensure 2D masks for GPU (no batch dimension)
            cell_mask_2d = np.asarray(cell_mask).squeeze()
            nucleus_mask_2d = np.asarray(nucleus_mask).squeeze()

            # Run GPU repair
            cell_matched_2d, nucleus_matched_2d, repair_metadata = repair_masks_gpu(
                cell_mask_2d,
                nucleus_mask_2d,
                use_gpu=True,
                batch_size=gpu_batch_size,
                use_bincount_overlap=use_bincount_overlap,
                fallback_to_cpu=fallback_to_cpu,
            )

            # Compute outside mask
            cell_outside_nucleus = cell_matched_2d.astype(np.int32) - nucleus_matched_2d.astype(np.int32)
            cell_outside_nucleus = np.clip(cell_outside_nucleus, 0, None).astype(np.uint32)

            # Expand dims to match CPU format (batch dimension)
            cell_matched_mask = np.expand_dims(cell_matched_2d, axis=0)
            nucleus_matched_mask = np.expand_dims(nucleus_matched_2d, axis=0)
            cell_outside_nucleus_mask = np.expand_dims(cell_outside_nucleus, axis=0)

            # Compute matched fraction
            n_matched = repair_metadata.get("n_matched_cells", 0)
            matched_fraction = n_matched / n_cells if n_cells > 0 else 0.0

            # Build result dict
            res = {
                "cell_matched_mask": cell_matched_mask,
                "nucleus_matched_mask": nucleus_matched_mask,
                "cell_outside_nucleus_mask": cell_outside_nucleus_mask,
                "matched_fraction": matched_fraction,
                "repair_metadata": repair_metadata,
            }

        else:
            # Use CPU implementation (existing code)
            # Add first dim to the masks to match implementation of repair_masks_single
            cell_mask = np.expand_dims(cell_mask, axis=0)
            nucleus_mask = np.expand_dims(nucleus_mask, axis=0)

            res = repair_masks_single(cell_mask, nucleus_mask)

        batch_elapsed = time.time() - batch_start

        # Log batch completion statistics
        matched_cells = len(np.unique(res['cell_matched_mask'])) - 1
        logging.info(f"Batch {idx} complete in {batch_elapsed / 3600:.2f} hours ({batch_elapsed / 60:.1f} minutes)")
        logging.info(f"Matched: {matched_cells:,}/{n_cells:,} cells ({matched_cells / n_cells * 100 if n_cells > 0 else 0:.1f}%)")
        if batch_elapsed > 0:
            logging.info(f"Rate: {n_cells / batch_elapsed:.1f} cells/sec")

        # Log GPU metadata if available
        if use_gpu and "repair_metadata" in res:
            metadata = res["repair_metadata"]
            if metadata.get("gpu_used"):
                logging.info(f"GPU repair used successfully")
                if "speedup" in metadata:
                    logging.info(f"Speedup vs CPU: {metadata['speedup']:.2f}x")
            elif metadata.get("fallback_to_cpu"):
                logging.warning(f"GPU repair fell back to CPU: {metadata.get('fallback_reason', 'unknown')}")

        res_list.append(res)

    return res_list


def repair_masks_single(cell_mask, nucleus_mask):
    """
    cell_mask.shape: (1, 1440, 1920)
    nucleus_mask.shape: (1, 1440, 1920)
    """
    cell_matched_mask, nucleus_matched_mask, cell_outside_nucleus_mask = (
        get_matched_masks(cell_mask, nucleus_mask)
    )
    cell_matched_mask = np.squeeze(cell_matched_mask)
    nucleus_matched_mask = np.squeeze(nucleus_matched_mask)
    cell_outside_nucleus_mask = np.squeeze(cell_outside_nucleus_mask)

    matched_fraction = get_matched_fraction(
        "nonrepaired_matched_mask",
        np.squeeze(cell_mask),
        cell_matched_mask,
        np.squeeze(nucleus_mask),
    )
    logging.info(f"cell_matched_mask.shape: {cell_matched_mask.shape}")
    logging.info(f"nucleus_matched_mask.shape: {nucleus_matched_mask.shape}")
    logging.info(f"cell_outside_nucleus_mask.shape: {cell_outside_nucleus_mask.shape}")
    logging.info(f"matched_fraction: {matched_fraction}")

    # Calculate statistics for the table
    matching_stats = calculate_matching_statistics(
        np.squeeze(cell_mask),
        np.squeeze(nucleus_mask),
        cell_matched_mask,
        nucleus_matched_mask,
    )

    return {
        "cell_matched_mask": cell_matched_mask.astype(np.uint32),
        "nucleus_matched_mask": nucleus_matched_mask.astype(np.uint32),
        "cell_outside_nucleus_mask": cell_outside_nucleus_mask.astype(np.uint32),
        "cell_matched_boundary": _compute_labeled_boundary(cell_matched_mask),
        "nucleus_matched_boundary": _compute_labeled_boundary(nucleus_matched_mask),
        "matched_fraction": matched_fraction,
        "matching_stats": matching_stats,
    }


def get_matched_fraction(repair_mask, mask, cell_matched_mask, nucleus_mask):
    if repair_mask == "repaired_matched_mask":
        fraction_matched_cells = 1
    elif repair_mask == "nonrepaired_matched_mask":
        # print(mask.shape,cell_matched_mask.shape,nucleus_mask.shape)
        matched_cell_num = len(np.unique(cell_matched_mask))
        total_cell_num = len(np.unique(mask))
        total_nuclei_num = len(np.unique(nucleus_mask))
        mismatched_cell_num = total_cell_num - matched_cell_num
        mismatched_nuclei_num = total_nuclei_num - matched_cell_num
        # print(matched_cell_num, total_cell_num, total_nuclei_num, mismatched_cell_num, mismatched_nuclei_num)
        fraction_matched_cells = matched_cell_num / (
            mismatched_cell_num + mismatched_nuclei_num + matched_cell_num
        )
    return fraction_matched_cells


def get_matched_masks(cell_mask, nucleus_mask):
    """Match cell and nucleus masks and compute matched/unmatched masks.

    This function handles non-contiguous label IDs correctly using a label→index
    mapping. Labels do not need to be contiguous (e.g., [1, 5, 7, 23] is valid).

    Args:
        cell_mask: Cell segmentation mask (H, W) with integer labels
        nucleus_mask: Nucleus segmentation mask (H, W) with integer labels

    Returns:
        tuple: (cell_matched_mask, nucleus_matched_mask, cell_outside_nucleus_mask)
    """
    # debug_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/debug"
    cell_membrane_mask = get_boundary(cell_mask)

    # Extract coordinates using NumPy (replaces pandas-based implementation)
    cell_coords_dict = get_indices_numpy(cell_mask)
    nucleus_coords_dict = get_indices_numpy(nucleus_mask)

    # Remove background label (0) and convert to list format
    # The dict uses labels as keys, so we need to create a list indexed by (label - 1)
    cell_labels = sorted([label for label in cell_coords_dict.keys() if label > 0])
    nucleus_labels = sorted([label for label in nucleus_coords_dict.keys() if label > 0])

    if cell_labels and cell_labels[0] != 1:
        logging.info(f"cell_coords first label: {cell_labels[0]}")
    if nucleus_labels and nucleus_labels[0] != 1:
        logging.info(f"nucleus_coords first label: {nucleus_labels[0]}")

    # Convert to list format: cell_coords[i] contains coordinates for label cell_labels[i]
    # Transpose to get shape (N, ndim) where N is number of pixels
    cell_coords = [np.array(cell_coords_dict[label]).T for label in cell_labels]
    nucleus_coords = [np.array(nucleus_coords_dict[label]).T for label in nucleus_labels]

    # Create mapping from nucleus label ID to list index for O(1) lookup
    # This is CRITICAL for handling non-contiguous label IDs (e.g., 1, 5, 7, 23)
    # Without this mapping, code that used nucleus_coords[j - 1] would fail with
    # IndexError when labels have gaps (e.g., j=23 but list has only 4 items)
    nucleus_label_to_idx = {label: idx for idx, label in enumerate(nucleus_labels)}

    # Get membrane coordinates using sparse method (unchanged)
    cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]
    cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
    # logging.info(f"cell_coords: {len(cell_coords)}")
    # logging.info(f"nucleus_coords: {len(nucleus_coords)}")
    # logging.info(f"cell_membrane_coords: {len(cell_membrane_coords)}")

    cell_matched_indices = set()
    nucleus_matched_indices = set()
    cell_matched_list = []
    nucleus_matched_list = []

    # # Save cell coords and nucleus coords for debugging by pickle
    # file_name = debug_dir + "/cell_coords.pkl"
    # with open(file_name, 'wb') as f:
    # 	pickle.dump(cell_coords, f)
    # file_name = debug_dir + "/nucleus_coords.pkl"
    # with open(file_name, 'wb') as f:
    # 	pickle.dump(nucleus_coords, f)

    repaired_num = 0
    total_cells = len(cell_coords)

    # Add progress bar for cell matching loop
    with tqdm(
        total=total_cells,
        desc="Matching cells to nuclei",
        unit="cell",
        mininterval=1.0,
        dynamic_ncols=True,
    ) as pbar:
        for i in range(total_cells):
            if len(cell_coords[i]) != 0:
                current_cell_coords = cell_coords[i]
                # Optimized: Use set instead of np.unique() to avoid O(n log n) sorting
                # We only need unique nucleus IDs, not sorted order
                nucleus_ids_set = set(nucleus_mask[tuple(current_cell_coords.T)])
                # Remove background (0) and convert to array
                nucleus_ids_set.discard(0)
                nucleus_search_num = np.array(list(nucleus_ids_set), dtype=nucleus_mask.dtype)
                # logging.info(f"nucleus_search_num: {nucleus_search_num}")
                best_mismatch_fraction = 1
                whole_cell_best = []
                for j in nucleus_search_num:
                    # logging.info(f"i: {i}; j: {j}")
                    # Use label→index mapping to handle non-contiguous label IDs
                    # Check that j exists in mapping (it should, but be defensive)
                    if j != 0 and j in nucleus_label_to_idx:
                        j_idx = nucleus_label_to_idx[j]  # Get correct index for label j
                        if (j_idx not in nucleus_matched_indices) and (
                            i not in cell_matched_indices
                        ):
                            whole_cell, nucleus, mismatch_fraction = get_matched_cells(
                                cell_coords[i],
                                cell_membrane_coords[i],
                                nucleus_coords[j_idx],  # Use mapped index, not j-1
                                mismatch_repair=1,
                            )
                            # This part was type(whole_cell) != bool
                            # Ref: https://github.com/murphygroup/CellSegmentationEvaluator/blob/6def33dd172ad9074bd856399535a5deea3e3fd6/full_pipeline/pipeline/segmentation/get_cellular_compartments.py#L176
                            # We changed it to arrary size to make sure whole_cell and nucleus are alwasy arrays
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
                else:
                    # logging.debug(f"Skipped cell#{str(i)}")
                    pass

            # Update progress bar
            pbar.update(1)

            # Update progress bar stats every 1000 cells
            if i % 1000 == 0 and i > 0:
                pbar.set_postfix({
                    'matched': len(cell_matched_indices),
                    'repaired': repaired_num,
                })

    if repaired_num > 0:
        logging.info(f"{repaired_num} cells repaired out of {total_cells} cells")

    cell_matched_mask = get_mask(cell_matched_list, cell_mask.shape)
    nucleus_matched_mask = get_mask(nucleus_matched_list, nucleus_mask.shape)
    cell_outside_nucleus_mask = cell_matched_mask - nucleus_matched_mask
    return cell_matched_mask, nucleus_matched_mask, cell_outside_nucleus_mask


def get_mask(cell_list, mask_shape):
    mask = np.zeros(mask_shape)
    for cell_num in range(len(cell_list)):
        mask[tuple(cell_list[cell_num].T)] = cell_num + 1
    return mask


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def get_boundary(mask):
    mask_boundary = find_boundaries(mask, mode="inner")
    mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
    return mask_boundary_indexed

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix(
        (cols, (data.ravel(), cols)), shape=(np.int64(data.max() + 1), data.size)
    )


def get_indices_sparse(data):
    data = data.astype(np.uint64)
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Match a single cell mask with a candidate nucleus mask.

    The inputs are coordinate arrays (z, y, x) belonging to an individual cell,
    its membrane, and a nucleus candidate. We first measure how much of the
    nucleus sits inside the cell interior (cell minus membrane). Then we decide
    whether the nucleus should be kept, possibly trimming it when
    ``mismatch_repair`` is enabled.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - full cell coordinates (possibly unchanged from the input)
            - repaired nucleus coordinates (may be empty if rejected)
            - mismatch_fraction measuring nucleus pixels outside the cell interior
    """

    cell_set = set(map(tuple, cell_arr))
    membrane_set = set(map(tuple, cell_membrane_arr))
    nucleus_set = set(map(tuple, nuclear_arr))

    if not nucleus_set:
        # No nucleus pixels exist; treat as a complete mismatch.
        return np.array([]), np.array([]), 1.0

    cell_interior = cell_set - membrane_set
    unmatched_pixels = nucleus_set - cell_interior
    mismatch_fraction = len(unmatched_pixels) / len(nucleus_set)

    if not mismatch_repair:
        # Without repair we only accept perfect overlap.
        if mismatch_fraction == 0:
            return np.array(list(cell_set)), np.array(list(nucleus_set)), 0.0
        return np.array([]), np.array([]), mismatch_fraction

    if mismatch_fraction >= 1.0 or not cell_interior:
        # Completely disjoint or cell has no interior -> reject.
        return np.array([]), np.array([]), 1.0

    matched_nucleus = nucleus_set & cell_interior
    if not matched_nucleus:
        # Safety guard: partial overlap should have produced interior pixels.
        return np.array([]), np.array([]), 1.0

    return (
        np.array(list(cell_set)),
        np.array(list(matched_nucleus)),
        mismatch_fraction,
    )


def calculate_matching_statistics(
    cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask
):
    """
    Calculate statistics about matched and unmatched cells and nuclei.

    Args:
        cell_mask: Original cell mask
        nucleus_mask: Original nucleus mask
        cell_matched_mask: Mask of matched cells
        nucleus_matched_mask: Mask of matched nuclei

    Returns:
        Dictionary containing statistics for the table
    """
    # Get counts (excluding background value 0)
    total_cells = len(np.unique(cell_mask)) - 1
    total_nuclei = len(np.unique(nucleus_mask)) - 1
    matched_cells = len(np.unique(cell_matched_mask)) - 1
    matched_nuclei = len(np.unique(nucleus_matched_mask)) - 1

    # Calculate unmatched counts
    unmatched_cells = total_cells - matched_cells
    unmatched_nuclei = total_nuclei - matched_nuclei

    # Calculate percentages
    cell_matched_pct = (matched_cells / total_cells * 100) if total_cells > 0 else 0
    cell_unmatched_pct = (unmatched_cells / total_cells * 100) if total_cells > 0 else 0
    nuclei_matched_pct = (
        (matched_nuclei / total_nuclei * 100) if total_nuclei > 0 else 0
    )
    nuclei_unmatched_pct = (
        (unmatched_nuclei / total_nuclei * 100) if total_nuclei > 0 else 0
    )

    # Assemble statistics dictionary
    stats = {
        "nucleus": {
            "total": (total_nuclei, 100.0),
            "matched": (matched_nuclei, nuclei_matched_pct),
            "unmatched": (unmatched_nuclei, nuclei_unmatched_pct),
        },
        "whole_cell": {
            "total": (total_cells, 100.0),
            "matched": (matched_cells, cell_matched_pct),
            "unmatched": (unmatched_cells, cell_unmatched_pct),
        },
    }

    return stats
