# repair_mask.py
import gc
from deepcell.applications import Mesmer
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from scipy.sparse import csr_matrix

from aegle.codex_patches import CodexPatches
from aegle.visualization import make_outline_overlay
import logging


def repair_masks_batch(seg_res_batch):
    res_list = []
    for idx, seg_res in enumerate(seg_res_batch):
        logging.info(f"Repairing masks for batch index: {idx}")
        cell_mask = seg_res["cell"]
        nucleus_mask = seg_res["nucleus"]
        # Add first dim to the masks to match to implementation of repair_masks_single
        cell_mask = np.expand_dims(cell_mask, axis=0)
        nucleus_mask = np.expand_dims(nucleus_mask, axis=0)

        # run core mask repair function
        res = repair_masks_single(cell_mask, nucleus_mask)
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
    # debug_dir = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/debug"
    cell_membrane_mask = get_boundary(cell_mask)
    # cell_coords = get_indices_sparse(cell_mask)[1:]
    # nucleus_coords = get_indices_sparse(nucleus_mask)[1:]
    cell_coords = get_indices_pandas(cell_mask)
    if cell_coords.index[0] == 0:
        cell_coords = cell_coords.drop(0)
    else:
        logging.info(f"cell_coords.index[0]: {cell_coords.index[0]}")

    nucleus_coords = get_indices_pandas(nucleus_mask)
    if nucleus_coords.index[0] == 0:
        nucleus_coords = nucleus_coords.drop(0)
    else:
        logging.info(f"nucleus_coords.index[0]: {nucleus_coords.index[0]}")

    cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]
    cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
    cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))
    nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
    # logging.info(f"cell_coords: {len(cell_coords)}")
    # logging.info(f"nucleus_coords: {len(nucleus_coords)}")
    # logging.info(f"cell_membrane_coords: {len(cell_membrane_coords)}")

    cell_matched_index_list = []
    nucleus_matched_index_list = []
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
    for i in range(len(cell_coords)):
        if len(cell_coords[i]) != 0:
            current_cell_coords = cell_coords[i]
            nucleus_search_num = np.unique(
                list(map(lambda x: nucleus_mask[tuple(x)], current_cell_coords))
            )
            # logging.info(f"nucleus_search_num: {nucleus_search_num}")
            best_mismatch_fraction = 1
            whole_cell_best = []
            for j in nucleus_search_num:
                # logging.info(f"i: {i}; j: {j}")
                if j != 0:
                    if (j - 1 not in nucleus_matched_index_list) and (
                        i not in cell_matched_index_list
                    ):
                        whole_cell, nucleus, mismatch_fraction = get_matched_cells(
                            cell_coords[i],
                            cell_membrane_coords[i],
                            nucleus_coords[j - 1],
                            mismatch_repair=1,
                        )
                        if type(whole_cell) != bool:
                            if mismatch_fraction < best_mismatch_fraction:
                                best_mismatch_fraction = mismatch_fraction
                                whole_cell_best = whole_cell
                                nucleus_best = nucleus
                                i_ind = i
                                j_ind = j - 1
            if best_mismatch_fraction < 1 and best_mismatch_fraction > 0:
                repaired_num += 1

            if len(whole_cell_best) > 0:
                cell_matched_list.append(whole_cell_best)
                nucleus_matched_list.append(nucleus_best)
                cell_matched_index_list.append(i_ind)
                nucleus_matched_index_list.append(j_ind)
            else:
                logging.debug(f"Skipped cell#{str(i)}")

    if repaired_num > 0:
        logging.info(f"{repaired_num} cells repaired out of {len(cell_coords)} cells")

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


def get_indices_pandas(data):
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix(
        (cols, (data.ravel(), cols)), shape=(np.int64(data.max() + 1), data.size)
    )


def get_indices_sparse(data):
    data = data.astype(np.uint64)
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def get_matched_cells(cell_arr, cell_membrane_arr, nucleus_arr, mismatch_repair):
    a = set((tuple(i) for i in cell_arr))
    b = set((tuple(i) for i in cell_membrane_arr))
    c = set((tuple(i) for i in nucleus_arr))
    d = a - b
    mismatch_pixel_num = len(list(c - d))
    mismatch_fraction = len(list(c - d)) / len(list(c))
    if not mismatch_repair:
        if mismatch_pixel_num == 0:
            return np.array(list(a)), np.array(list(c)), 0
        else:
            return False, False, False
    else:
        if mismatch_pixel_num < len(c):
            return np.array(list(a)), np.array(list(d & c)), mismatch_fraction
        else:
            return False, False, False


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
