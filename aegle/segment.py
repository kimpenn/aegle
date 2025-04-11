# segment.py
import gc
from deepcell.applications import Mesmer
from tensorflow.keras.models import load_model
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from scipy.sparse import csr_matrix

from aegle.codex_patches import CodexPatches
from aegle.visualization import make_outline_overlay
from aegle.repair_masks import repair_masks_batch

Image = np.ndarray
import pickle


def run_cell_segmentation(
    codex_patches: CodexPatches,
    config: dict,
    args: Optional[dict] = None,
):
    """
    Main function to run the cell segmentation module.

    Args:
        codex_patches (CodexPatches): CodexPatches object containing patches and metadata.
        config (dict): Configuration parameters.
        args (Optional[dict]): Command-line arguments.

    Returns:
        Tuple[List[dict], float, List[dict]]: Matched segmentation results, fraction matched, and original segmentation results.
    """

    patches_ndarray = codex_patches.get_patches()
    logging.info(f"Number of patches: {len(patches_ndarray)}")
    logging.info(f"dtype of patches: {patches_ndarray.dtype}")
    patches_metadata_df = codex_patches.get_patches_metadata()

    # Select only valid patches
    idx = patches_metadata_df["is_infomative"] == True
    valid_patches = patches_ndarray[idx]
    if valid_patches.size == 0:
        logging.warning("No valid patches available for segmentation.")
        return None
    codex_patches.valid_patches = valid_patches

    seg_res_batch = segment(valid_patches, config)
    # file_name = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/explore-eval-scores-dev/exp-10/seg_res_batch.pickle"
    # with open(file_name, "wb") as f:
    #     pickle.dump(seg_res_batch, f)

    repaired_seg_res_batch = repair_masks_batch(seg_res_batch)
    matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
    # add matched fraction to metadata dataframe with the same index as the patches
    patches_metadata_df.loc[idx, "matched_fraction"] = matched_fraction_list

    codex_patches.set_seg_res(repaired_seg_res_batch, seg_res_batch)
    codex_patches.set_metadata(patches_metadata_df)
    codex_patches.save_seg_res()


def segment(
    valid_patches: np.ndarray,
    config: dict,
) -> List[dict]:
    """
    Segment cells and nuclei in the input image patches.

    Args:
        patches_ndarray (np.ndarray): Image patches to segment.
        config (dict): Configuration parameters.
        args: Command-line arguments.

    Returns:
        segmentation_output (List[dict]): A list where each element is a dictionary containing segmentation results for a patch:
            - 'cell': np.ndarray, the cell mask.
            - 'nucleus': np.ndarray, the nucleus mask.
            - 'cell_boundary': np.ndarray, the cell boundary mask.
            - 'nucleus_boundary': np.ndarray, the nucleus boundary mask.
    """

    # Load the segmentation model
    model = _load_segmentation_model(config)

    # Perform segmentation
    image_mpp = config.get("data", {}).get("image_mpp", 0.5)
    segmentation_predictions = model.predict(
        valid_patches, image_mpp=image_mpp, compartment="both"
    )
    # is there any negative values in the segmentation predictions?
    if np.any(segmentation_predictions < 0):
        logging.warning("Negative values found in segmentation predictions.")
        # save segmentation_predictions to a file for debug
        # np.save("/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle/segmentation_predictions_debug.npy", segmentation_predictions)
    else:
        logging.info("No negative values found in segmentation predictions.")
    
    max_label = segmentation_predictions.max()
    logging.info(f"Max label value before casting: {max_label}")
    if max_label >= 2**32:
        raise ValueError("Segmentation label values exceed uint32 range. Consider using uint64 instead.")

    # transform segmentation_predictions from float32 to int32
    segmentation_predictions = segmentation_predictions.astype(np.uint32)
    logging.info("Segmentation completed successfully.")
    logging.info(f"=== dtype of segmentation predictions: {segmentation_predictions.dtype}")
    # reorganize the segmentation predictions
    cell_masks, nuc_masks = _separate_batch(segmentation_predictions)
    cell_boundaries = get_boundary(cell_masks)
    nuc_boundaries = get_boundary(nuc_masks)

    segmentation_output = _build_segmentation_output(
        cell_masks, nuc_masks, cell_boundaries, nuc_boundaries
    )

    # Force garbage collection to free memory
    gc.collect()

    return segmentation_output


# def get_matched_masks(
#     seg_res_batch: List[dict], do_mismatch_repair: bool
# ) -> Tuple[List[dict], float]:
#     """
#     Returns masks with matched cells and nuclei based on the segmentation output.

#     Args:
#         seg_res_batch (List[dict]): List of dictionaries containing segmentation results for each patch.
#         do_mismatch_repair (bool): Whether to apply mismatch repair during the matching process.

#     Returns:
#         Tuple[List[dict], float]: A tuple containing:
#             - matched_output (List[dict]): The updated segmentation output with matched masks for each patch.
#             - fraction_matched_cells (float): The fraction of matched cells across all patches.
#     """
#     matched_output = []
#     total_matched_cells = 0

#     # Iterate over each patch segmentation
#     for patch_segmentation in seg_res_batch:

#         whole_cell_mask = patch_segmentation["cell"]
#         nucleus_mask = patch_segmentation["nucleus"]
#         cell_membrane_mask = patch_segmentation["cell_boundary"]

#         # Extract coordinates for cell, nucleus, and cell membrane
#         cell_coords_dict = _get_mask_coordinates(whole_cell_mask)
#         nucleus_coords_dict = _get_mask_coordinates(nucleus_mask)
#         cell_membrane_coords_dict = _get_mask_coordinates(cell_membrane_mask)

#         # Perform matching between cells and nuclei
#         cell_matched_list, nucleus_matched_list = _match_cells_to_nuclei(
#             cell_coords_dict,
#             nucleus_coords_dict,
#             cell_membrane_coords_dict,
#             nucleus_mask,
#             do_mismatch_repair,
#         )

#         # Create new masks with matched cells and nuclei
#         cell_matched_mask = get_mask_from_labels(
#             cell_matched_list, whole_cell_mask.shape
#         )
#         nucleus_matched_mask = get_mask_from_labels(
#             nucleus_matched_list, nucleus_mask.shape
#         )
#         cell_membrane_matched_mask = get_boundary(cell_matched_mask)
#         nucleus_membrane_matched_mask = get_boundary(nucleus_matched_mask)

#         # Update the patch segmentation with the matched masks
#         matched_patch_segmentation = {
#             "cell": cell_matched_mask,
#             "nucleus": nucleus_matched_mask,
#             "cell_boundary": cell_membrane_matched_mask,
#             "nucleus_boundary": nucleus_membrane_matched_mask,
#         }
#         matched_output.append(matched_patch_segmentation)

#         # Calculate statistics for fraction of matched cells
#         matched_cell_num = len(np.unique(cell_matched_mask)) - 1
#         total_cell_num = len(np.unique(whole_cell_mask)) - 1
#         total_nuclei_num = len(np.unique(nucleus_mask)) - 1

#         mismatched_cell_num = total_cell_num - matched_cell_num
#         mismatched_nuclei_num = total_nuclei_num - matched_cell_num
#         denominator = mismatched_cell_num + mismatched_nuclei_num + total_matched_cells
#         fraction_matched_cells = (
#             matched_cell_num / denominator if denominator > 0 else 0
#         )
#         logging.info(f"Fraction of matched cells: {fraction_matched_cells}")

#     return matched_output, fraction_matched_cells


def _load_segmentation_model(config: dict) -> Mesmer:
    """
    Load the DeepCell Mesmer segmentation model and configure logging.

    Args:
        config (dict): Configuration parameters.

    Returns:
        Mesmer: Initialized DeepCell Mesmer application.
    """
    try:
        # Read `model_path` from the segmentation config
        model_path = config.get("segmentation", {}).get(
            "model_path",
            "/workspaces/codex-analysis/data/deepcell/MultiplexSegmentation",
        )
        keras_model = load_model(model_path)
        logging.info("Segmentation model loaded successfully.")
        # Initialize the Mesmer app with the loaded model
        model = Mesmer(model=keras_model)
        model.logger.setLevel(logging.DEBUG)
        logging.info(f"Training Resolution: {model.model_mpp} microns per pixel")
        return model

    except Exception as e:
        logging.error(f"Failed to load segmentation model: {e}", exc_info=True)
        raise


def _separate_batch(img_stack: np.ndarray) -> Tuple[List[Image], List[Image]]:
    """
    Separate cell and nucleus masks from the segmentation prediction batch.

    Args:
        img_stack (np.ndarray): Batch of segmentation predictions.

    Returns:
        Tuple[List[Image], List[Image]]: List of cell masks and nucleus masks.
    """
    cell_masks = [img_stack[i, :, :, 0] for i in range(img_stack.shape[0])]
    nuc_masks = [img_stack[i, :, :, 1] for i in range(img_stack.shape[0])]
    return cell_masks, nuc_masks


def get_boundary(masks: List[Image]) -> List[Image]:
    """
    Generate boundary masks for each segmented cell or nucleus.

    Args:
        masks (List[Image]): List of segmented masks.

    Returns:
        List[Image]: List of boundary masks.
    Description:
        DeepCell generates indentifiers for each cell.
        The mask is 2D array with each cell having a unique identifier.
        We convert the area mask to a boundary mask with the same cell identifier.
    """
    boundaries = []
    for mask in masks:
        # hubmap pipeline uses both default mode="thick" and "inner".
        # reference: https://github.com/hubmapconsortium/segmentations/blob/4f1f5e5aa9941274d3959932d43be15909c38177/bin/img_proc/match_masks.py#L61
        # mode="thick" means that the boundary is outside the mask
        # mode="inner" means that the boundary is inside the mask
        # I think "inner" is the correct mode since we do not want to add non-cell pixels to the boundary.
        # connectivity=1 means that the boundary is 1 pixel wide.
        mask_boundary = find_boundaries(mask, mode="inner", connectivity=1)
        boundaries.append(_get_indexed_mask(mask, mask_boundary))
    return boundaries


def _get_indexed_mask(mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """
    Convert a boundary mask into an indexed mask with the same cell identifiers.

    Args:
        mask (np.ndarray): Original mask with cell identifiers.
        boundary (np.ndarray): Binary boundary mask.

    Returns:
        np.ndarray: Indexed boundary mask.
    """
    boundary = boundary.astype(int)
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def _build_segmentation_output(
    cell_masks: List[Image],
    nuc_masks: List[Image],
    cell_boundaries: List[Image],
    nuc_boundaries: List[Image],
) -> List[dict]:
    """
    Build the segmentation output dictionary for each patch.

    Args:
        cell_masks (List[Image]): List of cell masks.
        nuc_masks (List[Image]): List of nucleus masks.
        cell_boundaries (List[Image]): List of cell boundary masks.
        nuc_boundaries (List[Image]): List of nucleus boundary masks.

    Returns:
        List[dict]: List of dictionaries containing segmentation results for each patch.
    """
    batch_size = len(cell_masks)
    segmentation_output = []
    for i in range(batch_size):
        img_set = {
            "cell": cell_masks[i],
            "nucleus": nuc_masks[i],
            "cell_boundary": cell_boundaries[i],
            "nucleus_boundary": nuc_boundaries[i],
        }
        segmentation_output.append(img_set)
    return segmentation_output


# def _get_mask_coordinates(mask: np.ndarray) -> Dict[int, np.ndarray]:
#     """
#     Get coordinates and labels for each unique object in the mask.
#     Returns:
#         Dict[int, np.ndarray]: Mapping from label to coordinates. np.ndarray is a list of (row, col) coordinates.
#     """
#     props = regionprops(mask)
#     coords_dict = {prop.label: prop.coords for prop in props}
#     return coords_dict


def get_mask_from_labels(
    cell_list: List[Tuple[int, np.ndarray]], shape: Tuple[int]
) -> np.ndarray:
    """
    Create a mask from a list of (label, cell coordinates), handling out-of-bounds coordinates.
    """
    mask = np.zeros(shape, dtype=int)
    max_row, max_col = shape

    for label, cell in cell_list:
        # Filter out out-of-bounds coordinates
        valid_coords = cell[
            (cell[:, 0] >= 0)
            & (cell[:, 0] < max_row)
            & (cell[:, 1] >= 0)
            & (cell[:, 1] < max_col)
        ]
        mask[valid_coords[:, 0], valid_coords[:, 1]] = label
    return mask


def _match_cells_to_nuclei(
    cell_coords: Dict[int, np.ndarray],
    nucleus_coords: Dict[int, np.ndarray],
    cell_membrane_coords: Dict[int, np.ndarray],
    nucleus_mask: np.ndarray,
    do_mismatch_repair: bool,
) -> Tuple[List[Tuple[int, np.ndarray]], List[Tuple[int, np.ndarray]]]:
    """
    Matches cells and nuclei based on coordinates.
    """
    cell_matched_list = []
    nucleus_matched_list = []
    cell_matched_labels = set()
    nucleus_matched_labels = set()

    for cell_label, cell_coord in cell_coords.items():
        if len(cell_coord) == 0:
            raise ValueError("Empty cell coordinates detected.")

        # The label of the cell membrane is the same as the cell
        cell_membrane_coord = cell_membrane_coords.get(cell_label, np.array([]))

        # Find candidate nuclei overlapping with the current cell
        nucleus_candidates = np.unique(nucleus_mask[tuple(cell_coord.T)])
        best_mismatch_fraction = 1
        best_match = None

        for nucleus_id in nucleus_candidates:
            if nucleus_id == 0 or nucleus_id in nucleus_matched_labels:
                continue

            current_nucleus_coords = nucleus_coords.get(nucleus_id)
            if current_nucleus_coords is None:
                continue

            match_result = get_matched_cells(
                cell_coord,
                cell_membrane_coord,
                current_nucleus_coords,
                mismatch_repair=do_mismatch_repair,
            )

            if match_result is not None:
                whole_cell, nucleus, mismatch_fraction = match_result
                if mismatch_fraction == 0:
                    # Perfect match found; add and break
                    cell_matched_list.append((cell_label, whole_cell))
                    nucleus_matched_list.append((nucleus_id, nucleus))
                    cell_matched_labels.add(cell_label)
                    nucleus_matched_labels.add(nucleus_id)
                    break
                elif mismatch_fraction < best_mismatch_fraction:
                    best_mismatch_fraction = mismatch_fraction
                    best_match = {
                        "cell": whole_cell,
                        "nucleus": nucleus,
                        "cell_label": cell_label,
                        "nucleus_label": nucleus_id,
                    }

        # If no perfect match was found, add the best match (if any)
        if best_match and cell_label not in cell_matched_labels:
            cell_matched_list.append((best_match["cell_label"], best_match["cell"]))
            nucleus_matched_list.append(
                (best_match["nucleus_label"], best_match["nucleus"])
            )
            cell_matched_labels.add(best_match["cell_label"])
            nucleus_matched_labels.add(best_match["nucleus_label"])

    return cell_matched_list, nucleus_matched_list


def get_matched_cells(
    cell_arr: np.ndarray,
    cell_membrane_arr: np.ndarray,
    nucleus_arr: np.ndarray,
    mismatch_repair: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Determine if a cell and nucleus match based on overlap, optionally performing mismatch repair.

    Args:
        cell_arr (np.ndarray): Coordinates of the cell.
        cell_membrane_arr (np.ndarray): Coordinates of the cell membrane.
        nucleus_arr (np.ndarray): Coordinates of the nucleus.
        mismatch_repair (bool): Whether to apply mismatch repair.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, float]]:
            - If a match is found, returns a tuple containing:
                - whole_cell (np.ndarray): Coordinates of the whole cell.
                - nucleus (np.ndarray): Coordinates of the nucleus.
                - mismatch_fraction (float): Fraction of mismatched pixels.
            - If no match is found, returns None.
    """
    cell_set = set(map(tuple, cell_arr))
    membrane_set = set(map(tuple, cell_membrane_arr))
    nucleus_set = set(map(tuple, nucleus_arr))

    cell_interior = cell_set - membrane_set
    mismatched_pixels = nucleus_set - cell_interior
    mismatch_fraction = len(mismatched_pixels) / len(nucleus_set) if nucleus_set else 1

    if not mismatch_repair:
        if len(mismatched_pixels) == 0:
            return (
                np.array(list(cell_set)),
                np.array(list(nucleus_set)),
                mismatch_fraction,
            )
        else:
            return None
    else:
        if len(mismatched_pixels) < len(nucleus_set):
            matched_nucleus = nucleus_set & cell_interior
            return (
                np.array(list(cell_set)),
                np.array(list(matched_nucleus)),
                mismatch_fraction,
            )
        else:
            return None


def visualize_cell_segmentation(
    data,
    seg_res: List[dict],
    config: dict,
    args: Optional[dict] = None,
):
    """
    Visualize the cell segmentation results.

    Args:
        matched_seg_res (List[dict]): List of dictionaries containing matched segmentation results for each patch.
        config (dict): Configuration parameters.
        args: Command-line arguments.

    Returns:
        None
    """

    make_outline_overlay(rgb_data, predictions)


# def get_matched_cells(
#     cell_arr: np.ndarray,
#     cell_membrane_arr: np.ndarray,
#     nucleus_arr: np.ndarray,
#     mismatch_repair: bool,
# ) -> Tuple[np.ndarray, np.ndarray, float]:
#     """
#     Determine if a cell and nucleus match based on overlap, optionally performing mismatch repair.

#     Args:
#         cell_arr (np.ndarray): Coordinates of the cell.
#         cell_membrane_arr (np.ndarray): Coordinates of the cell membrane.
#         nucleus_arr (np.ndarray): Coordinates of the nucleus.
#         mismatch_repair (bool): Whether to apply mismatch repair.

#     Returns:
#         Tuple[np.ndarray, np.ndarray, float]:
#             - whole_cell (np.ndarray): Coordinates of the whole cell.
#             - nucleus (np.ndarray): Coordinates of the nucleus.
#             - mismatch_fraction (float): Fraction of mismatched pixels.
#     """
#     cell_set = set(map(tuple, cell_arr))
#     membrane_set = set(map(tuple, cell_membrane_arr))
#     nucleus_set = set(map(tuple, nucleus_arr))

#     cell_interior = cell_set - membrane_set
#     mismatched_pixels = nucleus_set - cell_interior
#     mismatch_fraction = len(mismatched_pixels) / len(nucleus_set) if nucleus_set else 1

#     if not mismatch_repair:
#         if len(mismatched_pixels) == 0:
#             return (
#                 np.array(list(cell_set)),
#                 np.array(list(nucleus_set)),
#                 mismatch_fraction,
#             )
#         else:
#             return False, False, mismatch_fraction
#     else:
#         if len(mismatched_pixels) < len(nucleus_set):
#             matched_nucleus = nucleus_set & cell_interior
#             return (
#                 np.array(list(cell_set)),
#                 np.array(list(matched_nucleus)),
#                 mismatch_fraction,
#             )
#         else:
#             return False, False, mismatch_fraction


# def get_mask(cell_list: List[np.ndarray], shape: Tuple[int]) -> np.ndarray:
#     """
#     Create a mask from a list of cell coordinates, handling out-of-bounds coordinates.

#     Args:
#         cell_list (List[np.ndarray]): List of cell coordinates.
#         shape (Tuple[int]): Shape of the output mask.

#     Returns:
#         np.ndarray: Mask with cell labels.
#     """
#     mask = np.zeros(shape, dtype=int)
#     max_row, max_col = shape

#     for cell_num, cell in enumerate(cell_list):
#         # Filter out out-of-bounds coordinates
#         valid_coords = cell[
#             (cell[:, 0] >= 0)
#             & (cell[:, 0] < max_row)
#             & (cell[:, 1] >= 0)
#             & (cell[:, 1] < max_col)
#         ]
#         mask[valid_coords[:, 0], valid_coords[:, 1]] = cell_num + 1
#     return mask
