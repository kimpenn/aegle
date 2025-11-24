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
from aegle.memory_monitor import memory_monitor

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
    
    memory_monitor.log_memory_usage("Start of run_cell_segmentation")

    patches_info = codex_patches.get_patches()
    patches_metadata_df = codex_patches.get_patches_metadata()

    # Check if we're using disk-based patches
    if isinstance(patches_info, dict) and patches_info.get("disk_based", False):
        logging.info("Using disk-based patches for segmentation")
        
        # Select only valid patch indices
        idx = patches_metadata_df["is_informative"] == True
        valid_patch_indices = [i for i, is_valid in enumerate(idx) if is_valid]
        
        if not valid_patch_indices:
            logging.warning("No valid patches available for segmentation.")
            return None
            
        logging.info(f"Found {len(valid_patch_indices)} valid patches out of {len(patches_metadata_df)} total patches")
        
    else:
        # Traditional in-memory patches
        logging.info(f"Number of patches: {len(patches_info)}")
        logging.info(f"dtype of patches: {patches_info.dtype}")
        memory_monitor.log_array_info(patches_info, "patches_ndarray")
        
        # Select only valid patches
        idx = patches_metadata_df["is_informative"] == True
        valid_patches = patches_info[idx]
        if valid_patches.size == 0:
            logging.warning("No valid patches available for segmentation.")
            return None
        codex_patches.valid_patches = valid_patches
        
        memory_monitor.log_array_info(valid_patches, "valid_patches")
    memory_monitor.log_memory_usage("Before segmentation")

    # Get split mode to determine processing strategy
    patching_config = config.get("patching", {})
    split_mode = patching_config.get("split_mode", "patches")
    
    # Handle disk-based patches (always use sequential processing)
    if isinstance(patches_info, dict) and patches_info.get("disk_based", False):
        logging.info(f"Using sequential disk-based processing for split_mode='{split_mode}' to manage memory")
        logging.info(f"Processing {len(valid_patch_indices)} large patches individually from disk...")
        
        seg_res_batch = []
        for i, patch_idx in enumerate(valid_patch_indices):
            try:
                logging.info(f"Processing patch {i+1}/{len(valid_patch_indices)} (patch_index: {patch_idx}, split_mode: {split_mode})")
                memory_monitor.log_memory_usage(f"Before loading patch {i+1}")
                
                # Load patch from disk
                patch = codex_patches.load_patch_from_disk(patch_idx, "extracted")
                logging.info(f"Patch {i+1} shape: {patch.shape}")
                
                memory_monitor.log_memory_usage(f"After loading patch {i+1}")
                
                # Process single patch (reshape to batch format)
                single_patch = np.array([patch])  # Shape: (1, height, width, channels)
                
                # Clear original patch from memory before segmentation
                del patch
                import gc
                gc.collect()
                memory_monitor.log_memory_usage(f"After clearing loaded patch {i+1}")
                
                # Segment the single patch
                single_result = segment(single_patch, config)
                seg_res_batch.extend(single_result)
                
                # Clear segmentation input
                del single_patch
                gc.collect()
                
                logging.info(f"Patch {i+1}/{len(valid_patch_indices)} completed successfully")
                memory_monitor.log_memory_usage(f"After processing patch {i+1}")
                
            except Exception as e:
                error_msg = f"Failed to process patch {i+1}/{len(valid_patch_indices)} (patch_index: {patch_idx}) in split_mode '{split_mode}': {str(e)}"
                logging.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
                
        logging.info(f"Sequential disk-based processing completed successfully. Processed {len(seg_res_batch)} patches total.")
        
    elif split_mode in ["halves", "quarters", "full_image"]:
        # Sequential processing for memory management modes (in-memory patches)
        logging.info(f"Using sequential processing for split_mode='{split_mode}' to manage memory")
        logging.info(f"Processing {len(valid_patches)} large patches individually...")
        
        seg_res_batch = []
        for i, patch in enumerate(valid_patches):
            try:
                logging.info(f"Processing patch {i+1}/{len(valid_patches)} (split_mode: {split_mode})")
                memory_monitor.log_memory_usage(f"Before processing patch {i+1}")
                
                # Process single patch (reshape to batch format)
                single_patch = np.array([patch])  # Shape: (1, height, width, channels)
                logging.info(f"Patch {i+1} shape: {patch.shape}")
                
                # Segment the single patch
                single_result = segment(single_patch, config)
                seg_res_batch.extend(single_result)
                
                logging.info(f"Patch {i+1}/{len(valid_patches)} completed successfully")
                memory_monitor.log_memory_usage(f"After processing patch {i+1}")
                
                # Memory cleanup between patches
                import gc
                gc.collect()
                memory_monitor.log_memory_usage(f"After garbage collection for patch {i+1}")
                
            except Exception as e:
                error_msg = f"Failed to process patch {i+1}/{len(valid_patches)} in split_mode '{split_mode}': {str(e)}"
                logging.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
                
        logging.info(f"Sequential processing completed successfully. Processed {len(seg_res_batch)} patches total.")
        
    else:
        # Batch processing for small patches (existing behavior)
        logging.info(f"Using batch processing for split_mode='{split_mode}'")
        seg_res_batch = segment(valid_patches, config)
    
    memory_monitor.log_memory_usage("After segmentation, before repair")

    # Get repair configuration
    repair_config = config.get("segmentation", {}).get("repair", {})
    use_gpu = repair_config.get("use_gpu", False)
    gpu_batch_size = repair_config.get("gpu_batch_size", None)
    use_bincount_overlap = repair_config.get("use_bincount_overlap", True)
    mismatch_num_gpus = repair_config.get("mismatch_num_gpus", 1)
    fallback_to_cpu = repair_config.get("fallback_to_cpu", True)
    log_gpu_performance = repair_config.get("log_gpu_performance", False)

    # Log GPU repair configuration
    if use_gpu:
        logging.info("GPU-accelerated mask repair enabled")
        if use_bincount_overlap:
            logging.info("  Phase 5c bincount overlap: enabled (400-540x speedup expected)")
        else:
            logging.info("  Phase 4 sequential overlap: enabled (baseline)")
        if gpu_batch_size is not None:
            logging.info(f"  GPU batch size: {gpu_batch_size}")
        else:
            logging.info("  GPU batch size: auto-detect")
        if mismatch_num_gpus > 1:
            logging.info(f"  Multi-GPU mismatch: {mismatch_num_gpus} GPUs (1.87x speedup expected)")
        else:
            logging.info(f"  Single-GPU mismatch: enabled")
        logging.info(f"  Fallback to CPU: {'enabled' if fallback_to_cpu else 'disabled'}")
    else:
        logging.info("Using CPU mask repair (GPU disabled by config)")

    # Run mask repair with GPU support
    repaired_seg_res_batch = repair_masks_batch(
        seg_res_batch,
        use_gpu=use_gpu,
        gpu_batch_size=gpu_batch_size,
        use_bincount_overlap=use_bincount_overlap,
        mismatch_num_gpus=mismatch_num_gpus,
        fallback_to_cpu=fallback_to_cpu,
    )

    # Log GPU performance metrics if enabled
    if log_gpu_performance and use_gpu and repaired_seg_res_batch:
        # Aggregate metadata from all patches
        total_time = sum(res.get("repair_metadata", {}).get("total_time", 0)
                        for res in repaired_seg_res_batch)
        gpu_used_count = sum(1 for res in repaired_seg_res_batch
                            if res.get("repair_metadata", {}).get("gpu_used", False))

        if gpu_used_count > 0:
            logging.info(f"GPU repair performance summary:")
            logging.info(f"  - {gpu_used_count}/{len(repaired_seg_res_batch)} patches used GPU")
            logging.info(f"  - Total repair time: {total_time:.2f}s")

            # Log speedup if available
            for i, res in enumerate(repaired_seg_res_batch):
                metadata = res.get("repair_metadata", {})
                if metadata.get("speedup"):
                    logging.info(f"  - Patch {i}: {metadata['speedup']:.2f}x speedup")

    memory_monitor.log_memory_usage("After mask repair")

    # Add segmentation related metrics to the metadata
    matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
    
    # Handle metadata update for both disk-based and in-memory patches
    if isinstance(patches_info, dict) and patches_info.get("disk_based", False):
        # For disk-based patches, update only the valid patch indices
        for i, patch_idx in enumerate(valid_patch_indices):
            patches_metadata_df.loc[patch_idx, "matched_fraction"] = matched_fraction_list[i]
    else:
        # For in-memory patches, use the original logic
        patches_metadata_df.loc[idx, "matched_fraction"] = matched_fraction_list

    # Save everything
    codex_patches.set_seg_res(repaired_seg_res_batch, seg_res_batch)
    codex_patches.set_metadata(patches_metadata_df)
    
    memory_monitor.log_memory_usage("Before saving segmentation results")

    segmentation_cfg = config.get("segmentation", {})
    save_segmentation_pickle = segmentation_cfg.get("save_segmentation_pickle", True)
    if save_segmentation_pickle:
        codex_patches.save_seg_res()
    else:
        logging.info("Skipping segmentation pickle export (save_segmentation_pickle set to False)")
    codex_patches.save_metadata()

    if segmentation_cfg.get("save_segmentation_images", True):
        try:
            codex_patches.export_segmentation_masks(config, args)
        except Exception:
            logging.exception("Failed to export segmentation masks to OME-TIFF")
    
    memory_monitor.log_memory_usage("End of run_cell_segmentation")


def segment(
    valid_patches: np.ndarray,
    config: dict,
) -> List[dict]:
    """
    Segment cells and nuclei in the input image patches.

    Args:
        valid_patches (np.ndarray): Image patches to segment.
        config (dict): Configuration parameters.
        args: Command-line arguments.

    Returns:
        segmentation_output (List[dict]): A list where each element is a dictionary containing segmentation results for a patch:
            - 'cell': np.ndarray, the cell mask.
            - 'nucleus': np.ndarray, the nucleus mask.
            - 'cell_boundary': np.ndarray, the cell boundary mask.
            - 'nucleus_boundary': np.ndarray, the nucleus boundary mask.
    """

    memory_monitor.log_memory_usage("Start of segment function")
    memory_monitor.log_array_info(valid_patches, "valid_patches input")

    # Load the segmentation model
    model = _load_segmentation_model(config)
    
    memory_monitor.log_memory_usage("After loading segmentation model")

    # Perform segmentation
    image_mpp = config.get("data", {}).get("image_mpp", 0.5)
    logging.info(f"Starting model prediction with image_mpp={image_mpp}")
    
    segmentation_predictions = model.predict(
        valid_patches, image_mpp=image_mpp, compartment="both"
    )
    
    memory_monitor.log_memory_usage("After model.predict")
    memory_monitor.log_array_info(segmentation_predictions, "segmentation_predictions")
    
    # is there any negative values in the segmentation predictions?
    if np.any(segmentation_predictions < 0):
        logging.warning("Negative values found in segmentation predictions.")
    else:
        logging.info("No negative values found in segmentation predictions.")

    max_label = segmentation_predictions.max()
    logging.info(f"Max label value before casting: {max_label}")
    if max_label >= 2**32:
        raise ValueError(
            "Segmentation label values exceed uint32 range. Consider using uint64 instead."
        )

    # transform segmentation_predictions from float32 to int32
    segmentation_predictions = segmentation_predictions.astype(np.uint32)
    logging.info("Segmentation completed successfully.")
    logging.info(
        f"=== dtype of segmentation predictions: {segmentation_predictions.dtype}"
    )
    
    memory_monitor.log_memory_usage("After dtype conversion")
    memory_monitor.log_array_info(segmentation_predictions, "segmentation_predictions after conversion")
    
    # reorganize the segmentation predictions
    cell_masks, nuc_masks = _separate_batch(segmentation_predictions)
    
    memory_monitor.log_memory_usage("After separating masks")
    
    cell_boundaries = get_boundary(cell_masks)
    
    memory_monitor.log_memory_usage("After cell boundary generation")
    
    nuc_boundaries = get_boundary(nuc_masks)

    memory_monitor.log_memory_usage("After nucleus boundary generation")

    segmentation_output = _build_segmentation_output(
        cell_masks, nuc_masks, cell_boundaries, nuc_boundaries
    )

    memory_monitor.log_memory_usage("After building segmentation output")

    # Force garbage collection to free memory
    gc.collect()
    
    memory_monitor.log_memory_usage("End of segment function after gc.collect")

    return segmentation_output


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
        # Currently, the model weight is downloaded by
        # wget https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-7.tar.gz 
        # ref: https://github.com/hubmapconsortium/segmentations/blob/4f1f5e5aa9941274d3959932d43be15909c38177/Dockerfile#L63 
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
    # TODO: Fix this visualization function
    # make_outline_overlay(rgb_data, predictions)
    raise NotImplementedError("Visualization function not implemented.")


# def _match_cells_to_nuclei(
#     cell_coords: Dict[int, np.ndarray],
#     nucleus_coords: Dict[int, np.ndarray],
#     cell_membrane_coords: Dict[int, np.ndarray],
#     nucleus_mask: np.ndarray,
#     do_mismatch_repair: bool,
# ) -> Tuple[List[Tuple[int, np.ndarray]], List[Tuple[int, np.ndarray]]]:
#     """
#     Matches cells and nuclei based on coordinates.
#     """
#     cell_matched_list = []
#     nucleus_matched_list = []
#     cell_matched_labels = set()
#     nucleus_matched_labels = set()

#     for cell_label, cell_coord in cell_coords.items():
#         if len(cell_coord) == 0:
#             raise ValueError("Empty cell coordinates detected.")

#         # The label of the cell membrane is the same as the cell
#         cell_membrane_coord = cell_membrane_coords.get(cell_label, np.array([]))

#         # Find candidate nuclei overlapping with the current cell
#         nucleus_candidates = np.unique(nucleus_mask[tuple(cell_coord.T)])
#         best_mismatch_fraction = 1
#         best_match = None

#         for nucleus_id in nucleus_candidates:
#             if nucleus_id == 0 or nucleus_id in nucleus_matched_labels:
#                 continue

#             current_nucleus_coords = nucleus_coords.get(nucleus_id)
#             if current_nucleus_coords is None:
#                 continue

#             match_result = get_matched_cells(
#                 cell_coord,
#                 cell_membrane_coord,
#                 current_nucleus_coords,
#                 mismatch_repair=do_mismatch_repair,
#             )

#             if match_result is not None:
#                 whole_cell, nucleus, mismatch_fraction = match_result
#                 if mismatch_fraction == 0:
#                     # Perfect match found; add and break
#                     cell_matched_list.append((cell_label, whole_cell))
#                     nucleus_matched_list.append((nucleus_id, nucleus))
#                     cell_matched_labels.add(cell_label)
#                     nucleus_matched_labels.add(nucleus_id)
#                     break
#                 elif mismatch_fraction < best_mismatch_fraction:
#                     best_mismatch_fraction = mismatch_fraction
#                     best_match = {
#                         "cell": whole_cell,
#                         "nucleus": nucleus,
#                         "cell_label": cell_label,
#                         "nucleus_label": nucleus_id,
#                     }

#         # If no perfect match was found, add the best match (if any)
#         if best_match and cell_label not in cell_matched_labels:
#             cell_matched_list.append((best_match["cell_label"], best_match["cell"]))
#             nucleus_matched_list.append(
#                 (best_match["nucleus_label"], best_match["nucleus"])
#             )
#             cell_matched_labels.add(best_match["cell_label"])
#             nucleus_matched_labels.add(best_match["nucleus_label"])

#     return cell_matched_list, nucleus_matched_list
