# evaluation.py
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from skimage.filters import threshold_mean  # , threshold_otsu
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
import concurrent.futures
from functools import partial

from aegle.codex_patches import CodexPatches
from aegle.evaluation_func import *


def _process_single_patch(local_idx, patch_tensor, repaired_seg_res, original_seg_res_single, patch_metadata_index, config):
    """Evaluate a single patch and return the metrics dictionary."""
    cell_mask = repaired_seg_res["cell_matched_mask"]
    unique_cell_ids = np.unique(cell_mask)
    cell_count = len(unique_cell_ids) - (1 if 0 in unique_cell_ids else 0)

    patch_label = patch_metadata_index if patch_metadata_index is not None else local_idx

    if cell_count < 20:
        logging.warning(
            f"Skipping evaluation for patch {patch_label} with only {cell_count} cells (minimum 20 required)"
        )
        return {"QualityScore": float("nan")}

    try:
        img_dict_single = {
            "name": f"img_{patch_label}",
            "data": patch_tensor,
        }
        cell_matched_mask = repaired_seg_res["cell_matched_mask"]
        nucleus_matched_mask = repaired_seg_res["nucleus_matched_mask"]
        cell_outside_nucleus_mask = repaired_seg_res["cell_outside_nucleus_mask"]
        pixel_size = config["data"]["image_mpp"] * 1000
        res = evaluate_seg_single(
            img_dict_single,
            original_seg_res_single,
            cell_matched_mask,
            nucleus_matched_mask,
            cell_outside_nucleus_mask,
            unit="nanometer",
            pixelsizex=pixel_size,
            pixelsizey=pixel_size,
        )
        logging.info(
            f"Evaluated segmentation for patch {patch_label} (local index {local_idx}), cell count: {cell_count}, QualityScore: {res['QualityScore']}"
        )
        return res
    except Exception as e:
        logging.warning(f"Error evaluating patch {patch_label}: {str(e)}")
        return {"QualityScore": float("nan")}


def _prepare_patch_tensor(patch_array):
    """Convert a raw patch array (H, W, C) to the evaluation tensor shape (1, C, 1, H, W)."""
    patch_array = np.asarray(patch_array, dtype=np.float32)
    if patch_array.ndim != 3:
        raise ValueError(f"Expected patch array with 3 dimensions (H, W, C), got shape {patch_array.shape}")
    patch_tensor = np.transpose(patch_array, (2, 0, 1))  # C, H, W
    patch_tensor = np.expand_dims(patch_tensor, axis=1)  # C, 1, H, W
    patch_tensor = np.expand_dims(patch_tensor, axis=0)  # 1, C, 1, H, W
    return patch_tensor


def _load_patch_tensor(codex_patches: CodexPatches, patch_index: int):
    """Fetch a patch (in-memory or disk-based) and convert it to evaluation tensor format."""
    if patch_index is None:
        raise ValueError("patch_index must be provided for evaluation")

    if codex_patches.is_using_disk_based_patches():
        patch_array = codex_patches.load_patch_from_disk(patch_index, "extracted")
    else:
        patch_array = codex_patches.extracted_channel_patches[patch_index]

    return _prepare_patch_tensor(patch_array)


def run_seg_evaluation(
    codex_patches: CodexPatches,
    config: dict,
    args: Optional[dict] = None,
):
    original_seg_res_batch = codex_patches.original_seg_res_batch or []
    repaired_seg_res_batch = codex_patches.repaired_seg_res_batch or []

    if not repaired_seg_res_batch:
        logging.info("No repaired segmentation results available - skipping evaluation step.")
        codex_patches.seg_evaluation_metrics = []
        return

    patches_metadata_df = codex_patches.get_patches_metadata()
    informative_indices = []
    if patches_metadata_df is not None and "is_informative" in patches_metadata_df:
        informative_mask = patches_metadata_df["is_informative"] == True
        try:
            informative_indices = np.flatnonzero(informative_mask.to_numpy())
        except AttributeError:
            informative_indices = np.flatnonzero(np.asarray(informative_mask))

    if len(informative_indices) == 0:
        informative_indices = np.arange(len(repaired_seg_res_batch))

    if len(informative_indices) != len(repaired_seg_res_batch):
        logging.warning(
            "Length mismatch between informative patch indices (%d) and repaired segmentation batch (%d)",
            len(informative_indices),
            len(repaired_seg_res_batch),
        )
        informative_indices = informative_indices[: len(repaired_seg_res_batch)]

    logging.info(
        "Segmentation evaluation will process %d patch(es).", len(repaired_seg_res_batch)
    )

    res_list = [{"QualityScore": float("nan")} for _ in range(len(repaired_seg_res_batch))]

    max_workers = min(2, len(repaired_seg_res_batch))
    if len(repaired_seg_res_batch) > 200:
        logging.info(
            "Large patch count detected (%d) - running evaluation sequentially to reduce memory pressure.",
            len(repaired_seg_res_batch),
        )
        max_workers = 1

    if max_workers <= 1:
        logging.info("Starting sequential segmentation evaluation.")
        for local_idx in range(len(repaired_seg_res_batch)):
            patch_metadata_index = (
                informative_indices[local_idx] if local_idx < len(informative_indices) else None
            )
            try:
                patch_tensor = _load_patch_tensor(codex_patches, patch_metadata_index)
            except Exception as exc:
                logging.error(f"Failed to load patch {patch_metadata_index} for evaluation: {exc}")
                res_list[local_idx] = {"QualityScore": float("nan")}
                continue

            orig_single = (
                original_seg_res_batch[local_idx]
                if len(original_seg_res_batch) > local_idx
                else None
            )
            if orig_single is None:
                logging.error(
                    f"Missing original segmentation result for patch {patch_metadata_index}. Skipping evaluation."
                )
                res_list[local_idx] = {"QualityScore": float("nan")}
                continue

            res_list[local_idx] = _process_single_patch(
                local_idx,
                patch_tensor,
                repaired_seg_res_batch[local_idx],
                orig_single,
                patch_metadata_index,
                config,
            )
    else:
        logging.info("Starting parallel segmentation evaluation with %d worker(s)", max_workers)
        process_func = partial(_process_single_patch, config=config)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for local_idx in range(len(repaired_seg_res_batch)):
                patch_metadata_index = (
                    informative_indices[local_idx] if local_idx < len(informative_indices) else None
                )
                try:
                    patch_tensor = _load_patch_tensor(codex_patches, patch_metadata_index)
                except Exception as exc:
                    logging.error(f"Failed to load patch {patch_metadata_index} for evaluation: {exc}")
                    res_list[local_idx] = {"QualityScore": float("nan")}
                    continue

                orig_single = (
                    original_seg_res_batch[local_idx]
                    if len(original_seg_res_batch) > local_idx
                    else None
                )
                if orig_single is None:
                    logging.error(
                        f"Missing original segmentation result for patch {patch_metadata_index}. Skipping evaluation."
                    )
                    res_list[local_idx] = {"QualityScore": float("nan")}
                    continue

                future = executor.submit(
                    process_func,
                    local_idx,
                    patch_tensor,
                    repaired_seg_res_batch[local_idx],
                    orig_single,
                    patch_metadata_index,
                )
                future_to_idx[future] = local_idx

            for future in concurrent.futures.as_completed(future_to_idx):
                local_idx = future_to_idx[future]
                try:
                    res_list[local_idx] = future.result()
                except Exception as exc:
                    patch_metadata_index = (
                        informative_indices[local_idx] if local_idx < len(informative_indices) else None
                    )
                    patch_label = patch_metadata_index if patch_metadata_index is not None else local_idx
                    logging.error(f"Patch {patch_label} generated an exception during evaluation: {exc}")
                    res_list[local_idx] = {"QualityScore": float("nan")}

    logging.info("Segmentation evaluation completed")
    codex_patches.seg_evaluation_metrics = res_list


def evaluate_seg_single(
    img,
    seg_res,
    cell_matched_mask,
    nucleus_matched_mask,
    cell_outside_nucleus_mask,
    unit="nanometer",
    pixelsizex=377.5,
    pixelsizey=377.5,
):
    cell_mask = seg_res["cell"]
    nucleus_mask = seg_res["nucleus"]

    metric_mask = np.expand_dims(cell_matched_mask, 0)
    metric_mask = np.vstack((metric_mask, np.expand_dims(nucleus_matched_mask, 0)))
    metric_mask = np.vstack((metric_mask, np.expand_dims(cell_outside_nucleus_mask, 0)))
    # separate image foreground background

    thresholding_channels = range(img["data"].shape[1])
    seg_channel_provided = False
    img_thresholded = sum(
        thresholding(np.squeeze(img["data"][0, c, 0, :, :]))
        for c in thresholding_channels
    )

    # fg_bg_image = Image.fromarray(img_thresholded.astype(np.uint8) * 255, mode="L").convert("1")
    # fg_bg_image.save(output_dir / (img["name"] + "img_thresholded.png"))
    disksizes = (1, 2, 20, 10)  # these were used in CellSegmentationEvaluator v1.4
    # disksizes = (1, 2, 10, 3) #these were used by 3DCellComposer v1.1
    areasizes = (20000, 1000)  # these were used in CellSegmentationEvaluator v1.4
    # areasizes = (5000, 1000) #these were used by 3DCellComposer v1.1
    img_binary = foreground_separation(img_thresholded, disksizes, areasizes)
    img_binary = np.sign(img_binary)
    background_pixel_num = np.argwhere(img_binary == 0).shape[0]
    fraction_background = background_pixel_num / (
        img_binary.shape[0] * img_binary.shape[1]
    )
    # np.savetxt(output_dir / f"{img["name"]}_img_binary.txt.gz", img_binary)
    # fg_bg_image = Image.fromarray(img_binary.astype(np.uint8) * 255, mode="L").convert("1")
    # fg_bg_image.save(output_dir / (img["name"] + "img_binary.png"))

    # set mask channel names
    channel_names = [
        "Matched Cell",
        "Nucleus (including nucleus membrane)",
        "Cell Not Including Nucleus (cell membrane plus cytoplasm)",
    ]
    metrics = {}
    for channel in range(metric_mask.shape[0]):
        current_mask = metric_mask[channel]
        mask_binary = np.sign(current_mask)
        metrics[channel_names[channel]] = {}
        if channel_names[channel] == "Matched Cell":

            # matched_fraction = 1.0
            matched_fraction = get_matched_fraction(
                "nonrepaired_matched_mask",
                cell_mask,
                cell_matched_mask,
                nucleus_mask,
            )
            try:
                units, pixel_size = get_pixel_area(img["img"])
            except:
                reg = UnitRegistry()
                reg.define("cell = []")
                units = reg(unit)
                sizes = [pixelsizex * units, pixelsizey * units]
                # print(sizes)
                units = reg
                # pixel_size = math.prod(sizes)
                pixel_size = sizes[0] * sizes[1]
            pixel_num = mask_binary.shape[0] * mask_binary.shape[1]
            total_area = pixel_size * pixel_num
            # calculate number of cell per 100 squared micron
            cell_num = units("cell") * len(np.unique(current_mask)) - 1
            cells_per_area = cell_num / total_area
            units.define("hundred_square_micron = micrometer ** 2 * 100")
            cell_num_normalized = cells_per_area.to("cell / hundred_square_micron")
            # calculate the standard deviation of cell size

            _, _, cell_size_std = cell_size_uniformity(current_mask)

            # get coverage metrics
            foreground_fraction, background_fraction, mask_foreground_fraction = (
                fraction(img_binary, mask_binary)
            )

            img_channels = np.squeeze(img["data"][0, :, 0, :, :])

            foreground_CV, foreground_PCA = foreground_uniformity(
                img_binary, mask_binary, img_channels
            )
            # background_CV, background_PCA = background_uniformity(img_binary, img_channels)
            metrics[channel_names[channel]][
                "NumberOfCellsPer100SquareMicrons"
            ] = cell_num_normalized.magnitude
            metrics[channel_names[channel]][
                "FractionOfForegroundOccupiedByCells"
            ] = foreground_fraction
            metrics[channel_names[channel]]["1-FractionOfBackgroundOccupiedByCells"] = (
                1 - background_fraction
            )
            metrics[channel_names[channel]][
                "FractionOfCellMaskInForeground"
            ] = mask_foreground_fraction
            metrics[channel_names[channel]]["1/(ln(StandardDeviationOfCellSize)+1)"] = (
                1 / (np.log(cell_size_std) + 1)
            )
            metrics[channel_names[channel]][
                "FractionOfMatchedCellsAndNuclei"
            ] = matched_fraction
            metrics[channel_names[channel]]["1/(AvgCVForegroundOutsideCells+1)"] = 1 / (
                foreground_CV + 1
            )
            metrics[channel_names[channel]][
                "FractionOfFirstPCForegroundOutsideCells"
            ] = foreground_PCA

            # get cell type labels
            cell_type_labels = cell_type(current_mask, img_channels)
        else:
            img_channels = np.squeeze(img["data"][0, :, 0, :, :])
            # get cell uniformity
            cell_CV, cell_fraction, cell_silhouette = cell_uniformity(
                current_mask, img_channels, cell_type_labels
            )
            avg_cell_CV = np.average(cell_CV)
            avg_cell_fraction = np.average(cell_fraction)
            avg_cell_silhouette = np.average(cell_silhouette)

            metrics[channel_names[channel]][
                "1/(AvgOfWeightedAvgCVMeanCellIntensitiesOver1~10NumberOfClusters+1)"
            ] = 1 / (avg_cell_CV + 1)
            metrics[channel_names[channel]][
                "AvgOfWeightedAvgFractionOfFirstPCMeanCellIntensitiesOver1~10NumberOfClusters"
            ] = avg_cell_fraction
            metrics[channel_names[channel]][
                "AvgSilhouetteOver2~10NumberOfClusters"
            ] = avg_cell_silhouette

    metrics_flat = np.expand_dims(flatten_dict(metrics), 0)
    PCAmodel = getPCAmodel("2Dv1.5")
    try:
        quality_score = get_quality_score(metrics_flat, PCAmodel)
    except:
        quality_score = float("nan")
    metrics["QualityScore"] = quality_score

    # return metrics, fraction_background, 1 / (background_CV + 1), background_PCA
    return metrics
