import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import xmltodict
from skimage.filters import threshold_mean  # , threshold_otsu
from skimage.morphology import area_closing, closing, disk
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC

from aegle.codex_patches import CodexPatches
from aegle.evaluation_func import *


def run_seg_evaluation(
    codex_patches: CodexPatches,
    config: dict,
    args: Optional[dict] = None,
):
    original_seg_res_batch = codex_patches.original_seg_res_batch
    repaired_seg_res_batch = codex_patches.repaired_seg_res_batch
    patches_metadata_df = codex_patches.get_patches_metadata()
    idx = patches_metadata_df["is_infomative"] == True

    image_ndarray = codex_patches.extracted_channel_patches[idx]
    # reshape image_ndarray from batch, w, h, c to batch, c, w, h
    image_ndarray = image_ndarray.transpose(0, 3, 1, 2)
    # add an extra dimsion at axis 2
    image_ndarray = np.expand_dims(image_ndarray, axis=2)
    res_list = []
    for idx, repaired_seg_res in enumerate(repaired_seg_res_batch):
        original_seg_res_single = original_seg_res_batch[idx]

        logging.info(f"Evaluating segmentations for Patch: {idx}")
        img_dict_single = {
            "name": f"img_{idx}",
            "data": image_ndarray[idx : idx + 1, :, :, :, :],
        }
        cell_matched_mask = repaired_seg_res["cell_matched_mask"]
        nuclear_matched_mask = repaired_seg_res["nuclear_matched_mask"]
        cell_outside_nucleus_mask = repaired_seg_res["cell_outside_nucleus_mask"]
        pixel_size = config["data"]["image_mpp"] * 1000
        res = evaluate_seg_single(
            img_dict_single,
            original_seg_res_single,
            cell_matched_mask,
            nuclear_matched_mask,
            cell_outside_nucleus_mask,
            unit="nanometer",
            pixelsizex=pixel_size,
            pixelsizey=pixel_size,
        )
        res_list.append(res)
        logging.info(f"QualityScore: {res['QualityScore']}")

    codex_patches.seg_evaluation_metrics = res_list


def evaluate_seg_single(
    img,
    seg_res,
    cell_matched_mask,
    nuclear_matched_mask,
    cell_outside_nucleus_mask,
    unit="nanometer",
    pixelsizex=377.5,
    pixelsizey=377.5,
):
    cell_mask = seg_res["cell"]
    nucleus_mask = seg_res["nucleus"]

    metric_mask = np.expand_dims(cell_matched_mask, 0)
    metric_mask = np.vstack((metric_mask, np.expand_dims(nuclear_matched_mask, 0)))
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
        "Nucleus (including nuclear membrane)",
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
