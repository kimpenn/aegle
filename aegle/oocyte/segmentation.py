"""Local intensity-based segmentation of one raw UCHL1 patch."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .config import OocyteDetectionConfig
from .models import LocalSegmentationResult, SegmentationMetrics


def _threshold_value(image: np.ndarray, method: str) -> float:
    if method == "triangle":
        return float(filters.threshold_triangle(image))
    if method == "yen":
        return float(filters.threshold_yen(image))
    raise ValueError(f"unsupported threshold method: {method}")


def _select_component(
    labeled_mask: np.ndarray,
    intensity_image: np.ndarray,
    center_y: int,
    center_x: int,
):
    props = measure.regionprops(labeled_mask, intensity_image=intensity_image)
    if not props:
        raise ValueError("no connected components found after thresholding")

    center_label = int(labeled_mask[center_y, center_x])
    if center_label > 0:
        prop = next(region for region in props if region.label == center_label)
        return labeled_mask == prop.label, prop, "center_component"

    scored = []
    for prop in props:
        distance = float(
            np.hypot(prop.centroid[1] - center_x, prop.centroid[0] - center_y)
        )
        scored.append((float(prop.area) / (1.0 + distance), prop))
    _, prop = max(scored, key=lambda item: item[0])
    return labeled_mask == prop.label, prop, "distance_weighted_fallback"


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter**2))


def _annulus_values(
    image: np.ndarray,
    annulus_inner_px: int,
    annulus_outer_px: int,
) -> Tuple[np.ndarray, int, int]:
    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    annulus = image[
        (distance >= annulus_inner_px)
        & (distance <= annulus_outer_px)
    ]
    if annulus.size == 0:
        raise ValueError("local background annulus does not intersect the patch")
    return annulus, center_y, center_x


def _segment_oocyte_patch(
    patch: np.ndarray,
    config: OocyteDetectionConfig,
    *,
    annulus_inner_px: int | None = None,
    annulus_outer_px: int | None = None,
    annulus_floor_percentile: float | None = None,
) -> Tuple[np.ndarray, LocalSegmentationResult]:

    patch_array = np.asarray(patch)
    if patch_array.ndim != 2 or min(patch_array.shape) < 3:
        raise ValueError("UCHL1 patch must be a non-empty two-dimensional image")
    if not np.issubdtype(patch_array.dtype, np.number):
        raise TypeError("UCHL1 patch must have a numeric dtype")
    if not np.isfinite(patch_array).all():
        raise ValueError("UCHL1 patch contains non-finite values")

    local = config.local
    inner_px = local.annulus_inner_px if annulus_inner_px is None else annulus_inner_px
    outer_px = local.annulus_outer_px if annulus_outer_px is None else annulus_outer_px
    if not 0 < inner_px < outer_px:
        raise ValueError("annulus radii must satisfy 0 < inner < outer")
    smooth = ndi.gaussian_filter(
        patch_array.astype(np.float32),
        sigma=local.gaussian_sigma,
    )
    annulus, center_y, center_x = _annulus_values(smooth, inner_px, outer_px)
    base_threshold = _threshold_value(smooth, local.threshold_method)
    floor_percentile = (
        local.annulus_floor_percentile
        if annulus_floor_percentile is None
        else annulus_floor_percentile
    )
    annulus_floor = float(np.percentile(annulus, floor_percentile)) * (
        local.annulus_floor_multiplier
    )
    threshold = max(base_threshold, annulus_floor)

    binary = smooth >= threshold
    binary = morphology.remove_small_objects(
        binary,
        min_size=local.min_component_size_px,
    )
    if local.closing_radius_px > 0:
        binary = ndi.binary_closing(
            binary,
            structure=morphology.disk(local.closing_radius_px),
        )
    binary = ndi.binary_fill_holes(binary)

    labeled = measure.label(binary)
    mask, prop, selection_mode = _select_component(
        labeled,
        smooth,
        center_y,
        center_x,
    )
    pixel_size_um = config.pixel_size_um
    metrics = SegmentationMetrics(
        threshold_method=local.threshold_method,
        base_threshold=base_threshold,
        annulus_floor=annulus_floor,
        threshold=threshold,
        selection_mode=selection_mode,
        area_px=int(prop.area),
        equivalent_diameter_um=float(prop.equivalent_diameter_area * pixel_size_um),
        major_axis_um=float(prop.axis_major_length * pixel_size_um),
        minor_axis_um=float(prop.axis_minor_length * pixel_size_um),
        eccentricity=float(prop.eccentricity),
        solidity=float(prop.solidity),
        circularity=_circularity(prop.area, prop.perimeter),
        centroid_y_px=float(prop.centroid[0]),
        centroid_x_px=float(prop.centroid[1]),
        centroid_offset_px=float(
            np.hypot(prop.centroid[1] - center_x, prop.centroid[0] - center_y)
        ),
        mean_intensity=float(prop.mean_intensity),
        max_intensity=float(prop.max_intensity),
    )
    result = LocalSegmentationResult(
        mask=np.asarray(mask, dtype=np.bool_),
        metrics=metrics,
    )
    return smooth, result


def _segment_oocyte_patch_components(
    patch: np.ndarray,
    config: OocyteDetectionConfig,
    *,
    annulus_inner_px: int,
    annulus_outer_px: int,
    annulus_floor_percentile: float,
) -> Tuple[np.ndarray, List[LocalSegmentationResult]]:
    """Segment every thresholded component for secondary crowded-field discovery."""

    patch_array = np.asarray(patch)
    if patch_array.ndim != 2 or min(patch_array.shape) < 3:
        raise ValueError("UCHL1 patch must be a non-empty two-dimensional image")
    if not np.issubdtype(patch_array.dtype, np.number):
        raise TypeError("UCHL1 patch must have a numeric dtype")
    if not np.isfinite(patch_array).all():
        raise ValueError("UCHL1 patch contains non-finite values")
    if not 0 < annulus_inner_px < annulus_outer_px:
        raise ValueError("annulus radii must satisfy 0 < inner < outer")

    local = config.local
    smooth = ndi.gaussian_filter(
        patch_array.astype(np.float32),
        sigma=local.gaussian_sigma,
    )
    annulus, center_y, center_x = _annulus_values(
        smooth,
        annulus_inner_px,
        annulus_outer_px,
    )
    base_threshold = _threshold_value(smooth, local.threshold_method)
    annulus_floor = float(np.percentile(annulus, annulus_floor_percentile)) * (
        local.annulus_floor_multiplier
    )
    threshold = max(base_threshold, annulus_floor)
    binary = morphology.remove_small_objects(
        smooth >= threshold,
        min_size=local.min_component_size_px,
    )
    if local.closing_radius_px > 0:
        binary = ndi.binary_closing(
            binary,
            structure=morphology.disk(local.closing_radius_px),
        )
    labeled = measure.label(ndi.binary_fill_holes(binary))
    results: List[LocalSegmentationResult] = []
    for prop in measure.regionprops(labeled, intensity_image=smooth):
        mask = np.asarray(labeled == prop.label, dtype=np.bool_)
        metrics = SegmentationMetrics(
            threshold_method=local.threshold_method,
            base_threshold=base_threshold,
            annulus_floor=annulus_floor,
            threshold=threshold,
            selection_mode="all_components",
            area_px=int(prop.area),
            equivalent_diameter_um=float(
                prop.equivalent_diameter_area * config.pixel_size_um
            ),
            major_axis_um=float(prop.axis_major_length * config.pixel_size_um),
            minor_axis_um=float(prop.axis_minor_length * config.pixel_size_um),
            eccentricity=float(prop.eccentricity),
            solidity=float(prop.solidity),
            circularity=_circularity(prop.area, prop.perimeter),
            centroid_y_px=float(prop.centroid[0]),
            centroid_x_px=float(prop.centroid[1]),
            centroid_offset_px=float(
                np.hypot(prop.centroid[1] - center_x, prop.centroid[0] - center_y)
            ),
            mean_intensity=float(prop.mean_intensity),
            max_intensity=float(prop.max_intensity),
        )
        results.append(LocalSegmentationResult(mask=mask, metrics=metrics))
    return smooth, results


def segment_oocyte_patch(
    patch: np.ndarray,
    config: OocyteDetectionConfig,
) -> LocalSegmentationResult:
    """Segment the oocyte associated with the center of a raw UCHL1 patch."""

    _, result = _segment_oocyte_patch(patch, config)
    return result
