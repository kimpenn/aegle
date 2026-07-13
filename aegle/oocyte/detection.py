"""Whole-slide raw-UCHL1 proposal generation and candidate detection."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology

from .config import (
    OOCYTE_IMPLEMENTATION_VERSION,
    CoarseDetectionConfig,
    OocyteDetectionConfig,
)
from .export import LabelExportResult, export_whole_slide_labels
from .io import (
    extract_cyx_channel_patch,
    extract_padded_patch,
    find_channel_index,
    save_candidate_mask,
)
from .models import ExtractedPatch, LocalSegmentationResult, ScoredCandidateMask
from .qc import render_spatial_overview
from .rescue import run_secondary_rescue
from .segmentation import _segment_oocyte_patch


REFINED_CANDIDATE_COLUMNS = (
    "detector_component_id",
    "source_detector_component_id",
    "coarse_component_label",
    "coarse_seed_kind",
    "coarse_seed_rank",
    "coarse_area_ds",
    "coarse_equivalent_diameter_um",
    "coarse_center_x",
    "coarse_center_y",
    "coarse_mean_ds",
    "coarse_max_ds",
    "coarse_eccentricity",
    "coarse_solidity",
    "coarse_bbox_x0",
    "coarse_bbox_y0",
    "coarse_bbox_x1",
    "coarse_bbox_y1",
    "coarse_blob_sigma_ds",
    "coarse_blob_score",
    "seed_center_x",
    "seed_center_y",
    "local_context_mode",
    "evaluation_window_radius_px",
    "evaluation_annulus_inner_px",
    "evaluation_annulus_outer_px",
    "center_x",
    "center_y",
    "component_centroid_x",
    "component_centroid_y",
    "component_centroid_shift_px",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "threshold_method",
    "threshold",
    "selection_mode",
    "local_area_px",
    "local_equivalent_diameter_um",
    "local_major_axis_um",
    "local_minor_axis_um",
    "local_eccentricity",
    "local_solidity",
    "local_circularity",
    "local_centroid_offset_px",
    "local_mean_intensity",
    "local_max_intensity",
    "annulus_p95",
    "annulus_p99",
    "mean_to_annulus_p99_ratio",
    "detector_score",
    "accepted_strict",
    "accepted_rescue",
    "acceptance_mode",
    "accepted",
    "seed_source",
    "seed_rank",
    "seed_shift_px",
    "local_reseed_iteration",
    "local_instance_rank",
    "local_instance_count_from_coarse",
)


@dataclass(frozen=True)
class CoarseDetectionResult:
    image_shape_yx: Tuple[int, int]
    downsampled: np.ndarray
    mask: np.ndarray
    contrast: np.ndarray
    thresholds: Dict[str, Any]
    candidates: pd.DataFrame


@dataclass(frozen=True)
class RefinedDetectionResult:
    candidates: pd.DataFrame
    candidate_masks: Dict[str, ScoredCandidateMask]
    rescue_diagnostics: pd.DataFrame | None = None


@dataclass(frozen=True)
class OocyteDetectionResult:
    sample_id: str
    image_path: Path
    image_shape_yx: Tuple[int, int]
    channel_index: int
    profile_name: str
    profile_fingerprint: str
    implementation_version: str
    coarse_candidates: pd.DataFrame
    candidates: pd.DataFrame
    thresholds: Dict[str, Any]
    runtime_seconds: Dict[str, float]
    artifact_paths: Dict[str, Path]


@dataclass(frozen=True)
class _SegmentationEvaluation:
    row: Dict[str, Any]
    patch: ExtractedPatch
    segmentation: LocalSegmentationResult


@dataclass(frozen=True)
class _EvaluatedCandidate:
    row: Dict[str, Any]
    evaluation: _SegmentationEvaluation


def _reduce_strip(strip: np.ndarray, factor: int) -> np.ndarray:
    pad_h = (-strip.shape[0]) % factor
    pad_w = (-strip.shape[1]) % factor
    if pad_h or pad_w:
        strip = np.pad(strip, ((0, pad_h), (0, pad_w)), mode="edge")
    return strip.reshape(
        strip.shape[0] // factor,
        factor,
        strip.shape[1] // factor,
        factor,
    ).mean(axis=(1, 3))


def build_downsampled_mean_map_from_array(
    channel_image: np.ndarray,
    coarse: CoarseDetectionConfig,
) -> np.ndarray:
    """Build the detector mean map from a two-dimensional channel image."""

    image = np.asarray(channel_image)
    if image.ndim != 2:
        raise ValueError("channel image must be two-dimensional")
    image_h, image_w = image.shape
    factor = coarse.downsample_factor
    downsampled = np.zeros(
        (
            (image_h + factor - 1) // factor,
            (image_w + factor - 1) // factor,
        ),
        dtype=np.float32,
    )
    out_y = 0
    for y0 in range(0, image_h, coarse.strip_height):
        y1 = min(image_h, y0 + coarse.strip_height)
        reduced = _reduce_strip(
            np.asarray(image[y0:y1, :], dtype=np.float32),
            factor,
        )
        downsampled[out_y : out_y + reduced.shape[0], : reduced.shape[1]] = reduced
        out_y += reduced.shape[0]
    return downsampled


def _build_downsampled_mean_map_from_zarr(
    array: zarr.Array,
    channel_index: int,
    coarse: CoarseDetectionConfig,
) -> np.ndarray:
    image_h = int(array.shape[1])
    image_w = int(array.shape[2])
    factor = coarse.downsample_factor
    downsampled = np.zeros(
        (
            (image_h + factor - 1) // factor,
            (image_w + factor - 1) // factor,
        ),
        dtype=np.float32,
    )
    out_y = 0
    for y0 in range(0, image_h, coarse.strip_height):
        y1 = min(image_h, y0 + coarse.strip_height)
        strip = np.asarray(array[channel_index, y0:y1, :], dtype=np.float32)
        reduced = _reduce_strip(strip, factor)
        downsampled[out_y : out_y + reduced.shape[0], : reduced.shape[1]] = reduced
        out_y += reduced.shape[0]
    return downsampled


def _deduplicate_coarse_rows(
    rows: List[Dict[str, Any]],
    merge_distance_px: float,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    seed_kind_priority = {
        "peak": 3,
        "blob_log": 2,
        "local_peak": 2,
        "centroid_fallback": 1,
        "global_peak": 0,
    }
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["coarse_max_ds"]),
            seed_kind_priority.get(str(row["coarse_seed_kind"]), 0),
            float(row.get("coarse_area_ds", 0)),
        ),
        reverse=True,
    )
    kept_rows: List[Dict[str, Any]] = []
    for row in ranked:
        duplicate = any(
            np.hypot(
                float(row["coarse_center_x"]) - float(kept["coarse_center_x"]),
                float(row["coarse_center_y"]) - float(kept["coarse_center_y"]),
            )
            <= merge_distance_px
            for kept in kept_rows
        )
        if not duplicate:
            kept_rows.append(row)
    return kept_rows


def detect_coarse_candidates(
    downsampled: np.ndarray,
    config: OocyteDetectionConfig,
) -> CoarseDetectionResult:
    """Generate v6 coarse proposals from a downsampled UCHL1 mean map."""

    image = np.asarray(downsampled, dtype=np.float32)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("downsampled UCHL1 map must be a non-empty 2D array")
    coarse = config.coarse
    factor = coarse.downsample_factor
    log_image = np.log1p(image)
    smooth = ndi.gaussian_filter(log_image, sigma=coarse.gaussian_sigma_small)
    background = ndi.gaussian_filter(log_image, sigma=coarse.gaussian_sigma_large)
    contrast = smooth - background

    intensity_threshold = max(
        float(np.percentile(image, coarse.intensity_percentile)),
        float(np.percentile(image, coarse.intensity_floor_percentile))
        * coarse.intensity_floor_multiplier,
    )
    positive_contrast = contrast[contrast > 0]
    triangle_contrast = (
        float(filters.threshold_triangle(positive_contrast))
        if positive_contrast.size
        else 0.0
    )
    contrast_threshold = max(
        float(np.percentile(contrast, coarse.contrast_percentile)),
        triangle_contrast,
    )
    initial_mask = (image >= intensity_threshold) & (contrast >= contrast_threshold)

    def seed_rows_for_region(region, labeled: np.ndarray) -> List[Dict[str, Any]]:
        diameter_um = float(
            region.equivalent_diameter_area * factor * config.pixel_size_um
        )
        if diameter_um < coarse.min_diameter_um or diameter_um > coarse.max_diameter_um:
            return []

        y0, x0, y1, x1 = region.bbox
        local_intensity = image[y0:y1, x0:x1]
        local_labels = labeled[y0:y1, x0:x1] == region.label
        peak_threshold = float(
            np.percentile(local_intensity[local_labels], coarse.peak_percentile)
        )
        peak_coords = feature.peak_local_max(
            local_intensity,
            labels=local_labels.astype(np.uint8),
            min_distance=coarse.peak_min_distance_ds,
            threshold_abs=peak_threshold,
            num_peaks=coarse.max_peaks_per_component,
            exclude_border=False,
        )
        seeds = []
        if peak_coords.size:
            peak_rows = []
            for peak_y, peak_x in peak_coords:
                if local_labels[int(peak_y), int(peak_x)]:
                    peak_rows.append(
                        (
                            float(local_intensity[int(peak_y), int(peak_x)]),
                            int(peak_y),
                            int(peak_x),
                        )
                    )
            peak_rows.sort(reverse=True)
            for peak_rank, (_, peak_y, peak_x) in enumerate(
                peak_rows[: coarse.max_peaks_per_component],
                start=1,
            ):
                seeds.append((peak_y + y0, peak_x + x0, "peak", peak_rank))
        else:
            seeds.append(
                (
                    int(round(region.centroid[0])),
                    int(round(region.centroid[1])),
                    "centroid_fallback",
                    1,
                )
            )

        rows = []
        for seed_y, seed_x, seed_kind, seed_rank in seeds:
            rows.append(
                {
                    "coarse_component_label": int(region.label),
                    "coarse_seed_kind": seed_kind,
                    "coarse_seed_rank": int(seed_rank),
                    "coarse_area_ds": int(region.area),
                    "coarse_equivalent_diameter_um": diameter_um,
                    "coarse_center_x": int(round(seed_x * factor + factor / 2)),
                    "coarse_center_y": int(round(seed_y * factor + factor / 2)),
                    "coarse_mean_ds": float(region.mean_intensity),
                    "coarse_max_ds": float(region.max_intensity),
                    "coarse_eccentricity": float(region.eccentricity),
                    "coarse_solidity": float(region.solidity),
                    "coarse_bbox_x0": int(region.bbox[1] * factor),
                    "coarse_bbox_y0": int(region.bbox[0] * factor),
                    "coarse_bbox_x1": int(region.bbox[3] * factor),
                    "coarse_bbox_y1": int(region.bbox[2] * factor),
                }
            )
        return rows

    def regions_from_mask(
        mask: np.ndarray,
        *,
        dilate_first: bool,
        min_area_ds: int,
    ):
        working = mask.copy()
        if dilate_first:
            working = ndi.binary_dilation(working, structure=morphology.disk(1))
        if coarse.closing_radius_ds > 0:
            working = ndi.binary_closing(
                working,
                structure=morphology.disk(coarse.closing_radius_ds),
            )
        working = ndi.binary_fill_holes(working)
        working = morphology.remove_small_objects(working, min_size=min_area_ds)
        labeled = measure.label(working)
        regions_by_label = {}
        rows = []
        for region in measure.regionprops(labeled, intensity_image=image):
            regions_by_label[int(region.label)] = region
            rows.extend(seed_rows_for_region(region, labeled))
        return working, labeled, regions_by_label, rows

    cleaned_mask, labeled, regions_by_label, rows = regions_from_mask(
        initial_mask,
        dilate_first=False,
        min_area_ds=coarse.min_component_area_ds,
    )
    cleanup_mode = "standard"
    if len(rows) < coarse.fallback_min_candidate_count:
        cleaned_mask, labeled, regions_by_label, rows = regions_from_mask(
            initial_mask,
            dilate_first=True,
            min_area_ds=max(8, coarse.min_component_area_ds // 2),
        )
        cleanup_mode = "fallback_dilate"

    global_peaks = feature.peak_local_max(
        image,
        labels=(contrast >= contrast_threshold).astype(np.uint8),
        min_distance=coarse.global_peak_min_distance_ds,
        threshold_abs=intensity_threshold,
        num_peaks=coarse.global_peak_max_count,
        exclude_border=False,
    )
    for global_rank, (peak_y, peak_x) in enumerate(global_peaks, start=1):
        label = int(labeled[int(peak_y), int(peak_x)])
        region = regions_by_label.get(label)
        center_x = int(round(peak_x * factor + factor / 2))
        center_y = int(round(peak_y * factor + factor / 2))
        if region is None:
            rows.append(
                {
                    "coarse_component_label": label,
                    "coarse_seed_kind": "global_peak",
                    "coarse_seed_rank": int(global_rank),
                    "coarse_area_ds": 1,
                    "coarse_equivalent_diameter_um": config.pixel_size_um * factor,
                    "coarse_center_x": center_x,
                    "coarse_center_y": center_y,
                    "coarse_mean_ds": float(image[int(peak_y), int(peak_x)]),
                    "coarse_max_ds": float(image[int(peak_y), int(peak_x)]),
                    "coarse_eccentricity": 0.0,
                    "coarse_solidity": 1.0,
                    "coarse_bbox_x0": int(max(0, peak_x - 1) * factor),
                    "coarse_bbox_y0": int(max(0, peak_y - 1) * factor),
                    "coarse_bbox_x1": int((peak_x + 2) * factor),
                    "coarse_bbox_y1": int((peak_y + 2) * factor),
                }
            )
        else:
            rows.append(
                {
                    "coarse_component_label": int(region.label),
                    "coarse_seed_kind": "global_peak",
                    "coarse_seed_rank": int(global_rank),
                    "coarse_area_ds": int(region.area),
                    "coarse_equivalent_diameter_um": float(
                        region.equivalent_diameter_area
                        * factor
                        * config.pixel_size_um
                    ),
                    "coarse_center_x": center_x,
                    "coarse_center_y": center_y,
                    "coarse_mean_ds": float(region.mean_intensity),
                    "coarse_max_ds": float(image[int(peak_y), int(peak_x)]),
                    "coarse_eccentricity": float(region.eccentricity),
                    "coarse_solidity": float(region.solidity),
                    "coarse_bbox_x0": int(region.bbox[1] * factor),
                    "coarse_bbox_y0": int(region.bbox[0] * factor),
                    "coarse_bbox_x1": int(region.bbox[3] * factor),
                    "coarse_bbox_y1": int(region.bbox[2] * factor),
                }
            )

    positive_image = np.clip(contrast, 0, None)
    relaxed_intensity = max(
        float(np.percentile(image, coarse.blob_intensity_percentile)),
        intensity_threshold * coarse.blob_intensity_relax_multiplier,
    )
    relaxed_contrast = max(
        float(np.percentile(contrast, coarse.blob_contrast_percentile)),
        contrast_threshold * coarse.blob_contrast_relax_multiplier,
    )
    blob_input = ndi.gaussian_filter(
        positive_image,
        sigma=coarse.blob_response_sigma_ds,
    )
    blob_input_max = float(blob_input.max())
    if blob_input_max > 0 and coarse.blob_max_new_candidates > 0:
        blob_rows = []
        blobs = feature.blob_log(
            blob_input / blob_input_max,
            min_sigma=coarse.blob_min_sigma_ds,
            max_sigma=coarse.blob_max_sigma_ds,
            num_sigma=coarse.blob_num_sigma,
            threshold=coarse.blob_threshold,
            overlap=coarse.blob_overlap,
            exclude_border=False,
        )
        for blob_rank, (blob_y, blob_x, sigma_ds) in enumerate(blobs, start=1):
            yi = int(round(blob_y))
            xi = int(round(blob_x))
            if yi < 0 or yi >= image.shape[0] or xi < 0 or xi >= image.shape[1]:
                continue
            intensity_value = float(image[yi, xi])
            contrast_value = float(contrast[yi, xi])
            if intensity_value < relaxed_intensity or contrast_value < relaxed_contrast:
                continue
            estimated_diameter_um = float(
                np.sqrt(2.0) * float(sigma_ds) * factor * config.pixel_size_um * 2.0
            )
            if (
                estimated_diameter_um < coarse.min_diameter_um
                or estimated_diameter_um > coarse.max_diameter_um
            ):
                continue
            label = int(labeled[yi, xi])
            region = regions_by_label.get(label)
            radius_ds = max(2.0, 3.0 * float(sigma_ds))
            blob_rows.append(
                {
                    "coarse_component_label": label,
                    "coarse_seed_kind": "blob_log",
                    "coarse_seed_rank": int(blob_rank),
                    "coarse_area_ds": int(region.area)
                    if region is not None
                    else int(np.pi * radius_ds * radius_ds),
                    "coarse_equivalent_diameter_um": float(
                        region.equivalent_diameter_area
                        * factor
                        * config.pixel_size_um
                    )
                    if region is not None
                    else estimated_diameter_um,
                    "coarse_center_x": int(round(blob_x * factor + factor / 2)),
                    "coarse_center_y": int(round(blob_y * factor + factor / 2)),
                    "coarse_mean_ds": float(region.mean_intensity)
                    if region is not None
                    else intensity_value,
                    "coarse_max_ds": intensity_value,
                    "coarse_eccentricity": float(region.eccentricity)
                    if region is not None
                    else 0.0,
                    "coarse_solidity": float(region.solidity)
                    if region is not None
                    else 1.0,
                    "coarse_bbox_x0": int(
                        max(0, round((blob_x - radius_ds) * factor))
                    ),
                    "coarse_bbox_y0": int(
                        max(0, round((blob_y - radius_ds) * factor))
                    ),
                    "coarse_bbox_x1": int(round((blob_x + radius_ds) * factor)),
                    "coarse_bbox_y1": int(round((blob_y + radius_ds) * factor)),
                    "coarse_blob_sigma_ds": float(sigma_ds),
                    "coarse_blob_score": float(
                        blob_input[yi, xi] * max(contrast_value, 0.0)
                    ),
                }
            )
        blob_rows.sort(
            key=lambda row: (
                float(row.get("coarse_blob_score", 0.0)),
                float(row["coarse_max_ds"]),
                float(row["coarse_equivalent_diameter_um"]),
            ),
            reverse=True,
        )
        existing_centers = [
            (float(row["coarse_center_x"]), float(row["coarse_center_y"]))
            for row in rows
        ]
        kept_blob_rows = []
        kept_blob_centers = []
        for row in blob_rows:
            center = (
                float(row["coarse_center_x"]),
                float(row["coarse_center_y"]),
            )
            duplicate = any(
                np.hypot(center[0] - x, center[1] - y)
                <= coarse.seed_merge_distance_px
                for x, y in existing_centers + kept_blob_centers
            )
            if duplicate:
                continue
            kept_blob_rows.append(row)
            kept_blob_centers.append(center)
            if len(kept_blob_rows) >= coarse.blob_max_new_candidates:
                break
        rows.extend(kept_blob_rows)

    rows = _deduplicate_coarse_rows(rows, coarse.seed_merge_distance_px)
    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["coarse_max_ds", "coarse_area_ds"],
            ascending=[False, False],
        ).reset_index(drop=True)
        candidates.insert(
            0,
            "detector_component_id",
            [f"det_{index:04d}" for index in range(len(candidates))],
        )
    thresholds = {
        "intensity_threshold": intensity_threshold,
        "contrast_threshold": contrast_threshold,
        "triangle_contrast_threshold": triangle_contrast,
        "cleanup_mode": cleanup_mode,
        "coarse_blob_relaxed_intensity_floor": relaxed_intensity,
        "coarse_blob_relaxed_contrast_floor": relaxed_contrast,
    }
    return CoarseDetectionResult(
        image_shape_yx=(image.shape[0] * factor, image.shape[1] * factor),
        downsampled=image,
        mask=np.asarray(cleaned_mask, dtype=np.bool_),
        contrast=np.asarray(contrast, dtype=np.float32),
        thresholds=thresholds,
        candidates=candidates,
    )


def scan_coarse_candidates(
    image_path: Path,
    channel_index: int,
    config: OocyteDetectionConfig,
) -> CoarseDetectionResult:
    """Read one OME-TIFF channel in strips and generate coarse proposals."""

    path = Path(image_path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "CYX":
            raise ValueError(
                f"whole-slide coarse detection currently requires CYX axes, got {series.axes!r}"
            )
        array = zarr.open(series.aszarr(), mode="r")
        if not 0 <= channel_index < int(array.shape[0]):
            raise IndexError(f"channel index {channel_index} outside image channel range")
        image_shape = (int(array.shape[1]), int(array.shape[2]))
        downsampled = _build_downsampled_mean_map_from_zarr(
            array,
            channel_index,
            config.coarse,
        )
    result = detect_coarse_candidates(downsampled, config)
    return CoarseDetectionResult(
        image_shape_yx=image_shape,
        downsampled=result.downsampled,
        mask=result.mask,
        contrast=result.contrast,
        thresholds=result.thresholds,
        candidates=result.candidates,
    )


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _range_score(
    value: float,
    core_lo: float,
    core_hi: float,
    soft_lo: float,
    soft_hi: float,
) -> float:
    if value < soft_lo or value > soft_hi:
        return 0.0
    if core_lo <= value <= core_hi:
        return 1.0
    if value < core_lo:
        return _clip01((value - soft_lo) / max(core_lo - soft_lo, 1e-6))
    return _clip01((soft_hi - value) / max(soft_hi - core_hi, 1e-6))


def candidate_score(
    *,
    equivalent_diameter_um: float,
    circularity: float,
    solidity: float,
    brightness_ratio: float,
    offset_px: float,
    config: OocyteDetectionConfig,
) -> float:
    """Score one locally segmented candidate using the frozen profile weights."""

    scoring = config.scoring
    size_score = _range_score(
        equivalent_diameter_um,
        scoring.size_core_range_um[0],
        scoring.size_core_range_um[1],
        scoring.size_soft_range_um[0],
        scoring.size_soft_range_um[1],
    )
    circularity_score = _clip01(
        (circularity - scoring.circularity_floor) / scoring.circularity_span
    )
    solidity_score = _clip01(
        (solidity - scoring.solidity_floor) / scoring.solidity_span
    )
    shape_score = (
        scoring.shape_circularity_weight * circularity_score
        + scoring.shape_solidity_weight * solidity_score
    )
    brightness_score = _clip01(
        (brightness_ratio - scoring.brightness_ratio_floor)
        / scoring.brightness_ratio_span
    )
    centered_score = _clip01(1.0 - offset_px / scoring.centered_offset_limit_px)
    return float(
        scoring.brightness_weight * brightness_score
        + scoring.shape_weight * shape_score
        + scoring.size_weight * size_score
        + scoring.centered_weight * centered_score
    )


def _seed_candidates_from_patch(
    patch: np.ndarray,
    center_x: int,
    center_y: int,
    config: OocyteDetectionConfig,
) -> List[Dict[str, Any]]:
    local = config.local
    patch_center_y = patch.shape[0] // 2
    patch_center_x = patch.shape[1] // 2
    seeds: List[Dict[str, Any]] = [
        {
            "seed_center_x": int(center_x),
            "seed_center_y": int(center_y),
            "seed_source": "coarse_seed",
            "seed_rank": 0,
            "seed_shift_px": 0.0,
        }
    ]

    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    distance = np.sqrt((yy - patch_center_y) ** 2 + (xx - patch_center_x) ** 2)
    search_mask = distance <= local.peak_search_radius_px

    def append_peak_seeds(
        peak_image: np.ndarray,
        *,
        source_name: str,
        min_distance_px: int,
        percentile: float,
        max_count: int,
    ) -> None:
        if max_count <= 0:
            return
        search_values = peak_image[search_mask]
        if search_values.size == 0:
            return
        peak_threshold = float(np.percentile(search_values, percentile))
        peak_coords = feature.peak_local_max(
            peak_image,
            labels=search_mask.astype(np.uint8),
            min_distance=min_distance_px,
            threshold_abs=peak_threshold,
            num_peaks=max_count,
            exclude_border=False,
        )
        peak_rows = [
            (float(peak_image[int(peak_y), int(peak_x)]), int(peak_y), int(peak_x))
            for peak_y, peak_x in peak_coords
        ]
        peak_rows.sort(reverse=True)
        for peak_rank, (_, peak_y, peak_x) in enumerate(peak_rows, start=1):
            shift_x = int(peak_x - patch_center_x)
            shift_y = int(peak_y - patch_center_y)
            shift_px = float(np.hypot(shift_x, shift_y))
            if shift_px <= local.seed_merge_radius_px:
                continue
            candidate = {
                "seed_center_x": int(center_x + shift_x),
                "seed_center_y": int(center_y + shift_y),
                "seed_source": source_name,
                "seed_rank": int(peak_rank),
                "seed_shift_px": shift_px,
            }
            duplicate = any(
                np.hypot(
                    candidate["seed_center_x"] - seed["seed_center_x"],
                    candidate["seed_center_y"] - seed["seed_center_y"],
                )
                <= local.seed_merge_radius_px
                for seed in seeds
            )
            if not duplicate:
                seeds.append(candidate)

    smooth = ndi.gaussian_filter(
        patch.astype(np.float32),
        sigma=max(local.peak_gaussian_sigma, 0.5),
    )
    append_peak_seeds(
        smooth,
        source_name="local_peak",
        min_distance_px=local.peak_min_distance_px,
        percentile=local.peak_percentile,
        max_count=local.max_peak_seeds_per_candidate,
    )
    if local.max_broad_peak_seeds_per_candidate > 0:
        broad_smooth = ndi.gaussian_filter(
            patch.astype(np.float32),
            sigma=max(local.broad_peak_gaussian_sigma, 1.0),
        )
        append_peak_seeds(
            broad_smooth,
            source_name="local_broad_peak",
            min_distance_px=local.broad_peak_min_distance_px,
            percentile=local.broad_peak_percentile,
            max_count=local.max_broad_peak_seeds_per_candidate,
        )

    if local.offset_ring_step_px > 0 and local.max_offset_ring_seeds > 0:
        ring_offsets = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
        for ring_rank, (dx_sign, dy_sign) in enumerate(
            ring_offsets[: local.max_offset_ring_seeds],
            start=1,
        ):
            shift_x = int(dx_sign * local.offset_ring_step_px)
            shift_y = int(dy_sign * local.offset_ring_step_px)
            candidate = {
                "seed_center_x": int(center_x + shift_x),
                "seed_center_y": int(center_y + shift_y),
                "seed_source": "local_offset_ring",
                "seed_rank": int(ring_rank),
                "seed_shift_px": float(np.hypot(shift_x, shift_y)),
            }
            duplicate = any(
                np.hypot(
                    candidate["seed_center_x"] - seed["seed_center_x"],
                    candidate["seed_center_y"] - seed["seed_center_y"],
                )
                <= local.seed_merge_radius_px
                for seed in seeds
            )
            if not duplicate:
                seeds.append(candidate)
    return seeds


def _candidate_rank(candidate: _EvaluatedCandidate) -> Tuple[float, float, float, float]:
    row = candidate.row
    return (
        float(row["detector_score"]),
        float(row["local_circularity"]),
        float(row["mean_to_annulus_p99_ratio"]),
        -float(row["seed_shift_px"]),
    )


def _keep_distinct_candidate_rows(
    candidates: List[_EvaluatedCandidate],
    *,
    merge_distance_px: float,
    max_count: int,
) -> List[_EvaluatedCandidate]:
    ranked = sorted(candidates, key=_candidate_rank, reverse=True)
    kept: List[_EvaluatedCandidate] = []
    for candidate in ranked:
        row = candidate.row
        duplicate = any(
            np.hypot(
                float(row["center_x"]) - float(existing.row["center_x"]),
                float(row["center_y"]) - float(existing.row["center_y"]),
            )
            <= merge_distance_px
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= max_count:
            break
    return kept


def _deduplicate_refined_candidates(
    candidates: List[_EvaluatedCandidate],
    merge_distance_px: float,
) -> List[_EvaluatedCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.row["detector_score"]),
            float(candidate.row["local_circularity"]),
            float(candidate.row["local_equivalent_diameter_um"]),
        ),
        reverse=True,
    )
    kept: List[_EvaluatedCandidate] = []
    for candidate in ranked:
        row = candidate.row
        duplicate = any(
            np.hypot(
                float(row["center_x"]) - float(existing.row["center_x"]),
                float(row["center_y"]) - float(existing.row["center_y"]),
            )
            <= merge_distance_px
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)

    renumbered = []
    for index, candidate in enumerate(kept):
        source_id = str(candidate.row["detector_component_id"])
        row = {
            "detector_component_id": f"det_{index:04d}",
            "source_detector_component_id": source_id,
            **{
                key: value
                for key, value in candidate.row.items()
                if key != "detector_component_id"
            },
        }
        renumbered.append(_EvaluatedCandidate(row=row, evaluation=candidate.evaluation))
    return renumbered


def _refine_candidates(
    patch_reader: Callable[[int, int, int], ExtractedPatch],
    image_shape_yx: Tuple[int, int],
    coarse_candidates: pd.DataFrame,
    config: OocyteDetectionConfig,
) -> RefinedDetectionResult:
    local = config.local
    default_context = {
        "context_mode": "default",
        "window_radius_px": int(local.window_radius_px),
        "annulus_inner_px": int(local.annulus_inner_px),
        "annulus_outer_px": int(local.annulus_outer_px),
    }
    compact_context = {
        "context_mode": "compact",
        "window_radius_px": int(local.compact_window_radius_px),
        "annulus_inner_px": int(local.compact_annulus_inner_px),
        "annulus_outer_px": int(local.compact_annulus_outer_px),
    }
    patch_cache: Dict[Tuple[int, int, int], ExtractedPatch] = {}
    annulus_mask_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    evaluation_cache: Dict[
        Tuple[int, int, int, int, int],
        _SegmentationEvaluation | None,
    ] = {}

    def cached_patch(seed_x: int, seed_y: int, radius: int) -> ExtractedPatch:
        key = (int(seed_x), int(seed_y), int(radius))
        cached = patch_cache.get(key)
        if cached is None:
            cached = patch_reader(*key)
            patch_cache[key] = cached
        return cached

    def annulus_mask(
        shape: Tuple[int, int],
        annulus_inner_px: int,
        annulus_outer_px: int,
    ) -> np.ndarray:
        key = (
            int(shape[0]),
            int(shape[1]),
            int(annulus_inner_px),
            int(annulus_outer_px),
        )
        cached = annulus_mask_cache.get(key)
        if cached is None:
            center_y = shape[0] // 2
            center_x = shape[1] // 2
            yy, xx = np.ogrid[: shape[0], : shape[1]]
            distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
            cached = (distance >= annulus_inner_px) & (distance <= annulus_outer_px)
            annulus_mask_cache[key] = cached
        return cached

    def evaluate_seed_context(
        seed_x: int,
        seed_y: int,
        context: Dict[str, Any],
    ) -> _SegmentationEvaluation | None:
        cache_key = (
            int(seed_x),
            int(seed_y),
            int(context["window_radius_px"]),
            int(context["annulus_inner_px"]),
            int(context["annulus_outer_px"]),
        )
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]
        try:
            patch = cached_patch(
                seed_x,
                seed_y,
                int(context["window_radius_px"]),
            )
            smooth, segmentation = _segment_oocyte_patch(
                patch.image,
                config,
                annulus_inner_px=int(context["annulus_inner_px"]),
                annulus_outer_px=int(context["annulus_outer_px"]),
            )
        except (IndexError, ValueError):
            evaluation_cache[cache_key] = None
            return None

        metrics = segmentation.metrics
        mask = annulus_mask(
            smooth.shape,
            int(context["annulus_inner_px"]),
            int(context["annulus_outer_px"]),
        )
        annulus = smooth[mask]
        annulus_p99 = float(np.percentile(annulus, 99))
        annulus_p95 = float(np.percentile(annulus, 95))
        brightness_ratio = float(metrics.mean_intensity / max(annulus_p99, 1.0))
        score = candidate_score(
            equivalent_diameter_um=metrics.equivalent_diameter_um,
            circularity=metrics.circularity,
            solidity=metrics.solidity,
            brightness_ratio=brightness_ratio,
            offset_px=metrics.centroid_offset_px,
            config=config,
        )
        acceptance = config.acceptance
        strict_accept = bool(
            score >= acceptance.score_threshold
            and metrics.equivalent_diameter_um >= acceptance.strict_min_diameter_um
        )
        rescue_accept = bool(
            not strict_accept
            and acceptance.rescue_min_diameter_um
            <= metrics.equivalent_diameter_um
            <= acceptance.rescue_max_diameter_um
            and metrics.circularity >= acceptance.rescue_min_circularity
            and metrics.solidity >= acceptance.rescue_min_solidity
            and brightness_ratio >= acceptance.rescue_min_brightness_ratio
            and metrics.centroid_offset_px <= acceptance.rescue_max_offset_px
        )
        center_y_patch = smooth.shape[0] // 2
        center_x_patch = smooth.shape[1] // 2
        component_centroid_x = float(
            seed_x + metrics.centroid_x_px - center_x_patch
        )
        component_centroid_y = float(
            seed_y + metrics.centroid_y_px - center_y_patch
        )
        row = {
            "seed_center_x": int(seed_x),
            "seed_center_y": int(seed_y),
            "local_context_mode": str(context["context_mode"]),
            "evaluation_window_radius_px": int(context["window_radius_px"]),
            "evaluation_annulus_inner_px": int(context["annulus_inner_px"]),
            "evaluation_annulus_outer_px": int(context["annulus_outer_px"]),
            "center_x": int(seed_x),
            "center_y": int(seed_y),
            "component_centroid_x": component_centroid_x,
            "component_centroid_y": component_centroid_y,
            "component_centroid_shift_px": float(
                np.hypot(
                    component_centroid_x - seed_x,
                    component_centroid_y - seed_y,
                )
            ),
            "bbox_x0": patch.bbox.x0,
            "bbox_y0": patch.bbox.y0,
            "bbox_x1": patch.bbox.x1,
            "bbox_y1": patch.bbox.y1,
            "threshold_method": metrics.threshold_method,
            "threshold": float(metrics.threshold),
            "selection_mode": metrics.selection_mode,
            "local_area_px": int(metrics.area_px),
            "local_equivalent_diameter_um": float(metrics.equivalent_diameter_um),
            "local_major_axis_um": float(metrics.major_axis_um),
            "local_minor_axis_um": float(metrics.minor_axis_um),
            "local_eccentricity": float(metrics.eccentricity),
            "local_solidity": float(metrics.solidity),
            "local_circularity": float(metrics.circularity),
            "local_centroid_offset_px": float(metrics.centroid_offset_px),
            "local_mean_intensity": float(metrics.mean_intensity),
            "local_max_intensity": float(metrics.max_intensity),
            "annulus_p95": annulus_p95,
            "annulus_p99": annulus_p99,
            "mean_to_annulus_p99_ratio": brightness_ratio,
            "detector_score": float(score),
            "accepted_strict": strict_accept,
            "accepted_rescue": rescue_accept,
            "acceptance_mode": (
                "strict" if strict_accept else ("rescue" if rescue_accept else "rejected")
            ),
            "accepted": bool(strict_accept or rescue_accept),
        }
        evaluation = _SegmentationEvaluation(
            row=row,
            patch=patch,
            segmentation=segmentation,
        )
        evaluation_cache[cache_key] = evaluation
        return evaluation

    def evaluate_candidate(
        record: Dict[str, Any],
        seed_meta: Dict[str, Any],
        seed_x: int,
        seed_y: int,
        *,
        context: Dict[str, Any],
        reseed_iteration: int,
    ) -> _EvaluatedCandidate | None:
        evaluation = evaluate_seed_context(seed_x, seed_y, context)
        if evaluation is None:
            return None
        coarse_shift_px = float(
            np.hypot(
                seed_x - float(record["coarse_center_x"]),
                seed_y - float(record["coarse_center_y"]),
            )
        )
        source = str(seed_meta["seed_source"])
        row = {
            **record,
            **evaluation.row,
            "seed_source": (
                source if reseed_iteration == 0 else f"{source}_centroid_reseed"
            ),
            "seed_rank": int(seed_meta["seed_rank"]),
            "seed_shift_px": coarse_shift_px,
            "local_reseed_iteration": int(reseed_iteration),
        }
        return _EvaluatedCandidate(row=row, evaluation=evaluation)

    def evaluate_seed_variants(
        record: Dict[str, Any],
        seed: Dict[str, Any],
        *,
        contexts_to_try: List[Dict[str, Any]],
    ) -> List[_EvaluatedCandidate]:
        candidate_rows: List[_EvaluatedCandidate] = []
        context_index = 0
        while context_index < len(contexts_to_try):
            context = contexts_to_try[context_index]
            seed_x = int(seed["seed_center_x"])
            seed_y = int(seed["seed_center_y"])
            best_context_row = None
            for reseed_iteration in range(local.max_centroid_reseed_iterations + 1):
                candidate = evaluate_candidate(
                    record,
                    seed,
                    seed_x,
                    seed_y,
                    context=context,
                    reseed_iteration=reseed_iteration,
                )
                if candidate is None:
                    break
                if best_context_row is None or _candidate_rank(candidate) > _candidate_rank(
                    best_context_row
                ):
                    best_context_row = candidate
                if reseed_iteration >= local.max_centroid_reseed_iterations:
                    break
                centroid_shift_px = float(
                    candidate.row["component_centroid_shift_px"]
                )
                if centroid_shift_px < local.centroid_reseed_min_shift_px:
                    break
                if centroid_shift_px > local.centroid_reseed_max_shift_px:
                    break
                next_seed_x = int(round(candidate.row["component_centroid_x"]))
                next_seed_y = int(round(candidate.row["component_centroid_y"]))
                if np.hypot(next_seed_x - seed_x, next_seed_y - seed_y) <= 1.0:
                    break
                seed_x = next_seed_x
                seed_y = next_seed_y
            if best_context_row is not None:
                candidate_rows.append(best_context_row)
                should_retry_compact = bool(
                    compact_context["window_radius_px"]
                    < default_context["window_radius_px"]
                    and context["context_mode"] == "default"
                    and str(seed["seed_source"]) == "coarse_seed"
                    and (
                        not bool(best_context_row.row["accepted"])
                        or best_context_row.row["selection_mode"] != "center_component"
                        or float(
                            best_context_row.row["mean_to_annulus_p99_ratio"]
                        )
                        < local.compact_retry_max_brightness_ratio
                    )
                )
                if should_retry_compact:
                    contexts_to_try.append(compact_context)
            context_index += 1
        return candidate_rows

    rows: List[_EvaluatedCandidate] = []
    for record in coarse_candidates.to_dict("records"):
        center_x = int(record["coarse_center_x"])
        center_y = int(record["coarse_center_y"])
        try:
            initial_patch = cached_patch(
                center_x,
                center_y,
                local.window_radius_px,
            )
        except (IndexError, ValueError):
            continue
        candidate_rows: List[_EvaluatedCandidate] = []
        for seed in _seed_candidates_from_patch(
            initial_patch.image,
            center_x,
            center_y,
            config,
        ):
            contexts_to_try = [default_context]
            if (
                compact_context["window_radius_px"]
                < default_context["window_radius_px"]
                and str(seed["seed_source"]) != "coarse_seed"
            ):
                contexts_to_try.append(compact_context)
            candidate_rows.extend(
                evaluate_seed_variants(
                    record,
                    seed,
                    contexts_to_try=contexts_to_try,
                )
            )
        kept_rows = _keep_distinct_candidate_rows(
            candidate_rows,
            merge_distance_px=local.instance_merge_distance_px,
            max_count=local.max_instances_per_coarse_candidate,
        )
        for local_instance_rank, candidate in enumerate(kept_rows, start=1):
            row = {
                **candidate.row,
                "local_instance_rank": int(local_instance_rank),
                "local_instance_count_from_coarse": int(len(kept_rows)),
            }
            rows.append(_EvaluatedCandidate(row=row, evaluation=candidate.evaluation))
        # Final rows retain their own masks; discard all rejected evaluations before
        # moving to the next proposal so memory does not grow with every seed tested.
        patch_cache.clear()
        evaluation_cache.clear()

    rows.sort(
        key=lambda candidate: (
            float(candidate.row["detector_score"]),
            float(candidate.row["coarse_max_ds"]),
        ),
        reverse=True,
    )
    rows = _deduplicate_refined_candidates(
        rows,
        config.deduplication.refined_center_distance_px,
    )
    candidate_masks = {}
    for candidate in rows:
        candidate_id = str(candidate.row["detector_component_id"])
        evaluation = candidate.evaluation
        cropped_mask = evaluation.patch.crop_to_image_bounds(
            evaluation.segmentation.mask
        )
        candidate_masks[candidate_id] = ScoredCandidateMask(
            mask=np.asarray(cropped_mask, dtype=np.bool_).copy(),
            bbox=evaluation.patch.bbox,
            image_shape_yx=image_shape_yx,
            metrics=evaluation.segmentation.metrics,
        )
    return RefinedDetectionResult(
        candidates=pd.DataFrame([candidate.row for candidate in rows]).reindex(
            columns=REFINED_CANDIDATE_COLUMNS
        ),
        candidate_masks=candidate_masks,
    )


def refine_candidates_from_array(
    channel_image: np.ndarray,
    coarse_candidates: pd.DataFrame,
    config: OocyteDetectionConfig,
) -> RefinedDetectionResult:
    """Refine coarse candidates against an in-memory raw UCHL1 image."""

    image = np.asarray(channel_image)
    if image.ndim != 2:
        raise ValueError("channel image must be two-dimensional")

    def patch_reader(center_x: int, center_y: int, radius: int) -> ExtractedPatch:
        return extract_padded_patch(image, (center_x, center_y), radius)

    baseline = _refine_candidates(
        patch_reader,
        (int(image.shape[0]), int(image.shape[1])),
        coarse_candidates,
        config,
    )
    if config.secondary_rescue is None:
        return baseline
    rescued = run_secondary_rescue(
        patch_reader=patch_reader,
        image_shape_yx=(int(image.shape[0]), int(image.shape[1])),
        coarse_candidates=coarse_candidates,
        baseline_candidates=baseline.candidates,
        baseline_masks=baseline.candidate_masks,
        config=config,
        score_candidate=candidate_score,
    )
    return RefinedDetectionResult(
        candidates=rescued.candidates,
        candidate_masks=rescued.candidate_masks,
        rescue_diagnostics=rescued.diagnostics,
    )


def scan_refined_candidates(
    image_path: Path,
    channel_index: int,
    coarse_candidates: pd.DataFrame,
    config: OocyteDetectionConfig,
) -> RefinedDetectionResult:
    """Refine coarse candidates from one raw OME-TIFF channel."""

    path = Path(image_path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "CYX":
            raise ValueError(
                f"whole-slide refinement currently requires CYX axes, got {series.axes!r}"
            )
        array = zarr.open(series.aszarr(), mode="r")
        if not 0 <= channel_index < int(array.shape[0]):
            raise IndexError(f"channel index {channel_index} outside image channel range")
        image_shape = (int(array.shape[1]), int(array.shape[2]))

        def patch_reader(center_x: int, center_y: int, radius: int) -> ExtractedPatch:
            return extract_cyx_channel_patch(
                array,
                channel_index,
                (center_x, center_y),
                radius,
            )

        baseline = _refine_candidates(
            patch_reader,
            image_shape,
            coarse_candidates,
            config,
        )
        if config.secondary_rescue is None:
            return baseline
        rescued = run_secondary_rescue(
            patch_reader=patch_reader,
            image_shape_yx=image_shape,
            coarse_candidates=coarse_candidates,
            baseline_candidates=baseline.candidates,
            baseline_masks=baseline.candidate_masks,
            config=config,
            score_candidate=candidate_score,
        )
        return RefinedDetectionResult(
            candidates=rescued.candidates,
            candidate_masks=rescued.candidate_masks,
            rescue_diagnostics=rescued.diagnostics,
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(payload: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_csv(table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            table.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def detect_oocytes(
    image_path: Path,
    *,
    sample_id: str,
    out_dir: Path,
    config: OocyteDetectionConfig,
    channel_name: str | None = "UCHL1",
    channel_index: int | None = None,
    antibodies_path: Path | None = None,
    pixel_size_um: float | None = None,
) -> OocyteDetectionResult:
    """Run standalone raw-UCHL1 detection and persist one sample deliverable."""

    if not sample_id.strip():
        raise ValueError("sample_id must not be empty")
    source_path = Path(image_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"raw OME-TIFF not found: {source_path}")
    resolved_config = config
    if pixel_size_um is not None:
        if pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        if not np.isclose(pixel_size_um, config.pixel_size_um):
            resolved_config = dataclass_replace(
                config,
                pixel_size_um=float(pixel_size_um),
            )
    if channel_index is None:
        if antibodies_path is None:
            raise ValueError(
                "provide channel_index or antibodies_path for UCHL1 channel resolution"
            )
        channel_index = find_channel_index(
            Path(antibodies_path),
            channel_name or "UCHL1",
        )

    sample_dir = Path(out_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = sample_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = resolved_config.fingerprint()
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    coarse = scan_coarse_candidates(source_path, channel_index, resolved_config)
    coarse_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    refined = scan_refined_candidates(
        source_path,
        channel_index,
        coarse.candidates,
        resolved_config,
    )
    refinement_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    candidates = refined.candidates.copy()
    mask_paths = []
    for candidate_id in candidates.get(
        "detector_component_id",
        pd.Series(dtype=str),
    ).astype(str):
        relative_path = Path("masks") / f"{candidate_id}.npz"
        candidate_mask = refined.candidate_masks[candidate_id]
        save_candidate_mask(
            sample_dir / relative_path,
            mask=candidate_mask.mask,
            bbox=candidate_mask.bbox,
            image_shape_yx=candidate_mask.image_shape_yx,
            sample_id=sample_id,
            candidate_id=candidate_id,
            profile_name=resolved_config.profile_name,
            profile_fingerprint=fingerprint,
            metrics=candidate_mask.metrics,
            implementation_version=OOCYTE_IMPLEMENTATION_VERSION,
        )
        mask_paths.append(str(relative_path))
    candidates["mask_path"] = mask_paths
    persistence_seconds = time.perf_counter() - stage_start

    coarse_path = sample_dir / "coarse_candidates.csv"
    candidates_path = sample_dir / "candidates.csv"
    labels_path = sample_dir / "oocyte_labels.ome.tiff"
    labels_mapping_path = sample_dir / "oocyte_labels.csv"
    summary_path = sample_dir / "summary.json"
    runtime_path = sample_dir / "runtime.json"
    manifest_path = sample_dir / "run_manifest.json"
    overview_path = sample_dir / "overview.png"
    duplicate_suspects_path = sample_dir / "accepted_duplicate_suspects.csv"
    rescue_diagnostics_path = sample_dir / "rescue_diagnostics.csv"
    _atomic_write_csv(coarse.candidates, coarse_path)
    _atomic_write_csv(candidates, candidates_path)
    if refined.rescue_diagnostics is not None:
        _atomic_write_csv(refined.rescue_diagnostics, rescue_diagnostics_path)

    stage_start = time.perf_counter()
    label_export: LabelExportResult = export_whole_slide_labels(
        candidates,
        sample_dir=sample_dir,
        image_shape_yx=coarse.image_shape_yx,
        image_path=labels_path,
        mapping_path=labels_mapping_path,
    )
    export_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    spatial_qc = render_spatial_overview(
        coarse.downsampled,
        candidates,
        downsample_factor=resolved_config.coarse.downsample_factor,
        pixel_size_um=resolved_config.pixel_size_um,
        sample_id=sample_id,
        out_path=overview_path,
    )
    _atomic_write_csv(spatial_qc.duplicate_suspects, duplicate_suspects_path)
    spatial_qc_seconds = time.perf_counter() - stage_start
    total_seconds = time.perf_counter() - total_start
    runtime = {
        "coarse_detection": float(coarse_seconds),
        "local_refinement": float(refinement_seconds),
        "mask_persistence": float(persistence_seconds),
        "label_export": float(export_seconds),
        "spatial_qc": float(spatial_qc_seconds),
        "total": float(total_seconds),
    }
    artifact_paths = {
        "coarse_candidates": coarse_path,
        "candidates": candidates_path,
        "masks": masks_dir,
        "labels": labels_path,
        "label_mapping": labels_mapping_path,
        "summary": summary_path,
        "runtime": runtime_path,
        "run_manifest": manifest_path,
        "overview": overview_path,
        "duplicate_suspects": duplicate_suspects_path,
    }
    if refined.rescue_diagnostics is not None:
        artifact_paths["rescue_diagnostics"] = rescue_diagnostics_path
    accepted_count = int(candidates["accepted"].sum()) if not candidates.empty else 0
    rescue_candidate_count = int(
        (
            candidates.get("segmentation_pass", pd.Series(dtype=str))
            == "secondary_rescue"
        ).sum()
    )
    summary = {
        "schema_version": 1,
        "sample_id": sample_id,
        "status": "complete",
        "image_shape_yx": list(coarse.image_shape_yx),
        "channel_index": int(channel_index),
        "profile_name": resolved_config.profile_name,
        "profile_fingerprint": fingerprint,
        "implementation_version": OOCYTE_IMPLEMENTATION_VERSION,
        "coarse_candidate_count": int(len(coarse.candidates)),
        "refined_candidate_count": int(len(candidates)),
        "accepted_candidate_count": accepted_count,
        "secondary_rescue_candidate_count": rescue_candidate_count,
        "secondary_rescue_evaluation_count": int(
            0
            if refined.rescue_diagnostics is None
            else len(refined.rescue_diagnostics)
        ),
        "label_count": int(label_export.label_count),
        "assigned_label_pixel_count": int(label_export.assigned_pixel_count),
        "overlap_label_pixel_count": int(label_export.overlap_pixel_count),
        "accepted_duplicate_suspect_count": int(
            len(spatial_qc.duplicate_suspects)
        ),
        "thresholds": coarse.thresholds,
        "runtime_seconds": runtime,
        "artifact_paths": artifact_paths,
    }
    manifest = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_image": str(source_path),
        "source_image_size_bytes": int(source_path.stat().st_size),
        "antibodies_path": None
        if antibodies_path is None
        else str(Path(antibodies_path).resolve()),
        "requested_channel_name": channel_name,
        "resolved_channel_index": int(channel_index),
        "resolved_config": resolved_config.to_dict(),
        "profile_fingerprint": fingerprint,
        "implementation_version": OOCYTE_IMPLEMENTATION_VERSION,
        "outputs": artifact_paths,
    }
    _atomic_write_json(runtime, runtime_path)
    _atomic_write_json(summary, summary_path)
    _atomic_write_json(manifest, manifest_path)
    return OocyteDetectionResult(
        sample_id=sample_id,
        image_path=source_path,
        image_shape_yx=coarse.image_shape_yx,
        channel_index=int(channel_index),
        profile_name=resolved_config.profile_name,
        profile_fingerprint=fingerprint,
        implementation_version=OOCYTE_IMPLEMENTATION_VERSION,
        coarse_candidates=coarse.candidates,
        candidates=candidates,
        thresholds=coarse.thresholds,
        runtime_seconds=runtime,
        artifact_paths=artifact_paths,
    )
