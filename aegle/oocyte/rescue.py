"""Secondary crowded-field recovery for the raw-UCHL1 detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import OocyteDetectionConfig, SecondaryRescueConfig
from .models import ExtractedPatch, ScoredCandidateMask
from .segmentation import (
    _segment_oocyte_patch,
    _segment_oocyte_patch_components,
)


ScoreFunction = Callable[..., float]
PatchReader = Callable[[int, int, int], ExtractedPatch]


@dataclass(frozen=True)
class SecondaryRescueResult:
    candidates: pd.DataFrame
    candidate_masks: Dict[str, ScoredCandidateMask]
    diagnostics: pd.DataFrame


def _annulus_percentiles(
    image: np.ndarray,
    inner_px: int,
    outer_px: int,
    brightness_percentile: float,
) -> Tuple[float, float, float]:
    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    distance = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    values = image[(distance >= inner_px) & (distance <= outer_px)]
    if values.size == 0:
        raise ValueError("rescue background annulus does not intersect the patch")
    return (
        float(np.percentile(values, 95)),
        float(np.percentile(values, 99)),
        float(np.percentile(values, brightness_percentile)),
    )


def _mask_overlap_fraction(
    left: Any,
    right: Any,
) -> float:
    x0 = max(left.bbox.x0, right.bbox.x0)
    y0 = max(left.bbox.y0, right.bbox.y0)
    x1 = min(left.bbox.x1, right.bbox.x1)
    y1 = min(left.bbox.y1, right.bbox.y1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    left_crop = left.mask[
        y0 - left.bbox.y0 : y1 - left.bbox.y0,
        x0 - left.bbox.x0 : x1 - left.bbox.x0,
    ]
    right_crop = right.mask[
        y0 - right.bbox.y0 : y1 - right.bbox.y0,
        x0 - right.bbox.x0 : x1 - right.bbox.x0,
    ]
    intersection = int(np.logical_and(left_crop, right_crop).sum())
    if intersection == 0:
        return 0.0
    denominator = min(int(left.mask.sum()), int(right.mask.sum()))
    return float(intersection / max(denominator, 1))


def suppress_accepted_mask_duplicates(
    candidates: pd.DataFrame,
    candidate_masks: Mapping[str, Any],
    *,
    overlap_fraction: float,
    max_centroid_distance_px: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mark lower-scoring accepted masks as duplicates using actual mask overlap."""

    diagnostic_columns = [
        "detector_component_id",
        "duplicate_of",
        "mask_overlap_fraction_smaller",
        "detector_score",
        "status",
    ]
    output = candidates.copy()
    if output.empty:
        return output, pd.DataFrame(columns=diagnostic_columns)
    accepted = output[output["accepted"].astype(bool)].sort_values(
        ["detector_score", "detector_component_id"],
        ascending=[False, True],
        kind="stable",
    )
    kept: List[Tuple[int, Mapping[str, Any], Any]] = []
    diagnostics = []

    def centroid(record: Mapping[str, Any], axis: str) -> float:
        value = record.get(f"component_centroid_{axis}")
        if value is None or not np.isfinite(float(value)):
            value = record[f"center_{axis}"]
        return float(value)

    for index, row in accepted.iterrows():
        record = row.to_dict()
        candidate_id = str(record["detector_component_id"])
        candidate_mask = candidate_masks[candidate_id]
        duplicate_of = ""
        duplicate_overlap = 0.0
        for _, kept_record, kept_mask in kept:
            distance = float(
                np.hypot(
                    centroid(record, "x") - centroid(kept_record, "x"),
                    centroid(record, "y") - centroid(kept_record, "y"),
                )
            )
            if distance > max_centroid_distance_px:
                continue
            overlap = _mask_overlap_fraction(candidate_mask, kept_mask)
            if overlap > duplicate_overlap:
                duplicate_overlap = overlap
                duplicate_of = str(kept_record["detector_component_id"])
        if duplicate_overlap >= overlap_fraction:
            output.at[index, "accepted"] = False
            output.at[index, "acceptance_mode"] = "mask_duplicate_suppressed"
            output.at[index, "duplicate_suppressed"] = True
            diagnostics.append(
                {
                    "detector_component_id": candidate_id,
                    "duplicate_of": duplicate_of,
                    "mask_overlap_fraction_smaller": duplicate_overlap,
                    "detector_score": float(record["detector_score"]),
                    "status": "mask_duplicate_suppressed",
                }
            )
            continue
        output.at[index, "duplicate_suppressed"] = False
        kept.append((index, record, candidate_mask))
    return output, pd.DataFrame(diagnostics, columns=diagnostic_columns)


def _discovery_component_is_plausible(
    row: Mapping[str, Any],
    rescue: SecondaryRescueConfig,
) -> bool:
    return bool(
        rescue.discovery_min_diameter_um
        <= float(row["diameter_um"])
        <= rescue.discovery_max_diameter_um
        and float(row["circularity"]) >= rescue.discovery_min_circularity
        and float(row["solidity"]) >= rescue.discovery_min_solidity
        and float(row["offset_px"]) <= rescue.max_component_offset_px
    )


def _final_acceptance_reason(
    row: Mapping[str, Any],
    rescue: SecondaryRescueConfig,
) -> str:
    checks = (
        (
            rescue.final_min_diameter_um
            <= float(row["local_equivalent_diameter_um"])
            <= rescue.final_max_diameter_um,
            "diameter",
        ),
        (float(row["local_circularity"]) >= rescue.final_min_circularity, "circularity"),
        (float(row["local_solidity"]) >= rescue.final_min_solidity, "solidity"),
        (
            float(row["mean_to_background_ratio"])
            >= rescue.final_min_brightness_ratio,
            "brightness",
        ),
        (
            float(row["local_max_intensity"])
            >= rescue.final_min_max_intensity,
            "absolute_intensity",
        ),
        (float(row["detector_score"]) >= rescue.final_score_threshold, "score"),
        (
            float(row["local_centroid_offset_px"])
            <= rescue.final_max_centroid_offset_px,
            "centroid_offset",
        ),
        (
            not (
                float(row["local_max_intensity"])
                < rescue.low_intensity_shape_max_intensity
                and float(row["local_eccentricity"])
                > rescue.low_intensity_shape_max_eccentricity
            ),
            "low_intensity_elongation",
        ),
        (
            not (
                float(row["local_equivalent_diameter_um"])
                < rescue.small_candidate_diameter_um
                and float(row["detector_score"])
                < rescue.small_candidate_min_score
                and float(row["local_max_intensity"])
                < rescue.small_candidate_min_max_intensity
            ),
            "small_low_confidence",
        ),
    )
    failed = [name for passed, name in checks if not passed]
    if not failed:
        return "accepted_standard_pre_dedup"
    bright_irregular = bool(
        rescue.bright_irregular_min_diameter_um
        <= float(row["local_equivalent_diameter_um"])
        <= rescue.bright_irregular_max_diameter_um
        and float(row["local_circularity"])
        >= rescue.bright_irregular_min_circularity
        and float(row["local_solidity"]) >= rescue.bright_irregular_min_solidity
        and float(row["local_max_intensity"])
        >= rescue.bright_irregular_min_max_intensity
        and float(row["detector_score"]) >= rescue.bright_irregular_min_score
        and float(row["local_centroid_offset_px"])
        <= rescue.bright_irregular_max_centroid_offset_px
    )
    if bright_irregular:
        return "accepted_bright_irregular_pre_dedup"
    bright_fragment = bool(
        rescue.bright_fragment_min_diameter_um
        <= float(row["local_equivalent_diameter_um"])
        <= rescue.bright_fragment_max_diameter_um
        and float(row["local_circularity"])
        >= rescue.bright_fragment_min_circularity
        and float(row["local_solidity"]) >= rescue.bright_fragment_min_solidity
        and float(row["local_max_intensity"])
        >= rescue.bright_fragment_min_max_intensity
        and float(row["mean_to_background_ratio"])
        >= rescue.bright_fragment_min_brightness_ratio
        and float(row["detector_score"]) >= rescue.bright_fragment_min_score
        and float(row["local_centroid_offset_px"])
        <= rescue.bright_fragment_max_centroid_offset_px
    )
    return (
        "accepted_bright_fragment_pre_dedup"
        if bright_fragment
        else "failed_" + "+".join(failed)
    )


def _baseline_fallback_is_eligible(
    row: Mapping[str, Any],
    rescue: SecondaryRescueConfig,
) -> bool:
    return bool(
        rescue.baseline_fallback_min_diameter_um
        <= float(row["local_equivalent_diameter_um"])
        <= rescue.baseline_fallback_max_diameter_um
        and float(row["local_circularity"])
        >= rescue.baseline_fallback_min_circularity
        and float(row["local_solidity"]) >= rescue.baseline_fallback_min_solidity
        and float(row["local_max_intensity"])
        >= rescue.baseline_fallback_min_max_intensity
        and float(row["local_centroid_offset_px"])
        <= rescue.baseline_fallback_max_centroid_offset_px
    )


def _evaluate_recentered_component(
    *,
    patch_reader: PatchReader,
    image_shape_yx: Tuple[int, int],
    source_record: Mapping[str, Any],
    discovery_rank: int,
    seed_source: str,
    center_x: int,
    center_y: int,
    config: OocyteDetectionConfig,
    score_candidate: ScoreFunction,
) -> Tuple[Dict[str, Any], ScoredCandidateMask] | None:
    rescue = config.secondary_rescue
    if rescue is None:
        return None
    local = config.local
    if seed_source == "rejected_v6_seed":
        window_radius_px = int(source_record["evaluation_window_radius_px"])
        annulus_inner_px = int(source_record["evaluation_annulus_inner_px"])
        annulus_outer_px = int(source_record["evaluation_annulus_outer_px"])
        inherited_context = str(source_record["local_context_mode"])
    elif seed_source == "coarse_component_discovery_compact":
        window_radius_px = int(local.compact_window_radius_px)
        annulus_inner_px = int(local.compact_annulus_inner_px)
        annulus_outer_px = int(local.compact_annulus_outer_px)
        inherited_context = "compact"
    elif seed_source == "coarse_component_discovery_compact_relaxed":
        window_radius_px = int(local.compact_window_radius_px)
        annulus_inner_px = int(local.compact_annulus_inner_px)
        annulus_outer_px = int(local.compact_annulus_outer_px)
        inherited_context = "compact_relaxed"
    else:
        window_radius_px = int(local.window_radius_px)
        annulus_inner_px = int(local.annulus_inner_px)
        annulus_outer_px = int(local.annulus_outer_px)
        inherited_context = "default"
    floor_percentile = (
        rescue.compact_relaxed_annulus_floor_percentile
        if inherited_context == "compact_relaxed"
        else rescue.annulus_floor_percentile
    )
    brightness_percentile = (
        floor_percentile
        if inherited_context == "compact_relaxed"
        else rescue.brightness_reference_percentile
    )
    best: Tuple[Dict[str, Any], ScoredCandidateMask] | None = None
    current_x = int(center_x)
    current_y = int(center_y)
    for reseed_iteration in range(local.max_centroid_reseed_iterations + 1):
        try:
            patch = patch_reader(current_x, current_y, window_radius_px)
            smooth, segmentation = _segment_oocyte_patch(
                patch.image,
                config,
                annulus_inner_px=annulus_inner_px,
                annulus_outer_px=annulus_outer_px,
                annulus_floor_percentile=floor_percentile,
            )
            annulus_p95, annulus_p99, brightness_reference = _annulus_percentiles(
                smooth,
                annulus_inner_px,
                annulus_outer_px,
                brightness_percentile,
            )
        except (IndexError, ValueError):
            break

        metrics = segmentation.metrics
        brightness_ratio = float(
            metrics.mean_intensity / max(brightness_reference, 1.0)
        )
        score = score_candidate(
            equivalent_diameter_um=metrics.equivalent_diameter_um,
            circularity=metrics.circularity,
            solidity=metrics.solidity,
            brightness_ratio=brightness_ratio,
            offset_px=metrics.centroid_offset_px,
            config=config,
        )
        patch_center_y = smooth.shape[0] // 2
        patch_center_x = smooth.shape[1] // 2
        component_centroid_x = float(
            current_x + metrics.centroid_x_px - patch_center_x
        )
        component_centroid_y = float(
            current_y + metrics.centroid_y_px - patch_center_y
        )
        row = {
            **source_record,
            "source_detector_component_id": str(
                source_record["detector_component_id"]
            ),
            "seed_center_x": int(center_x),
            "seed_center_y": int(center_y),
            "local_context_mode": (
                f"secondary_rescue_p{floor_percentile:g}_{inherited_context}"
            ),
            "evaluation_window_radius_px": window_radius_px,
            "evaluation_annulus_inner_px": annulus_inner_px,
            "evaluation_annulus_outer_px": annulus_outer_px,
            "center_x": int(round(component_centroid_x)),
            "center_y": int(round(component_centroid_y)),
            "component_centroid_x": component_centroid_x,
            "component_centroid_y": component_centroid_y,
            "component_centroid_shift_px": float(metrics.centroid_offset_px),
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
            "mean_to_annulus_p99_ratio": float(
                metrics.mean_intensity / max(annulus_p99, 1.0)
            ),
            "detector_score": float(score),
            "accepted_strict": False,
            "accepted_rescue": True,
            "acceptance_mode": "secondary_rescue",
            "accepted": True,
            "seed_source": "secondary_rescue_component",
            "seed_rank": int(discovery_rank),
            "seed_shift_px": float(
                np.hypot(
                    center_x - float(source_record["coarse_center_x"]),
                    center_y - float(source_record["coarse_center_y"]),
                )
            ),
            "local_reseed_iteration": int(reseed_iteration),
            "local_instance_rank": int(discovery_rank),
            "local_instance_count_from_coarse": 0,
            "segmentation_pass": "secondary_rescue",
            "score_background_percentile": float(
                brightness_percentile
            ),
            "mean_to_background_ratio": brightness_ratio,
            "rescue_discovery_center_x": int(center_x),
            "rescue_discovery_center_y": int(center_y),
            "rescue_discovery_rank": int(discovery_rank),
            "rescue_seed_source": str(seed_source),
        }
        cropped_mask = patch.crop_to_image_bounds(segmentation.mask)
        candidate_mask = ScoredCandidateMask(
            mask=np.asarray(cropped_mask, dtype=np.bool_).copy(),
            bbox=patch.bbox,
            image_shape_yx=image_shape_yx,
            metrics=metrics,
        )
        evaluated = (row, candidate_mask)
        if best is None or (
            float(row["detector_score"]),
            float(row["local_circularity"]),
            -float(row["local_centroid_offset_px"]),
        ) > (
            float(best[0]["detector_score"]),
            float(best[0]["local_circularity"]),
            -float(best[0]["local_centroid_offset_px"]),
        ):
            best = evaluated

        shift = float(metrics.centroid_offset_px)
        if reseed_iteration >= local.max_centroid_reseed_iterations:
            break
        if not local.centroid_reseed_min_shift_px <= shift <= (
            local.centroid_reseed_max_shift_px
        ):
            break
        next_x = int(round(component_centroid_x))
        next_y = int(round(component_centroid_y))
        if np.hypot(next_x - current_x, next_y - current_y) <= 1.0:
            break
        current_x = next_x
        current_y = next_y
    return best


def run_secondary_rescue(
    *,
    patch_reader: PatchReader,
    image_shape_yx: Tuple[int, int],
    coarse_candidates: pd.DataFrame,
    baseline_candidates: pd.DataFrame,
    baseline_masks: Mapping[str, ScoredCandidateMask],
    config: OocyteDetectionConfig,
    score_candidate: ScoreFunction,
) -> SecondaryRescueResult:
    """Discover extra components, recenter them, and retain only nonduplicate rescues."""

    rescue = config.secondary_rescue
    if rescue is None:
        return SecondaryRescueResult(
            candidates=baseline_candidates.copy(),
            candidate_masks=dict(baseline_masks),
            diagnostics=pd.DataFrame(),
        )

    baseline_candidates, baseline_duplicate_diagnostics = (
        suppress_accepted_mask_duplicates(
            baseline_candidates,
            baseline_masks,
            overlap_fraction=rescue.duplicate_mask_overlap_fraction,
            max_centroid_distance_px=2.5 * rescue.duplicate_centroid_distance_px,
        )
    )
    accepted_existing: List[Tuple[str, Mapping[str, Any], ScoredCandidateMask]] = []
    for row in baseline_candidates.to_dict("records"):
        if bool(row.get("accepted", False)):
            candidate_id = str(row["detector_component_id"])
            accepted_existing.append((candidate_id, row, baseline_masks[candidate_id]))

    evaluated_centers: List[Tuple[float, float]] = []
    evaluated: List[Tuple[Dict[str, Any], ScoredCandidateMask]] = []
    diagnostics: List[Dict[str, Any]] = baseline_duplicate_diagnostics.to_dict(
        "records"
    )

    def evaluate_seed(
        source_record: Mapping[str, Any],
        *,
        center_x: float,
        center_y: float,
        discovery_rank: int,
        seed_source: str,
    ) -> bool:
        result = _evaluate_recentered_component(
            patch_reader=patch_reader,
            image_shape_yx=image_shape_yx,
            source_record=source_record,
            discovery_rank=discovery_rank,
            seed_source=seed_source,
            center_x=int(round(center_x)),
            center_y=int(round(center_y)),
            config=config,
            score_candidate=score_candidate,
        )
        if result is None:
            return False
        row, candidate_mask = result
        row["rescue_evaluation_id"] = f"eval_{len(diagnostics):06d}"
        row["rescue_status"] = _final_acceptance_reason(row, rescue)
        rule_by_status = {
            "accepted_bright_irregular_pre_dedup": "bright_irregular",
            "accepted_bright_fragment_pre_dedup": "bright_fragment",
        }
        row["rescue_acceptance_rule"] = rule_by_status.get(
            row["rescue_status"],
            "standard",
        )
        if row["rescue_acceptance_rule"] != "standard":
            row["acceptance_mode"] = (
                f"secondary_rescue_{row['rescue_acceptance_rule']}"
            )
        diagnostics.append(dict(row))
        if not str(row["rescue_status"]).endswith("_pre_dedup"):
            return False
        evaluated.append((row, candidate_mask))
        return True

    rejected = baseline_candidates[
        ~baseline_candidates["accepted"].astype(bool)
    ].sort_values("detector_score", ascending=False)
    for source_record in rejected.to_dict("records"):
        source_id = str(source_record["detector_component_id"])
        if _baseline_fallback_is_eligible(source_record, rescue):
            fallback_row = {
                **source_record,
                "source_detector_component_id": source_id,
                "center_x": int(round(float(source_record["component_centroid_x"]))),
                "center_y": int(round(float(source_record["component_centroid_y"]))),
                "accepted_strict": False,
                "accepted_rescue": True,
                "acceptance_mode": "secondary_rescue_baseline_shape_fallback",
                "accepted": True,
                "seed_source": "rejected_v6_mask_fallback",
                "segmentation_pass": "secondary_rescue",
                "score_background_percentile": 99.0,
                "mean_to_background_ratio": float(
                    source_record["mean_to_annulus_p99_ratio"]
                ),
                "rescue_discovery_center_x": int(source_record["seed_center_x"]),
                "rescue_discovery_center_y": int(source_record["seed_center_y"]),
                "rescue_discovery_rank": 0,
                "rescue_seed_source": "rejected_v6_mask_fallback",
                "rescue_evaluation_id": f"eval_{len(diagnostics):06d}",
                "rescue_status": "accepted_baseline_shape_fallback_pre_dedup",
                "rescue_acceptance_rule": "baseline_shape_fallback",
            }
            diagnostics.append(dict(fallback_row))
            evaluated.append((fallback_row, baseline_masks[source_id]))
        center_x = float(source_record["seed_center_x"])
        center_y = float(source_record["seed_center_y"])
        accepted = evaluate_seed(
            source_record,
            center_x=center_x,
            center_y=center_y,
            discovery_rank=1,
            seed_source="rejected_v6_seed",
        )
        if accepted:
            evaluated_centers.append((center_x, center_y))

    for source_record in coarse_candidates.to_dict("records"):
        coarse_x = int(source_record["coarse_center_x"])
        coarse_y = int(source_record["coarse_center_y"])
        try:
            patch = patch_reader(
                coarse_x,
                coarse_y,
                rescue.discovery_window_radius_px,
            )
            _, components = _segment_oocyte_patch_components(
                patch.image,
                config,
                annulus_inner_px=rescue.discovery_annulus_inner_px,
                annulus_outer_px=rescue.discovery_annulus_outer_px,
                annulus_floor_percentile=rescue.annulus_floor_percentile,
            )
        except (IndexError, ValueError):
            continue

        discovery_rows = []
        patch_center_y = patch.image.shape[0] // 2
        patch_center_x = patch.image.shape[1] // 2
        for component in components:
            metrics = component.metrics
            component_x = float(coarse_x + metrics.centroid_x_px - patch_center_x)
            component_y = float(coarse_y + metrics.centroid_y_px - patch_center_y)
            discovery_rows.append(
                {
                    "component": component,
                    "center_x": component_x,
                    "center_y": component_y,
                    "diameter_um": float(metrics.equivalent_diameter_um),
                    "circularity": float(metrics.circularity),
                    "solidity": float(metrics.solidity),
                    "offset_px": float(metrics.centroid_offset_px),
                    "mean_intensity": float(metrics.mean_intensity),
                }
            )
        plausible = [
            row
            for row in discovery_rows
            if _discovery_component_is_plausible(row, rescue)
        ]
        plausible.sort(
            key=lambda row: (
                float(row["mean_intensity"]),
                float(row["diameter_um"]),
                float(row["circularity"]),
            ),
            reverse=True,
        )
        plausible = plausible[: rescue.max_components_per_coarse_candidate]
        for discovery_rank, discovery in enumerate(plausible, start=1):
            center_x = float(discovery["center_x"])
            center_y = float(discovery["center_y"])
            repeated = any(
                np.hypot(center_x - prior_x, center_y - prior_y)
                <= rescue.discovery_seed_merge_distance_px
                for prior_x, prior_y in evaluated_centers
            )
            if repeated:
                diagnostics.append(
                    {
                        "source_detector_component_id": str(
                            source_record["detector_component_id"]
                        ),
                        "rescue_discovery_center_x": center_x,
                        "rescue_discovery_center_y": center_y,
                        "rescue_discovery_rank": discovery_rank,
                        "rescue_seed_source": "coarse_component_discovery",
                        "rescue_status": "duplicate_discovery_seed",
                    }
                )
                continue
            evaluated_centers.append((center_x, center_y))
            diagnostic_count_before = len(diagnostics)
            accepted = evaluate_seed(
                source_record,
                center_x=center_x,
                center_y=center_y,
                discovery_rank=discovery_rank,
                seed_source="coarse_component_discovery",
            )
            latest = diagnostics[-1] if len(diagnostics) > diagnostic_count_before else None
            should_retry_compact = bool(
                not accepted
                and (
                    latest is None
                    or str(latest.get("selection_mode", ""))
                    != "center_component"
                    or float(latest.get("local_centroid_offset_px", 0.0)) > 80.0
                )
            )
            if should_retry_compact:
                compact_diagnostic_count_before = len(diagnostics)
                compact_accepted = evaluate_seed(
                    source_record,
                    center_x=center_x,
                    center_y=center_y,
                    discovery_rank=discovery_rank,
                    seed_source="coarse_component_discovery_compact",
                )
                latest_compact = (
                    diagnostics[-1]
                    if len(diagnostics) > compact_diagnostic_count_before
                    else None
                )
                should_retry_relaxed = bool(
                    not compact_accepted
                    and (
                        latest_compact is None
                        or str(latest_compact.get("selection_mode", ""))
                        != "center_component"
                        or float(
                            latest_compact.get("local_centroid_offset_px", 0.0)
                        )
                        > 80.0
                    )
                )
                if should_retry_relaxed:
                    evaluate_seed(
                        source_record,
                        center_x=center_x,
                        center_y=center_y,
                        discovery_rank=discovery_rank,
                        seed_source=(
                            "coarse_component_discovery_compact_relaxed"
                        ),
                    )

    evaluated.sort(
        key=lambda item: (
            float(item[0]["detector_score"]),
            float(item[0]["local_circularity"]),
            float(item[0]["local_equivalent_diameter_um"]),
        ),
        reverse=True,
    )
    kept: List[Tuple[str, Dict[str, Any], ScoredCandidateMask]] = []
    for row, candidate_mask in evaluated:
        duplicate_id = ""
        duplicate_fraction = 0.0
        comparison = accepted_existing + kept
        for existing_id, existing_row, existing_mask in comparison:
            centroid_distance = float(
                np.hypot(
                    float(row["component_centroid_x"])
                    - float(existing_row["component_centroid_x"]),
                    float(row["component_centroid_y"])
                    - float(existing_row["component_centroid_y"]),
                )
            )
            if centroid_distance > 2.5 * rescue.duplicate_centroid_distance_px:
                continue
            overlap = _mask_overlap_fraction(candidate_mask, existing_mask)
            if overlap > duplicate_fraction:
                duplicate_fraction = overlap
                duplicate_id = existing_id
        is_duplicate = bool(
            duplicate_fraction >= rescue.duplicate_mask_overlap_fraction
        )
        if not is_duplicate:
            for existing_id, existing_row, _ in comparison:
                distance = float(
                    np.hypot(
                        float(row["component_centroid_x"])
                        - float(existing_row["component_centroid_x"]),
                        float(row["component_centroid_y"])
                        - float(existing_row["component_centroid_y"]),
                    )
                )
                if distance <= 8.0:
                    duplicate_id = existing_id
                    is_duplicate = True
                    break
        diagnostic_match = next(
            item
            for item in diagnostics
            if str(item.get("rescue_status", "")).endswith("_pre_dedup")
            and item.get("rescue_evaluation_id") == row.get("rescue_evaluation_id")
        )
        diagnostic_match["rescue_duplicate_of"] = duplicate_id
        diagnostic_match["rescue_duplicate_overlap_fraction"] = duplicate_fraction
        if is_duplicate:
            diagnostic_match["rescue_status"] = "duplicate_existing_mask"
            continue
        candidate_id = f"rescue_{len(kept):04d}"
        row["detector_component_id"] = candidate_id
        row["rescue_status"] = "accepted"
        row["rescue_duplicate_of"] = ""
        row["rescue_duplicate_overlap_fraction"] = duplicate_fraction
        diagnostic_match.update(row)
        kept.append((candidate_id, row, candidate_mask))

    baseline = baseline_candidates.copy()
    if not baseline.empty:
        baseline["segmentation_pass"] = "baseline_v6"
        baseline["score_background_percentile"] = 99.0
        baseline["mean_to_background_ratio"] = baseline[
            "mean_to_annulus_p99_ratio"
        ]
        baseline["rescue_status"] = "not_applicable"
    rescue_rows = pd.DataFrame([row for _, row, _ in kept])
    combined = pd.concat([baseline, rescue_rows], ignore_index=True, sort=False)
    combined_masks = dict(baseline_masks)
    combined_masks.update(
        {candidate_id: mask for candidate_id, _, mask in kept}
    )
    return SecondaryRescueResult(
        candidates=combined,
        candidate_masks=combined_masks,
        diagnostics=pd.DataFrame(diagnostics),
    )


__all__ = [
    "SecondaryRescueResult",
    "run_secondary_rescue",
    "suppress_accepted_mask_duplicates",
]
