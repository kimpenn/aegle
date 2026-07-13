"""Versioned configuration for standalone oocyte detection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


OOCYTE_IMPLEMENTATION_VERSION = "donor13_v6_engineering_1"


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _ordered_pair(name: str, values: Tuple[float, float]) -> None:
    if values[0] >= values[1]:
        raise ValueError(f"{name} must be increasing, got {values}")


@dataclass(frozen=True)
class CoarseDetectionConfig:
    downsample_factor: int = 8
    strip_height: int = 1024
    gaussian_sigma_small: float = 1.0
    gaussian_sigma_large: float = 10.0
    intensity_percentile: float = 99.9
    intensity_floor_percentile: float = 99.5
    intensity_floor_multiplier: float = 1.2
    contrast_percentile: float = 99.9
    min_component_area_ds: int = 20
    closing_radius_ds: int = 2
    fallback_min_candidate_count: int = 5
    min_diameter_um: float = 15.0
    max_diameter_um: float = 120.0
    max_peaks_per_component: int = 8
    peak_min_distance_ds: int = 6
    peak_percentile: float = 92.0
    global_peak_min_distance_ds: int = 6
    global_peak_max_count: int = 5000
    blob_threshold: float = 0.12
    blob_min_sigma_ds: float = 2.5
    blob_max_sigma_ds: float = 5.5
    blob_num_sigma: int = 5
    blob_overlap: float = 0.5
    blob_response_sigma_ds: float = 0.8
    blob_intensity_percentile: float = 99.5
    blob_contrast_percentile: float = 99.5
    blob_intensity_relax_multiplier: float = 0.35
    blob_contrast_relax_multiplier: float = 0.55
    blob_max_new_candidates: int = 120
    seed_merge_distance_px: float = 60.0

    def __post_init__(self) -> None:
        _positive("downsample_factor", self.downsample_factor)
        _positive("strip_height", self.strip_height)
        if self.gaussian_sigma_small >= self.gaussian_sigma_large:
            raise ValueError("coarse Gaussian sigmas must satisfy small < large")
        if self.min_diameter_um >= self.max_diameter_um:
            raise ValueError("coarse diameter bounds must satisfy min < max")
        if self.blob_min_sigma_ds > self.blob_max_sigma_ds:
            raise ValueError("blob sigma bounds must satisfy min <= max")


@dataclass(frozen=True)
class LocalSegmentationConfig:
    window_radius_px: int = 180
    threshold_method: str = "triangle"
    gaussian_sigma: float = 1.0
    annulus_inner_px: int = 80
    annulus_outer_px: int = 170
    annulus_floor_percentile: float = 99.0
    annulus_floor_multiplier: float = 1.0
    min_component_size_px: int = 200
    closing_radius_px: int = 3
    max_peak_seeds_per_candidate: int = 8
    peak_gaussian_sigma: float = 1.0
    peak_search_radius_px: int = 175
    peak_min_distance_px: int = 24
    peak_percentile: float = 98.5
    max_broad_peak_seeds_per_candidate: int = 3
    broad_peak_gaussian_sigma: float = 8.0
    broad_peak_min_distance_px: int = 36
    broad_peak_percentile: float = 99.0
    max_offset_ring_seeds: int = 8
    offset_ring_step_px: int = 24
    seed_merge_radius_px: float = 12.0
    max_centroid_reseed_iterations: int = 1
    centroid_reseed_min_shift_px: float = 8.0
    centroid_reseed_max_shift_px: float = 80.0
    compact_window_radius_px: int = 140
    compact_annulus_inner_px: int = 60
    compact_annulus_outer_px: int = 120
    compact_retry_max_brightness_ratio: float = 2.5
    max_instances_per_coarse_candidate: int = 6
    instance_merge_distance_px: float = 50.0

    def __post_init__(self) -> None:
        if self.threshold_method not in {"triangle", "yen"}:
            raise ValueError(f"unsupported threshold method: {self.threshold_method}")
        if not 0 < self.annulus_inner_px < self.annulus_outer_px < self.window_radius_px:
            raise ValueError("default annulus must fit inside the local window")
        if not (
            0
            < self.compact_annulus_inner_px
            < self.compact_annulus_outer_px
            < self.compact_window_radius_px
        ):
            raise ValueError("compact annulus must fit inside the compact window")
        if self.compact_window_radius_px >= self.window_radius_px:
            raise ValueError("compact window must be smaller than the default window")
        if self.centroid_reseed_min_shift_px >= self.centroid_reseed_max_shift_px:
            raise ValueError("centroid reseed bounds must satisfy min < max")


@dataclass(frozen=True)
class ScoringConfig:
    size_core_range_um: Tuple[float, float] = (25.0, 80.0)
    size_soft_range_um: Tuple[float, float] = (15.0, 110.0)
    circularity_floor: float = 0.45
    circularity_span: float = 0.40
    solidity_floor: float = 0.80
    solidity_span: float = 0.18
    brightness_ratio_floor: float = 3.0
    brightness_ratio_span: float = 12.0
    centered_offset_limit_px: float = 40.0
    brightness_weight: float = 0.35
    shape_weight: float = 0.30
    size_weight: float = 0.20
    centered_weight: float = 0.15
    shape_circularity_weight: float = 0.60
    shape_solidity_weight: float = 0.40

    def __post_init__(self) -> None:
        _ordered_pair("size_core_range_um", self.size_core_range_um)
        _ordered_pair("size_soft_range_um", self.size_soft_range_um)
        if self.size_soft_range_um[0] > self.size_core_range_um[0]:
            raise ValueError("soft size range must include the core lower bound")
        if self.size_soft_range_um[1] < self.size_core_range_um[1]:
            raise ValueError("soft size range must include the core upper bound")
        total = self.brightness_weight + self.shape_weight + self.size_weight + self.centered_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"candidate score weights must sum to 1, got {total}")
        shape_total = self.shape_circularity_weight + self.shape_solidity_weight
        if abs(shape_total - 1.0) > 1e-9:
            raise ValueError(f"shape score weights must sum to 1, got {shape_total}")


@dataclass(frozen=True)
class AcceptanceConfig:
    score_threshold: float = 0.45
    strict_min_diameter_um: float = 15.0
    rescue_min_diameter_um: float = 24.0
    rescue_max_diameter_um: float = 60.0
    rescue_min_circularity: float = 0.48
    rescue_min_solidity: float = 0.84
    rescue_min_brightness_ratio: float = 1.75
    rescue_max_offset_px: float = 12.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("acceptance score threshold must be in [0, 1]")
        if self.rescue_min_diameter_um >= self.rescue_max_diameter_um:
            raise ValueError("rescue diameter bounds must satisfy min < max")


@dataclass(frozen=True)
class DeduplicationConfig:
    refined_center_distance_px: float = 60.0


@dataclass(frozen=True)
class ComparisonConfig:
    reference_min_final_score: float = 0.35
    match_radius_px: float = 100.0


@dataclass(frozen=True)
class RuntimeConfig:
    automatic_sample_worker_cap: int = 4


@dataclass(frozen=True)
class ExperimentalConfig:
    border_rescue_enabled: bool = False


@dataclass(frozen=True)
class SecondaryRescueConfig:
    """Conservative second pass for crowded fields missed by the v6 refinement."""

    discovery_window_radius_px: int = 240
    discovery_annulus_inner_px: int = 170
    discovery_annulus_outer_px: int = 230
    annulus_floor_percentile: float = 95.0
    compact_relaxed_annulus_floor_percentile: float = 80.0
    brightness_reference_percentile: float = 95.0
    discovery_min_diameter_um: float = 14.0
    discovery_max_diameter_um: float = 115.0
    discovery_min_circularity: float = 0.30
    discovery_min_solidity: float = 0.70
    final_min_diameter_um: float = 18.0
    final_max_diameter_um: float = 100.0
    final_min_circularity: float = 0.65
    final_min_solidity: float = 0.88
    final_min_brightness_ratio: float = 1.10
    final_min_max_intensity: float = 1500.0
    final_score_threshold: float = 0.45
    low_intensity_shape_max_intensity: float = 3000.0
    low_intensity_shape_max_eccentricity: float = 0.82
    small_candidate_diameter_um: float = 20.0
    small_candidate_min_score: float = 0.48
    small_candidate_min_max_intensity: float = 2000.0
    bright_irregular_min_diameter_um: float = 20.0
    bright_irregular_max_diameter_um: float = 60.0
    bright_irregular_min_circularity: float = 0.58
    bright_irregular_min_solidity: float = 0.82
    bright_irregular_min_max_intensity: float = 6000.0
    bright_irregular_min_score: float = 0.43
    bright_irregular_max_centroid_offset_px: float = 50.0
    bright_fragment_min_diameter_um: float = 18.0
    bright_fragment_max_diameter_um: float = 30.0
    bright_fragment_min_circularity: float = 0.40
    bright_fragment_min_solidity: float = 0.70
    bright_fragment_min_max_intensity: float = 8000.0
    bright_fragment_min_brightness_ratio: float = 5.0
    bright_fragment_min_score: float = 0.38
    bright_fragment_max_centroid_offset_px: float = 15.0
    baseline_fallback_min_diameter_um: float = 20.0
    baseline_fallback_max_diameter_um: float = 60.0
    baseline_fallback_min_circularity: float = 0.60
    baseline_fallback_min_solidity: float = 0.85
    baseline_fallback_min_max_intensity: float = 6000.0
    baseline_fallback_max_centroid_offset_px: float = 40.0
    max_component_offset_px: float = 205.0
    final_max_centroid_offset_px: float = 60.0
    discovery_seed_merge_distance_px: float = 20.0
    duplicate_centroid_distance_px: float = 50.0
    duplicate_mask_overlap_fraction: float = 0.25
    max_components_per_coarse_candidate: int = 12

    def __post_init__(self) -> None:
        if not (
            0
            < self.discovery_annulus_inner_px
            < self.discovery_annulus_outer_px
            < self.discovery_window_radius_px
        ):
            raise ValueError("rescue discovery annulus must fit inside its window")
        if not 0.0 < self.annulus_floor_percentile <= 100.0:
            raise ValueError("rescue annulus percentile must be in (0, 100]")
        if not 0.0 < self.compact_relaxed_annulus_floor_percentile <= 100.0:
            raise ValueError("rescue compact-relaxed percentile must be in (0, 100]")
        if not 0.0 < self.brightness_reference_percentile <= 100.0:
            raise ValueError("rescue brightness percentile must be in (0, 100]")
        if self.discovery_min_diameter_um >= self.discovery_max_diameter_um:
            raise ValueError("rescue discovery diameter bounds must satisfy min < max")
        if self.final_min_diameter_um >= self.final_max_diameter_um:
            raise ValueError("rescue final diameter bounds must satisfy min < max")
        if (
            self.bright_irregular_min_diameter_um
            >= self.bright_irregular_max_diameter_um
        ):
            raise ValueError("bright-irregular diameter bounds must satisfy min < max")
        if self.bright_fragment_min_diameter_um >= self.bright_fragment_max_diameter_um:
            raise ValueError("bright-fragment diameter bounds must satisfy min < max")
        if (
            self.baseline_fallback_min_diameter_um
            >= self.baseline_fallback_max_diameter_um
        ):
            raise ValueError("baseline-fallback diameter bounds must satisfy min < max")
        if not 0.0 <= self.final_score_threshold <= 1.0:
            raise ValueError("rescue score threshold must be in [0, 1]")
        if not 0.0 <= self.duplicate_mask_overlap_fraction <= 1.0:
            raise ValueError("rescue mask-overlap threshold must be in [0, 1]")
        for name, value in (
            ("final circularity", self.final_min_circularity),
            ("final solidity", self.final_min_solidity),
            ("low-intensity eccentricity", self.low_intensity_shape_max_eccentricity),
            ("bright-irregular circularity", self.bright_irregular_min_circularity),
            ("bright-irregular solidity", self.bright_irregular_min_solidity),
            ("bright-fragment circularity", self.bright_fragment_min_circularity),
            ("bright-fragment solidity", self.bright_fragment_min_solidity),
            ("baseline-fallback circularity", self.baseline_fallback_min_circularity),
            ("baseline-fallback solidity", self.baseline_fallback_min_solidity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rescue {name} must be in [0, 1]")
        _positive("rescue max components", self.max_components_per_coarse_candidate)


@dataclass(frozen=True)
class OocyteDetectionConfig:
    profile_name: str = "donor13_v6"
    schema_version: int = 1
    pixel_size_um: float = 0.5
    coarse: CoarseDetectionConfig = CoarseDetectionConfig()
    local: LocalSegmentationConfig = LocalSegmentationConfig()
    scoring: ScoringConfig = ScoringConfig()
    acceptance: AcceptanceConfig = AcceptanceConfig()
    deduplication: DeduplicationConfig = DeduplicationConfig()
    comparison: ComparisonConfig = ComparisonConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    experimental: ExperimentalConfig = ExperimentalConfig()
    secondary_rescue: Optional[SecondaryRescueConfig] = None

    def __post_init__(self) -> None:
        if not self.profile_name:
            raise ValueError("profile_name must not be empty")
        _positive("schema_version", self.schema_version)
        _positive("pixel_size_um", self.pixel_size_um)
        if self.experimental.border_rescue_enabled:
            raise ValueError("donor13_v6 does not support experimental border rescue")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Keep the frozen donor13_v6 canonical JSON and fingerprint unchanged.
        if self.secondary_rescue is None:
            payload.pop("secondary_rescue")
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


DONOR13_V6 = OocyteDetectionConfig()
DONOR13_V6_RESCUE_V1 = OocyteDetectionConfig(
    profile_name="donor13_v6_rescue_v1",
    schema_version=2,
    secondary_rescue=SecondaryRescueConfig(),
)

_PROFILES = {
    DONOR13_V6.profile_name: DONOR13_V6,
    DONOR13_V6_RESCUE_V1.profile_name: DONOR13_V6_RESCUE_V1,
}


def available_profiles() -> Tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def get_profile(name: str) -> OocyteDetectionConfig:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(available_profiles())
        raise ValueError(f"unknown oocyte detector profile {name!r}; choose from: {choices}") from exc
