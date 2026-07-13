"""Standalone raw-UCHL1 oocyte detection."""

from .batch import (
    OocyteBatchResult,
    OocyteSampleInput,
    detect_oocyte_batch,
    load_sample_manifest,
)
from .config import (
    DONOR13_V6,
    DONOR13_V6_RESCUE_V1,
    OOCYTE_IMPLEMENTATION_VERSION,
    OocyteDetectionConfig,
    available_profiles,
    get_profile,
)
from .detection import (
    CoarseDetectionResult,
    OocyteDetectionResult,
    RefinedDetectionResult,
    candidate_score,
    detect_coarse_candidates,
    detect_oocytes,
    refine_candidates_from_array,
    scan_coarse_candidates,
    scan_refined_candidates,
)
from .delta import RescueDeltaBatchResult, generate_rescue_delta_batch
from .io import (
    find_channel_index,
    load_candidate_mask,
    read_ome_channel_patch,
    save_candidate_mask,
)
from .manual_seed_finalize import (
    MANUAL_SEED_PROFILE_NAME,
    ManualSeedFinalizeResult,
    finalize_manual_seed_review,
)
from .export import LabelExportResult, export_whole_slide_labels
from .models import (
    BoundingBox,
    LocalSegmentationResult,
    ScoredCandidateMask,
    SegmentationMetrics,
)
from .review import ReviewPackResult, generate_review_pack
from .report import HtmlReportResult, algorithm_document_html, generate_html_reports
from .release import (
    OOCYTE_RELEASE_IMPLEMENTATION_VERSION,
    OOCYTE_RELEASE_SCHEMA_VERSION,
    OocyteReleaseResult,
    OocyteReleaseSample,
    OocyteReleaseSpec,
    build_oocyte_release,
    load_oocyte_release_spec,
    validate_oocyte_release,
)
from .recall_review import (
    RecallReviewBundle,
    RecallReviewRuntime,
    analyze_recall_review,
    classify_recall_failure,
    generate_recall_review_bundle,
    serve_recall_review,
)
from .recall_review_batch import (
    BATCH_REVIEW_SCHEMA_VERSION,
    BatchRecallReviewBundle,
    generate_batch_recall_review_bundle,
    serve_batch_recall_review,
)
from .recall_overlay import (
    REVIEWED_OVERLAY_CANDIDATE_FILES,
    RecallMaskOverlay,
    load_recall_mask_overlay,
    overlay_dir_from_identity,
)
from .qc import (
    BatchSpatialQcResult,
    SpatialQcResult,
    accepted_duplicate_suspects,
    compile_batch_spatial_qc,
    render_spatial_overview,
)
from .segmentation import segment_oocyte_patch
from .profiling import (
    OOCYTE_PROFILING_VERSION,
    OocyteProfilingResult,
    profile_oocyte_labels,
)
from .precision_boundary_review import (
    BOUNDARY_RECOVERY_PARAMETERS,
    PrecisionBoundaryReviewResult,
    generate_precision_boundary_review,
)
from .precision_boundary_finalize import (
    PRECISION_BOUNDARY_CHOICES,
    PRECISION_RESOLVED_PROFILE_NAME,
    PrecisionBoundaryFinalizeResult,
    finalize_precision_boundary_review,
)
from .precision_manual_boundary_review import (
    PRECISION_MANUAL_BOUNDARY_RENDERER_VERSION,
    PRECISION_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION,
    PrecisionManualBoundaryReviewResult,
    generate_precision_manual_boundary_review,
)
from .precision_manual_boundary_finalize import (
    MANUAL_CONTOUR_MAX_DIAMETER_UM,
    MANUAL_CONTOUR_MIN_DIAMETER_UM,
    PRECISION_MANUAL_BOUNDARY_CHOICES,
    PRECISION_RESOLVED_V2_PROFILE_NAME,
    PrecisionManualBoundaryFinalizeResult,
    finalize_precision_manual_boundary_review,
)
from .shape_recovery_review import (
    SHAPE_RECOVERY_PARAMETERS,
    ShapeRecoveryReviewResult,
    generate_shape_recovery_review,
)
from .shape_recovery_finalize import (
    MANUAL_SEED_V2_PROFILE_NAME,
    SHAPE_REVIEW_CHOICES,
    finalize_shape_recovery_review,
)

__all__ = [
    "DONOR13_V6",
    "DONOR13_V6_RESCUE_V1",
    "OocyteBatchResult",
    "OOCYTE_IMPLEMENTATION_VERSION",
    "OOCYTE_PROFILING_VERSION",
    "OOCYTE_RELEASE_IMPLEMENTATION_VERSION",
    "OOCYTE_RELEASE_SCHEMA_VERSION",
    "OocyteDetectionConfig",
    "BoundingBox",
    "BATCH_REVIEW_SCHEMA_VERSION",
    "BatchRecallReviewBundle",
    "BatchSpatialQcResult",
    "BOUNDARY_RECOVERY_PARAMETERS",
    "CoarseDetectionResult",
    "LocalSegmentationResult",
    "LabelExportResult",
    "MANUAL_SEED_PROFILE_NAME",
    "MANUAL_SEED_V2_PROFILE_NAME",
    "MANUAL_CONTOUR_MAX_DIAMETER_UM",
    "MANUAL_CONTOUR_MIN_DIAMETER_UM",
    "ManualSeedFinalizeResult",
    "HtmlReportResult",
    "OocyteDetectionResult",
    "OocyteProfilingResult",
    "OocyteReleaseResult",
    "OocyteReleaseSample",
    "OocyteReleaseSpec",
    "OocyteSampleInput",
    "PRECISION_BOUNDARY_CHOICES",
    "PRECISION_MANUAL_BOUNDARY_CHOICES",
    "PRECISION_MANUAL_BOUNDARY_RENDERER_VERSION",
    "PRECISION_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION",
    "PRECISION_RESOLVED_PROFILE_NAME",
    "PRECISION_RESOLVED_V2_PROFILE_NAME",
    "PrecisionBoundaryFinalizeResult",
    "PrecisionBoundaryReviewResult",
    "PrecisionManualBoundaryFinalizeResult",
    "PrecisionManualBoundaryReviewResult",
    "RefinedDetectionResult",
    "RecallReviewBundle",
    "RecallReviewRuntime",
    "RecallMaskOverlay",
    "REVIEWED_OVERLAY_CANDIDATE_FILES",
    "RescueDeltaBatchResult",
    "ReviewPackResult",
    "ScoredCandidateMask",
    "SHAPE_REVIEW_CHOICES",
    "SHAPE_RECOVERY_PARAMETERS",
    "ShapeRecoveryReviewResult",
    "SpatialQcResult",
    "SegmentationMetrics",
    "available_profiles",
    "accepted_duplicate_suspects",
    "analyze_recall_review",
    "build_oocyte_release",
    "candidate_score",
    "classify_recall_failure",
    "compile_batch_spatial_qc",
    "detect_coarse_candidates",
    "detect_oocyte_batch",
    "detect_oocytes",
    "export_whole_slide_labels",
    "finalize_manual_seed_review",
    "finalize_precision_boundary_review",
    "finalize_precision_manual_boundary_review",
    "finalize_shape_recovery_review",
    "find_channel_index",
    "get_profile",
    "generate_review_pack",
    "generate_rescue_delta_batch",
    "generate_html_reports",
    "generate_batch_recall_review_bundle",
    "generate_recall_review_bundle",
    "generate_precision_boundary_review",
    "generate_precision_manual_boundary_review",
    "generate_shape_recovery_review",
    "algorithm_document_html",
    "load_candidate_mask",
    "load_oocyte_release_spec",
    "load_recall_mask_overlay",
    "load_sample_manifest",
    "profile_oocyte_labels",
    "overlay_dir_from_identity",
    "read_ome_channel_patch",
    "refine_candidates_from_array",
    "render_spatial_overview",
    "save_candidate_mask",
    "scan_coarse_candidates",
    "scan_refined_candidates",
    "segment_oocyte_patch",
    "serve_recall_review",
    "serve_batch_recall_review",
    "validate_oocyte_release",
]
