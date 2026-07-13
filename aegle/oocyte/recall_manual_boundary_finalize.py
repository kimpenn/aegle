"""Finalize reviewed Recall polygons into a new manual-seed v2 delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .export import LabelExportResult, export_whole_slide_labels
from .io import load_candidate_mask
from .manual_seed_finalize import (
    _atomic_write_csv,
    _overlap_metrics,
    _write_reviewed_mask,
)
from .precision_boundary_finalize import _atomic_copy
from .precision_manual_boundary_finalize import (
    _clean_text,
    _normalize_vertices,
    _rasterize_polygon,
)
from .recall_manual_boundary_review import (
    RECALL_MANUAL_BOUNDARY_REVIEW_TYPE,
    _resolved_mask_path,
    _verify_manual_seed_delivery,
)
from .recall_overlay import overlay_dir_from_identity
from .recall_review import (
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _read_json,
)


RECALL_MANUAL_CONTOUR_PROFILE_NAME = "manual_seed_manual_contour_v1"
RECALL_MANUAL_BOUNDARY_CHOICES = {
    "accept_manual_contour",
    "exclude",
    "unsure",
}


@dataclass(frozen=True)
class RecallManualBoundaryFinalizeResult:
    out_dir: Path
    decisions_path: Path
    candidates_path: Path
    combined_candidates_path: Path
    overlap_audit_path: Path
    labels: LabelExportResult
    manifest_path: Path
    manual_added_count: int
    manual_excluded_count: int
    combined_label_count: int


def _validate_review_identity(
    sample_identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("Recall manual-boundary review schema_version must be 1")
    if payload.get("review_type") != RECALL_MANUAL_BOUNDARY_REVIEW_TYPE:
        raise ValueError(f"review_type must be {RECALL_MANUAL_BOUNDARY_REVIEW_TYPE!r}")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Recall manual-boundary review identity is missing")
    for field, expected in sample_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"Recall manual-boundary identity mismatch for {field}")
    required = (
        "base_manual_seed_dir",
        "base_manual_seed_manifest",
        "base_manual_seed_manifest_sha256",
        "base_combined_candidates_sha256",
        "manual_seed_review",
        "manual_seed_review_sha256",
        "recall_manual_boundary_candidate_table",
        "recall_manual_boundary_candidate_table_sha256",
    )
    for field in required:
        if not identity.get(field):
            raise ValueError(f"Recall manual-boundary identity is missing {field}")
    return identity


def _validate_review_rows(
    payload: Mapping[str, Any],
    candidate_table: pd.DataFrame,
    *,
    image_shape_yx: Tuple[int, int],
) -> Dict[str, Dict[str, Any]]:
    if candidate_table["review_key"].astype(str).duplicated().any():
        raise ValueError("Recall manual-boundary candidate keys are not unique")
    candidates = {
        str(row["review_key"]): row for row in candidate_table.to_dict("records")
    }
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Recall manual-boundary review rows must be a list")
    review_by_key: Dict[str, Dict[str, Any]] = {}
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"Recall manual-boundary row {row_number} must be an object")
        key = str(raw_row.get("review_key", ""))
        if key in review_by_key:
            raise ValueError(f"Recall manual-boundary review contains duplicate key {key}")
        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError(f"Recall manual-boundary review contains unknown key {key}")
        for field in (
            "manual_index",
            "review_index",
            "boundary_index",
            "annotation_id",
            "detector_component_id",
            "detection_pass",
        ):
            if str(raw_row.get(field, "")) != str(candidate[field]):
                raise ValueError(f"Recall manual-boundary metadata changed for {key}: {field}")
        for field in ("center_x", "center_y"):
            if not np.isclose(
                float(raw_row.get(field, np.nan)),
                float(candidate[field]),
                atol=0.01,
            ):
                raise ValueError(
                    f"Recall manual-boundary coordinates changed for {key}: {field}"
                )
        choice = str(raw_row.get("manual_boundary_choice", "")).strip()
        if choice not in RECALL_MANUAL_BOUNDARY_CHOICES:
            raise ValueError(
                f"invalid or missing Recall manual-boundary choice for {key}"
            )
        if choice == "unsure":
            raise ValueError(f"Recall manual-boundary review remains unsure for {key}")
        vertices: list[tuple[float, float]] = []
        if choice == "accept_manual_contour":
            vertices = _normalize_vertices(
                raw_row.get("vertices_xy"),
                candidate=candidate,
                image_shape_yx=image_shape_yx,
                key=key,
            )
        review_by_key[key] = {
            "choice": choice,
            "notes": _clean_text(raw_row.get("manual_boundary_notes")),
            "vertices": vertices,
            "candidate": candidate,
        }
    if set(review_by_key) != set(candidates):
        raise ValueError("Recall manual-boundary rows do not match the candidate table")
    return review_by_key


def _contour_candidate_row(
    *,
    source: Mapping[str, Any],
    mask_path: Path,
    mask_sha256: str,
    destination: Path,
    metrics: Mapping[str, Any],
    notes: str,
    review_sha256: str,
    candidate_table_sha256: str,
) -> Dict[str, Any]:
    review_index = int(source["review_index"])
    candidate_id = f"manual_contour_{review_index:03d}"
    mask = load_candidate_mask(mask_path)
    ys, xs = np.nonzero(mask.mask)
    centroid_x = mask.bbox.x0 + float(xs.mean())
    centroid_y = mask.bbox.y0 + float(ys.mean())
    return {
        "detector_component_id": candidate_id,
        "source_annotation_id": str(source["annotation_id"]),
        "display_id": f"#R{review_index:03d}",
        "html_id": f"manual-contour-{review_index:03d}",
        "accepted": True,
        "accepted_strict": False,
        "accepted_rescue": False,
        "detector_score": 1.0,
        "acceptance_mode": "manual_seed_reviewed_contour",
        "detection_pass": RECALL_MANUAL_CONTOUR_PROFILE_NAME,
        "segmentation_pass": "manual_polygon",
        "center_x": int(round(centroid_x)),
        "center_y": int(round(centroid_y)),
        "component_centroid_x": centroid_x,
        "component_centroid_y": centroid_y,
        "bbox_x0": mask.bbox.x0,
        "bbox_y0": mask.bbox.y0,
        "bbox_x1": mask.bbox.x1,
        "bbox_y1": mask.bbox.y1,
        "local_area_px": int(mask.mask.sum()),
        "local_equivalent_diameter_um": metrics.get("equivalent_diameter_um"),
        "local_major_axis_um": metrics.get("major_axis_um"),
        "local_minor_axis_um": metrics.get("minor_axis_um"),
        "local_eccentricity": metrics.get("eccentricity"),
        "local_solidity": metrics.get("solidity"),
        "local_circularity": metrics.get("circularity"),
        "local_centroid_offset_px": metrics.get("centroid_offset_px"),
        "local_mean_intensity": metrics.get("mean_intensity"),
        "local_max_intensity": metrics.get("max_intensity"),
        "threshold_method": "manual_polygon",
        "threshold": None,
        "selection_mode": "reviewed_manual_polygon",
        "failure_class": "manual_boundary_required",
        "manual_review_index": review_index,
        "manual_mask_choice": "accept_manual_contour",
        "manual_notes": notes,
        "boundary_warning": False,
        "quality_class": "reviewed_manual_contour",
        "mask_path": str(mask_path.relative_to(destination)),
        "mask_source_dir": str(destination),
        "reviewed_mask_sha256": mask_sha256,
        "manual_boundary_review_sha256": review_sha256,
        "manual_boundary_candidate_table_sha256": candidate_table_sha256,
        "duplicate_suppressed": False,
    }


def finalize_recall_manual_boundary_review(
    sample_dir: Path,
    base_finalize_dir: Path,
    review_json: Path,
    out_dir: Path,
    *,
    tile_shape_yx: Tuple[int, int] = (512, 512),
) -> RecallManualBoundaryFinalizeResult:
    """Validate reviewed Recall contours and compose an immutable v2 delivery."""

    review_path = Path(review_json).resolve()
    payload = _read_json(review_path)
    raw_identity = payload.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("Recall manual-boundary review identity is missing")
    sample = _load_sample(
        sample_dir,
        overlay_dir=overlay_dir_from_identity(raw_identity),
    )
    identity = _validate_review_identity(sample.review_identity, payload)
    base_dir = Path(base_finalize_dir).resolve()
    if base_dir != Path(str(identity["base_manual_seed_dir"])).resolve():
        raise ValueError("requested base manual-seed directory does not match review")
    base_manifest_path = base_dir / "manual_seed_finalize_manifest.json"
    if base_manifest_path != Path(str(identity["base_manual_seed_manifest"])).resolve():
        raise ValueError("base manual-seed manifest path does not match review")
    if _file_sha256(base_manifest_path) != str(
        identity["base_manual_seed_manifest_sha256"]
    ):
        raise ValueError("base manual-seed manifest SHA-256 mismatch")
    manual_seed_review_path = Path(str(identity["manual_seed_review"])).resolve()
    if _file_sha256(manual_seed_review_path) != str(
        identity["manual_seed_review_sha256"]
    ):
        raise ValueError("manual-seed review SHA-256 mismatch")
    _verify_manual_seed_delivery(
        base_dir,
        sample_identity=sample.review_identity,
        review_sha256=str(identity["manual_seed_review_sha256"]),
    )
    combined_path = base_dir / "oocyte_candidates_rescue_v1_plus_manual_seed_v1.csv"
    if _file_sha256(combined_path) != str(identity["base_combined_candidates_sha256"]):
        raise ValueError("base combined candidate table SHA-256 mismatch")
    candidate_table_path = Path(
        str(identity["recall_manual_boundary_candidate_table"])
    ).resolve()
    if not candidate_table_path.is_file():
        raise FileNotFoundError(
            f"Recall manual-boundary candidate table is missing: {candidate_table_path}"
        )
    if _file_sha256(candidate_table_path) != str(
        identity["recall_manual_boundary_candidate_table_sha256"]
    ):
        raise ValueError("Recall manual-boundary candidate table SHA-256 mismatch")
    candidates = pd.read_csv(candidate_table_path)
    review_by_key = _validate_review_rows(
        payload,
        candidates,
        image_shape_yx=sample.image_shape_yx,
    )
    base_candidates = pd.read_csv(combined_path)
    base_masks = []
    normalized_base_rows = []
    for row in base_candidates.to_dict("records"):
        mask_path = _resolved_mask_path(base_dir, row)
        if not mask_path.is_file():
            raise FileNotFoundError(f"base combined mask is missing: {mask_path}")
        persisted = load_candidate_mask(mask_path)
        if persisted.image_shape_yx != sample.image_shape_yx:
            raise ValueError(f"base combined mask shape mismatch: {mask_path}")
        base_masks.append((str(row["detector_component_id"]), persisted))
        normalized = dict(row)
        normalized["mask_path"] = str(mask_path)
        normalized["mask_source_dir"] = ""
        normalized_base_rows.append(normalized)

    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "recall_manual_boundary_finalize_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    masks_dir = destination / "reviewed_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    review_sha256 = _file_sha256(review_path)
    candidate_table_sha256 = _file_sha256(candidate_table_path)
    decisions = []
    contour_rows = []
    overlap_rows = []
    accepted_contours = []
    expected_masks = set()
    for key, item in review_by_key.items():
        source = item["candidate"]
        choice = str(item["choice"])
        decision: Dict[str, Any] = {
            "review_key": key,
            "annotation_id": str(source["annotation_id"]),
            "review_index": int(source["review_index"]),
            "center_x": float(source["center_x"]),
            "center_y": float(source["center_y"]),
            "manual_boundary_choice": choice,
            "manual_boundary_notes": item["notes"],
            "accepted": choice == "accept_manual_contour",
            "reviewed_mask_path": "",
            "reviewed_mask_sha256": "",
            "vertex_count": len(item["vertices"]),
        }
        if choice == "accept_manual_contour":
            mask, metrics = _rasterize_polygon(
                item["vertices"],
                center_xy=(float(source["center_x"]), float(source["center_y"])),
                image_shape_yx=sample.image_shape_yx,
                pixel_size_um=sample.pixel_size_um,
                key=key,
            )
            for base_id, base_mask in base_masks:
                pixels, contour_fraction, base_fraction = _overlap_metrics(
                    mask,
                    base_mask,
                )
                if pixels:
                    overlap_rows.append(
                        {
                            "manual_id": key,
                            "base_id": base_id,
                            "overlap_pixel_count": pixels,
                            "manual_overlap_fraction": contour_fraction,
                            "base_overlap_fraction": base_fraction,
                        }
                    )
            for prior_id, prior_mask in accepted_contours:
                pixels, contour_fraction, prior_fraction = _overlap_metrics(
                    mask,
                    prior_mask,
                )
                if pixels:
                    overlap_rows.append(
                        {
                            "manual_id": key,
                            "base_id": prior_id,
                            "overlap_pixel_count": pixels,
                            "manual_overlap_fraction": contour_fraction,
                            "base_overlap_fraction": prior_fraction,
                        }
                    )
            if overlap_rows:
                pairs = ", ".join(
                    f"{row['manual_id']}/{row['base_id']}={row['overlap_pixel_count']} px"
                    for row in overlap_rows
                )
                raise ValueError(f"manual contour overlaps resolved masks: {pairs}")
            review_index = int(source["review_index"])
            candidate_id = f"manual_contour_{review_index:03d}"
            reviewed_path = masks_dir / f"{candidate_id}.npz"
            expected_masks.add(reviewed_path.name)
            metadata = {
                **dict(mask.metadata),
                "schema_version": 1,
                "sample_id": sample.sample_id,
                "candidate_id": candidate_id,
                "profile_name": RECALL_MANUAL_CONTOUR_PROFILE_NAME,
                "base_profile_name": sample.profile_name,
                "base_profile_fingerprint": sample.profile_fingerprint,
                "implementation_version": sample.implementation_version,
                "reviewed_manual_seed": True,
                "reviewed_manual_contour": True,
                "provisional_only": False,
                "source_annotation_id": str(source["annotation_id"]),
                "manual_notes": item["notes"],
                "manual_boundary_review_sha256": review_sha256,
                "manual_boundary_candidate_table_sha256": candidate_table_sha256,
                "metrics": metrics,
            }
            _write_reviewed_mask(reviewed_path, mask=mask, metadata=metadata)
            mask_sha256 = _file_sha256(reviewed_path)
            decision["reviewed_mask_path"] = str(reviewed_path)
            decision["reviewed_mask_sha256"] = mask_sha256
            contour_rows.append(
                _contour_candidate_row(
                    source=source,
                    mask_path=reviewed_path,
                    mask_sha256=mask_sha256,
                    destination=destination,
                    metrics=metrics,
                    notes=item["notes"],
                    review_sha256=review_sha256,
                    candidate_table_sha256=candidate_table_sha256,
                )
            )
            accepted_contours.append((key, mask))
        decisions.append(decision)
    for existing in masks_dir.glob("*.npz"):
        if existing.name not in expected_masks:
            existing.unlink()

    decisions_path = destination / "recall_manual_boundary_decisions.csv"
    candidates_path = destination / "recall_manual_contour_candidates.csv"
    combined_candidates_path = destination / "oocyte_candidates_reviewed_manual_seed_v2.csv"
    overlap_audit_path = destination / "mask_overlap_audit.csv"
    overlap_columns = (
        "manual_id",
        "base_id",
        "overlap_pixel_count",
        "manual_overlap_fraction",
        "base_overlap_fraction",
    )
    _atomic_write_csv(pd.DataFrame(decisions), decisions_path)
    contour_table = pd.DataFrame(contour_rows)
    _atomic_write_csv(contour_table, candidates_path)
    combined = pd.DataFrame([*normalized_base_rows, *contour_rows])
    if combined["detector_component_id"].astype(str).duplicated().any():
        raise ValueError("v2 combined candidates contain duplicate IDs")
    _atomic_write_csv(combined, combined_candidates_path)
    _atomic_write_csv(
        pd.DataFrame(overlap_rows, columns=overlap_columns),
        overlap_audit_path,
    )
    labels = export_whole_slide_labels(
        combined,
        sample_dir=destination,
        image_shape_yx=sample.image_shape_yx,
        image_path=destination / "oocyte_labels_reviewed_manual_seed_v2.ome.tiff",
        mapping_path=destination / "oocyte_labels_reviewed_manual_seed_v2_mapping.csv",
        tile_shape_yx=tile_shape_yx,
    )
    review_inputs = destination / "review_inputs"
    copied_inputs = [
        _atomic_copy(
            base_manifest_path,
            review_inputs / "base_manual_seed_finalize_manifest.json",
        ),
        _atomic_copy(
            manual_seed_review_path,
            review_inputs / "manual_seed_mask_review.json",
        ),
        _atomic_copy(
            candidate_table_path,
            review_inputs / "recall_manual_boundary_candidates.csv",
        ),
        _atomic_copy(
            review_path,
            review_inputs / "recall_manual_boundary_review.json",
        ),
    ]
    artifact_paths = [
        decisions_path,
        candidates_path,
        combined_candidates_path,
        overlap_audit_path,
        labels.image_path,
        labels.mapping_path,
        *masks_dir.glob("*.npz"),
        *copied_inputs,
    ]
    manifest = {
        "schema_version": 1,
        "delivery_name": "reviewed_manual_seed_delta_v2",
        "sample": sample.review_identity,
        "review_identity": dict(identity),
        "base_manual_seed_dir": str(base_dir),
        "base_manual_seed_manifest": str(base_manifest_path),
        "base_manual_seed_manifest_sha256": _file_sha256(base_manifest_path),
        "manual_boundary_review": str(review_path),
        "manual_boundary_review_sha256": review_sha256,
        "manual_boundary_candidate_table": str(candidate_table_path),
        "manual_boundary_candidate_table_sha256": candidate_table_sha256,
        "base_label_count": len(base_candidates),
        "manual_boundary_card_count": len(candidates),
        "manual_added_count": len(contour_rows),
        "manual_excluded_count": sum(
            row["manual_boundary_choice"] == "exclude" for row in decisions
        ),
        "combined_label_count": labels.label_count,
        "overlap_pixel_count": labels.overlap_pixel_count,
        "assigned_pixel_count": labels.assigned_pixel_count,
        "remaining_manual_boundary_count": 0,
        "production_outputs_modified": False,
        "base_manual_seed_outputs_modified": False,
        "artifacts": {
            str(path.relative_to(destination)): {
                "path": str(path),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False),
    )
    return RecallManualBoundaryFinalizeResult(
        out_dir=destination,
        decisions_path=decisions_path,
        candidates_path=candidates_path,
        combined_candidates_path=combined_candidates_path,
        overlap_audit_path=overlap_audit_path,
        labels=labels,
        manifest_path=manifest_path,
        manual_added_count=len(contour_rows),
        manual_excluded_count=int(manifest["manual_excluded_count"]),
        combined_label_count=labels.label_count,
    )


__all__ = [
    "RECALL_MANUAL_CONTOUR_PROFILE_NAME",
    "RECALL_MANUAL_BOUNDARY_CHOICES",
    "RecallManualBoundaryFinalizeResult",
    "finalize_recall_manual_boundary_review",
]
