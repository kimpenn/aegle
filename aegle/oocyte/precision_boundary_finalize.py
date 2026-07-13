"""Finalize reviewed Precision decisions into an immutable intermediate label set."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .export import LabelExportResult, export_whole_slide_labels
from .io import load_candidate_mask
from .manual_seed_finalize import (
    _atomic_write_csv,
    _overlap_metrics,
    _tight_mask,
    _write_reviewed_mask,
)
from .models import PersistedMask
from .precision_boundary_review import (
    _note_tokens,
    _review_key,
    _validate_precision_review,
)
from .recall_review import (
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _mask_path,
    _read_json,
)


PRECISION_RESOLVED_PROFILE_NAME = "precision_resolved_v1"
PRECISION_BOUNDARY_CHOICES = {
    "keep_current",
    "use_conservative",
    "use_expanded",
    "needs_manual",
    "exclude",
    "unsure",
}
_ACCEPTED_BOUNDARY_CHOICES = {
    "keep_current",
    "use_conservative",
    "use_expanded",
}
_PROPOSAL_VARIANTS = {
    "use_conservative": "conservative",
    "use_expanded": "expanded",
}
_PROPOSAL_METRICS = (
    "area_px",
    "equivalent_diameter_um",
    "circularity",
    "solidity",
    "centroid_offset_px",
)


@dataclass(frozen=True)
class PrecisionBoundaryFinalizeResult:
    out_dir: Path
    decisions_path: Path
    candidates_path: Path
    manual_queue_path: Path
    overlap_audit_path: Path
    labels: LabelExportResult
    manifest_path: Path
    resolved_count: int
    unresolved_manual_count: int
    excluded_count: int


def _clean_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _bool_value(value: Any, *, field: str) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value for {field}: {value!r}")


def _safe_mask_name(review_key: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "__", review_key).strip("._-")
    if not value:
        raise ValueError(f"review key cannot form a mask filename: {review_key!r}")
    return f"{value}.npz"


def _atomic_copy(source: Path, destination: Path) -> Path:
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with source_path.open("rb") as source_handle, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as target_handle:
            temporary_path = Path(target_handle.name)
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def _validate_boundary_identity(
    sample_identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("Precision boundary review schema_version must be 1")
    if payload.get("review_type") != "oocyte_precision_boundary_review":
        raise ValueError(
            "review_type must be 'oocyte_precision_boundary_review'"
        )
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Precision boundary review identity is missing")
    for field, expected in sample_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"Precision boundary identity mismatch for {field}")
    for field in (
        "precision_review_json",
        "precision_review_json_sha256",
        "boundary_candidate_table",
        "boundary_candidate_table_sha256",
    ):
        if not identity.get(field):
            raise ValueError(f"Precision boundary identity is missing {field}")
    return identity


def _validate_boundary_rows(
    payload: Mapping[str, Any],
    candidates: pd.DataFrame,
) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, Dict[str, Any]]]:
    required = {
        "boundary_index",
        "review_key",
        "display_id",
        "detector_component_id",
        "detection_pass",
        "x",
        "y",
        "current_mask_path",
        "conservative_available",
        "expanded_available",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"Precision boundary candidate table is missing columns: {sorted(missing)}"
        )
    if candidates["review_key"].astype(str).duplicated().any():
        raise ValueError("Precision boundary candidate keys are not unique")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Precision boundary review rows must be a list")
    by_key = {
        str(row["review_key"]): row for row in candidates.to_dict("records")
    }
    review_by_key: Dict[str, Dict[str, Any]] = {}
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(
                f"Precision boundary review row {row_number} must be an object"
            )
        key = str(raw_row.get("review_key", ""))
        if key in review_by_key:
            raise ValueError(f"Precision boundary review has duplicate key {key}")
        candidate = by_key.get(key)
        if candidate is None:
            raise ValueError(f"Precision boundary review contains unknown key {key}")
        for field in (
            "boundary_index",
            "detector_component_id",
            "detection_pass",
        ):
            if str(raw_row.get(field, "")) != str(candidate[field]):
                raise ValueError(
                    f"Precision boundary row metadata changed for {key}: {field}"
                )
        for field in ("x", "y"):
            if not np.isclose(
                float(raw_row.get(field, np.nan)),
                float(candidate[field]),
                atol=0.01,
            ):
                raise ValueError(
                    f"Precision boundary coordinates changed for {key}: {field}"
                )
        choice = str(raw_row.get("boundary_review_choice", "")).strip()
        if choice not in PRECISION_BOUNDARY_CHOICES:
            raise ValueError(
                f"invalid or missing Precision boundary choice for {key}"
            )
        if choice == "unsure":
            raise ValueError(f"Precision boundary review remains unsure for {key}")
        review_by_key[key] = {
            "choice": choice,
            "notes": _clean_text(raw_row.get("boundary_review_notes")),
            "candidate": candidate,
        }
    if set(review_by_key) != set(by_key):
        raise ValueError(
            "Precision boundary review rows do not match its candidate table"
        )
    return rows, review_by_key


def _validate_current_mask(
    path: Path,
    *,
    sample_id: str,
    candidate_id: str,
    image_shape_yx: Tuple[int, int],
) -> PersistedMask:
    if not path.is_file():
        raise FileNotFoundError(f"current candidate mask is missing: {path}")
    persisted = load_candidate_mask(path)
    if persisted.image_shape_yx != image_shape_yx:
        raise ValueError(f"current candidate mask image shape mismatch: {path}")
    if str(persisted.metadata.get("sample_id", "")) != sample_id:
        raise ValueError(f"current candidate mask sample mismatch: {path}")
    if str(persisted.metadata.get("candidate_id", "")) != candidate_id:
        raise ValueError(f"current candidate mask ID mismatch: {path}")
    if not persisted.mask.any():
        raise ValueError(f"current candidate mask is empty: {path}")
    return persisted


def _validate_proposal_mask(
    candidate: Mapping[str, Any],
    *,
    key: str,
    variant: str,
    boundary_root: Path,
    image_shape_yx: Tuple[int, int],
) -> Tuple[Path, PersistedMask]:
    available_field = f"{variant}_available"
    if not _bool_value(candidate.get(available_field), field=available_field):
        raise ValueError(f"selected {variant} proposal is unavailable for {key}")
    path_value = _clean_text(candidate.get(f"{variant}_mask_path"))
    if not path_value:
        raise ValueError(f"selected {variant} proposal has no path for {key}")
    path = Path(path_value).resolve()
    if not path.is_relative_to(boundary_root):
        raise ValueError(f"selected proposal is outside its review pack: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"selected proposal is missing: {path}")
    persisted = load_candidate_mask(path)
    if persisted.image_shape_yx != image_shape_yx:
        raise ValueError(f"selected proposal image shape mismatch: {path}")
    if str(persisted.metadata.get("annotation_id", "")) != key:
        raise ValueError(f"selected proposal annotation mismatch: {path}")
    metrics = persisted.metadata.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"selected proposal has no metrics: {path}")
    if int(persisted.mask.sum()) != int(metrics.get("area_px", -1)):
        raise ValueError(f"selected proposal mask area does not match metadata: {path}")
    for name in _PROPOSAL_METRICS:
        table_value = float(candidate.get(f"{variant}_{name}", np.nan))
        metadata_value = float(metrics.get(name, np.nan))
        if not np.isfinite(table_value) or not np.isfinite(metadata_value):
            raise ValueError(f"selected proposal metric {name} is missing: {path}")
        if not np.isclose(table_value, metadata_value, rtol=1e-7, atol=1e-7):
            raise ValueError(
                f"selected proposal metric {name} does not match table: {path}"
            )
    return path, persisted


def _resolved_overlap_audit(
    masks: Sequence[Tuple[str, PersistedMask]],
) -> pd.DataFrame:
    rows = []
    for left_index, (left_key, left) in enumerate(masks):
        for right_key, right in masks[left_index + 1 :]:
            pixels, left_fraction, right_fraction = _overlap_metrics(left, right)
            if not pixels:
                continue
            rows.append(
                {
                    "left_review_key": left_key,
                    "right_review_key": right_key,
                    "overlap_pixel_count": pixels,
                    "left_overlap_fraction": left_fraction,
                    "right_overlap_fraction": right_fraction,
                    "smaller_mask_overlap_fraction": max(
                        left_fraction, right_fraction
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "left_review_key",
            "right_review_key",
            "overlap_pixel_count",
            "left_overlap_fraction",
            "right_overlap_fraction",
            "smaller_mask_overlap_fraction",
        ],
    )


def _candidate_row(
    source: Mapping[str, Any],
    *,
    review_key: str,
    mask: PersistedMask,
    reviewed_path: Path,
    reviewed_sha256: str,
    destination: Path,
    final_source: str,
    precision_status: str,
    precision_notes: str,
    boundary_choice: str,
    boundary_notes: str,
    source_mask_path: Path,
    source_mask_sha256: str,
    precision_review_sha256: str,
    boundary_review_sha256: str,
) -> Dict[str, Any]:
    row = dict(source)
    source_metrics = mask.metadata.get("metrics", {})
    metrics = source_metrics if isinstance(source_metrics, Mapping) else {}
    ys, xs = np.nonzero(mask.mask)
    centroid_x = mask.bbox.x0 + float(xs.mean())
    centroid_y = mask.bbox.y0 + float(ys.mean())
    source_acceptance_mode = _clean_text(row.get("acceptance_mode"))
    source_segmentation_pass = _clean_text(row.get("segmentation_pass"))
    row.update(
        {
            "review_key": review_key,
            "resolved_oocyte_id": review_key,
            "source_center_x": source.get("center_x"),
            "source_center_y": source.get("center_y"),
            "source_acceptance_mode": source_acceptance_mode,
            "source_segmentation_pass": source_segmentation_pass,
            "accepted": True,
            "center_x": int(round(centroid_x)),
            "center_y": int(round(centroid_y)),
            "component_centroid_x": centroid_x,
            "component_centroid_y": centroid_y,
            "bbox_x0": mask.bbox.x0,
            "bbox_y0": mask.bbox.y0,
            "bbox_x1": mask.bbox.x1,
            "bbox_y1": mask.bbox.y1,
            "local_area_px": int(mask.mask.sum()),
            "local_equivalent_diameter_um": metrics.get(
                "equivalent_diameter_um"
            ),
            "local_major_axis_um": metrics.get("major_axis_um"),
            "local_minor_axis_um": metrics.get("minor_axis_um"),
            "local_eccentricity": metrics.get("eccentricity"),
            "local_solidity": metrics.get("solidity"),
            "local_circularity": metrics.get("circularity"),
            "local_centroid_offset_px": metrics.get("centroid_offset_px"),
            "local_mean_intensity": metrics.get("mean_intensity"),
            "local_max_intensity": metrics.get("max_intensity"),
            "threshold_method": metrics.get("threshold_method"),
            "threshold": metrics.get("threshold"),
            "selection_mode": metrics.get("selection_mode"),
            "acceptance_mode": final_source,
            "segmentation_pass": final_source,
            "quality_class": "precision_reviewed_boundary",
            "precision_review_status": precision_status,
            "precision_review_notes": precision_notes,
            "precision_boundary_choice": boundary_choice,
            "precision_boundary_notes": boundary_notes,
            "precision_resolution_source": final_source,
            "mask_path": str(reviewed_path.relative_to(destination)),
            "mask_source_dir": str(destination),
            "source_mask_path": str(source_mask_path),
            "source_mask_sha256": source_mask_sha256,
            "reviewed_mask_sha256": reviewed_sha256,
            "precision_review_json_sha256": precision_review_sha256,
            "precision_boundary_review_json_sha256": boundary_review_sha256,
            "duplicate_suppressed": False,
        }
    )
    return row


def finalize_precision_boundary_review(
    sample_dir: Path,
    precision_review_json: Path,
    boundary_review_json: Path,
    out_dir: Path,
    *,
    tile_shape_yx: Tuple[int, int] = (512, 512),
    max_smaller_overlap_fraction: float = 0.25,
) -> PrecisionBoundaryFinalizeResult:
    """Apply completed Precision and boundary decisions without claiming Recall."""

    sample = _load_sample(sample_dir)
    precision_path = Path(precision_review_json).resolve()
    boundary_review_path = Path(boundary_review_json).resolve()
    precision_payload = _read_json(precision_path)
    precision_rows = _validate_precision_review(
        sample.review_identity,
        sample.candidates,
        precision_payload,
    )
    for row in precision_rows:
        if str(row.get("manual_status", "")) == "unsure":
            raise ValueError(
                f"Precision review remains unsure for {row.get('review_key', '')}"
            )
    precision_sha256 = _file_sha256(precision_path)

    boundary_payload = _read_json(boundary_review_path)
    boundary_identity = _validate_boundary_identity(
        sample.review_identity,
        boundary_payload,
    )
    identity_precision_path = Path(
        str(boundary_identity["precision_review_json"])
    ).resolve()
    if identity_precision_path != precision_path:
        raise ValueError("Precision review path does not match boundary identity")
    if precision_sha256 != str(boundary_identity["precision_review_json_sha256"]):
        raise ValueError("Precision review SHA-256 does not match boundary identity")

    boundary_table_path = Path(
        str(boundary_identity["boundary_candidate_table"])
    ).resolve()
    if not boundary_table_path.is_file():
        raise FileNotFoundError(
            f"Precision boundary candidate table is missing: {boundary_table_path}"
        )
    boundary_table_sha256 = _file_sha256(boundary_table_path)
    if boundary_table_sha256 != str(
        boundary_identity["boundary_candidate_table_sha256"]
    ):
        raise ValueError("Precision boundary candidate table SHA-256 mismatch")
    boundary_root = boundary_table_path.parent.resolve()
    boundary_candidates = pd.read_csv(boundary_table_path)
    _, boundary_by_key = _validate_boundary_rows(
        boundary_payload,
        boundary_candidates,
    )
    boundary_review_sha256 = _file_sha256(boundary_review_path)

    candidate_records = sample.candidates.to_dict("records")
    candidate_by_key = {_review_key(row): row for row in candidate_records}
    if len(candidate_by_key) != len(candidate_records):
        raise ValueError("sample candidate review keys are not unique")
    component_ids = [str(row["detector_component_id"]) for row in candidate_records]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("sample detector_component_id values are not unique")
    expected_boundary_keys = {
        str(row["review_key"])
        for row in precision_rows
        if str(row.get("manual_status")) == "reject"
        and "true_oocyte" in _note_tokens(row.get("manual_notes"))
    }
    if set(boundary_by_key) != expected_boundary_keys:
        raise ValueError(
            "Precision boundary candidate set does not match true-oocyte rejects"
        )

    precision_by_key = {
        str(row["review_key"]): row for row in precision_rows
    }
    selections = []
    decisions = []
    manual_queue = []
    choice_counts: Dict[str, int] = {}
    false_positive_count = 0
    for review_index, precision_row in enumerate(precision_rows, start=1):
        key = str(precision_row["review_key"])
        source = candidate_by_key[key]
        status = str(precision_row["manual_status"])
        precision_notes = _clean_text(precision_row.get("manual_notes"))
        boundary = boundary_by_key.get(key)
        boundary_choice = "" if boundary is None else str(boundary["choice"])
        boundary_notes = "" if boundary is None else str(boundary["notes"])
        if boundary_choice:
            choice_counts[boundary_choice] = choice_counts.get(boundary_choice, 0) + 1

        accepted = False
        resolution_state = "excluded"
        final_source = "precision_reject"
        selected_variant = ""
        selected_path: Path | None = None
        selected_mask: PersistedMask | None = None
        if status == "accept":
            accepted = True
            resolution_state = "resolved"
            final_source = "precision_accept_current"
            selected_path = _mask_path(sample.sample_dir, source).resolve()
            selected_mask = _validate_current_mask(
                selected_path,
                sample_id=sample.sample_id,
                candidate_id=str(source["detector_component_id"]),
                image_shape_yx=sample.image_shape_yx,
            )
        elif boundary_choice in _ACCEPTED_BOUNDARY_CHOICES:
            accepted = True
            resolution_state = "resolved"
            if boundary_choice == "keep_current":
                final_source = "precision_boundary_keep_current"
                selected_path = _mask_path(sample.sample_dir, source).resolve()
                table_current_path = Path(
                    str(boundary["candidate"]["current_mask_path"])
                ).resolve()
                if table_current_path != selected_path:
                    raise ValueError(
                        f"current boundary mask path changed for {key}"
                    )
                selected_mask = _validate_current_mask(
                    selected_path,
                    sample_id=sample.sample_id,
                    candidate_id=str(source["detector_component_id"]),
                    image_shape_yx=sample.image_shape_yx,
                )
            else:
                selected_variant = _PROPOSAL_VARIANTS[boundary_choice]
                final_source = f"precision_boundary_{selected_variant}"
                selected_path, selected_mask = _validate_proposal_mask(
                    boundary["candidate"],
                    key=key,
                    variant=selected_variant,
                    boundary_root=boundary_root,
                    image_shape_yx=sample.image_shape_yx,
                )
        elif boundary_choice == "needs_manual":
            resolution_state = "manual_boundary_required"
            final_source = "precision_boundary_needs_manual"
            manual_queue.append(
                {
                    "review_index": review_index,
                    "boundary_index": int(boundary["candidate"]["boundary_index"]),
                    "review_key": key,
                    "display_id": str(source.get("display_id", "")),
                    "detector_component_id": str(source["detector_component_id"]),
                    "detection_pass": str(source["detection_pass"]),
                    "center_x": float(source["center_x"]),
                    "center_y": float(source["center_y"]),
                    "precision_notes": precision_notes,
                    "boundary_review_notes": boundary_notes,
                    "current_mask_path": str(
                        Path(str(boundary["candidate"]["current_mask_path"])).resolve()
                    ),
                    "required_action": "draw_and_review_manual_boundary",
                }
            )
        elif boundary_choice == "exclude":
            final_source = "precision_boundary_excluded"
        elif status == "reject":
            false_positive_count += 1
        else:
            raise ValueError(f"unsupported Precision decision for {key}")

        source_sha256 = ""
        if accepted:
            if selected_path is None or selected_mask is None:
                raise AssertionError(f"accepted selection has no mask: {key}")
            source_sha256 = _file_sha256(selected_path)
            selections.append(
                {
                    "review_index": review_index,
                    "review_key": key,
                    "source": source,
                    "precision_status": status,
                    "precision_notes": precision_notes,
                    "boundary_choice": boundary_choice,
                    "boundary_notes": boundary_notes,
                    "selected_variant": selected_variant,
                    "final_source": final_source,
                    "source_path": selected_path,
                    "source_sha256": source_sha256,
                    "mask": _tight_mask(selected_mask),
                }
            )
        decisions.append(
            {
                "review_index": review_index,
                "review_key": key,
                "display_id": str(source.get("display_id", "")),
                "detector_component_id": str(source["detector_component_id"]),
                "detection_pass": str(source["detection_pass"]),
                "center_x": float(source["center_x"]),
                "center_y": float(source["center_y"]),
                "precision_status": status,
                "precision_notes": precision_notes,
                "boundary_review_choice": boundary_choice,
                "boundary_review_notes": boundary_notes,
                "final_accepted": accepted,
                "resolution_state": resolution_state,
                "final_source": final_source,
                "selected_variant": selected_variant,
                "source_mask_path": "" if selected_path is None else str(selected_path),
                "source_mask_sha256": source_sha256,
                "reviewed_mask_path": "",
                "reviewed_mask_sha256": "",
            }
        )

    if set(precision_by_key) != set(candidate_by_key):
        raise AssertionError("validated Precision review unexpectedly changed candidate set")
    audit = _resolved_overlap_audit(
        [(str(item["review_key"]), item["mask"]) for item in selections]
    )
    blocking = audit[
        audit["smaller_mask_overlap_fraction"] >= max_smaller_overlap_fraction
    ]
    if not blocking.empty:
        pairs = ", ".join(
            f"{row.left_review_key}/{row.right_review_key}="
            f"{row.smaller_mask_overlap_fraction:.3f}"
            for row in blocking.itertuples()
        )
        raise ValueError(f"Precision-resolved masks have blocking overlap: {pairs}")

    destination = Path(out_dir).resolve()
    manifest_path = destination / "precision_resolved_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"immutable Precision delivery already exists: {manifest_path}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    masks_dir = destination / "reviewed_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    expected_mask_names = {
        _safe_mask_name(str(item["review_key"])) for item in selections
    }
    for existing in masks_dir.glob("*.npz"):
        if existing.name not in expected_mask_names:
            existing.unlink()

    input_dir = destination / "review_inputs"
    input_copies = {
        "automatic_candidates": (
            sample.sample_dir / "html_candidates.csv",
            input_dir / "automatic_candidates.csv",
        ),
        "precision_review": (
            precision_path,
            input_dir / "precision_review.json",
        ),
        "boundary_candidates": (
            boundary_table_path,
            input_dir / "precision_boundary_candidates.csv",
        ),
        "boundary_review": (
            boundary_review_path,
            input_dir / "precision_boundary_review.json",
        ),
    }
    for source_path, copy_path in input_copies.values():
        _atomic_copy(source_path, copy_path)
        if _file_sha256(source_path) != _file_sha256(copy_path):
            raise ValueError(f"review-input copy SHA-256 mismatch: {copy_path}")

    decision_by_key = {str(row["review_key"]): row for row in decisions}
    candidate_rows = []
    for item in selections:
        key = str(item["review_key"])
        reviewed_path = masks_dir / _safe_mask_name(key)
        metadata = {
            **dict(item["mask"].metadata),
            "schema_version": 1,
            "sample_id": sample.sample_id,
            "candidate_id": str(item["source"]["detector_component_id"]),
            "review_key": key,
            "resolved_oocyte_id": key,
            "profile_name": PRECISION_RESOLVED_PROFILE_NAME,
            "base_profile_name": sample.profile_name,
            "base_profile_fingerprint": sample.profile_fingerprint,
            "implementation_version": sample.implementation_version,
            "precision_resolved": True,
            "provisional_only": False,
            "precision_resolution_source": item["final_source"],
            "precision_review_status": item["precision_status"],
            "precision_review_notes": item["precision_notes"],
            "precision_boundary_choice": item["boundary_choice"],
            "precision_boundary_notes": item["boundary_notes"],
            "precision_review_json_sha256": precision_sha256,
            "precision_boundary_review_json_sha256": boundary_review_sha256,
            "boundary_candidate_table_sha256": boundary_table_sha256,
            "source_mask_path": str(item["source_path"]),
            "source_mask_sha256": item["source_sha256"],
        }
        _write_reviewed_mask(reviewed_path, mask=item["mask"], metadata=metadata)
        reviewed_sha256 = _file_sha256(reviewed_path)
        decision = decision_by_key[key]
        decision["reviewed_mask_path"] = str(reviewed_path)
        decision["reviewed_mask_sha256"] = reviewed_sha256
        candidate_rows.append(
            _candidate_row(
                item["source"],
                review_key=key,
                mask=item["mask"],
                reviewed_path=reviewed_path,
                reviewed_sha256=reviewed_sha256,
                destination=destination,
                final_source=str(item["final_source"]),
                precision_status=str(item["precision_status"]),
                precision_notes=str(item["precision_notes"]),
                boundary_choice=str(item["boundary_choice"]),
                boundary_notes=str(item["boundary_notes"]),
                source_mask_path=Path(item["source_path"]),
                source_mask_sha256=str(item["source_sha256"]),
                precision_review_sha256=precision_sha256,
                boundary_review_sha256=boundary_review_sha256,
            )
        )

    decisions_path = destination / "precision_review_decisions.csv"
    candidates_path = destination / "precision_resolved_candidates.csv"
    manual_queue_path = destination / "manual_boundary_queue.csv"
    overlap_audit_path = destination / "mask_overlap_audit.csv"
    decisions_table = pd.DataFrame(decisions)
    _atomic_write_csv(decisions_table, decisions_path)
    if candidate_rows:
        resolved_candidates = pd.DataFrame(candidate_rows)
    else:
        resolved_candidates = sample.candidates.iloc[:0].copy()
        empty_columns = {
            "accepted": "bool",
            "detector_score": "float64",
            "acceptance_mode": "object",
            "review_key": "object",
            "resolved_oocyte_id": "object",
            "precision_review_status": "object",
            "precision_review_notes": "object",
            "precision_boundary_choice": "object",
            "precision_boundary_notes": "object",
            "precision_resolution_source": "object",
            "reviewed_mask_sha256": "object",
        }
        for column, dtype in empty_columns.items():
            if column not in resolved_candidates:
                resolved_candidates[column] = pd.Series(dtype=dtype)
    _atomic_write_csv(resolved_candidates, candidates_path)
    manual_columns = [
        "review_index",
        "boundary_index",
        "review_key",
        "display_id",
        "detector_component_id",
        "detection_pass",
        "center_x",
        "center_y",
        "precision_notes",
        "boundary_review_notes",
        "current_mask_path",
        "required_action",
    ]
    _atomic_write_csv(pd.DataFrame(manual_queue, columns=manual_columns), manual_queue_path)
    _atomic_write_csv(audit, overlap_audit_path)

    labels = export_whole_slide_labels(
        resolved_candidates,
        sample_dir=destination,
        image_shape_yx=sample.image_shape_yx,
        image_path=destination / "oocyte_labels_precision_resolved_v1.ome.tiff",
        mapping_path=destination / "oocyte_labels_precision_resolved_v1_mapping.csv",
        tile_shape_yx=tile_shape_yx,
    )
    if labels.overlap_pixel_count:
        raise AssertionError("preflight and label-export overlap counts disagree")
    expected_pixels = sum(int(item["mask"].mask.sum()) for item in selections)
    if labels.assigned_pixel_count != expected_pixels:
        raise AssertionError("label-export pixel count does not match resolved masks")

    excluded_count = int((~decisions_table["final_accepted"]).sum()) - len(manual_queue)
    artifact_paths = [
        decisions_path,
        candidates_path,
        manual_queue_path,
        overlap_audit_path,
        labels.image_path,
        labels.mapping_path,
        *(copy_path for _, copy_path in input_copies.values()),
        *(Path(row["reviewed_mask_path"]) for row in decisions if row["reviewed_mask_path"]),
    ]
    source_counts = (
        resolved_candidates["precision_resolution_source"]
        .value_counts()
        .sort_index()
        .to_dict()
        if len(resolved_candidates)
        else {}
    )
    manifest = {
        "schema_version": 1,
        "delivery_name": PRECISION_RESOLVED_PROFILE_NAME,
        "delivery_status": "intermediate_precision_only",
        "release_ready": False,
        "precision_complete": True,
        "manual_boundary_complete": len(manual_queue) == 0,
        "recall_complete": False,
        "sample": sample.review_identity,
        "precision_review_json": str(precision_path),
        "precision_review_json_sha256": precision_sha256,
        "precision_review_exported_at": precision_payload.get("exported_at"),
        "precision_boundary_review_json": str(boundary_review_path),
        "precision_boundary_review_json_sha256": boundary_review_sha256,
        "precision_boundary_review_exported_at": boundary_payload.get("exported_at"),
        "boundary_candidate_table": str(boundary_table_path),
        "boundary_candidate_table_sha256": boundary_table_sha256,
        "automatic_candidate_count": len(sample.candidates),
        "resolved_label_count": labels.label_count,
        "unresolved_manual_count": len(manual_queue),
        "excluded_count": excluded_count,
        "false_positive_count": false_positive_count,
        "boundary_choice_counts": choice_counts,
        "resolution_source_counts": source_counts,
        "unresolved_review_keys": [row["review_key"] for row in manual_queue],
        "overlap_audit_row_count": len(audit),
        "max_smaller_overlap_fraction": (
            0.0
            if audit.empty
            else float(audit["smaller_mask_overlap_fraction"].max())
        ),
        "overlap_blocking_threshold": max_smaller_overlap_fraction,
        "label_export": {
            "label_count": labels.label_count,
            "assigned_pixel_count": labels.assigned_pixel_count,
            "overlap_pixel_count": labels.overlap_pixel_count,
        },
        "production_outputs_modified": False,
        "review_pack_outputs_modified": False,
        "input_copies": {
            name: {
                "source_path": str(source_path),
                "copied_path": str(copy_path),
                "sha256": _file_sha256(copy_path),
            }
            for name, (source_path, copy_path) in input_copies.items()
        },
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
    return PrecisionBoundaryFinalizeResult(
        out_dir=destination,
        decisions_path=decisions_path,
        candidates_path=candidates_path,
        manual_queue_path=manual_queue_path,
        overlap_audit_path=overlap_audit_path,
        labels=labels,
        manifest_path=manifest_path,
        resolved_count=labels.label_count,
        unresolved_manual_count=len(manual_queue),
        excluded_count=excluded_count,
    )


__all__ = [
    "PRECISION_BOUNDARY_CHOICES",
    "PRECISION_RESOLVED_PROFILE_NAME",
    "PrecisionBoundaryFinalizeResult",
    "finalize_precision_boundary_review",
]
