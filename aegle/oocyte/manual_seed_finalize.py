"""Finalize reviewed manual-seed masks into versioned label artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .export import LabelExportResult, export_whole_slide_labels
from .io import load_candidate_mask
from .models import BoundingBox, PersistedMask
from .recall_review import (
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _mask_path,
    _read_json,
)
from .recall_overlay import overlay_dir_from_identity


MANUAL_SEED_PROFILE_NAME = "manual_seed_review_v1"
ACCEPTED_CHOICES = {
    "accept_manual_conservative": "conservative",
    "accept_manual_expanded": "expanded",
}
ALLOWED_CHOICES = set(ACCEPTED_CHOICES) | {"neither", "duplicate", "unsure"}


@dataclass(frozen=True)
class ManualSeedFinalizeResult:
    out_dir: Path
    decisions_path: Path
    candidates_path: Path
    overlap_audit_path: Path
    delta_labels: LabelExportResult
    combined_labels: LabelExportResult | None
    combined_candidates_path: Path | None
    manifest_path: Path
    accepted_count: int
    boundary_warning_count: int


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
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


def _boundary_warning(notes: str) -> bool:
    normalized = notes.strip().casefold()
    return bool(
        normalized
        and any(
            term in normalized
            for term in (
                "boundary",
                "misses part",
                "undersegment",
                "under-segment",
                "oversegment",
                "over-segment",
            )
        )
    )


def _tight_mask(mask: PersistedMask) -> PersistedMask:
    ys, xs = np.nonzero(mask.mask)
    if not len(xs):
        raise ValueError("reviewed mask contains no foreground pixels")
    local_x0, local_x1 = int(xs.min()), int(xs.max()) + 1
    local_y0, local_y1 = int(ys.min()), int(ys.max()) + 1
    bbox = BoundingBox(
        mask.bbox.x0 + local_x0,
        mask.bbox.y0 + local_y0,
        mask.bbox.x0 + local_x1,
        mask.bbox.y0 + local_y1,
    )
    return PersistedMask(
        mask=np.asarray(
            mask.mask[local_y0:local_y1, local_x0:local_x1],
            dtype=np.bool_,
        ),
        bbox=bbox,
        image_shape_yx=mask.image_shape_yx,
        metadata=mask.metadata,
    )


def _write_reviewed_mask(
    path: Path,
    *,
    mask: PersistedMask,
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                mask=mask.mask,
                bbox_xyxy=np.asarray(mask.bbox.as_tuple(), dtype=np.int64),
                image_shape_yx=np.asarray(mask.image_shape_yx, dtype=np.int64),
                metadata_json=np.asarray(
                    json.dumps(
                        _json_safe(dict(metadata)),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _overlap_metrics(left: PersistedMask, right: PersistedMask) -> Tuple[int, float, float]:
    x0 = max(left.bbox.x0, right.bbox.x0)
    y0 = max(left.bbox.y0, right.bbox.y0)
    x1 = min(left.bbox.x1, right.bbox.x1)
    y1 = min(left.bbox.y1, right.bbox.y1)
    if x0 >= x1 or y0 >= y1:
        return 0, 0.0, 0.0
    left_region = left.mask[
        y0 - left.bbox.y0 : y1 - left.bbox.y0,
        x0 - left.bbox.x0 : x1 - left.bbox.x0,
    ]
    right_region = right.mask[
        y0 - right.bbox.y0 : y1 - right.bbox.y0,
        x0 - right.bbox.x0 : x1 - right.bbox.x0,
    ]
    overlap = int(np.logical_and(left_region, right_region).sum())
    return (
        overlap,
        overlap / max(int(left.mask.sum()), 1),
        overlap / max(int(right.mask.sum()), 1),
    )


def _overlap_audit(
    manual_masks: Sequence[Tuple[str, PersistedMask]],
    production_masks: Sequence[Tuple[str, PersistedMask]],
) -> pd.DataFrame:
    rows = []

    def add_overlap(
        left_scope: str,
        left_id: str,
        left: PersistedMask,
        right_scope: str,
        right_id: str,
        right: PersistedMask,
    ) -> None:
        pixels, left_fraction, right_fraction = _overlap_metrics(left, right)
        if not pixels:
            return
        rows.append(
            {
                "left_scope": left_scope,
                "left_id": left_id,
                "right_scope": right_scope,
                "right_id": right_id,
                "overlap_pixel_count": pixels,
                "left_overlap_fraction": left_fraction,
                "right_overlap_fraction": right_fraction,
                "smaller_mask_overlap_fraction": max(left_fraction, right_fraction),
            }
        )

    for left_index, (left_id, left) in enumerate(manual_masks):
        for right_id, right in manual_masks[left_index + 1 :]:
            add_overlap("manual", left_id, left, "manual", right_id, right)
        for right_id, right in production_masks:
            add_overlap("manual", left_id, left, "production", right_id, right)
    columns = [
        "left_scope",
        "left_id",
        "right_scope",
        "right_id",
        "overlap_pixel_count",
        "left_overlap_fraction",
        "right_overlap_fraction",
        "smaller_mask_overlap_fraction",
    ]
    return pd.DataFrame(rows, columns=columns)


def _validate_review_identity(sample_identity: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("manual-seed review schema_version must be 1")
    if payload.get("review_type") != "manual_seed_mask_review":
        raise ValueError("review_type must be 'manual_seed_mask_review'")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("manual-seed review is missing its identity object")
    for field, expected in sample_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"manual-seed review identity mismatch for {field}")
    if not identity.get("analysis_sha256"):
        raise ValueError("manual-seed review identity is missing analysis_sha256")


def finalize_manual_seed_review(
    sample_dir: Path,
    review_json: Path,
    out_dir: Path,
    *,
    analysis_dir: Path | None = None,
    write_combined_labels: bool = True,
    tile_shape_yx: Tuple[int, int] = (512, 512),
    max_smaller_overlap_fraction: float = 0.25,
) -> ManualSeedFinalizeResult:
    """Validate a completed boundary review and write immutable label artifacts."""

    review_path = Path(review_json).resolve()
    payload = _read_json(review_path)
    review_identity = payload.get("identity")
    if not isinstance(review_identity, Mapping):
        raise ValueError("manual-seed review is missing its identity object")
    sample = _load_sample(
        sample_dir,
        overlay_dir=overlay_dir_from_identity(review_identity),
    )
    _validate_review_identity(sample.review_identity, payload)
    identity = payload["identity"]
    if analysis_dir is None:
        analysis_path = Path(str(identity.get("analysis_table", ""))).resolve()
    else:
        analysis_path = Path(analysis_dir).resolve() / "recall_failure_analysis.csv"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"recall analysis table does not exist: {analysis_path}")
    analysis_sha256 = _file_sha256(analysis_path)
    if analysis_sha256 != str(identity["analysis_sha256"]):
        raise ValueError("recall analysis SHA-256 does not match the review identity")

    review_rows = payload.get("rows")
    if not isinstance(review_rows, list):
        raise ValueError("manual-seed review rows must be a list")
    analysis = pd.read_csv(analysis_path)
    if analysis["annotation_id"].duplicated().any():
        raise ValueError("recall analysis contains duplicate annotation_id values")
    analysis_by_id = analysis.set_index("annotation_id", drop=False)
    review_ids = [str(row.get("annotation_id", "")) for row in review_rows]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("manual-seed review contains duplicate annotation_id values")
    if set(review_ids) != set(str(value) for value in analysis["annotation_id"]):
        raise ValueError("manual-seed review rows do not match the recall analysis")

    review_sha256 = _file_sha256(review_path)
    selections = []
    decisions = []
    choice_counts: Dict[str, int] = {}
    analysis_root = analysis_path.parent.resolve()
    for review_index, raw_row in enumerate(review_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"manual-seed review row {review_index} must be an object")
        annotation_id = str(raw_row["annotation_id"])
        source = analysis_by_id.loc[annotation_id]
        if not (
            np.isclose(float(raw_row["x"]), float(source["x"]), atol=0.01)
            and np.isclose(float(raw_row["y"]), float(source["y"]), atol=0.01)
        ):
            raise ValueError(f"manual-seed review coordinates changed for {annotation_id}")
        choice = str(raw_row.get("manual_mask_choice", "")).strip()
        if choice not in ALLOWED_CHOICES:
            raise ValueError(f"invalid or missing manual mask choice for {annotation_id}")
        choice_counts[choice] = choice_counts.get(choice, 0) + 1
        notes = str(raw_row.get("manual_notes", "")).strip()
        warning = _boundary_warning(notes)
        decision: Dict[str, Any] = {
            "review_index": review_index,
            "annotation_id": annotation_id,
            "x": float(source["x"]),
            "y": float(source["y"]),
            "failure_class": str(source["failure_class"]),
            "manual_mask_choice": choice,
            "accepted": choice in ACCEPTED_CHOICES,
            "selected_variant": ACCEPTED_CHOICES.get(choice, ""),
            "manual_notes": notes,
            "boundary_warning": warning,
            "source_mask_path": "",
            "source_mask_sha256": "",
            "reviewed_mask_path": "",
            "reviewed_mask_sha256": "",
        }
        if choice in ACCEPTED_CHOICES:
            variant = ACCEPTED_CHOICES[choice]
            source_column = f"manual_{variant}_mask_path"
            source_mask_path = Path(str(source[source_column])).resolve()
            if not source_mask_path.is_file():
                raise FileNotFoundError(f"selected provisional mask is missing: {source_mask_path}")
            if not source_mask_path.is_relative_to(analysis_root):
                raise ValueError(
                    f"selected provisional mask is outside the analysis directory: {source_mask_path}"
                )
            provisional = load_candidate_mask(source_mask_path)
            if provisional.image_shape_yx != sample.image_shape_yx:
                raise ValueError(f"selected mask image shape mismatch: {source_mask_path}")
            if str(provisional.metadata.get("annotation_id", "")) != annotation_id:
                raise ValueError(f"selected mask annotation mismatch: {source_mask_path}")
            candidate_id = f"manual_seed_{review_index:03d}"
            selections.append(
                {
                    "review_index": review_index,
                    "candidate_id": candidate_id,
                    "annotation_id": annotation_id,
                    "variant": variant,
                    "choice": choice,
                    "notes": notes,
                    "boundary_warning": warning,
                    "source": source,
                    "source_mask_path": source_mask_path,
                    "source_mask_sha256": _file_sha256(source_mask_path),
                    "mask": _tight_mask(provisional),
                }
            )
            decision["source_mask_path"] = str(source_mask_path)
            decision["source_mask_sha256"] = selections[-1]["source_mask_sha256"]
        decisions.append(decision)

    if not selections:
        raise ValueError("manual-seed review accepted no masks to finalize")

    production_masks = []
    for record in sample.candidates.to_dict("records"):
        candidate_id = str(record["detector_component_id"])
        persisted = load_candidate_mask(_mask_path(sample.sample_dir, record))
        if persisted.image_shape_yx != sample.image_shape_yx:
            raise ValueError(f"production mask image shape mismatch: {candidate_id}")
        production_masks.append((candidate_id, persisted))
    audit = _overlap_audit(
        [(str(item["candidate_id"]), item["mask"]) for item in selections],
        production_masks,
    )
    blocking = audit[
        audit["smaller_mask_overlap_fraction"] >= max_smaller_overlap_fraction
    ]
    if not blocking.empty:
        pairs = ", ".join(
            f"{row.left_id}/{row.right_id}={row.smaller_mask_overlap_fraction:.3f}"
            for row in blocking.itertuples()
        )
        raise ValueError(f"reviewed masks have blocking overlap: {pairs}")

    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manual_seed_finalize_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    masks_dir = destination / "reviewed_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    expected_mask_names = {f"{item['candidate_id']}.npz" for item in selections}
    for existing in masks_dir.glob("*.npz"):
        if existing.name not in expected_mask_names:
            existing.unlink()

    candidate_rows = []
    decision_by_id = {str(row["annotation_id"]): row for row in decisions}
    for item in selections:
        candidate_id = str(item["candidate_id"])
        reviewed_path = masks_dir / f"{candidate_id}.npz"
        source = item["source"]
        variant = str(item["variant"])
        metrics = dict(item["mask"].metadata.get("metrics", {}))
        metadata = {
            **dict(item["mask"].metadata),
            "schema_version": 1,
            "sample_id": sample.sample_id,
            "candidate_id": candidate_id,
            "profile_name": MANUAL_SEED_PROFILE_NAME,
            "base_profile_name": sample.profile_name,
            "base_profile_fingerprint": sample.profile_fingerprint,
            "implementation_version": sample.implementation_version,
            "reviewed_manual_seed": True,
            "provisional_only": False,
            "source_annotation_id": item["annotation_id"],
            "manual_mask_choice": item["choice"],
            "manual_notes": item["notes"],
            "boundary_warning": item["boundary_warning"],
            "review_json_sha256": review_sha256,
            "review_exported_at": payload.get("exported_at"),
            "analysis_sha256": analysis_sha256,
            "source_mask_path": str(item["source_mask_path"]),
            "source_mask_sha256": item["source_mask_sha256"],
        }
        _write_reviewed_mask(reviewed_path, mask=item["mask"], metadata=metadata)
        reviewed_sha256 = _file_sha256(reviewed_path)
        decision = decision_by_id[str(item["annotation_id"])]
        decision["reviewed_mask_path"] = str(reviewed_path)
        decision["reviewed_mask_sha256"] = reviewed_sha256
        ys, xs = np.nonzero(item["mask"].mask)
        centroid_x = item["mask"].bbox.x0 + float(xs.mean())
        centroid_y = item["mask"].bbox.y0 + float(ys.mean())
        percentile = source[f"manual_{variant}_percentile"]
        candidate_rows.append(
            {
                "detector_component_id": candidate_id,
                "source_annotation_id": item["annotation_id"],
                "display_id": f"#R{int(item['review_index']):03d}",
                "html_id": f"manual-seed-{int(item['review_index']):03d}",
                "accepted": True,
                "accepted_strict": False,
                "accepted_rescue": False,
                "detector_score": 1.0,
                "acceptance_mode": f"manual_seed_reviewed_{variant}",
                "detection_pass": MANUAL_SEED_PROFILE_NAME,
                "segmentation_pass": f"manual_{variant}",
                "center_x": int(round(centroid_x)),
                "center_y": int(round(centroid_y)),
                "component_centroid_x": centroid_x,
                "component_centroid_y": centroid_y,
                "bbox_x0": item["mask"].bbox.x0,
                "bbox_y0": item["mask"].bbox.y0,
                "bbox_x1": item["mask"].bbox.x1,
                "bbox_y1": item["mask"].bbox.y1,
                "local_area_px": int(item["mask"].mask.sum()),
                "local_equivalent_diameter_um": metrics.get("equivalent_diameter_um"),
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
                "score_background_percentile": percentile,
                "failure_class": source["failure_class"],
                "manual_review_index": int(item["review_index"]),
                "manual_mask_choice": item["choice"],
                "manual_notes": item["notes"],
                "boundary_warning": bool(item["boundary_warning"]),
                "quality_class": (
                    "reviewed_boundary_warning"
                    if item["boundary_warning"]
                    else "reviewed_manual_seed"
                ),
                "mask_path": str(reviewed_path.relative_to(destination)),
                "mask_source_dir": str(destination),
                "source_provisional_mask_path": str(item["source_mask_path"]),
                "source_provisional_mask_sha256": item["source_mask_sha256"],
                "reviewed_mask_sha256": reviewed_sha256,
                "review_json_sha256": review_sha256,
                "duplicate_suppressed": False,
            }
        )

    decisions_path = destination / "manual_seed_review_decisions.csv"
    candidates_path = destination / "manual_seed_accepted_candidates.csv"
    overlap_audit_path = destination / "mask_overlap_audit.csv"
    _atomic_write_csv(pd.DataFrame(decisions), decisions_path)
    candidates = pd.DataFrame(candidate_rows)
    _atomic_write_csv(candidates, candidates_path)
    _atomic_write_csv(audit, overlap_audit_path)

    delta_labels = export_whole_slide_labels(
        candidates,
        sample_dir=destination,
        image_shape_yx=sample.image_shape_yx,
        image_path=destination / "oocyte_labels_manual_seed_delta_v1.ome.tiff",
        mapping_path=destination / "oocyte_labels_manual_seed_delta_v1_mapping.csv",
        tile_shape_yx=tile_shape_yx,
    )

    combined_labels = None
    combined_candidates_path = None
    if write_combined_labels:
        production = sample.candidates.copy()
        production["mask_path"] = [
            str(_mask_path(sample.sample_dir, record).resolve())
            for record in sample.candidates.to_dict("records")
        ]
        production["mask_source_dir"] = ""
        combined = pd.concat([production, candidates], ignore_index=True, sort=False)
        combined_candidates_path = (
            destination / "oocyte_candidates_rescue_v1_plus_manual_seed_v1.csv"
        )
        _atomic_write_csv(combined, combined_candidates_path)
        combined_labels = export_whole_slide_labels(
            combined,
            sample_dir=destination,
            image_shape_yx=sample.image_shape_yx,
            image_path=(
                destination
                / "oocyte_labels_rescue_v1_plus_manual_seed_v1.ome.tiff"
            ),
            mapping_path=(
                destination
                / "oocyte_labels_rescue_v1_plus_manual_seed_v1_mapping.csv"
            ),
            tile_shape_yx=tile_shape_yx,
        )

    artifact_paths = [
        decisions_path,
        candidates_path,
        overlap_audit_path,
        delta_labels.image_path,
        delta_labels.mapping_path,
        *(Path(row["reviewed_mask_path"]) for row in decisions if row["reviewed_mask_path"]),
    ]
    if combined_labels is not None and combined_candidates_path is not None:
        artifact_paths.extend(
            [
                combined_candidates_path,
                combined_labels.image_path,
                combined_labels.mapping_path,
            ]
        )
    manifest = {
        "schema_version": 1,
        "delivery_name": "reviewed_manual_seed_delta_v1",
        "sample": sample.review_identity,
        "review_identity": dict(identity),
        "review_json": str(review_path),
        "review_json_sha256": review_sha256,
        "review_exported_at": payload.get("exported_at"),
        "analysis_table": str(analysis_path),
        "analysis_sha256": analysis_sha256,
        "choice_counts": choice_counts,
        "reviewed_row_count": len(review_rows),
        "accepted_manual_mask_count": len(candidates),
        "boundary_warning_count": int(candidates["boundary_warning"].sum()),
        "production_candidate_count": int(sample.candidates["accepted"].astype(bool).sum()),
        "combined_label_count": (
            None if combined_labels is None else combined_labels.label_count
        ),
        "manual_overlap_audit_row_count": len(audit),
        "max_smaller_overlap_fraction": (
            0.0 if audit.empty else float(audit["smaller_mask_overlap_fraction"].max())
        ),
        "label_export": {
            "delta_label_count": delta_labels.label_count,
            "delta_assigned_pixel_count": delta_labels.assigned_pixel_count,
            "delta_overlap_pixel_count": delta_labels.overlap_pixel_count,
            "combined_assigned_pixel_count": (
                None if combined_labels is None else combined_labels.assigned_pixel_count
            ),
            "combined_overlap_pixel_count": (
                None if combined_labels is None else combined_labels.overlap_pixel_count
            ),
        },
        "production_outputs_modified": False,
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
    return ManualSeedFinalizeResult(
        out_dir=destination,
        decisions_path=decisions_path,
        candidates_path=candidates_path,
        overlap_audit_path=overlap_audit_path,
        delta_labels=delta_labels,
        combined_labels=combined_labels,
        combined_candidates_path=combined_candidates_path,
        manifest_path=manifest_path,
        accepted_count=len(candidates),
        boundary_warning_count=int(candidates["boundary_warning"].sum()),
    )


__all__ = [
    "MANUAL_SEED_PROFILE_NAME",
    "ManualSeedFinalizeResult",
    "finalize_manual_seed_review",
]
