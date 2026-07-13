"""Finalize a reviewed shape-recovery delta into a new manual-mask delivery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .export import export_whole_slide_labels
from .io import load_candidate_mask
from .manual_seed_finalize import (
    ManualSeedFinalizeResult,
    _atomic_write_csv,
    _boundary_warning,
    _overlap_audit,
    _tight_mask,
    _write_reviewed_mask,
)
from .recall_review import (
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _mask_path,
    _read_json,
)


MANUAL_SEED_V2_PROFILE_NAME = "manual_seed_review_v2"
SHAPE_REVIEW_CHOICES = {
    "keep_v4",
    "accept_shape_recovery",
    "exclude",
    "unsure",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _clean_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _validate_shape_identity(sample_identity: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("shape-recovery review schema_version must be 1")
    if payload.get("review_type") != "manual_seed_shape_recovery_review":
        raise ValueError("review_type must be 'manual_seed_shape_recovery_review'")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("shape-recovery review is missing its identity object")
    for field, expected in sample_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"shape-recovery review identity mismatch for {field}")
    for field in (
        "analysis_sha256",
        "manual_review_json_sha256",
        "shape_candidate_table_sha256",
    ):
        if not identity.get(field):
            raise ValueError(f"shape-recovery review identity is missing {field}")


def _verify_base_delivery(base_dir: Path, expected_review_sha256: str) -> Dict[str, Any]:
    manifest_path = base_dir / "manual_seed_finalize_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("delivery_name") != "reviewed_manual_seed_delta_v1":
        raise ValueError("base finalize directory is not a reviewed manual-seed v1 delivery")
    if manifest.get("review_json_sha256") != expected_review_sha256:
        raise ValueError("base v1 review SHA-256 does not match shape-review identity")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("base v1 manifest is missing artifacts")
    for relative_path, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid base artifact record: {relative_path}")
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"base v1 artifact is missing: {path}")
        if not path.resolve().is_relative_to(base_dir.resolve()):
            raise ValueError(f"base v1 artifact is outside its delivery: {path}")
        if _file_sha256(path) != str(record.get("sha256", "")):
            raise ValueError(f"base v1 artifact SHA-256 mismatch: {relative_path}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"base v1 artifact size mismatch: {relative_path}")
    return manifest


def finalize_shape_recovery_review(
    sample_dir: Path,
    shape_review_json: Path,
    base_finalize_dir: Path,
    out_dir: Path,
    *,
    write_combined_labels: bool = True,
    tile_shape_yx: Tuple[int, int] = (512, 512),
    max_smaller_overlap_fraction: float = 0.25,
) -> ManualSeedFinalizeResult:
    """Apply shape-delta decisions to v1 and write a separate v2 delivery."""

    sample = _load_sample(sample_dir)
    review_path = Path(shape_review_json).resolve()
    payload = _read_json(review_path)
    _validate_shape_identity(sample.review_identity, payload)
    identity = payload["identity"]
    candidate_table_path = Path(str(identity.get("shape_candidate_table", ""))).resolve()
    if not candidate_table_path.is_file():
        raise FileNotFoundError(
            f"shape-recovery candidate table does not exist: {candidate_table_path}"
        )
    candidate_table_sha256 = _file_sha256(candidate_table_path)
    if candidate_table_sha256 != str(identity["shape_candidate_table_sha256"]):
        raise ValueError("shape-recovery candidate table SHA-256 mismatch")
    shape_root = candidate_table_path.parent.resolve()
    shape_candidates = pd.read_csv(candidate_table_path)
    if shape_candidates["annotation_id"].duplicated().any():
        raise ValueError("shape-recovery candidate table has duplicate annotations")
    shape_by_id = shape_candidates.set_index("annotation_id", drop=False)

    review_rows = payload.get("rows")
    if not isinstance(review_rows, list):
        raise ValueError("shape-recovery review rows must be a list")
    review_ids = [str(row.get("annotation_id", "")) for row in review_rows]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("shape-recovery review has duplicate annotations")
    if set(review_ids) != set(str(value) for value in shape_candidates["annotation_id"]):
        raise ValueError("shape-recovery review rows do not match its candidate table")
    shape_decisions: Dict[int, Dict[str, Any]] = {}
    choice_counts: Dict[str, int] = {}
    for row_number, raw_row in enumerate(review_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"shape-recovery review row {row_number} must be an object")
        annotation_id = str(raw_row["annotation_id"])
        candidate = shape_by_id.loc[annotation_id]
        review_index = int(candidate["review_index"])
        if int(raw_row["review_index"]) != review_index:
            raise ValueError(f"shape-recovery review index changed for {annotation_id}")
        if not (
            np.isclose(float(raw_row["x"]), float(candidate["x"]), atol=0.01)
            and np.isclose(float(raw_row["y"]), float(candidate["y"]), atol=0.01)
        ):
            raise ValueError(f"shape-recovery coordinates changed for {annotation_id}")
        choice = str(raw_row.get("shape_review_choice", "")).strip()
        if choice not in SHAPE_REVIEW_CHOICES:
            raise ValueError(f"invalid or missing shape-review choice for {annotation_id}")
        if choice == "unsure":
            raise ValueError(f"shape-recovery review remains unsure for {annotation_id}")
        choice_counts[choice] = choice_counts.get(choice, 0) + 1
        shape_decisions[review_index] = {
            "annotation_id": annotation_id,
            "choice": choice,
            "notes": str(raw_row.get("shape_review_notes", "")).strip(),
            "candidate": candidate,
        }

    base_dir = Path(base_finalize_dir).resolve()
    base_manifest = _verify_base_delivery(
        base_dir,
        str(identity["manual_review_json_sha256"]),
    )
    if base_manifest.get("sample") != sample.review_identity:
        raise ValueError("base v1 sample identity does not match shape review")
    base_decisions = pd.read_csv(base_dir / "manual_seed_review_decisions.csv")
    base_candidates = pd.read_csv(base_dir / "manual_seed_accepted_candidates.csv")
    if base_decisions["review_index"].duplicated().any():
        raise ValueError("base v1 decisions contain duplicate review indices")
    base_decision_by_index = base_decisions.set_index("review_index", drop=False)
    base_candidate_by_index = base_candidates.set_index("manual_review_index", drop=False)
    analysis_path = Path(str(identity.get("analysis_table", ""))).resolve()
    if not analysis_path.is_file() or _file_sha256(analysis_path) != str(
        identity["analysis_sha256"]
    ):
        raise ValueError("shape review analysis table is missing or stale")
    analysis = pd.read_csv(analysis_path).set_index("annotation_id", drop=False)

    selections = []
    final_decisions = []
    replacement_count = 0
    addition_count = 0
    exclusion_count = 0
    for review_index, base in base_decision_by_index.sort_index().iterrows():
        review_index = int(review_index)
        annotation_id = str(base["annotation_id"])
        base_accepted = _as_bool(base["accepted"])
        shape = shape_decisions.get(review_index)
        final_accepted = base_accepted
        final_source = "v1_reviewed_mask" if base_accepted else "excluded_v1"
        source_path = (
            None
            if not base_accepted
            else Path(str(base["reviewed_mask_path"])).resolve()
        )
        final_notes = _clean_text(base.get("manual_notes", ""))
        boundary_warning = _as_bool(base.get("boundary_warning", False))
        shape_choice = ""
        shape_notes = ""
        if shape is not None:
            shape_choice = str(shape["choice"])
            shape_notes = str(shape["notes"])
            if shape_choice == "accept_shape_recovery":
                source_path = Path(
                    str(shape["candidate"]["shape_recovery_mask_path"])
                ).resolve()
                if not source_path.is_relative_to(shape_root):
                    raise ValueError(
                        f"shape-recovery mask is outside its review directory: {source_path}"
                    )
                final_accepted = True
                final_source = "shape_recovery"
                final_notes = shape_notes
                boundary_warning = _boundary_warning(shape_notes)
                if base_accepted:
                    replacement_count += 1
                else:
                    addition_count += 1
            elif shape_choice == "exclude":
                source_path = None
                final_accepted = False
                final_source = "excluded_shape_review"
                final_notes = shape_notes
                boundary_warning = False
                if base_accepted:
                    exclusion_count += 1
            elif shape_choice == "keep_v4":
                final_source = "v1_reviewed_mask" if base_accepted else "excluded_v1"
        decision = {
            "review_index": review_index,
            "annotation_id": annotation_id,
            "x": float(base["x"]),
            "y": float(base["y"]),
            "failure_class": str(base["failure_class"]),
            "base_manual_mask_choice": str(base["manual_mask_choice"]),
            "shape_review_choice": shape_choice,
            "base_manual_notes": _clean_text(base.get("manual_notes", "")),
            "shape_review_notes": shape_notes,
            "final_accepted": final_accepted,
            "final_source": final_source,
            "final_notes": final_notes,
            "boundary_warning": boundary_warning,
            "source_mask_path": "" if source_path is None else str(source_path),
            "source_mask_sha256": "",
            "reviewed_mask_path": "",
            "reviewed_mask_sha256": "",
        }
        if final_accepted:
            if source_path is None or not source_path.is_file():
                raise FileNotFoundError(f"final reviewed mask is missing: {source_path}")
            if final_source == "v1_reviewed_mask" and not source_path.is_relative_to(
                base_dir
            ):
                raise ValueError(f"base reviewed mask is outside v1 delivery: {source_path}")
            persisted = load_candidate_mask(source_path)
            if persisted.image_shape_yx != sample.image_shape_yx:
                raise ValueError(f"final mask image shape mismatch: {source_path}")
            source_sha256 = _file_sha256(source_path)
            decision["source_mask_sha256"] = source_sha256
            selections.append(
                {
                    "review_index": review_index,
                    "annotation_id": annotation_id,
                    "source": final_source,
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                    "mask": _tight_mask(persisted),
                    "notes": final_notes,
                    "boundary_warning": boundary_warning,
                    "base_candidate": (
                        None
                        if review_index not in base_candidate_by_index.index
                        else base_candidate_by_index.loc[review_index]
                    ),
                    "analysis": analysis.loc[annotation_id],
                    "shape_choice": shape_choice,
                }
            )
        final_decisions.append(decision)

    production_masks = []
    for record in sample.candidates.to_dict("records"):
        production_masks.append(
            (
                str(record["detector_component_id"]),
                load_candidate_mask(_mask_path(sample.sample_dir, record)),
            )
        )
    audit = _overlap_audit(
        [
            (f"manual_seed_{int(item['review_index']):03d}", item["mask"])
            for item in selections
        ],
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
        raise ValueError(f"shape-finalized masks have blocking overlap: {pairs}")

    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manual_seed_finalize_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    masks_dir = destination / "reviewed_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"manual_seed_{int(item['review_index']):03d}.npz" for item in selections
    }
    for existing in masks_dir.glob("*.npz"):
        if existing.name not in expected_names:
            existing.unlink()

    review_sha256 = _file_sha256(review_path)
    base_manifest_sha256 = _file_sha256(
        base_dir / "manual_seed_finalize_manifest.json"
    )
    decision_by_index = {
        int(row["review_index"]): row for row in final_decisions
    }
    candidate_rows = []
    for item in selections:
        review_index = int(item["review_index"])
        candidate_id = f"manual_seed_{review_index:03d}"
        reviewed_path = masks_dir / f"{candidate_id}.npz"
        source_metadata = dict(item["mask"].metadata)
        metrics = dict(source_metadata.get("metrics", {}))
        metadata = {
            **source_metadata,
            "schema_version": 1,
            "sample_id": sample.sample_id,
            "candidate_id": candidate_id,
            "profile_name": MANUAL_SEED_V2_PROFILE_NAME,
            "base_profile_name": sample.profile_name,
            "base_profile_fingerprint": sample.profile_fingerprint,
            "implementation_version": sample.implementation_version,
            "reviewed_manual_seed": True,
            "provisional_only": False,
            "review_generation": 2,
            "source_annotation_id": item["annotation_id"],
            "final_source": item["source"],
            "shape_review_choice": item["shape_choice"],
            "final_notes": item["notes"],
            "boundary_warning": item["boundary_warning"],
            "shape_review_json": str(review_path),
            "shape_review_json_sha256": review_sha256,
            "shape_review_exported_at": payload.get("exported_at"),
            "base_delivery_manifest": str(
                base_dir / "manual_seed_finalize_manifest.json"
            ),
            "base_delivery_manifest_sha256": base_manifest_sha256,
            "source_mask_path": str(item["source_path"]),
            "source_mask_sha256": item["source_sha256"],
        }
        _write_reviewed_mask(reviewed_path, mask=item["mask"], metadata=metadata)
        reviewed_sha256 = _file_sha256(reviewed_path)
        decision = decision_by_index[review_index]
        decision["reviewed_mask_path"] = str(reviewed_path)
        decision["reviewed_mask_sha256"] = reviewed_sha256
        ys, xs = np.nonzero(item["mask"].mask)
        centroid_x = item["mask"].bbox.x0 + float(xs.mean())
        centroid_y = item["mask"].bbox.y0 + float(ys.mean())
        row = (
            {}
            if item["base_candidate"] is None
            else item["base_candidate"].to_dict()
        )
        row.update(
            {
                "detector_component_id": candidate_id,
                "source_annotation_id": item["annotation_id"],
                "display_id": f"#R{review_index:03d}",
                "html_id": f"manual-seed-{review_index:03d}",
                "accepted": True,
                "accepted_strict": False,
                "accepted_rescue": False,
                "detector_score": 1.0,
                "acceptance_mode": (
                    "manual_seed_shape_recovery_reviewed"
                    if item["source"] == "shape_recovery"
                    else "manual_seed_reviewed_v2_carry_forward"
                ),
                "detection_pass": MANUAL_SEED_V2_PROFILE_NAME,
                "segmentation_pass": (
                    "manual_shape_recovery"
                    if item["source"] == "shape_recovery"
                    else "manual_v1_carry_forward"
                ),
                "center_x": int(round(centroid_x)),
                "center_y": int(round(centroid_y)),
                "component_centroid_x": centroid_x,
                "component_centroid_y": centroid_y,
                "bbox_x0": item["mask"].bbox.x0,
                "bbox_y0": item["mask"].bbox.y0,
                "bbox_x1": item["mask"].bbox.x1,
                "bbox_y1": item["mask"].bbox.y1,
                "local_area_px": int(item["mask"].mask.sum()),
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
                "score_background_percentile": source_metadata.get(
                    "annulus_floor_percentile"
                ),
                "failure_class": item["analysis"]["failure_class"],
                "manual_review_index": review_index,
                "base_manual_mask_choice": decision["base_manual_mask_choice"],
                "shape_review_choice": decision["shape_review_choice"],
                "manual_notes": item["notes"],
                "boundary_warning": bool(item["boundary_warning"]),
                "quality_class": (
                    "reviewed_boundary_warning"
                    if item["boundary_warning"]
                    else "reviewed_manual_seed_v2"
                ),
                "mask_path": str(reviewed_path.relative_to(destination)),
                "mask_source_dir": str(destination),
                "source_reviewed_mask_path": str(item["source_path"]),
                "source_reviewed_mask_sha256": item["source_sha256"],
                "reviewed_mask_sha256": reviewed_sha256,
                "shape_review_json_sha256": review_sha256,
                "duplicate_suppressed": False,
            }
        )
        candidate_rows.append(row)

    decisions_path = destination / "manual_seed_review_decisions_v2.csv"
    candidates_path = destination / "manual_seed_accepted_candidates_v2.csv"
    overlap_audit_path = destination / "mask_overlap_audit.csv"
    _atomic_write_csv(pd.DataFrame(final_decisions), decisions_path)
    candidates = pd.DataFrame(candidate_rows)
    _atomic_write_csv(candidates, candidates_path)
    _atomic_write_csv(audit, overlap_audit_path)
    delta_labels = export_whole_slide_labels(
        candidates,
        sample_dir=destination,
        image_shape_yx=sample.image_shape_yx,
        image_path=destination / "oocyte_labels_manual_seed_delta_v2.ome.tiff",
        mapping_path=destination / "oocyte_labels_manual_seed_delta_v2_mapping.csv",
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
            destination / "oocyte_candidates_rescue_v1_plus_manual_seed_v2.csv"
        )
        _atomic_write_csv(combined, combined_candidates_path)
        combined_labels = export_whole_slide_labels(
            combined,
            sample_dir=destination,
            image_shape_yx=sample.image_shape_yx,
            image_path=(
                destination
                / "oocyte_labels_rescue_v1_plus_manual_seed_v2.ome.tiff"
            ),
            mapping_path=(
                destination
                / "oocyte_labels_rescue_v1_plus_manual_seed_v2_mapping.csv"
            ),
            tile_shape_yx=tile_shape_yx,
        )

    artifact_paths = [
        decisions_path,
        candidates_path,
        overlap_audit_path,
        delta_labels.image_path,
        delta_labels.mapping_path,
        *(Path(row["reviewed_mask_path"]) for row in final_decisions if row["reviewed_mask_path"]),
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
        "delivery_name": "reviewed_manual_seed_delta_v2",
        "sample": sample.review_identity,
        "shape_review_json": str(review_path),
        "shape_review_json_sha256": review_sha256,
        "shape_review_exported_at": payload.get("exported_at"),
        "shape_choice_counts": choice_counts,
        "shape_candidate_table": str(candidate_table_path),
        "shape_candidate_table_sha256": candidate_table_sha256,
        "base_delivery": str(base_dir),
        "base_delivery_manifest_sha256": base_manifest_sha256,
        "reviewed_row_count": len(final_decisions),
        "accepted_manual_mask_count": len(candidates),
        "shape_replacement_count": replacement_count,
        "shape_addition_count": addition_count,
        "shape_exclusion_count": exclusion_count,
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
        "base_v1_outputs_modified": False,
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
    "MANUAL_SEED_V2_PROFILE_NAME",
    "SHAPE_REVIEW_CHOICES",
    "finalize_shape_recovery_review",
]
