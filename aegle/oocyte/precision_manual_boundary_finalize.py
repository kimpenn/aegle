"""Finalize reviewed manual polygons into a Precision-resolved v2 delivery."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from skimage import draw, measure

from .export import LabelExportResult, export_whole_slide_labels
from .io import load_candidate_mask
from .manual_seed_finalize import (
    _atomic_write_csv,
    _tight_mask,
    _write_reviewed_mask,
)
from .models import BoundingBox, PersistedMask
from .precision_boundary_finalize import (
    _atomic_copy,
    _resolved_overlap_audit,
)
from .precision_boundary_review import _review_key
from .precision_manual_boundary_review import (
    _verify_precision_resolved_delivery,
)
from .recall_review import (
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _mask_path,
    _read_json,
)


PRECISION_RESOLVED_V2_PROFILE_NAME = "precision_resolved_v2"
PRECISION_MANUAL_BOUNDARY_CHOICES = {
    "accept_manual_contour",
    "exclude",
    "unsure",
}
MANUAL_CONTOUR_MIN_DIAMETER_UM = 10.0
MANUAL_CONTOUR_MAX_DIAMETER_UM = 100.0


@dataclass(frozen=True)
class PrecisionManualBoundaryFinalizeResult:
    out_dir: Path
    decisions_path: Path
    candidates_path: Path
    manual_decisions_path: Path
    remaining_queue_path: Path
    overlap_audit_path: Path
    labels: LabelExportResult
    manifest_path: Path
    resolved_count: int
    manual_added_count: int
    manual_excluded_count: int


def _clean_text(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _orientation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
        c[0] - a[0]
    )


def _on_segment(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    return bool(
        min(a[0], b[0]) - tolerance <= p[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance
        <= p[1]
        <= max(a[1], b[1]) + tolerance
        and abs(_orientation(a, b, p)) <= tolerance
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return bool(
        (abs(values[0]) <= 1e-9 and _on_segment(a, b, c))
        or (abs(values[1]) <= 1e-9 and _on_segment(a, b, d))
        or (abs(values[2]) <= 1e-9 and _on_segment(c, d, a))
        or (abs(values[3]) <= 1e-9 and _on_segment(c, d, b))
    )


def _validate_simple_polygon(
    vertices: Sequence[tuple[float, float]],
    *,
    key: str,
) -> None:
    edge_count = len(vertices)
    for left_index in range(edge_count):
        a = vertices[left_index]
        b = vertices[(left_index + 1) % edge_count]
        for right_index in range(left_index + 1, edge_count):
            if right_index in {
                left_index,
                (left_index + 1) % edge_count,
                (left_index - 1) % edge_count,
            }:
                continue
            if left_index == 0 and right_index == edge_count - 1:
                continue
            c = vertices[right_index]
            d = vertices[(right_index + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"manual contour self-intersects for {key}")


def _normalize_vertices(
    value: Any,
    *,
    candidate: Mapping[str, Any],
    image_shape_yx: Tuple[int, int],
    key: str,
) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise ValueError(f"manual contour vertices must be a list for {key}")
    vertices = []
    for index, raw_vertex in enumerate(value, start=1):
        if not isinstance(raw_vertex, (list, tuple)) or len(raw_vertex) != 2:
            raise ValueError(f"manual contour vertex {index} is invalid for {key}")
        x, y = float(raw_vertex[0]), float(raw_vertex[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"manual contour vertex {index} is non-finite for {key}")
        point = (x, y)
        if not vertices or not np.allclose(point, vertices[-1], atol=1e-6):
            vertices.append(point)
    if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1], atol=1e-6):
        vertices.pop()
    if len(vertices) < 3 or len({(round(x, 6), round(y, 6)) for x, y in vertices}) < 3:
        raise ValueError(f"manual contour requires three unique vertices for {key}")
    origin_x = float(candidate["patch_origin_x"])
    origin_y = float(candidate["patch_origin_y"])
    width = int(candidate["asset_width_px"])
    height = int(candidate["asset_height_px"])
    image_h, image_w = image_shape_yx
    for x, y in vertices:
        if not (
            origin_x <= x <= origin_x + width - 1
            and origin_y <= y <= origin_y + height - 1
        ):
            raise ValueError(f"manual contour leaves its reviewed patch for {key}")
        if not (0 <= x < image_w and 0 <= y < image_h):
            raise ValueError(f"manual contour leaves the source image for {key}")
    _validate_simple_polygon(vertices, key=key)
    signed_area = 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(vertices, vertices[1:] + vertices[:1])
    )
    if abs(signed_area) < 1.0:
        raise ValueError(f"manual contour has negligible area for {key}")
    return vertices


def _rasterize_polygon(
    vertices: Sequence[tuple[float, float]],
    *,
    center_xy: tuple[float, float],
    image_shape_yx: Tuple[int, int],
    pixel_size_um: float,
    key: str,
) -> tuple[PersistedMask, Dict[str, float | int | str | None]]:
    image_h, image_w = image_shape_yx
    xs = np.asarray([point[0] for point in vertices], dtype=np.float64)
    ys = np.asarray([point[1] for point in vertices], dtype=np.float64)
    x0 = max(0, int(math.floor(float(xs.min()))))
    y0 = max(0, int(math.floor(float(ys.min()))))
    x1 = min(image_w, int(math.ceil(float(xs.max()))) + 1)
    y1 = min(image_h, int(math.ceil(float(ys.max()))) + 1)
    bbox = BoundingBox(x0, y0, x1, y1)
    rows, columns = draw.polygon(ys - y0, xs - x0, shape=bbox.shape_yx)
    mask = np.zeros(bbox.shape_yx, dtype=np.bool_)
    mask[rows, columns] = True
    if not mask.any():
        raise ValueError(f"manual contour rasterized to an empty mask for {key}")
    center_x, center_y = center_xy
    local_center_x = int(round(center_x)) - bbox.x0
    local_center_y = int(round(center_y)) - bbox.y0
    if not (
        0 <= local_center_x < bbox.width
        and 0 <= local_center_y < bbox.height
        and mask[local_center_y, local_center_x]
    ):
        raise ValueError(f"manual contour does not contain its reviewed center for {key}")
    prop = measure.regionprops(mask.astype(np.uint8))[0]
    equivalent_diameter_um = float(prop.equivalent_diameter_area * pixel_size_um)
    if not (
        MANUAL_CONTOUR_MIN_DIAMETER_UM
        <= equivalent_diameter_um
        <= MANUAL_CONTOUR_MAX_DIAMETER_UM
    ):
        raise ValueError(
            f"manual contour diameter {equivalent_diameter_um:.1f} um is outside "
            f"[{MANUAL_CONTOUR_MIN_DIAMETER_UM:.1f}, "
            f"{MANUAL_CONTOUR_MAX_DIAMETER_UM:.1f}] for {key}"
        )
    centroid_x = bbox.x0 + float(prop.centroid[1])
    centroid_y = bbox.y0 + float(prop.centroid[0])
    circularity = (
        0.0
        if prop.perimeter <= 0
        else float(4.0 * np.pi * prop.area / (prop.perimeter**2))
    )
    metrics: Dict[str, float | int | str | None] = {
        "threshold_method": "manual_polygon",
        "base_threshold": None,
        "annulus_floor": None,
        "threshold": None,
        "selection_mode": "reviewed_manual_polygon",
        "area_px": int(prop.area),
        "equivalent_diameter_um": equivalent_diameter_um,
        "major_axis_um": float(prop.axis_major_length * pixel_size_um),
        "minor_axis_um": float(prop.axis_minor_length * pixel_size_um),
        "eccentricity": float(prop.eccentricity),
        "solidity": float(prop.solidity),
        "circularity": circularity,
        "centroid_y_px": float(prop.centroid[0]),
        "centroid_x_px": float(prop.centroid[1]),
        "centroid_offset_px": float(
            np.hypot(centroid_x - center_x, centroid_y - center_y)
        ),
        "mean_intensity": None,
        "max_intensity": None,
    }
    return (
        PersistedMask(
            mask=mask,
            bbox=bbox,
            image_shape_yx=image_shape_yx,
            metadata={"schema_version": 1, "metrics": metrics},
        ),
        metrics,
    )


def _validate_review_identity(
    sample_identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("manual boundary review schema_version must be 1")
    if payload.get("review_type") != "oocyte_precision_manual_boundary_review":
        raise ValueError(
            "review_type must be 'oocyte_precision_manual_boundary_review'"
        )
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("manual boundary review identity is missing")
    for field, expected in sample_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"manual boundary identity mismatch for {field}")
    for field in (
        "base_precision_resolved_dir",
        "base_precision_resolved_manifest",
        "base_precision_resolved_manifest_sha256",
        "base_resolved_candidates_sha256",
        "manual_boundary_queue_sha256",
        "manual_boundary_candidate_table",
        "manual_boundary_candidate_table_sha256",
    ):
        if not identity.get(field):
            raise ValueError(f"manual boundary identity is missing {field}")
    return identity


def _validate_review_rows(
    payload: Mapping[str, Any],
    candidate_table: pd.DataFrame,
    *,
    image_shape_yx: Tuple[int, int],
) -> Dict[str, Dict[str, Any]]:
    if candidate_table["review_key"].astype(str).duplicated().any():
        raise ValueError("manual boundary candidate keys are not unique")
    candidates = {
        str(row["review_key"]): row for row in candidate_table.to_dict("records")
    }
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manual boundary review rows must be a list")
    review_by_key: Dict[str, Dict[str, Any]] = {}
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"manual boundary row {row_number} must be an object")
        key = str(raw_row.get("review_key", ""))
        if key in review_by_key:
            raise ValueError(f"manual boundary review contains duplicate key {key}")
        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError(f"manual boundary review contains unknown key {key}")
        for field in (
            "manual_index",
            "review_index",
            "boundary_index",
            "detector_component_id",
            "detection_pass",
        ):
            if str(raw_row.get(field, "")) != str(candidate[field]):
                raise ValueError(f"manual boundary metadata changed for {key}: {field}")
        for field in ("center_x", "center_y"):
            if not np.isclose(
                float(raw_row.get(field, np.nan)),
                float(candidate[field]),
                atol=0.01,
            ):
                raise ValueError(f"manual boundary coordinates changed for {key}: {field}")
        choice = str(raw_row.get("manual_boundary_choice", "")).strip()
        if choice not in PRECISION_MANUAL_BOUNDARY_CHOICES:
            raise ValueError(f"invalid or missing manual boundary choice for {key}")
        if choice == "unsure":
            raise ValueError(f"manual boundary review remains unsure for {key}")
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
        raise ValueError("manual boundary review rows do not match its candidate table")
    return review_by_key


def _manual_candidate_row(
    source: Mapping[str, Any],
    *,
    key: str,
    mask: PersistedMask,
    metrics: Mapping[str, Any],
    reviewed_path: Path,
    reviewed_sha256: str,
    destination: Path,
    notes: str,
    review_sha256: str,
    candidate_table_sha256: str,
    current_mask_path: Path,
    current_mask_sha256: str,
) -> Dict[str, Any]:
    row = dict(source)
    ys, xs = np.nonzero(mask.mask)
    centroid_x = mask.bbox.x0 + float(xs.mean())
    centroid_y = mask.bbox.y0 + float(ys.mean())
    row.update(
        {
            "review_key": key,
            "resolved_oocyte_id": key,
            "source_center_x": source.get("center_x"),
            "source_center_y": source.get("center_y"),
            "source_acceptance_mode": source.get("acceptance_mode"),
            "source_segmentation_pass": source.get("segmentation_pass"),
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
            "local_equivalent_diameter_um": metrics.get("equivalent_diameter_um"),
            "local_major_axis_um": metrics.get("major_axis_um"),
            "local_minor_axis_um": metrics.get("minor_axis_um"),
            "local_eccentricity": metrics.get("eccentricity"),
            "local_solidity": metrics.get("solidity"),
            "local_circularity": metrics.get("circularity"),
            "local_centroid_offset_px": metrics.get("centroid_offset_px"),
            "local_mean_intensity": None,
            "local_max_intensity": None,
            "threshold_method": "manual_polygon",
            "threshold": None,
            "selection_mode": "reviewed_manual_polygon",
            "acceptance_mode": "precision_manual_contour",
            "segmentation_pass": "precision_manual_contour",
            "quality_class": "precision_reviewed_manual_boundary",
            "precision_review_status": "reject",
            "precision_boundary_choice": "needs_manual",
            "precision_resolution_source": "precision_manual_contour",
            "manual_boundary_choice": "accept_manual_contour",
            "manual_boundary_notes": notes,
            "mask_path": str(reviewed_path.relative_to(destination)),
            "mask_source_dir": str(destination),
            "source_mask_path": str(current_mask_path),
            "source_mask_sha256": current_mask_sha256,
            "reviewed_mask_sha256": reviewed_sha256,
            "precision_manual_boundary_review_json_sha256": review_sha256,
            "precision_manual_boundary_candidate_table_sha256": (
                candidate_table_sha256
            ),
            "duplicate_suppressed": False,
        }
    )
    return row


def finalize_precision_manual_boundary_review(
    sample_dir: Path,
    base_resolved_dir: Path,
    manual_review_json: Path,
    out_dir: Path,
    *,
    tile_shape_yx: Tuple[int, int] = (512, 512),
) -> PrecisionManualBoundaryFinalizeResult:
    """Rasterize reviewed polygons and write an immutable Precision v2."""

    sample = _load_sample(sample_dir)
    base_dir = Path(base_resolved_dir).resolve()
    base_manifest = _verify_precision_resolved_delivery(
        base_dir,
        sample_identity=sample.review_identity,
    )
    base_manifest_path = base_dir / "precision_resolved_manifest.json"
    base_manifest_sha256 = _file_sha256(base_manifest_path)
    review_path = Path(manual_review_json).resolve()
    payload = _read_json(review_path)
    identity = _validate_review_identity(sample.review_identity, payload)
    if Path(str(identity["base_precision_resolved_dir"])).resolve() != base_dir:
        raise ValueError("manual boundary base directory does not match review identity")
    if Path(str(identity["base_precision_resolved_manifest"])).resolve() != (
        base_manifest_path
    ):
        raise ValueError("manual boundary base manifest path does not match")
    if str(identity["base_precision_resolved_manifest_sha256"]) != (
        base_manifest_sha256
    ):
        raise ValueError("manual boundary base manifest SHA-256 mismatch")

    base_candidates_path = base_dir / "precision_resolved_candidates.csv"
    base_queue_path = base_dir / "manual_boundary_queue.csv"
    if _file_sha256(base_candidates_path) != str(
        identity["base_resolved_candidates_sha256"]
    ):
        raise ValueError("manual boundary base candidate SHA-256 mismatch")
    if _file_sha256(base_queue_path) != str(identity["manual_boundary_queue_sha256"]):
        raise ValueError("manual boundary queue SHA-256 mismatch")
    candidate_table_path = Path(
        str(identity["manual_boundary_candidate_table"])
    ).resolve()
    if not candidate_table_path.is_file():
        raise FileNotFoundError(
            f"manual boundary candidate table is missing: {candidate_table_path}"
        )
    candidate_table_sha256 = _file_sha256(candidate_table_path)
    if candidate_table_sha256 != str(
        identity["manual_boundary_candidate_table_sha256"]
    ):
        raise ValueError("manual boundary candidate table SHA-256 mismatch")
    manual_candidates = pd.read_csv(candidate_table_path)
    review_by_key = _validate_review_rows(
        payload,
        manual_candidates,
        image_shape_yx=sample.image_shape_yx,
    )
    review_sha256 = _file_sha256(review_path)

    sample_records = sample.candidates.to_dict("records")
    sample_by_key = {_review_key(row): row for row in sample_records}
    base_candidates = pd.read_csv(base_candidates_path)
    base_decisions = pd.read_csv(base_dir / "precision_review_decisions.csv")
    if base_candidates["review_key"].astype(str).duplicated().any():
        raise ValueError("base Precision candidates contain duplicate review keys")
    if base_decisions["review_key"].astype(str).duplicated().any():
        raise ValueError("base Precision decisions contain duplicate review keys")

    base_masks = []
    for row in base_candidates.to_dict("records"):
        path = Path(str(row["mask_path"]))
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.is_relative_to(base_dir) or not path.is_file():
            raise ValueError(f"base Precision mask is outside its delivery: {path}")
        mask = load_candidate_mask(path)
        if mask.image_shape_yx != sample.image_shape_yx:
            raise ValueError(f"base Precision mask image shape mismatch: {path}")
        base_masks.append((str(row["review_key"]), mask, path, row))

    manual_selections = []
    manual_decisions = []
    choice_counts: Dict[str, int] = {}
    for key, review in review_by_key.items():
        choice = str(review["choice"])
        choice_counts[choice] = choice_counts.get(choice, 0) + 1
        candidate = review["candidate"]
        source = sample_by_key.get(key)
        if source is None:
            raise ValueError(f"manual boundary source candidate is missing: {key}")
        current_path = _mask_path(sample.sample_dir, source).resolve()
        if current_path != Path(str(candidate["current_mask_path"])).resolve():
            raise ValueError(f"manual boundary current mask changed for {key}")
        if not current_path.is_file():
            raise FileNotFoundError(f"manual boundary current mask is missing: {current_path}")
        current_sha256 = _file_sha256(current_path)
        if current_sha256 != str(candidate["current_mask_sha256"]):
            raise ValueError(f"manual boundary current mask SHA-256 mismatch for {key}")
        accepted = choice == "accept_manual_contour"
        mask = None
        metrics = None
        if accepted:
            mask, metrics = _rasterize_polygon(
                review["vertices"],
                center_xy=(float(candidate["center_x"]), float(candidate["center_y"])),
                image_shape_yx=sample.image_shape_yx,
                pixel_size_um=sample.pixel_size_um,
                key=key,
            )
            manual_selections.append(
                {
                    "key": key,
                    "choice": choice,
                    "notes": str(review["notes"]),
                    "vertices": review["vertices"],
                    "candidate": candidate,
                    "source": source,
                    "current_path": current_path,
                    "current_sha256": current_sha256,
                    "mask": _tight_mask(mask),
                    "metrics": metrics,
                }
            )
        manual_decisions.append(
            {
                "manual_index": int(candidate["manual_index"]),
                "review_index": int(candidate["review_index"]),
                "boundary_index": int(candidate["boundary_index"]),
                "review_key": key,
                "display_id": str(candidate["display_id"]),
                "detector_component_id": str(candidate["detector_component_id"]),
                "detection_pass": str(candidate["detection_pass"]),
                "center_x": float(candidate["center_x"]),
                "center_y": float(candidate["center_y"]),
                "manual_boundary_choice": choice,
                "manual_boundary_notes": str(review["notes"]),
                "vertex_count": len(review["vertices"]),
                "vertices_xy_json": json.dumps(review["vertices"]),
                "final_accepted": accepted,
                "final_source": (
                    "precision_manual_contour"
                    if accepted
                    else "precision_manual_boundary_excluded"
                ),
                "current_mask_path": str(current_path),
                "current_mask_sha256": current_sha256,
                "reviewed_mask_path": "",
                "reviewed_mask_sha256": "",
            }
        )

    all_selected_masks = [
        (key, mask) for key, mask, _, _ in base_masks
    ] + [
        (str(item["key"]), item["mask"]) for item in manual_selections
    ]
    audit = _resolved_overlap_audit(all_selected_masks)
    if not audit.empty:
        pairs = ", ".join(
            f"{row.left_review_key}/{row.right_review_key}="
            f"{int(row.overlap_pixel_count)}px"
            for row in audit.itertuples()
        )
        raise ValueError(f"Precision v2 masks overlap: {pairs}")

    destination = Path(out_dir).resolve()
    manifest_path = destination / "precision_resolved_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable Precision v2 already exists: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=True)
    masks_dir = destination / "reviewed_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {path.name for _, _, path, _ in base_masks} | {
        f"{str(item['key']).replace(':', '__')}.npz" for item in manual_selections
    }
    for existing in masks_dir.glob("*.npz"):
        if existing.name not in expected_names:
            existing.unlink()

    final_candidate_rows = []
    for key, _, source_path, raw_row in base_masks:
        copied_path = masks_dir / source_path.name
        _atomic_copy(source_path, copied_path)
        if _file_sha256(source_path) != _file_sha256(copied_path):
            raise ValueError(f"base mask copy SHA-256 mismatch for {key}")
        row = dict(raw_row)
        row["mask_path"] = str(copied_path.relative_to(destination))
        row["mask_source_dir"] = str(destination)
        row["precision_v2_source"] = "precision_resolved_v1_carry_forward"
        final_candidate_rows.append(row)

    manual_decision_by_key = {
        str(row["review_key"]): row for row in manual_decisions
    }
    for item in manual_selections:
        key = str(item["key"])
        reviewed_path = masks_dir / f"{key.replace(':', '__')}.npz"
        metadata = {
            **dict(item["mask"].metadata),
            "schema_version": 1,
            "sample_id": sample.sample_id,
            "candidate_id": str(item["source"]["detector_component_id"]),
            "review_key": key,
            "resolved_oocyte_id": key,
            "profile_name": PRECISION_RESOLVED_V2_PROFILE_NAME,
            "base_profile_name": sample.profile_name,
            "base_profile_fingerprint": sample.profile_fingerprint,
            "implementation_version": sample.implementation_version,
            "precision_resolved": True,
            "reviewed_manual_polygon": True,
            "provisional_only": False,
            "manual_boundary_choice": item["choice"],
            "manual_boundary_notes": item["notes"],
            "manual_polygon_vertices_xy": item["vertices"],
            "precision_manual_boundary_review_json": str(review_path),
            "precision_manual_boundary_review_json_sha256": review_sha256,
            "precision_manual_boundary_candidate_table_sha256": (
                candidate_table_sha256
            ),
            "base_precision_resolved_manifest_sha256": base_manifest_sha256,
            "source_current_mask_path": str(item["current_path"]),
            "source_current_mask_sha256": item["current_sha256"],
            "metrics": item["metrics"],
        }
        _write_reviewed_mask(reviewed_path, mask=item["mask"], metadata=metadata)
        reviewed_sha256 = _file_sha256(reviewed_path)
        decision = manual_decision_by_key[key]
        decision["reviewed_mask_path"] = str(reviewed_path)
        decision["reviewed_mask_sha256"] = reviewed_sha256
        final_candidate_rows.append(
            _manual_candidate_row(
                item["source"],
                key=key,
                mask=item["mask"],
                metrics=item["metrics"],
                reviewed_path=reviewed_path,
                reviewed_sha256=reviewed_sha256,
                destination=destination,
                notes=str(item["notes"]),
                review_sha256=review_sha256,
                candidate_table_sha256=candidate_table_sha256,
                current_mask_path=item["current_path"],
                current_mask_sha256=str(item["current_sha256"]),
            )
        )

    final_candidates = pd.DataFrame(final_candidate_rows)
    if final_candidates["review_key"].astype(str).duplicated().any():
        raise ValueError("Precision v2 candidates contain duplicate review keys")
    updated_decisions = base_decisions.copy()
    for column in ("selected_variant", "reviewed_mask_path", "reviewed_mask_sha256"):
        updated_decisions[column] = updated_decisions[column].fillna("").astype(str)
    updated_decisions["manual_boundary_choice"] = ""
    updated_decisions["manual_boundary_notes"] = ""
    updated_decisions["manual_boundary_review_json_sha256"] = ""
    for manual in manual_decisions:
        key = str(manual["review_key"])
        matches = updated_decisions["review_key"].astype(str) == key
        if int(matches.sum()) != 1:
            raise ValueError(f"base Precision decision is missing for {key}")
        updated_decisions.loc[matches, "manual_boundary_choice"] = manual[
            "manual_boundary_choice"
        ]
        updated_decisions.loc[matches, "manual_boundary_notes"] = manual[
            "manual_boundary_notes"
        ]
        updated_decisions.loc[
            matches, "manual_boundary_review_json_sha256"
        ] = review_sha256
        updated_decisions.loc[matches, "final_accepted"] = manual["final_accepted"]
        updated_decisions.loc[matches, "resolution_state"] = (
            "resolved" if manual["final_accepted"] else "excluded"
        )
        updated_decisions.loc[matches, "final_source"] = manual["final_source"]
        updated_decisions.loc[matches, "selected_variant"] = (
            "manual_polygon" if manual["final_accepted"] else ""
        )
        updated_decisions.loc[matches, "reviewed_mask_path"] = manual[
            "reviewed_mask_path"
        ]
        updated_decisions.loc[matches, "reviewed_mask_sha256"] = manual[
            "reviewed_mask_sha256"
        ]

    decisions_path = destination / "precision_review_decisions_v2.csv"
    candidates_path = destination / "precision_resolved_candidates_v2.csv"
    manual_decisions_path = destination / "manual_boundary_decisions.csv"
    remaining_queue_path = destination / "manual_boundary_queue.csv"
    overlap_audit_path = destination / "mask_overlap_audit.csv"
    _atomic_write_csv(updated_decisions, decisions_path)
    _atomic_write_csv(final_candidates, candidates_path)
    _atomic_write_csv(pd.DataFrame(manual_decisions), manual_decisions_path)
    queue_columns = list(pd.read_csv(base_queue_path).columns)
    _atomic_write_csv(pd.DataFrame(columns=queue_columns), remaining_queue_path)
    _atomic_write_csv(audit, overlap_audit_path)

    labels = export_whole_slide_labels(
        final_candidates,
        sample_dir=destination,
        image_shape_yx=sample.image_shape_yx,
        image_path=destination / "oocyte_labels_precision_resolved_v2.ome.tiff",
        mapping_path=destination / "oocyte_labels_precision_resolved_v2_mapping.csv",
        tile_shape_yx=tile_shape_yx,
    )
    if labels.overlap_pixel_count:
        raise AssertionError("manual-boundary preflight and label export disagree")
    expected_pixels = sum(int(mask.mask.sum()) for _, mask in all_selected_masks)
    if labels.assigned_pixel_count != expected_pixels:
        raise AssertionError("Precision v2 label pixel count does not match exact masks")

    input_dir = destination / "review_inputs"
    input_copies = {
        "base_precision_resolved_manifest": (
            base_manifest_path,
            input_dir / "base_precision_resolved_manifest.json",
        ),
        "manual_boundary_candidates": (
            candidate_table_path,
            input_dir / "precision_manual_boundary_candidates.csv",
        ),
        "manual_boundary_review": (
            review_path,
            input_dir / "precision_manual_boundary_review.json",
        ),
    }
    for source_path, copy_path in input_copies.values():
        _atomic_copy(source_path, copy_path)
        if _file_sha256(source_path) != _file_sha256(copy_path):
            raise ValueError(f"review-input copy SHA-256 mismatch: {copy_path}")

    artifact_paths = [
        decisions_path,
        candidates_path,
        manual_decisions_path,
        remaining_queue_path,
        overlap_audit_path,
        labels.image_path,
        labels.mapping_path,
        *(copy_path for _, copy_path in input_copies.values()),
        *(masks_dir / name for name in sorted(expected_names)),
    ]
    manual_excluded_count = choice_counts.get("exclude", 0)
    manifest = {
        "schema_version": 1,
        "delivery_name": PRECISION_RESOLVED_V2_PROFILE_NAME,
        "delivery_status": "intermediate_precision_only",
        "release_ready": False,
        "precision_complete": True,
        "manual_boundary_complete": True,
        "recall_complete": False,
        "sample": sample.review_identity,
        "base_precision_resolved_dir": str(base_dir),
        "base_precision_resolved_manifest_sha256": base_manifest_sha256,
        "manual_boundary_review_json": str(review_path),
        "manual_boundary_review_json_sha256": review_sha256,
        "manual_boundary_review_exported_at": payload.get("exported_at"),
        "manual_boundary_candidate_table": str(candidate_table_path),
        "manual_boundary_candidate_table_sha256": candidate_table_sha256,
        "manual_boundary_choice_counts": choice_counts,
        "base_resolved_label_count": int(base_manifest["resolved_label_count"]),
        "manual_added_count": len(manual_selections),
        "manual_excluded_count": manual_excluded_count,
        "resolved_label_count": labels.label_count,
        "unresolved_manual_count": 0,
        "overlap_audit_row_count": len(audit),
        "label_export": {
            "label_count": labels.label_count,
            "assigned_pixel_count": labels.assigned_pixel_count,
            "overlap_pixel_count": labels.overlap_pixel_count,
        },
        "production_outputs_modified": False,
        "base_precision_outputs_modified": False,
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
    return PrecisionManualBoundaryFinalizeResult(
        out_dir=destination,
        decisions_path=decisions_path,
        candidates_path=candidates_path,
        manual_decisions_path=manual_decisions_path,
        remaining_queue_path=remaining_queue_path,
        overlap_audit_path=overlap_audit_path,
        labels=labels,
        manifest_path=manifest_path,
        resolved_count=labels.label_count,
        manual_added_count=len(manual_selections),
        manual_excluded_count=manual_excluded_count,
    )


__all__ = [
    "MANUAL_CONTOUR_MAX_DIAMETER_UM",
    "MANUAL_CONTOUR_MIN_DIAMETER_UM",
    "PRECISION_MANUAL_BOUNDARY_CHOICES",
    "PRECISION_RESOLVED_V2_PROFILE_NAME",
    "PrecisionManualBoundaryFinalizeResult",
    "finalize_precision_manual_boundary_review",
]
