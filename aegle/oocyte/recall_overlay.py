"""Validated exact-mask overlays for coverage-based Recall review."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .io import load_candidate_mask


REVIEWED_OVERLAY_CANDIDATE_FILES = {
    "precision_resolved_v1": "precision_resolved_candidates.csv",
    "precision_resolved_v2": "precision_resolved_candidates_v2.csv",
}


@dataclass(frozen=True)
class RecallMaskOverlay:
    candidates: pd.DataFrame
    identity_fields: Dict[str, Any]
    delivery_dir: Path | None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> Dict[str, Any]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"reviewed overlay manifest must be an object: {path}")
    return payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _validate_manifest_artifacts(root: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("reviewed overlay manifest is missing artifacts")
    for relative_path, raw_record in artifacts.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"invalid reviewed overlay artifact: {relative_path}")
        path = Path(str(raw_record.get("path", ""))).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"reviewed overlay artifact is outside delivery: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"reviewed overlay artifact is missing: {path}")
        if path.stat().st_size != int(raw_record.get("size_bytes", -1)):
            raise ValueError(f"reviewed overlay artifact size mismatch: {relative_path}")
        if _file_sha256(path) != str(raw_record.get("sha256", "")):
            raise ValueError(
                f"reviewed overlay artifact SHA-256 mismatch: {relative_path}"
            )


def _validate_candidates(
    candidates: pd.DataFrame,
    *,
    root: Path,
    sample_id: str,
    image_shape_yx: Tuple[int, int],
) -> None:
    required = {
        "detector_component_id",
        "review_key",
        "accepted",
        "detector_score",
        "acceptance_mode",
        "center_x",
        "center_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "mask_path",
        "reviewed_mask_sha256",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"reviewed overlay candidate table is missing columns: {sorted(missing)}"
        )
    if candidates["review_key"].astype(str).duplicated().any():
        raise ValueError("reviewed overlay review_key values are not unique")
    if candidates["detector_component_id"].astype(str).duplicated().any():
        raise ValueError("reviewed overlay detector_component_id values are not unique")
    if not all(_as_bool(value) for value in candidates["accepted"]):
        raise ValueError("reviewed overlay candidate table contains unaccepted rows")
    for row in candidates.to_dict("records"):
        path = Path(str(row["mask_path"]))
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"reviewed overlay mask is outside delivery: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"reviewed overlay mask is missing: {path}")
        if _file_sha256(path) != str(row["reviewed_mask_sha256"]):
            raise ValueError(f"reviewed overlay mask SHA-256 mismatch: {path}")
        persisted = load_candidate_mask(path)
        if persisted.image_shape_yx != image_shape_yx:
            raise ValueError(f"reviewed overlay mask image shape mismatch: {path}")
        if str(persisted.metadata.get("sample_id", "")) != sample_id:
            raise ValueError(f"reviewed overlay mask sample mismatch: {path}")
        if persisted.bbox.as_tuple() != tuple(
            int(round(float(row[field])))
            for field in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
        ):
            raise ValueError(f"reviewed overlay mask bounding box mismatch: {path}")


def load_recall_mask_overlay(
    detector_candidates: pd.DataFrame,
    *,
    sample_identity: Mapping[str, Any],
    image_shape_yx: Tuple[int, int],
    overlay_dir: Path | None = None,
) -> RecallMaskOverlay:
    """Return automatic masks or a fully validated reviewed delivery."""

    if overlay_dir is None:
        return RecallMaskOverlay(
            candidates=detector_candidates.copy(),
            identity_fields={},
            delivery_dir=None,
        )
    root = Path(overlay_dir).resolve()
    manifest_path = root / "precision_resolved_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"reviewed overlay manifest does not exist: {manifest_path}"
        )
    manifest = _read_manifest(manifest_path)
    delivery_name = str(manifest.get("delivery_name", ""))
    candidate_name = REVIEWED_OVERLAY_CANDIDATE_FILES.get(delivery_name)
    if candidate_name is None:
        raise ValueError(
            f"unsupported reviewed Recall overlay delivery: {delivery_name!r}"
        )
    if manifest.get("sample") != dict(sample_identity):
        raise ValueError("reviewed overlay sample identity does not match detector output")
    if manifest.get("precision_complete") is not True:
        raise ValueError("reviewed overlay Precision review is not complete")
    if int(manifest.get("unresolved_manual_count", -1)) != 0:
        raise ValueError("reviewed overlay still contains unresolved manual boundaries")
    _validate_manifest_artifacts(root, manifest)
    candidates_path = root / candidate_name
    if not candidates_path.is_file():
        raise FileNotFoundError(
            f"reviewed overlay candidate table is missing: {candidates_path}"
        )
    candidates = pd.read_csv(candidates_path)
    expected_count = int(manifest.get("resolved_label_count", -1))
    if len(candidates) != expected_count:
        raise ValueError("reviewed overlay candidate count does not match manifest")
    if "mask_source_dir" not in candidates:
        candidates["mask_source_dir"] = str(root)
    else:
        candidates["mask_source_dir"] = str(root)
    if "display_id" not in candidates:
        candidates["display_id"] = [
            f"#R{index:03d}" for index in range(1, len(candidates) + 1)
        ]
    if "detection_pass" not in candidates:
        candidates["detection_pass"] = "reviewed_precision"
    _validate_candidates(
        candidates,
        root=root,
        sample_id=str(sample_identity["sample_id"]),
        image_shape_yx=image_shape_yx,
    )
    identity_fields = {
        "overlay_mode": "reviewed_delivery",
        "overlay_delivery_name": delivery_name,
        "overlay_delivery_dir": str(root),
        "overlay_manifest": str(manifest_path),
        "overlay_manifest_sha256": _file_sha256(manifest_path),
        "overlay_candidate_table": str(candidates_path),
        "overlay_candidate_table_sha256": _file_sha256(candidates_path),
        "overlay_candidate_count": len(candidates),
    }
    return RecallMaskOverlay(
        candidates=candidates,
        identity_fields=identity_fields,
        delivery_dir=root,
    )


def overlay_dir_from_identity(identity: Mapping[str, Any]) -> Path | None:
    """Recover a reviewed overlay path from a bound Recall identity."""

    mode = str(identity.get("overlay_mode", ""))
    if not mode:
        return None
    if mode != "reviewed_delivery":
        raise ValueError(f"unsupported Recall overlay mode: {mode!r}")
    value = str(identity.get("overlay_delivery_dir", "")).strip()
    if not value:
        raise ValueError("Recall overlay identity is missing overlay_delivery_dir")
    return Path(value).resolve()


__all__ = [
    "REVIEWED_OVERLAY_CANDIDATE_FILES",
    "RecallMaskOverlay",
    "load_recall_mask_overlay",
    "overlay_dir_from_identity",
]
