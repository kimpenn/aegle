"""Build and validate immutable reviewed oocyte release packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import tifffile
import yaml
import zarr

from .release_console import write_embedded_release_console


OOCYTE_RELEASE_SCHEMA_VERSION = 1
OOCYTE_RELEASE_IMPLEMENTATION_VERSION = "oocyte_release_v5"
RELEASE_ROLES = {"positive", "negative_control"}
PROFILE_ARTIFACT_NAMES = (
    "oocyte_by_marker.csv",
    "oocyte_metadata.csv",
    "oocyte_overview.csv",
    "channel_manifest.csv",
)


@dataclass(frozen=True)
class OocyteReleaseSample:
    sample_id: str
    role: Literal["positive", "negative_control"]
    image_path: Path
    antibodies_path: Path
    final_labels_path: Path
    final_mapping_path: Path
    final_candidates_path: Path
    profiling_dir: Path
    review_exports: tuple[Path, ...]
    provenance_files: tuple[Path, ...]
    detector_candidates_path: Path | None = None
    rescue_diagnostics_path: Path | None = None


@dataclass(frozen=True)
class OocyteReleaseSpec:
    release_name: str
    samples: tuple[OocyteReleaseSample, ...]
    algorithm_document: Path | None = None


@dataclass(frozen=True)
class OocyteReleaseResult:
    release_dir: Path
    manifest_path: Path
    sample_count: int
    oocyte_count: int
    validation: Dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "sha256": _file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _concat_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    populated = [table for table in tables if not table.empty]
    if populated:
        return pd.concat(populated, ignore_index=True)
    return tables[0].iloc[0:0].copy()


def _read_object(path: Path) -> Dict[str, Any]:
    source = Path(path)
    with source.open() as handle:
        if source.suffix.casefold() == ".json":
            value = json.load(handle)
        else:
            value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"release spec must be an object: {source}")
    return value


def _resolve_path(base: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"release sample is missing {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_optional_path(base: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_path_list(base: Path, value: Any, *, field: str) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"release sample {field} must be a list")
    return tuple(_resolve_path(base, item, field=field) for item in value)


def load_oocyte_release_spec(path: Path) -> OocyteReleaseSpec:
    """Load and resolve one YAML or JSON release specification."""

    source = Path(path).resolve()
    payload = _read_object(source)
    release_name = str(payload.get("release_name", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", release_name):
        raise ValueError("release_name must be a safe non-empty identifier")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("release spec samples must be a non-empty list")
    samples = []
    seen = set()
    base = source.parent
    for index, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"release sample {index} must be an object")
        sample_id = str(raw.get("sample_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id):
            raise ValueError(f"invalid release sample_id: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"duplicate release sample_id: {sample_id}")
        seen.add(sample_id)
        role = str(raw.get("role", "")).strip()
        if role not in RELEASE_ROLES:
            raise ValueError(f"invalid role for {sample_id}: {role!r}")
        samples.append(
            OocyteReleaseSample(
                sample_id=sample_id,
                role=role,  # type: ignore[arg-type]
                image_path=_resolve_path(base, raw.get("image"), field="image"),
                antibodies_path=_resolve_path(
                    base,
                    raw.get("antibodies"),
                    field="antibodies",
                ),
                final_labels_path=_resolve_path(
                    base,
                    raw.get("final_labels"),
                    field="final_labels",
                ),
                final_mapping_path=_resolve_path(
                    base,
                    raw.get("final_mapping"),
                    field="final_mapping",
                ),
                final_candidates_path=_resolve_path(
                    base,
                    raw.get("final_candidates"),
                    field="final_candidates",
                ),
                profiling_dir=_resolve_path(
                    base,
                    raw.get("profiling_dir"),
                    field="profiling_dir",
                ),
                review_exports=_resolve_path_list(
                    base,
                    raw.get("review_exports"),
                    field="review_exports",
                ),
                provenance_files=_resolve_path_list(
                    base,
                    raw.get("provenance_files"),
                    field="provenance_files",
                ),
                detector_candidates_path=_resolve_optional_path(
                    base,
                    raw.get("detector_candidates"),
                ),
                rescue_diagnostics_path=_resolve_optional_path(
                    base,
                    raw.get("rescue_diagnostics"),
                ),
            )
        )
    image_paths = [sample.image_path for sample in samples]
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("release samples must not reuse a source image")
    return OocyteReleaseSpec(
        release_name=release_name,
        samples=tuple(samples),
        algorithm_document=_resolve_optional_path(
            base,
            payload.get("algorithm_document"),
        ),
    )


def _require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def _label_summary(path: Path, *, chunk_height: int = 1024) -> Dict[str, Any]:
    labels_seen: set[int] = set()
    counts = np.zeros(1, dtype=np.int64)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        if axes != "YX":
            raise ValueError(f"release label image must have YX axes: {path} has {axes}")
        if np.dtype(series.dtype) != np.dtype(np.uint16):
            raise ValueError(f"release label image must be uint16: {path}")
        store = series.aszarr()
        try:
            array = zarr.open(store, mode="r")
            height, width = shape
            for y0 in range(0, height, chunk_height):
                block = np.asarray(array[y0 : min(height, y0 + chunk_height), :])
                block_counts = np.bincount(block.ravel())
                if len(block_counts) > len(counts):
                    counts = np.pad(counts, (0, len(block_counts) - len(counts)))
                counts[: len(block_counts)] += block_counts
                labels_seen.update(int(value) for value in np.unique(block) if value)
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()
    positive = sorted(labels_seen)
    return {
        "shape_yx": [int(shape[0]), int(shape[1])],
        "positive_labels": positive,
        "positive_label_count": len(positive),
        "assigned_pixel_count": int(counts[1:].sum()),
        "label_pixel_counts": {
            int(label): int(counts[label]) for label in positive
        },
    }


def _resolve_mask_path(candidates_path: Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("mask_path", "")))
    if path.is_absolute():
        return path.resolve()
    source_dir = str(row.get("mask_source_dir", "")).strip()
    if source_dir:
        return (Path(source_dir) / path).resolve()
    return (Path(candidates_path).parent / path).resolve()


def _load_packaged_mask(path: Path) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"mask", "bbox_xyxy", "image_shape_yx"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"release mask archive missing fields: {sorted(missing)}")
        mask = np.asarray(archive["mask"], dtype=np.bool_)
        bbox = tuple(int(value) for value in archive["bbox_xyxy"].tolist())
        image_shape = tuple(
            int(value) for value in archive["image_shape_yx"].tolist()
        )
    return mask, bbox, image_shape


def _validate_packaged_masks(
    labels_path: Path,
    final_dir: Path,
    mapping: pd.DataFrame,
    candidates: pd.DataFrame,
    image_shape_yx: Sequence[int],
) -> None:
    candidate_paths = {
        str(row["detector_component_id"]): str(row["mask_path"])
        for row in candidates.to_dict("records")
    }
    with tifffile.TiffFile(labels_path) as tif:
        store = tif.series[0].aszarr()
        try:
            labels = zarr.open(store, mode="r")
            for row in mapping.to_dict("records"):
                component_id = str(row["detector_component_id"])
                relative_mask_path = str(row["mask_path"])
                if candidate_paths.get(component_id) != relative_mask_path:
                    raise ValueError(
                        f"release mask paths differ for object: {component_id}"
                    )
                mask_path = (final_dir / relative_mask_path).resolve()
                if not mask_path.is_relative_to(final_dir) or not mask_path.is_file():
                    raise ValueError(f"release mask path is invalid: {component_id}")
                mask, bbox, mask_image_shape = _load_packaged_mask(mask_path)
                if len(bbox) != 4 or tuple(image_shape_yx) != mask_image_shape:
                    raise ValueError(f"release mask geometry is invalid: {component_id}")
                x0, y0, x1, y1 = bbox
                if not (0 <= x0 < x1 <= image_shape_yx[1]):
                    raise ValueError(f"release mask X bounds are invalid: {component_id}")
                if not (0 <= y0 < y1 <= image_shape_yx[0]):
                    raise ValueError(f"release mask Y bounds are invalid: {component_id}")
                if mask.shape != (y1 - y0, x1 - x0):
                    raise ValueError(f"release mask shape is invalid: {component_id}")
                expected_bbox = tuple(
                    int(row[name])
                    for name in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
                )
                if bbox != expected_bbox:
                    raise ValueError(f"release mask bbox mismatch: {component_id}")
                assigned = int(row["assigned_pixel_count"])
                if int(mask.sum()) != assigned:
                    raise ValueError(f"release mask pixel count mismatch: {component_id}")
                label = int(row["label"])
                label_patch = np.asarray(labels[y0:y1, x0:x1]) == label
                if not np.array_equal(mask, label_patch):
                    raise ValueError(f"release mask/label pixels differ: {component_id}")
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()


def _safe_mask_name(index: int, component_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", component_id).strip("._")
    if not safe:
        safe = "object"
    return f"mask-{index:04d}__{safe}.npz"


def _validate_profile_source(sample: OocyteReleaseSample) -> Dict[str, Any]:
    manifest_path = sample.profiling_dir / "profiling_manifest.json"
    _require_files([manifest_path, *(sample.profiling_dir / name for name in PROFILE_ARTIFACT_NAMES)])
    manifest = _read_object(manifest_path)
    if str(manifest.get("sample_id")) != sample.sample_id:
        raise ValueError(f"profiling sample mismatch for {sample.sample_id}")
    image = manifest.get("source_image")
    if not isinstance(image, dict):
        raise ValueError(f"profiling source image identity missing for {sample.sample_id}")
    stat = sample.image_path.stat()
    expected_image = {
        "path": str(sample.image_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": str(stat.st_mtime_ns),
    }
    for field, expected in expected_image.items():
        if str(image.get(field)) != str(expected):
            raise ValueError(
                f"profiling source image {field} mismatch for {sample.sample_id}"
            )
    antibodies = manifest.get("antibodies")
    if not isinstance(antibodies, dict):
        raise ValueError(f"profiling antibody identity missing for {sample.sample_id}")
    if str(antibodies.get("sha256")) != _file_sha256(sample.antibodies_path):
        raise ValueError(f"profiling antibody SHA mismatch for {sample.sample_id}")
    for name, record in manifest.get("artifacts", {}).items():
        path = sample.profiling_dir / str(name)
        if not path.is_file() or _artifact_record(path) != {
            "sha256": str(record.get("sha256")),
            "size_bytes": int(record.get("size_bytes", -1)),
        }:
            raise ValueError(f"profiling artifact mismatch for {sample.sample_id}: {name}")
    return manifest


def _validate_negative_control_inputs(sample: OocyteReleaseSample) -> Dict[str, Any]:
    if sample.role != "negative_control":
        return {}
    if sample.detector_candidates_path is None or sample.rescue_diagnostics_path is None:
        raise ValueError(
            f"negative control {sample.sample_id} requires detector and rescue diagnostics"
        )
    _require_files([sample.detector_candidates_path, sample.rescue_diagnostics_path])
    detector = pd.read_csv(sample.detector_candidates_path)
    rescue = pd.read_csv(sample.rescue_diagnostics_path)
    accepted = int(detector["accepted"].fillna(False).astype(bool).sum())
    rescue_accepted = int(rescue["rescue_status"].astype(str).eq("accepted").sum())
    if accepted or rescue_accepted:
        raise ValueError(
            f"negative control {sample.sample_id} contains accepted detector objects"
        )
    return {
        "baseline_refined_count": len(detector),
        "baseline_accepted_count": accepted,
        "rescue_evaluation_count": len(rescue),
        "rescue_accepted_count": rescue_accepted,
        "detector_candidates_sha256": _file_sha256(sample.detector_candidates_path),
        "rescue_diagnostics_sha256": _file_sha256(sample.rescue_diagnostics_path),
    }


def _copy_evidence(
    sample: OocyteReleaseSample,
    review_dir: Path,
    negative_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    review_dir.mkdir(parents=True, exist_ok=True)
    records = []
    used_names: set[str] = set()
    evidence_groups = [
        ("review_export", sample.review_exports),
        ("provenance", sample.provenance_files),
    ]
    if sample.detector_candidates_path is not None:
        evidence_groups.append(
            ("negative_control_detector", (sample.detector_candidates_path,))
        )
    if sample.rescue_diagnostics_path is not None:
        evidence_groups.append(
            ("negative_control_rescue", (sample.rescue_diagnostics_path,))
        )
    for category, paths in evidence_groups:
        for index, source in enumerate(paths, start=1):
            name = source.name
            if name in used_names:
                name = f"{category}-{index:02d}__{name}"
            used_names.add(name)
            destination = review_dir / name
            shutil.copy2(source, destination)
            records.append(
                {
                    "category": category,
                    "name": name,
                    "source_path": str(source),
                    **_artifact_record(destination),
                }
            )
    payload: Dict[str, Any] = {
        "schema_version": OOCYTE_RELEASE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "role": sample.role,
        "records": records,
    }
    if negative_evidence:
        payload["negative_control_signoff"] = {
            **dict(negative_evidence),
            "biological_expectation": "no_oocytes",
            "status": "passed",
        }
    _write_json(review_dir / "review_manifest.json", payload)
    return payload


def _release_profile_manifest(
    source_manifest: Mapping[str, Any],
    *,
    profile_dir: Path,
    final_dir: Path,
    source_manifest_path: Path,
) -> Dict[str, Any]:
    payload = dict(source_manifest)
    payload["release_packaged"] = True
    payload["source_profiling_manifest"] = {
        "path": str(source_manifest_path),
        **_artifact_record(source_manifest_path),
    }
    payload["label_image"] = {
        "path": "../final/oocyte_labels.ome.tiff",
        **_artifact_record(final_dir / "oocyte_labels.ome.tiff"),
    }
    payload["mapping"] = {
        "path": "../final/oocyte_labels.csv",
        **_artifact_record(final_dir / "oocyte_labels.csv"),
    }
    payload["candidates"] = {
        "path": "../final/oocyte_candidates.csv",
        **_artifact_record(final_dir / "oocyte_candidates.csv"),
    }
    payload["artifacts"] = {
        name: _artifact_record(profile_dir / name) for name in PROFILE_ARTIFACT_NAMES
    }
    return payload


def _batch_index_html(
    release_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    cards = "".join(
        f'<li><a href="samples/{row["sample_id"]}/review_console.html">{row["sample_id"]}</a> '
        f'<span>{row["role"]} · {row["oocyte_count"]} oocytes</span></li>'
        for row in rows
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{release_name}</title><style>body{{margin:0;background:linear-gradient(135deg,#eee4d3,#dcebe3);color:#172522;font-family:Georgia,serif}}main{{width:min(1000px,calc(100% - 32px));margin:30px auto}}header,li{{background:#fffaf0;border:1px solid #cbbfa9;border-radius:18px;padding:22px;margin:12px 0}}h1{{font-size:3.6rem;margin:.2em 0}}ul{{list-style:none;padding:0}}li{{display:flex;justify-content:space-between}}a{{color:#087b78;font-weight:bold}}</style></head><body><main><header><div>Aegle / raw UCHL1</div><h1>{release_name}</h1><p>Immutable reviewed masks and raw within-mask marker profiles.</p></header><ul>{cards}</ul></main></body></html>'''


def _build_sample(
    sample: OocyteReleaseSample,
    destination: Path,
) -> Dict[str, Any]:
    required = [
        sample.image_path,
        sample.antibodies_path,
        sample.final_labels_path,
        sample.final_mapping_path,
        sample.final_candidates_path,
        *sample.review_exports,
        *sample.provenance_files,
    ]
    _require_files(required)
    source_profile_manifest = _validate_profile_source(sample)
    negative_evidence = _validate_negative_control_inputs(sample)
    candidates = pd.read_csv(sample.final_candidates_path)
    mapping = pd.read_csv(sample.final_mapping_path)
    markers = pd.read_csv(sample.profiling_dir / "oocyte_by_marker.csv")
    metadata = pd.read_csv(sample.profiling_dir / "oocyte_metadata.csv")
    if len(candidates) != len(mapping) or len(mapping) != len(markers):
        raise ValueError(f"release row count mismatch for {sample.sample_id}")
    required_columns = {"detector_component_id", "mask_path"}
    for name, table in (("candidates", candidates), ("mapping", mapping)):
        missing = required_columns.difference(table.columns)
        if missing:
            raise ValueError(
                f"{sample.sample_id} {name} missing columns: {sorted(missing)}"
            )
    candidate_ids = candidates["detector_component_id"].astype(str)
    mapping_ids = mapping["detector_component_id"].astype(str)
    if candidate_ids.duplicated().any() or mapping_ids.duplicated().any():
        raise ValueError(f"duplicate final object IDs for {sample.sample_id}")
    if set(candidate_ids) != set(mapping_ids):
        raise ValueError(f"candidate/mapping IDs differ for {sample.sample_id}")
    expected_oocyte_ids = {f"{sample.sample_id}__{value}" for value in mapping_ids}
    if set(markers.get("oocyte_id", pd.Series(dtype=str)).astype(str)) != expected_oocyte_ids:
        raise ValueError(f"profiling object IDs differ for {sample.sample_id}")
    label_summary = _label_summary(sample.final_labels_path)
    expected_labels = set(int(value) for value in mapping["label"])
    if set(label_summary["positive_labels"]) != expected_labels:
        raise ValueError(f"label/mapping IDs differ for {sample.sample_id}")
    if sample.role == "positive" and not len(mapping):
        raise ValueError(f"positive sample has no oocytes: {sample.sample_id}")
    if sample.role == "negative_control" and len(mapping):
        raise ValueError(f"negative control has positive labels: {sample.sample_id}")

    final_dir = destination / "final"
    masks_dir = final_dir / "masks"
    profile_dir = destination / "profiling"
    review_dir = destination / "review"
    masks_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    mask_names: Dict[str, str] = {}
    candidate_records = candidates.to_dict("records")
    for index, row in enumerate(candidate_records, start=1):
        component_id = str(row["detector_component_id"])
        source_mask = _resolve_mask_path(sample.final_candidates_path, row)
        if not source_mask.is_file():
            raise FileNotFoundError(source_mask)
        expected_sha = str(row.get("reviewed_mask_sha256", "")).strip()
        if expected_sha and _file_sha256(source_mask) != expected_sha:
            raise ValueError(f"final mask SHA mismatch: {source_mask}")
        mask_name = _safe_mask_name(index, component_id)
        shutil.copy2(source_mask, masks_dir / mask_name)
        mask_names[component_id] = mask_name
    if len(mask_names) != len(candidates):
        raise ValueError(f"release mask IDs are not unique for {sample.sample_id}")
    candidates = candidates.copy()
    mapping = mapping.copy()
    candidates["mask_path"] = [
        f"masks/{mask_names[str(value)]}" for value in candidate_ids
    ]
    mapping["mask_path"] = [
        f"masks/{mask_names[str(value)]}" for value in mapping_ids
    ]
    for table in (candidates, mapping):
        if "mask_source_dir" in table:
            table.drop(columns=["mask_source_dir"], inplace=True)
    shutil.copy2(sample.final_labels_path, final_dir / "oocyte_labels.ome.tiff")
    _write_csv(final_dir / "oocyte_labels.csv", mapping)
    _write_csv(final_dir / "oocyte_candidates.csv", candidates)

    for name in PROFILE_ARTIFACT_NAMES:
        source = sample.profiling_dir / name
        if name == "oocyte_metadata.csv":
            table = metadata.copy()
            if "mask_path" in table:
                table["mask_path"] = [
                    f"../final/masks/{mask_names[str(value)]}"
                    for value in table["detector_component_id"].astype(str)
                ]
            _write_csv(profile_dir / name, table)
        else:
            shutil.copy2(source, profile_dir / name)
    release_profile_manifest = _release_profile_manifest(
        source_profile_manifest,
        profile_dir=profile_dir,
        final_dir=final_dir,
        source_manifest_path=sample.profiling_dir / "profiling_manifest.json",
    )
    _write_json(profile_dir / "profiling_manifest.json", release_profile_manifest)
    review_manifest = _copy_evidence(sample, review_dir, negative_evidence)
    console_summary = write_embedded_release_console(
        sample_id=sample.sample_id,
        role=sample.role,
        source_image_path=sample.image_path,
        antibodies_path=sample.antibodies_path,
        final_dir=final_dir,
        profiling_dir=profile_dir,
        destination=destination / "review_console.html",
    )

    sample_artifacts = {
        str(path.relative_to(destination)): _artifact_record(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "sample_release_manifest.json"
    }
    sample_manifest = {
        "schema_version": OOCYTE_RELEASE_SCHEMA_VERSION,
        "implementation_version": OOCYTE_RELEASE_IMPLEMENTATION_VERSION,
        "sample_id": sample.sample_id,
        "role": sample.role,
        "oocyte_count": len(mapping),
        "channel_count": int(source_profile_manifest.get("channel_count", -1)),
        "assigned_pixel_count": label_summary["assigned_pixel_count"],
        "image_shape_yx": label_summary["shape_yx"],
        "source_image": {
            "path": str(sample.image_path),
            "size_bytes": int(sample.image_path.stat().st_size),
            "mtime_ns": str(sample.image_path.stat().st_mtime_ns),
        },
        "antibodies": {
            "path": str(sample.antibodies_path),
            **_artifact_record(sample.antibodies_path),
        },
        "source_final_artifacts": {
            "labels": {
                "path": str(sample.final_labels_path),
                **_artifact_record(sample.final_labels_path),
            },
            "mapping": {
                "path": str(sample.final_mapping_path),
                **_artifact_record(sample.final_mapping_path),
            },
            "candidates": {
                "path": str(sample.final_candidates_path),
                **_artifact_record(sample.final_candidates_path),
            },
        },
        "review_record_count": len(review_manifest["records"]),
        "embedded_console": console_summary,
        "artifacts": sample_artifacts,
    }
    _write_json(destination / "sample_release_manifest.json", sample_manifest)
    return sample_manifest


def _readme_text(release_name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    positives = sum(int(row["oocyte_count"]) for row in rows if row["role"] == "positive")
    channel_counts = sorted({int(row["channel_count"]) for row in rows})
    channels = str(channel_counts[0]) if len(channel_counts) == 1 else str(channel_counts)
    return f"""# {release_name}

Reviewed raw-UCHL1 oocyte segmentation release for panel1.

- Positive oocytes: {positives}
- Samples: {len(rows)}
- Measurement: raw within-mask mean fluorescence
- Channels: {channels}, including DAPI for acquisition traceability
- DeepCell dependency: none

`batch_oocyte_by_marker.csv` and `batch_oocyte_metadata.csv` are the cohort-level tables. Each sample directory contains exact masks, a whole-slide label image, mapping, candidates, profiling outputs, review evidence, and a checksum manifest. Raw OME-TIFFs are referenced by immutable file identity and are not copied into this release.

Run `python src/run_oocyte_release.py validate --release-dir <path>` from the Aegle repository to verify hashes and cross-file invariants.
"""


def build_oocyte_release(spec_path: Path, out_dir: Path) -> OocyteReleaseResult:
    """Build one atomic immutable release and validate it before publication."""

    spec_source = Path(spec_path).resolve()
    spec = load_oocyte_release_spec(spec_source)
    destination = Path(out_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"release output is immutable and already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        packaged_spec = temporary / f"release_spec{spec_source.suffix.casefold()}"
        shutil.copy2(spec_source, packaged_spec)
        sample_rows = []
        for sample in spec.samples:
            sample_destination = temporary / "samples" / sample.sample_id
            manifest = _build_sample(sample, sample_destination)
            sample_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "role": sample.role,
                    "oocyte_count": int(manifest["oocyte_count"]),
                    "channel_count": int(manifest["channel_count"]),
                    "assigned_pixel_count": int(manifest["assigned_pixel_count"]),
                    "source_image": str(sample.image_path),
                    "sample_manifest_sha256": _file_sha256(
                        sample_destination / "sample_release_manifest.json"
                    ),
                }
            )
        marker_tables = [
            pd.read_csv(temporary / "samples" / row["sample_id"] / "profiling" / "oocyte_by_marker.csv")
            for row in sample_rows
        ]
        metadata_tables = [
            pd.read_csv(temporary / "samples" / row["sample_id"] / "profiling" / "oocyte_metadata.csv")
            for row in sample_rows
        ]
        marker_columns = [tuple(table.columns) for table in marker_tables]
        metadata_columns = [tuple(table.columns) for table in metadata_tables]
        if len(set(marker_columns)) != 1 or len(set(metadata_columns)) != 1:
            raise ValueError("sample profiling schemas differ within release")
        batch_markers = _concat_tables(marker_tables)
        batch_metadata = _concat_tables(metadata_tables)
        _write_csv(temporary / "batch_summary.csv", pd.DataFrame(sample_rows))
        _write_csv(temporary / "batch_oocyte_by_marker.csv", batch_markers)
        _write_csv(temporary / "batch_oocyte_metadata.csv", batch_metadata)
        if spec.algorithm_document is not None:
            _require_files([spec.algorithm_document])
            shutil.copy2(
                spec.algorithm_document,
                temporary / "oocyte_detection_algorithm.html",
            )
        _atomic_write_text(
            temporary / "oocyte_review_index.html",
            _batch_index_html(spec.release_name, sample_rows),
        )
        _atomic_write_text(
            temporary / "README.md",
            _readme_text(spec.release_name, sample_rows),
        )
        package_artifacts = {
            str(path.relative_to(temporary)): _artifact_record(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name != "release_manifest.json"
        }
        manifest = {
            "schema_version": OOCYTE_RELEASE_SCHEMA_VERSION,
            "implementation_version": OOCYTE_RELEASE_IMPLEMENTATION_VERSION,
            "release_name": spec.release_name,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "spec": {
                "source_path": str(spec_source),
                "path": packaged_spec.name,
                **_artifact_record(packaged_spec),
            },
            "sample_count": len(sample_rows),
            "positive_oocyte_count": int(
                sum(row["oocyte_count"] for row in sample_rows if row["role"] == "positive")
            ),
            "negative_control_count": int(
                sum(row["role"] == "negative_control" for row in sample_rows)
            ),
            "samples": sample_rows,
            "artifacts": package_artifacts,
        }
        _write_json(temporary / "release_manifest.json", manifest)
        validate_oocyte_release(temporary)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validation = validate_oocyte_release(destination)
    return OocyteReleaseResult(
        release_dir=destination,
        manifest_path=destination / "release_manifest.json",
        sample_count=int(validation["sample_count"]),
        oocyte_count=int(validation["positive_oocyte_count"]),
        validation=validation,
    )


def _verify_artifact_index(root: Path, records: Mapping[str, Any]) -> None:
    for relative, raw in records.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid artifact record: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"release artifact is missing: {relative}")
        expected = {
            "sha256": str(raw.get("sha256", "")),
            "size_bytes": int(raw.get("size_bytes", -1)),
        }
        if _artifact_record(path) != expected:
            raise ValueError(f"release artifact mismatch: {relative}")


def validate_oocyte_release(release_dir: Path) -> Dict[str, Any]:
    """Validate every checksum and cross-file invariant in a release."""

    root = Path(release_dir).resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != OOCYTE_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported oocyte release schema_version")
    implementation_version = str(manifest.get("implementation_version"))
    embedded_console_required = implementation_version != "oocyte_release_v1"
    overview_console_required = implementation_version in {
        "oocyte_release_v3",
        "oocyte_release_v4",
        "oocyte_release_v5",
    }
    nearest_hotspot_required = implementation_version in {
        "oocyte_release_v4",
        "oocyte_release_v5",
    }
    hotspot_toggle_required = implementation_version == "oocyte_release_v5"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("release manifest is missing artifacts")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "release_manifest.json"
    }
    if set(artifacts) != actual_files:
        raise ValueError("release artifact index does not match package files")
    _verify_artifact_index(root, artifacts)
    sample_rows = manifest.get("samples")
    if not isinstance(sample_rows, list) or not sample_rows:
        raise ValueError("release manifest has no samples")
    seen_samples = set()
    marker_columns: tuple[str, ...] | None = None
    total_positive = 0
    all_marker_ids: set[str] = set()
    per_sample = []
    for row in sample_rows:
        sample_id = str(row.get("sample_id", ""))
        role = str(row.get("role", ""))
        if sample_id in seen_samples or role not in RELEASE_ROLES:
            raise ValueError("invalid or duplicate sample in release manifest")
        seen_samples.add(sample_id)
        sample_root = root / "samples" / sample_id
        sample_manifest_path = sample_root / "sample_release_manifest.json"
        sample_manifest = _read_object(sample_manifest_path)
        if sample_manifest.get("sample_id") != sample_id or sample_manifest.get("role") != role:
            raise ValueError(f"sample manifest identity mismatch: {sample_id}")
        sample_artifacts = sample_manifest.get("artifacts")
        if not isinstance(sample_artifacts, Mapping):
            raise ValueError(f"sample artifact index is missing: {sample_id}")
        _verify_artifact_index(sample_root, sample_artifacts)
        labels_path = sample_root / "final" / "oocyte_labels.ome.tiff"
        mapping = pd.read_csv(sample_root / "final" / "oocyte_labels.csv")
        candidates = pd.read_csv(sample_root / "final" / "oocyte_candidates.csv")
        markers = pd.read_csv(sample_root / "profiling" / "oocyte_by_marker.csv")
        metadata = pd.read_csv(sample_root / "profiling" / "oocyte_metadata.csv")
        label_summary = _label_summary(labels_path)
        count = int(sample_manifest.get("oocyte_count", -1))
        if not (
            count == len(mapping) == len(candidates) == len(markers) == len(metadata)
            == label_summary["positive_label_count"]
        ):
            raise ValueError(f"release row/label count mismatch: {sample_id}")
        if embedded_console_required:
            console_path = sample_root / "review_console.html"
            console_text = console_path.read_text(encoding="utf-8")
            console_summary = sample_manifest.get("embedded_console")
            if not isinstance(console_summary, Mapping):
                raise ValueError(
                    f"release embedded console summary missing: {sample_id}"
                )
            embedded_card_count = len(
                re.findall(
                    r"<article\b[^>]*\bdata-embedded-review-card(?:\s|>)",
                    console_text,
                )
            )
            embedded_webp_count = console_text.count("data:image/webp;base64,")
            overview_webp_count = 2 if count else 1
            expected_webp_count = count * 2
            if overview_console_required:
                expected_webp_count += overview_webp_count
            if (
                int(console_summary.get("embedded_card_count", -1)) != count
                or int(console_summary.get("embedded_webp_count", -1))
                != expected_webp_count
                or embedded_card_count != count
                or embedded_webp_count != expected_webp_count
                or f'data-embedded-card-count="{count}"' not in console_text
            ):
                raise ValueError(
                    f"release embedded console count mismatch: {sample_id}"
                )
            if overview_console_required:
                hotspot_count = len(
                    re.findall(r"<circle\b[^>]*\bdata-global-hotspot(?:\s|>)", console_text)
                )
                if (
                    int(console_summary.get("overview_webp_count", -1))
                    != overview_webp_count
                    or int(console_summary.get("global_hotspot_count", -1))
                    != count
                    or int(console_summary.get("overview_downsample", -1)) <= 1
                    or console_text.count("data-global-overview") != 1
                    or hotspot_count != count
                    or console_text.count('href="#oocyte-') != count
                    or console_text.count('id="oocyte-') != count
                    or "{overview_html}" in console_text
                ):
                    raise ValueError(
                        f"release whole-slide console mismatch: {sample_id}"
                    )
            if nearest_hotspot_required and (
                console_text.count("data-card-target=") != count
                or "best>Number(nearest.getAttribute('r'))**2" not in console_text
            ):
                raise ValueError(
                    f"release nearest-hotspot navigation mismatch: {sample_id}"
                )
            if hotspot_toggle_required and (
                "hotspots.toggleAttribute('hidden',showRaw)" not in console_text
            ):
                raise ValueError(
                    f"release whole-slide mask toggle mismatch: {sample_id}"
                )
            required_console_links = (
                "profiling/oocyte_by_marker.csv",
                "profiling/oocyte_metadata.csv",
                "final/oocyte_labels.ome.tiff",
                "final/oocyte_labels.csv",
                "review/review_manifest.json",
                "sample_release_manifest.json",
            )
            if any(link not in console_text for link in required_console_links):
                raise ValueError(f"release embedded console links missing: {sample_id}")
            if "/api/" in console_text or "http://" in console_text:
                raise ValueError(f"release console requires a server: {sample_id}")
        labels = set(int(value) for value in mapping["label"])
        if labels != set(label_summary["positive_labels"]):
            raise ValueError(f"release mapping labels mismatch: {sample_id}")
        if labels != set(range(1, count + 1)):
            raise ValueError(f"release labels are not contiguous: {sample_id}")
        if mapping["detector_component_id"].astype(str).duplicated().any():
            raise ValueError(f"release object IDs are duplicated: {sample_id}")
        if set(mapping["detector_component_id"].astype(str)) != set(
            candidates["detector_component_id"].astype(str)
        ):
            raise ValueError(f"release candidate IDs mismatch: {sample_id}")
        expected_oocyte_ids = {
            f"{sample_id}__{value}"
            for value in mapping["detector_component_id"].astype(str)
        }
        marker_ids = set(markers["oocyte_id"].astype(str))
        if marker_ids != expected_oocyte_ids or set(metadata["oocyte_id"].astype(str)) != expected_oocyte_ids:
            raise ValueError(f"release profiling IDs mismatch: {sample_id}")
        if all_marker_ids.intersection(marker_ids):
            raise ValueError("release oocyte IDs are not unique across samples")
        all_marker_ids.update(marker_ids)
        for mapping_row in mapping.to_dict("records"):
            label = int(mapping_row["label"])
            expected_pixels = int(mapping_row["assigned_pixel_count"])
            if expected_pixels <= 0:
                raise ValueError(f"release label is empty: {sample_id} label {label}")
            if label_summary["label_pixel_counts"].get(label) != expected_pixels:
                raise ValueError(f"release assigned pixel mismatch: {sample_id} label {label}")
            if not (
                float(mapping_row["bbox_x0"])
                <= float(mapping_row["center_x"])
                < float(mapping_row["bbox_x1"])
                and float(mapping_row["bbox_y0"])
                <= float(mapping_row["center_y"])
                < float(mapping_row["bbox_y1"])
            ):
                raise ValueError(f"release centroid is outside bbox: {sample_id}")
        _validate_packaged_masks(
            labels_path,
            sample_root / "final",
            mapping,
            candidates,
            label_summary["shape_yx"],
        )
        current_columns = tuple(markers.columns)
        if marker_columns is None:
            marker_columns = current_columns
        elif current_columns != marker_columns:
            raise ValueError("release marker schemas differ")
        if role == "negative_control" and count != 0:
            raise ValueError(f"negative control is nonzero: {sample_id}")
        if role == "positive" and count <= 0:
            raise ValueError(f"positive sample is empty: {sample_id}")
        if role == "positive":
            total_positive += count
        profile_manifest = _read_object(sample_root / "profiling" / "profiling_manifest.json")
        profile_artifacts = profile_manifest.get("artifacts")
        if not isinstance(profile_artifacts, Mapping):
            raise ValueError(f"release profiling manifest missing artifacts: {sample_id}")
        _verify_artifact_index(sample_root / "profiling", profile_artifacts)
        if role == "negative_control":
            review_manifest = _read_object(
                sample_root / "review" / "review_manifest.json"
            )
            signoff = review_manifest.get("negative_control_signoff")
            if not isinstance(signoff, Mapping):
                raise ValueError(f"negative control signoff is missing: {sample_id}")
            if (
                signoff.get("status") != "passed"
                or int(signoff.get("baseline_accepted_count", -1)) != 0
                or int(signoff.get("rescue_accepted_count", -1)) != 0
            ):
                raise ValueError(f"negative control signoff failed: {sample_id}")
        per_sample.append({"sample_id": sample_id, "role": role, "oocyte_count": count})
    batch_summary = pd.read_csv(root / "batch_summary.csv")
    if len(batch_summary) != len(per_sample):
        raise ValueError("release batch summary row count mismatch")
    expected_summary = {
        (row["sample_id"], row["role"], row["oocyte_count"])
        for row in per_sample
    }
    actual_summary = {
        (str(row.sample_id), str(row.role), int(row.oocyte_count))
        for row in batch_summary.itertuples(index=False)
    }
    if actual_summary != expected_summary:
        raise ValueError("release batch summary values mismatch")
    for row in batch_summary.itertuples(index=False):
        sample_manifest_path = (
            root / "samples" / str(row.sample_id) / "sample_release_manifest.json"
        )
        if str(row.sample_manifest_sha256) != _file_sha256(sample_manifest_path):
            raise ValueError(f"release sample manifest SHA mismatch: {row.sample_id}")
    batch_markers = pd.read_csv(root / "batch_oocyte_by_marker.csv")
    batch_metadata = pd.read_csv(root / "batch_oocyte_metadata.csv")
    if len(batch_markers) != total_positive or len(batch_metadata) != total_positive:
        raise ValueError("release cohort table row count mismatch")
    if set(batch_markers["oocyte_id"].astype(str)) != all_marker_ids:
        raise ValueError("release cohort marker IDs mismatch")
    if set(batch_metadata["oocyte_id"].astype(str)) != all_marker_ids:
        raise ValueError("release cohort metadata IDs mismatch")
    if int(manifest.get("positive_oocyte_count", -1)) != total_positive:
        raise ValueError("release positive oocyte total mismatch")
    if int(manifest.get("sample_count", -1)) != len(sample_rows):
        raise ValueError("release sample count mismatch")
    return {
        "status": "valid",
        "release_name": str(manifest.get("release_name", "")),
        "sample_count": len(sample_rows),
        "positive_oocyte_count": total_positive,
        "negative_control_count": sum(row["role"] == "negative_control" for row in per_sample),
        "channel_count": 0 if marker_columns is None else len(marker_columns) - 3,
        "artifact_count": len(artifacts),
        "samples": per_sample,
    }


__all__ = [
    "OOCYTE_RELEASE_IMPLEMENTATION_VERSION",
    "OOCYTE_RELEASE_SCHEMA_VERSION",
    "OocyteReleaseResult",
    "OocyteReleaseSample",
    "OocyteReleaseSpec",
    "build_oocyte_release",
    "load_oocyte_release_spec",
    "validate_oocyte_release",
]
