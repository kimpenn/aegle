"""Profile final oocyte labels directly against registered raw channels."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
import zarr
from skimage.measure import regionprops

from .models import BoundingBox


logger = logging.getLogger(__name__)

OOCYTE_PROFILING_VERSION = "oocyte_profiling_v1"
MAX_LABEL_VALUE = int(np.iinfo(np.uint16).max)

MARKER_ID_COLUMNS = ["sample_id", "oocyte_id", "label_id"]
METADATA_COLUMNS = [
    "sample_id",
    "oocyte_id",
    "label_id",
    "detector_component_id",
    "detector_score",
    "acceptance_mode",
    "center_x",
    "center_y",
    "centroid_x",
    "centroid_y",
    "area_px",
    "area_um2",
    "equivalent_diameter_um",
    "perimeter_px",
    "perimeter_um",
    "major_axis_um",
    "minor_axis_um",
    "eccentricity",
    "solidity",
    "circularity",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "assigned_pixel_count",
    "overlap_pixel_count",
    "mask_path",
    "detection_pass",
    "segmentation_pass",
    "seed_source",
    "source_annotation_id",
    "failure_class",
    "manual_mask_choice",
    "shape_review_choice",
    "boundary_warning",
]
OVERVIEW_COLUMNS = [
    "sample_id",
    "oocyte_id",
    "label_id",
    "centroid_y",
    "centroid_x",
    "area_px",
    "area_um2",
]
PROVENANCE_COLUMNS = [
    "detection_pass",
    "segmentation_pass",
    "seed_source",
    "source_annotation_id",
    "failure_class",
    "manual_mask_choice",
    "shape_review_choice",
    "boundary_warning",
]


@dataclass(frozen=True)
class OocyteProfilingResult:
    sample_id: str
    oocyte_count: int
    channel_count: int
    output_dir: Path
    artifact_paths: Dict[str, Path]
    runtime_seconds: float


@dataclass(frozen=True)
class _ChannelRecord:
    source_row_index: int
    channel_index: int
    channel_id: str
    antibody_name: str
    exported_column_name: str
    measurement_class: str


@dataclass(frozen=True)
class _ObjectGeometry:
    label_id: int
    oocyte_id: str
    bbox: BoundingBox
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _ProfileRegion:
    bbox: BoundingBox


class _OmeZarrReader:
    """Context-managed bounded reader for one OME-TIFF series."""

    def __init__(self, path: Path, *, require_channels: bool) -> None:
        self.path = Path(path).resolve()
        self.require_channels = require_channels
        self._tif: tifffile.TiffFile | None = None
        self._store: Any = None
        self._array: Any = None
        self.axes = ""
        self.shape: Tuple[int, ...] = ()
        self.dtype = np.dtype(np.uint8)
        self.image_shape_yx: Tuple[int, int] = (0, 0)
        self.channel_count = 0
        self.storage: Dict[str, Any] = {}

    def __enter__(self) -> "_OmeZarrReader":
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._tif = tifffile.TiffFile(self.path)
        series = self._tif.series[0]
        self.axes = str(series.axes)
        self.shape = tuple(int(value) for value in series.shape)
        self.dtype = np.dtype(series.dtype)
        if self.axes.count("Y") != 1 or self.axes.count("X") != 1:
            self.close()
            raise ValueError(
                f"OME-TIFF must contain exactly one Y and X axis: {self.path} "
                f"has axes {self.axes!r}"
            )
        if self.axes.count("C") > 1:
            self.close()
            raise ValueError(f"OME-TIFF has multiple channel axes: {self.axes!r}")
        if self.require_channels and "C" not in self.axes:
            self.close()
            raise ValueError(f"raw OME-TIFF has no channel axis: {self.axes!r}")
        if not self.require_channels and "C" in self.axes:
            channel_size = self.shape[self.axes.index("C")]
            if channel_size != 1:
                self.close()
                raise ValueError(
                    f"label OME-TIFF must be two-dimensional, got {self.axes!r} "
                    f"with {channel_size} channels"
                )
        self.image_shape_yx = (
            self.shape[self.axes.index("Y")],
            self.shape[self.axes.index("X")],
        )
        self.channel_count = (
            self.shape[self.axes.index("C")] if "C" in self.axes else 0
        )
        page = series.pages[0]
        self.storage = {
            "is_tiled": bool(page.is_tiled),
            "tile_width": int(page.tilewidth or 0),
            "tile_height": int(page.tilelength or 0),
            "rows_per_strip": int(page.rowsperstrip or 0),
            "compression": str(page.compression.name),
        }
        self._store = series.aszarr()
        self._array = zarr.open(self._store, mode="r")
        return self

    def close(self) -> None:
        if self._store is not None:
            close = getattr(self._store, "close", None)
            if close is not None:
                close()
            self._store = None
        if self._tif is not None:
            self._tif.close()
            self._tif = None
        self._array = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def read_region(
        self,
        bbox: BoundingBox,
        *,
        channel_index: int | None = None,
    ) -> np.ndarray:
        if self._array is None:
            raise RuntimeError("OME reader is not open")
        image_h, image_w = self.image_shape_yx
        if bbox.x1 > image_w or bbox.y1 > image_h:
            raise ValueError("requested OME region exceeds image dimensions")
        resolved_channel_index = channel_index
        if "C" in self.axes:
            if not self.require_channels and resolved_channel_index is None:
                resolved_channel_index = 0
            if resolved_channel_index is None:
                raise ValueError("channel index is required for a channel image")
            if not 0 <= resolved_channel_index < self.channel_count:
                raise IndexError(
                    f"channel index {resolved_channel_index} outside "
                    f"[0, {self.channel_count})"
                )
        elif channel_index is not None:
            raise ValueError("channel index is invalid for a label image")

        indexer: list[Any] = []
        remaining_axes: list[str] = []
        for axis, axis_size in zip(self.axes, self.shape):
            if axis == "C":
                indexer.append(resolved_channel_index)
            elif axis == "Y":
                indexer.append(slice(bbox.y0, bbox.y1))
                remaining_axes.append(axis)
            elif axis == "X":
                indexer.append(slice(bbox.x0, bbox.x1))
                remaining_axes.append(axis)
            elif axis_size == 1:
                indexer.append(0)
            else:
                raise ValueError(
                    f"unsupported non-singleton OME axis {axis!r} with size "
                    f"{axis_size} in {self.path}"
                )

        region = np.asarray(self._array[tuple(indexer)])
        if set(remaining_axes) != {"Y", "X"} or region.ndim != 2:
            raise ValueError(
                f"OME region did not resolve to Y/X: axes={remaining_axes}, "
                f"shape={region.shape}"
            )
        if remaining_axes != ["Y", "X"]:
            region = np.transpose(
                region,
                (remaining_axes.index("Y"), remaining_axes.index("X")),
            )
        return region


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_csv(table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{destination.name}.",
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


def _atomic_write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _unique_channel_names(names: Sequence[str]) -> list[str]:
    counts: Dict[str, int] = {}
    used: set[str] = set()
    unique: list[str] = []
    for name in names:
        occurrence = counts.get(name, 0)
        candidate = name if occurrence == 0 else f"{name}_{occurrence}"
        while candidate in used:
            occurrence += 1
            candidate = f"{name}_{occurrence}"
        counts[name] = occurrence + 1
        used.add(candidate)
        unique.append(candidate)
    return unique


def _read_channels(antibodies_path: Path, channel_count: int) -> list[_ChannelRecord]:
    path = Path(antibodies_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"antibody table has no header: {path}")
        name_column = next(
            (
                name
                for name in ("antibody_name", "marker_name", "name")
                if name in reader.fieldnames
            ),
            None,
        )
        if name_column is None:
            raise ValueError(f"antibody table has no marker-name column: {path}")
        raw_rows = list(reader)

    if len(raw_rows) != channel_count:
        raise ValueError(
            f"antibody row count {len(raw_rows)} does not match raw image channel "
            f"count {channel_count}"
        )
    names = [str(row.get(name_column, "")).strip() for row in raw_rows]
    if any(not name for name in names):
        raise ValueError("antibody names must not be empty")
    exported_names = _unique_channel_names(names)
    if set(exported_names).intersection(MARKER_ID_COLUMNS):
        raise ValueError("antibody names conflict with profiling identifier columns")

    records: list[_ChannelRecord] = []
    for row_index, (row, name, exported_name) in enumerate(
        zip(raw_rows, names, exported_names)
    ):
        channel_id = str(row.get("channel_id", "")).strip()
        match = re.search(r":(\d+)$", channel_id)
        channel_index = int(match.group(1)) if match else row_index
        records.append(
            _ChannelRecord(
                source_row_index=row_index,
                channel_index=channel_index,
                channel_id=channel_id,
                antibody_name=name,
                exported_column_name=exported_name,
                measurement_class=(
                    "nuclear_stain"
                    if name.casefold() == "dapi"
                    else "protein_marker"
                ),
            )
        )
    indices = [record.channel_index for record in records]
    if sorted(indices) != list(range(channel_count)):
        raise ValueError(
            "antibody channel IDs must map one-to-one onto raw image channels "
            f"0..{channel_count - 1}"
        )
    return records


def _coerce_int_series(table: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    if values.isna().any() or not np.all(np.equal(values, np.floor(values))):
        raise ValueError(f"mapping column {column!r} must contain integers")
    return values.astype(np.int64)


def _read_mapping(mapping_path: Path) -> pd.DataFrame:
    path = Path(mapping_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping = pd.read_csv(path)
    required = {
        "label",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
    }
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"label mapping missing columns: {sorted(missing)}")
    if "oocyte_id" not in mapping.columns and "detector_component_id" not in mapping.columns:
        raise ValueError(
            "label mapping requires oocyte_id or detector_component_id"
        )
    for column in required:
        mapping[column] = _coerce_int_series(mapping, column)
    if (mapping["label"] <= 0).any():
        raise ValueError("mapping labels must be positive integers")
    if mapping["label"].duplicated().any():
        raise ValueError("mapping labels must be unique")
    return mapping.sort_values("label", kind="stable").reset_index(drop=True)


def _read_provenance(candidates_path: Path | None) -> Dict[str, Dict[str, Any]]:
    if candidates_path is None:
        return {}
    path = Path(candidates_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if "detector_component_id" not in table.columns:
        raise ValueError("candidate table has no detector_component_id column")
    ids = table["detector_component_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("candidate table detector_component_id values must be unique")
    rows: Dict[str, Dict[str, Any]] = {}
    for row in table.to_dict("records"):
        rows[str(row["detector_component_id"])] = row
    return rows


def _scan_label_counts(
    reader: _OmeZarrReader,
    *,
    strip_height_px: int,
) -> np.ndarray:
    if strip_height_px < 1:
        raise ValueError("label scan strip height must be positive")
    image_h, image_w = reader.image_shape_yx
    counts = np.zeros(1, dtype=np.int64)
    for y0 in range(0, image_h, strip_height_px):
        y1 = min(image_h, y0 + strip_height_px)
        region = reader.read_region(BoundingBox(0, y0, image_w, y1))
        if not np.issubdtype(region.dtype, np.integer):
            raise ValueError(f"label image must use an integer dtype, got {region.dtype}")
        if np.issubdtype(region.dtype, np.signedinteger) and int(region.min()) < 0:
            raise ValueError("label image contains negative values")
        maximum = int(region.max())
        if maximum > MAX_LABEL_VALUE:
            raise ValueError(
                f"label value {maximum} exceeds uint16 release limit {MAX_LABEL_VALUE}"
            )
        chunk_counts = np.bincount(region.ravel(), minlength=maximum + 1)
        if len(chunk_counts) > len(counts):
            counts = np.pad(counts, (0, len(chunk_counts) - len(counts)))
        counts[: len(chunk_counts)] += chunk_counts.astype(np.int64, copy=False)
    return counts


def _optional_value(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return default
    return value


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0", ""}:
        return False
    raise ValueError(f"invalid boolean value for {field_name}: {value!r}")


def _make_oocyte_id(sample_id: str, row: Mapping[str, Any]) -> str:
    supplied = str(_optional_value(row, "oocyte_id", "")).strip()
    if supplied:
        return supplied
    component = str(_optional_value(row, "detector_component_id", "")).strip()
    if not component:
        raise ValueError("mapping row has no stable component identifier")
    return f"{sample_id}__{component}"


def _extract_object_geometries(
    reader: _OmeZarrReader,
    mapping: pd.DataFrame,
    label_counts: np.ndarray,
    *,
    sample_id: str,
    pixel_size_um: float,
    provenance: Mapping[str, Mapping[str, Any]],
) -> list[_ObjectGeometry]:
    image_h, image_w = reader.image_shape_yx
    geometries: list[_ObjectGeometry] = []
    seen_oocyte_ids: set[str] = set()
    for row in mapping.to_dict("records"):
        label_id = int(row["label"])
        mapping_bbox = BoundingBox(
            int(row["bbox_x0"]),
            int(row["bbox_y0"]),
            int(row["bbox_x1"]),
            int(row["bbox_y1"]),
        )
        if mapping_bbox.x1 > image_w or mapping_bbox.y1 > image_h:
            raise ValueError(f"mapping bbox exceeds image dimensions for label {label_id}")
        patch = reader.read_region(mapping_bbox)
        mask = patch == label_id
        expected_count = int(label_counts[label_id])
        if expected_count <= 0:
            raise ValueError(f"mapping label {label_id} is absent from label image")
        if int(mask.sum()) != expected_count:
            raise ValueError(
                f"label {label_id} has pixels outside its mapping bbox: "
                f"bbox count {int(mask.sum())}, whole-image count {expected_count}"
            )
        local_y, local_x = np.nonzero(mask)
        local_y0 = int(local_y.min())
        local_y1 = int(local_y.max()) + 1
        local_x0 = int(local_x.min())
        local_x1 = int(local_x.max()) + 1
        tight_mask = mask[local_y0:local_y1, local_x0:local_x1]
        bbox = BoundingBox(
            mapping_bbox.x0 + local_x0,
            mapping_bbox.y0 + local_y0,
            mapping_bbox.x0 + local_x1,
            mapping_bbox.y0 + local_y1,
        )
        props = regionprops(tight_mask.astype(np.uint8, copy=False))
        if len(props) != 1:
            raise ValueError(f"label {label_id} did not produce one morphology region")
        prop = props[0]
        area_px = int(prop.area)
        perimeter_px = float(prop.perimeter)
        circularity = (
            float(4.0 * math.pi * area_px / (perimeter_px * perimeter_px))
            if perimeter_px > 0
            else 0.0
        )
        oocyte_id = _make_oocyte_id(sample_id, row)
        if oocyte_id in seen_oocyte_ids:
            raise ValueError(f"duplicate stable oocyte_id: {oocyte_id}")
        seen_oocyte_ids.add(oocyte_id)

        component_id = str(
            _optional_value(row, "detector_component_id", oocyte_id)
        )
        candidate = provenance.get(component_id, {})
        metadata: Dict[str, Any] = {
            "sample_id": sample_id,
            "oocyte_id": oocyte_id,
            "label_id": label_id,
            "detector_component_id": component_id,
            "detector_score": float(_optional_value(row, "detector_score", np.nan)),
            "acceptance_mode": str(_optional_value(row, "acceptance_mode", "")),
            "center_x": _optional_value(row, "center_x", np.nan),
            "center_y": _optional_value(row, "center_y", np.nan),
            "centroid_x": float(bbox.x0 + prop.centroid[1]),
            "centroid_y": float(bbox.y0 + prop.centroid[0]),
            "area_px": area_px,
            "area_um2": float(area_px * pixel_size_um * pixel_size_um),
            "equivalent_diameter_um": float(
                prop.equivalent_diameter_area * pixel_size_um
            ),
            "perimeter_px": perimeter_px,
            "perimeter_um": float(perimeter_px * pixel_size_um),
            "major_axis_um": float(prop.axis_major_length * pixel_size_um),
            "minor_axis_um": float(prop.axis_minor_length * pixel_size_um),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
            "circularity": circularity,
            "bbox_x0": bbox.x0,
            "bbox_y0": bbox.y0,
            "bbox_x1": bbox.x1,
            "bbox_y1": bbox.y1,
            "assigned_pixel_count": int(
                _optional_value(row, "assigned_pixel_count", expected_count)
            ),
            "overlap_pixel_count": int(
                _optional_value(row, "overlap_pixel_count", 0)
            ),
            "mask_path": str(_optional_value(row, "mask_path", "")),
        }
        if metadata["assigned_pixel_count"] != expected_count:
            raise ValueError(
                f"mapping assigned_pixel_count mismatch for label {label_id}: "
                f"{metadata['assigned_pixel_count']} != {expected_count}"
            )
        for column in PROVENANCE_COLUMNS:
            source = row if column in row and not pd.isna(row[column]) else candidate
            default = False if column == "boundary_warning" else ""
            value = _optional_value(source, column, default)
            if column == "boundary_warning":
                value = _coerce_bool(value, field_name=column)
            metadata[column] = value
        geometries.append(
            _ObjectGeometry(
                label_id=label_id,
                oocyte_id=oocyte_id,
                bbox=bbox,
                metadata=metadata,
            )
        )
    return geometries


def _build_profile_regions(
    geometries: Sequence[_ObjectGeometry],
    *,
    max_height_px: int,
    merge_gap_px: int,
) -> list[_ProfileRegion]:
    if max_height_px < 1:
        raise ValueError("profile region height must be positive")
    if merge_gap_px < 0:
        raise ValueError("profile region merge gap must be non-negative")
    if not geometries:
        return []

    intervals = sorted((item.bbox.y0, item.bbox.y1) for item in geometries)
    merged: list[Tuple[int, int]] = []
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end + merge_gap_px:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))

    regions: list[_ProfileRegion] = []
    for merged_y0, merged_y1 in merged:
        for y0 in range(merged_y0, merged_y1, max_height_px):
            y1 = min(merged_y1, y0 + max_height_px)
            intersecting = [
                item.bbox
                for item in geometries
                if item.bbox.y0 < y1 and item.bbox.y1 > y0
            ]
            if not intersecting:
                continue
            regions.append(
                _ProfileRegion(
                    bbox=BoundingBox(
                        min(bbox.x0 for bbox in intersecting),
                        y0,
                        max(bbox.x1 for bbox in intersecting),
                        y1,
                    )
                )
            )
    return regions


def _profile_marker_means(
    source: _OmeZarrReader,
    labels: _OmeZarrReader,
    channels: Sequence[_ChannelRecord],
    geometries: Sequence[_ObjectGeometry],
    label_counts: np.ndarray,
    *,
    max_region_height_px: int,
    merge_gap_px: int,
) -> tuple[np.ndarray, list[_ProfileRegion]]:
    regions = _build_profile_regions(
        geometries,
        max_height_px=max_region_height_px,
        merge_gap_px=merge_gap_px,
    )
    if not geometries:
        return np.empty((0, len(channels)), dtype=np.float64), regions

    maximum_label = max(item.label_id for item in geometries)
    sums = np.zeros((maximum_label + 1, len(channels)), dtype=np.float64)
    covered_counts = np.zeros(maximum_label + 1, dtype=np.int64)
    for region_index, region in enumerate(regions, start=1):
        label_patch = labels.read_region(region.bbox)
        patch_counts = np.bincount(label_patch.ravel(), minlength=maximum_label + 1)
        covered_counts += patch_counts[: maximum_label + 1].astype(
            np.int64, copy=False
        )
        logger.info(
            "Profiling region %d/%d: bbox=%s, shape=%s",
            region_index,
            len(regions),
            region.bbox.as_tuple(),
            region.bbox.shape_yx,
        )
        for output_index, channel in enumerate(channels):
            raw_patch = source.read_region(
                region.bbox,
                channel_index=channel.channel_index,
            )
            channel_sums = np.bincount(
                label_patch.ravel(),
                weights=raw_patch.ravel(),
                minlength=maximum_label + 1,
            )
            sums[:, output_index] += channel_sums[: maximum_label + 1]

    ordered_labels = np.asarray([item.label_id for item in geometries], dtype=np.int64)
    expected_counts = label_counts[ordered_labels]
    if not np.array_equal(covered_counts[ordered_labels], expected_counts):
        raise RuntimeError("profile regions did not cover every final label pixel")
    means = sums[ordered_labels] / expected_counts[:, None]
    return means, regions


def _path_identity(path: Path, *, include_sha256: bool) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    identity: Dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": str(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = _file_sha256(resolved)
    return identity


def profile_oocyte_labels(
    *,
    sample_id: str,
    image_path: Path,
    antibodies_path: Path,
    label_path: Path,
    mapping_path: Path,
    out_dir: Path,
    pixel_size_um: float,
    candidates_path: Path | None = None,
    max_region_height_px: int = 512,
    merge_gap_px: int = 16,
    label_scan_height_px: int = 1024,
) -> OocyteProfilingResult:
    """Write one raw marker-intensity row per final oocyte label.

    The final label image, not a nucleus mask or detector candidate table, defines
    which pixels belong to each profiled object. The optional candidate table only
    enriches provenance fields and cannot add or remove labels.
    """

    started = time.perf_counter()
    sample = str(sample_id).strip()
    if not sample:
        raise ValueError("sample_id must not be empty")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive and finite")

    raw_path = Path(image_path).resolve()
    antibodies = Path(antibodies_path).resolve()
    labels_path = Path(label_path).resolve()
    mapping_csv = Path(mapping_path).resolve()
    candidate_csv = Path(candidates_path).resolve() if candidates_path else None
    mapping = _read_mapping(mapping_csv)
    provenance = _read_provenance(candidate_csv)

    with _OmeZarrReader(raw_path, require_channels=True) as source, _OmeZarrReader(
        labels_path, require_channels=False
    ) as labels:
        if source.image_shape_yx != labels.image_shape_yx:
            raise ValueError(
                "raw image and final label dimensions differ: "
                f"{source.image_shape_yx} != {labels.image_shape_yx}"
            )
        channels = _read_channels(antibodies, source.channel_count)
        label_counts = _scan_label_counts(
            labels,
            strip_height_px=label_scan_height_px,
        )
        observed_labels = set(np.flatnonzero(label_counts[1:]) + 1)
        mapped_labels = set(int(value) for value in mapping["label"].tolist())
        if observed_labels != mapped_labels:
            missing_from_mapping = sorted(observed_labels - mapped_labels)
            missing_from_image = sorted(mapped_labels - observed_labels)
            raise ValueError(
                "label image and mapping label sets differ; "
                f"unmapped image labels={missing_from_mapping}, "
                f"mapping labels absent from image={missing_from_image}"
            )
        geometries = _extract_object_geometries(
            labels,
            mapping,
            label_counts,
            sample_id=sample,
            pixel_size_um=float(pixel_size_um),
            provenance=provenance,
        )
        means, profile_regions = _profile_marker_means(
            source,
            labels,
            channels,
            geometries,
            label_counts,
            max_region_height_px=max_region_height_px,
            merge_gap_px=merge_gap_px,
        )
        source_series = {
            "axes": source.axes,
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "storage": source.storage,
        }
        label_series = {
            "axes": labels.axes,
            "shape": list(labels.shape),
            "dtype": str(labels.dtype),
            "storage": labels.storage,
        }

    channel_names = [record.exported_column_name for record in channels]
    marker_rows = []
    for object_index, geometry in enumerate(geometries):
        row: Dict[str, Any] = {
            "sample_id": sample,
            "oocyte_id": geometry.oocyte_id,
            "label_id": geometry.label_id,
        }
        row.update(
            {
                channel_name: float(means[object_index, channel_index])
                for channel_index, channel_name in enumerate(channel_names)
            }
        )
        marker_rows.append(row)
    marker_table = pd.DataFrame(
        marker_rows,
        columns=MARKER_ID_COLUMNS + channel_names,
    )
    metadata_table = pd.DataFrame(
        [geometry.metadata for geometry in geometries],
        columns=METADATA_COLUMNS,
    )
    overview_table = metadata_table[OVERVIEW_COLUMNS].merge(
        marker_table,
        on=MARKER_ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    overview_table = overview_table[
        OVERVIEW_COLUMNS + channel_names
    ]
    channel_table = pd.DataFrame(
        [
            {
                "source_row_index": record.source_row_index,
                "channel_index": record.channel_index,
                "channel_id": record.channel_id,
                "antibody_name": record.antibody_name,
                "exported_column_name": record.exported_column_name,
                "measurement_class": record.measurement_class,
                "included": True,
            }
            for record in channels
        ]
    )

    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "markers": destination / "oocyte_by_marker.csv",
        "metadata": destination / "oocyte_metadata.csv",
        "overview": destination / "oocyte_overview.csv",
        "channels": destination / "channel_manifest.csv",
        "manifest": destination / "profiling_manifest.json",
    }
    paths["manifest"].unlink(missing_ok=True)
    _atomic_write_csv(marker_table, paths["markers"])
    _atomic_write_csv(metadata_table, paths["metadata"])
    _atomic_write_csv(overview_table, paths["overview"])
    _atomic_write_csv(channel_table, paths["channels"])

    runtime_seconds = time.perf_counter() - started
    profile_pixel_count = int(sum(region.bbox.width * region.bbox.height for region in profile_regions))
    manifest = {
        "schema_version": 1,
        "implementation_version": OOCYTE_PROFILING_VERSION,
        "sample_id": sample,
        "pixel_size_um": float(pixel_size_um),
        "oocyte_count": len(geometries),
        "channel_count": len(channels),
        "runtime_seconds": runtime_seconds,
        "source_image": {
            **_path_identity(raw_path, include_sha256=False),
            **source_series,
        },
        "label_image": {
            **_path_identity(labels_path, include_sha256=True),
            **label_series,
        },
        "mapping": _path_identity(mapping_csv, include_sha256=True),
        "antibodies": _path_identity(antibodies, include_sha256=True),
        "candidates": (
            _path_identity(candidate_csv, include_sha256=True)
            if candidate_csv is not None
            else None
        ),
        "measurement": {
            "statistic": "raw_within_mask_mean",
            "background_subtracted": False,
            "transformed": False,
            "normalized": False,
            "object_source": "final_label_image",
        },
        "bounded_read_plan": {
            "region_count": len(profile_regions),
            "max_region_height_px": int(max_region_height_px),
            "merge_gap_px": int(merge_gap_px),
            "label_scan_height_px": int(label_scan_height_px),
            "profile_region_pixel_count_per_channel": profile_pixel_count,
            "profile_region_channel_pixel_count": profile_pixel_count * len(channels),
            "largest_region_shape_yx": (
                list(
                    max(
                        (region.bbox.shape_yx for region in profile_regions),
                        key=lambda shape: shape[0] * shape[1],
                    )
                )
                if profile_regions
                else [0, 0]
            ),
        },
        "artifacts": {
            path.name: {
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    _atomic_write_json(manifest, paths["manifest"])
    logger.info(
        "Oocyte profiling complete: sample=%s, oocytes=%d, channels=%d, runtime=%.1fs",
        sample,
        len(geometries),
        len(channels),
        runtime_seconds,
    )
    return OocyteProfilingResult(
        sample_id=sample,
        oocyte_count=len(geometries),
        channel_count=len(channels),
        output_dir=destination,
        artifact_paths=paths,
        runtime_seconds=runtime_seconds,
    )


__all__ = [
    "OOCYTE_PROFILING_VERSION",
    "OocyteProfilingResult",
    "profile_oocyte_labels",
]
