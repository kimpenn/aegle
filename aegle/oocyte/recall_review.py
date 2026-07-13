"""Coverage-based recall review for standalone raw-UCHL1 oocyte detection."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import tempfile
import threading
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tifffile
import zarr
from matplotlib import colormaps
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation

from .config import DONOR13_V6
from .io import load_candidate_mask
from .models import BoundingBox, ExtractedPatch, LocalSegmentationResult, PersistedMask
from .recall_overlay import load_recall_mask_overlay, overlay_dir_from_identity
from .recall_review_page import recall_review_page_html, review_console_page_html
from .segmentation import _segment_oocyte_patch_components


LOGGER = logging.getLogger(__name__)
RECALL_REVIEW_SCHEMA_VERSION = 1
RECALL_COVERAGE_IDENTITY_VERSION = 1
SURVEY_COVERAGE_PROFILE = "survey_v1"
DETAIL_COVERAGE_PROFILE = "detail_v1"
SURVEY_WINDOW_RADIUS_PX = 1280
SURVEY_WINDOW_STRIDE_PX = 2304
DETAIL_WINDOW_RADIUS_PX = 512
DETAIL_WINDOW_STRIDE_PX = 768
DEFAULT_WINDOW_RADIUS_PX = SURVEY_WINDOW_RADIUS_PX
DEFAULT_WINDOW_STRIDE_PX = SURVEY_WINDOW_STRIDE_PX
DEFAULT_OVERVIEW_DOWNSAMPLE = 16
MIN_RECALL_WINDOW_RADIUS_PX = 128
MAX_RECALL_WINDOW_RADIUS_PX = SURVEY_WINDOW_RADIUS_PX


@dataclass(frozen=True)
class RecallReviewBundle:
    sample_id: str
    sample_dir: Path
    page_path: Path
    overview_path: Path
    metadata_path: Path
    console_path: Path
    window_count: int


@dataclass(frozen=True)
class RecallReviewSample:
    sample_id: str
    sample_dir: Path
    source_image: Path
    source_image_size_bytes: int
    channel_index: int
    image_shape_yx: Tuple[int, int]
    pixel_size_um: float
    profile_name: str
    profile_fingerprint: str
    implementation_version: str
    candidates: pd.DataFrame
    detector_candidates: pd.DataFrame
    refined_candidates: pd.DataFrame
    coarse_candidates: pd.DataFrame
    suppressed_candidates: pd.DataFrame
    overlay_delivery_dir: Path | None
    review_identity: Dict[str, Any]


@dataclass(frozen=True)
class ProbeSegmentation:
    patch: ExtractedPatch
    p99: LocalSegmentationResult | None
    p95: LocalSegmentationResult | None
    p99_error: str | None
    p95_error: str | None


@dataclass(frozen=True)
class ManualSeedSegmentation:
    patch: ExtractedPatch
    conservative: LocalSegmentationResult | None
    expanded: LocalSegmentationResult | None
    conservative_percentile: float | None
    expanded_percentile: float | None
    error: str | None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value if isinstance(value, str) else str(value)


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
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
            handle.flush()
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_image(path: Path, image: Image.Image, *, format_name: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=f".{format_name.lower()}",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        image.save(temporary_path, format=format_name, quality=90, method=6)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_path(sample_dir: Path, row: Mapping[str, Any]) -> Path:
    raw_mask_path = Path(str(row["mask_path"]))
    if raw_mask_path.is_absolute():
        return raw_mask_path
    source_value = row.get("mask_source_dir")
    if source_value is None or pd.isna(source_value) or not str(source_value).strip():
        source_dir = sample_dir
    else:
        source_dir = Path(str(source_value))
    return source_dir / raw_mask_path


def _load_sample(
    sample_dir: Path,
    *,
    overlay_dir: Path | None = None,
) -> RecallReviewSample:
    root = Path(sample_dir).resolve()
    manifest_path = root / "run_manifest.json"
    summary_path = root / "summary.json"
    candidates_path = root / "html_candidates.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("sample directory must contain run_manifest.json and summary.json")
    if not candidates_path.is_file():
        raise FileNotFoundError(
            f"missing {candidates_path}; generate the combined HTML report before recall review"
        )
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    sample_id = str(manifest["sample_id"])
    source_image = Path(str(manifest["source_image"])).resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"raw source image does not exist: {source_image}")
    image_shape = tuple(int(value) for value in summary["image_shape_yx"])
    if len(image_shape) != 2:
        raise ValueError("summary image_shape_yx must contain Y and X")
    detector_candidates = pd.read_csv(candidates_path)
    required = {
        "detector_component_id",
        "center_x",
        "center_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "mask_path",
    }
    missing = required.difference(detector_candidates.columns)
    if missing:
        raise ValueError(f"html_candidates.csv missing columns: {sorted(missing)}")
    if "mask_source_dir" not in detector_candidates:
        detector_candidates["mask_source_dir"] = str(root)
    if "display_id" not in detector_candidates:
        detector_candidates["display_id"] = [
            f"#{index:03d}" for index in range(1, len(detector_candidates) + 1)
        ]
    if "detection_pass" not in detector_candidates:
        detector_candidates["detection_pass"] = "baseline_v6"
    if "detector_score" not in detector_candidates:
        detector_candidates["detector_score"] = 0.0
    for row in detector_candidates.to_dict("records"):
        path = _mask_path(root, row)
        if not path.is_file():
            raise FileNotFoundError(f"candidate mask does not exist: {path}")

    refined_path = root / "candidates.csv"
    coarse_path = root / "coarse_candidates.csv"
    refined = pd.read_csv(refined_path) if refined_path.is_file() else pd.DataFrame()
    coarse = pd.read_csv(coarse_path) if coarse_path.is_file() else pd.DataFrame()
    suppressed_path = root / "combined_duplicate_suppressed.csv"
    suppressed = pd.read_csv(suppressed_path) if suppressed_path.is_file() else pd.DataFrame()
    if not suppressed.empty and "detector_component_id" in suppressed and not refined.empty:
        suppressed = suppressed.merge(
            refined,
            on="detector_component_id",
            how="left",
            suffixes=("", "_refined"),
        )

    source_stat = source_image.stat()
    candidate_sha = _file_sha256(candidates_path)
    detector_identity = {
        "sample_id": sample_id,
        "source_image": str(source_image),
        "source_image_size_bytes": int(source_stat.st_size),
        # Nanosecond timestamps exceed JavaScript's exact integer range.
        "source_image_mtime_ns": str(source_stat.st_mtime_ns),
        "profile_name": str(manifest["profile_name"] if "profile_name" in manifest else summary["profile_name"]),
        "profile_fingerprint": str(manifest["profile_fingerprint"]),
        "implementation_version": str(manifest["implementation_version"]),
        "candidate_table_sha256": candidate_sha,
        "combined_candidate_count": int(len(detector_candidates)),
    }
    overlay = load_recall_mask_overlay(
        detector_candidates,
        sample_identity=detector_identity,
        image_shape_yx=(image_shape[0], image_shape[1]),
        overlay_dir=overlay_dir,
    )
    review_identity = {**detector_identity, **overlay.identity_fields}
    resolved_config = manifest.get("resolved_config", {})
    pixel_size_um = float(resolved_config.get("pixel_size_um", 0.5))
    return RecallReviewSample(
        sample_id=sample_id,
        sample_dir=root,
        source_image=source_image,
        source_image_size_bytes=int(source_stat.st_size),
        channel_index=int(manifest["resolved_channel_index"]),
        image_shape_yx=(image_shape[0], image_shape[1]),
        pixel_size_um=pixel_size_um,
        profile_name=str(review_identity["profile_name"]),
        profile_fingerprint=str(review_identity["profile_fingerprint"]),
        implementation_version=str(review_identity["implementation_version"]),
        candidates=overlay.candidates,
        detector_candidates=detector_candidates,
        refined_candidates=refined,
        coarse_candidates=coarse,
        suppressed_candidates=suppressed,
        overlay_delivery_dir=overlay.delivery_dir,
        review_identity=review_identity,
    )


class _OmeChannelSource:
    """Keep one tifffile-backed zarr source open for bounded channel reads."""

    def __init__(self, path: Path, channel_index: int):
        self.path = Path(path)
        self.channel_index = int(channel_index)
        self._tif = tifffile.TiffFile(self.path)
        self._series = self._tif.series[0]
        self.axes = self._series.axes
        self.shape = tuple(int(value) for value in self._series.shape)
        if "C" not in self.axes or "Y" not in self.axes or "X" not in self.axes:
            self.close()
            raise ValueError(f"expected C/Y/X axes in OME-TIFF, got {self.axes!r}")
        channel_count = self.shape[self.axes.index("C")]
        if not 0 <= self.channel_index < channel_count:
            self.close()
            raise IndexError(f"channel {self.channel_index} outside [0, {channel_count})")
        for axis, size in zip(self.axes, self.shape):
            if axis not in {"C", "Y", "X"} and size != 1:
                self.close()
                raise ValueError(f"unsupported non-singleton OME axis {axis!r}: {size}")
        self.image_shape_yx = (
            self.shape[self.axes.index("Y")],
            self.shape[self.axes.index("X")],
        )
        self._store = self._series.aszarr()
        self._array = zarr.open(self._store, mode="r")
        self._lock = threading.Lock()

    def close(self) -> None:
        tif = getattr(self, "_tif", None)
        if tif is not None:
            tif.close()
            self._tif = None

    def __enter__(self) -> "_OmeChannelSource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _indexer(self, y_slice: slice, x_slice: slice) -> Tuple[Any, ...]:
        output = []
        for axis, size in zip(self.axes, self.shape):
            if axis == "C":
                output.append(self.channel_index)
            elif axis == "Y":
                output.append(y_slice)
            elif axis == "X":
                output.append(x_slice)
            elif size == 1:
                output.append(0)
            else:
                raise ValueError(f"unsupported OME axis {axis!r}")
        return tuple(output)

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        image_h, image_w = self.image_shape_yx
        if not (0 <= y0 < y1 <= image_h and 0 <= x0 < x1 <= image_w):
            raise ValueError("region must be a positive image-bounded rectangle")
        with self._lock:
            region = np.asarray(self._array[self._indexer(slice(y0, y1), slice(x0, x1))])
        if region.ndim != 2:
            raise ValueError(f"channel region did not resolve to Y/X: {region.shape}")
        return region

    def read_patch(self, center_xy: Tuple[int, int], radius: int) -> ExtractedPatch:
        if radius < 1:
            raise ValueError("patch radius must be positive")
        image_h, image_w = self.image_shape_yx
        center_x, center_y = (int(center_xy[0]), int(center_xy[1]))
        if not 0 <= center_x < image_w or not 0 <= center_y < image_h:
            raise ValueError("patch center must be inside the source image")
        requested_x0 = center_x - radius
        requested_y0 = center_y - radius
        requested_x1 = center_x + radius + 1
        requested_y1 = center_y + radius + 1
        x0, y0 = max(0, requested_x0), max(0, requested_y0)
        x1, y1 = min(image_w, requested_x1), min(image_h, requested_y1)
        bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
        padding = (
            y0 - requested_y0,
            requested_y1 - y1,
            x0 - requested_x0,
            requested_x1 - x1,
        )
        patch = self.read_region(y0, y1, x0, x1)
        if any(padding):
            top, bottom, left, right = padding
            patch = np.pad(patch, ((top, bottom), (left, right)), mode="edge")
        return ExtractedPatch(
            image=np.asarray(patch),
            bbox=bbox,
            image_shape_yx=self.image_shape_yx,
            padding_tblr=padding,
        )


def _reduce_strip(strip: np.ndarray, factor: int) -> np.ndarray:
    pad_h = (-strip.shape[0]) % factor
    pad_w = (-strip.shape[1]) % factor
    if pad_h or pad_w:
        strip = np.pad(strip, ((0, pad_h), (0, pad_w)), mode="edge")
    return strip.reshape(
        strip.shape[0] // factor,
        factor,
        strip.shape[1] // factor,
        factor,
    ).mean(axis=(1, 3))


def _build_overview(
    source: _OmeChannelSource,
    *,
    downsample: int,
    strip_height: int = 1024,
) -> Tuple[np.ndarray, float, float]:
    if downsample < 1:
        raise ValueError("overview downsample must be positive")
    image_h, image_w = source.image_shape_yx
    output = np.zeros(
        (
            math.ceil(image_h / downsample),
            math.ceil(image_w / downsample),
        ),
        dtype=np.float32,
    )
    out_y = 0
    for y0 in range(0, image_h, strip_height):
        y1 = min(image_h, y0 + strip_height)
        strip = source.read_region(y0, y1, 0, image_w).astype(np.float32, copy=False)
        reduced = _reduce_strip(strip, downsample)
        output[out_y : out_y + reduced.shape[0], : reduced.shape[1]] = reduced
        out_y += reduced.shape[0]
    transformed = np.log1p(np.maximum(output, 0.0))
    finite = transformed[np.isfinite(transformed)]
    if finite.size:
        low, high = (float(value) for value in np.percentile(finite, [1.0, 99.9]))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    return output, low, high


def _magma_image(array: np.ndarray, low: float, high: float) -> Image.Image:
    transformed = np.log1p(np.maximum(np.asarray(array, dtype=np.float32), 0.0))
    normalized = np.clip((transformed - low) / max(high - low, 1e-6), 0.0, 1.0)
    rgb = np.asarray(colormaps["magma"](normalized)[..., :3] * 255.0, dtype=np.uint8)
    return Image.fromarray(rgb)


def _axis_centers(length: int, radius: int, stride: int) -> Sequence[int]:
    if length < 1 or radius < 1 or stride < 1:
        raise ValueError("axis length, radius, and stride must be positive")
    if length <= 2 * radius + 1:
        return [length // 2]
    first = radius
    last = length - radius - 1
    centers = list(range(first, last + 1, stride))
    if centers[-1] != last:
        centers.append(last)
    return centers


def _coverage_profile_name(
    window_radius_px: int,
    window_stride_px: int,
    overview_downsample: int,
) -> str:
    geometry = (window_radius_px, window_stride_px, overview_downsample)
    if geometry == (
        SURVEY_WINDOW_RADIUS_PX,
        SURVEY_WINDOW_STRIDE_PX,
        DEFAULT_OVERVIEW_DOWNSAMPLE,
    ):
        return SURVEY_COVERAGE_PROFILE
    if geometry == (
        DETAIL_WINDOW_RADIUS_PX,
        DETAIL_WINDOW_STRIDE_PX,
        DEFAULT_OVERVIEW_DOWNSAMPLE,
    ):
        return DETAIL_COVERAGE_PROFILE
    return "custom"


def _coverage_geometry_sha256(windows: Sequence[Mapping[str, Any]]) -> str:
    """Hash spatial coverage only, independent of risk-based review ordering."""

    geometry = []
    for row in sorted(
        windows,
        key=lambda item: (int(item["spatial_row"]), int(item["spatial_column"])),
    ):
        bbox = row["bbox"]
        geometry.append(
            {
                "bbox": {
                    "x0": int(bbox["x0"]),
                    "x1": int(bbox["x1"]),
                    "y0": int(bbox["y0"]),
                    "y1": int(bbox["y1"]),
                },
                "center_x": int(row["center_x"]),
                "center_y": int(row["center_y"]),
                "spatial_column": int(row["spatial_column"]),
                "spatial_row": int(row["spatial_row"]),
                "window_id": str(row["window_id"]),
            }
        )
    canonical = json.dumps(
        geometry,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _coverage_identity(
    windows: Sequence[Mapping[str, Any]],
    *,
    window_radius_px: int,
    window_stride_px: int,
    overview_downsample: int,
) -> Dict[str, Any]:
    return {
        "recall_coverage_identity_version": RECALL_COVERAGE_IDENTITY_VERSION,
        "recall_coverage_profile": _coverage_profile_name(
            window_radius_px,
            window_stride_px,
            overview_downsample,
        ),
        "recall_window_radius_px": int(window_radius_px),
        "recall_window_stride_px": int(window_stride_px),
        "recall_overview_downsample": int(overview_downsample),
        "recall_window_count": len(windows),
        "recall_window_geometry_sha256": _coverage_geometry_sha256(windows),
    }


def _identity_contains_required(
    actual: Mapping[str, Any],
    required: Mapping[str, Any],
) -> bool:
    return all(actual.get(field) == expected for field, expected in required.items())


def _validate_bundle_coverage_identity(metadata: Mapping[str, Any]) -> None:
    identity = metadata.get("review_identity")
    windows = metadata.get("windows")
    if not isinstance(identity, Mapping) or not isinstance(windows, list):
        return
    geometry_fields = {
        "recall_coverage_identity_version",
        "recall_coverage_profile",
        "recall_window_radius_px",
        "recall_window_stride_px",
        "recall_overview_downsample",
        "recall_window_count",
        "recall_window_geometry_sha256",
    }
    present_fields = geometry_fields.intersection(identity)
    if not present_fields:
        return
    if present_fields != geometry_fields:
        missing = sorted(geometry_fields.difference(identity))
        raise ValueError(f"recall review bundle coverage identity is incomplete: {missing}")
    expected = _coverage_identity(
        windows,
        window_radius_px=int(metadata["window_radius_px"]),
        window_stride_px=int(metadata["window_stride_px"]),
        overview_downsample=int(metadata["overview_downsample"]),
    )
    if not _identity_contains_required(identity, expected):
        raise ValueError("recall review bundle coverage identity does not match its grid")


def _points_in_box(table: pd.DataFrame, x_column: str, y_column: str, bbox: BoundingBox) -> int:
    if table.empty or x_column not in table or y_column not in table:
        return 0
    x = pd.to_numeric(table[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(table[y_column], errors="coerce").to_numpy(dtype=float)
    return int(((x >= bbox.x0) & (x < bbox.x1) & (y >= bbox.y0) & (y < bbox.y1)).sum())


def _browser_candidate_rows(candidates: pd.DataFrame) -> Sequence[Dict[str, Any]]:
    output = []
    for row in candidates.to_dict("records"):
        output.append(
            {
                "detector_component_id": str(row["detector_component_id"]),
                "display_id": str(row.get("display_id", row["detector_component_id"])),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "bbox": {
                    "x0": int(round(float(row["bbox_x0"]))),
                    "y0": int(round(float(row["bbox_y0"]))),
                    "x1": int(round(float(row["bbox_x1"]))),
                    "y1": int(round(float(row["bbox_y1"]))),
                },
                "detector_score": float(row.get("detector_score", 0.0)),
                "detection_pass": str(row.get("detection_pass", "baseline_v6")),
                "resolution_source": str(
                    row.get(
                        "precision_resolution_source",
                        row.get("acceptance_mode", "automatic_candidate"),
                    )
                ),
            }
        )
    return output


def _coverage_windows(
    sample: RecallReviewSample,
    overview: np.ndarray,
    *,
    window_radius_px: int,
    window_stride_px: int,
    overview_downsample: int,
) -> Sequence[Dict[str, Any]]:
    image_h, image_w = sample.image_shape_yx
    accepted = sample.candidates
    refined = sample.refined_candidates
    if not refined.empty and "accepted" in refined:
        accepted_flag = refined["accepted"].astype(str).str.lower().isin({"true", "1"})
        score = pd.to_numeric(refined.get("detector_score", 0.0), errors="coerce").fillna(0.0)
        near_rejected = refined[(~accepted_flag) & (score >= 0.30)]
    else:
        near_rejected = pd.DataFrame()
    spatial_rows = []
    for row_index, center_y in enumerate(_axis_centers(image_h, window_radius_px, window_stride_px)):
        for column_index, center_x in enumerate(_axis_centers(image_w, window_radius_px, window_stride_px)):
            bbox = BoundingBox(
                x0=max(0, center_x - window_radius_px),
                y0=max(0, center_y - window_radius_px),
                x1=min(image_w, center_x + window_radius_px + 1),
                y1=min(image_h, center_y + window_radius_px + 1),
            )
            accepted_count = _points_in_box(accepted, "center_x", "center_y", bbox)
            near_rejected_count = _points_in_box(
                near_rejected, "center_x", "center_y", bbox
            )
            coarse_count = _points_in_box(
                sample.coarse_candidates,
                "coarse_center_x",
                "coarse_center_y",
                bbox,
            )
            overview_x0 = max(0, bbox.x0 // overview_downsample)
            overview_y0 = max(0, bbox.y0 // overview_downsample)
            overview_x1 = min(overview.shape[1], math.ceil(bbox.x1 / overview_downsample))
            overview_y1 = min(overview.shape[0], math.ceil(bbox.y1 / overview_downsample))
            overview_crop = overview[overview_y0:overview_y1, overview_x0:overview_x1]
            mean_signal = float(np.mean(np.log1p(np.maximum(overview_crop, 0.0))))
            risk_score = (
                5.0 * near_rejected_count
                + 1.25 * coarse_count
                + 0.5 * accepted_count
                + mean_signal
            )
            spatial_rows.append(
                {
                    "window_id": f"grid-r{row_index:03d}-c{column_index:03d}",
                    "spatial_row": row_index,
                    "spatial_column": column_index,
                    "center_x": int(center_x),
                    "center_y": int(center_y),
                    "bbox": {
                        "x0": bbox.x0,
                        "y0": bbox.y0,
                        "x1": bbox.x1,
                        "y1": bbox.y1,
                    },
                    "risk_score": float(risk_score),
                    "accepted_count": accepted_count,
                    "near_rejected_count": near_rejected_count,
                    "coarse_count": coarse_count,
                    "overview_log_mean": mean_signal,
                }
            )
    ordered = sorted(
        spatial_rows,
        key=lambda row: (
            -int(row["near_rejected_count"]),
            -int(row["coarse_count"]),
            -float(row["overview_log_mean"]),
            int(row["spatial_row"]),
            int(row["spatial_column"]),
        ),
    )
    for index, row in enumerate(ordered, start=1):
        row["review_order"] = index
    return ordered


def generate_recall_review_bundle(
    sample_dir: Path,
    *,
    overlay_dir: Path | None = None,
    window_radius_px: int = DEFAULT_WINDOW_RADIUS_PX,
    window_stride_px: int = DEFAULT_WINDOW_STRIDE_PX,
    overview_downsample: int = DEFAULT_OVERVIEW_DOWNSAMPLE,
) -> RecallReviewBundle:
    """Generate deterministic metadata, overview, and recall-review HTML."""

    if not MIN_RECALL_WINDOW_RADIUS_PX <= window_radius_px <= MAX_RECALL_WINDOW_RADIUS_PX:
        raise ValueError(
            "window radius must be between "
            f"{MIN_RECALL_WINDOW_RADIUS_PX} and {MAX_RECALL_WINDOW_RADIUS_PX} px"
        )
    if window_stride_px < 1 or window_stride_px > 2 * window_radius_px + 1:
        raise ValueError("window stride must be positive and cannot leave coverage gaps")
    if overview_downsample < 1:
        raise ValueError("overview downsample must be positive")
    sample = _load_sample(sample_dir, overlay_dir=overlay_dir)
    assets_dir = sample.sample_dir / "recall_review"
    overview_path = assets_dir / "overview.webp"
    metadata_path = assets_dir / "metadata.json"
    page_path = sample.sample_dir / "recall_review.html"
    console_path = sample.sample_dir / "review_console.html"
    LOGGER.info("Recall bundle reading raw UCHL1 overview for %s", sample.sample_id)
    with _OmeChannelSource(sample.source_image, sample.channel_index) as source:
        if source.image_shape_yx != sample.image_shape_yx:
            raise ValueError(
                f"manifest/source image shape mismatch: {sample.image_shape_yx} != {source.image_shape_yx}"
            )
        overview, global_low, global_high = _build_overview(
            source,
            downsample=overview_downsample,
        )
    overview_image = _magma_image(overview, global_low, global_high)
    _atomic_save_image(overview_path, overview_image, format_name="WEBP")
    windows = _coverage_windows(
        sample,
        overview,
        window_radius_px=window_radius_px,
        window_stride_px=window_stride_px,
        overview_downsample=overview_downsample,
    )
    coverage_identity = _coverage_identity(
        windows,
        window_radius_px=window_radius_px,
        window_stride_px=window_stride_px,
        overview_downsample=overview_downsample,
    )
    review_identity = {**sample.review_identity, **coverage_identity}
    metadata = {
        "schema_version": RECALL_REVIEW_SCHEMA_VERSION,
        "review_type": "oocyte_recall",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample.sample_id,
        "image_height": sample.image_shape_yx[0],
        "image_width": sample.image_shape_yx[1],
        "pixel_size_um": sample.pixel_size_um,
        "channel_index": sample.channel_index,
        "window_radius_px": window_radius_px,
        "window_stride_px": window_stride_px,
        "overview_downsample": overview_downsample,
        "recall_coverage_profile": coverage_identity["recall_coverage_profile"],
        "overview_width": int(overview.shape[1]),
        "overview_height": int(overview.shape[0]),
        "global_log_low": global_low,
        "global_log_high": global_high,
        "review_identity": review_identity,
        "mask_overlay": {
            "mode": sample.review_identity.get("overlay_mode", "automatic_candidates"),
            "delivery_name": sample.review_identity.get(
                "overlay_delivery_name", sample.profile_name
            ),
            "candidate_count": len(sample.candidates),
            "candidate_table_sha256": sample.review_identity.get(
                "overlay_candidate_table_sha256",
                sample.review_identity["candidate_table_sha256"],
            ),
            "manifest_sha256": sample.review_identity.get("overlay_manifest_sha256"),
        },
        "candidates": _browser_candidate_rows(sample.candidates),
        "windows": windows,
    }
    _atomic_write_text(
        metadata_path,
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True, allow_nan=False),
    )
    _atomic_write_text(page_path, recall_review_page_html(sample.sample_id))
    _atomic_write_text(
        console_path,
        review_console_page_html(
            sample_id=sample.sample_id,
            profile_name=sample.profile_name,
            candidate_count=len(sample.candidates),
            window_count=len(windows),
            image_shape_yx=sample.image_shape_yx,
            overlay_name=str(
                sample.review_identity.get("overlay_delivery_name", "automatic detector")
            ),
        ),
    )
    LOGGER.info(
        "Recall bundle completed %s: %s windows",
        sample.sample_id,
        len(windows),
    )
    return RecallReviewBundle(
        sample_id=sample.sample_id,
        sample_dir=sample.sample_dir,
        page_path=page_path,
        overview_path=overview_path,
        metadata_path=metadata_path,
        console_path=console_path,
        window_count=len(windows),
    )


def _distance_to_rows(
    table: pd.DataFrame,
    x: float,
    y: float,
    *,
    x_columns: Iterable[str],
    y_columns: Iterable[str],
) -> float | None:
    if table.empty:
        return None
    x_column = next((name for name in x_columns if name in table), None)
    y_column = next((name for name in y_columns if name in table), None)
    if x_column is None or y_column is None:
        return None
    xs = pd.to_numeric(table[x_column], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(table[y_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not valid.any():
        return None
    return float(np.hypot(xs[valid] - x, ys[valid] - y).min())


def classify_recall_failure(
    *,
    already_covered: bool,
    nearest_suppressed_distance_px: float | None,
    nearest_coarse_distance_px: float | None,
    nearest_refined_distance_px: float | None,
    suppressed_radius_px: float = 100.0,
    coarse_radius_px: float = 160.0,
    refined_radius_px: float = 100.0,
) -> str:
    """Classify the earliest detector stage associated with a manual click."""

    if already_covered:
        return "already_covered"
    if (
        nearest_suppressed_distance_px is not None
        and nearest_suppressed_distance_px <= suppressed_radius_px
    ):
        return "dedup_error"
    if nearest_refined_distance_px is not None and nearest_refined_distance_px <= refined_radius_px:
        return "acceptance_miss"
    if nearest_coarse_distance_px is not None and nearest_coarse_distance_px <= coarse_radius_px:
        return "segmentation_miss"
    return "proposal_miss"


def _split_touching_manual_component(
    selected: LocalSegmentationResult,
    smooth: np.ndarray,
    center_y: int,
    center_x: int,
) -> LocalSegmentationResult:
    """Split a multi-lobed component and retain the basin nearest the click."""

    distance = ndi.distance_transform_edt(selected.mask)
    peaks = feature.peak_local_max(
        distance,
        min_distance=30,
        threshold_abs=5.0,
        labels=selected.mask.astype(np.uint8),
        exclude_border=False,
    )
    if len(peaks) < 2:
        return selected
    ordered = sorted(
        (tuple(int(value) for value in peak) for peak in peaks),
        key=lambda peak: float(np.hypot(peak[0] - center_y, peak[1] - center_x)),
    )
    markers = np.zeros(selected.mask.shape, dtype=np.int32)
    for label, (peak_y, peak_x) in enumerate(ordered, start=1):
        markers[peak_y, peak_x] = label
    basins = segmentation.watershed(-distance, markers, mask=selected.mask)
    basin_areas = np.bincount(basins.ravel())
    target = np.asarray(basins == 1, dtype=np.bool_)
    min_basin_area_px = 400
    other_areas = basin_areas[2:] if len(basin_areas) > 2 else np.asarray([])
    if int(target.sum()) < min_basin_area_px or not np.any(
        other_areas >= min_basin_area_px
    ):
        return selected

    prop = measure.regionprops(
        target.astype(np.uint8),
        intensity_image=smooth,
    )[0]
    pixel_size_um = DONOR13_V6.pixel_size_um
    circularity = (
        0.0
        if prop.perimeter <= 0
        else float(4.0 * np.pi * prop.area / (prop.perimeter**2))
    )
    metrics = dataclass_replace(
        selected.metrics,
        selection_mode="manual_seed_watershed_component",
        area_px=int(prop.area),
        equivalent_diameter_um=float(prop.equivalent_diameter_area * pixel_size_um),
        major_axis_um=float(prop.axis_major_length * pixel_size_um),
        minor_axis_um=float(prop.axis_minor_length * pixel_size_um),
        eccentricity=float(prop.eccentricity),
        solidity=float(prop.solidity),
        circularity=circularity,
        centroid_y_px=float(prop.centroid[0]),
        centroid_x_px=float(prop.centroid[1]),
        centroid_offset_px=float(
            np.hypot(prop.centroid[1] - center_x, prop.centroid[0] - center_y)
        ),
        mean_intensity=float(prop.mean_intensity),
        max_intensity=float(prop.max_intensity),
    )
    return LocalSegmentationResult(mask=target, metrics=metrics)


def _segment_manual_seed_patch(
    patch: np.ndarray,
    *,
    annulus_floor_percentile: float,
    annulus_inner_px: int | None = None,
    annulus_outer_px: int | None = None,
) -> LocalSegmentationResult:
    """Select the plausible thresholded component nearest the manual click."""

    smooth, components = _segment_oocyte_patch_components(
        patch,
        DONOR13_V6,
        annulus_inner_px=(
            DONOR13_V6.local.annulus_inner_px
            if annulus_inner_px is None
            else annulus_inner_px
        ),
        annulus_outer_px=(
            DONOR13_V6.local.annulus_outer_px
            if annulus_outer_px is None
            else annulus_outer_px
        ),
        annulus_floor_percentile=annulus_floor_percentile,
    )
    if not components:
        raise ValueError("no connected components found after thresholding")
    plausible = [
        component
        for component in components
        if 10.0 <= component.metrics.equivalent_diameter_um <= 100.0
        and component.metrics.centroid_offset_px <= 120.0
    ]
    pool = plausible or components
    center_y = patch.shape[0] // 2
    center_x = patch.shape[1] // 2

    def rank(component: LocalSegmentationResult) -> Tuple[float, float, float]:
        click_distance = float(
            ndi.distance_transform_edt(~component.mask)[center_y, center_x]
        )
        return (
            click_distance,
            float(component.metrics.centroid_offset_px),
            -float(component.metrics.area_px),
        )

    selected = _split_touching_manual_component(
        min(pool, key=rank),
        smooth,
        center_y,
        center_x,
    )
    if selected.metrics.selection_mode == "manual_seed_watershed_component":
        return selected
    metrics = dataclass_replace(
        selected.metrics,
        selection_mode="manual_seed_nearest_component",
    )
    return LocalSegmentationResult(mask=selected.mask, metrics=metrics)


class RecallReviewRuntime:
    """Read-only runtime used by both HTTP routes and offline analysis."""

    def __init__(
        self,
        sample_dir: Path,
        *,
        overlay_dir: Path | None = None,
    ):
        sample_root = Path(sample_dir).resolve()
        self.bundle_dir = sample_root / "recall_review"
        self.metadata_path = self.bundle_dir / "metadata.json"
        self.page_path = sample_root / "recall_review.html"
        self.overview_path = self.bundle_dir / "overview.webp"
        if not self.metadata_path.is_file() or not self.page_path.is_file():
            raise FileNotFoundError("recall review bundle is missing; generate it first")
        self.metadata = _read_json(self.metadata_path)
        _validate_bundle_coverage_identity(self.metadata)
        identity = self.metadata.get("review_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("recall review bundle is missing its review identity")
        bound_overlay_dir = overlay_dir_from_identity(identity)
        if overlay_dir is not None and bound_overlay_dir != Path(overlay_dir).resolve():
            raise ValueError("requested Recall overlay does not match generated bundle")
        effective_overlay_dir = (
            Path(overlay_dir).resolve() if overlay_dir is not None else bound_overlay_dir
        )
        self.sample = _load_sample(
            sample_root,
            overlay_dir=effective_overlay_dir,
        )
        if not _identity_contains_required(identity, self.sample.review_identity):
            raise ValueError("recall review bundle identity does not match current sample output")
        self.source = _OmeChannelSource(
            self.sample.source_image,
            self.sample.channel_index,
        )
        self._mask_cache: Dict[str, PersistedMask] = {}

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> "RecallReviewRuntime":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _candidate_mask(self, row: Mapping[str, Any]) -> PersistedMask:
        path = _mask_path(self.sample.sample_dir, row).resolve()
        key = str(path)
        if key not in self._mask_cache:
            self._mask_cache[key] = load_candidate_mask(path)
        return self._mask_cache[key]

    def _intersecting_rows(self, patch: ExtractedPatch) -> Sequence[Dict[str, Any]]:
        rows = []
        for row in self.sample.candidates.to_dict("records"):
            x0, y0 = float(row["bbox_x0"]), float(row["bbox_y0"])
            x1, y1 = float(row["bbox_x1"]), float(row["bbox_y1"])
            if x0 >= patch.bbox.x1 or y0 >= patch.bbox.y1 or x1 <= patch.bbox.x0 or y1 <= patch.bbox.y0:
                continue
            rows.append(row)
        return rows

    @staticmethod
    def _place_mask(mask: PersistedMask, patch: ExtractedPatch) -> np.ndarray:
        placed = np.zeros(patch.image.shape, dtype=np.bool_)
        x0 = max(mask.bbox.x0, patch.bbox.x0)
        y0 = max(mask.bbox.y0, patch.bbox.y0)
        x1 = min(mask.bbox.x1, patch.bbox.x1)
        y1 = min(mask.bbox.y1, patch.bbox.y1)
        if x0 >= x1 or y0 >= y1:
            return placed
        top, _, left, _ = patch.padding_tblr
        source = mask.mask[
            y0 - mask.bbox.y0 : y1 - mask.bbox.y0,
            x0 - mask.bbox.x0 : x1 - mask.bbox.x0,
        ]
        target_y = top + y0 - patch.bbox.y0
        target_x = left + x0 - patch.bbox.x0
        placed[target_y : target_y + source.shape[0], target_x : target_x + source.shape[1]] = source
        return placed

    def render_patch(self, center_xy: Tuple[int, int], radius: int, contrast: str) -> bytes:
        patch = self.source.read_patch(center_xy, radius)
        transformed = np.log1p(np.maximum(patch.image.astype(np.float32), 0.0))
        finite = transformed[np.isfinite(transformed)]
        if contrast == "global":
            low = float(self.metadata["global_log_low"])
            high = float(self.metadata["global_log_high"])
        elif contrast == "local":
            if finite.size:
                low, high = (float(value) for value in np.percentile(finite, [2.0, 99.8]))
            else:
                low, high = 0.0, 1.0
        else:
            raise ValueError("contrast must be 'local' or 'global'")
        if high <= low:
            high = low + 1.0
        normalized = np.clip((transformed - low) / (high - low), 0.0, 1.0)
        rgb = np.asarray(colormaps["magma"](normalized)[..., :3] * 255.0, dtype=np.uint8)
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="WEBP", quality=90, method=5)
        return output.getvalue()

    def render_overlay(self, center_xy: Tuple[int, int], radius: int) -> bytes:
        patch = self.source.read_patch(center_xy, radius)
        rgba = np.zeros((*patch.image.shape, 4), dtype=np.uint8)
        rows = self._intersecting_rows(patch)
        label_positions = []
        for row in rows:
            mask = self._candidate_mask(row)
            placed = self._place_mask(mask, patch)
            if not placed.any():
                continue
            rgba[placed] = np.array([0, 235, 222, 42], dtype=np.uint8)
            boundary = np.logical_xor(placed, ndi.binary_erosion(placed))
            boundary = ndi.binary_dilation(boundary, iterations=1)
            rgba[boundary] = np.array([0, 255, 242, 245], dtype=np.uint8)
            requested_x0 = center_xy[0] - radius
            requested_y0 = center_xy[1] - radius
            label_positions.append(
                (
                    int(round(float(row["center_x"]))) - requested_x0,
                    int(round(float(row["center_y"]))) - requested_y0,
                    str(row.get("display_id", row["detector_component_id"])),
                )
            )
        image = Image.fromarray(rgba)
        draw = ImageDraw.Draw(image)
        for x, y, label in label_positions:
            draw.rectangle((x + 5, y - 15, x + 49, y + 2), fill=(10, 25, 24, 210))
            draw.text((x + 8, y - 14), label, fill=(255, 255, 244, 255))
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def window_payload(self, center_xy: Tuple[int, int], radius: int) -> Dict[str, Any]:
        patch = self.source.read_patch(center_xy, radius)
        candidates = []
        for row in self._intersecting_rows(patch):
            candidates.append(
                {
                    "detector_component_id": str(row["detector_component_id"]),
                    "display_id": str(row.get("display_id", row["detector_component_id"])),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "detector_score": float(row.get("detector_score", 0.0)),
                    "detection_pass": str(row.get("detection_pass", "baseline_v6")),
                    "resolution_source": str(
                        row.get(
                            "precision_resolution_source",
                            row.get("acceptance_mode", "automatic_candidate"),
                        )
                    ),
                }
            )
        return {
            "center_x": center_xy[0],
            "center_y": center_xy[1],
            "radius": radius,
            "bbox": {
                "x0": patch.bbox.x0,
                "y0": patch.bbox.y0,
                "x1": patch.bbox.x1,
                "y1": patch.bbox.y1,
            },
            "padding_tblr": list(patch.padding_tblr),
            "candidates": candidates,
        }

    def _point_covered(self, x: float, y: float) -> bool:
        px, py = int(round(x)), int(round(y))
        for row in self.sample.candidates.to_dict("records"):
            if not (
                float(row["bbox_x0"]) <= px < float(row["bbox_x1"])
                and float(row["bbox_y0"]) <= py < float(row["bbox_y1"])
            ):
                continue
            mask = self._candidate_mask(row)
            if bool(mask.mask[py - mask.bbox.y0, px - mask.bbox.x0]):
                return True
        return False

    def segment_probe(self, x: float, y: float) -> ProbeSegmentation:
        radius = DONOR13_V6.local.window_radius_px
        patch = self.source.read_patch((int(round(x)), int(round(y))), radius)
        p99 = p95 = None
        p99_error = p95_error = None
        try:
            p99 = _segment_manual_seed_patch(
                patch.image,
                annulus_floor_percentile=99.0,
            )
        except (ValueError, TypeError) as exc:
            p99_error = str(exc)
        try:
            p95 = _segment_manual_seed_patch(
                patch.image,
                annulus_floor_percentile=95.0,
            )
        except (ValueError, TypeError) as exc:
            p95_error = str(exc)
        return ProbeSegmentation(
            patch=patch,
            p99=p99,
            p95=p95,
            p99_error=p99_error,
            p95_error=p95_error,
        )

    def segment_manual_provisionals(
        self,
        x: float,
        y: float,
        *,
        exclude_points_xy: Sequence[Tuple[float, float]] = (),
        allow_shape_recovery: bool = False,
    ) -> ManualSeedSegmentation:
        radius = 100
        rounded_x = int(round(x))
        rounded_y = int(round(y))
        patch = self.source.read_patch((rounded_x, rounded_y), radius)
        center_y = patch.image.shape[0] // 2
        center_x = patch.image.shape[1] // 2
        options = []
        errors = []
        for percentile in (95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0):
            try:
                result = _segment_manual_seed_patch(
                    patch.image,
                    annulus_floor_percentile=percentile,
                    annulus_inner_px=40,
                    annulus_outer_px=90,
                )
            except (ValueError, TypeError) as exc:
                errors.append(f"P{percentile:g}: {exc}")
                continue
            distance_to_mask = ndi.distance_transform_edt(~result.mask)
            click_distance = float(distance_to_mask[center_y, center_x])
            near_other_seed = False
            for other_x, other_y in exclude_points_xy:
                if float(np.hypot(other_x - x, other_y - y)) < 25.0:
                    continue
                local_x = center_x + int(round(other_x)) - rounded_x
                local_y = center_y + int(round(other_y)) - rounded_y
                if not (
                    0 <= local_y < result.mask.shape[0]
                    and 0 <= local_x < result.mask.shape[1]
                ):
                    continue
                if float(distance_to_mask[local_y, local_x]) <= 5.0:
                    near_other_seed = True
                    break
            options.append((percentile, result, click_distance, near_other_seed))
        valid = [
            option
            for option in options
            if option[2] <= 25.0
            and not option[3]
            and option[1].metrics.centroid_offset_px <= 50.0
            and 12.0 <= option[1].metrics.equivalent_diameter_um <= 100.0
        ]
        if not valid:
            return ManualSeedSegmentation(
                patch=patch,
                conservative=None,
                expanded=None,
                conservative_percentile=None,
                expanded_percentile=None,
                error="; ".join(errors) or "no click-targeted component passed geometry gates",
            )
        conservative_percentile, conservative, _, _ = valid[0]
        conservative_area = max(float(conservative.metrics.area_px), 1.0)
        expansion_options = []
        for percentile, result, click_distance, _ in valid:
            intersection = int(np.logical_and(conservative.mask, result.mask).sum())
            conservative_overlap = intersection / conservative_area
            area_ratio = float(result.metrics.area_px) / conservative_area
            standard_expansion = area_ratio <= 4.0
            shape_recovery = bool(
                allow_shape_recovery
                and area_ratio <= 6.0
                and result.metrics.equivalent_diameter_um >= 20.0
                and result.metrics.circularity >= 0.80
                and result.metrics.solidity >= 0.90
                and result.metrics.centroid_offset_px <= 25.0
            )
            if (
                click_distance <= 25.0
                and conservative_overlap >= 0.70
                and (standard_expansion or shape_recovery)
            ):
                expansion_options.append((percentile, result, area_ratio))
        expanded_percentile, expanded, _ = max(
            expansion_options,
            key=lambda item: (float(item[1].metrics.area_px), item[0]),
        )
        return ManualSeedSegmentation(
            patch=patch,
            conservative=conservative,
            expanded=expanded,
            conservative_percentile=conservative_percentile,
            expanded_percentile=expanded_percentile,
            error=None,
        )

    def probe(self, x: float, y: float) -> Dict[str, Any]:
        image_h, image_w = self.sample.image_shape_yx
        if not (0 <= x < image_w and 0 <= y < image_h):
            raise ValueError("probe coordinate must be inside the source image")
        nearest_accepted = _distance_to_rows(
            self.sample.candidates,
            x,
            y,
            x_columns=("component_centroid_x", "center_x"),
            y_columns=("component_centroid_y", "center_y"),
        )
        nearest_refined = _distance_to_rows(
            self.sample.refined_candidates,
            x,
            y,
            x_columns=("component_centroid_x", "center_x", "seed_center_x"),
            y_columns=("component_centroid_y", "center_y", "seed_center_y"),
        )
        nearest_coarse = _distance_to_rows(
            self.sample.coarse_candidates,
            x,
            y,
            x_columns=("coarse_center_x",),
            y_columns=("coarse_center_y",),
        )
        nearest_suppressed = _distance_to_rows(
            self.sample.suppressed_candidates,
            x,
            y,
            x_columns=("component_centroid_x", "center_x", "seed_center_x"),
            y_columns=("component_centroid_y", "center_y", "seed_center_y"),
        )
        covered = self._point_covered(x, y)
        failure_class = classify_recall_failure(
            already_covered=covered,
            nearest_suppressed_distance_px=nearest_suppressed,
            nearest_coarse_distance_px=nearest_coarse,
            nearest_refined_distance_px=nearest_refined,
        )
        segmentation = self.segment_probe(x, y)
        provisionals = self.segment_manual_provisionals(x, y)
        return {
            "x": float(x),
            "y": float(y),
            "already_covered": covered,
            "failure_class": failure_class,
            "nearest_accepted_distance_px": nearest_accepted,
            "nearest_refined_distance_px": nearest_refined,
            "nearest_coarse_distance_px": nearest_coarse,
            "nearest_suppressed_distance_px": nearest_suppressed,
            "p99_metrics": None if segmentation.p99 is None else segmentation.p99.metrics.to_dict(),
            "p95_metrics": None if segmentation.p95 is None else segmentation.p95.metrics.to_dict(),
            "p99_error": segmentation.p99_error,
            "p95_error": segmentation.p95_error,
            "manual_conservative_metrics": (
                None
                if provisionals.conservative is None
                else provisionals.conservative.metrics.to_dict()
            ),
            "manual_expanded_metrics": (
                None
                if provisionals.expanded is None
                else provisionals.expanded.metrics.to_dict()
            ),
            "manual_conservative_percentile": provisionals.conservative_percentile,
            "manual_expanded_percentile": provisionals.expanded_percentile,
            "manual_provisional_error": provisionals.error,
        }


def _query_number(query: Mapping[str, Sequence[str]], name: str) -> float:
    values = query.get(name)
    if not values or len(values) != 1:
        raise ValueError(f"query parameter {name!r} is required exactly once")
    try:
        value = float(values[0])
    except ValueError as exc:
        raise ValueError(f"query parameter {name!r} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"query parameter {name!r} must be finite")
    return value


def _request_geometry(
    runtime: RecallReviewRuntime,
    query: Mapping[str, Sequence[str]],
) -> Tuple[Tuple[int, int], int]:
    x = int(round(_query_number(query, "x")))
    y = int(round(_query_number(query, "y")))
    radius = int(round(_query_number(query, "radius")))
    image_h, image_w = runtime.sample.image_shape_yx
    if not 0 <= x < image_w or not 0 <= y < image_h:
        raise ValueError("x and y must be inside the source image")
    if not MIN_RECALL_WINDOW_RADIUS_PX <= radius <= MAX_RECALL_WINDOW_RADIUS_PX:
        raise ValueError(
            "radius must be between "
            f"{MIN_RECALL_WINDOW_RADIUS_PX} and {MAX_RECALL_WINDOW_RADIUS_PX} px"
        )
    return (x, y), radius


def _handler_for(runtime: RecallReviewRuntime):
    class RecallReviewHandler(BaseHTTPRequestHandler):
        server_version = "AegleRecallReview/1"

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info("Recall HTTP %s - %s", self.address_string(), format_string % args)

        def _send(self, payload: bytes, content_type: str, *, status: int = 200, head: bool = False) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if not head:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers cancel stale patch requests while the focal window moves.
                LOGGER.debug("Recall HTTP client cancelled %s", self.path)

        def _json(self, payload: Mapping[str, Any], *, status: int = 200, head: bool = False) -> None:
            body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status=status, head=head)

        def _handle(self, *, head: bool) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if parsed.path == "/health":
                    self._json(
                        {
                            "status": "ok",
                            "sample_id": runtime.sample.sample_id,
                            "read_only": True,
                        },
                        head=head,
                    )
                    return
                if parsed.path in {"/", "/recall_review.html"}:
                    self._send(
                        runtime.page_path.read_bytes(),
                        "text/html; charset=utf-8",
                        head=head,
                    )
                    return
                if parsed.path == "/review_console.html":
                    console_page = runtime.sample.sample_dir / "review_console.html"
                    if not console_page.is_file():
                        raise FileNotFoundError("sample review console is unavailable")
                    self._send(
                        console_page.read_bytes(),
                        "text/html; charset=utf-8",
                        head=head,
                    )
                    return
                if parsed.path == "/oocytes.html":
                    precision_page = runtime.sample.sample_dir / "oocytes.html"
                    if not precision_page.is_file():
                        raise FileNotFoundError("precision review page is unavailable")
                    self._send(
                        precision_page.read_bytes(),
                        "text/html; charset=utf-8",
                        head=head,
                    )
                    return
                if parsed.path.startswith("/html_assets/"):
                    asset_name = Path(parsed.path).name
                    if (
                        parsed.path != f"/html_assets/{asset_name}"
                        or not asset_name.endswith(".webp")
                    ):
                        raise ValueError("invalid precision-review asset path")
                    asset_path = runtime.sample.sample_dir / "html_assets" / asset_name
                    if not asset_path.is_file():
                        raise FileNotFoundError("precision-review asset is unavailable")
                    self._send(asset_path.read_bytes(), "image/webp", head=head)
                    return
                if parsed.path in {
                    "/oocyte_review_index.html",
                    "/oocyte_review_console.html",
                    "/oocyte_detection_algorithm.html",
                }:
                    shared_path = runtime.sample.sample_dir.parent / parsed.path.lstrip("/")
                    if not shared_path.is_file():
                        raise FileNotFoundError("shared review page is unavailable")
                    self._send(
                        shared_path.read_bytes(),
                        "text/html; charset=utf-8",
                        head=head,
                    )
                    return
                if parsed.path.startswith("/recall_analysis"):
                    relative_path = Path(parsed.path.lstrip("/"))
                    requested_path = (runtime.sample.sample_dir / relative_path).resolve()
                    try:
                        requested_path.relative_to(runtime.sample.sample_dir)
                    except ValueError as exc:
                        raise ValueError("invalid recall-analysis asset path") from exc
                    content_types = {
                        ".html": "text/html; charset=utf-8",
                        ".webp": "image/webp",
                        ".json": "application/json; charset=utf-8",
                        ".csv": "text/csv; charset=utf-8",
                    }
                    content_type = content_types.get(requested_path.suffix.lower())
                    if content_type is None or not requested_path.is_file():
                        raise FileNotFoundError("recall-analysis asset is unavailable")
                    self._send(requested_path.read_bytes(), content_type, head=head)
                    return
                if parsed.path == "/recall_review/overview.webp":
                    self._send(runtime.overview_path.read_bytes(), "image/webp", head=head)
                    return
                if parsed.path == "/api/metadata":
                    self._json(runtime.metadata, head=head)
                    return
                if parsed.path in {"/api/patch.webp", "/api/overlay.png", "/api/window"}:
                    center, radius = _request_geometry(runtime, query)
                    if parsed.path == "/api/patch.webp":
                        contrast = query.get("contrast", ["local"])[0]
                        self._send(
                            runtime.render_patch(center, radius, contrast),
                            "image/webp",
                            head=head,
                        )
                    elif parsed.path == "/api/overlay.png":
                        self._send(runtime.render_overlay(center, radius), "image/png", head=head)
                    else:
                        self._json(runtime.window_payload(center, radius), head=head)
                    return
                if parsed.path == "/api/probe":
                    x = _query_number(query, "x")
                    y = _query_number(query, "y")
                    self._json(runtime.probe(x, y), head=head)
                    return
                self._json({"error": "route not found"}, status=HTTPStatus.NOT_FOUND, head=head)
            except (ValueError, FileNotFoundError) as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST, head=head)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                LOGGER.exception("Recall review request failed")
                self._json(
                    {"error": f"internal server error: {type(exc).__name__}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    head=head,
                )

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP method name
            self._handle(head=False)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib HTTP method name
            self._handle(head=True)

    return RecallReviewHandler


def serve_recall_review(
    sample_dir: Path,
    *,
    overlay_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8767,
    generate: bool = True,
    window_radius_px: int = DEFAULT_WINDOW_RADIUS_PX,
    window_stride_px: int = DEFAULT_WINDOW_STRIDE_PX,
    overview_downsample: int = DEFAULT_OVERVIEW_DOWNSAMPLE,
) -> None:
    """Generate if requested, then serve one sample until interrupted."""

    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if generate:
        generate_recall_review_bundle(
            sample_dir,
            overlay_dir=overlay_dir,
            window_radius_px=window_radius_px,
            window_stride_px=window_stride_px,
            overview_downsample=overview_downsample,
        )
    with RecallReviewRuntime(sample_dir, overlay_dir=overlay_dir) as runtime:
        server = ThreadingHTTPServer((host, int(port)), _handler_for(runtime))
        server.daemon_threads = True
        LOGGER.info(
            "Recall review serving sample=%s shape=%s channel=%s source=%s",
            runtime.sample.sample_id,
            runtime.sample.image_shape_yx,
            runtime.sample.channel_index,
            runtime.sample.source_image,
        )
        LOGGER.info("Review console URL: http://%s:%s/review_console.html", host, port)
        LOGGER.info("Recall review URL: http://%s:%s/recall_review.html", host, port)
        LOGGER.info("Production detector outputs are read-only")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            LOGGER.info("Recall review server interrupted")
        finally:
            server.server_close()


def _validate_review_payload(sample: RecallReviewSample, payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != RECALL_REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported recall review schema_version")
    if payload.get("review_type") != "oocyte_recall":
        raise ValueError("review_type must be 'oocyte_recall'")
    identity = payload.get("sample")
    if not isinstance(identity, Mapping):
        raise ValueError("review JSON must contain a sample identity object")
    for field, expected in sample.review_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"review sample identity mismatch for {field}")
    misses = payload.get("missing_oocytes")
    if not isinstance(misses, list):
        raise ValueError("review JSON missing_oocytes must be a list")


def _save_provisional_mask(
    path: Path,
    *,
    result: LocalSegmentationResult,
    patch: ExtractedPatch,
    annotation_id: str,
    percentile: float,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bounded_mask = patch.crop_to_image_bounds(result.mask).astype(np.bool_)
    metadata = {
        "schema_version": 1,
        "annotation_id": annotation_id,
        "annulus_floor_percentile": percentile,
        "metrics": result.metrics.to_dict(),
        "provisional_only": True,
    }
    np.savez_compressed(
        destination,
        mask=bounded_mask,
        bbox_xyxy=np.asarray(patch.bbox.as_tuple(), dtype=np.int64),
        image_shape_yx=np.asarray(patch.image_shape_yx, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(_json_safe(metadata), sort_keys=True)),
    )
    return destination


def analyze_recall_review(
    sample_dir: Path,
    review_json: Path,
    out_dir: Path,
    *,
    overlay_dir: Path | None = None,
) -> Path:
    """Validate an exported review and classify every missing-oocyte click."""

    payload = _read_json(Path(review_json))
    identity = payload.get("sample")
    if not isinstance(identity, Mapping):
        raise ValueError("review JSON must contain a sample identity object")
    bundle_metadata_path = Path(sample_dir).resolve() / "recall_review/metadata.json"
    if not bundle_metadata_path.is_file():
        raise FileNotFoundError(
            f"current Recall bundle metadata is missing: {bundle_metadata_path}"
        )
    bundle_metadata = _read_json(bundle_metadata_path)
    bundle_identity = bundle_metadata.get("review_identity")
    if not isinstance(bundle_identity, Mapping):
        raise ValueError("current Recall bundle identity is missing")
    for field in sorted(set(identity) | set(bundle_identity)):
        if identity.get(field) != bundle_identity.get(field):
            raise ValueError(
                f"review sample identity mismatch for current bundle field {field}"
            )
    bound_overlay_dir = overlay_dir_from_identity(bundle_identity)
    if overlay_dir is not None and bound_overlay_dir != Path(overlay_dir).resolve():
        raise ValueError("requested Recall overlay does not match review JSON")
    effective_overlay_dir = (
        Path(overlay_dir).resolve() if overlay_dir is not None else bound_overlay_dir
    )
    sample = _load_sample(sample_dir, overlay_dir=effective_overlay_dir)
    _validate_review_payload(sample, payload)
    destination = Path(out_dir).resolve()
    masks_dir = destination / "provisional_masks"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    annotations = payload["missing_oocytes"]
    centers = [
        (float(annotation["x"]), float(annotation["y"]))
        for annotation in annotations
        if isinstance(annotation, Mapping)
    ]
    with RecallReviewRuntime(
        sample.sample_dir,
        overlay_dir=effective_overlay_dir,
    ) as runtime:
        for index, annotation in enumerate(annotations, start=1):
            if not isinstance(annotation, Mapping):
                raise ValueError(f"missing_oocytes[{index - 1}] must be an object")
            annotation_id = str(annotation.get("annotation_id", f"miss-{index:04d}"))
            x = float(annotation["x"])
            y = float(annotation["y"])
            probe = runtime.probe(x, y)
            provisionals = runtime.segment_manual_provisionals(
                x,
                y,
                exclude_points_xy=tuple(
                    point
                    for point_index, point in enumerate(centers)
                    if point_index != index - 1
                ),
            )
            conservative_path = expanded_path = ""
            if provisionals.conservative is not None:
                conservative_path = str(
                    _save_provisional_mask(
                        masks_dir / f"{annotation_id}_conservative.npz",
                        result=provisionals.conservative,
                        patch=provisionals.patch,
                        annotation_id=annotation_id,
                        percentile=float(provisionals.conservative_percentile),
                    )
                )
            if provisionals.expanded is not None:
                expanded_path = str(
                    _save_provisional_mask(
                        masks_dir / f"{annotation_id}_expanded.npz",
                        result=provisionals.expanded,
                        patch=provisionals.patch,
                        annotation_id=annotation_id,
                        percentile=float(provisionals.expanded_percentile),
                    )
                )
            row = {
                "annotation_id": annotation_id,
                "window_id": str(annotation.get("window_id", "")),
                "x": x,
                "y": y,
                "notes": str(annotation.get("notes", "")),
                "failure_class": probe["failure_class"],
                "already_covered": bool(probe["already_covered"]),
                "nearest_accepted_distance_px": probe["nearest_accepted_distance_px"],
                "nearest_refined_distance_px": probe["nearest_refined_distance_px"],
                "nearest_coarse_distance_px": probe["nearest_coarse_distance_px"],
                "nearest_suppressed_distance_px": probe["nearest_suppressed_distance_px"],
                "manual_conservative_mask_path": conservative_path,
                "manual_expanded_mask_path": expanded_path,
                "p99_error": probe["p99_error"],
                "p95_error": probe["p95_error"],
                "manual_conservative_percentile": provisionals.conservative_percentile,
                "manual_expanded_percentile": provisionals.expanded_percentile,
                "manual_provisional_error": provisionals.error,
            }
            for prefix in ("p99", "p95"):
                metrics = probe[f"{prefix}_metrics"] or {}
                for field in (
                    "equivalent_diameter_um",
                    "circularity",
                    "solidity",
                    "centroid_offset_px",
                    "mean_intensity",
                    "max_intensity",
                ):
                    row[f"{prefix}_{field}"] = metrics.get(field)
            manual_results = {
                "manual_conservative": provisionals.conservative,
                "manual_expanded": provisionals.expanded,
            }
            for prefix, result in manual_results.items():
                metrics = {} if result is None else result.metrics.to_dict()
                for field in (
                    "equivalent_diameter_um",
                    "circularity",
                    "solidity",
                    "centroid_offset_px",
                    "mean_intensity",
                    "max_intensity",
                ):
                    row[f"{prefix}_{field}"] = metrics.get(field)
            rows.append(row)
    columns = [
        "annotation_id",
        "window_id",
        "x",
        "y",
        "notes",
        "failure_class",
        "already_covered",
        "nearest_accepted_distance_px",
        "nearest_refined_distance_px",
        "nearest_coarse_distance_px",
        "nearest_suppressed_distance_px",
        "manual_conservative_mask_path",
        "manual_expanded_mask_path",
        "p99_error",
        "p95_error",
        "manual_conservative_percentile",
        "manual_expanded_percentile",
        "manual_provisional_error",
        "p99_equivalent_diameter_um",
        "p99_circularity",
        "p99_solidity",
        "p99_centroid_offset_px",
        "p99_mean_intensity",
        "p99_max_intensity",
        "p95_equivalent_diameter_um",
        "p95_circularity",
        "p95_solidity",
        "p95_centroid_offset_px",
        "p95_mean_intensity",
        "p95_max_intensity",
        "manual_conservative_equivalent_diameter_um",
        "manual_conservative_circularity",
        "manual_conservative_solidity",
        "manual_conservative_centroid_offset_px",
        "manual_conservative_mean_intensity",
        "manual_conservative_max_intensity",
        "manual_expanded_equivalent_diameter_um",
        "manual_expanded_circularity",
        "manual_expanded_solidity",
        "manual_expanded_centroid_offset_px",
        "manual_expanded_mean_intensity",
        "manual_expanded_max_intensity",
    ]
    table_path = destination / "recall_failure_analysis.csv"
    pd.DataFrame(rows, columns=columns).to_csv(table_path, index=False)
    summary = {
        "schema_version": 1,
        "sample": dict(identity),
        "review_json": str(Path(review_json).resolve()),
        "reviewed_window_count": len(payload.get("windows", [])),
        "missing_oocyte_count": len(rows),
        "failure_class_counts": (
            pd.Series([row["failure_class"] for row in rows]).value_counts().to_dict()
            if rows
            else {}
        ),
        "analysis_table": str(table_path),
        "production_outputs_modified": False,
    }
    _atomic_write_text(
        destination / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
    )
    from .manual_seed_review import generate_manual_seed_review

    review = generate_manual_seed_review(sample.sample_dir, destination)
    summary["manual_seed_review_page"] = str(review.page_path)
    summary["manual_seed_review_card_count"] = review.card_count
    _atomic_write_text(
        destination / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
    )
    return table_path


__all__ = [
    "DEFAULT_OVERVIEW_DOWNSAMPLE",
    "DEFAULT_WINDOW_RADIUS_PX",
    "DEFAULT_WINDOW_STRIDE_PX",
    "MAX_RECALL_WINDOW_RADIUS_PX",
    "MIN_RECALL_WINDOW_RADIUS_PX",
    "RECALL_REVIEW_SCHEMA_VERSION",
    "RecallReviewBundle",
    "RecallReviewRuntime",
    "analyze_recall_review",
    "classify_recall_failure",
    "generate_recall_review_bundle",
    "serve_recall_review",
]
