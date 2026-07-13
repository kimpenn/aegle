"""Raw-image patch access and exact candidate-mask persistence."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tifffile
import zarr

from .models import BoundingBox, ExtractedPatch, PersistedMask, SegmentationMetrics


def find_channel_index(antibodies_path: Path, channel_name: str = "UCHL1") -> int:
    """Resolve a marker name to its image channel index from a TSV table."""

    path = Path(antibodies_path)
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

        target = channel_name.strip().casefold()
        for row_index, row in enumerate(reader):
            if str(row.get(name_column, "")).strip().casefold() != target:
                continue
            channel_id = str(row.get("channel_id", ""))
            match = re.search(r":(\d+)$", channel_id)
            return int(match.group(1)) if match else row_index
    raise ValueError(f"channel {channel_name!r} not found in antibody table: {path}")


def _patch_geometry(
    image_shape_yx: Tuple[int, int],
    center_xy: Tuple[int, int],
    radius: int,
) -> Tuple[BoundingBox, Tuple[int, int, int, int]]:
    image_h, image_w = image_shape_yx
    center_x, center_y = center_xy
    if radius < 1:
        raise ValueError("patch radius must be positive")

    requested_x0 = center_x - radius
    requested_x1 = center_x + radius + 1
    requested_y0 = center_y - radius
    requested_y1 = center_y + radius + 1
    if (
        requested_x1 <= 0
        or requested_y1 <= 0
        or requested_x0 >= image_w
        or requested_y0 >= image_h
    ):
        raise ValueError("requested patch does not intersect the source image")
    x0 = max(0, requested_x0)
    x1 = min(image_w, requested_x1)
    y0 = max(0, requested_y0)
    y1 = min(image_h, requested_y1)
    bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    padding = (
        y0 - requested_y0,
        requested_y1 - y1,
        x0 - requested_x0,
        requested_x1 - x1,
    )
    return bbox, padding


def extract_padded_patch(
    image: np.ndarray,
    center_xy: Tuple[int, int],
    radius: int,
) -> ExtractedPatch:
    """Extract an edge-padded fixed-size patch from a two-dimensional image."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("source image must be two-dimensional")
    image_shape = (int(array.shape[0]), int(array.shape[1]))
    bbox, padding = _patch_geometry(image_shape, center_xy, radius)
    patch = np.asarray(array[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1])
    top, bottom, left, right = padding
    if any(padding):
        patch = np.pad(patch, ((top, bottom), (left, right)), mode="edge")
    return ExtractedPatch(
        image=patch,
        bbox=bbox,
        image_shape_yx=image_shape,
        padding_tblr=padding,
    )


def extract_cyx_channel_patch(
    array: Any,
    channel_index: int,
    center_xy: Tuple[int, int],
    radius: int,
) -> ExtractedPatch:
    """Extract a padded patch from an array-like object with C/Y/X axes."""

    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 3:
        raise ValueError("channel source must use C/Y/X axes")
    if not 0 <= channel_index < shape[0]:
        raise IndexError(f"channel index {channel_index} outside image channel range")
    image_shape = (shape[1], shape[2])
    bbox, padding = _patch_geometry(image_shape, center_xy, radius)
    patch = np.asarray(
        array[channel_index, bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    )
    top, bottom, left, right = padding
    if any(padding):
        patch = np.pad(patch, ((top, bottom), (left, right)), mode="edge")
    return ExtractedPatch(
        image=patch,
        bbox=bbox,
        image_shape_yx=image_shape,
        padding_tblr=padding,
    )


def read_ome_channel_patch(
    image_path: Path,
    channel_index: int,
    center_xy: Tuple[int, int],
    radius: int,
) -> ExtractedPatch:
    """Read one cropped channel patch without loading a whole OME-TIFF plane."""

    path = Path(image_path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = tuple(int(value) for value in series.shape)
        if "C" not in axes or "Y" not in axes or "X" not in axes:
            raise ValueError(f"expected C/Y/X axes in OME-TIFF, got {axes!r}")

        channel_axis = axes.index("C")
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        if not 0 <= channel_index < shape[channel_axis]:
            raise IndexError(
                f"channel index {channel_index} outside [0, {shape[channel_axis]})"
            )
        image_shape = (shape[y_axis], shape[x_axis])
        bbox, padding = _patch_geometry(image_shape, center_xy, radius)

        store = series.aszarr()
        array = zarr.open(store, mode="r")
        indexer = []
        for axis, axis_size in zip(axes, shape):
            if axis == "C":
                indexer.append(channel_index)
            elif axis == "Y":
                indexer.append(slice(bbox.y0, bbox.y1))
            elif axis == "X":
                indexer.append(slice(bbox.x0, bbox.x1))
            elif axis_size == 1:
                indexer.append(0)
            else:
                raise ValueError(
                    f"unsupported non-singleton OME axis {axis!r} with size {axis_size}"
                )
        patch = np.asarray(array[tuple(indexer)])

    if patch.ndim != 2:
        raise ValueError(f"channel patch did not resolve to two dimensions: {patch.shape}")
    top, bottom, left, right = padding
    if any(padding):
        patch = np.pad(patch, ((top, bottom), (left, right)), mode="edge")
    return ExtractedPatch(
        image=patch,
        bbox=bbox,
        image_shape_yx=image_shape,
        padding_tblr=padding,
    )


def save_candidate_mask(
    path: Path,
    *,
    mask: np.ndarray,
    bbox: BoundingBox,
    image_shape_yx: Tuple[int, int],
    sample_id: str,
    candidate_id: str,
    profile_name: str,
    profile_fingerprint: str,
    metrics: SegmentationMetrics,
    implementation_version: str | None = None,
) -> Path:
    """Atomically persist the exact cropped mask used for candidate scoring."""

    mask_array = np.asarray(mask, dtype=np.bool_)
    if mask_array.shape != bbox.shape_yx:
        raise ValueError("mask shape must match its image-space bounding box")
    image_h, image_w = image_shape_yx
    if bbox.x1 > image_w or bbox.y1 > image_h:
        raise ValueError("mask bounding box exceeds source image dimensions")
    if not sample_id or not candidate_id or not profile_name or not profile_fingerprint:
        raise ValueError("mask identity and profile fields must not be empty")

    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "profile_name": profile_name,
        "profile_fingerprint": profile_fingerprint,
        "implementation_version": implementation_version,
        "metrics": metrics.to_dict(),
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                mask=mask_array,
                bbox_xyxy=np.asarray(bbox.as_tuple(), dtype=np.int64),
                image_shape_yx=np.asarray(image_shape_yx, dtype=np.int64),
                metadata_json=np.asarray(metadata_json),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_candidate_mask(path: Path) -> PersistedMask:
    """Load and validate a cropped mask archive without pickle support."""

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"mask", "bbox_xyxy", "image_shape_yx", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"candidate mask archive missing fields: {sorted(missing)}")
        mask = np.asarray(archive["mask"], dtype=np.bool_)
        bbox_values = tuple(int(value) for value in archive["bbox_xyxy"].tolist())
        image_shape = tuple(int(value) for value in archive["image_shape_yx"].tolist())
        metadata = json.loads(str(archive["metadata_json"].item()))

    if len(bbox_values) != 4 or len(image_shape) != 2:
        raise ValueError("candidate mask archive has invalid geometry")
    return PersistedMask(
        mask=mask,
        bbox=BoundingBox(*bbox_values),
        image_shape_yx=image_shape,
        metadata=metadata,
    )
