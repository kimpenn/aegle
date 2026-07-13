"""Derived whole-slide label exports built from persisted candidate masks."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tifffile

from .io import load_candidate_mask


@dataclass(frozen=True)
class LabelExportResult:
    image_path: Path
    mapping_path: Path
    label_count: int
    assigned_pixel_count: int
    overlap_pixel_count: int


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


def export_whole_slide_labels(
    candidates: pd.DataFrame,
    *,
    sample_dir: Path,
    image_shape_yx: Tuple[int, int],
    image_path: Path,
    mapping_path: Path,
    tile_shape_yx: Tuple[int, int] = (512, 512),
) -> LabelExportResult:
    """Compose accepted masks into one sparse whole-slide label OME-TIFF."""

    required = {
        "detector_component_id",
        "accepted",
        "detector_score",
        "acceptance_mode",
        "center_x",
        "center_y",
        "mask_path",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidate table missing label-export columns: {sorted(missing)}")
    image_h, image_w = (int(image_shape_yx[0]), int(image_shape_yx[1]))
    if image_h <= 0 or image_w <= 0:
        raise ValueError("whole-slide label dimensions must be positive")
    if any(value <= 0 or value % 16 for value in tile_shape_yx):
        raise ValueError("TIFF tile dimensions must be positive multiples of 16")

    accepted = candidates[candidates["accepted"].astype(bool)].copy()
    accepted = accepted.sort_values(
        ["detector_score", "detector_component_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    if len(accepted) > np.iinfo(np.uint16).max:
        raise ValueError("uint16 label export supports at most 65535 accepted candidates")

    destination = Path(image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_path = None
    temporary_tiff = None
    mapping_rows = []
    total_assigned = 0
    total_overlap = 0
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".labels.raw",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            raw_path = Path(handle.name)
            handle.truncate(image_h * image_w * np.dtype(np.uint16).itemsize)
        labels = np.memmap(
            raw_path,
            mode="r+",
            dtype=np.uint16,
            shape=(image_h, image_w),
        )
        labels[:] = 0

        sample_root = Path(sample_dir)
        for label_value, record in enumerate(accepted.to_dict("records"), start=1):
            candidate_mask_path = Path(str(record["mask_path"]))
            if not candidate_mask_path.is_absolute():
                candidate_mask_path = sample_root / candidate_mask_path
            persisted = load_candidate_mask(candidate_mask_path)
            if persisted.image_shape_yx != (image_h, image_w):
                raise ValueError(
                    f"candidate mask image shape mismatch: {candidate_mask_path}"
                )
            bbox = persisted.bbox
            region = labels[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
            foreground = persisted.mask
            overlap = foreground & (region != 0)
            writable = foreground & (region == 0)
            overlap_count = int(overlap.sum())
            assigned_count = int(writable.sum())
            region[writable] = label_value
            total_overlap += overlap_count
            total_assigned += assigned_count
            mapping_rows.append(
                {
                    "label": int(label_value),
                    "detector_component_id": str(record["detector_component_id"]),
                    "detector_score": float(record["detector_score"]),
                    "acceptance_mode": str(record["acceptance_mode"]),
                    "center_x": int(record["center_x"]),
                    "center_y": int(record["center_y"]),
                    "bbox_x0": bbox.x0,
                    "bbox_y0": bbox.y0,
                    "bbox_x1": bbox.x1,
                    "bbox_y1": bbox.y1,
                    "mask_path": str(record["mask_path"]),
                    "assigned_pixel_count": assigned_count,
                    "overlap_pixel_count": overlap_count,
                }
            )
        labels.flush()

        with tempfile.NamedTemporaryFile(
            suffix=".ome.tiff",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_tiff = Path(handle.name)
        tifffile.imwrite(
            temporary_tiff,
            labels,
            dtype=np.uint16,
            bigtiff=True,
            ome=True,
            metadata={"axes": "YX"},
            photometric="minisblack",
            tile=tile_shape_yx,
            compression="zlib",
        )
        del labels
        temporary_tiff.replace(destination)
        temporary_tiff = None
    finally:
        if temporary_tiff is not None and temporary_tiff.exists():
            temporary_tiff.unlink()
        if raw_path is not None and raw_path.exists():
            raw_path.unlink()

    mapping_columns = [
        "label",
        "detector_component_id",
        "detector_score",
        "acceptance_mode",
        "center_x",
        "center_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "mask_path",
        "assigned_pixel_count",
        "overlap_pixel_count",
    ]
    mapping = pd.DataFrame(mapping_rows, columns=mapping_columns)
    _atomic_write_csv(mapping, Path(mapping_path))
    return LabelExportResult(
        image_path=destination,
        mapping_path=Path(mapping_path),
        label_count=len(mapping),
        assigned_pixel_count=total_assigned,
        overlap_pixel_count=total_overlap,
    )
