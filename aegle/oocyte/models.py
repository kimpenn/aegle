"""Typed values shared by standalone oocyte modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Half-open image-space bounding box in X/Y order."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError("bounding box coordinates must be non-negative")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive width and height")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def shape_yx(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class ExtractedPatch:
    """A fixed-size image patch plus its clipped full-image geometry."""

    image: np.ndarray
    bbox: BoundingBox
    image_shape_yx: Tuple[int, int]
    padding_tblr: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.image.ndim != 2:
            raise ValueError("extracted patch must be two-dimensional")
        top, bottom, left, right = self.padding_tblr
        if min(top, bottom, left, right) < 0:
            raise ValueError("patch padding must be non-negative")
        expected_shape = (
            self.bbox.height + top + bottom,
            self.bbox.width + left + right,
        )
        if self.image.shape != expected_shape:
            raise ValueError(
                f"patch shape {self.image.shape} does not match geometry {expected_shape}"
            )

    def crop_to_image_bounds(self, array: np.ndarray) -> np.ndarray:
        """Remove edge padding from a patch-aligned array."""

        if array.shape[:2] != self.image.shape:
            raise ValueError("array must share the extracted patch Y/X shape")
        top, bottom, left, right = self.padding_tblr
        y1 = array.shape[0] - bottom if bottom else array.shape[0]
        x1 = array.shape[1] - right if right else array.shape[1]
        return array[top:y1, left:x1]


@dataclass(frozen=True)
class SegmentationMetrics:
    threshold_method: str
    base_threshold: float
    annulus_floor: float
    threshold: float
    selection_mode: str
    area_px: int
    equivalent_diameter_um: float
    major_axis_um: float
    minor_axis_um: float
    eccentricity: float
    solidity: float
    circularity: float
    centroid_y_px: float
    centroid_x_px: float
    centroid_offset_px: float
    mean_intensity: float
    max_intensity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalSegmentationResult:
    mask: np.ndarray
    metrics: SegmentationMetrics

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("segmentation mask must be a two-dimensional boolean array")
        if not self.mask.any():
            raise ValueError("segmentation mask must contain at least one foreground pixel")
        if int(self.mask.sum()) != self.metrics.area_px:
            raise ValueError("segmentation mask area does not match metrics")


@dataclass(frozen=True)
class ScoredCandidateMask:
    """Image-bounded mask and metrics retained for one refined candidate."""

    mask: np.ndarray
    bbox: BoundingBox
    image_shape_yx: Tuple[int, int]
    metrics: SegmentationMetrics

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("candidate mask must be a two-dimensional boolean array")
        if self.mask.shape != self.bbox.shape_yx:
            raise ValueError("candidate mask shape must match its image-space bounding box")
        image_h, image_w = self.image_shape_yx
        if self.bbox.x1 > image_w or self.bbox.y1 > image_h:
            raise ValueError("candidate mask bounding box exceeds the source image")


@dataclass(frozen=True)
class PersistedMask:
    mask: np.ndarray
    bbox: BoundingBox
    image_shape_yx: Tuple[int, int]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("persisted mask must be a two-dimensional boolean array")
        if self.mask.shape != self.bbox.shape_yx:
            raise ValueError("persisted mask shape must match its image-space bounding box")
        image_h, image_w = self.image_shape_yx
        if self.bbox.x1 > image_w or self.bbox.y1 > image_h:
            raise ValueError("persisted mask bounding box exceeds the source image")
