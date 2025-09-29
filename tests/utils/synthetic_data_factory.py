"""Utilities for generating synthetic CODEX-like patches for testing.

The helpers here intentionally keep the construction explicit so tests can
reason about the exact pixel allocations and signal statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from skimage.segmentation import find_boundaries


ChannelMap = Mapping[str, float]


@dataclass(frozen=True)
class SyntheticPatch:
    """Container holding synthetic patch data and expected statistics."""

    image_dict: Dict[str, np.ndarray]
    nucleus_mask: np.ndarray
    cell_mask: np.ndarray
    channels: Tuple[str, ...]
    expected_means: Dict[int, Dict[str, Dict[str, float]]]

    def cytoplasm_mask(self) -> np.ndarray:
        """Return boolean mask where whole-cell minus nucleus."""
        return np.logical_and(self.cell_mask > 0, self.nucleus_mask == 0)


def _validate_channels(channels: Sequence[str]) -> Tuple[str, ...]:
    if not channels:
        raise ValueError("At least one channel must be provided.")
    if len({ch.lower() for ch in channels}) != len(channels):
        raise ValueError("Channel names must be unique ignoring case for synthetic data.")
    return tuple(channels)


def _ensure_intensity_map(
    channels: Tuple[str, ...],
    nucleus_intensity: Optional[ChannelMap],
    cytoplasm_intensity: Optional[ChannelMap],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    nucleus_defaults = {ch: 10.0 for ch in channels}
    cytoplasm_defaults = {ch: 3.0 for ch in channels}

    if nucleus_intensity:
        nucleus_defaults.update(nucleus_intensity)
    if cytoplasm_intensity:
        cytoplasm_defaults.update(cytoplasm_intensity)

    return nucleus_defaults, cytoplasm_defaults


def make_disk_cells_patch(
    *,
    shape: Tuple[int, int] = (32, 32),
    cells: Optional[Sequence[Mapping[str, object]]] = None,
    channels: Sequence[str] = ("chan0", "chan1"),
    background_intensity: Optional[ChannelMap] = None,
) -> SyntheticPatch:
    """Build a synthetic patch using circular nuclei and whole-cell regions.

    Parameters
    ----------
    shape:
        Height and width of the synthetic patch.
    cells:
        Sequence of dict definitions. Each entry may contain:
            - label: int (defaults to incrementing from 1)
            - center: Tuple[int, int] (y, x) center of the cell
            - nucleus_radius: float
            - cell_radius: float (must be >= nucleus_radius)
            - nucleus_intensity: mapping channel->value
            - cytoplasm_intensity: mapping channel->value
    channels:
        Names of channels to synthesise.
    background_intensity:
        Optional mapping of channel->value for the background (default 0).
    """

    channels_tuple = _validate_channels(channels)
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape must be positive in both dimensions")

    yy, xx = np.indices(shape)

    if not cells:
        # Default: single well separated cell.
        center = (height // 2, width // 2)
        cells = [
            {
                "label": 1,
                "center": center,
                "nucleus_radius": min(height, width) * 0.12,
                "cell_radius": min(height, width) * 0.25,
            }
        ]

    image_dict: Dict[str, np.ndarray] = {
        ch: np.full(shape, (background_intensity or {}).get(ch, 0.0), dtype=np.float32)
        for ch in channels_tuple
    }
    nucleus_mask = np.zeros(shape, dtype=np.uint32)
    cell_mask = np.zeros(shape, dtype=np.uint32)

    expected: Dict[int, Dict[str, Dict[str, float]]] = {}

    for default_label, cell_def in enumerate(cells, start=1):
        label = int(cell_def.get("label", default_label))
        if label <= 0:
            raise ValueError("Cell label must be positive integer")
        if label in expected:
            raise ValueError(f"Duplicate cell label detected: {label}")

        center = cell_def.get("center", (height // 2, width // 2))
        cy, cx = center
        nucleus_radius = float(cell_def.get("nucleus_radius", min(height, width) * 0.12))
        cell_radius = float(cell_def.get("cell_radius", nucleus_radius * 1.8))
        if cell_radius < nucleus_radius:
            raise ValueError("cell_radius must be >= nucleus_radius")

        nucleus_intensity_map, cytoplasm_intensity_map = _ensure_intensity_map(
            channels_tuple,
            cell_def.get("nucleus_intensity"),
            cell_def.get("cytoplasm_intensity"),
        )

        distances = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        nucleus_area = distances <= nucleus_radius
        cell_area = distances <= cell_radius

        # Update masks - later cells override overlaps to keep determinism.
        nucleus_mask[nucleus_area] = label
        cell_mask[cell_area] = label

        # Paint channel intensities (cytoplasm first, nucleus overrides).
        for ch in channels_tuple:
            img = image_dict[ch]
            if cell_area.any():
                img[cell_area] = cytoplasm_intensity_map[ch]
            if nucleus_area.any():
                img[nucleus_area] = nucleus_intensity_map[ch]

        expected[label] = {}

        nucleus_pixels = nucleus_area
        cell_pixels = cell_area
        cytoplasm_pixels = np.logical_and(cell_pixels, np.logical_not(nucleus_pixels))

        for ch in channels_tuple:
            metrics: Dict[str, float] = {}
            # Whole-cell mean over cell mask.
            if cell_pixels.any():
                metrics["wholecell_mean"] = float(np.mean(image_dict[ch][cell_pixels]))
            else:
                metrics["wholecell_mean"] = float("nan")
            # Nucleus mean.
            if nucleus_pixels.any():
                metrics["nucleus_mean"] = float(np.mean(image_dict[ch][nucleus_pixels]))
            else:
                metrics["nucleus_mean"] = float("nan")
            # Cytoplasm mean.
            if cytoplasm_pixels.any():
                metrics["cytoplasm_mean"] = float(
                    np.mean(image_dict[ch][cytoplasm_pixels])
                )
            else:
                metrics["cytoplasm_mean"] = float("nan")

            expected[label][ch] = metrics

    return SyntheticPatch(
        image_dict=image_dict,
        nucleus_mask=nucleus_mask,
        cell_mask=cell_mask,
        channels=channels_tuple,
        expected_means=expected,
    )


def make_single_cell_patch(
    *,
    shape: Tuple[int, int] = (32, 32),
    channels: Sequence[str] = ("chan0", "chan1"),
    nucleus_intensity: Optional[ChannelMap] = None,
    cytoplasm_intensity: Optional[ChannelMap] = None,
) -> SyntheticPatch:
    """Convenience wrapper for a single centred cell."""
    return make_disk_cells_patch(
        shape=shape,
        channels=channels,
        cells=[
            {
                "label": 1,
                "center": (shape[0] // 2, shape[1] // 2),
                "nucleus_radius": min(shape) * 0.12,
                "cell_radius": min(shape) * 0.25,
                "nucleus_intensity": nucleus_intensity or {},
                "cytoplasm_intensity": cytoplasm_intensity or {},
            }
        ],
    )


def make_nucleus_only_patch(
    *,
    shape: Tuple[int, int] = (32, 32),
    channels: Sequence[str] = ("chan0", "chan1"),
    nucleus_intensity: Optional[ChannelMap] = None,
) -> SyntheticPatch:
    """Create a patch where whole-cell matches the nucleus (no cytoplasm)."""
    return make_disk_cells_patch(
        shape=shape,
        channels=channels,
        cells=[
            {
                "label": 1,
                "center": (shape[0] // 2, shape[1] // 2),
                "nucleus_radius": min(shape) * 0.2,
                "cell_radius": min(shape) * 0.2,
                "nucleus_intensity": nucleus_intensity or {},
                "cytoplasm_intensity": nucleus_intensity or {},
            }
        ],
    )


def make_empty_patch(
    *,
    shape: Tuple[int, int] = (32, 32),
    channels: Sequence[str] = ("chan0", "chan1"),
    background_intensity: Optional[ChannelMap] = None,
) -> SyntheticPatch:
    """Return an empty patch with zero cells."""
    channels_tuple = _validate_channels(channels)
    img = {
        ch: np.full(shape, (background_intensity or {}).get(ch, 0.0), dtype=np.float32)
        for ch in channels_tuple
    }
    nucleus_mask = np.zeros(shape, dtype=np.uint32)
    cell_mask = np.zeros(shape, dtype=np.uint32)
    return SyntheticPatch(
        image_dict=img,
        nucleus_mask=nucleus_mask,
        cell_mask=cell_mask,
        channels=channels_tuple,
        expected_means={},
    )



def make_segmentation_result(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    *,
    include_boundaries: bool = True,
) -> Dict[str, np.ndarray]:
    """Build a segmentation result dictionary compatible with repair helpers."""
    result = {
        "cell": np.asarray(cell_mask, dtype=np.uint32),
        "nucleus": np.asarray(nucleus_mask, dtype=np.uint32),
    }

    if include_boundaries:
        cell_boundary_bool = find_boundaries(cell_mask, mode="inner")
        nucleus_boundary_bool = find_boundaries(nucleus_mask, mode="inner")
        result["cell_boundary"] = np.where(
            cell_boundary_bool, result["cell"], 0
        ).astype(np.uint32)
        result["nucleus_boundary"] = np.where(
            nucleus_boundary_bool, result["nucleus"], 0
        ).astype(np.uint32)

    return result


def render_patch(
    patch: SyntheticPatch,
    output_path: Path,
    *,
    overlay_masks: bool = True,
    cmap: str = "viridis",
    dpi: int = 200,
) -> None:
    """Save a quicklook image for manual inspection of a synthetic patch."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for render_patch") from exc

    channels = patch.channels
    num_panels = len(channels) + (2 if overlay_masks else 0)
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))
    if num_panels == 1:
        axes = [axes]

    for idx, channel in enumerate(channels):
        ax = axes[idx]
        im = ax.imshow(patch.image_dict[channel], cmap=cmap)
        ax.set_title(f"{channel} intensity")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    panel_idx = len(channels)
    if overlay_masks:
        cell_ax = axes[panel_idx]
        cell_ax.imshow(patch.cell_mask, cmap="nipy_spectral")
        cell_ax.set_title("cell mask")
        cell_ax.axis("off")
        panel_idx += 1

        nucleus_ax = axes[panel_idx]
        nucleus_ax.imshow(patch.nucleus_mask, cmap="nipy_spectral")
        nucleus_ax.set_title("nucleus mask")
        nucleus_ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

