"""Spatial quality-control plots and duplicate diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class SpatialQcResult:
    overview_path: Path
    duplicate_suspects: pd.DataFrame


@dataclass(frozen=True)
class BatchSpatialQcResult:
    qc_dir: Path
    overview_index_path: Path
    duplicate_suspects_path: Path
    overview_atlas_path: Path | None


def accepted_duplicate_suspects(
    candidates: pd.DataFrame,
    *,
    pixel_size_um: float,
    overlap_fraction_threshold: float = 0.25,
) -> pd.DataFrame:
    """Flag accepted pairs whose diameter-derived circles substantially overlap."""

    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    accepted = candidates[candidates["accepted"].astype(bool)]
    rows = []
    for left, right in combinations(accepted.to_dict("records"), 2):
        left_radius_px = (
            0.5 * float(left["local_equivalent_diameter_um"]) / pixel_size_um
        )
        right_radius_px = (
            0.5 * float(right["local_equivalent_diameter_um"]) / pixel_size_um
        )
        distance_px = float(
            np.hypot(
                float(left["center_x"]) - float(right["center_x"]),
                float(left["center_y"]) - float(right["center_y"]),
            )
        )
        overlap_px = max(0.0, left_radius_px + right_radius_px - distance_px)
        overlap_fraction = overlap_px / max(
            min(left_radius_px, right_radius_px),
            1e-6,
        )
        if overlap_fraction < overlap_fraction_threshold:
            continue
        rows.append(
            {
                "detector_component_id_a": str(left["detector_component_id"]),
                "detector_component_id_b": str(right["detector_component_id"]),
                "center_x_a": float(left["center_x"]),
                "center_y_a": float(left["center_y"]),
                "center_x_b": float(right["center_x"]),
                "center_y_b": float(right["center_y"]),
                "diameter_um_a": float(left["local_equivalent_diameter_um"]),
                "diameter_um_b": float(right["local_equivalent_diameter_um"]),
                "pair_distance_px": distance_px,
                "radius_px_a": left_radius_px,
                "radius_px_b": right_radius_px,
                "overlap_fraction_smaller": overlap_fraction,
            }
        )
    columns = [
        "detector_component_id_a",
        "detector_component_id_b",
        "center_x_a",
        "center_y_a",
        "center_x_b",
        "center_y_b",
        "diameter_um_a",
        "diameter_um_b",
        "pair_distance_px",
        "radius_px_a",
        "radius_px_b",
        "overlap_fraction_smaller",
    ]
    output = pd.DataFrame(rows, columns=columns)
    if not output.empty:
        output = output.sort_values(
            "overlap_fraction_smaller",
            ascending=False,
        ).reset_index(drop=True)
    return output


def render_spatial_overview(
    downsampled_uchl1: np.ndarray,
    candidates: pd.DataFrame,
    *,
    downsample_factor: int,
    pixel_size_um: float,
    sample_id: str,
    out_path: Path,
) -> SpatialQcResult:
    """Render whole-slide candidate locations over the raw UCHL1 mean map."""

    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive")
    background = np.log1p(np.asarray(downsampled_uchl1, dtype=np.float32))
    accepted = candidates[candidates["accepted"].astype(bool)].copy()
    rejected = candidates[~candidates["accepted"].astype(bool)].copy()
    duplicates = accepted_duplicate_suspects(
        candidates,
        pixel_size_um=pixel_size_um,
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis in axes:
        axis.imshow(background, cmap="gray", interpolation="nearest")
        axis.set_axis_off()

    axes[0].set_title(
        f"{sample_id} raw UCHL1 proposals\n"
        f"accepted={len(accepted)} rejected={len(rejected)}"
    )
    if not rejected.empty:
        axes[0].scatter(
            rejected["center_x"] / downsample_factor,
            rejected["center_y"] / downsample_factor,
            s=7,
            c="#7f8c8d",
            alpha=0.45,
            linewidths=0,
            label="rejected",
        )
    if not accepted.empty:
        axes[0].scatter(
            accepted["center_x"] / downsample_factor,
            accepted["center_y"] / downsample_factor,
            s=16,
            c="#00d4c8",
            edgecolors="#102a2a",
            linewidths=0.3,
            label="accepted",
        )
        axes[0].legend(loc="lower right", framealpha=0.8)

    axes[1].set_title(
        f"Accepted mask-scale circles\n"
        f"duplicate suspects={len(duplicates)}"
    )
    score_min = float(accepted["detector_score"].min()) if not accepted.empty else 0.0
    score_max = float(accepted["detector_score"].max()) if not accepted.empty else 1.0
    score_span = max(score_max - score_min, 1e-6)
    for record in accepted.to_dict("records"):
        score = float(record["detector_score"])
        normalized = (score - score_min) / score_span
        color = plt.cm.turbo(0.15 + 0.75 * normalized)
        radius_ds = (
            0.5
            * float(record["local_equivalent_diameter_um"])
            / pixel_size_um
            / downsample_factor
        )
        axes[1].add_patch(
            Circle(
                (
                    float(record["center_x"]) / downsample_factor,
                    float(record["center_y"]) / downsample_factor,
                ),
                radius=max(radius_ds, 1.0),
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.9,
            )
        )
    for record in duplicates.to_dict("records"):
        axes[1].plot(
            [
                float(record["center_x_a"]) / downsample_factor,
                float(record["center_x_b"]) / downsample_factor,
            ],
            [
                float(record["center_y_a"]) / downsample_factor,
                float(record["center_y_b"]) / downsample_factor,
            ],
            color="#ff2e63",
            linewidth=0.8,
            alpha=0.8,
        )
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return SpatialQcResult(
        overview_path=destination,
        duplicate_suspects=duplicates,
    )


def compile_batch_spatial_qc(
    batch_dir: Path,
    summary: pd.DataFrame,
    *,
    atlas_columns: int = 2,
    thumbnail_size: tuple[int, int] = (900, 420),
) -> BatchSpatialQcResult:
    """Build a batch index, duplicate table, and overview contact sheet."""

    if atlas_columns <= 0:
        raise ValueError("atlas_columns must be positive")
    batch_root = Path(batch_dir)
    qc_dir = batch_root / "spatial_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    duplicate_tables = []
    overview_paths = []
    complete = summary[summary["status"] == "complete"]
    for record in complete.to_dict("records"):
        sample_id = str(record["sample_id"])
        sample_dir = batch_root / sample_id
        overview_path = sample_dir / "overview.png"
        duplicates_path = sample_dir / "accepted_duplicate_suspects.csv"
        duplicates = pd.read_csv(duplicates_path)
        if not duplicates.empty:
            duplicates.insert(0, "sample_id", sample_id)
            duplicate_tables.append(duplicates)
        index_rows.append(
            {
                "sample_id": sample_id,
                "status": "complete",
                "accepted_candidate_count": int(record["accepted_candidate_count"]),
                "duplicate_suspect_count": int(len(duplicates)),
                "overview_path": str(overview_path),
                "duplicate_suspects_path": str(duplicates_path),
            }
        )
        overview_paths.append((sample_id, overview_path))

    index_path = qc_dir / "overview_index.csv"
    duplicate_path = qc_dir / "accepted_duplicate_suspects.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    duplicate_columns = [
        "sample_id",
        "detector_component_id_a",
        "detector_component_id_b",
        "center_x_a",
        "center_y_a",
        "center_x_b",
        "center_y_b",
        "diameter_um_a",
        "diameter_um_b",
        "pair_distance_px",
        "radius_px_a",
        "radius_px_b",
        "overlap_fraction_smaller",
    ]
    combined_duplicates = (
        pd.concat(duplicate_tables, ignore_index=True)
        if duplicate_tables
        else pd.DataFrame(columns=duplicate_columns)
    )
    combined_duplicates.to_csv(duplicate_path, index=False)

    atlas_path = None
    if overview_paths:
        cell_w, cell_h = thumbnail_size
        atlas_rows = math.ceil(len(overview_paths) / atlas_columns)
        atlas = Image.new(
            "RGB",
            (cell_w * atlas_columns, cell_h * atlas_rows),
            color="white",
        )
        for index, (_, overview_path) in enumerate(overview_paths):
            with Image.open(overview_path) as image:
                tile = image.convert("RGB")
                tile.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                x0 = (index % atlas_columns) * cell_w + (cell_w - tile.width) // 2
                y0 = (index // atlas_columns) * cell_h + (cell_h - tile.height) // 2
                atlas.paste(tile, (x0, y0))
        atlas_path = qc_dir / "all_samples_overview.png"
        atlas.save(atlas_path, format="PNG", optimize=True)
    return BatchSpatialQcResult(
        qc_dir=qc_dir,
        overview_index_path=index_path,
        duplicate_suspects_path=duplicate_path,
        overview_atlas_path=atlas_path,
    )
