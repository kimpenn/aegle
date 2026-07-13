"""Persisted-mask review montages for standalone oocyte detection outputs."""

from __future__ import annotations

import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr

from .io import extract_cyx_channel_patch, load_candidate_mask
from .models import ExtractedPatch, PersistedMask


@dataclass(frozen=True)
class ReviewPackResult:
    review_dir: Path
    accepted_count: int
    novel_count: int | None
    missed_reference_count: int | None
    artifact_paths: Dict[str, Path]


@dataclass(frozen=True)
class _SampleSource:
    sample_id: str
    sample_dir: Path
    image_path: Path
    channel_index: int


def _load_sample_sources(batch_dir: Path) -> List[_SampleSource]:
    summary_path = batch_dir / "batch_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"batch summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    sources = []
    for record in summary.to_dict("records"):
        if str(record["status"]) != "complete":
            continue
        sample_id = str(record["sample_id"])
        sample_dir = batch_dir / sample_id
        manifest_path = sample_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        sources.append(
            _SampleSource(
                sample_id=sample_id,
                sample_dir=sample_dir,
                image_path=Path(manifest["source_image"]),
                channel_index=int(manifest["resolved_channel_index"]),
            )
        )
    return sources


def _load_candidates(sources: List[_SampleSource]) -> pd.DataFrame:
    tables = []
    for source in sources:
        table = pd.read_csv(source.sample_dir / "candidates.csv")
        table.insert(0, "sample_id", source.sample_id)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _load_references(
    path: Path,
    sample_ids: set[str],
    min_final_score: float,
) -> pd.DataFrame:
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if sample_id not in sample_ids:
                continue
            final_score = float(record.get("final_score", 0.0))
            if final_score < min_final_score:
                continue
            center = record.get("center")
            if center is None or len(center) != 2:
                raise ValueError(
                    f"reference line {line_number} has no two-coordinate center"
                )
            records.append(
                {
                    "sample_id": sample_id,
                    "reference_oocyte_id": int(record["oocyte_id"]),
                    "reference_center_x": float(center[0]),
                    "reference_center_y": float(center[1]),
                    "reference_final_score": final_score,
                    "reference_quality": str(record.get("quality", "")),
                }
            )
    return pd.DataFrame(records)


def _attach_reference_fields(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
    match_radius_px: float,
) -> pd.DataFrame:
    output = candidates.copy()
    output["nearest_reference_oocyte_id"] = np.nan
    output["nearest_reference_distance_px"] = np.nan
    output["nearest_reference_final_score"] = np.nan
    output["nearest_reference_quality"] = ""
    output["matches_reference_within_radius"] = False
    if output.empty or references.empty:
        return output

    for sample_id, index in output.groupby("sample_id").groups.items():
        sample_refs = references[references["sample_id"] == sample_id]
        if sample_refs.empty:
            continue
        ref_xy = sample_refs[["reference_center_x", "reference_center_y"]].to_numpy()
        candidate_xy = output.loc[index, ["center_x", "center_y"]].to_numpy()
        distances = np.sqrt(
            ((candidate_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2)
        )
        nearest_indices = distances.argmin(axis=1)
        nearest_distances = distances[np.arange(len(index)), nearest_indices]
        nearest = sample_refs.iloc[nearest_indices]
        output.loc[index, "nearest_reference_oocyte_id"] = nearest[
            "reference_oocyte_id"
        ].to_numpy()
        output.loc[index, "nearest_reference_distance_px"] = nearest_distances
        output.loc[index, "nearest_reference_final_score"] = nearest[
            "reference_final_score"
        ].to_numpy()
        output.loc[index, "nearest_reference_quality"] = nearest[
            "reference_quality"
        ].to_numpy()
        output.loc[index, "matches_reference_within_radius"] = (
            nearest_distances <= match_radius_px
        )
    return output


def _reference_comparison(
    references: pd.DataFrame,
    candidates: pd.DataFrame,
    match_radius_px: float,
) -> pd.DataFrame:
    rows = []
    accepted = candidates[candidates["accepted"].astype(bool)]
    for reference in references.to_dict("records"):
        sample_candidates = candidates[candidates["sample_id"] == reference["sample_id"]]
        sample_accepted = accepted[accepted["sample_id"] == reference["sample_id"]]

        def nearest(table: pd.DataFrame) -> Dict[str, Any] | None:
            if table.empty:
                return None
            distances = np.hypot(
                table["center_x"].to_numpy(dtype=float)
                - reference["reference_center_x"],
                table["center_y"].to_numpy(dtype=float)
                - reference["reference_center_y"],
            )
            position = int(distances.argmin())
            row = table.iloc[position]
            return {
                "component_id": str(row["detector_component_id"]),
                "distance_px": float(distances[position]),
                "detector_score": float(row["detector_score"]),
                "accepted": bool(row["accepted"]),
            }

        nearest_refined = nearest(sample_candidates)
        nearest_accepted = nearest(sample_accepted)
        rows.append(
            {
                **reference,
                "nearest_refined_component_id": None
                if nearest_refined is None
                else nearest_refined["component_id"],
                "nearest_refined_distance_px": None
                if nearest_refined is None
                else nearest_refined["distance_px"],
                "nearest_refined_detector_score": None
                if nearest_refined is None
                else nearest_refined["detector_score"],
                "nearest_refined_accepted": False
                if nearest_refined is None
                else nearest_refined["accepted"],
                "matched_by_refined": bool(
                    nearest_refined is not None
                    and nearest_refined["distance_px"] <= match_radius_px
                ),
                "nearest_accepted_component_id": None
                if nearest_accepted is None
                else nearest_accepted["component_id"],
                "nearest_accepted_distance_px": None
                if nearest_accepted is None
                else nearest_accepted["distance_px"],
                "nearest_accepted_detector_score": None
                if nearest_accepted is None
                else nearest_accepted["detector_score"],
                "matched_by_accepted": bool(
                    nearest_accepted is not None
                    and nearest_accepted["distance_px"] <= match_radius_px
                ),
            }
        )
    return pd.DataFrame(rows)


def _candidate_review_table(
    candidates: pd.DataFrame,
    *,
    novel_only: bool,
    references_available: bool,
) -> pd.DataFrame:
    table = candidates[candidates["accepted"].astype(bool)].copy()
    if novel_only:
        table = table[~table["matches_reference_within_radius"].astype(bool)].copy()
    if table.empty:
        return table
    table = table.sort_values(
        ["detector_score", "mean_to_annulus_p99_ratio"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "review_rank", np.arange(1, len(table) + 1))
    if references_available:
        table["review_bucket"] = np.where(
            table["matches_reference_within_radius"],
            "near_reference",
            "novel",
        )
    else:
        table["review_bucket"] = "accepted"
    table["manual_is_oocyte"] = ""
    table["manual_mask_quality"] = ""
    table["manual_duplicate_group"] = ""
    table["manual_notes"] = ""
    return table


def _missed_review_table(reference_comparison: pd.DataFrame) -> pd.DataFrame:
    table = reference_comparison[
        ~reference_comparison["matched_by_accepted"].astype(bool)
    ].copy()
    if table.empty:
        return table
    table = table.sort_values(
        ["reference_final_score", "reference_quality"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "review_rank", np.arange(1, len(table) + 1))
    table["center_x"] = table["reference_center_x"].round().astype(int)
    table["center_y"] = table["reference_center_y"].round().astype(int)
    table["manual_true_oocyte_at_reference"] = ""
    table["manual_failure_mode"] = ""
    table["manual_duplicate_group"] = ""
    table["manual_notes"] = ""
    return table


def _place_mask_in_patch(
    persisted: PersistedMask,
    patch: ExtractedPatch,
) -> np.ndarray | None:
    ix0 = max(persisted.bbox.x0, patch.bbox.x0)
    iy0 = max(persisted.bbox.y0, patch.bbox.y0)
    ix1 = min(persisted.bbox.x1, patch.bbox.x1)
    iy1 = min(persisted.bbox.y1, patch.bbox.y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return None
    top, _, left, _ = patch.padding_tblr
    overlay = np.zeros(patch.image.shape, dtype=np.bool_)
    target_x0 = left + ix0 - patch.bbox.x0
    target_y0 = top + iy0 - patch.bbox.y0
    source_x0 = ix0 - persisted.bbox.x0
    source_y0 = iy0 - persisted.bbox.y0
    width = ix1 - ix0
    height = iy1 - iy0
    overlay[target_y0 : target_y0 + height, target_x0 : target_x0 + width] = (
        persisted.mask[
            source_y0 : source_y0 + height,
            source_x0 : source_x0 + width,
        ]
    )
    return overlay if overlay.any() else None


def _format_float(value: Any, digits: int, missing: str = "n/a") -> str:
    if value is None or pd.isna(value):
        return missing
    return f"{float(value):.{digits}f}"


def _render_pages(
    table: pd.DataFrame,
    *,
    mode: str,
    pages_dir: Path,
    page_title: str,
    columns: int,
    rows: int,
    sources: Dict[str, _SampleSource],
    arrays: Dict[str, Any],
    accepted_by_sample: Dict[str, pd.DataFrame],
    mask_cache: Dict[Path, PersistedMask],
    neighbor_overlays: bool,
    max_neighbor_overlays: int,
    neighbor_margin_px: float,
    missed_patch_radius_px: int,
) -> pd.DataFrame:
    output = table.copy()
    if output.empty:
        return output
    pages_dir.mkdir(parents=True, exist_ok=True)
    per_page = columns * rows
    output["page_number"] = np.arange(len(output)) // per_page + 1
    output["panel_index"] = np.arange(len(output)) % per_page + 1
    output["page_path"] = output["page_number"].map(
        lambda page: str(pages_dir / f"page_{int(page):02d}.png")
    )
    rank_lookup = {
        (str(row["sample_id"]), str(row["detector_component_id"])): int(
            row["review_rank"]
        )
        for row in output.to_dict("records")
        if "detector_component_id" in row
    }

    def persisted_mask(sample_dir: Path, value: Any) -> PersistedMask:
        path = Path(str(value))
        if not path.is_absolute():
            path = sample_dir / path
        path = path.resolve()
        if path not in mask_cache:
            mask_cache[path] = load_candidate_mask(path)
        return mask_cache[path]

    for page_number in range(1, math.ceil(len(output) / per_page) + 1):
        page = output[output["page_number"] == page_number]
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.3 * columns, 4.4 * rows),
            constrained_layout=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()
        for ax in axes_flat:
            ax.set_axis_off()
        for ax, record in zip(axes_flat, page.to_dict("records")):
            sample_id = str(record["sample_id"])
            source = sources[sample_id]
            radius = (
                int(record.get("evaluation_window_radius_px", missed_patch_radius_px))
                if mode == "candidate"
                else missed_patch_radius_px
            )
            center_x = int(record["center_x"])
            center_y = int(record["center_y"])
            patch = extract_cyx_channel_patch(
                arrays[sample_id],
                source.channel_index,
                (center_x, center_y),
                radius,
            )
            current_mask = None
            if mode == "candidate":
                current_mask = _place_mask_in_patch(
                    persisted_mask(source.sample_dir, record["mask_path"]),
                    patch,
                )
            overlays = []
            if neighbor_overlays:
                neighbors = accepted_by_sample.get(sample_id, pd.DataFrame()).copy()
                if mode == "candidate" and not neighbors.empty:
                    neighbors = neighbors[
                        neighbors["detector_component_id"]
                        != record["detector_component_id"]
                    ]
                if not neighbors.empty:
                    distances = np.hypot(
                        neighbors["center_x"].to_numpy(dtype=float) - center_x,
                        neighbors["center_y"].to_numpy(dtype=float) - center_y,
                    )
                    neighbors = neighbors.assign(_distance_px=distances)
                    neighbors = neighbors[
                        neighbors["_distance_px"] <= radius + neighbor_margin_px
                    ].sort_values("_distance_px")
                for neighbor in neighbors.head(max_neighbor_overlays).to_dict("records"):
                    overlay = _place_mask_in_patch(
                        persisted_mask(source.sample_dir, neighbor["mask_path"]),
                        patch,
                    )
                    if overlay is None:
                        continue
                    rank = rank_lookup.get(
                        (sample_id, str(neighbor["detector_component_id"]))
                    )
                    overlays.append((overlay, "" if rank is None else f"#{rank}"))

            ax.imshow(np.log1p(patch.image), cmap="magma")
            ax.scatter(
                [patch.image.shape[1] // 2],
                [patch.image.shape[0] // 2],
                s=24,
                c="white",
                marker="+",
                linewidths=0.8,
            )
            if current_mask is not None:
                ax.contour(
                    current_mask.astype(np.uint8),
                    levels=[0.5],
                    colors=["cyan"],
                    linewidths=1.2,
                )
            for overlay, label in overlays:
                ax.contour(
                    overlay.astype(np.uint8),
                    levels=[0.5],
                    colors=["#a7f432"],
                    linewidths=0.9,
                )
                if label:
                    label_y, label_x = np.argwhere(overlay).mean(axis=0)
                    ax.text(
                        float(label_x),
                        float(label_y),
                        label,
                        color="#a7f432",
                        fontsize=7,
                        ha="center",
                        va="center",
                        bbox={
                            "boxstyle": "round,pad=0.12",
                            "fc": "#111111",
                            "ec": "none",
                            "alpha": 0.7,
                        },
                    )
            rank = int(record["review_rank"])
            if mode == "candidate":
                title_lines = [
                    f"#{rank:03d} | {sample_id} {record['detector_component_id']}",
                    f"{record['review_bucket']} {record['acceptance_mode']}",
                    f"score={_format_float(record['detector_score'], 3)}",
                    f"d={_format_float(record['local_equivalent_diameter_um'], 1)}um "
                    f"circ={_format_float(record['local_circularity'], 3)}",
                ]
                if "nearest_reference_distance_px" in record:
                    title_lines.append(
                        "refdist="
                        + _format_float(
                            record["nearest_reference_distance_px"],
                            0,
                            missing="none",
                        )
                        + ("px" if not pd.isna(record["nearest_reference_distance_px"]) else "")
                    )
            else:
                title_lines = [
                    f"#{rank:03d} | {sample_id} ref#{int(record['reference_oocyte_id'])}",
                    f"old={_format_float(record['reference_final_score'], 3)} "
                    f"{record['reference_quality']}",
                    "nearest_acc="
                    + _format_float(
                        record.get("nearest_accepted_distance_px"),
                        0,
                        missing="none",
                    )
                    + (
                        "px"
                        if not pd.isna(record.get("nearest_accepted_distance_px"))
                        else ""
                    ),
                ]
            ax.set_title("\n".join(title_lines), fontsize=9, color="#202124")
            ax.set_axis_off()
        fig.suptitle(f"{page_title} | page {page_number}", fontsize=13)
        fig.savefig(
            pages_dir / f"page_{page_number:02d}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)
    return output


def generate_review_pack(
    batch_dir: Path,
    *,
    references_path: Path | None = None,
    reference_min_final_score: float = 0.35,
    match_radius_px: float = 100.0,
    columns: int = 3,
    rows: int = 4,
    neighbor_overlays: bool = True,
    max_neighbor_overlays: int = 8,
    neighbor_margin_px: float = 40.0,
    missed_patch_radius_px: int = 180,
) -> ReviewPackResult:
    """Generate accepted and optional novel/missed review tables and pages."""

    if columns <= 0 or rows <= 0:
        raise ValueError("montage columns and rows must be positive")
    batch_root = Path(batch_dir).resolve()
    review_dir = batch_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    sources_list = _load_sample_sources(batch_root)
    sources = {source.sample_id: source for source in sources_list}
    candidates = _load_candidates(sources_list)
    references_available = references_path is not None
    if references_available:
        references = _load_references(
            Path(references_path),
            set(sources),
            reference_min_final_score,
        )
    else:
        references = pd.DataFrame()
    candidates = _attach_reference_fields(candidates, references, match_radius_px)
    accepted = _candidate_review_table(
        candidates,
        novel_only=False,
        references_available=references_available,
    )
    novel = (
        _candidate_review_table(
            candidates,
            novel_only=True,
            references_available=True,
        )
        if references_available
        else pd.DataFrame()
    )
    reference_comparison = (
        _reference_comparison(references, candidates, match_radius_px)
        if references_available
        else pd.DataFrame()
    )
    missed = (
        _missed_review_table(reference_comparison)
        if references_available
        else pd.DataFrame()
    )
    accepted_by_sample = {
        sample_id: table.copy()
        for sample_id, table in candidates[
            candidates["accepted"].astype(bool)
        ].groupby("sample_id")
    }
    arrays = {}
    mask_cache: Dict[Path, PersistedMask] = {}
    with ExitStack() as stack:
        for source in sources_list:
            tif = stack.enter_context(tifffile.TiffFile(source.image_path))
            arrays[source.sample_id] = zarr.open(tif.series[0].aszarr(), mode="r")
        accepted = _render_pages(
            accepted,
            mode="candidate",
            pages_dir=review_dir / "accepted_pages",
            page_title="Accepted raw-UCHL1 oocyte candidates",
            columns=columns,
            rows=rows,
            sources=sources,
            arrays=arrays,
            accepted_by_sample=accepted_by_sample,
            mask_cache=mask_cache,
            neighbor_overlays=neighbor_overlays,
            max_neighbor_overlays=max_neighbor_overlays,
            neighbor_margin_px=neighbor_margin_px,
            missed_patch_radius_px=missed_patch_radius_px,
        )
        if references_available:
            novel = _render_pages(
                novel,
                mode="candidate",
                pages_dir=review_dir / "novel_pages",
                page_title="Novel raw-UCHL1 oocyte candidates",
                columns=columns,
                rows=rows,
                sources=sources,
                arrays=arrays,
                accepted_by_sample=accepted_by_sample,
                mask_cache=mask_cache,
                neighbor_overlays=neighbor_overlays,
                max_neighbor_overlays=max_neighbor_overlays,
                neighbor_margin_px=neighbor_margin_px,
                missed_patch_radius_px=missed_patch_radius_px,
            )
            missed = _render_pages(
                missed,
                mode="miss",
                pages_dir=review_dir / "missed_pages",
                page_title="Legacy references missed by accepted detector candidates",
                columns=columns,
                rows=rows,
                sources=sources,
                arrays=arrays,
                accepted_by_sample=accepted_by_sample,
                mask_cache=mask_cache,
                neighbor_overlays=neighbor_overlays,
                max_neighbor_overlays=max_neighbor_overlays,
                neighbor_margin_px=neighbor_margin_px,
                missed_patch_radius_px=missed_patch_radius_px,
            )

    accepted_path = review_dir / "accepted_candidates.csv"
    novel_path = review_dir / "novel_candidates.csv"
    missed_path = review_dir / "missed_references.csv"
    comparison_path = review_dir / "reference_comparison.csv"
    accepted.to_csv(accepted_path, index=False)
    if references_available:
        novel.to_csv(novel_path, index=False)
        missed.to_csv(missed_path, index=False)
        reference_comparison.to_csv(comparison_path, index=False)
    summary_path = review_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "references_available": references_available,
                "references_path": None
                if references_path is None
                else str(Path(references_path).resolve()),
                "accepted_count": int(len(accepted)),
                "novel_count": None if not references_available else int(len(novel)),
                "missed_reference_count": None
                if not references_available
                else int(len(missed)),
                "match_radius_px": float(match_radius_px),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifacts = {
        "accepted_candidates": accepted_path,
        "summary": summary_path,
        "accepted_pages": review_dir / "accepted_pages",
    }
    if references_available:
        artifacts.update(
            {
                "novel_candidates": novel_path,
                "missed_references": missed_path,
                "reference_comparison": comparison_path,
                "novel_pages": review_dir / "novel_pages",
                "missed_pages": review_dir / "missed_pages",
            }
        )
    return ReviewPackResult(
        review_dir=review_dir,
        accepted_count=len(accepted),
        novel_count=None if not references_available else len(novel),
        missed_reference_count=None if not references_available else len(missed),
        artifact_paths=artifacts,
    )
