"""Identity-bound polygon review for unresolved Recall manual-seed boundaries."""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
from PIL import Image

from .io import load_candidate_mask
from .manual_seed_finalize import _atomic_write_csv, _validate_review_identity
from .manual_seed_review import _draw_mask
from .precision_manual_boundary_review import (
    _CSS as _MANUAL_BOUNDARY_CSS,
    _JS as _PRECISION_MANUAL_BOUNDARY_JS,
    _card_html,
    _intersects_patch,
)
from .recall_overlay import overlay_dir_from_identity
from .recall_review import (
    RecallReviewRuntime,
    _atomic_save_image,
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _read_json,
)


RECALL_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION = 1
RECALL_MANUAL_BOUNDARY_RENDERER_VERSION = "recall_manual_polygon_v1"
RECALL_MANUAL_BOUNDARY_REVIEW_TYPE = "oocyte_recall_manual_boundary_review"


@dataclass(frozen=True)
class RecallManualBoundaryReviewResult:
    page_path: Path
    candidates_path: Path
    assets_dir: Path
    card_count: int


def _verify_manual_seed_delivery(
    base_dir: Path,
    *,
    sample_identity: Mapping[str, Any],
    review_sha256: str,
) -> Dict[str, Any]:
    root = Path(base_dir).resolve()
    manifest_path = root / "manual_seed_finalize_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manual-seed finalize manifest does not exist: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("delivery_name") != "reviewed_manual_seed_delta_v1":
        raise ValueError("base directory is not a reviewed_manual_seed_delta_v1 delivery")
    if manifest.get("sample") != sample_identity:
        raise ValueError("base manual-seed sample identity does not match")
    if str(manifest.get("review_json_sha256", "")) != review_sha256:
        raise ValueError("base manual-seed review SHA-256 does not match")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("base manual-seed manifest is missing artifacts")
    for relative_path, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid base artifact record: {relative_path}")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise FileNotFoundError(f"base manual-seed artifact is missing: {path}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"base artifact size mismatch: {relative_path}")
        if _file_sha256(path) != str(record.get("sha256", "")):
            raise ValueError(f"base artifact SHA-256 mismatch: {relative_path}")
    required = (
        "manual_seed_review_decisions.csv",
        "manual_seed_accepted_candidates.csv",
        "oocyte_candidates_rescue_v1_plus_manual_seed_v1.csv",
        "oocyte_labels_rescue_v1_plus_manual_seed_v1.ome.tiff",
        "oocyte_labels_rescue_v1_plus_manual_seed_v1_mapping.csv",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"base manual-seed delivery is missing {name}")
    return manifest


def _resolved_mask_path(base_dir: Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row["mask_path"]))
    if path.is_absolute():
        return path.resolve()
    source_dir = row.get("mask_source_dir")
    root = (
        Path(base_dir)
        if source_dir is None or pd.isna(source_dir) or not str(source_dir).strip()
        else Path(str(source_dir))
    )
    return (root / path).resolve()


def _render_assets(
    runtime: RecallReviewRuntime,
    *,
    center_xy: tuple[int, int],
    radius: int,
    current_mask_path: Path,
    resolved_candidates: pd.DataFrame,
    base_dir: Path,
    allowed_mask_roots: Sequence[Path],
    raw_path: Path,
    context_path: Path,
) -> tuple[Any, int]:
    patch = runtime.source.read_patch(center_xy, radius)
    raw = Image.open(
        io.BytesIO(runtime.render_patch(center_xy, radius, "local"))
    ).convert("RGBA")
    context = raw.copy()
    neighbor_count = 0
    roots = tuple(Path(root).resolve() for root in allowed_mask_roots)
    for row in resolved_candidates.to_dict("records"):
        if not _intersects_patch(row, patch):
            continue
        mask_path = _resolved_mask_path(base_dir, row)
        if not mask_path.is_file() or not any(
            mask_path.is_relative_to(root) for root in roots
        ):
            raise ValueError(f"resolved mask is outside an identity-bound delivery: {mask_path}")
        mask = load_candidate_mask(mask_path)
        if mask.image_shape_yx != runtime.sample.image_shape_yx:
            raise ValueError(f"resolved mask image shape mismatch: {mask_path}")
        context = _draw_mask(
            context,
            runtime._place_mask(mask, patch),
            color=(0, 235, 220),
        )
        neighbor_count += 1
    current = load_candidate_mask(current_mask_path)
    if current.image_shape_yx != runtime.sample.image_shape_yx:
        raise ValueError(f"current target mask image shape mismatch: {current_mask_path}")
    context = _draw_mask(
        context,
        runtime._place_mask(current, patch),
        color=(255, 205, 55),
    )
    _atomic_save_image(raw_path, raw.convert("RGB"), format_name="WEBP")
    _atomic_save_image(context_path, context.convert("RGB"), format_name="WEBP")
    return patch, neighbor_count


def _recall_manual_boundary_javascript() -> str:
    script = _PRECISION_MANUAL_BOUNDARY_JS
    replacements = {
        "aegle-oocyte-manual-boundary:": "aegle-oocyte-recall-manual-boundary:",
        "DATA.identity.manual_boundary_candidate_table_sha256": (
            "DATA.identity.recall_manual_boundary_candidate_table_sha256"
        ),
        "['sample_id','candidate_table_sha256','base_precision_resolved_manifest_sha256','manual_boundary_candidate_table_sha256']": (
            "['sample_id','recall_window_geometry_sha256',"
            "'base_manual_seed_manifest_sha256',"
            "'recall_manual_boundary_candidate_table_sha256']"
        ),
        "oocyte_precision_manual_boundary_review": RECALL_MANUAL_BOUNDARY_REVIEW_TYPE,
        "_precision_manual_boundary_review.json": "_recall_manual_boundary_review.json",
    }
    for old, new in replacements.items():
        if old not in script:
            raise RuntimeError(f"manual-boundary JavaScript template changed: {old}")
        script = script.replace(old, new)
    return script


def generate_recall_manual_boundary_review(
    sample_dir: Path,
    manual_review_json: Path,
    base_finalize_dir: Path,
    out_dir: Path,
    *,
    patch_radius_px: int = 220,
) -> RecallManualBoundaryReviewResult:
    """Generate a polygon editor for confirmed oocytes with two bad masks."""

    if patch_radius_px < 64:
        raise ValueError("manual-boundary patch radius must be at least 64 pixels")
    review_path = Path(manual_review_json).resolve()
    payload = _read_json(review_path)
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("manual-seed review is missing its identity object")
    overlay_dir = overlay_dir_from_identity(identity)
    sample = _load_sample(sample_dir, overlay_dir=overlay_dir)
    _validate_review_identity(sample.review_identity, payload)
    review_sha256 = _file_sha256(review_path)
    base_dir = Path(base_finalize_dir).resolve()
    _verify_manual_seed_delivery(
        base_dir,
        sample_identity=sample.review_identity,
        review_sha256=review_sha256,
    )
    base_manifest_path = base_dir / "manual_seed_finalize_manifest.json"
    base_manifest_sha256 = _file_sha256(base_manifest_path)
    analysis_path = Path(str(identity.get("analysis_table", ""))).resolve()
    if not analysis_path.is_file():
        raise FileNotFoundError(f"recall analysis table does not exist: {analysis_path}")
    if _file_sha256(analysis_path) != str(identity.get("analysis_sha256", "")):
        raise ValueError("recall analysis SHA-256 does not match manual review identity")
    analysis = pd.read_csv(analysis_path)
    analysis_by_id = analysis.set_index("annotation_id", drop=False)
    review_rows = payload.get("rows")
    if not isinstance(review_rows, list):
        raise ValueError("manual-seed review rows must be a list")
    unresolved = []
    for review_index, row in enumerate(review_rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("manual-seed review row must be an object")
        choice = str(row.get("manual_mask_choice", "")).strip()
        notes = str(row.get("manual_notes", "")).strip()
        if choice == "neither" and "needs_manual_boundary" in notes.casefold():
            unresolved.append((review_index, row))
    if not unresolved:
        raise ValueError("manual-seed review has no confirmed manual-boundary queue")

    combined_path = base_dir / "oocyte_candidates_rescue_v1_plus_manual_seed_v1.csv"
    resolved_candidates = pd.read_csv(combined_path)
    allowed_roots = [base_dir, sample.sample_dir]
    if overlay_dir is not None:
        allowed_roots.append(overlay_dir)
    root = Path(out_dir).resolve()
    assets_dir = root / "review_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    with RecallReviewRuntime(sample.sample_dir, overlay_dir=overlay_dir) as runtime:
        for manual_index, (review_index, review_row) in enumerate(
            unresolved,
            start=1,
        ):
            annotation_id = str(review_row.get("annotation_id", ""))
            if annotation_id not in analysis_by_id.index:
                raise ValueError(
                    f"manual boundary annotation is absent from analysis: {annotation_id}"
                )
            source = analysis_by_id.loc[annotation_id]
            current_path = Path(str(source["manual_conservative_mask_path"])).resolve()
            if not current_path.is_file() or not current_path.is_relative_to(
                analysis_path.parent
            ):
                raise ValueError(
                    f"manual boundary current mask is outside analysis: {current_path}"
                )
            center = (
                int(round(float(source["x"]))),
                int(round(float(source["y"]))),
            )
            raw_name = f"manual-{manual_index:03d}-raw.webp"
            context_name = f"manual-{manual_index:03d}-context.webp"
            patch, neighbor_count = _render_assets(
                runtime,
                center_xy=center,
                radius=patch_radius_px,
                current_mask_path=current_path,
                resolved_candidates=resolved_candidates,
                base_dir=base_dir,
                allowed_mask_roots=allowed_roots,
                raw_path=assets_dir / raw_name,
                context_path=assets_dir / context_name,
            )
            patch_origin_x = center[0] - patch_radius_px
            patch_origin_y = center[1] - patch_radius_px
            output_rows.append(
                {
                    "manual_index": manual_index,
                    "review_index": review_index,
                    "boundary_index": manual_index,
                    "review_key": annotation_id,
                    "annotation_id": annotation_id,
                    "display_id": f"#R{review_index:03d}",
                    "detector_component_id": annotation_id,
                    "detection_pass": "manual_seed_review_v1",
                    "center_x": float(source["x"]),
                    "center_y": float(source["y"]),
                    "center_local_x": float(source["x"]) - patch_origin_x,
                    "center_local_y": float(source["y"]) - patch_origin_y,
                    "precision_notes": "",
                    "boundary_review_notes": str(review_row.get("manual_notes", "")),
                    "current_mask_path": str(current_path),
                    "current_mask_sha256": _file_sha256(current_path),
                    "expanded_mask_path": str(
                        Path(str(source["manual_expanded_mask_path"])).resolve()
                    ),
                    "patch_origin_x": patch_origin_x,
                    "patch_origin_y": patch_origin_y,
                    "patch_bbox_x0": patch.bbox.x0,
                    "patch_bbox_y0": patch.bbox.y0,
                    "patch_bbox_x1": patch.bbox.x1,
                    "patch_bbox_y1": patch.bbox.y1,
                    "asset_width_px": int(patch.image.shape[1]),
                    "asset_height_px": int(patch.image.shape[0]),
                    "resolved_neighbor_count": neighbor_count,
                    "raw_asset_name": raw_name,
                    "raw_asset_sha256": _file_sha256(assets_dir / raw_name),
                    "context_asset_name": context_name,
                    "context_asset_sha256": _file_sha256(assets_dir / context_name),
                }
            )
    expected_assets = {
        str(row[field])
        for row in output_rows
        for field in ("raw_asset_name", "context_asset_name")
    }
    for existing in assets_dir.glob("*.webp"):
        if existing.name not in expected_assets:
            existing.unlink()
    candidates_path = root / "recall_manual_boundary_candidates.csv"
    _atomic_write_csv(pd.DataFrame(output_rows), candidates_path)
    page_identity: Dict[str, Any] = {
        **dict(identity),
        "base_manual_seed_dir": str(base_dir),
        "base_manual_seed_manifest": str(base_manifest_path),
        "base_manual_seed_manifest_sha256": base_manifest_sha256,
        "base_combined_candidates_sha256": _file_sha256(combined_path),
        "manual_seed_review": str(review_path),
        "manual_seed_review_sha256": review_sha256,
        "recall_manual_boundary_candidate_table": str(candidates_path),
        "recall_manual_boundary_candidate_table_sha256": _file_sha256(
            candidates_path
        ),
        "renderer_version": RECALL_MANUAL_BOUNDARY_RENDERER_VERSION,
        "patch_radius_px": patch_radius_px,
    }
    page_payload = {
        "identity": page_identity,
        "rows": [_json_safe(row) for row in output_rows],
    }
    cards = "".join(_card_html(row) for row in output_rows)
    script_payload = json.dumps(page_payload, allow_nan=False).replace("</", "<\\/")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>{html.escape(sample.sample_id)} Recall manual boundary</title><style>{_MANUAL_BOUNDARY_CSS}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle / Recall delta / exact manual boundary</div><h1>{html.escape(sample.sample_id)} recall contour desk</h1><p>Trace the intended outer oocyte boundary on native UCHL1. Cyan marks all 128 currently resolved masks; yellow is the rejected conservative fragment; orange is your polygon. The expanded proposal was rejected because it merged a neighbor. The exported contour remains provisional until Python validates and rasterizes it.</p><div class="steps"><span><strong>1.</strong> Click around the intended boundary.</span><span><strong>2.</strong> Drag handles to refine; hide masks when needed.</span><span><strong>3.</strong> Accept the contour and export JSON.</span></div></header><nav class="toolbar"><button data-filter="all">All</button><button data-filter="unreviewed">Unreviewed</button><span id="progress" class="mono"></span><span class="grow"></span><button id="import-button">Import JSON</button><input id="import-json" type="file" accept="application/json" hidden><button id="export-json">Export JSON</button><a class="button" href="../recall_analysis_v1/manual_seed_review.html">Back to mask review</a></nav><section class="cards">{cards}</section><footer class="footer">The first vertex is yellow. Avoid cyan neighbors and follicular halo. Any edited contour returns to Unreviewed until explicitly accepted again.</footer></main><script id="manual-boundary-data" type="application/json">{script_payload}</script><script>{_recall_manual_boundary_javascript()}</script></body></html>'''
    page_path = root / "recall_manual_boundary_review.html"
    _atomic_write_text(page_path, page)
    summary = {
        "schema_version": RECALL_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION,
        "review_type": "oocyte_recall_manual_boundary_review_pack",
        "sample": sample.review_identity,
        "base_manual_seed_manifest": str(base_manifest_path),
        "base_manual_seed_manifest_sha256": base_manifest_sha256,
        "manual_boundary_card_count": len(output_rows),
        "page": str(page_path),
        "candidates": str(candidates_path),
        "production_outputs_modified": False,
        "base_manual_seed_outputs_modified": False,
    }
    _atomic_write_text(
        root / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
    )
    return RecallManualBoundaryReviewResult(
        page_path=page_path,
        candidates_path=candidates_path,
        assets_dir=assets_dir,
        card_count=len(output_rows),
    )


__all__ = [
    "RECALL_MANUAL_BOUNDARY_RENDERER_VERSION",
    "RECALL_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION",
    "RECALL_MANUAL_BOUNDARY_REVIEW_TYPE",
    "RecallManualBoundaryReviewResult",
    "generate_recall_manual_boundary_review",
]
