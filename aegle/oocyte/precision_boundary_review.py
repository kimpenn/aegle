"""Review-gated boundary replacements for accepted raw-UCHL1 candidates."""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .io import load_candidate_mask
from .manual_seed_finalize import _atomic_write_csv
from .manual_seed_review import _crosshair, _draw_mask, _load_provisional, _panel_label
from .models import LocalSegmentationResult, PersistedMask
from .recall_review import (
    RecallReviewRuntime,
    _atomic_save_image,
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _mask_path,
    _read_json,
    _save_provisional_mask,
)


PRECISION_BOUNDARY_REVIEW_SCHEMA_VERSION = 1
BOUNDARY_CANDIDATE_COLUMNS = (
    "boundary_index",
    "review_key",
    "display_id",
    "detector_component_id",
    "detection_pass",
    "x",
    "y",
    "current_mask_path",
    "conservative_available",
    "expanded_available",
)
BOUNDARY_RECOVERY_PARAMETERS = {
    "annulus_percentiles": [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0],
    "min_area_growth_ratio": 1.10,
    "standard_max_area_growth_ratio": 4.0,
    "shape_max_area_growth_ratio": 6.0,
    "min_current_overlap": 0.95,
    "max_equivalent_diameter_um": 100.0,
    "max_centroid_offset_px": 50.0,
    "shape_min_circularity": 0.80,
    "shape_min_solidity": 0.90,
    "shape_max_centroid_offset_px": 25.0,
}


@dataclass(frozen=True)
class PrecisionBoundaryReviewResult:
    page_path: Path
    candidates_path: Path
    assets_dir: Path
    masks_dir: Path
    card_count: int
    proposal_count: int
    manual_only_count: int
    automatic_review_path: Path | None


def _review_key(row: Mapping[str, Any]) -> str:
    return (
        f"{str(row.get('detection_pass', 'baseline_v6'))}:"
        f"{str(row['detector_component_id'])}"
    )


def _note_tokens(value: Any) -> set[str]:
    return {
        token.strip().casefold()
        for token in str(value or "").split(";")
        if token.strip()
    }


def _validate_precision_review(
    sample_identity: Mapping[str, Any],
    candidates: pd.DataFrame,
    payload: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    if payload.get("schema_version") != PRECISION_BOUNDARY_REVIEW_SCHEMA_VERSION:
        raise ValueError("precision review schema_version must be 1")
    if payload.get("review_type") != "oocyte_precision_review":
        raise ValueError("review_type must be 'oocyte_precision_review'")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("precision review identity is missing")
    for field, expected in sample_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"precision review identity mismatch for {field}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("precision review rows must be a list")

    current_by_key = {
        _review_key(row): row for row in candidates.to_dict("records")
    }
    if len(current_by_key) != len(candidates):
        raise ValueError("current candidate review keys are not unique")
    seen = set()
    allowed_statuses = {"accept", "reject", "unsure"}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"precision review row {index} must be an object")
        key = str(row.get("review_key", ""))
        if key in seen:
            raise ValueError(f"precision review contains duplicate key {key}")
        seen.add(key)
        current = current_by_key.get(key)
        if current is None:
            raise ValueError(f"precision review contains unknown candidate {key}")
        for field in ("detector_component_id", "detection_pass"):
            if str(row.get(field, "")) != str(current[field]):
                raise ValueError(f"precision review metadata mismatch for {key}: {field}")
        for field in ("center_x", "center_y"):
            if float(row.get(field, np.nan)) != float(current[field]):
                raise ValueError(f"precision review metadata mismatch for {key}: {field}")
        if str(row.get("manual_status", "")) not in allowed_statuses:
            raise ValueError(f"precision review status is unresolved for {key}")
    if seen != set(current_by_key):
        raise ValueError("precision review does not cover the current candidate table")
    return rows


def _proposal_is_safe(
    result: LocalSegmentationResult,
    current_local: np.ndarray,
) -> Tuple[bool, Dict[str, float], Sequence[str]]:
    current_area = max(int(current_local.sum()), 1)
    proposal_area = int(result.metrics.area_px)
    intersection = int(np.logical_and(current_local, result.mask).sum())
    union = int(np.logical_or(current_local, result.mask).sum())
    area_ratio = proposal_area / current_area
    current_overlap = intersection / current_area
    iou = intersection / max(union, 1)
    shape_extension = bool(
        area_ratio <= BOUNDARY_RECOVERY_PARAMETERS["shape_max_area_growth_ratio"]
        and result.metrics.circularity
        >= BOUNDARY_RECOVERY_PARAMETERS["shape_min_circularity"]
        and result.metrics.solidity
        >= BOUNDARY_RECOVERY_PARAMETERS["shape_min_solidity"]
        and result.metrics.centroid_offset_px
        <= BOUNDARY_RECOVERY_PARAMETERS["shape_max_centroid_offset_px"]
    )
    safe = bool(
        area_ratio >= BOUNDARY_RECOVERY_PARAMETERS["min_area_growth_ratio"]
        and (
            area_ratio
            <= BOUNDARY_RECOVERY_PARAMETERS["standard_max_area_growth_ratio"]
            or shape_extension
        )
        and current_overlap
        >= BOUNDARY_RECOVERY_PARAMETERS["min_current_overlap"]
        and result.metrics.equivalent_diameter_um
        <= BOUNDARY_RECOVERY_PARAMETERS["max_equivalent_diameter_um"]
        and result.metrics.centroid_offset_px
        <= BOUNDARY_RECOVERY_PARAMETERS["max_centroid_offset_px"]
    )
    warnings = []
    if result.metrics.circularity < 0.65:
        warnings.append("low_circularity")
    if result.metrics.solidity < 0.85:
        warnings.append("low_solidity")
    if area_ratio > 3.0:
        warnings.append("large_area_growth")
    if result.metrics.centroid_offset_px > 25.0:
        warnings.append("large_centroid_offset")
    metrics = {
        "area_px": float(proposal_area),
        "area_ratio": float(area_ratio),
        "current_overlap": float(current_overlap),
        "iou": float(iou),
        "equivalent_diameter_um": float(result.metrics.equivalent_diameter_um),
        "circularity": float(result.metrics.circularity),
        "solidity": float(result.metrics.solidity),
        "centroid_offset_px": float(result.metrics.centroid_offset_px),
    }
    return safe, metrics, warnings


def _current_metrics(mask: PersistedMask) -> Mapping[str, Any]:
    metrics = mask.metadata.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _add_proposal_fields(
    row: Dict[str, Any],
    *,
    prefix: str,
    result: LocalSegmentationResult | None,
    percentile: float | None,
    current_local: np.ndarray,
    path: Path,
    patch: Any,
    annotation_id: str,
) -> bool:
    if result is None or percentile is None:
        row[f"{prefix}_available"] = False
        return False
    safe, metrics, warnings = _proposal_is_safe(result, current_local)
    row[f"{prefix}_available"] = safe
    row[f"{prefix}_percentile"] = float(percentile)
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value
    row[f"{prefix}_warnings"] = ";".join(warnings)
    if not safe:
        return False
    _save_provisional_mask(
        path,
        result=result,
        patch=patch,
        annotation_id=annotation_id,
        percentile=float(percentile),
    )
    row[f"{prefix}_mask_path"] = str(path)
    return True


def _render_card(
    runtime: RecallReviewRuntime,
    row: Mapping[str, Any],
    destination: Path,
    *,
    radius: int,
) -> None:
    center = (int(round(float(row["x"]))), int(round(float(row["y"]))))
    patch = runtime.source.read_patch(center, radius)
    raw = Image.open(io.BytesIO(runtime.render_patch(center, radius, "local"))).convert(
        "RGBA"
    )
    existing = Image.open(io.BytesIO(runtime.render_overlay(center, radius))).convert(
        "RGBA"
    )
    base = Image.alpha_composite(raw, existing)
    click_x = center[0] - (center[0] - radius)
    click_y = center[1] - (center[1] - radius)
    context = base.copy()
    _crosshair(context, click_x, click_y)
    panels = [
        _panel_label(
            context,
            f"#{int(row['boundary_index']):03d} RAW + ALL CURRENT MASKS",
            color=(0, 255, 242),
        )
    ]

    current = load_candidate_mask(Path(str(row["current_mask_path"])))
    current_panel = _draw_mask(
        base.copy(), runtime._place_mask(current, patch), color=(255, 211, 72)
    )
    _crosshair(current_panel, click_x, click_y)
    panels.append(
        _panel_label(
            current_panel,
            f"CURRENT / d {float(row['current_equivalent_diameter_um']):.1f} um",
            color=(255, 211, 72),
        )
    )

    for prefix, color, label in (
        ("conservative", (80, 214, 132), "CONSERVATIVE"),
        ("expanded", (255, 132, 67), "EXPANDED"),
    ):
        panel = base.copy()
        if bool(row.get(f"{prefix}_available", False)):
            persisted = _load_provisional(str(row[f"{prefix}_mask_path"]))
            if persisted is None:
                raise FileNotFoundError(f"boundary proposal is missing: {prefix}")
            panel = _draw_mask(
                panel,
                runtime._place_mask(persisted, patch),
                color=color,
            )
            panel_label = (
                f"{label} / P{float(row[f'{prefix}_percentile']):.0f} / "
                f"d {float(row[f'{prefix}_equivalent_diameter_um']):.1f} um"
            )
        else:
            panel_label = f"{label} / NO SAFE PROPOSAL"
        _crosshair(panel, click_x, click_y)
        panels.append(_panel_label(panel, panel_label, color=color))

    gutter = 5
    canvas = Image.new(
        "RGBA",
        (
            sum(panel.width for panel in panels) + gutter * (len(panels) - 1),
            panels[0].height,
        ),
        (245, 238, 224, 255),
    )
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width + gutter
    _atomic_save_image(destination, canvas.convert("RGB"), format_name="WEBP")


def _metric(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


def _card_html(row: Mapping[str, Any]) -> str:
    key = html.escape(str(row["review_key"]), quote=True)
    index = int(row["boundary_index"])
    conservative_disabled = "" if row.get("conservative_available") else " disabled"
    expanded_disabled = "" if row.get("expanded_available") else " disabled"
    warnings = "; ".join(
        token
        for token in (
            str(row.get("conservative_warnings", "")),
            str(row.get("expanded_warnings", "")),
        )
        if token
    ) or "none"
    return f'''<article class="card" data-id="{key}" data-review="unreviewed">
  <img src="review_assets/boundary-{index:03d}.webp" loading="lazy" alt="Boundary comparison for {key}">
  <div class="body"><div class="title"><h2>#{index:03d} / {html.escape(str(row['display_id']))}</h2><span>{html.escape(str(row['proposal_status']))}</span></div>
  <div class="mono">{key} &middot; x {int(round(float(row['x'])))} &middot; y {int(round(float(row['y'])))}</div>
  <div class="metrics"><span>current d {_metric(row['current_equivalent_diameter_um'],1)} um</span><span>current circ {_metric(row['current_circularity'])}</span><span>current solid {_metric(row['current_solidity'])}</span><span>conservative growth {_metric(row.get('conservative_area_ratio'))}x</span><span>expanded growth {_metric(row.get('expanded_area_ratio'))}x</span><span>warnings {html.escape(warnings)}</span></div>
  <div class="actions"><button data-choice="keep_current">Keep current</button><button data-choice="use_conservative"{conservative_disabled}>Use conservative</button><button data-choice="use_expanded"{expanded_disabled}>Use expanded</button><button data-choice="needs_manual">Needs manual</button><button data-choice="exclude">Not oocyte</button><button data-choice="unsure">Unsure</button></div>
  <input class="notes" value="{html.escape(str(row.get('precision_notes', '')), quote=True)}" placeholder="Boundary-review note">
  </div></article>'''


_CSS = r'''
:root{--ink:#172522;--panel:#fffaf0;--line:#d1c6b3;--teal:#087b78;--green:#3f9b68;--orange:#d9662b;--yellow:#b58a17;--red:#a94336}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 10% 0,#fff8e8 0,transparent 30%),linear-gradient(135deg,#eadfcd,#f8f2e7 66%,#dcebe3);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}.shell{width:min(1600px,calc(100% - 24px));margin:auto}.hero{margin:16px 0;padding:24px 28px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(115deg,#fffaf0,#e4f3ec);box-shadow:0 15px 35px rgba(30,45,38,.12)}.hero h1{font-size:clamp(2rem,4.5vw,4.2rem);line-height:.95;margin:.2em 0}.hero p{max-width:1050px;line-height:1.5}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{text-transform:uppercase;letter-spacing:.14em;color:var(--teal);font-size:.75rem;font-weight:700}.toolbar{position:sticky;top:6px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 12px;margin:14px 0;border:1px solid var(--line);border-radius:14px;background:rgba(255,250,240,.95)}button,.button,input{font:inherit;border:1px solid #b9ae9d;border-radius:10px;padding:8px 11px;background:#fffaf0;color:var(--ink)}button{cursor:pointer}button:disabled{opacity:.35;cursor:not-allowed}.button{text-decoration:none}.grow{flex:1}.cards{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:50px}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:0 9px 24px rgba(32,45,39,.08)}.card.hidden{display:none}.card img{width:100%;display:block;background:#171f1c}.body{padding:14px}.title{display:flex;align-items:center;justify-content:space-between}.title h2{margin:0}.title span{font-family:monospace;color:var(--orange)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:10px 0;font:12px monospace}.metrics span{background:#f0e7d8;padding:7px;border-radius:7px}.actions{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}.actions button.selected[data-choice=keep_current]{background:#f6e6a9;border-color:var(--yellow)}.actions button.selected[data-choice=use_conservative]{background:#d4eedf;border-color:var(--green)}.actions button.selected[data-choice=use_expanded]{background:#ffd7bd;border-color:var(--orange)}.actions button.selected[data-choice=needs_manual],.actions button.selected[data-choice=exclude]{background:#f1d6cf;border-color:var(--red)}.actions button.selected[data-choice=unsure]{background:#e5e1d7;border-color:#847d70}.notes{width:100%;margin-top:8px}.footer{padding:10px 0 50px;color:#68726c}@media(max-width:900px){.actions{grid-template-columns:1fr 1fr 1fr}.metrics{grid-template-columns:1fr 1fr}.toolbar{position:static}}
'''


_JS = r'''
const DATA=JSON.parse(document.getElementById('boundary-data').textContent),KEY='aegle-oocyte-precision-boundary:'+DATA.identity.sample_id+':'+DATA.identity.boundary_candidate_table_sha256;let state=JSON.parse(localStorage.getItem(KEY)||'{}'),filter='all';const cards=[...document.querySelectorAll('.card')];function save(){localStorage.setItem(KEY,JSON.stringify(state));progress()}function paint(card){const id=card.dataset.id,s=state[id]||{};card.dataset.review=s.choice||'unreviewed';card.querySelectorAll('[data-choice]').forEach(b=>b.classList.toggle('selected',b.dataset.choice===s.choice));card.querySelector('.notes').value=s.notes??card.querySelector('.notes').value}function progress(){const n=DATA.rows.filter(r=>(state[r.review_key]||{}).choice).length;document.getElementById('progress').textContent=n+' / '+DATA.rows.length+' reviewed'}function apply(){cards.forEach(c=>c.classList.toggle('hidden',filter==='unreviewed'&&c.dataset.review!=='unreviewed'))}cards.forEach(paint);document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{const card=b.closest('.card'),id=card.dataset.id;state[id]={...(state[id]||{}),choice:b.dataset.choice};paint(card);save();apply()});document.querySelectorAll('.notes').forEach(n=>n.onchange=()=>{const card=n.closest('.card'),id=card.dataset.id;state[id]={...(state[id]||{}),notes:n.value};save()});document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;apply()});function exportData(type){const rows=DATA.rows.map(r=>({...r,boundary_review_choice:(state[r.review_key]||{}).choice||'',boundary_review_notes:(state[r.review_key]||{}).notes||''})),payload={schema_version:1,review_type:'oocyte_precision_boundary_review',identity:DATA.identity,exported_at:new Date().toISOString(),rows};let blob,name;if(type==='json'){blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});name=DATA.identity.sample_id+'_precision_boundary_review.json'}else{const keys=Object.keys(rows[0]||{}),esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';blob=new Blob([[keys.join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\n')],{type:'text/csv'});name=DATA.identity.sample_id+'_precision_boundary_review.csv'}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}document.getElementById('export-json').onclick=()=>exportData('json');document.getElementById('export-csv').onclick=()=>exportData('csv');progress();apply();
'''


def generate_precision_boundary_review(
    sample_dir: Path,
    precision_review_json: Path,
    out_dir: Path,
    *,
    patch_radius_px: int = 220,
) -> PrecisionBoundaryReviewResult:
    """Build a review pack for true oocytes rejected only for mask quality."""

    sample = _load_sample(sample_dir)
    precision_path = Path(precision_review_json).resolve()
    precision_payload = _read_json(precision_path)
    precision_rows = _validate_precision_review(
        sample.review_identity,
        sample.candidates,
        precision_payload,
    )
    boundary_rows = [
        row
        for row in precision_rows
        if str(row.get("manual_status")) == "reject"
        and "true_oocyte" in _note_tokens(row.get("manual_notes"))
    ]

    candidate_by_key = {
        _review_key(row): row for row in sample.candidates.to_dict("records")
    }
    true_centers = [
        (float(row["center_x"]), float(row["center_y"]))
        for row in precision_rows
        if str(row.get("manual_status")) == "accept"
        or "true_oocyte" in _note_tokens(row.get("manual_notes"))
    ]
    root = Path(out_dir).resolve()
    masks_dir = root / "proposal_masks"
    assets_dir = root / "review_assets"
    masks_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []

    with RecallReviewRuntime(sample.sample_dir) as runtime:
        for index, review_row in enumerate(boundary_rows, start=1):
            key = str(review_row["review_key"])
            candidate = candidate_by_key[key]
            x = float(candidate["center_x"])
            y = float(candidate["center_y"])
            other_centers = tuple(
                center
                for center in true_centers
                if float(np.hypot(center[0] - x, center[1] - y)) >= 25.0
            )
            segmentation = runtime.segment_manual_provisionals(
                x,
                y,
                exclude_points_xy=other_centers,
                allow_shape_recovery=True,
            )
            current = runtime._candidate_mask(candidate)
            current_local = runtime._place_mask(current, segmentation.patch)
            metrics = _current_metrics(current)
            output: Dict[str, Any] = {
                "boundary_index": index,
                "review_key": key,
                "display_id": str(review_row["display_id"]),
                "detector_component_id": str(candidate["detector_component_id"]),
                "detection_pass": str(candidate["detection_pass"]),
                "x": x,
                "y": y,
                "precision_status": str(review_row["manual_status"]),
                "precision_notes": str(review_row.get("manual_notes", "")),
                "current_mask_path": str(_mask_path(sample.sample_dir, candidate).resolve()),
                "current_area_px": int(current.mask.sum()),
                "current_equivalent_diameter_um": float(
                    metrics.get("equivalent_diameter_um", np.nan)
                ),
                "current_circularity": float(metrics.get("circularity", np.nan)),
                "current_solidity": float(metrics.get("solidity", np.nan)),
                "segmentation_error": segmentation.error,
            }
            conservative_available = _add_proposal_fields(
                output,
                prefix="conservative",
                result=segmentation.conservative,
                percentile=segmentation.conservative_percentile,
                current_local=current_local,
                path=masks_dir / f"boundary-{index:03d}-conservative.npz",
                patch=segmentation.patch,
                annotation_id=key,
            )
            expanded_available = _add_proposal_fields(
                output,
                prefix="expanded",
                result=segmentation.expanded,
                percentile=segmentation.expanded_percentile,
                current_local=current_local,
                path=masks_dir / f"boundary-{index:03d}-expanded.npz",
                patch=segmentation.patch,
                annotation_id=key,
            )
            output["proposal_status"] = (
                "automatic_options"
                if conservative_available or expanded_available
                else "manual_required"
            )
            output_rows.append(output)

        expected_masks = {
            Path(str(row[path_key])).name
            for row in output_rows
            for path_key in ("conservative_mask_path", "expanded_mask_path")
            if row.get(path_key)
        }
        for existing in masks_dir.glob("*.npz"):
            if existing.name not in expected_masks:
                existing.unlink()
        table = pd.DataFrame(output_rows)
        if table.empty:
            table = pd.DataFrame(columns=BOUNDARY_CANDIDATE_COLUMNS)
        candidates_path = root / "precision_boundary_candidates.csv"
        _atomic_write_csv(table, candidates_path)
        expected_assets = {
            f"boundary-{int(row['boundary_index']):03d}.webp"
            for row in output_rows
        }
        for existing in assets_dir.glob("*.webp"):
            if existing.name not in expected_assets:
                existing.unlink()
        for row in output_rows:
            _render_card(
                runtime,
                row,
                assets_dir / f"boundary-{int(row['boundary_index']):03d}.webp",
                radius=patch_radius_px,
            )

    identity: Dict[str, Any] = {
        **sample.review_identity,
        "precision_review_json": str(precision_path),
        "precision_review_json_sha256": _file_sha256(precision_path),
        "boundary_candidate_table": str(candidates_path),
        "boundary_candidate_table_sha256": _file_sha256(candidates_path),
        "boundary_recovery_parameters": BOUNDARY_RECOVERY_PARAMETERS,
        "patch_radius_px": int(patch_radius_px),
    }
    safe_rows = [_json_safe(row) for row in output_rows]
    cards = "".join(_card_html(row) for row in output_rows)
    payload = {"identity": identity, "rows": safe_rows}
    automatic_review_path = None
    if not output_rows:
        automatic_review_path = root / "precision_boundary_review_empty.json"
        automatic_review = {
            "schema_version": PRECISION_BOUNDARY_REVIEW_SCHEMA_VERSION,
            "review_type": "oocyte_precision_boundary_review",
            "identity": identity,
            "exported_at": None,
            "rows": [],
            "generated_automatically": True,
            "reason": "precision review contains no true-oocyte boundary failures",
        }
        _atomic_write_text(
            automatic_review_path,
            json.dumps(
                _json_safe(automatic_review),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
        )
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sample.sample_id)} Precision boundary review</title><style>{_CSS}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle / reviewed Precision boundary delta</div><h1>{html.escape(sample.sample_id)} boundary recovery</h1><p>These are true oocytes whose current masks were rejected for boundary quality. Cyan shows all current masks in context, yellow is the frozen current candidate, green is the conservative lower-annulus proposal, and orange is the largest bounded expansion. Choose a proposal only when it follows the intended outer oocyte boundary without absorbing a neighbor. This page never modifies detector output.</p></header><nav class="toolbar"><button data-filter="all">All</button><button data-filter="unreviewed">Unreviewed</button><span id="progress" class="mono"></span><span class="grow"></span><button id="export-json">Export JSON</button><button id="export-csv">Export CSV</button><a class="button" href="../oocytes.html">Back to Precision</a></nav><section class="cards">{cards}</section><footer class="footer">Selections remain browser-local until exported. Final labels change only after an identity-validated boundary review is finalized.</footer></main><script id="boundary-data" type="application/json">{json.dumps(payload, allow_nan=False)}</script><script>{_JS}</script></body></html>'''
    page_path = root / "precision_boundary_review.html"
    _atomic_write_text(page_path, page)
    proposal_count = sum(
        bool(row.get("conservative_available"))
        or bool(row.get("expanded_available"))
        for row in output_rows
    )
    summary = {
        "schema_version": 1,
        "review_type": "oocyte_precision_boundary_review_pack",
        "sample": sample.review_identity,
        "precision_review_json": str(precision_path),
        "precision_review_json_sha256": _file_sha256(precision_path),
        "boundary_card_count": len(output_rows),
        "proposal_count": proposal_count,
        "manual_only_count": len(output_rows) - proposal_count,
        "page": str(page_path),
        "candidates": str(candidates_path),
        "automatic_review": (
            None if automatic_review_path is None else str(automatic_review_path)
        ),
        "production_outputs_modified": False,
    }
    _atomic_write_text(
        root / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
    )
    return PrecisionBoundaryReviewResult(
        page_path=page_path,
        candidates_path=candidates_path,
        assets_dir=assets_dir,
        masks_dir=masks_dir,
        card_count=len(output_rows),
        proposal_count=proposal_count,
        manual_only_count=len(output_rows) - proposal_count,
        automatic_review_path=automatic_review_path,
    )


__all__ = [
    "BOUNDARY_RECOVERY_PARAMETERS",
    "PrecisionBoundaryReviewResult",
    "generate_precision_boundary_review",
]
