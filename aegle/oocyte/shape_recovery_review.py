"""Targeted review pack for shape-gated manual-seed mask expansion."""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from .manual_seed_finalize import _atomic_write_csv, _validate_review_identity
from .manual_seed_review import _crosshair, _draw_mask, _load_provisional, _panel_label
from .recall_review import (
    RecallReviewRuntime,
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _read_json,
    _save_provisional_mask,
    _validate_review_payload,
)


SHAPE_RECOVERY_PARAMETERS = {
    "max_area_ratio": 6.0,
    "min_equivalent_diameter_um": 20.0,
    "min_circularity": 0.80,
    "min_solidity": 0.90,
    "max_centroid_offset_px": 25.0,
    "min_conservative_overlap": 0.70,
}


@dataclass(frozen=True)
class ShapeRecoveryReviewResult:
    page_path: Path
    candidates_path: Path
    assets_dir: Path
    card_count: int


def _render_shape_card(
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
    click_x = int(round(float(row["x"]))) - (center[0] - radius)
    click_y = int(round(float(row["y"]))) - (center[1] - radius)
    first = base.copy()
    _crosshair(first, click_x, click_y)
    panels = [
        _panel_label(
            first,
            f"#{int(row['review_index']):03d} RAW + EXISTING MASKS + CENTER",
            color=(0, 255, 242),
        )
    ]
    for path_key, color, label in (
        (
            "current_mask_path",
            (255, 211, 72),
            f"V4 REVIEWED / P{float(row['current_percentile']):.0f} / "
            f"d {float(row['current_equivalent_diameter_um']):.1f} um",
        ),
        (
            "shape_recovery_mask_path",
            (255, 132, 67),
            f"SHAPE RECOVERY / P{float(row['shape_recovery_percentile']):.0f} / "
            f"d {float(row['shape_recovery_equivalent_diameter_um']):.1f} um",
        ),
    ):
        persisted = _load_provisional(row[path_key])
        if persisted is None:
            raise FileNotFoundError(f"shape-review mask is unavailable: {row[path_key]}")
        panel = _draw_mask(
            base.copy(),
            runtime._place_mask(persisted, patch),
            color=color,
        )
        _crosshair(panel, click_x, click_y)
        panels.append(_panel_label(panel, label, color=color))
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(destination, format="WEBP", quality=89, method=6)


def _metric(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


def _card_html(row: Mapping[str, Any]) -> str:
    annotation_id = html.escape(str(row["annotation_id"]), quote=True)
    review_index = int(row["review_index"])
    previous_note = "" if pd.isna(row.get("previous_manual_notes")) else str(
        row.get("previous_manual_notes", "")
    )
    previous_choice = html.escape(str(row.get("previous_manual_mask_choice", "")))
    return f'''<article class="card" data-id="{annotation_id}" data-review="unreviewed">
  <img src="review_assets/shape-{review_index:03d}.webp" loading="lazy" alt="Shape recovery comparison for {annotation_id}">
  <div class="body"><div class="title"><h2>#{review_index:03d}</h2><span>{html.escape(str(row['failure_class']))}</span></div>
  <div class="mono">{annotation_id} · previous {previous_choice} · area growth {_metric(row['shape_to_current_area_ratio'])}x</div>
  <div class="metrics"><span>v4 d {_metric(row['current_equivalent_diameter_um'],1)} um</span><span>v4 circ {_metric(row['current_circularity'])}</span><span>v4 solid {_metric(row['current_solidity'])}</span><span>recovery d {_metric(row['shape_recovery_equivalent_diameter_um'],1)} um</span><span>recovery circ {_metric(row['shape_recovery_circularity'])}</span><span>recovery solid {_metric(row['shape_recovery_solidity'])}</span></div>
  <div class="actions"><button data-choice="keep_v4">Keep v4</button><button data-choice="accept_shape_recovery">Use recovery</button><button data-choice="exclude">Exclude</button><button data-choice="unsure">Unsure</button></div>
  <input class="notes" value="{html.escape(previous_note, quote=True)}" placeholder="Shape-recovery review note">
  </div></article>'''


_CSS = r'''
:root{--ink:#172522;--panel:#fffaf0;--line:#d1c6b3;--teal:#087b78;--orange:#d9662b;--yellow:#b58a17;--red:#a94336}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 10% 0,#fff8e8 0,transparent 30%),linear-gradient(135deg,#eadfcd,#f8f2e7 66%,#dcebe3);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}.shell{width:min(1500px,calc(100% - 24px));margin:auto}.hero{margin:16px 0;padding:24px 28px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(115deg,#fffaf0,#e4f3ec);box-shadow:0 15px 35px rgba(30,45,38,.12)}.hero h1{font-size:clamp(2rem,4.5vw,4.2rem);line-height:.95;margin:.2em 0}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{text-transform:uppercase;letter-spacing:.14em;color:var(--teal);font-size:.75rem;font-weight:700}.hero p{max-width:1000px;line-height:1.5}.toolbar{position:sticky;top:6px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 12px;margin:14px 0;border:1px solid var(--line);border-radius:14px;background:rgba(255,250,240,.95);backdrop-filter:blur(8px)}button,.button,input{font:inherit;border:1px solid #b9ae9d;border-radius:10px;padding:8px 11px;background:#fffaf0;color:var(--ink)}button{cursor:pointer}.button{text-decoration:none}.grow{flex:1}.cards{display:grid;grid-template-columns:1fr;gap:16px;margin-bottom:50px}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:0 9px 24px rgba(32,45,39,.08)}.card.hidden{display:none}.card img{width:100%;display:block;background:#171f1c}.body{padding:14px}.title{display:flex;align-items:center;justify-content:space-between}.title h2{margin:0}.title span{font-family:monospace;color:var(--orange)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:10px 0;font:12px monospace}.metrics span{background:#f0e7d8;padding:7px;border-radius:7px}.actions{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.actions button.selected[data-choice=keep_v4]{background:#f6e6a9;border-color:var(--yellow)}.actions button.selected[data-choice=accept_shape_recovery]{background:#ffd7bd;border-color:var(--orange)}.actions button.selected[data-choice=exclude]{background:#f1d6cf;border-color:var(--red)}.actions button.selected[data-choice=unsure]{background:#e5e1d7;border-color:#847d70}.notes{width:100%;margin-top:8px}.footer{padding:10px 0 50px;color:#68726c}@media(max-width:720px){.shell{width:calc(100% - 10px)}.hero{padding:18px}.metrics{grid-template-columns:1fr 1fr}.actions{grid-template-columns:1fr 1fr}.toolbar{position:static}}
'''


_JS = r'''
const DATA=JSON.parse(document.getElementById('shape-data').textContent),KEY='aegle-oocyte-shape-review:'+DATA.identity.sample_id+':'+DATA.identity.shape_candidate_table_sha256;let state=JSON.parse(localStorage.getItem(KEY)||'{}'),filter='all';const cards=[...document.querySelectorAll('.card')];function save(){localStorage.setItem(KEY,JSON.stringify(state));paintProgress()}function paint(card){const id=card.dataset.id,s=state[id]||{};card.dataset.review=s.choice||'unreviewed';card.querySelectorAll('[data-choice]').forEach(b=>b.classList.toggle('selected',b.dataset.choice===s.choice));card.querySelector('.notes').value=s.notes??card.querySelector('.notes').value}function paintProgress(){const reviewed=Object.values(state).filter(s=>s.choice).length;document.getElementById('progress').textContent=reviewed+' / '+DATA.rows.length+' reviewed'}function apply(){cards.forEach(c=>c.classList.toggle('hidden',filter==='unreviewed'&&c.dataset.review!=='unreviewed'))}cards.forEach(paint);document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{const card=b.closest('.card'),id=card.dataset.id;state[id]={...(state[id]||{}),choice:b.dataset.choice};paint(card);save();apply()});document.querySelectorAll('.notes').forEach(n=>n.onchange=()=>{const card=n.closest('.card'),id=card.dataset.id;state[id]={...(state[id]||{}),notes:n.value};save()});document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;apply()});function exportData(type){const rows=DATA.rows.map(r=>({...r,shape_review_choice:(state[r.annotation_id]||{}).choice||'',shape_review_notes:(state[r.annotation_id]||{}).notes||''})),payload={schema_version:1,review_type:'manual_seed_shape_recovery_review',identity:DATA.identity,exported_at:new Date().toISOString(),rows};let blob,name;if(type==='json'){blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});name=DATA.identity.sample_id+'_shape_recovery_review.json'}else{const keys=Object.keys(rows[0]||{}),esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';blob=new Blob([[keys.join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\n')],{type:'text/csv'});name=DATA.identity.sample_id+'_shape_recovery_review.csv'}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}document.getElementById('export-json').onclick=()=>exportData('json');document.getElementById('export-csv').onclick=()=>exportData('csv');paintProgress();apply();
'''


def generate_shape_recovery_review(
    sample_dir: Path,
    recall_review_json: Path,
    manual_review_json: Path,
    out_dir: Path,
    *,
    patch_radius_px: int = 220,
) -> ShapeRecoveryReviewResult:
    """Generate a delta page only for masks changed by shape-gated expansion."""

    sample = _load_sample(sample_dir)
    recall_payload = _read_json(Path(recall_review_json))
    _validate_review_payload(sample, recall_payload)
    manual_path = Path(manual_review_json).resolve()
    manual_payload = _read_json(manual_path)
    _validate_review_identity(sample.review_identity, manual_payload)
    identity = manual_payload["identity"]
    analysis_path = Path(str(identity.get("analysis_table", ""))).resolve()
    if not analysis_path.is_file():
        raise FileNotFoundError(f"recall analysis table does not exist: {analysis_path}")
    if _file_sha256(analysis_path) != str(identity.get("analysis_sha256", "")):
        raise ValueError("recall analysis SHA-256 does not match manual review identity")
    analysis = pd.read_csv(analysis_path)
    analysis_by_id = analysis.set_index("annotation_id", drop=False)
    manual_rows = manual_payload.get("rows")
    if not isinstance(manual_rows, list):
        raise ValueError("manual-seed mask review rows must be a list")
    centers = [
        (float(annotation["x"]), float(annotation["y"]))
        for annotation in recall_payload["missing_oocytes"]
    ]
    center_by_id = {
        str(annotation["annotation_id"]): center
        for annotation, center in zip(recall_payload["missing_oocytes"], centers)
    }
    root = Path(out_dir).resolve()
    masks_dir = root / "shape_recovery_masks"
    assets_dir = root / "review_assets"
    masks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with RecallReviewRuntime(sample.sample_dir) as runtime:
        for review_index, manual_row in enumerate(manual_rows, start=1):
            if not isinstance(manual_row, Mapping):
                raise ValueError(f"manual review row {review_index} must be an object")
            annotation_id = str(manual_row["annotation_id"])
            if annotation_id not in center_by_id or annotation_id not in analysis_by_id.index:
                raise ValueError(f"manual review annotation is not in recall analysis: {annotation_id}")
            x, y = center_by_id[annotation_id]
            other_centers = tuple(
                center for other_id, center in center_by_id.items() if other_id != annotation_id
            )
            standard = runtime.segment_manual_provisionals(
                x,
                y,
                exclude_points_xy=other_centers,
            )
            recovered = runtime.segment_manual_provisionals(
                x,
                y,
                exclude_points_xy=other_centers,
                allow_shape_recovery=True,
            )
            if standard.expanded is None or recovered.expanded is None:
                continue
            if np.array_equal(standard.expanded.mask, recovered.expanded.mask):
                continue
            if recovered.expanded.metrics.area_px <= standard.expanded.metrics.area_px:
                continue
            source = analysis_by_id.loc[annotation_id]
            current_path = Path(str(source["manual_expanded_mask_path"])).resolve()
            if not current_path.is_file():
                raise FileNotFoundError(f"current v4 mask is missing: {current_path}")
            recovery_path = masks_dir / f"shape-{review_index:03d}.npz"
            _save_provisional_mask(
                recovery_path,
                result=recovered.expanded,
                patch=recovered.patch,
                annotation_id=annotation_id,
                percentile=float(recovered.expanded_percentile),
            )
            rows.append(
                {
                    "review_index": review_index,
                    "annotation_id": annotation_id,
                    "x": x,
                    "y": y,
                    "failure_class": str(source["failure_class"]),
                    "previous_manual_mask_choice": str(
                        manual_row.get("manual_mask_choice", "")
                    ),
                    "previous_manual_notes": str(manual_row.get("manual_notes", "")),
                    "current_mask_path": str(current_path),
                    "current_percentile": float(source["manual_expanded_percentile"]),
                    "current_area_px": int(standard.expanded.metrics.area_px),
                    "current_equivalent_diameter_um": float(
                        standard.expanded.metrics.equivalent_diameter_um
                    ),
                    "current_circularity": float(standard.expanded.metrics.circularity),
                    "current_solidity": float(standard.expanded.metrics.solidity),
                    "shape_recovery_mask_path": str(recovery_path),
                    "shape_recovery_percentile": float(recovered.expanded_percentile),
                    "shape_recovery_area_px": int(recovered.expanded.metrics.area_px),
                    "shape_recovery_equivalent_diameter_um": float(
                        recovered.expanded.metrics.equivalent_diameter_um
                    ),
                    "shape_recovery_circularity": float(
                        recovered.expanded.metrics.circularity
                    ),
                    "shape_recovery_solidity": float(recovered.expanded.metrics.solidity),
                    "shape_recovery_centroid_offset_px": float(
                        recovered.expanded.metrics.centroid_offset_px
                    ),
                    "shape_to_current_area_ratio": float(
                        recovered.expanded.metrics.area_px
                        / max(standard.expanded.metrics.area_px, 1)
                    ),
                }
            )
        table = pd.DataFrame(rows)
        candidates_path = root / "shape_recovery_candidates.csv"
        _atomic_write_csv(table, candidates_path)
        expected_assets = {
            f"shape-{int(row['review_index']):03d}.webp" for row in rows
        }
        assets_dir.mkdir(parents=True, exist_ok=True)
        for existing in assets_dir.glob("*.webp"):
            if existing.name not in expected_assets:
                existing.unlink()
        for row in rows:
            _render_shape_card(
                runtime,
                row,
                assets_dir / f"shape-{int(row['review_index']):03d}.webp",
                radius=patch_radius_px,
            )
    if not rows:
        raise ValueError("shape recovery did not change any reviewed masks")

    page_identity: Dict[str, Any] = {
        **sample.review_identity,
        "analysis_table": str(analysis_path),
        "analysis_sha256": _file_sha256(analysis_path),
        "manual_review_json": str(manual_path),
        "manual_review_json_sha256": _file_sha256(manual_path),
        "recall_review_json": str(Path(recall_review_json).resolve()),
        "recall_review_json_sha256": _file_sha256(Path(recall_review_json)),
        "shape_candidate_table": str(candidates_path),
        "shape_candidate_table_sha256": _file_sha256(candidates_path),
        "shape_recovery_parameters": SHAPE_RECOVERY_PARAMETERS,
        "patch_radius_px": patch_radius_px,
    }
    safe_rows = [_json_safe(row) for row in rows]
    cards = "".join(_card_html(row) for row in rows)
    payload = {"identity": page_identity, "rows": safe_rows}
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sample.sample_id)} shape recovery review</title><style>{_CSS}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle / manual-seed shape delta</div><h1>{html.escape(sample.sample_id)} shape recovery review</h1><p>Only masks changed by the shape-gated expansion are shown. Yellow is the frozen v4 reviewed candidate; orange is the proposed recovery. Use recovery only when its outer contour better matches the intended oocyte without absorbing a neighbor. This page does not modify the completed v1 delivery.</p></header><nav class="toolbar"><button data-filter="all">All</button><button data-filter="unreviewed">Unreviewed</button><span id="progress" class="mono"></span><span class="grow"></span><button id="export-json">Export JSON</button><button id="export-csv">Export CSV</button><a class="button" href="/recall_analysis_v4/manual_seed_review.html">Back to v4 review</a></nav><section class="cards">{cards}</section><footer class="footer">Selections remain browser-local until exported. A new combined v2 is created only after this delta review is ingested.</footer></main><script id="shape-data" type="application/json">{json.dumps(payload, allow_nan=False)}</script><script>{_JS}</script></body></html>'''
    page_path = root / "shape_recovery_review.html"
    _atomic_write_text(page_path, page)
    summary = {
        "schema_version": 1,
        "sample": sample.review_identity,
        "shape_recovery_card_count": len(rows),
        "review_indices": [int(row["review_index"]) for row in rows],
        "page": str(page_path),
        "candidates": str(candidates_path),
        "production_outputs_modified": False,
    }
    _atomic_write_text(
        root / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
    )
    return ShapeRecoveryReviewResult(
        page_path=page_path,
        candidates_path=candidates_path,
        assets_dir=assets_dir,
        card_count=len(rows),
    )


__all__ = [
    "SHAPE_RECOVERY_PARAMETERS",
    "ShapeRecoveryReviewResult",
    "generate_shape_recovery_review",
]
