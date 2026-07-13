"""Identity-bound polygon review for unresolved Precision boundaries."""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .io import load_candidate_mask
from .manual_seed_finalize import _atomic_write_csv
from .manual_seed_review import _draw_mask
from .precision_boundary_review import _review_key
from .recall_review import (
    RecallReviewRuntime,
    _atomic_save_image,
    _atomic_write_text,
    _file_sha256,
    _json_safe,
    _load_sample,
    _mask_path,
    _read_json,
)


PRECISION_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION = 1
PRECISION_MANUAL_BOUNDARY_RENDERER_VERSION = "precision_manual_polygon_v1"


@dataclass(frozen=True)
class PrecisionManualBoundaryReviewResult:
    page_path: Path
    candidates_path: Path
    assets_dir: Path
    card_count: int


def _verify_precision_resolved_delivery(
    base_dir: Path,
    *,
    sample_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    root = Path(base_dir).resolve()
    manifest_path = root / "precision_resolved_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Precision-resolved manifest does not exist: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("delivery_name") != "precision_resolved_v1":
        raise ValueError("base directory is not a precision_resolved_v1 delivery")
    if manifest.get("sample") != sample_identity:
        raise ValueError("base Precision-resolved sample identity does not match")
    if manifest.get("release_ready") is not False:
        raise ValueError("base Precision intermediate has an invalid release state")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("base Precision manifest is missing artifacts")
    for relative_path, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid base artifact record: {relative_path}")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"base Precision artifact is missing: {path}")
        if not path.is_relative_to(root):
            raise ValueError(f"base Precision artifact is outside its delivery: {path}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"base Precision artifact size mismatch: {relative_path}")
        if _file_sha256(path) != str(record.get("sha256", "")):
            raise ValueError(f"base Precision artifact SHA-256 mismatch: {relative_path}")
    required = (
        "precision_resolved_candidates.csv",
        "precision_review_decisions.csv",
        "manual_boundary_queue.csv",
        "oocyte_labels_precision_resolved_v1.ome.tiff",
        "oocyte_labels_precision_resolved_v1_mapping.csv",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"base Precision delivery is missing {name}")
    return manifest


def _resolved_mask_path(base_dir: Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row["mask_path"]))
    if path.is_absolute():
        return path.resolve()
    source_dir = row.get("mask_source_dir")
    if source_dir is None or pd.isna(source_dir) or not str(source_dir).strip():
        root = Path(base_dir)
    else:
        root = Path(str(source_dir))
    return (root / path).resolve()


def _intersects_patch(row: Mapping[str, Any], patch: Any) -> bool:
    return not (
        float(row["bbox_x0"]) >= patch.bbox.x1
        or float(row["bbox_y0"]) >= patch.bbox.y1
        or float(row["bbox_x1"]) <= patch.bbox.x0
        or float(row["bbox_y1"]) <= patch.bbox.y0
    )


def _render_assets(
    runtime: RecallReviewRuntime,
    *,
    center_xy: tuple[int, int],
    radius: int,
    current_mask_path: Path,
    resolved_candidates: pd.DataFrame,
    base_dir: Path,
    raw_path: Path,
    context_path: Path,
) -> tuple[Any, int]:
    patch = runtime.source.read_patch(center_xy, radius)
    raw = Image.open(
        io.BytesIO(runtime.render_patch(center_xy, radius, "local"))
    ).convert("RGBA")
    context = raw.copy()
    neighbor_count = 0
    for row in resolved_candidates.to_dict("records"):
        if not _intersects_patch(row, patch):
            continue
        mask_path = _resolved_mask_path(base_dir, row)
        if not mask_path.is_file() or not mask_path.is_relative_to(base_dir):
            raise ValueError(f"resolved mask is outside the base delivery: {mask_path}")
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


def _card_html(row: Mapping[str, Any]) -> str:
    key = html.escape(str(row["review_key"]), quote=True)
    note = html.escape(str(row["boundary_review_notes"]), quote=True)
    return f'''<article class="card" data-id="{key}" data-review="unreviewed">
  <div class="canvas-shell"><canvas width="{int(row['asset_width_px'])}" height="{int(row['asset_height_px'])}" data-raw="review_assets/{html.escape(str(row['raw_asset_name']), quote=True)}" data-context="review_assets/{html.escape(str(row['context_asset_name']), quote=True)}" aria-label="Manual boundary editor for {key}"></canvas></div>
  <div class="body"><div class="title"><h2>#{int(row['manual_index']):03d} / {html.escape(str(row['display_id']))}</h2><span>manual polygon</span></div>
  <div class="mono">{key} &middot; x {float(row['center_x']):.1f} &middot; y {float(row['center_y']):.1f}</div>
  <p class="instruction"><strong>Prior instruction:</strong> {html.escape(str(row['boundary_review_notes']))}</p>
  <div class="status"><span class="point-count">0 vertices</span><span class="choice-label">Unreviewed</span></div>
  <div class="edit-actions"><button data-action="undo">Undo point</button><button data-action="clear">Clear</button><button data-action="toggle-masks">Hide masks</button></div>
  <div class="review-actions"><button data-choice="accept_manual_contour">Accept contour</button><button data-choice="exclude">Not oocyte</button><button data-choice="unsure">Unsure</button></div>
  <input class="notes" value="{note}" placeholder="Manual-boundary note">
  </div></article>'''


_CSS = r'''
:root{--ink:#172522;--paper:#f4ecdd;--panel:#fffaf0;--line:#c8bca7;--teal:#047f78;--cyan:#00ebdc;--orange:#f2662e;--yellow:#ffcd37;--red:#a94336;--shadow:rgba(30,45,38,.14)}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 9% 0,#fff9e9 0,transparent 31%),linear-gradient(135deg,#e5dac7,#f6f0e4 62%,#dbe9e1);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}.shell{width:min(1500px,calc(100% - 24px));margin:auto}.hero{margin:16px 0;padding:25px 29px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(112deg,#fffaf0,#dff1e9);box-shadow:0 16px 38px var(--shadow)}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{text-transform:uppercase;letter-spacing:.15em;color:var(--teal);font-size:.74rem;font-weight:700}.hero h1{font-size:clamp(2.4rem,5vw,5rem);line-height:.92;margin:.18em 0}.hero p{max-width:1080px;line-height:1.5}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:15px}.steps span{padding:10px 12px;border-radius:10px;background:rgba(255,250,240,.78);border:1px solid #d7cbb8}.toolbar{position:sticky;top:6px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 12px;margin:14px 0;border:1px solid var(--line);border-radius:14px;background:rgba(255,250,240,.95);backdrop-filter:blur(9px)}button,.button,input{font:inherit;border:1px solid #b9ae9d;border-radius:10px;padding:8px 11px;background:#fffaf0;color:var(--ink)}button{cursor:pointer}.button{text-decoration:none}.grow{flex:1}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px;margin-bottom:50px}.card{overflow:hidden;border:1px solid var(--line);border-radius:20px;background:var(--panel);box-shadow:0 11px 27px rgba(32,45,39,.1)}.card.hidden{display:none}.canvas-shell{padding:12px;background:#17201d}.card canvas{display:block;width:100%;height:auto;border-radius:10px;touch-action:none;cursor:crosshair;background:#080b0a}.body{padding:15px}.title{display:flex;align-items:center;justify-content:space-between}.title h2{margin:0;font-size:1.8rem}.title span{font:12px monospace;color:var(--orange);text-transform:uppercase}.instruction{min-height:3em;padding:9px 11px;border-left:4px solid var(--yellow);background:#f7ead0;line-height:1.35}.status{display:flex;justify-content:space-between;margin:8px 0;font:12px monospace}.edit-actions,.review-actions{display:grid;gap:7px;margin-top:7px}.edit-actions{grid-template-columns:repeat(3,1fr)}.review-actions{grid-template-columns:2fr 1fr 1fr}.review-actions button.selected[data-choice=accept_manual_contour]{background:#cfece0;border-color:var(--teal)}.review-actions button.selected[data-choice=exclude]{background:#f0d4cd;border-color:var(--red)}.review-actions button.selected[data-choice=unsure]{background:#ebe1c8;border-color:#9d8243}.notes{width:100%;margin-top:8px}.footer{padding:8px 0 48px;color:#5f6a64}@media(max-width:760px){.shell{width:min(100% - 10px,1500px)}.hero{padding:18px}.steps{grid-template-columns:1fr}.toolbar{position:static}.cards{grid-template-columns:1fr}.review-actions{grid-template-columns:1fr}}
'''


_JS = r'''
const DATA=JSON.parse(document.getElementById('manual-boundary-data').textContent),KEY='aegle-oocyte-manual-boundary:'+DATA.identity.sample_id+':'+DATA.identity.manual_boundary_candidate_table_sha256;let state=JSON.parse(localStorage.getItem(KEY)||'{}'),filter='all';const cards=[...document.querySelectorAll('.card')],images=new Map(),dragging=new Map();function rowFor(id){return DATA.rows.find(r=>r.review_key===id)}function cardState(id){return state[id]||(state[id]={points:[],choice:'',notes:'',showMasks:true})}function save(){localStorage.setItem(KEY,JSON.stringify(state));progress()}function canvasPoint(canvas,event){const rect=canvas.getBoundingClientRect();return[(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height]}function nearest(points,p,canvas){const scale=canvas.width/canvas.getBoundingClientRect().width,limit=12*scale;let best=-1,d=Infinity;points.forEach((q,i)=>{const v=Math.hypot(q[0]-p[0],q[1]-p[1]);if(v<d){d=v;best=i}});return d<=limit?best:-1}function invalidate(s){s.choice=''}function render(card){const id=card.dataset.id,row=rowFor(id),s=cardState(id),canvas=card.querySelector('canvas'),ctx=canvas.getContext('2d'),pair=images.get(id);if(pair){ctx.drawImage(s.showMasks?pair.context:pair.raw,0,0,canvas.width,canvas.height)}else{ctx.fillStyle='#101714';ctx.fillRect(0,0,canvas.width,canvas.height)}const pts=s.points||[];if(pts.length){ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);pts.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));if(pts.length>=3){ctx.closePath();ctx.fillStyle='rgba(242,102,46,.18)';ctx.fill()}ctx.strokeStyle='#f2662e';ctx.lineWidth=2;ctx.stroke();pts.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],i===0?5:4,0,Math.PI*2);ctx.fillStyle=i===0?'#ffcd37':'#fffaf0';ctx.fill();ctx.strokeStyle='#172522';ctx.lineWidth=1;ctx.stroke()})}ctx.strokeStyle='#ff5739';ctx.lineWidth=2;const x=row.center_local_x,y=row.center_local_y;ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.moveTo(x-13,y);ctx.lineTo(x+13,y);ctx.moveTo(x,y-13);ctx.lineTo(x,y+13);ctx.stroke();card.dataset.review=s.choice||'unreviewed';card.querySelector('.point-count').textContent=pts.length+' vertices';card.querySelector('.choice-label').textContent=({accept_manual_contour:'Contour accepted',exclude:'Not oocyte',unsure:'Unsure'})[s.choice]||'Unreviewed';card.querySelectorAll('[data-choice]').forEach(b=>b.classList.toggle('selected',b.dataset.choice===s.choice));card.querySelector('[data-action=toggle-masks]').textContent=s.showMasks?'Hide masks':'Show masks';if(!card.querySelector('.notes').dataset.loaded){card.querySelector('.notes').value=s.notes||row.boundary_review_notes||'';card.querySelector('.notes').dataset.loaded='1'}}function loadImages(card){const id=card.dataset.id,canvas=card.querySelector('canvas'),raw=new Image(),context=new Image();let count=0;function ready(){if(++count===2){images.set(id,{raw,context});render(card)}}raw.onload=ready;context.onload=ready;raw.src=canvas.dataset.raw;context.src=canvas.dataset.context}cards.forEach(card=>{const id=card.dataset.id,canvas=card.querySelector('canvas');cardState(id);loadImages(card);canvas.onpointerdown=e=>{e.preventDefault();canvas.setPointerCapture(e.pointerId);const s=cardState(id),p=canvasPoint(canvas,e),hit=nearest(s.points,p,canvas);if(hit>=0){dragging.set(id,hit)}else{s.points.push(p);dragging.set(id,s.points.length-1);invalidate(s);save();render(card)}};canvas.onpointermove=e=>{if(!dragging.has(id))return;const s=cardState(id),p=canvasPoint(canvas,e),i=dragging.get(id);s.points[i]=[Math.max(0,Math.min(canvas.width-1,p[0])),Math.max(0,Math.min(canvas.height-1,p[1]))];invalidate(s);render(card)};canvas.onpointerup=()=>{if(dragging.delete(id)){save();render(card)}};canvas.onpointercancel=canvas.onpointerup;card.querySelector('[data-action=undo]').onclick=()=>{const s=cardState(id);s.points.pop();invalidate(s);save();render(card)};card.querySelector('[data-action=clear]').onclick=()=>{const s=cardState(id);s.points=[];invalidate(s);save();render(card)};card.querySelector('[data-action=toggle-masks]').onclick=()=>{const s=cardState(id);s.showMasks=!s.showMasks;save();render(card)};card.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{const s=cardState(id);if(b.dataset.choice==='accept_manual_contour'&&s.points.length<3){alert('Add at least three contour vertices first.');return}s.choice=b.dataset.choice;save();render(card);apply()});card.querySelector('.notes').onchange=e=>{const s=cardState(id);s.notes=e.target.value;save()};render(card)});function progress(){const n=DATA.rows.filter(r=>cardState(r.review_key).choice).length;document.getElementById('progress').textContent=n+' / '+DATA.rows.length+' reviewed'}function apply(){cards.forEach(c=>c.classList.toggle('hidden',filter==='unreviewed'&&c.dataset.review!=='unreviewed'))}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;apply()});function identityMatches(a,b){const keys=['sample_id','candidate_table_sha256','base_precision_resolved_manifest_sha256','manual_boundary_candidate_table_sha256'];return keys.every(k=>a&&b&&a[k]===b[k])}function exportReview(){const rows=DATA.rows.map(r=>{const s=cardState(r.review_key),vertices=(s.points||[]).map(p=>[Number((r.patch_origin_x+p[0]).toFixed(3)),Number((r.patch_origin_y+p[1]).toFixed(3))]);return{...r,manual_boundary_choice:s.choice||'',manual_boundary_notes:s.notes||r.boundary_review_notes||'',vertices_xy:vertices,vertex_count:vertices.length}}),payload={schema_version:1,review_type:'oocyte_precision_manual_boundary_review',identity:DATA.identity,exported_at:new Date().toISOString(),rows},blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=DATA.identity.sample_id+'_precision_manual_boundary_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}document.getElementById('export-json').onclick=exportReview;const importer=document.getElementById('import-json');document.getElementById('import-button').onclick=()=>importer.click();importer.onchange=async()=>{const file=importer.files[0];if(!file)return;try{const payload=JSON.parse(await file.text());if(payload.schema_version!==1||payload.review_type!=='oocyte_precision_manual_boundary_review'||!identityMatches(payload.identity,DATA.identity))throw new Error('Review identity does not match this page.');const incoming=new Map((payload.rows||[]).map(r=>[r.review_key,r]));if(incoming.size!==DATA.rows.length||DATA.rows.some(r=>!incoming.has(r.review_key)))throw new Error('Review rows do not match this queue.');DATA.rows.forEach(r=>{const v=incoming.get(r.review_key),points=(v.vertices_xy||[]).map(p=>[Number(p[0])-r.patch_origin_x,Number(p[1])-r.patch_origin_y]);state[r.review_key]={points,choice:v.manual_boundary_choice||'',notes:v.manual_boundary_notes||'',showMasks:true}});save();cards.forEach(render);apply()}catch(error){alert(error.message)}finally{importer.value=''}};progress();apply();
'''


def generate_precision_manual_boundary_review(
    sample_dir: Path,
    base_resolved_dir: Path,
    out_dir: Path,
    *,
    patch_radius_px: int = 220,
) -> PrecisionManualBoundaryReviewResult:
    """Generate a polygon editor for a Precision intermediate's manual queue."""

    if patch_radius_px < 64:
        raise ValueError("manual-boundary patch radius must be at least 64 pixels")
    sample = _load_sample(sample_dir)
    base_dir = Path(base_resolved_dir).resolve()
    base_manifest = _verify_precision_resolved_delivery(
        base_dir,
        sample_identity=sample.review_identity,
    )
    base_manifest_path = base_dir / "precision_resolved_manifest.json"
    base_manifest_sha256 = _file_sha256(base_manifest_path)
    queue_path = base_dir / "manual_boundary_queue.csv"
    queue = pd.read_csv(queue_path)
    required_queue = {
        "review_index",
        "boundary_index",
        "review_key",
        "display_id",
        "detector_component_id",
        "detection_pass",
        "center_x",
        "center_y",
        "boundary_review_notes",
        "current_mask_path",
    }
    missing = required_queue.difference(queue.columns)
    if missing:
        raise ValueError(f"manual boundary queue is missing columns: {sorted(missing)}")
    if queue.empty:
        raise ValueError("Precision intermediate has no unresolved manual boundaries")
    if queue["review_key"].astype(str).duplicated().any():
        raise ValueError("manual boundary queue contains duplicate review keys")
    expected_count = int(base_manifest.get("unresolved_manual_count", -1))
    if len(queue) != expected_count:
        raise ValueError("manual boundary queue count does not match base manifest")
    resolved_candidates_path = base_dir / "precision_resolved_candidates.csv"
    resolved_candidates = pd.read_csv(resolved_candidates_path)
    decisions = pd.read_csv(base_dir / "precision_review_decisions.csv")
    decision_by_key = decisions.set_index("review_key", drop=False)
    sample_by_key = {
        _review_key(row): row for row in sample.candidates.to_dict("records")
    }

    root = Path(out_dir).resolve()
    assets_dir = root / "review_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    with RecallReviewRuntime(sample.sample_dir) as runtime:
        for manual_index, queue_row in enumerate(queue.to_dict("records"), start=1):
            key = str(queue_row["review_key"])
            if key not in sample_by_key or key not in decision_by_key.index:
                raise ValueError(f"manual boundary key is absent from source decisions: {key}")
            decision = decision_by_key.loc[key]
            if str(decision["resolution_state"]) != "manual_boundary_required":
                raise ValueError(f"manual boundary decision state changed for {key}")
            source = sample_by_key[key]
            if str(source["detector_component_id"]) != str(
                queue_row["detector_component_id"]
            ):
                raise ValueError(f"manual boundary candidate ID changed for {key}")
            current_path = _mask_path(sample.sample_dir, source).resolve()
            if current_path != Path(str(queue_row["current_mask_path"])).resolve():
                raise ValueError(f"manual boundary current mask path changed for {key}")
            if not current_path.is_file():
                raise FileNotFoundError(f"manual boundary current mask is missing: {current_path}")
            center = (
                int(round(float(queue_row["center_x"]))),
                int(round(float(queue_row["center_y"]))),
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
                raw_path=assets_dir / raw_name,
                context_path=assets_dir / context_name,
            )
            patch_origin_x = center[0] - patch_radius_px
            patch_origin_y = center[1] - patch_radius_px
            output_rows.append(
                {
                    "manual_index": manual_index,
                    "review_index": int(queue_row["review_index"]),
                    "boundary_index": int(queue_row["boundary_index"]),
                    "review_key": key,
                    "display_id": str(queue_row["display_id"]),
                    "detector_component_id": str(queue_row["detector_component_id"]),
                    "detection_pass": str(queue_row["detection_pass"]),
                    "center_x": float(queue_row["center_x"]),
                    "center_y": float(queue_row["center_y"]),
                    "center_local_x": float(queue_row["center_x"]) - patch_origin_x,
                    "center_local_y": float(queue_row["center_y"]) - patch_origin_y,
                    "precision_notes": str(queue_row.get("precision_notes", "")),
                    "boundary_review_notes": str(queue_row["boundary_review_notes"]),
                    "current_mask_path": str(current_path),
                    "current_mask_sha256": _file_sha256(current_path),
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
        str(row[name])
        for row in output_rows
        for name in ("raw_asset_name", "context_asset_name")
    }
    for existing in assets_dir.glob("*.webp"):
        if existing.name not in expected_assets:
            existing.unlink()
    candidates_path = root / "precision_manual_boundary_candidates.csv"
    _atomic_write_csv(pd.DataFrame(output_rows), candidates_path)
    identity = {
        **sample.review_identity,
        "base_precision_resolved_dir": str(base_dir),
        "base_precision_resolved_manifest": str(base_manifest_path),
        "base_precision_resolved_manifest_sha256": base_manifest_sha256,
        "base_resolved_candidates_sha256": _file_sha256(resolved_candidates_path),
        "manual_boundary_queue_sha256": _file_sha256(queue_path),
        "manual_boundary_candidate_table": str(candidates_path),
        "manual_boundary_candidate_table_sha256": _file_sha256(candidates_path),
        "renderer_version": PRECISION_MANUAL_BOUNDARY_RENDERER_VERSION,
        "patch_radius_px": patch_radius_px,
    }
    payload = {
        "identity": identity,
        "rows": [_json_safe(row) for row in output_rows],
    }
    cards = "".join(_card_html(row) for row in output_rows)
    script_payload = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>{html.escape(sample.sample_id)} Manual boundary review</title><style>{_CSS}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle / exact manual oocyte boundary</div><h1>{html.escape(sample.sample_id)} contour desk</h1><p>Trace only the intended outer oocyte boundary on native UCHL1. Cyan marks already resolved oocytes; yellow is the rejected current target mask; orange is your polygon. Browser points are evidence only until the Python finalizer validates and rasterizes the exported JSON.</p><div class="steps"><span><strong>1.</strong> Click around the intended boundary.</span><span><strong>2.</strong> Drag handles to refine; hide masks when needed.</span><span><strong>3.</strong> Accept each contour and export JSON.</span></div></header><nav class="toolbar"><button data-filter="all">All</button><button data-filter="unreviewed">Unreviewed</button><span id="progress" class="mono"></span><span class="grow"></span><button id="import-button">Import JSON</button><input id="import-json" type="file" accept="application/json" hidden><button id="export-json">Export JSON</button><a class="button" href="../review_console.html">Sample console</a></nav><section class="cards">{cards}</section><footer class="footer">The first vertex is yellow. Avoid cyan neighbors and follicular halo. Any edited contour returns to Unreviewed until explicitly accepted again.</footer></main><script id="manual-boundary-data" type="application/json">{script_payload}</script><script>{_JS}</script></body></html>'''
    page_path = root / "precision_manual_boundary_review.html"
    _atomic_write_text(page_path, page)
    summary = {
        "schema_version": 1,
        "review_type": "oocyte_precision_manual_boundary_review_pack",
        "sample": sample.review_identity,
        "base_precision_resolved_manifest": str(base_manifest_path),
        "base_precision_resolved_manifest_sha256": base_manifest_sha256,
        "manual_boundary_card_count": len(output_rows),
        "page": str(page_path),
        "candidates": str(candidates_path),
        "production_outputs_modified": False,
        "base_precision_outputs_modified": False,
    }
    _atomic_write_text(
        root / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
    )
    return PrecisionManualBoundaryReviewResult(
        page_path=page_path,
        candidates_path=candidates_path,
        assets_dir=assets_dir,
        card_count=len(output_rows),
    )


__all__ = [
    "PRECISION_MANUAL_BOUNDARY_RENDERER_VERSION",
    "PRECISION_MANUAL_BOUNDARY_REVIEW_SCHEMA_VERSION",
    "PrecisionManualBoundaryReviewResult",
    "generate_precision_manual_boundary_review",
]
