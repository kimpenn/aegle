"""Static side-by-side review for human-seeded provisional oocyte masks."""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .models import BoundingBox, PersistedMask
from .recall_review import (
    RecallReviewRuntime,
    _atomic_write_text,
    _file_sha256,
    _identity_contains_required,
    _json_safe,
    _load_sample,
    _read_json,
)
from .recall_overlay import overlay_dir_from_identity


@dataclass(frozen=True)
class ManualSeedReviewResult:
    page_path: Path
    assets_dir: Path
    card_count: int


def _load_provisional(path: Any) -> PersistedMask | None:
    if path is None or pd.isna(path) or not str(path).strip():
        return None
    source = Path(str(path))
    if not source.is_file():
        return None
    with np.load(source, allow_pickle=False) as archive:
        mask = np.asarray(archive["mask"], dtype=np.bool_)
        bbox = BoundingBox(*(int(value) for value in archive["bbox_xyxy"].tolist()))
        image_shape = tuple(int(value) for value in archive["image_shape_yx"].tolist())
        metadata = json.loads(str(archive["metadata_json"].item()))
    return PersistedMask(
        mask=mask,
        bbox=bbox,
        image_shape_yx=(image_shape[0], image_shape[1]),
        metadata=metadata,
    )


def _draw_mask(
    image: Image.Image,
    placed: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA")).copy()
    if placed.any():
        rgb = rgba[..., :3]
        foreground = rgb[placed].astype(np.float32)
        rgb[placed] = np.asarray(
            0.78 * foreground + 0.22 * np.asarray(color, dtype=np.float32),
            dtype=np.uint8,
        )
        boundary = np.logical_xor(placed, ndi.binary_erosion(placed))
        boundary = ndi.binary_dilation(boundary, iterations=1)
        rgb[boundary] = np.asarray(color, dtype=np.uint8)
        rgba[..., 3][boundary] = 255
    return Image.fromarray(rgba)


def _crosshair(image: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(image)
    color = (255, 87, 57, 255)
    draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=color, width=3)
    draw.line((x - 16, y, x + 16, y), fill=color, width=3)
    draw.line((x, y - 16, x, y + 16), fill=color, width=3)


def _panel_label(image: Image.Image, label: str, *, color: tuple[int, int, int]) -> Image.Image:
    output = Image.new("RGBA", (image.width, image.height + 34), (23, 31, 28, 255))
    output.paste(image.convert("RGBA"), (0, 34))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, image.width, 34), fill=(23, 31, 28, 255))
    draw.text((12, 9), label, fill=(*color, 255))
    return output


def _render_card(
    runtime: RecallReviewRuntime,
    row: Mapping[str, Any],
    destination: Path,
    *,
    radius: int,
) -> None:
    center = (int(round(float(row["x"]))), int(round(float(row["y"]))))
    patch = runtime.source.read_patch(center, radius)
    raw = Image.open(io.BytesIO(runtime.render_patch(center, radius, "local"))).convert("RGBA")
    existing = Image.open(io.BytesIO(runtime.render_overlay(center, radius))).convert("RGBA")
    base = Image.alpha_composite(raw, existing)
    requested_x0 = center[0] - radius
    requested_y0 = center[1] - radius
    click_x = int(round(float(row["x"]))) - requested_x0
    click_y = int(round(float(row["y"]))) - requested_y0

    existing_panel = base.copy()
    _crosshair(existing_panel, click_x, click_y)
    existing_panel = _panel_label(
        existing_panel,
        "RAW + EXISTING MASKS + MANUAL CENTER",
        color=(0, 255, 242),
    )

    panels = [existing_panel]
    for prefix, color, label in (
        ("manual_conservative", (87, 255, 127), "CONSERVATIVE PROVISIONAL MASK"),
        ("manual_expanded", (255, 211, 72), "EXPANDED PROVISIONAL MASK"),
    ):
        provisional = _load_provisional(row.get(f"{prefix}_mask_path"))
        panel = base.copy()
        if provisional is not None:
            placed = runtime._place_mask(provisional, patch)
            panel = _draw_mask(panel, placed, color=color)
        _crosshair(panel, click_x, click_y)
        suffix = "" if provisional is not None else " (UNAVAILABLE)"
        panels.append(_panel_label(panel, label + suffix, color=color))

    gutter = 5
    canvas = Image.new(
        "RGBA",
        (sum(panel.width for panel in panels) + gutter * (len(panels) - 1), panels[0].height),
        (245, 238, 224, 255),
    )
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width + gutter
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(destination, format="WEBP", quality=88, method=6)


def _metric(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _card(row: Mapping[str, Any], index: int) -> str:
    annotation_id = html.escape(str(row["annotation_id"]), quote=True)
    card_id = f"seed-{index:03d}"
    failure_class = html.escape(str(row["failure_class"]))
    note_value = "" if pd.isna(row.get("notes")) else str(row.get("notes", ""))
    return f'''<article class="card" id="{card_id}" data-id="{annotation_id}" data-review="unreviewed">
  <img src="review_assets/{card_id}.webp" loading="lazy" alt="Conservative and expanded provisional masks for {annotation_id}">
  <div class="body"><div class="title"><h2>#{index:03d}</h2><span>{failure_class}</span></div>
  <div class="mono">{annotation_id} · x {_metric(row['x'],1)} · y {_metric(row['y'],1)}</div>
  <div class="metrics"><span>conservative P{_metric(row.get('manual_conservative_percentile'),0)} · d {_metric(row.get('manual_conservative_equivalent_diameter_um'),1)} um</span><span>conservative circ {_metric(row.get('manual_conservative_circularity'))}</span><span>conservative solid {_metric(row.get('manual_conservative_solidity'))}</span><span>expanded P{_metric(row.get('manual_expanded_percentile'),0)} · d {_metric(row.get('manual_expanded_equivalent_diameter_um'),1)} um</span><span>expanded circ {_metric(row.get('manual_expanded_circularity'))}</span><span>expanded solid {_metric(row.get('manual_expanded_solidity'))}</span></div>
  <div class="actions"><button data-choice="accept_manual_conservative">Use conservative</button><button data-choice="accept_manual_expanded">Use expanded</button><button data-choice="neither">Neither</button><button data-choice="duplicate">Duplicate</button><button data-choice="unsure">Unsure</button></div>
  <input class="notes" value="{html.escape(note_value, quote=True)}" placeholder="Mask review note">
  </div></article>'''


_CSS = r'''
:root{--ink:#172522;--panel:#fffaf0;--paper:#eee6d8;--teal:#087b78;--line:#d1c6b3;--amber:#d97b24;--red:#a94336;--green:#458d68}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 12% 0,#fff9ec 0,transparent 28%),linear-gradient(135deg,#e9dfce,#f7f1e5 65%,#e0ece5);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}.shell{width:min(1680px,calc(100% - 24px));margin:auto}.hero{margin:16px 0;padding:24px 28px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(115deg,#fffaf0,#e4f3ec);box-shadow:0 15px 35px rgba(30,45,38,.12)}.hero h1{font-size:clamp(2rem,4.5vw,4.4rem);line-height:.95;margin:.2em 0}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{text-transform:uppercase;letter-spacing:.14em;color:var(--teal);font-size:.75rem;font-weight:700}.hero p{max-width:1000px;line-height:1.5}.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{padding:10px 14px;border:1px solid var(--line);border-radius:12px;background:#fffaf0}.toolbar{position:sticky;top:6px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 12px;margin:14px 0;border:1px solid var(--line);border-radius:14px;background:rgba(255,250,240,.95);backdrop-filter:blur(8px)}button,.button,input{font:inherit;border:1px solid #b9ae9d;border-radius:10px;padding:8px 11px;background:#fffaf0;color:var(--ink)}button{cursor:pointer}.button{text-decoration:none}.grow{flex:1}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(650px,1fr));gap:14px;margin-bottom:50px}.card{background:var(--panel);border:1px solid var(--line);border-radius:17px;overflow:hidden;box-shadow:0 9px 24px rgba(32,45,39,.08)}.card.hidden{display:none}.card img{width:100%;display:block;background:#171f1c}.body{padding:13px}.title{display:flex;align-items:center;justify-content:space-between}.title h2{margin:0}.title span{font-family:monospace;color:var(--amber)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:10px 0;font:12px monospace}.metrics span{background:#f0e7d8;padding:6px;border-radius:7px}.actions{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.actions button.selected[data-choice=accept_manual_conservative]{background:#d8f0dd;border-color:var(--green)}.actions button.selected[data-choice=accept_manual_expanded]{background:#f6e6a9;border-color:#bd9022}.actions button.selected[data-choice=neither]{background:#f1d6cf;border-color:var(--red)}.actions button.selected[data-choice=duplicate]{background:#dae5ee;border-color:#557b98}.actions button.selected[data-choice=unsure]{background:#eee3ca;border-color:#9f834a}.notes{width:100%;margin-top:8px}.footer{color:#68726c;padding:10px 0 50px}@media(max-width:720px){.shell{width:min(100% - 10px,1680px)}.hero{padding:18px}.cards{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.actions{grid-template-columns:1fr 1fr}.toolbar{position:static}}
'''


_JS = r'''
const DATA=JSON.parse(document.getElementById('seed-data').textContent),KEY='aegle-oocyte-manual-seed-review:'+DATA.identity.sample_id+':'+DATA.identity.analysis_sha256;let state=JSON.parse(localStorage.getItem(KEY)||'{}'),filter='all';const cards=[...document.querySelectorAll('.card')];function save(){localStorage.setItem(KEY,JSON.stringify(state));paintProgress()}function paint(card){const id=card.dataset.id,s=state[id]||{};card.dataset.review=s.choice||'unreviewed';card.querySelectorAll('[data-choice]').forEach(b=>b.classList.toggle('selected',b.dataset.choice===s.choice));card.querySelector('.notes').value=s.notes??card.querySelector('.notes').value}function paintProgress(){const reviewed=Object.values(state).filter(s=>s.choice).length;document.getElementById('progress').textContent=reviewed+' / '+DATA.rows.length+' masks reviewed'}function apply(){cards.forEach(c=>c.classList.toggle('hidden',filter==='unreviewed'&&c.dataset.review!=='unreviewed'))}cards.forEach(paint);document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>{const card=b.closest('.card'),id=card.dataset.id;state[id]={...(state[id]||{}),choice:b.dataset.choice};paint(card);save();apply()});document.querySelectorAll('.notes').forEach(n=>n.onchange=()=>{const card=n.closest('.card'),id=card.dataset.id;state[id]={...(state[id]||{}),notes:n.value};save()});document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;apply()});function exportData(type){const rows=DATA.rows.map(r=>({...r,manual_mask_choice:(state[r.annotation_id]||{}).choice||'',manual_notes:(state[r.annotation_id]||{}).notes||''})),payload={schema_version:1,review_type:'manual_seed_mask_review',identity:DATA.identity,exported_at:new Date().toISOString(),rows};let blob,name;if(type==='json'){blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});name=DATA.identity.sample_id+'_manual_seed_mask_review.json'}else{const keys=Object.keys(rows[0]||{}),esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';blob=new Blob([[keys.join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\n')],{type:'text/csv'});name=DATA.identity.sample_id+'_manual_seed_mask_review.csv'}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}document.getElementById('export-json').onclick=()=>exportData('json');document.getElementById('export-csv').onclick=()=>exportData('csv');paintProgress();apply();
'''


def generate_manual_seed_review(
    sample_dir: Path,
    analysis_dir: Path,
    *,
    patch_radius_px: int = 220,
) -> ManualSeedReviewResult:
    """Render each manual click with conservative and expanded mask overlays."""

    root = Path(analysis_dir).resolve()
    analysis_summary_path = root / "summary.json"
    overlay_dir = None
    expected_identity = None
    if analysis_summary_path.is_file():
        analysis_summary = _read_json(analysis_summary_path)
        expected_identity = analysis_summary.get("sample")
        if isinstance(expected_identity, Mapping):
            overlay_dir = overlay_dir_from_identity(expected_identity)
    sample = _load_sample(sample_dir, overlay_dir=overlay_dir)
    if expected_identity is not None and not _identity_contains_required(
        expected_identity,
        sample.review_identity,
    ):
        raise ValueError("recall analysis identity does not match current sample overlay")
    table_path = root / "recall_failure_analysis.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"recall analysis table does not exist: {table_path}")
    table = pd.read_csv(table_path)
    required = {
        "annotation_id",
        "x",
        "y",
        "manual_conservative_mask_path",
        "manual_expanded_mask_path",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"recall analysis table missing columns: {sorted(missing)}")
    assets_dir = root / "review_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"seed-{index:03d}.webp" for index in range(1, len(table) + 1)}
    for existing in assets_dir.glob("*.webp"):
        if existing.name not in expected:
            existing.unlink()
    with RecallReviewRuntime(sample.sample_dir, overlay_dir=overlay_dir) as runtime:
        for index, row in enumerate(table.to_dict("records"), start=1):
            destination = assets_dir / f"seed-{index:03d}.webp"
            _render_card(runtime, row, destination, radius=patch_radius_px)

    rows = [_json_safe(row) for row in table.to_dict("records")]
    identity: Dict[str, Any] = {
        **(
            dict(expected_identity)
            if isinstance(expected_identity, Mapping)
            else sample.review_identity
        ),
        "analysis_sha256": _file_sha256(table_path),
        "analysis_table": str(table_path),
        "patch_radius_px": patch_radius_px,
    }
    payload = {"identity": identity, "rows": rows}
    cards = "".join(_card(row, index) for index, row in enumerate(table.to_dict("records"), start=1))
    failure_counts = table["failure_class"].value_counts().to_dict() if not table.empty else {}
    stats = "".join(
        f'<span class="stat"><strong>{int(count)}</strong> {html.escape(str(name))}</span>'
        for name, count in failure_counts.items()
    )
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sample.sample_id)} manual-seed mask review</title><style>{_CSS}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle / human-seeded recall delta</div><h1>{html.escape(sample.sample_id)} provisional mask review</h1><p>Each row compares the same raw UCHL1 patch and manual center. Cyan is an existing accepted mask, green is the highest-percentile click-targeted conservative result, and yellow is a bounded expansion of the same component. Choose one mask only when it isolates the intended oocyte; choose Neither for a false click or two bad segmentations, and Duplicate when this click repeats another card.</p><div class="stats"><span class="stat"><strong>{len(table)}</strong> manual clicks</span>{stats}</div></header><nav class="toolbar"><button data-filter="all">All</button><button data-filter="unreviewed">Unreviewed</button><span id="progress" class="mono"></span><span class="grow"></span><button id="export-json">Export JSON</button><button id="export-csv">Export CSV</button><a class="button" href="../recall_review.html">Back to recall review</a></nav><section class="cards">{cards}</section><footer class="footer">Selections remain browser-local until exported. No provisional mask is part of the production label image.</footer></main><script id="seed-data" type="application/json">{json.dumps(payload, allow_nan=False)}</script><script>{_JS}</script></body></html>'''
    page_path = root / "manual_seed_review.html"
    _atomic_write_text(page_path, page)
    return ManualSeedReviewResult(
        page_path=page_path,
        assets_dir=assets_dir,
        card_count=len(table),
    )


__all__ = ["ManualSeedReviewResult", "generate_manual_seed_review"]
