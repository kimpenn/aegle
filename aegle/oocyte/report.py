"""Self-contained algorithm documentation and per-sample HTML review pages."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tifffile
import zarr
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage as ndi

from .config import DONOR13_V6, DONOR13_V6_RESCUE_V1, OocyteDetectionConfig
from .export import export_whole_slide_labels
from .io import extract_cyx_channel_patch, load_candidate_mask
from .models import ExtractedPatch, PersistedMask
from .rescue import suppress_accepted_mask_duplicates


LOGGER = logging.getLogger(__name__)
THUMBNAIL_CACHE_SCHEMA_VERSION = 1
THUMBNAIL_RENDER_VERSION = "raw-mask-thumbnail-v2"
PRECISION_REVIEW_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HtmlReportResult:
    batch_dir: Path
    algorithm_document: Path
    batch_index: Path
    sample_pages: Dict[str, Path]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _thumbnail_fingerprint(
    row: Mapping[str, Any],
    *,
    source_image: Path,
    source_size_bytes: int,
    source_mtime_ns: int,
    channel_index: int,
    patch_radius_px: int,
    mask_path: Path,
) -> str:
    payload = {
        "renderer": THUMBNAIL_RENDER_VERSION,
        "source_image": str(Path(source_image).resolve()),
        "source_size_bytes": int(source_size_bytes),
        "source_mtime_ns": str(source_mtime_ns),
        "channel_index": int(channel_index),
        "patch_radius_px": int(patch_radius_px),
        "candidate_id": str(row["detector_component_id"]),
        "detection_pass": str(row.get("detection_pass", "baseline_v6")),
        "center_x": int(round(float(row["center_x"]))),
        "center_y": int(round(float(row["center_y"]))),
        "mask_path": str(Path(mask_path).resolve()),
        "mask_sha256": _file_sha256(mask_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_thumbnail_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        payload.get("schema_version") != THUMBNAIL_CACHE_SCHEMA_VERSION
        or payload.get("renderer") != THUMBNAIL_RENDER_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        return {}
    return payload


def _thumbnail_is_current(
    asset_path: Path,
    entry: Any,
    expected_fingerprint: str,
) -> bool:
    if not asset_path.is_file() or not isinstance(entry, dict):
        return False
    if entry.get("render_fingerprint") != expected_fingerprint:
        return False
    expected_sha = entry.get("asset_sha256")
    return isinstance(expected_sha, str) and _file_sha256(asset_path) == expected_sha


def _format_number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return str(value)


def _place_mask(mask: PersistedMask, patch: ExtractedPatch) -> np.ndarray:
    placed = np.zeros(patch.image.shape, dtype=np.bool_)
    x0 = max(mask.bbox.x0, patch.bbox.x0)
    y0 = max(mask.bbox.y0, patch.bbox.y0)
    x1 = min(mask.bbox.x1, patch.bbox.x1)
    y1 = min(mask.bbox.y1, patch.bbox.y1)
    if x0 >= x1 or y0 >= y1:
        return placed
    top, _, left, _ = patch.padding_tblr
    source = mask.mask[
        y0 - mask.bbox.y0 : y1 - mask.bbox.y0,
        x0 - mask.bbox.x0 : x1 - mask.bbox.x0,
    ]
    target_y = top + y0 - patch.bbox.y0
    target_x = left + x0 - patch.bbox.x0
    placed[target_y : target_y + source.shape[0], target_x : target_x + source.shape[1]] = source
    return placed


def _raw_mask_thumbnail_image(
    patch: ExtractedPatch,
    persisted_mask: PersistedMask,
    *,
    show_mask: bool = True,
    output_size_px: int = 420,
) -> Image.Image:
    raw = np.asarray(patch.image, dtype=np.float32)
    transformed = np.log1p(np.maximum(raw, 0.0))
    finite = transformed[np.isfinite(transformed)]
    lo, hi = np.percentile(finite, [2.0, 99.8]) if finite.size else (0.0, 1.0)
    normalized = np.clip((transformed - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    rgb = np.asarray(colormaps["magma"](normalized)[..., :3] * 255.0, dtype=np.uint8)
    mask = _place_mask(persisted_mask, patch)
    if show_mask and mask.any():
        rgb[mask] = np.asarray(
            0.82 * rgb[mask] + 0.18 * np.array([0, 245, 236]),
            dtype=np.uint8,
        )
        boundary = np.logical_xor(mask, ndi.binary_erosion(mask))
        rgb[ndi.binary_dilation(boundary, iterations=1)] = np.array([0, 255, 246])
    center_y = rgb.shape[0] // 2
    center_x = rgb.shape[1] // 2
    rgb[max(0, center_y - 6) : center_y + 7, center_x] = 255
    rgb[center_y, max(0, center_x - 6) : center_x + 7] = 255
    image = Image.fromarray(rgb)
    return image.resize((output_size_px, output_size_px), Image.Resampling.LANCZOS)


def render_raw_mask_thumbnail_bytes(
    patch: ExtractedPatch,
    persisted_mask: PersistedMask,
    *,
    show_mask: bool = True,
    output_size_px: int = 420,
    quality: int = 88,
) -> bytes:
    """Render one raw-UCHL1 patch as self-contained WebP bytes."""

    image = _raw_mask_thumbnail_image(
        patch,
        persisted_mask,
        show_mask=show_mask,
        output_size_px=output_size_px,
    )
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=6)
    return buffer.getvalue()


def _raw_mask_thumbnail(
    patch: ExtractedPatch,
    persisted_mask: PersistedMask,
    destination: Path,
    *,
    output_size_px: int = 420,
) -> None:
    image_bytes = render_raw_mask_thumbnail_bytes(
        patch,
        persisted_mask,
        output_size_px=output_size_px,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".webp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        temporary_path.write_bytes(image_bytes)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_references(path: Path | None, sample_id: str) -> np.ndarray:
    if path is None:
        return np.empty((0, 2), dtype=np.float64)
    coordinates = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("sample_id", "")) != sample_id:
                continue
            if float(record.get("final_score", 0.0)) < 0.35:
                continue
            center = record.get("center")
            if center is not None and len(center) == 2:
                coordinates.append((float(center[0]), float(center[1])))
    return np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)


def _annotate_reference_distance(table: pd.DataFrame, reference_xy: np.ndarray) -> pd.DataFrame:
    output = table.copy()
    if output.empty or reference_xy.size == 0:
        output["nearest_reference_distance_px"] = np.nan
        output["reference_class"] = "unavailable" if reference_xy.size == 0 else "novel"
        return output
    candidate_xy = output[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    distances = np.sqrt(
        ((candidate_xy[:, None, :] - reference_xy[None, :, :]) ** 2).sum(axis=2)
    )
    nearest = distances.min(axis=1)
    output["nearest_reference_distance_px"] = nearest
    output["reference_class"] = np.where(nearest <= 100.0, "reference-linked", "detector-only")
    return output


def _candidate_quality(row: Mapping[str, Any]) -> str:
    score = float(row.get("detector_score", 0.0))
    diameter = float(row.get("local_equivalent_diameter_um", 0.0))
    circularity = float(row.get("local_circularity", 0.0))
    if (
        str(row.get("rescue_acceptance_rule", ""))
        in {"bright_irregular", "bright_fragment", "baseline_shape_fallback"}
        or score < 0.50
        or diameter < 20.0
        or circularity < 0.70
    ):
        return "review-priority"
    return "standard"


def _precision_review_key(row: Mapping[str, Any]) -> str:
    return (
        f"{str(row.get('detection_pass', 'baseline_v6'))}:"
        f"{str(row['detector_component_id'])}"
    )


def _precision_review_rows(candidates: pd.DataFrame) -> List[Dict[str, Any]]:
    payload_columns = [
        "html_id",
        "display_id",
        "detector_component_id",
        "detection_pass",
        "detector_score",
        "center_x",
        "center_y",
        "local_equivalent_diameter_um",
        "local_circularity",
        "local_solidity",
        "reference_class",
        "nearest_reference_distance_px",
        "quality_class",
        "acceptance_mode",
        "rescue_acceptance_rule",
    ]
    rows = []
    for row in candidates.reindex(columns=payload_columns).to_dict("records"):
        safe_row = {key: _json_safe(value) for key, value in row.items()}
        safe_row["review_key"] = _precision_review_key(safe_row)
        rows.append(safe_row)
    keys = [str(row["review_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("precision review keys must be unique within a sample")
    return rows


def _precision_review_identity(
    *,
    sample_id: str,
    source_image: Path,
    manifest: Mapping[str, Any],
    sample_summary: Mapping[str, Any],
    candidate_table_path: Path,
    candidate_count: int,
) -> Dict[str, Any]:
    source_path = Path(source_image).resolve()
    source_stat = source_path.stat()
    return {
        "sample_id": str(sample_id),
        "source_image": str(source_path),
        "source_image_size_bytes": int(source_stat.st_size),
        "source_image_mtime_ns": str(source_stat.st_mtime_ns),
        "profile_name": str(
            manifest.get("profile_name", sample_summary.get("profile_name", ""))
        ),
        "profile_fingerprint": str(manifest.get("profile_fingerprint", "")),
        "implementation_version": str(manifest.get("implementation_version", "")),
        "candidate_table_sha256": _file_sha256(candidate_table_path),
        "combined_candidate_count": int(candidate_count),
    }


def _spatial_svg(
    candidates: pd.DataFrame,
    image_shape_yx: Tuple[int, int],
) -> str:
    height, width = image_shape_yx
    view_width = 1000.0
    view_height = max(280.0, view_width * height / max(width, 1))
    circles = []
    for row in candidates.to_dict("records"):
        x = float(row["center_x"]) / max(width, 1) * view_width
        y = float(row["center_y"]) / max(height, 1) * view_height
        css = "rescue-dot" if row["detection_pass"] == "secondary_rescue" else "baseline-dot"
        candidate_id = html.escape(str(row["html_id"]), quote=True)
        label = html.escape(str(row["display_id"]), quote=True)
        circles.append(
            f'<circle class="map-dot {css}" cx="{x:.2f}" cy="{y:.2f}" r="7" '
            f'data-card="{candidate_id}" tabindex="0"><title>{label}</title></circle>'
        )
    return (
        f'<svg class="spatial-map" viewBox="0 0 {view_width:.0f} {view_height:.0f}" '
        'role="img" aria-label="Whole-slide oocyte coordinate map">'
        '<defs><pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">'
        '<path d="M 50 0 L 0 0 0 50" fill="none" stroke="#d8d0bf" stroke-width="1"/>'
        '</pattern></defs>'
        f'<rect width="{view_width:.0f}" height="{view_height:.0f}" rx="18" fill="#f7f0df"/>'
        f'<rect width="{view_width:.0f}" height="{view_height:.0f}" rx="18" fill="url(#grid)"/>'
        + "".join(circles)
        + "</svg>"
    )


_REPORT_CSS = r"""
:root{--ink:#172522;--paper:#f2ecdf;--panel:#fffaf0;--teal:#087b78;--cyan:#00d9cf;--amber:#d77a1f;--red:#a53b2d;--muted:#6e756f;--line:#d8d0bf;--shadow:0 16px 38px rgba(41,51,46,.12)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:radial-gradient(circle at 15% 0,#fff9e9 0,transparent 32%),linear-gradient(135deg,#eee5d5,#f7f2e8 65%,#e7eee7);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}.shell{width:min(1480px,calc(100% - 32px));margin:auto}.hero{margin:20px 0;padding:30px 34px;border:1px solid #c9c0ad;border-radius:24px;background:linear-gradient(120deg,rgba(255,250,240,.97),rgba(229,244,238,.92));box-shadow:var(--shadow);position:relative;overflow:hidden}.hero:after{content:"";position:absolute;width:260px;height:260px;border:40px solid rgba(8,123,120,.08);border-radius:50%;right:-90px;top:-110px}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.76rem;color:var(--teal);font-weight:700}.hero h1{font-size:clamp(2.1rem,5vw,4.6rem);line-height:.94;margin:.25em 0 .18em;max-width:930px}.hero p{max-width:850px;font-size:1.06rem;line-height:1.55}.stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}.stat{background:#fffaf0;border:1px solid var(--line);border-radius:14px;padding:12px 16px;min-width:145px}.stat strong{font-size:1.55rem;display:block}.toolbar{position:sticky;top:8px;z-index:20;margin:18px 0;padding:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border:1px solid var(--line);border-radius:16px;background:rgba(255,250,240,.94);backdrop-filter:blur(10px);box-shadow:0 8px 20px rgba(30,40,35,.1)}button,select,input{font:inherit}button,.button,select,input[type=search]{border:1px solid #bdb5a6;background:#fffaf0;color:var(--ink);border-radius:999px;padding:9px 14px;text-decoration:none}button{cursor:pointer}button:hover,.button:hover,.filter.active{border-color:var(--teal);background:#e2f2ed}.toolbar .spacer{flex:1}.review-message{width:100%;min-height:1em;color:var(--teal);font-size:.75rem}.review-message.error{color:var(--red)}.map-panel{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:18px;box-shadow:var(--shadow);margin:18px 0}.map-panel h2{margin:0 0 8px}.spatial-map{width:100%;max-height:520px}.map-dot{cursor:pointer;stroke:#fffaf0;stroke-width:2;transition:r .18s,opacity .18s}.map-dot:hover,.map-dot:focus{r:12;outline:none}.baseline-dot{fill:var(--teal)}.rescue-dot{fill:var(--amber)}.legend{display:flex;gap:18px;align-items:center;font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;font-size:.8rem}.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:16px;margin:18px 0 60px}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;overflow:hidden;box-shadow:0 8px 22px rgba(30,40,35,.08);animation:rise .45s both;transition:transform .18s,border-color .18s}.card:hover{transform:translateY(-3px);border-color:#9b968b}.card.review-priority{border-color:#d69a66}.card.hidden{display:none}.card-image{position:relative;aspect-ratio:1;background:#180c21}.card-image img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}.card-image img[hidden]{display:none!important}.mask-toggle{position:absolute;right:9px;bottom:9px;z-index:3;padding:6px 10px;border-radius:9px;background:rgba(255,250,240,.92);box-shadow:0 3px 10px rgba(20,22,20,.28);font:700 .7rem "IBM Plex Mono","Aptos Mono","Courier New",monospace}.mask-toggle.raw-visible{background:rgba(8,123,120,.94);border-color:#8ee6df;color:white}.mask-toggle:disabled{cursor:not-allowed;opacity:.82}.card-body{padding:14px}.card-title{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.card h3{margin:0;font-size:1.15rem}.badge{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;font-size:.69rem;padding:4px 7px;border-radius:999px;background:#dceeea;color:#075c5a}.badge.rescue{background:#f7dfbf;color:#8a4712}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:7px;margin:12px 0;font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;font-size:.73rem}.metrics span{background:#f3ecde;border-radius:8px;padding:6px}.review-actions{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.review-actions button{padding:7px 4px;font-size:.78rem}.review-actions button.selected[data-status=accept]{background:#d8eee5;border-color:#4b9b7d}.review-actions button.selected[data-status=reject]{background:#f3d9d2;border-color:#b65343}.review-actions button.selected[data-status=unsure]{background:#f6e8bd;border-color:#c18a2d}.note-presets{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.note-preset{padding:5px 8px;border-radius:8px;color:#59645f;background:#f3ecde;font:600 .63rem "IBM Plex Mono","Aptos Mono","Courier New",monospace}.note-preset.selected{color:#fff;background:var(--teal);border-color:var(--teal)}.notes{width:100%;border:1px solid var(--line);border-radius:9px;margin-top:8px;padding:7px;background:#fffdf7}.empty{padding:60px;text-align:center;border:1px dashed #bdb5a6;border-radius:18px;background:#fffaf0}.footer{padding:30px 0 60px;color:var(--muted)}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}@media(max-width:700px){.shell{width:min(100% - 18px,1480px)}.hero{padding:24px 20px}.toolbar{position:static}.toolbar .spacer{display:none}.cards{grid-template-columns:1fr}.card{animation:none}}
"""


_REPORT_JS = r"""
const PACKAGE=JSON.parse(document.getElementById('candidate-data').textContent),DATA=PACKAGE.rows,IDENTITY=PACKAGE.identity,KEY='aegle-oocyte-precision-review:'+IDENTITY.sample_id+':'+IDENTITY.candidate_table_sha256;let state={};try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(_error){state={}}let filter='all';const cards=[...document.querySelectorAll('.card')],allowedStatuses=new Set(['accept','reject','unsure']);function save(){localStorage.setItem(KEY,JSON.stringify(state));progress()}function paint(card){const key=card.dataset.reviewKey,s=state[key]||{},notes=s.notes||'';card.dataset.review=s.status||'unreviewed';card.querySelectorAll('.review-actions button').forEach(b=>b.classList.toggle('selected',b.dataset.status===s.status));card.querySelectorAll('.note-preset').forEach(b=>b.classList.toggle('selected',b.dataset.note===notes));card.querySelector('.notes').value=notes}function progress(){const reviewed=DATA.filter(row=>allowedStatuses.has((state[row.review_key]||{}).status)).length;document.getElementById('review-progress').textContent=reviewed+' / '+DATA.length+' reviewed'}function apply(){const q=document.getElementById('search').value.toLowerCase();cards.forEach(c=>{const matchFilter=filter==='all'||(filter==='unreviewed'&&c.dataset.review==='unreviewed')||(filter==='flagged'&&c.dataset.quality==='review-priority')||c.dataset.pass===filter||c.dataset.rule===filter;const matchSearch=!q||c.dataset.search.includes(q);c.classList.toggle('hidden',!(matchFilter&&matchSearch))})}function message(text,isError=false){const node=document.getElementById('review-message');node.textContent=text;node.classList.toggle('error',isError)}cards.forEach(paint);document.querySelectorAll('.review-actions button').forEach(b=>b.addEventListener('click',()=>{const card=b.closest('.card'),key=card.dataset.reviewKey;state[key]={...(state[key]||{}),status:b.dataset.status};paint(card);save();apply()}));document.querySelectorAll('.note-preset').forEach(b=>b.addEventListener('click',()=>{const card=b.closest('.card'),key=card.dataset.reviewKey,current=(state[key]||{}).notes||'',notes=current===b.dataset.note?'':b.dataset.note;state[key]={...(state[key]||{}),notes};paint(card);save()}));document.querySelectorAll('.notes').forEach(n=>n.addEventListener('input',()=>{const card=n.closest('.card'),key=card.dataset.reviewKey;state[key]={...(state[key]||{}),notes:n.value};card.querySelectorAll('.note-preset').forEach(b=>b.classList.toggle('selected',b.dataset.note===n.value));save()}));document.querySelectorAll('.filter').forEach(b=>b.addEventListener('click',()=>{filter=b.dataset.filter;document.querySelectorAll('.filter').forEach(x=>x.classList.toggle('active',x===b));apply()}));document.getElementById('search').addEventListener('input',apply);document.getElementById('sort').addEventListener('change',e=>{const mode=e.target.value,grid=document.querySelector('.cards');cards.sort((a,b)=>mode==='score'?+b.dataset.score-+a.dataset.score:mode==='position'?+a.dataset.y-+b.dataset.y:+a.dataset.index-+b.dataset.index).forEach(c=>grid.appendChild(c))});document.querySelectorAll('.map-dot').forEach(dot=>dot.addEventListener('click',()=>document.getElementById(dot.dataset.card).scrollIntoView({behavior:'smooth',block:'center'})));function reviewedRows(){return DATA.map(row=>({...row,manual_status:(state[row.review_key]||{}).status||'',manual_notes:(state[row.review_key]||{}).notes||''}))}function exportRows(type){const rows=reviewedRows();let blob,name;if(type==='json'){const payload={schema_version:1,review_type:'oocyte_precision_review',identity:IDENTITY,exported_at:new Date().toISOString(),rows};blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});name=IDENTITY.sample_id+'_oocyte_review.json'}else{const keys=Object.keys(rows[0]||{}),esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';blob=new Blob([[keys.join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\n')],{type:'text/csv'});name=IDENTITY.sample_id+'_oocyte_review.csv'}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}function validateImport(payload){if(!payload||Array.isArray(payload)||payload.schema_version!==1||payload.review_type!=='oocyte_precision_review')throw new Error('Expected an identity-bound oocyte_precision_review JSON export');for(const field of ['sample_id','source_image','source_image_size_bytes','source_image_mtime_ns','profile_fingerprint','implementation_version','candidate_table_sha256','combined_candidate_count'])if(String((payload.identity||{})[field])!==String(IDENTITY[field]))throw new Error('Review identity mismatch: '+field);if(!Array.isArray(payload.rows))throw new Error('Review rows are missing');const expected=new Map(DATA.map(row=>[row.review_key,row])),next={},seen=new Set();for(const row of payload.rows){const key=row.review_key||String(row.detection_pass||'baseline_v6')+':'+String(row.detector_component_id||'');if(seen.has(key))throw new Error('Duplicate review row: '+key);seen.add(key);const current=expected.get(key);if(!current)throw new Error('Unknown candidate: '+key);if(String(row.detector_component_id)!==String(current.detector_component_id)||String(row.detection_pass)!==String(current.detection_pass)||Number(row.center_x)!==Number(current.center_x)||Number(row.center_y)!==Number(current.center_y))throw new Error('Candidate metadata mismatch: '+key);const status=String(row.manual_status||'').toLowerCase();if(status&&!allowedStatuses.has(status))throw new Error('Invalid review status for '+key);next[key]={status,notes:String(row.manual_notes||'')}}return next}document.getElementById('import-json').onclick=()=>document.getElementById('import-json-file').click();document.getElementById('import-json-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{state=validateImport(JSON.parse(await file.text()));cards.forEach(paint);save();apply();message('Imported '+progressCount()+' reviewed decisions from '+file.name)}catch(error){message(error.message,true)}finally{event.target.value=''}};function progressCount(){return DATA.filter(row=>allowedStatuses.has((state[row.review_key]||{}).status)).length}document.getElementById('export-csv').onclick=()=>exportRows('csv');document.getElementById('export-json').onclick=()=>exportRows('json');progress();apply();
document.querySelectorAll('.mask-toggle').forEach(button=>button.addEventListener('click',()=>{const host=button.closest('.card-image'),masked=host.querySelector('.masked-thumbnail'),raw=host.querySelector('.raw-thumbnail');if(button.dataset.rawVisible==='true'){raw.hidden=true;masked.hidden=false;button.dataset.rawVisible='false';button.textContent='Hide mask';button.classList.remove('raw-visible');return}const reveal=()=>{masked.hidden=true;raw.hidden=false;button.dataset.rawVisible='true';button.textContent='Show mask';button.classList.add('raw-visible');button.disabled=false};if(raw.dataset.loaded==='true'){reveal();return}button.disabled=true;button.textContent='Loading raw';raw.onload=()=>{raw.dataset.loaded='true';reveal()};raw.onerror=()=>{button.textContent='Raw requires review server';button.title='Open this page through the dynamic review server';button.disabled=true};raw.src=raw.dataset.src}));
"""


_PRECISION_NOTE_PRESETS = (
    ("Halo artifact", "halo_artifact"),
    ("True oocyte, bad mask", "true_oocyte; mask_truncated; mask_off_target"),
    ("Non-oocyte tissue", "non_oocyte_tissue; irregular_bright_patch"),
    ("Mask off target", "mask_off_target"),
)


def _candidate_card(
    row: Mapping[str, Any],
    order: int,
    *,
    patch_radius_px: int,
) -> str:
    card_id = html.escape(str(row["html_id"]), quote=True)
    display_id = html.escape(str(row["display_id"]))
    candidate_id = html.escape(str(row["detector_component_id"]))
    detection_pass = str(row["detection_pass"])
    rescue_rule = str(row.get("rescue_acceptance_rule", ""))
    rescue_labels = {
        "bright_irregular": "Rescue irregular",
        "bright_fragment": "Rescue fragment",
        "baseline_shape_fallback": "P99 fallback",
    }
    pass_label = rescue_labels.get(
        rescue_rule,
        "Rescue P95" if detection_pass == "secondary_rescue" else "Baseline v6",
    )
    badge_class = "rescue" if detection_pass == "secondary_rescue" else ""
    quality = str(row["quality_class"])
    search = html.escape(
        f"{display_id} {candidate_id} {detection_pass} {row.get('reference_class', '')}".lower(),
        quote=True,
    )
    image_path = html.escape(str(row["thumbnail_path"]), quote=True)
    raw_path = html.escape(
        "api/patch.webp?"
        f"x={int(round(float(row['center_x'])))}&"
        f"y={int(round(float(row['center_y'])))}&"
        f"radius={int(patch_radius_px)}&contrast=local",
        quote=True,
    )
    review_key = html.escape(_precision_review_key(row), quote=True)
    note_presets = "".join(
        '<button class="note-preset" type="button" '
        f'data-note="{html.escape(value, quote=True)}">'
        f"{html.escape(label)}</button>"
        for label, value in _PRECISION_NOTE_PRESETS
    )
    return f"""
<article class="card {quality}" id="{card_id}" data-id="{card_id}" data-review-key="{review_key}" data-index="{order}" data-score="{float(row['detector_score']):.8f}" data-y="{float(row['center_y']):.2f}" data-pass="{html.escape(detection_pass, quote=True)}" data-rule="{html.escape(rescue_rule, quote=True)}" data-quality="{quality}" data-review="unreviewed" data-search="{search}" style="animation-delay:{min(order,20)*25}ms">
  <div class="card-image">
    <img class="masked-thumbnail" src="{image_path}" loading="lazy" alt="Raw UCHL1 and exact mask for {display_id}">
    <img class="raw-thumbnail" data-src="{raw_path}" alt="Raw UCHL1 without mask for {display_id}" hidden>
    <button class="mask-toggle" type="button" data-raw-visible="false">Hide mask</button>
  </div>
  <div class="card-body">
    <div class="card-title"><h3>{display_id}</h3><span class="badge {badge_class}">{pass_label}</span></div>
    <div class="mono">{candidate_id}</div>
    <div class="metrics">
      <span>score {_format_number(row.get('detector_score'),3)}</span><span>d {_format_number(row.get('local_equivalent_diameter_um'),1)} um</span>
      <span>circ {_format_number(row.get('local_circularity'),2)}</span><span>solid {_format_number(row.get('local_solidity'),2)}</span>
      <span>x {int(round(float(row['center_x'])))}</span><span>y {int(round(float(row['center_y'])))}</span>
      <span>{html.escape(str(row.get('reference_class','unavailable')))}</span><span>ref {_format_number(row.get('nearest_reference_distance_px'),0)} px</span>
    </div>
    <div class="review-actions"><button data-status="accept">Accept</button><button data-status="reject">Reject</button><button data-status="unsure">Unsure</button></div>
    <div class="note-presets" aria-label="Biologist note presets">{note_presets}</div>
    <input class="notes" type="text" placeholder="Biologist note" aria-label="Review note for {display_id}">
  </div>
</article>"""


def _sample_html(
    sample_id: str,
    candidates: pd.DataFrame,
    image_shape_yx: Tuple[int, int],
    *,
    combined_labels_available: bool,
    patch_radius_px: int,
    review_identity: Mapping[str, Any],
) -> str:
    baseline_count = int((candidates["detection_pass"] == "baseline_v6").sum()) if not candidates.empty else 0
    rescue_count = int((candidates["detection_pass"] == "secondary_rescue").sum()) if not candidates.empty else 0
    flagged_count = int((candidates["quality_class"] == "review-priority").sum()) if not candidates.empty else 0
    cards = "".join(
        _candidate_card(row, index, patch_radius_px=patch_radius_px)
        for index, row in enumerate(candidates.to_dict("records"), start=1)
    )
    if not cards:
        cards = '<div class="empty"><h2>No machine-accepted oocytes</h2><p>This is a detector result, not evidence that the tissue contains no oocytes.</p></div>'
    payload = {
        "schema_version": PRECISION_REVIEW_SCHEMA_VERSION,
        "review_type": "oocyte_precision_review_workspace",
        "identity": dict(review_identity),
        "rows": _precision_review_rows(candidates),
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    spatial = _spatial_svg(candidates, image_shape_yx)
    labels_link = (
        '<a class="button" href="oocyte_labels_rescue_v1.ome.tiff">Combined OME labels</a>'
        if combined_labels_available
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sample_id)} oocyte review</title><style>{_REPORT_CSS}</style></head>
<body data-sample="{html.escape(sample_id,quote=True)}"><main class="shell"><header class="hero"><div class="eyebrow">Aegle / raw UCHL1 / biological review</div><h1>{html.escape(sample_id)} oocyte atlas</h1><p>Every cyan contour is loaded from the exact persisted mask used by the detector. Baseline v6 candidates are retained unchanged; secondary-rescue candidates recover crowded fields using robust P95 background estimation and mask-overlap deduplication.</p>{labels_link}<div class="stats"><div class="stat"><strong>{len(candidates)}</strong>combined</div><div class="stat"><strong>{baseline_count}</strong>baseline v6</div><div class="stat"><strong>{rescue_count}</strong>rescue delta</div><div class="stat"><strong>{flagged_count}</strong>review priority</div></div></header>
<nav class="toolbar"><button class="filter active" data-filter="all">All</button><button class="filter" data-filter="baseline_v6">Baseline</button><button class="filter" data-filter="secondary_rescue">Rescue</button><button class="filter" data-filter="bright_irregular">Bright irregular</button><button class="filter" data-filter="bright_fragment">Bright fragments</button><button class="filter" data-filter="baseline_shape_fallback">P99 fallback</button><button class="filter" data-filter="flagged">Flagged</button><button class="filter" data-filter="unreviewed">Unreviewed</button><input id="search" type="search" placeholder="Search ID or class"><select id="sort"><option value="index">Review order</option><option value="score">Score high-low</option><option value="position">Slide Y position</option></select><span class="spacer"></span><span id="review-progress" class="mono"></span><button id="import-json">Import JSON</button><input id="import-json-file" type="file" accept="application/json,.json" hidden><button id="export-csv">Export CSV</button><button id="export-json">Export JSON</button><a class="button" href="../oocyte_detection_algorithm.html">Algorithm</a><span id="review-message" class="review-message mono" role="status"></span></nav>
<section class="map-panel"><h2>Whole-slide position</h2><div class="legend"><span><i style="background:var(--teal)"></i>baseline v6</span><span><i style="background:var(--amber)"></i>secondary rescue</span></div>{spatial}</section><section class="cards">{cards}</section><footer class="footer">Review state is stored in this browser only until exported. Coordinates are full-resolution image pixels; physical scale is 0.5 um/px.</footer></main><script id="candidate-data" type="application/json">{payload_json}</script><script>{_REPORT_JS}</script></body></html>"""


def _config_table(config: OocyteDetectionConfig) -> str:
    rows = []
    for section, values in config.to_dict().items():
        if isinstance(values, dict):
            for key, value in values.items():
                rows.append(
                    f"<tr><td>{html.escape(section)}</td><td class=mono>{html.escape(str(key))}</td><td class=mono>{html.escape(str(value))}</td></tr>"
                )
    return "".join(rows)


def algorithm_document_html() -> str:
    """Return the complete standalone detector algorithm document."""

    v6_rows = _config_table(DONOR13_V6)
    rescue_rows = _config_table(DONOR13_V6_RESCUE_V1)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aegle raw-UCHL1 oocyte detector</title><style>{_REPORT_CSS}article{{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:24px;margin:18px 0;box-shadow:var(--shadow)}}article h2{{font-size:2rem;margin-top:0}}article p,article li{{line-height:1.62}}table{{border-collapse:collapse;width:100%;font-size:.82rem}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left}}.formula{{font-size:1.15rem;padding:16px;border-left:5px solid var(--teal);background:#e8f1eb}}.flow{{width:100%;height:auto}}.flow text{{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;fill:#172522}}.flow .box{{fill:#fffaf0;stroke:#087b78;stroke-width:3}}.flow .rescue-box{{fill:#fff0d9;stroke:#d77a1f;stroke-width:3}}.flow .arrow{{stroke:#172522;stroke-width:3;fill:none;marker-end:url(#arrow)}}code{{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle standalone scientific imaging module</div><h1>Raw-UCHL1 oocyte detection</h1><p>A detector and mask generator designed for ovarian oocytes that are fragmented by general-purpose DeepCell/Mesmer segmentation. It requires the registered raw fluorescence image and UCHL1 channel metadata, not DeepCell labels, nuclei, cell tables, or a legacy candidate list.</p><div class="stats"><div class="stat"><strong>UCHL1</strong>signal substrate</div><div class="stat"><strong>2-pass</strong>v6 + rescue</div><div class="stat"><strong>exact</strong>persisted masks</div></div></header>
<article><h2>1. Contract and rationale</h2><p>Oocytes are much larger than surrounding cells and can contain a dark germinal vesicle. DeepCell therefore splits one biological oocyte into several labels or leaves a nuclear hole. The standalone module instead uses the roughly 300-fold UCHL1 contrast, detects candidate locations over the whole slide, then performs local intensity segmentation at native resolution.</p><ul><li>Input: one CYX registered OME-TIFF, resolved UCHL1 channel index, pixel size, sample ID, and named profile.</li><li>Required external segmentation: none.</li><li>Output: candidate CSV, exact cropped NPZ masks, tiled uint16 whole-slide OME label image, mapping table, review/QC artifacts, and HTML review page.</li><li>Legacy references affect comparison displays only and never affect detection, masks, scores, or acceptance.</li></ul></article>
<article><h2>2. End-to-end flow</h2><svg class="flow" viewBox="0 0 1180 500" role="img" aria-label="Detector flow"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#172522"/></marker></defs><rect class="box" x="35" y="70" width="195" height="90" rx="18"/><text x="132" y="105" text-anchor="middle">Raw OME-TIFF</text><text x="132" y="132" text-anchor="middle">resolve UCHL1</text><path class="arrow" d="M230 115 H300"/><rect class="box" x="300" y="70" width="210" height="90" rx="18"/><text x="405" y="105" text-anchor="middle">Strip mean map</text><text x="405" y="132" text-anchor="middle">8x downsample</text><path class="arrow" d="M510 115 H580"/><rect class="box" x="580" y="70" width="230" height="90" rx="18"/><text x="695" y="105" text-anchor="middle">Coarse proposals</text><text x="695" y="132" text-anchor="middle">components + peaks + LoG</text><path class="arrow" d="M810 115 H880"/><rect class="box" x="880" y="70" width="255" height="90" rx="18"/><text x="1008" y="105" text-anchor="middle">Native local refinement</text><text x="1008" y="132" text-anchor="middle">multi-seed + compact retry</text><path class="arrow" d="M1008 160 V230"/><rect class="box" x="880" y="230" width="255" height="90" rx="18"/><text x="1008" y="265" text-anchor="middle">Score + v6 acceptance</text><text x="1008" y="292" text-anchor="middle">shape / size / contrast</text><path class="arrow" d="M880 275 H810"/><rect class="rescue-box" x="535" y="230" width="275" height="90" rx="18"/><text x="672" y="265" text-anchor="middle">Secondary rescue</text><text x="672" y="292" text-anchor="middle">P95 + all components</text><path class="arrow" d="M535 275 H465"/><rect class="box" x="245" y="230" width="220" height="90" rx="18"/><text x="355" y="265" text-anchor="middle">Mask-overlap dedup</text><text x="355" y="292" text-anchor="middle">actual component centroid</text><path class="arrow" d="M245 275 H175"/><rect class="box" x="35" y="230" width="140" height="90" rx="18"/><text x="105" y="265" text-anchor="middle">Persist</text><text x="105" y="292" text-anchor="middle">NPZ + OME</text><path class="arrow" d="M105 320 V390"/><rect class="box" x="35" y="390" width="1100" height="75" rx="18"/><text x="585" y="423" text-anchor="middle">Montage + spatial QC + filterable per-sample HTML biological review</text><text x="585" y="449" text-anchor="middle">all visualizations load the persisted scored mask</text></svg></article>
<article><h2>3. Whole-slide proposal generation</h2><p>The reader streams horizontal strips from the selected CYX channel. Each 8 x 8 block is reduced to its mean, avoiding a full-resolution in-memory copy. A log-transformed difference of Gaussians separates compact bright objects from broad tissue background. Candidate seeds are the union of thresholded connected components, component-local intensity peaks, globally separated peaks, and Laplacian-of-Gaussian blobs. Physical diameter limits and 60 px seed merging suppress puncta and redundant proposals while retaining several peaks in crowded follicles.</p></article>
<article><h2>4. Native-resolution v6 segmentation</h2><svg class="flow" viewBox="0 0 1180 300" role="img" aria-label="Local segmentation"><defs><marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#172522"/></marker></defs><circle cx="140" cy="145" r="95" fill="#351243"/><circle cx="140" cy="145" r="42" fill="#ffbd58"/><circle cx="140" cy="145" r="70" fill="none" stroke="#00d9cf" stroke-width="4"/><circle cx="140" cy="145" r="92" fill="none" stroke="#00d9cf" stroke-width="4" stroke-dasharray="8 8"/><text x="140" y="275" text-anchor="middle">patch + annulus</text><path d="M245 145 H340" stroke="#172522" stroke-width="3" marker-end="url(#arrow2)"/><rect class="box" x="340" y="75" width="250" height="140" rx="18"/><text x="465" y="115" text-anchor="middle">Gaussian smooth</text><text x="465" y="145" text-anchor="middle">triangle threshold</text><text x="465" y="175" text-anchor="middle">max(base, annulus P99)</text><path d="M590 145 H680" stroke="#172522" stroke-width="3" marker-end="url(#arrow2)"/><rect class="box" x="680" y="75" width="220" height="140" rx="18"/><text x="790" y="115" text-anchor="middle">remove small</text><text x="790" y="145" text-anchor="middle">close + fill holes</text><text x="790" y="175" text-anchor="middle">select component</text><path d="M900 145 H980" stroke="#172522" stroke-width="3" marker-end="url(#arrow2)"/><path d="M1010 90 C1090 70 1140 130 1110 200 C1050 235 985 190 1010 90Z" fill="#00d9cf" opacity=".65"/><text x="1060" y="275" text-anchor="middle">exact boolean mask</text></svg><p>Each coarse proposal generates a coarse seed, sharp local peaks, broad peaks, and offset-ring seeds. Default and compact contexts are evaluated, with one component-centroid reseed when appropriate. The center component wins; when a dark germinal vesicle leaves the seed in background, an area-over-distance fallback selects the nearby large component. Morphology removes small objects, closes short gaps, and fills enclosed nuclear holes.</p><div class="formula">score = 0.35 brightness + 0.30 shape + 0.20 size + 0.15 centeredness</div><p>Shape combines circularity and solidity. Size uses a soft physical range with a 25–80 um core. Acceptance retains the original v6 strict threshold and its conservative morphology rescue. Global v6 deduplication remains frozen for numerical reproducibility.</p></article>
<article><h2>5. Crowded-field secondary rescue</h2><p>P99 is precise for isolated oocytes, but a neighboring oocyte inside the background annulus can raise the floor until the target becomes a small fragment. The separate <code>donor13_v6_rescue_v1</code> profile never alters a v6 acceptance. It retries rejected v6 seeds in their original default or compact context at annulus P95, and independently enumerates all plausible P95 components in a 240 px discovery window around each coarse proposal. Every discovery is recentered and segmented again at native resolution.</p><p>If both default and compact P95 select a distant fallback component, one compact P80 evaluation is permitted. This is a targeted crowded-field escape hatch, not a global lower threshold. The rescue can also retain the original P99 mask when P95 merges neighbors. High-intensity irregular and bright-fragment rules are separately named and always marked review-priority.</p><p>Rescue candidates must satisfy physical size, circularity, solidity, score, local contrast, native intensity, and centroid-offset gates. Deduplication compares the actual persisted component masks: a rescue is suppressed when at least 25% of the smaller mask overlaps an accepted v6 or higher-ranked rescue mask. Seed coordinates are not used as the biological object center.</p></article>
<article><h2>6. Persistence and review invariants</h2><ul><li>Each candidate NPZ stores a boolean cropped mask, image-space bounding box, source shape, candidate ID, profile fingerprint, implementation version, and metrics.</li><li>The OME label image is recomposed only from accepted NPZ masks in descending score order; overlap pixels retain the higher-scoring label.</li><li>Montages and HTML thumbnails load persisted masks. A display can never silently rerun thresholding with newer code.</li><li>Thumbnail cache reuse requires a matching raw-source stat, channel, center, radius, renderer version, exact NPZ SHA-256, and output WebP SHA-256; display-rank changes therefore invalidate stale assets.</li><li>Precision browser state uses stable detector keys plus the candidate-table SHA-256. Identity-bound JSON can be imported or exported, and never mutates detector output.</li><li>Reference-linked and detector-only labels are review metadata, not training truth.</li></ul></article>
<article><h2>7. Validation status and limitations</h2><p>The frozen v6 profile reproduces donor13 sample counts and numerical rows, including the representative 13-23/#680 mask. Rescue is an explicitly review-gated extension: the first-cohort delta was inspected card by card and reference-backed donor13 failures were used only as diagnostic evidence, never as proposal input. Donor11 executes safely but has not yet received biological examples or a validated donor-specific intensity regime. Open-ring oocytes with a very large dark germinal vesicle can still yield an incomplete crescent mask when the fluorescent rim is not closed; these remain review-priority rather than being silently completed by an assumed circle.</p></article>
<article><h2>8. Frozen profile parameters</h2><h3>donor13_v6</h3><div style="overflow:auto"><table><thead><tr><th>section</th><th>parameter</th><th>value</th></tr></thead><tbody>{v6_rows}</tbody></table></div><h3>donor13_v6_rescue_v1</h3><div style="overflow:auto"><table><thead><tr><th>section</th><th>parameter</th><th>value</th></tr></thead><tbody>{rescue_rows}</tbody></table></div></article><footer class="footer"><a class="button" href="oocyte_review_index.html">Open batch review index</a></footer></main></body></html>"""


def _load_combined_candidates(
    sample_id: str,
    sample_dir: Path,
    rescue_delta_dir: Path | None,
) -> pd.DataFrame:
    baseline = pd.read_csv(sample_dir / "candidates.csv")
    baseline = baseline[baseline["accepted"].astype(bool)].copy()
    if "segmentation_pass" in baseline.columns:
        baseline["detection_pass"] = np.where(
            baseline["segmentation_pass"].astype(str) == "secondary_rescue",
            "secondary_rescue",
            "baseline_v6",
        )
    else:
        baseline["detection_pass"] = "baseline_v6"
    baseline["mask_source_dir"] = str(sample_dir)
    tables = [baseline]
    if rescue_delta_dir is not None:
        rescue_path = rescue_delta_dir / sample_id / "candidates.csv"
        if rescue_path.is_file():
            rescue = pd.read_csv(rescue_path)
            rescue = rescue[rescue["accepted"].astype(bool)].copy()
            rescue["detection_pass"] = "secondary_rescue"
            rescue["mask_source_dir"] = str(rescue_path.parent)
            tables.append(rescue)
    combined = pd.concat(tables, ignore_index=True, sort=False)
    persisted_masks = {}
    for row in combined.to_dict("records"):
        candidate_id = str(row["detector_component_id"])
        persisted_masks[candidate_id] = load_candidate_mask(
            Path(str(row["mask_source_dir"])) / str(row["mask_path"])
        )
    combined, suppressed = suppress_accepted_mask_duplicates(
        combined,
        persisted_masks,
        overlap_fraction=0.25,
        max_centroid_distance_px=125.0,
    )
    suppressed.to_csv(sample_dir / "combined_duplicate_suppressed.csv", index=False)
    combined = combined[combined["accepted"].astype(bool)].copy()
    combined = combined.sort_values(
        ["detector_score", "center_y", "center_x"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    combined["display_id"] = [f"#{index:03d}" for index in range(1, len(combined) + 1)]
    combined["html_id"] = [f"oocyte-{index:04d}" for index in range(1, len(combined) + 1)]
    combined["quality_class"] = [
        _candidate_quality(row) for row in combined.to_dict("records")
    ]
    return combined


def generate_html_reports(
    batch_dir: Path,
    *,
    rescue_delta_dir: Path | None = None,
    references_path: Path | None = None,
    patch_radius_px: int = 180,
    export_combined_labels: bool = True,
) -> HtmlReportResult:
    """Generate the algorithm document, batch index, and one review page per sample."""

    root = Path(batch_dir).resolve()
    rescue_root = None if rescue_delta_dir is None else Path(rescue_delta_dir).resolve()
    summary = pd.read_csv(root / "batch_summary.csv")
    algorithm_path = root / "oocyte_detection_algorithm.html"
    algorithm_path.write_text(algorithm_document_html())
    sample_pages: Dict[str, Path] = {}
    index_rows = []
    for summary_row in summary.to_dict("records"):
        if str(summary_row["status"]) != "complete":
            continue
        sample_id = str(summary_row["sample_id"])
        LOGGER.info("HTML report starting sample %s", sample_id)
        sample_dir = root / sample_id
        manifest = json.loads((sample_dir / "run_manifest.json").read_text())
        image_path = Path(manifest["source_image"])
        channel_index = int(manifest["resolved_channel_index"])
        sample_summary = json.loads((sample_dir / "summary.json").read_text())
        image_shape = tuple(int(value) for value in sample_summary["image_shape_yx"])
        candidates = _load_combined_candidates(sample_id, sample_dir, rescue_root)
        candidates = _annotate_reference_distance(
            candidates,
            _load_references(references_path, sample_id),
        )
        rescue_count = int(
            (candidates["detection_pass"] == "secondary_rescue").sum()
        ) if not candidates.empty else 0
        combined_label_path = sample_dir / "oocyte_labels_rescue_v1.ome.tiff"
        should_export_combined_labels = bool(
            export_combined_labels and rescue_root is not None and rescue_count > 0
        )
        combined_labels_available = bool(
            rescue_root is not None
            and rescue_count > 0
            and (should_export_combined_labels or combined_label_path.is_file())
        )
        if should_export_combined_labels:
            combined_export = candidates.copy()
            combined_export["mask_path"] = [
                str(Path(str(row["mask_source_dir"])) / str(row["mask_path"]))
                for row in combined_export.to_dict("records")
            ]
            combined_export.to_csv(
                sample_dir / "candidates_rescue_v1_combined.csv",
                index=False,
            )
            export_whole_slide_labels(
                combined_export,
                sample_dir=sample_dir,
                image_shape_yx=image_shape,
                image_path=combined_label_path,
                mapping_path=sample_dir / "oocyte_labels_rescue_v1.csv",
            )
        assets_dir = sample_dir / "html_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_manifest_path = assets_dir / "manifest.json"
        thumbnail_manifest = _load_thumbnail_manifest(thumbnail_manifest_path)
        previous_entries = thumbnail_manifest.get("entries", {})
        expected_assets = {
            f"{html_id}.webp" for html_id in candidates["html_id"].astype(str)
        }
        for existing_asset in assets_dir.glob("*.webp"):
            if existing_asset.name not in expected_assets:
                existing_asset.unlink()
        thumbnail_paths = [
            str(Path("html_assets") / f"{html_id}.webp")
            for html_id in candidates["html_id"].astype(str)
        ]
        source_stat = image_path.stat()
        current_entries: Dict[str, Dict[str, str]] = {}
        render_jobs = []
        if not candidates.empty:
            for row in candidates.to_dict("records"):
                asset_name = f"{row['html_id']}.webp"
                asset_path = assets_dir / asset_name
                source_dir = Path(str(row["mask_source_dir"]))
                mask_path = source_dir / str(row["mask_path"])
                fingerprint = _thumbnail_fingerprint(
                    row,
                    source_image=image_path,
                    source_size_bytes=int(source_stat.st_size),
                    source_mtime_ns=int(source_stat.st_mtime_ns),
                    channel_index=channel_index,
                    patch_radius_px=patch_radius_px,
                    mask_path=mask_path,
                )
                previous_entry = previous_entries.get(asset_name)
                if _thumbnail_is_current(asset_path, previous_entry, fingerprint):
                    current_entries[asset_name] = {
                        "render_fingerprint": fingerprint,
                        "asset_sha256": str(previous_entry["asset_sha256"]),
                    }
                else:
                    render_jobs.append((row, asset_name, mask_path, fingerprint))
            if render_jobs:
                with tifffile.TiffFile(image_path) as tif:
                    array = zarr.open(tif.series[0].aszarr(), mode="r")
                    for row, asset_name, mask_path, fingerprint in render_jobs:
                        destination = assets_dir / asset_name
                        persisted = load_candidate_mask(mask_path)
                        patch = extract_cyx_channel_patch(
                            array,
                            channel_index,
                            (
                                int(round(float(row["center_x"]))),
                                int(round(float(row["center_y"]))),
                            ),
                            patch_radius_px,
                        )
                        _raw_mask_thumbnail(patch, persisted, destination)
                        current_entries[asset_name] = {
                            "render_fingerprint": fingerprint,
                            "asset_sha256": _file_sha256(destination),
                        }
        _atomic_write_json(
            thumbnail_manifest_path,
            {
                "schema_version": THUMBNAIL_CACHE_SCHEMA_VERSION,
                "renderer": THUMBNAIL_RENDER_VERSION,
                "entries": current_entries,
            },
        )
        candidates["thumbnail_path"] = thumbnail_paths
        candidate_table_path = sample_dir / "html_candidates.csv"
        candidates.to_csv(candidate_table_path, index=False)
        precision_identity = _precision_review_identity(
            sample_id=sample_id,
            source_image=image_path,
            manifest=manifest,
            sample_summary=sample_summary,
            candidate_table_path=candidate_table_path,
            candidate_count=len(candidates),
        )
        page_path = sample_dir / "oocytes.html"
        page_path.write_text(
            _sample_html(
                sample_id,
                candidates,
                image_shape,
                combined_labels_available=combined_labels_available,
                patch_radius_px=patch_radius_px,
                review_identity=precision_identity,
            )
        )
        sample_pages[sample_id] = page_path
        baseline_count = int((candidates["detection_pass"] == "baseline_v6").sum()) if not candidates.empty else 0
        index_rows.append((sample_id, len(candidates), baseline_count, rescue_count, page_path))
        LOGGER.info(
            "HTML report completed sample %s: %s candidates",
            sample_id,
            len(candidates),
        )

    cards = "".join(
        f'<a class="stat" style="text-decoration:none;color:inherit;min-width:240px" href="{html.escape(str(path.relative_to(root)),quote=True)}"><span class="eyebrow">sample</span><strong>{html.escape(sample_id)}</strong><span>{total} total / {baseline} baseline / {rescue} rescue</span></a>'
        for sample_id, total, baseline, rescue, path in index_rows
    )
    index_path = root / "oocyte_review_index.html"
    index_path.write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Oocyte review index</title><style>{_REPORT_CSS}</style></head><body><main class="shell"><header class="hero"><div class="eyebrow">Aegle raw-UCHL1 cohort review</div><h1>Oocyte review index</h1><p>Choose a sample to review exact detector masks in full-resolution image coordinates.</p><a class="button" href="oocyte_detection_algorithm.html">Read algorithm</a></header><section class="stats">{cards}</section><footer class="footer">Machine counts require biological review; zero candidates do not prove biological absence.</footer></main></body></html>'
    )
    return HtmlReportResult(
        batch_dir=root,
        algorithm_document=algorithm_path,
        batch_index=index_path,
        sample_pages=sample_pages,
    )


__all__ = [
    "HtmlReportResult",
    "algorithm_document_html",
    "generate_html_reports",
]
