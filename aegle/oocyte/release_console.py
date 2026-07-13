"""Self-contained static review pages for packaged oocyte releases."""

from __future__ import annotations

import base64
import html
import math
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping

from matplotlib import colormaps
import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image
from scipy import ndimage as ndi

from .io import extract_cyx_channel_patch, find_channel_index, load_candidate_mask
from .report import render_raw_mask_thumbnail_bytes


EMBEDDED_CONSOLE_SCHEMA_VERSION = 1
EMBEDDED_CONSOLE_RENDER_VERSION = "release-embedded-uchl1-v4"
OVERVIEW_DOWNSAMPLE = 16


def _atomic_write_text(path: Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            handle.write(content)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _number(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def _image_data_uri(image: Image.Image, *, quality: int = 84) -> str:
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=6)
    return _data_uri(buffer.getvalue())


def _reduce_strip(strip: np.ndarray, factor: int) -> np.ndarray:
    pad_h = (-strip.shape[0]) % factor
    pad_w = (-strip.shape[1]) % factor
    if pad_h or pad_w:
        strip = np.pad(strip, ((0, pad_h), (0, pad_w)), mode="edge")
    return strip.reshape(
        strip.shape[0] // factor,
        factor,
        strip.shape[1] // factor,
        factor,
    ).mean(axis=(1, 3))


def _raw_overview_image(
    raw_array: Any,
    channel_index: int,
    *,
    downsample: int = OVERVIEW_DOWNSAMPLE,
    strip_height: int = 1024,
) -> Image.Image:
    channel_count, image_h, image_w = (int(value) for value in raw_array.shape)
    if not 0 <= channel_index < channel_count:
        raise IndexError("overview channel index is outside the source image")
    overview = np.zeros(
        (math.ceil(image_h / downsample), math.ceil(image_w / downsample)),
        dtype=np.float32,
    )
    output_y = 0
    for y0 in range(0, image_h, strip_height):
        y1 = min(image_h, y0 + strip_height)
        strip = np.asarray(raw_array[channel_index, y0:y1, :], dtype=np.float32)
        reduced = _reduce_strip(strip, downsample)
        overview[
            output_y : output_y + reduced.shape[0],
            : reduced.shape[1],
        ] = reduced
        output_y += reduced.shape[0]
    transformed = np.log1p(np.maximum(overview, 0.0))
    finite = transformed[np.isfinite(transformed)]
    low, high = (
        (float(value) for value in np.percentile(finite, [1.0, 99.9]))
        if finite.size
        else (0.0, 1.0)
    )
    normalized = np.clip(
        (transformed - low) / max(float(high - low), 1e-6),
        0.0,
        1.0,
    )
    rgb = np.asarray(colormaps["magma"](normalized)[..., :3] * 255.0, dtype=np.uint8)
    return Image.fromarray(rgb)


def _masked_overview_image(
    raw_overview: Image.Image,
    persisted_masks: Mapping[str, Any],
    *,
    downsample: int = OVERVIEW_DOWNSAMPLE,
) -> Image.Image:
    boundary_map = np.zeros(
        (raw_overview.height, raw_overview.width),
        dtype=np.bool_,
    )
    for persisted in persisted_masks.values():
        boundary = np.logical_xor(
            persisted.mask,
            ndi.binary_erosion(persisted.mask),
        )
        yy, xx = np.nonzero(boundary)
        if not len(xx):
            continue
        overview_y = (persisted.bbox.y0 + yy) // downsample
        overview_x = (persisted.bbox.x0 + xx) // downsample
        valid = (
            (overview_y >= 0)
            & (overview_y < raw_overview.height)
            & (overview_x >= 0)
            & (overview_x < raw_overview.width)
        )
        boundary_map[overview_y[valid], overview_x[valid]] = True
    if boundary_map.any():
        boundary_map = ndi.binary_dilation(boundary_map, iterations=1)
    rgb = np.asarray(raw_overview.convert("RGB")).copy()
    rgb[boundary_map] = np.array([0, 255, 246], dtype=np.uint8)
    return Image.fromarray(rgb)


def _provenance_label(row: Mapping[str, Any]) -> str:
    detection_pass = str(row.get("detection_pass", "")).strip()
    manual_choice = str(row.get("manual_mask_choice", "")).strip()
    if "manual_contour" in detection_pass or "contour" in manual_choice:
        return "manual contour"
    if detection_pass.startswith("manual_seed"):
        return "reviewed recall"
    if detection_pass.startswith("rescue"):
        return "rescue"
    return "automatic"


def _artifact_links() -> str:
    links = (
        ("Batch index", "../../oocyte_review_index.html"),
        ("Batch expression", "../../batch_oocyte_by_marker.csv"),
        ("Sample expression", "profiling/oocyte_by_marker.csv"),
        ("Metadata", "profiling/oocyte_metadata.csv"),
        ("Label OME-TIFF", "final/oocyte_labels.ome.tiff"),
        ("Label mapping", "final/oocyte_labels.csv"),
        ("Candidates", "final/oocyte_candidates.csv"),
        ("Review evidence", "review/review_manifest.json"),
        ("Checksums", "sample_release_manifest.json"),
    )
    return "".join(
        f'<a href="{href}">{html.escape(label)}</a>' for label, href in links
    )


def _card_html(
    row: Mapping[str, Any],
    *,
    raw_uri: str,
    masked_uri: str,
) -> str:
    label_id = int(row["label_id"])
    oocyte_id = html.escape(str(row["oocyte_id"]))
    component_id = html.escape(str(row["detector_component_id"]))
    provenance = _provenance_label(row)
    warning = bool(row.get("boundary_warning", False))
    warning_html = '<span class="warning">boundary note</span>' if warning else ""
    return f"""
    <article id="oocyte-{label_id:03d}" class="cell-card" data-embedded-review-card data-provenance="{html.escape(provenance)}">
      <div class="image-frame">
        <img class="masked" src="{masked_uri}" loading="lazy" alt="Raw UCHL1 with final mask for {oocyte_id}">
        <img class="raw" src="{raw_uri}" loading="lazy" alt="Raw UCHL1 without mask for {oocyte_id}" hidden>
        <button class="mask-toggle" type="button">Hide mask</button>
      </div>
      <div class="card-body">
        <div class="card-heading"><h2>#{label_id:03d}</h2><span>{html.escape(provenance)}</span></div>
        <code>{oocyte_id}</code><small>{component_id}</small>
        <div class="metrics">
          <span><b>{_number(row.get('equivalent_diameter_um'))}</b> um diameter</span>
          <span><b>{_number(row.get('area_um2'), 0)}</b> um2 area</span>
          <span><b>{_number(row.get('circularity'), 2)}</b> circularity</span>
          <span><b>{_number(row.get('solidity'), 2)}</b> solidity</span>
          <span><b>{_number(row.get('center_x'), 0)}</b> x</span>
          <span><b>{_number(row.get('center_y'), 0)}</b> y</span>
        </div>
        {warning_html}
      </div>
    </article>
    """


def _overview_html(
    *,
    rows: list[Mapping[str, Any]],
    image_shape_yx: tuple[int, int],
    raw_uri: str,
    masked_uri: str | None,
) -> str:
    image_h, image_w = image_shape_yx
    hotspots = []
    for row in rows:
        label_id = int(row["label_id"])
        center_x = float(row["center_x"])
        center_y = float(row["center_y"])
        try:
            bbox_width = float(row["bbox_x1"]) - float(row["bbox_x0"])
            bbox_height = float(row["bbox_y1"]) - float(row["bbox_y0"])
        except (TypeError, ValueError):
            bbox_width = bbox_height = 0.0
        radius = max(70.0, min(180.0, max(bbox_width, bbox_height) * 0.75))
        title = html.escape(f"#{label_id:03d} {row['oocyte_id']}")
        hotspots.append(
            f'<a href="#oocyte-{label_id:03d}" aria-label="{title}">'
            f'<circle data-global-hotspot cx="{center_x:.2f}" cy="{center_y:.2f}" '
            f'r="{radius:.2f}" data-card-target="oocyte-{label_id:03d}">'
            f'<title>{title}</title></circle></a>'
        )
    if masked_uri is None:
        images = (
            f'<img class="overview-raw-only" src="{raw_uri}" '
            'alt="Whole-slide raw UCHL1 overview">'
        )
        controls = (
            '<span class="overview-status">No final masks in this negative control</span>'
        )
        svg = ""
    else:
        images = (
            f'<img class="overview-masked" src="{masked_uri}" '
            'alt="Whole-slide raw UCHL1 with all final masks">'
            f'<img class="overview-raw" src="{raw_uri}" '
            'alt="Whole-slide raw UCHL1 without masks" hidden>'
        )
        controls = (
            '<button class="overview-toggle" type="button">Hide masks</button>'
            '<span class="overview-status">Click a mask location to jump to its card</span>'
        )
        svg = (
            f'<svg class="overview-hotspots" viewBox="0 0 {image_w} {image_h}" '
            f'preserveAspectRatio="none" aria-label="{len(rows)} clickable oocyte locations">'
            f'{"".join(hotspots)}</svg>'
        )
    return f"""
    <section class="overview-panel" data-global-overview>
      <div class="overview-heading"><div><div class="eyebrow">Whole-slide navigator</div><h2>Final masks in tissue context</h2></div><div class="overview-controls">{controls}</div></div>
      <p>Downsampled raw UCHL1 across the complete registered section. Cyan boundaries are the exact final NPZ masks projected at overview scale.</p>
      <div class="overview-stage"><div class="overview-frame">{images}{svg}</div></div>
    </section>
    """


def _page_html(
    *,
    sample_id: str,
    role: str,
    cards: list[str],
    provenance_counts: Mapping[str, int],
    overview_html: str,
) -> str:
    count = len(cards)
    filters = "".join(
        f'<button type="button" data-filter="{html.escape(name)}">'
        f"{html.escape(name)} <b>{value}</b></button>"
        for name, value in sorted(provenance_counts.items())
    )
    if count:
        content = f"""
        {overview_html}
        <section class="controls"><button type="button" data-filter="all" class="active">All <b>{count}</b></button>{filters}</section>
        <section class="grid">{''.join(cards)}</section>
        """
        description = (
            "Each card embeds the registered raw UCHL1 patch and the exact final "
            "release mask. Cyan is the delivered boundary; the white cross marks "
            "the recorded center. No server or raw-image file is needed to view "
            "these cards."
        )
    else:
        content = f"""
        {overview_html}
        <section class="empty"><h2>No final oocytes</h2><p>This sample is a validated no-oocyte negative control. Its label image, mapping, candidate table, and expression tables contain zero positive rows.</p></section>
        """
        description = (
            "The frozen detector and rescue diagnostics contain zero accepted "
            "objects, and the final whole-slide label contains background only."
        )
    return f"""<!doctype html>
<html lang="en" data-console-schema="{EMBEDDED_CONSOLE_SCHEMA_VERSION}" data-renderer="{EMBEDDED_CONSOLE_RENDER_VERSION}" data-embedded-card-count="{count}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(sample_id)} final oocytes</title>
  <style>
    :root{{--ink:#17231d;--muted:#65716a;--paper:#eee6d7;--panel:#fffaf0;--line:#cbbfa9;--teal:#087d79;--cyan:#00d6ca;--coral:#c94f3e;--shadow:0 16px 36px rgba(35,48,40,.13)}}
    *{{box-sizing:border-box}}html,body{{margin:0;min-height:100%;background:radial-gradient(circle at 10% -5%,#fff8e8,transparent 32%),linear-gradient(145deg,#e9e0d2,#f7f3e8 58%,#dcebe4);color:var(--ink);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}}body{{border-top:8px solid #263725}}main{{width:min(1500px,calc(100% - 28px));margin:auto;padding:22px 0 60px}}.hero{{border:1px solid var(--line);border-radius:26px;background:rgba(255,250,240,.96);box-shadow:var(--shadow);padding:28px}}.eyebrow,code,small,.metrics,.warning{{font-family:"IBM Plex Mono","Courier New",monospace}}.eyebrow{{color:var(--teal);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase}}h1{{font-size:clamp(3rem,7vw,6.5rem);line-height:.86;margin:.18em 0}}.hero p{{max-width:900px;font-size:1.05rem;line-height:1.5}}.links{{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}}.links a,.controls button{{color:var(--ink);text-decoration:none;border:1px solid #aa9e8a;border-radius:11px;background:#fffaf0;padding:9px 12px;font:inherit}}.links a:first-child{{background:var(--teal);border-color:var(--teal);color:white}}.controls{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}.controls button{{cursor:pointer}}.controls button.active{{background:#17231d;color:white}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.cell-card{{overflow:hidden;border:1px solid var(--line);border-radius:20px;background:var(--panel);box-shadow:0 9px 24px rgba(35,48,40,.09)}}.cell-card[hidden]{{display:none}}.image-frame{{position:relative;aspect-ratio:1;background:#130f17}}.image-frame img{{display:block;width:100%;height:100%;object-fit:cover}}.image-frame img[hidden]{{display:none}}.mask-toggle{{position:absolute;right:10px;bottom:10px;border:1px solid #9d8d76;border-radius:999px;padding:7px 12px;background:rgba(255,250,240,.94);color:var(--ink);font:700 .72rem "IBM Plex Mono","Courier New",monospace;cursor:pointer;box-shadow:0 3px 10px rgba(0,0,0,.2)}}.card-body{{padding:15px}}.card-heading{{display:flex;justify-content:space-between;align-items:center;gap:10px}}.card-heading h2{{font-size:1.7rem;margin:0}}.card-heading span{{border-radius:999px;background:#dcefe9;padding:5px 8px;color:#24655f;font-size:.72rem}}code,small{{display:block;overflow-wrap:anywhere}}code{{font-size:.8rem;margin-top:7px}}small{{color:var(--muted);margin-top:4px}}.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:12px;font-size:.66rem}}.metrics span{{background:#efe7d8;border-radius:9px;padding:7px;color:var(--muted)}}.metrics b{{display:block;color:var(--ink);font-size:.8rem}}.warning{{display:inline-block;margin-top:10px;border-radius:999px;background:#f7d9cf;color:#87392e;padding:5px 8px;font-size:.67rem}}.empty{{margin-top:18px;border:1px solid var(--line);border-radius:22px;background:var(--panel);padding:32px;box-shadow:var(--shadow)}}.empty h2{{font-size:2.4rem;margin:.1em 0}}.empty p{{max-width:760px;line-height:1.5;color:var(--muted)}}footer{{margin-top:20px;color:var(--muted);font-size:.84rem}}
    .overview-panel{{margin-top:18px;border:1px solid var(--line);border-radius:24px;background:rgba(255,250,240,.96);padding:24px;box-shadow:var(--shadow)}}.overview-heading{{display:flex;align-items:end;justify-content:space-between;gap:18px}}.overview-heading h2{{font-size:2rem;margin:.15em 0}}.overview-panel>p{{max-width:900px;color:var(--muted);line-height:1.45}}.overview-controls{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}.overview-toggle{{border:1px solid #aa9e8a;border-radius:11px;background:#17231d;color:white;padding:9px 12px;font:inherit;cursor:pointer}}.overview-status{{color:var(--muted);font-size:.85rem}}.overview-stage{{overflow:auto;border-radius:18px;background:#110d16;padding:12px;text-align:center}}.overview-frame{{position:relative;display:inline-block;max-width:100%;line-height:0}}.overview-frame img{{display:block;width:auto;height:auto;max-width:100%;max-height:780px;border-radius:10px}}.overview-frame img[hidden],.overview-hotspots[hidden]{{display:none}}.overview-hotspots{{position:absolute;inset:0;width:100%;height:100%}}.overview-hotspots circle{{fill:rgba(0,214,202,.10);stroke:rgba(0,255,246,.55);stroke-width:2;vector-effect:non-scaling-stroke;cursor:pointer}}.overview-hotspots a:hover circle,.overview-hotspots a:focus circle{{fill:rgba(255,212,56,.26);stroke:#ffd438;stroke-width:4}}.cell-card{{scroll-margin-top:15px}}.cell-card:target{{outline:5px solid #ffd438;outline-offset:3px}}
    @media(max-width:1150px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:820px){{.grid{{grid-template-columns:repeat(2,1fr)}}.overview-heading{{align-items:start;flex-direction:column}}}}@media(max-width:520px){{.grid{{grid-template-columns:1fr}}.hero,.overview-panel{{padding:21px}}}}
  </style>
</head>
<body><main>
  <header class="hero"><div class="eyebrow">Aegle / raw UCHL1 / final reviewed release / {html.escape(role)}</div><h1>{html.escape(sample_id)}</h1><p><strong>{count}</strong> final reviewed oocyte labels. {html.escape(description)}</p><nav class="links">{_artifact_links()}</nav></header>
  {content}
  <footer>Static release console. All review images are embedded in this HTML; artifact links are relative to the unpacked release directory.</footer>
</main>
<script>
document.querySelectorAll('.mask-toggle').forEach(button=>button.addEventListener('click',()=>{{const frame=button.closest('.image-frame');const masked=frame.querySelector('.masked');const raw=frame.querySelector('.raw');const showRaw=raw.hidden;raw.hidden=!showRaw;masked.hidden=showRaw;button.textContent=showRaw?'Show mask':'Hide mask'}}));
document.querySelectorAll('.overview-toggle').forEach(button=>button.addEventListener('click',()=>{{const panel=button.closest('.overview-panel');const masked=panel.querySelector('.overview-masked');const raw=panel.querySelector('.overview-raw');const hotspots=panel.querySelector('.overview-hotspots');const showRaw=raw.hidden;raw.hidden=!showRaw;masked.hidden=showRaw;hotspots.toggleAttribute('hidden',showRaw);button.textContent=showRaw?'Show masks':'Hide masks'}}));
document.querySelectorAll('.overview-hotspots').forEach(svg=>svg.addEventListener('click',event=>{{const rect=svg.getBoundingClientRect(),view=svg.viewBox.baseVal,x=view.x+(event.clientX-rect.left)*view.width/rect.width,y=view.y+(event.clientY-rect.top)*view.height/rect.height;let nearest=null,best=Infinity;svg.querySelectorAll('[data-global-hotspot]').forEach(circle=>{{const dx=Number(circle.getAttribute('cx'))-x,dy=Number(circle.getAttribute('cy'))-y,distance=dx*dx+dy*dy;if(distance<best){{best=distance;nearest=circle}}}});if(!nearest||best>Number(nearest.getAttribute('r'))**2)return;event.preventDefault();const target=document.getElementById(nearest.dataset.cardTarget);if(target){{location.hash=nearest.dataset.cardTarget;target.scrollIntoView({{behavior:'smooth',block:'start'}})}}}}));
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{const value=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(item=>item.classList.toggle('active',item===button));document.querySelectorAll('[data-embedded-review-card]').forEach(card=>card.hidden=value!=='all'&&card.dataset.provenance!==value)}}));
</script></body></html>"""


def write_embedded_release_console(
    *,
    sample_id: str,
    role: str,
    source_image_path: Path,
    antibodies_path: Path,
    final_dir: Path,
    profiling_dir: Path,
    destination: Path,
    output_size_px: int = 360,
    webp_quality: int = 82,
) -> Dict[str, Any]:
    """Write a server-free release page containing exact raw/mask image pairs."""

    metadata = pd.read_csv(Path(profiling_dir) / "oocyte_metadata.csv")
    mapping = pd.read_csv(Path(final_dir) / "oocyte_labels.csv")
    if len(metadata) != len(mapping):
        raise ValueError(f"release console row count mismatch for {sample_id}")
    joined = metadata.merge(
        mapping[["detector_component_id", "mask_path"]],
        on="detector_component_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_final"),
    )
    if len(joined) and joined["mask_path_final"].isna().any():
        raise ValueError(f"release console mask mapping is incomplete for {sample_id}")
    joined = joined.sort_values("label_id", kind="stable")

    rows = joined.to_dict("records")
    persisted_masks = {
        str(row["detector_component_id"]): load_candidate_mask(
            (Path(final_dir) / str(row["mask_path_final"])).resolve()
        )
        for row in rows
    }
    cards: list[str] = []
    provenance_counts: Dict[str, int] = {}
    channel_index = find_channel_index(antibodies_path, "UCHL1")
    with tifffile.TiffFile(source_image_path) as tif:
        series = tif.series[0]
        if str(series.axes) != "CYX":
            raise ValueError(
                f"embedded release console requires CYX source axes: {series.axes}"
            )
        store = series.aszarr()
        try:
            raw_array = zarr.open(store, mode="r")
            raw_overview = _raw_overview_image(raw_array, channel_index)
            raw_overview_uri = _image_data_uri(raw_overview)
            masked_overview_uri = None
            if persisted_masks:
                masked_overview_uri = _image_data_uri(
                    _masked_overview_image(raw_overview, persisted_masks)
                )
            for row in rows:
                persisted = persisted_masks[str(row["detector_component_id"])]
                max_mask_dimension = max(persisted.mask.shape)
                radius = max(96, min(240, math.ceil(max_mask_dimension * 1.35)))
                patch = extract_cyx_channel_patch(
                    raw_array,
                    channel_index,
                    (
                        int(round(float(row["center_x"]))),
                        int(round(float(row["center_y"]))),
                    ),
                    radius,
                )
                raw_uri = _data_uri(
                    render_raw_mask_thumbnail_bytes(
                        patch,
                        persisted,
                        show_mask=False,
                        output_size_px=output_size_px,
                        quality=webp_quality,
                    )
                )
                masked_uri = _data_uri(
                    render_raw_mask_thumbnail_bytes(
                        patch,
                        persisted,
                        show_mask=True,
                        output_size_px=output_size_px,
                        quality=webp_quality,
                    )
                )
                cards.append(_card_html(row, raw_uri=raw_uri, masked_uri=masked_uri))
                provenance = _provenance_label(row)
                provenance_counts[provenance] = (
                    provenance_counts.get(provenance, 0) + 1
                )
            image_shape_yx = (int(raw_array.shape[1]), int(raw_array.shape[2]))
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()

    overview = _overview_html(
        rows=rows,
        image_shape_yx=image_shape_yx,
        raw_uri=raw_overview_uri,
        masked_uri=masked_overview_uri,
    )

    page = _page_html(
        sample_id=sample_id,
        role=role,
        cards=cards,
        provenance_counts=provenance_counts,
        overview_html=overview,
    )
    _atomic_write_text(destination, page)
    overview_webp_count = 2 if masked_overview_uri is not None else 1
    return {
        "schema_version": EMBEDDED_CONSOLE_SCHEMA_VERSION,
        "renderer": EMBEDDED_CONSOLE_RENDER_VERSION,
        "embedded_card_count": len(cards),
        "embedded_webp_count": len(cards) * 2 + overview_webp_count,
        "overview_webp_count": overview_webp_count,
        "overview_downsample": OVERVIEW_DOWNSAMPLE,
        "global_hotspot_count": len(rows),
        "html_size_bytes": len(page.encode("utf-8")),
        "provenance_counts": provenance_counts,
    }


__all__ = [
    "EMBEDDED_CONSOLE_RENDER_VERSION",
    "EMBEDDED_CONSOLE_SCHEMA_VERSION",
    "write_embedded_release_console",
]
