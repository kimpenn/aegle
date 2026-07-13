"""Batch index and one-port server for per-sample oocyte review consoles."""

from __future__ import annotations

import html
import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import parse_qs, urlparse

from .recall_review import (
    DEFAULT_OVERVIEW_DOWNSAMPLE,
    DEFAULT_WINDOW_RADIUS_PX,
    DEFAULT_WINDOW_STRIDE_PX,
    RecallReviewRuntime,
    _atomic_write_text,
    _file_sha256,
    _identity_contains_required,
    _json_safe,
    _load_sample,
    _query_number,
    _read_json,
    _request_geometry,
    generate_recall_review_bundle,
)
from .recall_review_page import recall_review_page_html, review_console_page_html
from .recall_overlay import overlay_dir_from_identity


LOGGER = logging.getLogger(__name__)
BATCH_REVIEW_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchRecallReviewBundle:
    batch_dir: Path
    index_path: Path
    manifest_path: Path
    sample_ids: tuple[str, ...]
    total_window_count: int
    total_candidate_count: int


def _sample_ids(batch_dir: Path, requested: Sequence[str] | None) -> tuple[str, ...]:
    root = Path(batch_dir).resolve()
    if requested is None:
        values = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path / "run_manifest.json").is_file()
            and (path / "html_candidates.csv").is_file()
        )
    else:
        values = [str(value).strip() for value in requested]
    if not values:
        raise ValueError("batch recall review requires at least one sample")
    if len(set(values)) != len(values):
        raise ValueError("batch recall review sample IDs must be unique")
    for sample_id in values:
        if (
            not sample_id
            or sample_id in {".", ".."}
            or Path(sample_id).name != sample_id
            or "/" in sample_id
            or "\\" in sample_id
        ):
            raise ValueError(f"invalid batch sample ID: {sample_id!r}")
        sample_dir = (root / sample_id).resolve()
        try:
            sample_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"sample directory is outside batch root: {sample_id}") from exc
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")
    return tuple(values)


def _batch_index_html(records: Sequence[Mapping[str, Any]]) -> str:
    cards = []
    for record in records:
        sample_id = html.escape(str(record["sample_id"]))
        cards.append(
            f"""
            <article class="sample-card">
              <div class="sample-top"><div><div class="mono role">PANEL1 / OVARY</div><h2>{sample_id}</h2></div><span class="status">review required</span></div>
              <div class="metrics"><span><strong>{int(record['candidate_count'])}</strong> masks</span><span><strong>{int(record['window_count'])}</strong> windows</span><span><strong>{int(record['image_width'])} x {int(record['image_height'])}</strong> px</span></div>
              <p>Profile <span class="mono">{html.escape(str(record['profile_name']))}</span>; overlay <span class="mono">{html.escape(str(record['overlay_name']))}</span>. Review completion requires exported decisions; recall exports are source- and overlay-identity matched.</p>
              <div class="actions"><a class="primary" href="{sample_id}/review_console.html">Open console</a><a href="{sample_id}/oocytes.html">Precision</a><a href="{sample_id}/recall_review.html">Recall</a></div>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Panel1 oocyte review consoles</title>
  <style>
    :root{{--ink:#17231d;--muted:#68736c;--paper:#f4ecdd;--panel:#fffaf0;--teal:#087d79;--cyan:#00cfc5;--amber:#d17b27;--line:#cec3af;--shadow:0 18px 44px rgba(35,48,40,.14)}}
    *{{box-sizing:border-box}}html,body{{margin:0;min-height:100%;color:var(--ink);background:radial-gradient(circle at 12% -8%,#fff8e8 0,transparent 34%),linear-gradient(145deg,#e7ded0,#f8f4e9 58%,#dcebe4);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}}.shell{{width:min(1260px,calc(100% - 28px));margin:auto;padding:24px 0 56px}}.hero{{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:26px;padding:32px;background:linear-gradient(120deg,rgba(255,250,240,.98),rgba(222,242,234,.95));box-shadow:var(--shadow)}}.hero:after{{content:"";position:absolute;width:260px;height:260px;border:44px solid rgba(0,207,197,.08);border-radius:50%;right:-65px;top:-100px}}.eyebrow,.mono{{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}}.eyebrow{{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:700}}h1{{font-size:clamp(2.8rem,7vw,6rem);line-height:.88;margin:.2em 0;max-width:850px}}.hero p{{max-width:850px;line-height:1.5;font-size:1.05rem}}.hero-links{{display:flex;gap:9px;flex-wrap:wrap;margin-top:18px}}a{{color:inherit;text-decoration:none}}.hero-links a,.actions a{{border:1px solid #aa9f8c;border-radius:11px;padding:9px 13px;background:#fffaf0}}.hero-links a:hover,.actions a:hover{{border-color:var(--cyan);transform:translateY(-1px)}}.workflow{{margin-top:14px;border:1px solid var(--line);border-radius:20px;padding:23px;background:rgba(255,250,240,.96);box-shadow:0 11px 29px rgba(35,48,40,.09)}}.workflow-head{{display:flex;gap:18px;align-items:end;justify-content:space-between}}.workflow h2{{font-size:1.8rem;margin:.15em 0}}.workflow-head p{{color:var(--muted);margin:0;max-width:600px;line-height:1.4}}.flow-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:17px}}.flow-step{{position:relative;border-top:3px solid var(--teal);border-radius:10px;background:#f1e8d8;padding:14px;min-height:150px}}.flow-step:nth-child(3){{border-color:var(--amber)}}.flow-number{{font:700 .7rem "IBM Plex Mono","Courier New",monospace;color:var(--teal);letter-spacing:.1em}}.flow-step h3{{font-size:1.15rem;margin:.45em 0}}.flow-step p{{color:var(--muted);font-size:.85rem;line-height:1.4;margin:0}}.flow-step code{{font-size:.72rem;overflow-wrap:anywhere}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:14px}}.sample-card{{border:1px solid var(--line);border-radius:19px;padding:22px;background:rgba(255,250,240,.96);box-shadow:0 11px 29px rgba(35,48,40,.09)}}.sample-top{{display:flex;justify-content:space-between;gap:14px;align-items:start}}.sample-card h2{{font-size:2.2rem;margin:.12em 0}}.role{{font-size:.68rem;letter-spacing:.12em;color:var(--teal)}}.status{{font:700 .68rem "IBM Plex Mono","Courier New",monospace;text-transform:uppercase;color:#8c531b;background:#f8e2c4;border-radius:999px;padding:7px 9px}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:12px 0}}.metrics span{{background:#f0e7d7;border-radius:10px;padding:9px;font-size:.75rem;color:var(--muted)}}.metrics strong{{display:block;color:var(--ink);font-size:1rem}}.sample-card p{{color:var(--muted);line-height:1.45}}.actions{{display:flex;gap:7px;flex-wrap:wrap}}.actions .primary{{background:var(--teal);border-color:var(--teal);color:white}}.notice{{margin-top:14px;border-left:4px solid var(--amber);border-radius:0 12px 12px 0;background:#fff0d8;padding:13px 15px;line-height:1.45}}footer{{margin-top:18px;color:var(--muted);font-size:.86rem}}
    @media(max-width:980px){{.flow-grid{{grid-template-columns:repeat(2,1fr)}}.workflow-head{{align-items:start;flex-direction:column}}}}@media(max-width:760px){{.grid,.flow-grid{{grid-template-columns:1fr}}.hero,.workflow{{padding:22px}}.metrics{{grid-template-columns:1fr 1fr}}.flow-step{{min-height:0}}}}
  </style>
</head>
<body>
<main class="shell">
  <header class="hero"><div class="eyebrow">Aegle / raw UCHL1 / panel1</div><h1>Oocyte review consoles</h1><p>Sample-specific workspaces keep precision review, whole-slide recall review, and mask finalization distinct. Start with the algorithm contract, then export both review records before any final labels or expression matrices are created.</p><div class="hero-links"><a href="oocyte_detection_algorithm.html">Algorithm</a><a href="oocyte_review_index.html">Precision batch index</a></div></header>
  <section class="workflow">
    <div class="workflow-head"><div><div class="eyebrow">Review workflow</div><h2>Complete each sample in this order</h2></div><p><strong>Open console</strong> is navigation only. Opening a page does not create or complete any review decision.</p></div>
    <div class="flow-grid">
      <article class="flow-step"><div class="flow-number">01 / OPEN CONSOLE</div><h3>Enter the sample workspace</h3><p>Read the algorithm summary and choose Precision or Recall. Page visits alone are not review evidence.</p></article>
      <article class="flow-step"><div class="flow-number">02 / PRECISION</div><h3>Inspect existing masks</h3><p>Decide whether each mask is an oocyte, whether its boundary is acceptable, and whether it is duplicated. Export <code>&lt;sample&gt;_oocyte_review.json</code> when complete.</p></article>
      <article class="flow-step"><div class="flow-number">03 / RECALL</div><h3>Find missing oocytes</h3><p>Move through the whole-slide windows, mark each one Complete, Has misses, or Unsure, and click the center of every missed oocyte. Export <code>&lt;sample&gt;_recall_review.json</code>.</p></article>
      <article class="flow-step"><div class="flow-number">04 / RETURN</div><h3>Return the review records</h3><p>Place both JSON files in <code>notes/oocytes_detection/reviews/</code>. Recall clicks are centers, not final masks; boundary review and finalization follow.</p></article>
    </div>
  </section>
  <section class="grid">{''.join(cards)}</section>
  <div class="notice"><strong>Human action remains required.</strong> These consoles prepare and preserve review evidence; they do not mark candidates or windows complete automatically.</div>
  <footer>Serve this directory with the Aegle batch review server. Raw images and existing detector outputs remain read-only.</footer>
</main>
</body>
</html>"""


def generate_batch_recall_review_bundle(
    batch_dir: Path,
    *,
    sample_ids: Sequence[str] | None = None,
    overlay_dirs: Mapping[str, Path] | None = None,
    generate_samples: bool = True,
    window_radius_px: int = DEFAULT_WINDOW_RADIUS_PX,
    window_stride_px: int = DEFAULT_WINDOW_STRIDE_PX,
    overview_downsample: int = DEFAULT_OVERVIEW_DOWNSAMPLE,
) -> BatchRecallReviewBundle:
    """Generate sample consoles plus one identity-bound batch landing page."""

    root = Path(batch_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected_ids = _sample_ids(root, sample_ids)
    resolved_overlays = {
        str(sample_id): Path(path).resolve()
        for sample_id, path in (overlay_dirs or {}).items()
    }
    unknown_overlays = set(resolved_overlays).difference(selected_ids)
    if unknown_overlays:
        raise ValueError(
            f"Recall overlays reference unselected samples: {sorted(unknown_overlays)}"
        )
    records: list[Dict[str, Any]] = []
    for sample_id in selected_ids:
        sample_dir = root / sample_id
        requested_overlay = resolved_overlays.get(sample_id)
        if generate_samples:
            bundle = generate_recall_review_bundle(
                sample_dir,
                overlay_dir=requested_overlay,
                window_radius_px=window_radius_px,
                window_stride_px=window_stride_px,
                overview_downsample=overview_downsample,
            )
            metadata_path = bundle.metadata_path
            console_path = bundle.console_path
        else:
            metadata_path = sample_dir / "recall_review/metadata.json"
            console_path = sample_dir / "review_console.html"
            if not metadata_path.is_file() or not (sample_dir / "recall_review.html").is_file():
                raise FileNotFoundError(
                    f"recall review bundle is missing for sample {sample_id}"
                )
        metadata = _read_json(metadata_path)
        identity = metadata.get("review_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"recall metadata identity is invalid for {sample_id}")
        bound_overlay = overlay_dir_from_identity(identity)
        if requested_overlay is not None and bound_overlay != requested_overlay:
            raise ValueError(f"Recall overlay identity mismatch for sample {sample_id}")
        sample = _load_sample(sample_dir, overlay_dir=bound_overlay)
        if not _identity_contains_required(identity, sample.review_identity):
            raise ValueError(
                f"recall review bundle identity mismatch for sample {sample_id}"
            )
        windows = metadata.get("windows")
        if not isinstance(windows, list):
            raise ValueError(f"recall metadata windows are invalid for {sample_id}")
        _atomic_write_text(
            sample_dir / "recall_review.html",
            recall_review_page_html(sample.sample_id),
        )
        _atomic_write_text(
            console_path,
            review_console_page_html(
                sample_id=sample.sample_id,
                profile_name=sample.profile_name,
                candidate_count=len(sample.candidates),
                window_count=len(windows),
                image_shape_yx=sample.image_shape_yx,
                overlay_name=str(
                    sample.review_identity.get(
                        "overlay_delivery_name", "automatic detector"
                    )
                ),
            ),
        )
        records.append(
            {
                "sample_id": sample.sample_id,
                "sample_dir": str(sample.sample_dir),
                "profile_name": sample.profile_name,
                "profile_fingerprint": sample.profile_fingerprint,
                "implementation_version": sample.implementation_version,
                "image_height": sample.image_shape_yx[0],
                "image_width": sample.image_shape_yx[1],
                "candidate_count": len(sample.candidates),
                "overlay_name": str(
                    sample.review_identity.get(
                        "overlay_delivery_name", "automatic detector"
                    )
                ),
                "overlay_manifest_sha256": sample.review_identity.get(
                    "overlay_manifest_sha256"
                ),
                "window_count": len(windows),
                "metadata_path": str(metadata_path),
                "metadata_sha256": _file_sha256(metadata_path),
                "console_path": str(console_path),
                "review_status": "requires_export",
            }
        )

    index_path = root / "oocyte_review_console.html"
    manifest_path = root / "oocyte_review_console_manifest.json"
    _atomic_write_text(index_path, _batch_index_html(records))
    manifest = {
        "schema_version": BATCH_REVIEW_SCHEMA_VERSION,
        "review_type": "oocyte_batch_review_console",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_dir": str(root),
        "sample_count": len(records),
        "total_candidate_count": sum(int(row["candidate_count"]) for row in records),
        "total_window_count": sum(int(row["window_count"]) for row in records),
        "samples": records,
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False),
    )
    return BatchRecallReviewBundle(
        batch_dir=root,
        index_path=index_path,
        manifest_path=manifest_path,
        sample_ids=selected_ids,
        total_window_count=int(manifest["total_window_count"]),
        total_candidate_count=int(manifest["total_candidate_count"]),
    )


def _batch_handler_for(
    runtimes: Mapping[str, RecallReviewRuntime],
    bundle: BatchRecallReviewBundle,
):
    class BatchRecallReviewHandler(BaseHTTPRequestHandler):
        server_version = "AegleBatchRecallReview/1"

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info(
                "Batch recall HTTP %s - %s",
                self.address_string(),
                format_string % args,
            )

        def _send(
            self,
            payload: bytes,
            content_type: str,
            *,
            status: int = 200,
            head: bool = False,
        ) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if not head:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                LOGGER.debug("Batch recall HTTP client cancelled %s", self.path)

        def _json(
            self,
            payload: Mapping[str, Any],
            *,
            status: int = 200,
            head: bool = False,
        ) -> None:
            body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
            self._send(
                body,
                "application/json; charset=utf-8",
                status=status,
                head=head,
            )

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _shared(self, route: str, *, head: bool) -> bool:
            if route in {"/", "/oocyte_review_console.html"}:
                self._send(
                    bundle.index_path.read_bytes(),
                    "text/html; charset=utf-8",
                    head=head,
                )
                return True
            if route == "/health":
                self._json(
                    {
                        "status": "ok",
                        "read_only": True,
                        "sample_ids": list(bundle.sample_ids),
                    },
                    head=head,
                )
                return True
            if route in {
                "/oocyte_review_index.html",
                "/oocyte_detection_algorithm.html",
            }:
                path = bundle.batch_dir / route.lstrip("/")
                if not path.is_file():
                    raise FileNotFoundError("shared review page is unavailable")
                self._send(path.read_bytes(), "text/html; charset=utf-8", head=head)
                return True
            return False

        def _sample_route(
            self,
            runtime: RecallReviewRuntime,
            route: str,
            query: Mapping[str, Sequence[str]],
            *,
            head: bool,
        ) -> bool:
            if route == "/health":
                self._json(
                    {
                        "status": "ok",
                        "sample_id": runtime.sample.sample_id,
                        "read_only": True,
                    },
                    head=head,
                )
                return True
            if route in {"/", "/review_console.html"}:
                page = runtime.sample.sample_dir / "review_console.html"
                if not page.is_file():
                    raise FileNotFoundError("sample review console is unavailable")
                self._send(page.read_bytes(), "text/html; charset=utf-8", head=head)
                return True
            if route == "/recall_review.html":
                self._send(
                    runtime.page_path.read_bytes(),
                    "text/html; charset=utf-8",
                    head=head,
                )
                return True
            if route == "/oocytes.html":
                page = runtime.sample.sample_dir / "oocytes.html"
                if not page.is_file():
                    raise FileNotFoundError("precision review page is unavailable")
                self._send(page.read_bytes(), "text/html; charset=utf-8", head=head)
                return True
            if route.startswith("/html_assets/"):
                asset_name = Path(route).name
                if route != f"/html_assets/{asset_name}" or not asset_name.endswith(
                    ".webp"
                ):
                    raise ValueError("invalid precision-review asset path")
                path = runtime.sample.sample_dir / "html_assets" / asset_name
                if not path.is_file():
                    raise FileNotFoundError("precision-review asset is unavailable")
                self._send(path.read_bytes(), "image/webp", head=head)
                return True
            if route.startswith("/recall_analysis"):
                relative_path = Path(route.lstrip("/"))
                path = (runtime.sample.sample_dir / relative_path).resolve()
                try:
                    path.relative_to(runtime.sample.sample_dir)
                except ValueError as exc:
                    raise ValueError("invalid recall-analysis asset path") from exc
                content_types = {
                    ".html": "text/html; charset=utf-8",
                    ".webp": "image/webp",
                    ".json": "application/json; charset=utf-8",
                    ".csv": "text/csv; charset=utf-8",
                }
                content_type = content_types.get(path.suffix.lower())
                if content_type is None or not path.is_file():
                    raise FileNotFoundError("recall-analysis asset is unavailable")
                self._send(path.read_bytes(), content_type, head=head)
                return True
            if route == "/recall_review/overview.webp":
                self._send(runtime.overview_path.read_bytes(), "image/webp", head=head)
                return True
            if route == "/api/metadata":
                self._json(runtime.metadata, head=head)
                return True
            if route in {"/api/patch.webp", "/api/overlay.png", "/api/window"}:
                center, radius = _request_geometry(runtime, query)
                if route == "/api/patch.webp":
                    contrast = query.get("contrast", ["local"])[0]
                    self._send(
                        runtime.render_patch(center, radius, contrast),
                        "image/webp",
                        head=head,
                    )
                elif route == "/api/overlay.png":
                    self._send(
                        runtime.render_overlay(center, radius),
                        "image/png",
                        head=head,
                    )
                else:
                    self._json(runtime.window_payload(center, radius), head=head)
                return True
            if route == "/api/probe":
                x = _query_number(query, "x")
                y = _query_number(query, "y")
                self._json(runtime.probe(x, y), head=head)
                return True
            return False

        def _handle(self, *, head: bool) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if self._shared(parsed.path, head=head):
                    return
                parts = parsed.path.lstrip("/").split("/", 1)
                sample_id = parts[0] if parts else ""
                runtime = runtimes.get(sample_id)
                if runtime is None:
                    self._json(
                        {"error": "route not found"},
                        status=HTTPStatus.NOT_FOUND,
                        head=head,
                    )
                    return
                if len(parts) == 1 and not parsed.path.endswith("/"):
                    self._redirect(f"/{sample_id}/")
                    return
                route = "/" + parts[1] if len(parts) == 2 and parts[1] else "/"
                if not self._sample_route(runtime, route, query, head=head):
                    self._json(
                        {"error": "route not found"},
                        status=HTTPStatus.NOT_FOUND,
                        head=head,
                    )
            except (ValueError, FileNotFoundError) as exc:
                self._json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                    head=head,
                )
            except Exception as exc:  # pragma: no cover - defensive server boundary
                LOGGER.exception("Batch recall review request failed")
                self._json(
                    {"error": f"internal server error: {type(exc).__name__}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    head=head,
                )

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP method name
            self._handle(head=False)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib HTTP method name
            self._handle(head=True)

    return BatchRecallReviewHandler


def serve_batch_recall_review(
    batch_dir: Path,
    *,
    sample_ids: Sequence[str] | None = None,
    overlay_dirs: Mapping[str, Path] | None = None,
    host: str = "127.0.0.1",
    port: int = 8767,
    generate: bool = True,
    window_radius_px: int = DEFAULT_WINDOW_RADIUS_PX,
    window_stride_px: int = DEFAULT_WINDOW_STRIDE_PX,
    overview_downsample: int = DEFAULT_OVERVIEW_DOWNSAMPLE,
) -> None:
    """Serve multiple identity-isolated sample consoles on one local port."""

    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be in [1, 65535]")
    bundle = generate_batch_recall_review_bundle(
        batch_dir,
        sample_ids=sample_ids,
        overlay_dirs=overlay_dirs,
        generate_samples=generate,
        window_radius_px=window_radius_px,
        window_stride_px=window_stride_px,
        overview_downsample=overview_downsample,
    )
    with ExitStack() as stack:
        runtimes = {
            sample_id: stack.enter_context(
                RecallReviewRuntime(bundle.batch_dir / sample_id)
            )
            for sample_id in bundle.sample_ids
        }
        server = ThreadingHTTPServer(
            (host, int(port)),
            _batch_handler_for(runtimes, bundle),
        )
        server.daemon_threads = True
        LOGGER.info(
            "Batch review serving samples=%s candidates=%s windows=%s",
            ",".join(bundle.sample_ids),
            bundle.total_candidate_count,
            bundle.total_window_count,
        )
        LOGGER.info("Batch review URL: http://%s:%s/", host, port)
        LOGGER.info("Source images and production detector outputs are read-only")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            LOGGER.info("Batch review server interrupted")
        finally:
            server.server_close()


__all__ = [
    "BATCH_REVIEW_SCHEMA_VERSION",
    "BatchRecallReviewBundle",
    "generate_batch_recall_review_bundle",
    "serve_batch_recall_review",
]
