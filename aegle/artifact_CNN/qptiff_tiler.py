"""
Tile a (QPTIFF / pyramidal TIFF / WSI) image into 960x720 non-overlapping patches for CNN training.

- Memory‑efficient via tifffile + zarr; reads one window at a time.
- Automatically selects the proper zarr **array** even if `series.aszarr()` returns a **Group** (fixes
  "path is not a string" errors).
- Skips edge remainders so tiles do **not** overlap and all are exactly 960x720.
- Saves patches as TIFF.
- Optional GeoJSON grouping into subfolders by majority annotation


Usage (CLI):
    python qptiff_tiler.py \
        --input /path/to/image.qptiff \
        --output_dir ./tiles \
        --tile_w 960 --tile_h 720 \
        --level 0 \
        --geojson "/path/to/**.geojson"

Or import and call `tile_tiff(...)` from Python code.

Requirements:
    pip install tifffile zarr pillow numpy shapely imagecodecs

Notes:
- For pyramidal TIFF/WSI, `--level 0` typically refers to the highest resolution series.
- This script drops any partial tiles on the right/bottom edges to ensure all tiles are exactly 960x720 with **no overlaps**.
- The GeoJSON is assumed to be in the **same pixel coordinate space** as the chosen `--level`.
- If a tile has no overlap with any region, it is saved under `Unlabeled/` when `--geojson` is provided.
- If your input TIFF uses a compressed codec (e.g., LZW), install `imagecodecs` for reading via tifffile.

"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, Any, Iterable, Union, Sequence
import json
from shapely.geometry import shape, box
from shapely.ops import unary_union

import numpy as np

# Dependencies
_have_tifffile = False
_have_imagecodecs = False

try:
    import tifffile as tiff  # type: ignore

    _have_tifffile = True
except Exception:
    _have_tifffile = False

try:
    import imagecodecs  # type: ignore  # noqa: F401

    _have_imagecodecs = True
except Exception:
    _have_imagecodecs = False

import zarr  # type: ignore


# -------------------- helpers --------------------


def _pick_series(tf: "tiff.TiffFile", level: Optional[int]):
    # Select appropriate series/page; favor specified level if provided
    if level is None:
        return tf.series[0]
    try:
        return tf.series[level]
    except Exception:
        return tf.series[0]


def _yx_indices(axes: str) -> tuple[int, int]:
    y_idx = axes.find("Y")
    x_idx = axes.find("X")
    if y_idx == -1 or x_idx == -1:
        raise RuntimeError(f"Could not find Y/X axes in series axes '{axes}'")
    return y_idx, x_idx


def _samples_index(axes: str) -> int:
    if "S" in axes:
        return axes.find("S")
    if "C" in axes:
        return axes.find("C")
    return -1


def _first_zarr_array(zroot: Any):
    if hasattr(zroot, "dtype") and hasattr(zroot, "ndim"):
        return zroot
    if hasattr(zroot, "arrays"):
        try:
            # Newer tifffile returns a group when using aszarr() on pyramids
            # Pick the first array child
            children = list(zroot.arrays())
            if children:
                return children[0][1]
        except Exception:
            pass
    # Fallback: attempt known attribute
    if hasattr(zroot, "get"):
        for k in ("0", 0, "array", "data"):
            try:
                v = zroot.get(k)
                if v is not None:
                    return v
            except Exception:
                continue
    # Give up
    return zroot


def _read_window(
    zarr_node, axes: str, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """
    Slice a window and **return with axes ordered as (Y, X[, C])**.
    Other axes (e.g., Z, T, R) are fixed at 0 and removed.
    """
    s_idx = _samples_index(axes)
    # Build slices in original axes order
    slices = []
    kept_axes = []  # axes that will remain after indexing
    for i, ax in enumerate(axes):
        if ax == "Y":
            slices.append(slice(y0, y1))
            kept_axes.append("Y")
        elif ax == "X":
            slices.append(slice(x0, x1))
            kept_axes.append("X")
        elif i == s_idx:
            slices.append(slice(None))
            kept_axes.append("S" if "S" in axes else ("C" if "C" in axes else "S"))
        else:
            # collapse other axes
            slices.append(0)

    tile = zarr_node[tuple(slices)]

    # Determine axis positions in the resulting tile
    # kept_axes aligns 1:1 with tile.ndim
    # Normalize channel label
    kept_axes_norm = []
    for ax in kept_axes:
        if ax in ("S", "C"):
            kept_axes_norm.append("C")
        elif ax in ("Y", "X"):
            kept_axes_norm.append(ax)
        else:
            # shouldn't occur but ignore if present
            pass

    # Build permutation to (Y, X[, C])
    try:
        y_pos = kept_axes_norm.index("Y")
        x_pos = kept_axes_norm.index("X")
    except ValueError:
        # If missing, bail out
        return tile

    perm = [y_pos, x_pos]
    if "C" in kept_axes_norm:
        c_pos = kept_axes_norm.index("C")
        perm.append(c_pos)

    if tile.ndim != len(perm):
        # If there are accidental leftovers, drop extras by moving desired axes first
        tile = np.moveaxis(tile, [y_pos, x_pos], [0, 1])
        if "C" in kept_axes_norm and tile.ndim >= 3:
            # Move channel to last axis (2)
            tile = np.moveaxis(tile, c_pos, 2)
    else:
        tile = tile.transpose(perm)

    return tile


def _yx_from_axes(shape: Tuple[int, ...], axes: str) -> Tuple[int, int]:
    y_idx = axes.find("Y")
    x_idx = axes.find("X")
    if y_idx == -1 or x_idx == -1:
        raise RuntimeError(f"Could not find Y/X in axes='{axes}'")
    return int(shape[y_idx]), int(shape[x_idx])


def _get_best_level_index(series) -> int:
    # Choose the level (in series.levels) with the largest spatial area
    best_i, best_area = 0, -1
    for i, lvl in enumerate(getattr(series, "levels", [])):
        try:
            y, x = _yx_from_axes(lvl.shape, lvl.axes)
            area = y * x
            if area > best_area:
                best_i, best_area = i, area
        except Exception:
            continue
    return best_i


def _open_zarr_array_for_series(series, preferred_level: Optional[int]):
    """
    Return (zarr_array, axes) for the correct pyramid level:
    - If preferred_level is given and valid, use it.
    - Else pick the level with the largest Y*X.
    - If no levels are present, fall back to series.aszarr() and pick
      the child array with the largest last-2-dim area.
    """
    # Case 1: series has explicit levels (tifffile WSIs)
    lvls = getattr(series, "levels", None)
    if lvls:
        if preferred_level is not None and 0 <= preferred_level < len(lvls):
            lvl_idx = preferred_level
        else:
            lvl_idx = _get_best_level_index(series)
        zr = series.aszarr(level=lvl_idx)  # zarr store for that level
        znode = zarr.open(zr, mode="r")
        # Some tifffile versions return an Array; some return a Group (rare).
        if hasattr(znode, "dtype") and hasattr(znode, "ndim"):
            arr = znode
        else:
            # Pick child with largest last-2-dim area
            best, best_area = None, -1
            for _, a in znode.arrays():
                shape = getattr(a, "shape", ())
                area = int(shape[-1]) * int(shape[-2]) if len(shape) >= 2 else -1
                if area > best_area:
                    best, best_area = a, area
            arr = best if best is not None else znode
        return arr, lvls[lvl_idx].axes

    # Case 2: no levels → open root and pick largest child
    zr = series.aszarr()
    zroot = zarr.open(zr, mode="r")
    if hasattr(zroot, "dtype") and hasattr(zroot, "ndim"):
        return zroot, series.axes
    if hasattr(zroot, "arrays"):
        best, best_area = None, -1
        for _, a in zroot.arrays():
            shp = getattr(a, "shape", ())
            area = int(shp[-1]) * int(shp[-2]) if len(shp) >= 2 else -1
            if area > best_area:
                best, best_area = a, area
        if best is not None:
            return best, series.axes
    return zroot, series.axes


# -------------------- region utils (GeoJSON) --------------------

from typing import Dict
LabelGeomMap = Dict[str, "shapely.geometry.base.BaseGeometry"]


def _extract_label(props: dict) -> str | None:
    # Preferred (QuPath-style)
    cls = props.get("classification")
    if isinstance(cls, dict) and isinstance(cls.get("name"), str):
        return cls["name"]

    # Fallbacks
    for key in ("name", "Name", "label", "Label", "region", "Region", "class", "Class"):
        if isinstance(props.get(key), str):
            return props[key]
    return None


def _iter_features(gj: dict | list) -> Iterable[dict]:
    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
        yield from gj.get("features", [])
    elif isinstance(gj, dict) and gj.get("type") == "Feature":
        yield gj
    elif isinstance(gj, list):
        for x in gj:
            if isinstance(x, dict):
                yield x


def _load_regions_from_geojson(
    geojson_paths: Union[str, Path, Iterable[Union[str, Path]]],
) -> LabelGeomMap | None:
    """
    Read one or many GeoJSON files and return: {label -> unary_union(Multi)Polygon}.
    Labels are discovered from properties (no hard-coded buckets).
    Returns None if nothing could be parsed.
    """
    # normalize to list of paths
    if isinstance(geojson_paths, (str, Path)):
        paths = [geojson_paths]
    else:
        paths = list(geojson_paths)

    per_label_geoms: dict[str, list] = {}

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                gj = json.load(f)
        except Exception:
            continue

        for feat in _iter_features(gj):
            try:
                geom = shape(feat.get("geometry"))
            except Exception:
                continue
            props = feat.get("properties", {}) if isinstance(feat, dict) else {}
            label = _extract_label(props)
            if not label:
                continue
            per_label_geoms.setdefault(label, []).append(geom)

    if not per_label_geoms:
        return None

    unions: LabelGeomMap = {}
    for label, geoms in per_label_geoms.items():
        if not geoms:
            continue
        if len(geoms) == 1:
            unions[label] = geoms[0]
        else:
            try:
                unions[label] = unary_union(geoms)
            except Exception:
                unions[label] = geoms[0]  # conservative fallback

    return unions if unions else None


def _majority_label_for_rect(
    x0: int, y0: int, x1: int, y1: int, unions: LabelGeomMap | None
) -> str:
    """
    Given a tile rectangle in pixel coordinates and dict of label -> (Multi)Polygon,
    route the tile to:
      - "minimal_tissue" if the GeoJSON contains a "tissue" label and the tile has
        <30% tissue area.
      - Otherwise, the label with maximum intersection area (excluding "tissue").
    Returns "Unlabeled" if no (non-tissue) overlap is found, or if unions is None.
    """
    if not unions:
        return "Unlabeled"

    rect = box(x0, y0, x1, y1)
    tile_area = float(rect.area) if rect.area else 0.0

    # Tissue gate: only tiles with >=30% tissue proceed to normal labeling.
    # The "tissue" label may vary in case ("Tissue", "tissue", ...), so match case-insensitively.
    tissue_key = None
    for k in unions.keys():
        if isinstance(k, str) and k.lower() == "tissue":
            tissue_key = k
            break

    if tissue_key is not None and tile_area > 0:
        try:
            tissue_poly = unions.get(tissue_key)
            inter = rect.intersection(tissue_poly) if tissue_poly is not None else None
            tissue_area = float(inter.area) if (inter is not None and not inter.is_empty) else 0.0
        except Exception:
            tissue_area = 0.0

        if (tissue_area / tile_area) < 0.30:
            return "minimal_tissue"

    # Existing scheme (majority intersection), but exclude "tissue" so it doesn't dominate.
    best_label, best_area = "Unlabeled", 0.0
    for label, poly in unions.items():
        if isinstance(label, str) and label.lower() == "tissue":
            continue
        try:
            inter = rect.intersection(poly)
            a = inter.area if not inter.is_empty else 0.0
        except Exception:
            a = 0.0
        if a > best_area:
            best_area = a
            best_label = label

    return best_label if best_area > 0 else "Unlabeled"


# -------------------- tifffile-based tiling --------------------



def _tile_with_tifffile(
    input_path: Path,
    output_dir: Path,
    tile_w: int,
    tile_h: int,
    level: Optional[int],
    region_unions: Optional[dict] = None,
) -> Tuple[int, int, int]:
    if not _have_tifffile:
        raise SystemExit(
            "tifffile is not installed. Please install via: pip install tifffile"
        )
    if not _have_imagecodecs:
        raise SystemExit(
            "Your TIFF likely uses a codec (e.g., LZW) that requires 'imagecodecs'. Please: pip install imagecodecs"
        )

    # Keep the underlying file handle open during async zarr reads
    with tiff.TiffFile(str(input_path), _multifile=False) as tf:
        series = _pick_series(tf, level)
        znode, axes = _open_zarr_array_for_series(series, preferred_level=level)

        # Log what we chose (optional but useful)
        try:
            print(f"[tiler] Using axes='{axes}', zarr shape={tuple(znode.shape)}")
        except Exception:
            pass

        y_idx, x_idx = _yx_indices(axes)
        H = int(znode.shape[y_idx])
        W = int(znode.shape[x_idx])

        n_rows = H // tile_h
        n_cols = W // tile_w

        for r in range(n_rows):
            y0, y1 = r * tile_h, r * tile_h + tile_h
            for c in range(n_cols):
                x0, x1 = c * tile_w, c * tile_w + tile_w
                arr = _read_window(znode, axes, y0, y1, x0, x1)
                if arr.size == 0:
                    continue
                # Choose output subdirectory by majority annotation if provided
                label = (
                    _majority_label_for_rect(x0, y0, x1, y1, region_unions)
                    if region_unions is not None
                    else ""
                )
                if not label:
                    label = "Unlabeled"
                out_dir = output_dir / label
                out_dir.mkdir(parents=True, exist_ok=True)
                _save_tile(arr, out_dir, r, c, input_path.stem)
        total = n_rows * n_cols
        return n_rows, n_cols, total


# -------------------- common save --------------------


def _save_tile(arr: np.ndarray, output_dir: Path, r: int, c: int, image_stem: str):
    """Save a tile as TIFF with correct channel semantics.
    
    - Writes standard TIFF with correct SamplesPerPixel and planar config (no OME-XML)
      to avoid shape/axes mismatches. Keeps all channels & dtype.
    """
    stem = f"{image_stem}_tile_{r}_{c}"
    out_path = output_dir / f"{stem}.tiff"

    # Ensure channel-last for multi-channel
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        # Heuristic: if channels are first, move to last
        arr = np.moveaxis(arr, 0, -1)

    # Always save as TIFF using tifffile
    import tifffile as _tif

    # Convert channel order if needed handled above; write as contiguous
    if arr.ndim == 2:
        _tif.imwrite(out_path, arr, photometric="minisblack", planarconfig="contig")
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            _tif.imwrite(
                out_path, arr[:, :, 0], photometric="minisblack", planarconfig="contig"
            )
        else:
            _tif.imwrite(
                out_path,
                arr,
                photometric="rgb" if arr.shape[2] in (3, 4) else "minisblack",
                planarconfig="contig",
            )
    else:
        raise RuntimeError(f"Unexpected tile array shape {arr.shape}")


# -------------------- unified API --------------------


def tile_tiff(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    tile_w: int = 960,
    tile_h: int = 720,
    level: Optional[int] = 0,
    geojson: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
) -> Tuple[int, int, int]:
    """Tile a (QP)TIFF into non-overlapping patches of size tile_w x tile_h.

    Returns (n_rows, n_cols, n_tiles).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_unions = _load_regions_from_geojson(geojson) if geojson else None

    if geojson:
        # Normalize the list of geojson paths for display
        if isinstance(geojson, (list, tuple)):
            gj_list = [str(p) for p in geojson]
        else:
            gj_list = [str(geojson)]
        print(f"[GeoJSON] Sources: {gj_list}")

    if region_unions:
        labels = sorted(region_unions.keys())
        print(f"[GeoJSON] Discovered {len(labels)} label(s): {labels}")
    else:
        print("[GeoJSON] No labels discovered. Tiles will be routed to 'Unlabeled/'.")

    # Proceed with tiling using the unions (routing happens inside _tile_with_tifffile)
    return _tile_with_tifffile(
        input_path, output_dir, tile_w, tile_h, level, region_unions
    )


# -------------------- CLI --------------------


def main():
    p = argparse.ArgumentParser(
        description="Tile a (QP)TIFF into 960x720 non-overlapping patches"
    )
    p.add_argument(
        "--input", required=True, help="Path to input QPTIFF / pyramidal TIFF"
    )
    p.add_argument("--output_dir", required=True, help="Directory to write tiles")
    p.add_argument(
        "--tile_w", type=int, default=960, help="Tile width in pixels (default: 960)"
    )
    p.add_argument(
        "--tile_h", type=int, default=720, help="Tile height in pixels (default: 720)"
    )
    p.add_argument(
        "--level",
        type=int,
        default=0,
        help="Series level for tifffile reader (0 = highest-res in most files)",
    )
    p.add_argument(
        "--geojson",
        type=str,
        default=None,
        help="Path to GeoJSON file with regions (Inner_Medulla, Outer_Medulla, Cortex)",
    )

    args = p.parse_args()

    n_rows, n_cols, total = tile_tiff(
        input_path=args.input,
        output_dir=args.output_dir,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        level=args.level,
        geojson=args.geojson,
    )
    print(f"Completed. Rows: {n_rows}, Cols: {n_cols}, Total tiles: {total}")


if __name__ == "__main__":
    main()
