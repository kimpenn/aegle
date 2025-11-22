"""Lightweight helpers for constructing tiny test assets on the fly."""
from __future__ import annotations

import os
from types import SimpleNamespace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
import zstandard as zstd


def write_tiny_ome_tiff(path: os.PathLike, shape: Sequence[int] = (16, 16, 4), *, dtype=np.uint16) -> Path:
    """Create a deterministic multi-channel OME-TIFF for tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    data = rng.integers(low=0, high=1000, size=shape, dtype=dtype)
    # tifffile expects channel-first by default; we transpose to match CodexImage expectations later.
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(data.transpose(2, 0, 1), photometric="minisblack")
    return path


def write_antibodies_tsv(path: os.PathLike, channel_names: Iterable[str]) -> Path:
    """Write a minimal antibodies.tsv with given channel names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["antibody_name"]
    rows.extend(channel_names)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def make_dummy_args(*, out_dir: os.PathLike, data_dir: os.PathLike = "."):
    """Return a SimpleNamespace with the minimal attrs Codex helpers expect."""
    return SimpleNamespace(out_dir=str(out_dir), data_dir=str(data_dir))


def write_zstd_npy(array: np.ndarray, path: os.PathLike, *, threads: int = 0) -> Path:
    """Compress a numpy array to a .npy.zst file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(threads=threads)
    with open(path, "wb") as raw:
        with compressor.stream_writer(raw) as writer:
            np.save(writer, array, allow_pickle=False)
    return path
