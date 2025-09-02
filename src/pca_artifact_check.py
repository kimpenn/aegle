#!/usr/bin/env python3
"""
pca_artifact_check.py
---------------------
Principal-component sweep of a multi-channel CODEX (or any OME-TIFF) with an
*optional* synthetic horizontal + vertical gradient (“technical artifact”).
Steps
1) Optionally add the artifact to all channels.
2) Fit StandardScaler + PCA on a pixel subset.
3) Stream-project the whole image onto the first N PCs.
4) Report PC ↔ (x, y) correlations and save outputs.

Author: Da Kuang (adapted & scripted)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import logging
import sys # Added import
import csv # Added for CSV output

import numpy as np
import psutil # Added for memory tracking
from tifffile import TiffFile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PCA on a CODEX OME-TIFF with optional synthetic "
                    "horizontal / vertical gradient artifacts."
    )
    p.add_argument("file", help="Path to input OME-TIFF (C × H × W).")
    p.add_argument("--components", "-k", type=int, default=3,
                   help="Number of PCs to keep (default: 3).")
    p.add_argument("--sample_pixels", "-s", type=int, default=128_000_000,
                   help="Number of pixels used to fit PCA (default: 128 M).")
    p.add_argument("--chunk_size", "-c", type=int, default=2_000_000,
                   help="Chunk size for streaming projection (default: 2 M).")
    p.add_argument("--full_scale", "-a", type=float, default=0.0,
                   help="Amplitude of synthetic artifact. "
                        "0 = no artifact (default: 0).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for pixel sampling (default: 0).")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging level (default: INFO).")
    p.add_argument("--outdir", "-o", type=Path, default=None,
                help=("Output directory root. "
                        "If set, results will be written to "
                        "<outdir>/<sample_name>/ . "
                        "Default: alongside the input file."))
    return p.parse_args()


# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ----- logging setup -----------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=getattr(logging, args.log_level.upper()),
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(args.seed)
    img_path = Path(args.file).expanduser().resolve()

    # Derive the directory that will hold this sample's outputs
    sample_name = img_path.stem                         # e.g. "D16_Scan1_manual_0"
    # Get rid of . and suffixes, if any
    if "." in sample_name:
        sample_name = sample_name.split(".")[0]          # e.g. "D16_Scan1_manual_0"
    if args.outdir is None:
        sample_dir = img_path.parent / sample_name      # default side-by-side folder
    else:
        sample_dir = args.outdir.expanduser().resolve() / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # File names inside that folder
    out_npy     = sample_dir / f"{sample_name}_pc_maps.npy"
    out_art_png = sample_dir / f"{sample_name}_artifact.png"
    out_pc_png  = sample_dir / f"{sample_name}_pc_maps.png"
    out_csv     = sample_dir / f"{sample_name}_correlations.csv"
    logger.info("Run with the following parameters:")
    logger.info("Input file: %s", img_path)
    logger.info("Output directory: %s", sample_dir)
    logger.info("Input image: %s", img_path)
    logger.info("Output PC maps: %s", out_npy)
    logger.info("Output artifact plot: %s", out_art_png)
    logger.info("Output PC maps plot: %s", out_pc_png)
    logger.info("Output correlations CSV: %s", out_csv)


    # 1. Load image -----------------------------------------------------------
    with TiffFile(img_path) as tif:
        mmap_uint16 = tif.series[0].asarray(out="memmap")  # (C, H, W)    
    C, H, W = mmap_uint16.shape
    logger.info(
        "Loaded image: C=%d, H=%d, W=%d (%s pixels / channel)",
        C, H, W, f"{H*W:,}",
    )
    logger.info(f"Memory usage after loading image: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")

    # 2. Inject artifact ------------------------------------------------------
    if args.full_scale > 0:
        y_rel = (np.arange(H, dtype=np.float32) / (H - 1) - 0.5)[:, None]
        x_rel = (np.arange(W, dtype=np.float32) / (W - 1) - 0.5)[None, :]
        artifact = (y_rel + x_rel) * args.full_scale          # (H, W)

        # Transform to float32 is memory extensive.
        # because astype creates a copy.
        logger.info("Converting to float32...")
        data = mmap_uint16.astype(np.float32)
        data += artifact                                      # broadcast
        logger.info(f"Memory usage after artifact injection and float32 conversion: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")
        logger.info(
            "Injected artifact ±%.0f intensity units (%.2f%% of 16-bit range)",
            args.full_scale / 2,
            100 * args.full_scale / 65535,
        )
        plt.figure(figsize=(5, 4))
        plt.imshow(artifact, cmap="coolwarm")
        plt.title("Synthetic artifact (ΔI)")
        plt.colorbar(); plt.axis("off")
        plt.savefig(out_art_png, dpi=150, bbox_inches="tight")
        logger.info("[saved] %s", out_art_png)
    else:
        data = mmap_uint16.astype(np.float32)
        logger.info(f"Memory usage after float32 conversion (no artifact): {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")

    # 3. Sample pixels for PCA -----------------------------------------------
    total_px = H * W
    n_sample = min(args.sample_pixels, total_px)
    sample_pct = n_sample / total_px * 100
    logger.info(
        "Sampling %s pixels for PCA fit (%.2f%% of total)…",
        f"{n_sample:,}", sample_pct
    )    
    sample_idx = rng.choice(total_px, size=n_sample, replace=False)
    sample_idx.sort()

    flat_view = data.reshape(C, -1).T
    logger.info(f"Memory usage after creating flat_view: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")
    X_sample = flat_view[sample_idx].astype(np.float32)
    logger.info(f"Memory usage after creating X_sample: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")

    logger.info("Sampling %s pixels for PCA fit …", f"{n_sample:,}")
    scaler = StandardScaler().fit(X_sample)
    X_sample_z = scaler.transform(X_sample)

    pca = PCA(n_components=args.components,
              svd_solver="randomized",
              random_state=args.seed).fit(X_sample_z)
    pc_vecs = pca.components_.astype(np.float32)
    logger.info(f"Memory usage after PCA fit: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")
    logger.info("Explained variance ratio: %s",
                np.round(pca.explained_variance_ratio_, 4))

    # Transpose pc_vecs once for repeated use in matrix multiplication
    # pc_vecs shape is (n_components, C), pc_vecs_T shape is (C, n_components)
    pc_vecs_T = pc_vecs.T.astype(np.float32)

    del X_sample, X_sample_z  # free RAM

    # 4. Stream-project the full image ---------------------------------------
    pc_maps_flat = np.empty((args.components, total_px), dtype=np.float32)
    logger.info(f"Memory usage after allocating pc_maps_flat: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")
    logger.info("Projecting full image …")
    for start in tqdm(
        range(0, total_px, args.chunk_size),
        disable=not sys.stdout.isatty()  # Disable progress bar if not TTY
    ):
        end = min(start + args.chunk_size, total_px)
        logger.info(f"Memory usage before processing chunk {start}-{end}: {psutil.Process().memory_info().rss / 1024 ** 2:,.2f} MB")
        X_blk = flat_view[start:end]
        X_blk = (X_blk - scaler.mean_) / scaler.scale_
        pc_maps_flat[:, start:end] = (X_blk @ pc_vecs_T).T

    pc_maps = pc_maps_flat.reshape(args.components, H, W)
    # np.save(out_npy, pc_maps)
    logger.info("[saved] %s", out_npy)

    # 5. Correlation with coordinates ----------------------------------------
    y_coords = np.repeat(np.arange(H, dtype=np.float32), W)
    x_coords = np.tile(np.arange(W, dtype=np.float32), H)

    correlation_data = []
    for k in range(args.components):
        rho_y = np.corrcoef(pc_maps_flat[k], y_coords)[0, 1]
        rho_x = np.corrcoef(pc_maps_flat[k], x_coords)[0, 1]
        logger.info("PC%d: ρ_y=%+.3f, ρ_x=%+.3f", k + 1, rho_y, rho_x)
        correlation_data.append({
            'PC': k + 1,
            'rho_y': rho_y,
            'rho_x': rho_x
        })
    
    # Save correlations to CSV
    with open(out_csv, 'w', newline='') as csvfile:
        fieldnames = ['PC', 'rho_y', 'rho_x']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(correlation_data)
    logger.info("[saved] %s", out_csv)

    # 6. Save PC maps plot -------------------------------------------------------
    cols = args.components
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        pc_img = pc_maps[k]
        vmin, vmax = np.percentile(pc_img, [0.5, 99.5])
        im = ax.imshow(pc_img, vmin=vmin, vmax=vmax, cmap="turbo")
        rho_y = np.corrcoef(pc_maps_flat[k], y_coords)[0, 1]
        rho_x = np.corrcoef(pc_maps_flat[k], x_coords)[0, 1]
        ax.set_title(f"PC{k+1}: ρy={rho_y:+.2f}, ρx={rho_x:+.2f}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_pc_png, dpi=150, bbox_inches="tight")
    logger.info("[saved] %s", out_pc_png)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
