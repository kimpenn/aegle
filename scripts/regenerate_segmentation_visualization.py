#!/usr/bin/env python3
"""Regenerate segmentation visualization assets for an existing pipeline run."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegle.pipeline import (
    LARGE_SAMPLE_PIXEL_THRESHOLD,
    PATCH_SELECTION_SEED,
    PATCH_VIS_MAX,
    PATCH_VIS_MIN,
    PATCH_VIS_SIZE,
    _compute_labeled_boundary,
    _generate_patch_overview,
    _generate_composite_patch_infos,
    _select_representative_patches,
    _count_labels,
)
from aegle.visualization_segmentation import (
    create_nucleus_mask_visualization,
    create_segmentation_overlay,
    create_wholecell_mask_visualization,
)

REASON_LABELS = {
    "high_density": "High density region",
    "repair_hotspot": "Repair hotspot",
    "low_density": "Sparse region",
    "representative": "Representative area",
    "coverage": "Additional coverage",
    "all": "Full-sample patch",
}


class VisualizationRegenerator:
    """Regenerate segmentation ROI assets from saved masks without rerunning the pipeline."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.viz_dir = output_dir / "visualization" / "segmentation"
        self.seg_dir = output_dir / "segmentations"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        config_path = self.output_dir / "copied_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing copied_config.yaml at {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_extracted_image(self) -> np.ndarray:
        array_path = self.output_dir / "extracted_channel_patches.npy.gz"
        if not array_path.exists():
            raise FileNotFoundError(f"Missing extracted_channel_patches at {array_path}")
        with gzip.open(array_path, "rb") as f:
            data = np.load(f)
        if data.ndim == 3:
            return data
        if data.ndim == 4:
            return data[0]
        raise ValueError(f"Unexpected extracted_channel_patches shape: {data.shape}")

    @staticmethod
    def _load_mask(path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            logging.warning("Mask not found: %s", path)
            return None
        return tifffile.imread(str(path))

    def _load_aggregated_masks(self) -> Dict[str, Optional[np.ndarray]]:
        prefix = self.output_dir.name
        masks = {
            "cell_matched_mask": self.seg_dir / f"{prefix}.cell_matched_mask.segmentations.ome.tiff",
            "nucleus_matched_mask": self.seg_dir / f"{prefix}.nucleus_matched_mask.segmentations.ome.tiff",
            "cell": self.seg_dir / f"{prefix}.cell.segmentations.ome.tiff",
            "nucleus": self.seg_dir / f"{prefix}.nucleus.segmentations.ome.tiff",
        }
        return {key: self._load_mask(path) for key, path in masks.items()}

    def regenerate(self) -> None:
        split_mode = (
            self.config.get("patching", {}).get("split_mode", "patches")
        )
        if split_mode != "full_image":
            raise NotImplementedError(
                "This helper currently supports split_mode=full_image only."
            )

        aggregated_masks = self._load_aggregated_masks()
        cell_matched = aggregated_masks.get("cell_matched_mask")
        nucleus_matched = aggregated_masks.get("nucleus_matched_mask")
        if cell_matched is None or nucleus_matched is None:
            raise FileNotFoundError("Missing required matched segmentation masks.")

        extended_image = self._load_extracted_image()
        image_height, image_width = cell_matched.shape
        total_pixels = image_height * image_width
        is_large_sample = total_pixels > LARGE_SAMPLE_PIXEL_THRESHOLD

        composite_infos = _generate_composite_patch_infos(
            image_width,
            image_height,
            aggregated_masks,
            PATCH_VIS_SIZE,
        )
        if not composite_infos:
            raise RuntimeError("No composite patches could be generated from masks.")

        suggested = max(
            1,
            math.ceil(total_pixels / float(PATCH_VIS_SIZE * PATCH_VIS_SIZE * 2)),
        )
        target_patch_count = max(
            PATCH_VIS_MIN,
            min(PATCH_VIS_MAX, suggested),
        )
        target_patch_count = min(target_patch_count, len(composite_infos))

        selected_infos = _select_representative_patches(
            composite_infos,
            target_patch_count,
            seed=PATCH_SELECTION_SEED,
        )
        if not selected_infos:
            raise RuntimeError("Representative selection returned no patches.")

        for display_idx, info in enumerate(selected_infos):
            info.setdefault("selection_reason", "representative")
            info["display_index"] = display_idx
            info.setdefault("display_name", f"ROI {display_idx + 1}")
            info["selection_reason_label"] = REASON_LABELS.get(
                info.get("selection_reason"),
                "Representative area",
            )
            info["source_patch_index"] = None
            info["metadata_index"] = None
            info["seg_idx"] = info.get("seg_idx")

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        patch_entries = []

        base_channel = extended_image[:, :, 0] if extended_image.ndim >= 3 else extended_image

        for info in selected_infos:
            display_idx = info["display_index"]
            display_name = info["display_name"]
            selection_label = info.get("selection_reason_label")

            x_start = int(info.get("x_start", 0))
            y_start = int(info.get("y_start", 0))
            width = int(info.get("width", PATCH_VIS_SIZE))
            height = int(info.get("height", PATCH_VIS_SIZE))
            x_end = min(image_width, x_start + width)
            y_end = min(image_height, y_start + height)

            image_patch = extended_image[y_start:y_end, x_start:x_end]
            base_patch = image_patch[:, :, 0] if image_patch.ndim >= 3 else image_patch

            matched_cell_patch = cell_matched[y_start:y_end, x_start:x_end]
            matched_nucleus_patch = nucleus_matched[y_start:y_end, x_start:x_end]
            matched_cell_boundary = _compute_labeled_boundary(matched_cell_patch)
            matched_nucleus_boundary = _compute_labeled_boundary(matched_nucleus_patch)

            original_cell = aggregated_masks.get("cell")
            original_nucleus = aggregated_masks.get("nucleus")
            original_cell_patch = (
                original_cell[y_start:y_end, x_start:x_end]
                if original_cell is not None
                else None
            )
            original_nucleus_patch = (
                original_nucleus[y_start:y_end, x_start:x_end]
                if original_nucleus is not None
                else None
            )
            original_cell_boundary = _compute_labeled_boundary(original_cell_patch)
            original_nucleus_boundary = _compute_labeled_boundary(original_nucleus_patch)

            matched_cell_count = int(info.get("cell_count", 0))
            matched_nucleus_count = int(info.get("nucleus_count", 0))
            original_cell_count = (
                _count_labels(original_cell_patch) if original_cell_patch is not None else 0
            )
            original_nucleus_count = (
                _count_labels(original_nucleus_patch) if original_nucleus_patch is not None else 0
            )

            files_record: Dict[str, str] = {}

            def save_figure(fig, filename: str) -> None:
                fig.savefig(self.viz_dir / filename, dpi=150, bbox_inches="tight")
                plt.close(fig)

            # Repaired overlay
            fig = create_segmentation_overlay(
                base_patch,
                matched_nucleus_patch,
                matched_cell_patch,
                show_ids=False,
                alpha=0.6,
                reference_nucleus_mask=original_nucleus_patch,
                reference_cell_mask=original_cell_patch,
                show_reference_highlights=False,
                fill_cell_mask=False,
                fill_nucleus_mask=False,
                cell_boundary_mask=matched_cell_boundary,
                nucleus_boundary_mask=matched_nucleus_boundary,
                custom_title=f"Segmentation Overlay — {display_name}",
                cell_count=matched_cell_count,
                nucleus_count=matched_nucleus_count,
            )
            repaired_name = f"vis_segmentation_overlay_patch_{display_idx}.png"
            save_figure(fig, repaired_name)
            files_record["overlay_repaired"] = repaired_name

            if original_nucleus_patch is not None or original_cell_patch is not None:
                fig = create_segmentation_overlay(
                    base_patch,
                    original_nucleus_patch,
                    original_cell_patch,
                    show_ids=False,
                    alpha=0.6,
                    fill_cell_mask=False,
                    fill_nucleus_mask=False,
                    cell_boundary_mask=original_cell_boundary,
                    nucleus_boundary_mask=original_nucleus_boundary,
                    custom_title=f"Segmentation Overlay (Pre-Repair) — {display_name}",
                    cell_count=original_cell_count or None,
                    nucleus_count=original_nucleus_count or None,
                )
                pre_name = f"vis_segmentation_overlay_pre_repair_patch_{display_idx}.png"
                save_figure(fig, pre_name)
                files_record["overlay_pre_repair"] = pre_name

            fig = create_nucleus_mask_visualization(base_patch, matched_nucleus_patch, show_ids=False)
            nucleus_name = f"vis_nucleus_mask_patch_{display_idx}.png"
            save_figure(fig, nucleus_name)
            files_record["nucleus_mask"] = nucleus_name

            fig = create_wholecell_mask_visualization(base_patch, matched_cell_patch, show_ids=False)
            cell_name = f"vis_wholecell_mask_patch_{display_idx}.png"
            save_figure(fig, cell_name)
            files_record["wholecell_mask"] = cell_name

            if original_nucleus_patch is not None:
                fig = create_nucleus_mask_visualization(base_patch, original_nucleus_patch, show_ids=False)
                pre_nucleus_name = f"vis_nucleus_mask_pre_repair_patch_{display_idx}.png"
                save_figure(fig, pre_nucleus_name)
                files_record["nucleus_mask_pre_repair"] = pre_nucleus_name

            if original_cell_patch is not None:
                fig = create_wholecell_mask_visualization(base_patch, original_cell_patch, show_ids=False)
                pre_cell_name = f"vis_wholecell_mask_pre_repair_patch_{display_idx}.png"
                save_figure(fig, pre_cell_name)
                files_record["wholecell_mask_pre_repair"] = pre_cell_name

            if original_nucleus_patch is not None:
                fig = create_segmentation_overlay(
                    base_patch,
                    original_nucleus_patch,
                    matched_cell_patch,
                    show_ids=False,
                    alpha=0.6,
                    reference_nucleus_mask=original_nucleus_patch,
                    reference_cell_mask=None,
                    show_cell_overlay=False,
                    show_nucleus_overlay=False,
                    custom_title=f"Unmatched Nuclei — {display_name}\n\n",


                    show_reference_highlights=True,
                    cell_boundary_mask=matched_cell_boundary,
                    nucleus_boundary_mask=matched_nucleus_boundary,
                )
                unmatched_nucleus_name = (
                    f"vis_segmentation_overlay_unmatched_nucleus_patch_{display_idx}.png"
                )
                save_figure(fig, unmatched_nucleus_name)
                files_record["overlay_unmatched_nucleus"] = unmatched_nucleus_name

            if original_cell_patch is not None:
                fig = create_segmentation_overlay(
                    base_patch,
                    None,
                    matched_cell_patch,
                    show_ids=False,
                    alpha=0.6,
                    reference_nucleus_mask=None,
                    reference_cell_mask=original_cell_patch,
                    show_cell_overlay=False,
                    show_nucleus_overlay=False,
                    custom_title=f"Unmatched Cells — {display_name}\n\n",


                    show_reference_highlights=True,
                    cell_boundary_mask=matched_cell_boundary,
                    nucleus_boundary_mask=matched_nucleus_boundary,
                )
                unmatched_cell_name = (
                    f"vis_segmentation_overlay_unmatched_cell_patch_{display_idx}.png"
                )
                save_figure(fig, unmatched_cell_name)
                files_record["overlay_unmatched_cell"] = unmatched_cell_name

            patch_entries.append(
                {
                    "display_index": display_idx,
                    "display_name": display_name,
                    "selection_reason": info.get("selection_reason"),
                    "selection_reason_label": selection_label,
                    "segmentation_index": None,
                    "source_patch_index": None,
                    "metadata_index": None,
                    "x_start": x_start,
                    "y_start": y_start,
                    "width": x_end - x_start,
                    "height": y_end - y_start,
                    "cell_count": int(info.get("cell_count", 0)),
                    "nucleus_count": int(info.get("nucleus_count", 0)),
                    "matched_fraction": info.get("matched_fraction"),
                    "unmatched_cells": float(info.get("unmatched_cells", 0.0)),
                    "unmatched_nuclei": float(info.get("unmatched_nuclei", 0.0)),
                    "cell_density": float(info.get("cell_density", 0.0)),
                    "files": files_record,
                }
            )

        overview_path: Optional[str] = None
        if is_large_sample and selected_infos:
            try:
                overview_channel = base_channel if base_channel.ndim == 2 else base_channel[:, :]
                overview_rel = _generate_patch_overview(
                    str(self.viz_dir),
                    overview_channel,
                    (image_height, image_width),
                    selected_infos,
                )
                overview_path = os.path.basename(overview_rel) if overview_rel else None
            except Exception as exc:
                logging.warning("Failed to regenerate overview image: %s", exc)
                overview_path = None

        summary = {
            "is_large_sample": bool(is_large_sample),
            "large_sample_threshold": LARGE_SAMPLE_PIXEL_THRESHOLD,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "total_pixels": int(total_pixels),
            "patch_visualization_size": PATCH_VIS_SIZE,
            "target_patch_count": int(target_patch_count),
            "selected_patch_count": len(patch_entries),
            "overview_image": overview_path,
            "patches": patch_entries,
        }

        summary_path = self.viz_dir / "segmentation_patch_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info("Wrote updated segmentation summary: %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate segmentation visualization from saved masks (full_image mode only)."
    )
    parser.add_argument("output_dir", help="Path to pipeline output directory")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    regenerator = VisualizationRegenerator(output_dir)
    regenerator.regenerate()
    logging.info("Segmentation visualization regeneration completed.")


if __name__ == "__main__":
    main()
