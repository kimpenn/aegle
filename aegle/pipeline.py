import os
import json
import math
import random
import gzip
import numpy as np
import logging
import pickle
import shutil
import time
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
from typing import Dict, List, Optional, Tuple
from skimage.segmentation import find_boundaries

from aegle.codex_image import CodexImage
from aegle.codex_patches import CodexPatches

from aegle.visualization import save_patches_rgb

from aegle.segment import run_cell_segmentation, visualize_cell_segmentation
from aegle.evaluation import run_seg_evaluation
from aegle.cell_profiling import run_cell_profiling
from aegle.segmentation_analysis.segmentation_analysis import run_segmentation_analysis
from aegle.visualization_segmentation import (
    create_segmentation_overlay,
    create_quality_heatmaps,
    plot_cell_morphology_stats,
    visualize_segmentation_errors,
    create_nucleus_mask_visualization,
    create_wholecell_mask_visualization
)
from aegle.report_generator import generate_pipeline_report
# from aegle.segmentation_analysis.intensity_analysis import bias_analysis, distribution_analysis
# from aegle.segmentation_analysis.spatial_analysis import density_metrics
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def _count_labels(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    unique_labels = np.unique(mask)
    return int(np.sum(unique_labels > 0))


def _extract_matching_value(stats: Dict, key: str) -> float:
    """Safely extract a numeric value from matching statistics."""
    if not isinstance(stats, dict):
        return 0.0

    value = stats.get(key)
    if value is None:
        return 0.0

    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    elif hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        value = arr.flat[0]

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_patch_metrics(seg_result: Dict, original_result: Optional[Dict]) -> Dict[str, float]:
    """Compute segmentation metrics useful for visualization selection."""
    matched_cell_mask = seg_result.get("cell_matched_mask", seg_result.get("cell"))
    matched_nucleus_mask = seg_result.get("nucleus_matched_mask", seg_result.get("nucleus"))

    metrics = {
        "cell_count": _count_labels(matched_cell_mask),
        "nucleus_count": _count_labels(matched_nucleus_mask),
        "matched_fraction": seg_result.get("matched_fraction"),
    }

    matching_stats = seg_result.get("matching_stats") or {}
    whole_stats = matching_stats.get("whole_cell", {})
    nucleus_stats = matching_stats.get("nucleus", {})

    metrics["unmatched_cells"] = _extract_matching_value(whole_stats, "unmatched")
    metrics["unmatched_nuclei"] = _extract_matching_value(nucleus_stats, "unmatched")
    metrics["total_cells_original"] = _count_labels(original_result.get("cell")) if original_result else None
    metrics["total_nuclei_original"] = _count_labels(original_result.get("nucleus")) if original_result else None

    matched_fraction = metrics.get("matched_fraction")
    if matched_fraction is not None:
        try:
            matched_fraction = float(matched_fraction)
        except (TypeError, ValueError):
            matched_fraction = None
    metrics["matched_fraction"] = matched_fraction

    return metrics


def _select_representative_patches(
    patch_infos: List[Dict],
    target_count: int,
    seed: int = 17,
) -> List[Dict]:
    """Select a representative subset of patches for visualization."""

    def has_content(info: Dict) -> bool:
        """Return True when a patch contains meaningful segmentation content."""

        def to_float(key: str) -> float:
            value = info.get(key)
            if value is None:
                return 0.0
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        cell_count = to_float("cell_count")
        nucleus_count = to_float("nucleus_count")
        if cell_count >= 200.0 or nucleus_count >= 200.0:
            return True

        unmatched_nuclei = to_float("unmatched_nuclei")
        if unmatched_nuclei >= 200.0:
            return True

        unmatched_cells = to_float("unmatched_cells")
        return unmatched_cells >= 200.0

    if target_count <= 0 or len(patch_infos) <= target_count:
        # Copy to avoid mutating original references
        result = [info.copy() for info in patch_infos if has_content(info)]
        if not result:
            result = [info.copy() for info in patch_infos]
        for info in result:
            info.setdefault("selection_reason", "all")
        return result

    info_copies = [info.copy() for info in patch_infos]
    selected: List[Dict] = []
    selected_ids = set()
    rng = random.Random(seed)

    def pick_best(key_func, reverse: bool, reason: str) -> None:
        candidates = []
        for info in info_copies:
            if info["seg_idx"] in selected_ids:
                continue
            if not has_content(info):
                continue
            value = key_func(info)
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            candidates.append((value, info))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0], reverse=reverse)
        chosen = candidates[0][1]
        chosen["selection_reason"] = reason
        selected.append(chosen)
        selected_ids.add(chosen["seg_idx"])

    def pick_random(reason: str) -> None:
        remaining = [info for info in info_copies if info["seg_idx"] not in selected_ids]
        if not remaining:
            return
        prioritized = [info for info in remaining if has_content(info)]
        pool = prioritized if prioritized else remaining
        chosen = rng.choice(pool)
        chosen["selection_reason"] = reason
        selected.append(chosen)
        selected_ids.add(chosen["seg_idx"])

    # Priority selections
    pick_best(lambda info: info.get("cell_count", 0), True, "high_density")
    pick_best(lambda info: info.get("repair_score", 0), True, "repair_hotspot")
    pick_best(lambda info: info.get("cell_count", 0), False, "low_density")
    pick_random("representative")

    # Fill remaining slots prioritising high cell counts
    if len(selected) < target_count:
        remaining = [info for info in info_copies if info["seg_idx"] not in selected_ids]
        remaining.sort(key=lambda info: info.get("cell_count", 0), reverse=True)
        prioritized = [info for info in remaining if has_content(info)]
        fallback = [info for info in remaining if not has_content(info)]

        for pool in (prioritized, fallback):
            for info in pool:
                info.setdefault("selection_reason", "coverage")
                selected.append(info)
                selected_ids.add(info["seg_idx"])
                if len(selected) >= target_count:
                    break
            if len(selected) >= target_count:
                break

    # Clamp to target count and ensure deterministic order by selection_reason priority then seg_idx
    priority_order = {
        "high_density": 0,
        "repair_hotspot": 1,
        "low_density": 2,
        "representative": 3,
        "coverage": 4,
        "all": 5,
    }

    selected = selected[:target_count]
    selected.sort(
        key=lambda info: (
            priority_order.get(info.get("selection_reason", "coverage"), 9),
            info.get("seg_idx", 0),
        )
    )

    return selected


def _generate_patch_overview(
    viz_dir: str,
    base_image: np.ndarray,
    image_size: Tuple[int, int],
    patches: List[Dict],
    max_side: int = 512,
) -> Optional[str]:
    """Create a downsampled overview image with patch extents highlighted."""
    if base_image is None or base_image.size == 0 or not patches:
        return None

    height, width = image_size
    scale = max_side / max(height, width)
    scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))

    if base_image.ndim == 2:
        display_img = np.stack([base_image] * 3, axis=-1)
    else:
        display_img = base_image.copy()

    display_img = display_img.astype(np.float32)
    img_min, img_max = display_img.min(), display_img.max()
    if img_max > img_min:
        display_img = (display_img - img_min) / (img_max - img_min)
    else:
        display_img = np.zeros_like(display_img)

    resized = cv2.resize(display_img, scaled_size, interpolation=cv2.INTER_AREA)
    overview_img = (resized * 255).clip(0, 255).astype(np.uint8)

    colors = [
        (220, 20, 60),  # crimson
        (65, 105, 225),  # royal blue
        (60, 179, 113),  # medium sea green
        (238, 130, 238),  # violet
        (255, 165, 0),  # orange
        (72, 61, 139),  # dark slate blue
        (70, 130, 180),  # steel blue
        (205, 92, 92),   # indian red
    ]

    label_entries = []
    max_x = max(1, scaled_size[0] - 30)
    max_y = max(1, scaled_size[1] - 10)

    for idx, patch in enumerate(patches):
        color = colors[idx % len(colors)]
        x0 = int(patch["x_start"] * scale)
        y0 = int(patch["y_start"] * scale)
        x1 = int((patch["x_start"] + patch["width"]) * scale)
        y1 = int((patch["y_start"] + patch["height"]) * scale)
        cv2.rectangle(overview_img, (x0, y0), (x1, y1), color, thickness=2)
        label = f"{idx + 1}"
        text_x = min(max(5, x0 + 8), max_x)
        text_y = min(max(18, y0 + 24), max_y)
        label_entries.append((text_x, text_y, color, label))

    for text_x, text_y, color, label in label_entries:
        cv2.putText(
            overview_img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overview_img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    output_path = os.path.join(viz_dir, "segmentation_patch_overview.png")
    cv2.imwrite(output_path, cv2.cvtColor(overview_img, cv2.COLOR_RGB2BGR))
    return output_path


def _load_cell_metadata_dataframe(profiling_out_dir: str) -> Optional[pd.DataFrame]:
    """Load merged cell metadata if available, concatenating patch files as needed."""
    combined_path = os.path.join(profiling_out_dir, 'cell_metadata.csv')
    if os.path.exists(combined_path):
        try:
            return pd.read_csv(combined_path)
        except Exception as exc:
            logging.warning('Failed to read %s: %s', combined_path, exc)

    patch_paths = sorted(glob.glob(os.path.join(profiling_out_dir, 'patch-*-cell_metadata.csv')))
    if not patch_paths:
        return None

    frames: List[pd.DataFrame] = []
    for path_str in patch_paths:
        try:
            frames.append(pd.read_csv(path_str))
        except Exception as exc:
            logging.warning('Failed to read %s: %s', path_str, exc)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return None


LARGE_SAMPLE_PIXEL_THRESHOLD = 1_000_000
PATCH_VIS_SIZE = 1024
PATCH_VIS_MIN = 4
PATCH_VIS_MAX = 8
PATCH_SELECTION_SEED = 17


def run_pipeline(config, args):
    """
    Run the CODEX image analysis pipeline.

    Args:
        config (dict): Configuration parameters loaded from YAML file.
        args (Namespace): Command-line arguments parsed by argparse.
    """
    logging.info("----- Running pipeline with provided configuration and arguments.")
    os.makedirs(args.out_dir, exist_ok=True)
    logging.info(f"Output directory set to: {args.out_dir}")
    copied_config_path = os.path.join(args.out_dir, "copied_config.yaml")
    shutil.copy(args.config_file, copied_config_path)

    cell_metadata_df: Optional[pd.DataFrame] = None
    resume_stage = getattr(args, "resume_stage", None)
    # Extract split_mode early so it's available in all code paths (including resume mode)
    split_mode = config.get("patching", {}).get("split_mode", "patches")
    if resume_stage:
        logging.info("Resume mode detected. Stage: %s", resume_stage)

    # Handle resume mode - load existing data and run requested stage
    if resume_stage == "cell_profiling":
        logging.info(
            "Skipping image loading and segmentation; resuming cell profiling using existing outputs in %s",
            args.out_dir,
        )
        codex_patches = CodexPatches.load_from_outputs(config, args)
        run_cell_profiling(codex_patches, config, args)
        logging.info("Cell profiling completed (resume mode).")
        # Continue to post-profiling stages (visualization, analysis, report) instead of returning
    elif resume_stage == "visualization":
        logging.info(
            "Skipping to visualization/report; loading existing outputs from %s",
            args.out_dir,
        )
        codex_patches = CodexPatches.load_from_outputs(config, args)
        logging.info("Loaded existing outputs. Proceeding to visualization and report generation.")
        # Continue to post-profiling stages (visualization, analysis, report)
    else:
        # ---------------------------------
        # (A) Load Image and Antibodies Data
        # ---------------------------------
        # Step 1: Initialize CodexImage object
        # - Read and Codex image as well the dataframe about antibodies
        logging.info("----- Initializing CodexImage object.")
        codex_image = CodexImage(config, args)
        logging.info("CodexImage object initialized successfully.")
        if config["data"]["generate_channel_stats"]:
            logging.info("----- Generating channel statistics.")
            codex_image.calculate_channel_statistics()
            logging.info("Channel statistics generated successfully.")

        # Step 2: Extract target channels from the image based on configuration
        logging.info("----- Extracting target channels from the image.")
        codex_image.extract_target_channels()
        logging.info("Target channels extracted successfully.")

        # ---------------------------------
        # (B) Patched Image Preprocessing
        # ---------------------------------
        # Step 1: Extend the image for full patch coverage
        logging.info("----- Extending image for full patch coverage.")
        codex_image.extend_image()
        logging.info("Image extension completed successfully.")

        # Whole-sample visualization is handled in the preprocess overview module.
        if config.get("visualization", {}).get("visualize_whole_sample", False):
            logging.info(
                "Skipping whole-sample visualization in main pipeline; overview is generated during preprocess."
            )

        # Step 2: Initialize CodexPatches object and generate patches
        logging.info("----- Initializing CodexPatches object and generating patches.")
        codex_patches = CodexPatches(codex_image, config, args)
        codex_patches.save_patches()
        codex_patches.save_metadata()
        logging.info("Patches generated and metadata saved successfully.")

        # Optional: Add disruptions to patches for testing
        # Extract distruption type and level from config
        disruption_config = config.get("testing", {}).get("data_disruption", {})
        logging.info(f"Disruption config: {disruption_config}")
        has_disruptions = False
        if disruption_config and disruption_config.get("type", None) is not None:
            disruption_type = disruption_config.get("type", None)
            disruption_level = disruption_config.get("level", 1)
            logging.info(
                f"Adding disruptions {disruption_type} at level {disruption_level} to patches for testing."
            )
            codex_patches.add_disruptions(disruption_type, disruption_level)
            logging.info("Disruptions added to patches.")
            has_disruptions = True
            if disruption_config.get("save_disrupted_patches", False):
                logging.info("Saving disrupted patches.")
                codex_patches.save_disrupted_patches()
                logging.info("Disrupted patches saved.")

        # Optional: Visualize patches
        # Priority: if disruptions exist and visualize_patches is True, visualize disrupted patches
        # Otherwise, visualize original patches
        # Skip visualization for full_image mode (only 1 patch = entire image, provides no QC value)
        if config.get("visualization", {}).get("visualize_patches", False):
            if split_mode == "full_image":
                logging.info(
                    "Skipping patch visualization for split_mode='full_image' "
                    "(only 1 patch = entire image, provides no QC value)."
                )
            elif has_disruptions and disruption_config.get("visualize_disrupted", True):
                logging.info("Visualizing disrupted patches.")
                save_patches_rgb(
                    codex_patches.disrupted_extracted_channel_patches,
                    codex_patches.patches_metadata,
                    config,
                    args,
                    max_workers=config.get("visualization", {}).get("workers", None),
                )
                logging.info("Disrupted patch visualization completed.")
            else:
                logging.info("Visualizing original patches.")
                save_patches_rgb(
                    codex_patches.extracted_channel_patches,
                    codex_patches.patches_metadata,
                    config,
                    args,
                    max_workers=config.get("visualization", {}).get("workers", None),
                )
                logging.info("Original patch visualization completed.")

        # ---------------------------------
        # (C) Cell Segmentation and auto evaluation
        # ---------------------------------
        logging.info("Running cell segmentation.")
        run_cell_segmentation(codex_patches, config, args)
        try:
            codex_patches.write_segmentation_manifest()
        except Exception as exc:
            logging.warning(f"Failed to write segmentation manifest: {exc}")

        if config["evaluation"]["compute_metrics"]:
            # TODO: if the number of cells are too large we should skip the evaluation
            run_seg_evaluation(codex_patches, config, args)
            # save the seg_evaluation_metrics to a gzip-compressed pickle file
            metrics_path = os.path.join(args.out_dir, "seg_evaluation_metrics.pkl.gz")
            with gzip.open(metrics_path, "wb") as file_handle:
                pickle.dump(codex_patches.seg_evaluation_metrics, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logging.info("The calculation of evaluation metrics is skipped (evaluation.compute_metrics is False).")
            for metrics_name in ("seg_evaluation_metrics.pkl.gz", "seg_evaluation_metrics.pkl"):
                metrics_path = os.path.join(args.out_dir, metrics_name)
                if os.path.exists(metrics_path):
                    try:
                        os.remove(metrics_path)
                        logging.info(f"Removed stale segmentation metrics file: {metrics_path}")
                    except OSError as exc:
                        logging.warning(f"Failed to remove stale metrics file {metrics_path}: {exc}")

        # ---------------------------------
        # (D) Cell Profiling
        # ---------------------------------
        logging.info("Running cell profiling.")
        run_cell_profiling(codex_patches, config, args)
        logging.info("Cell profiling completed.")

    # Post-profiling stages (run regardless of resume mode)
    profiling_out_dir = os.path.join(args.out_dir, "cell_profiling")
    try:
        cell_metadata_df = _load_cell_metadata_dataframe(profiling_out_dir)
    except Exception as exc:
        logging.warning(f"Failed to load cell metadata dataframe: {exc}")
        cell_metadata_df = None

    # Segmentation Visualization
    if config.get("visualization", {}).get("visualize_segmentation", False):
        logging.info("Starting segmentation visualization...")
        
        # Create visualization directory
        viz_dir = os.path.join(args.out_dir, "visualization", "segmentation")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get patches metadata
        patches_metadata_df = codex_patches.get_patches_metadata()

        # split_mode already extracted earlier in pipeline
        use_segmentation_patches = split_mode == "patches"
        original_seg_results = getattr(codex_patches, "original_seg_res_batch", None)

        timing_totals = {
            "load_patch": 0.0,
            "overlay_repaired": 0.0,
            "overlay_pre_repair": 0.0,
            "segmentation_errors": 0.0,
            "nucleus_mask": 0.0,
            "cell_mask": 0.0,
            "highlight_unmatched_nucleus": 0.0,
            "highlight_unmatched_cell": 0.0,
        }

        informative_mask = patches_metadata_df["is_informative"] == True
        informative_indices = patches_metadata_df[informative_mask].index.tolist()

        # Get extended image for visualization (works in both normal and resume modes)
        extended_image = codex_patches.get_extended_image_for_visualization()
        if extended_image is not None:
            image_height, image_width = extended_image.shape[:2]
        else:
            image_height = int(patches_metadata_df.get("height", 0))
            image_width = int(patches_metadata_df.get("width", 0))

        total_pixels = max(1, int(image_height) * int(image_width))
        is_large_sample = total_pixels > LARGE_SAMPLE_PIXEL_THRESHOLD

        patch_summary_entries: List[Dict] = []
        aggregated_masks: Dict[str, np.ndarray] = {}
        overview_path = None
        target_patch_count = 0
        selected_patch_infos: List[Dict] = []

        reason_labels = {
            "high_density": "High density region",
            "repair_hotspot": "Repair hotspot",
            "low_density": "Sparse region",
            "representative": "Representative area",
            "coverage": "Additional coverage",
            "all": "Full-sample patch",
        }

        def prepare_segmentation_patch_visuals() -> Tuple[List[Dict], int]:
            patch_infos: List[Dict] = []

            if informative_indices:
                for idx, seg_result in enumerate(codex_patches.repaired_seg_res_batch):
                    if seg_result is None:
                        continue
                    if idx >= len(informative_indices):
                        logging.warning(
                            "Segmentation result index %d has no matching metadata entry; skipping visualization.",
                            idx,
                        )
                        continue

                    patch_meta_idx = informative_indices[idx]
                    patch_meta = patches_metadata_df.loc[patch_meta_idx]
                    orig_seg_result = None
                    if original_seg_results and idx < len(original_seg_results):
                        orig_seg_result = original_seg_results[idx]

                    metrics = _compute_patch_metrics(seg_result, orig_seg_result)

                    patch_width = patch_meta.get("patch_width", patch_meta.get("width", 0))
                    patch_height = patch_meta.get("patch_height", patch_meta.get("height", 0))

                    if not patch_width or not patch_height:
                        fallback_mask = seg_result.get("cell_matched_mask", seg_result.get("cell"))
                        if fallback_mask is not None:
                            patch_height, patch_width = fallback_mask.shape[:2]

                    patch_width = int(round(float(patch_width))) if patch_width else 0
                    patch_height = int(round(float(patch_height))) if patch_height else 0
                    patch_area = max(patch_width * patch_height, 1)

                    cell_density = metrics["cell_count"] / patch_area
                    repair_score = metrics["unmatched_cells"] + metrics["unmatched_nuclei"]
                    if metrics.get("matched_fraction") is not None:
                        repair_score += max(0.0, 1.0 - metrics["matched_fraction"]) * max(metrics["cell_count"], 1)

                    patch_infos.append(
                        {
                            "seg_idx": idx,
                            "metadata_index": int(patch_meta_idx),
                            "x_start": int(round(float(patch_meta.get("x_start", 0)))),
                            "y_start": int(round(float(patch_meta.get("y_start", 0)))),
                            "width": patch_width,
                            "height": patch_height,
                            "cell_count": metrics["cell_count"],
                            "nucleus_count": metrics["nucleus_count"],
                            "matched_fraction": metrics["matched_fraction"],
                            "unmatched_cells": metrics["unmatched_cells"],
                            "unmatched_nuclei": metrics["unmatched_nuclei"],
                            "cell_density": cell_density,
                            "repair_score": repair_score,
                        }
                    )
            else:
                logging.warning("No informative patches found for segmentation visualization.")

            target = len(patch_infos)
            selected: List[Dict] = []

            if patch_infos:
                if is_large_sample:
                    approx_ratio = total_pixels / float(PATCH_VIS_SIZE * PATCH_VIS_SIZE * 2)
                    suggested = max(1, math.ceil(approx_ratio))
                    target = max(PATCH_VIS_MIN, min(PATCH_VIS_MAX, suggested))
                    target = min(target, len(patch_infos))
                    selected = _select_representative_patches(
                        patch_infos,
                        target,
                        seed=PATCH_SELECTION_SEED,
                    )
                    logging.info(
                        "Large sample detected (%d pixels). Selected %d patch(es) out of %d for visualization.",
                        total_pixels,
                        len(selected),
                        len(patch_infos),
                    )
                else:
                    selected = [info.copy() for info in patch_infos]
                    for info in selected:
                        info.setdefault("selection_reason", "all")
                    target = len(selected)
            else:
                logging.warning("No segmentation results available for visualization.")

            for display_idx, info in enumerate(selected):
                info["display_index"] = display_idx
                info.setdefault("display_name", f"Patch {display_idx + 1}")
                info["selection_reason_label"] = reason_labels.get(
                    info.get("selection_reason"),
                    "Representative patch",
                )
                info["source_patch_index"] = info.get("seg_idx")

            return selected, target

        if use_segmentation_patches:
            selected_patch_infos, target_patch_count = prepare_segmentation_patch_visuals()

        else:
            # Composite visualization from merged masks (halves/quarters/full image)
            if not informative_indices:
                logging.warning("No informative patches found for segmentation visualization.")

            def merge_mask(result_key: str) -> Optional[np.ndarray]:
                if not result_key:
                    return None
                return _merge_masks_for_visual(
                    patches_metadata_df,
                    informative_indices,
                    codex_patches.repaired_seg_res_batch,
                    original_seg_results,
                    result_key,
                    (image_height, image_width),
                )

            aggregated_masks["cell_matched_mask"] = merge_mask("cell_matched_mask")
            aggregated_masks["nucleus_matched_mask"] = merge_mask("nucleus_matched_mask")
            aggregated_masks["cell"] = merge_mask("cell")
            aggregated_masks["nucleus"] = merge_mask("nucleus")

            if aggregated_masks["cell_matched_mask"] is None or aggregated_masks["nucleus_matched_mask"] is None:
                logging.warning("Unable to merge segmentation masks for composite visualization; falling back to raw patches.")
                use_segmentation_patches = True
                selected_patch_infos = []
            else:
                composite_infos = _generate_composite_patch_infos(
                    image_width,
                    image_height,
                    aggregated_masks,
                    PATCH_VIS_SIZE,
                )

                target_patch_count = len(composite_infos)
                if target_patch_count == 0:
                    logging.warning("No composite patches generated for visualization.")
                if composite_infos:
                    suggested = max(1, math.ceil(total_pixels / float(PATCH_VIS_SIZE * PATCH_VIS_SIZE * 2)))
                    target_patch_count = max(PATCH_VIS_MIN, min(PATCH_VIS_MAX, suggested))
                    target_patch_count = min(target_patch_count, len(composite_infos))
                    selected_patch_infos = _select_representative_patches(
                        composite_infos,
                        target_patch_count,
                        seed=PATCH_SELECTION_SEED,
                    )
                    for display_idx, info in enumerate(selected_patch_infos):
                        info.setdefault("selection_reason", "representative")
                        info["display_index"] = display_idx
                        info.setdefault("display_name", f"ROI {display_idx + 1}")
                        info["selection_reason_label"] = reason_labels.get(
                            info.get("selection_reason"),
                            "Representative area",
                        )
                        info["source_patch_index"] = None
                else:
                    selected_patch_infos = []

                if extended_image is not None and selected_patch_infos:
                    try:
                        overview_channel = (
                            extended_image[:, :, 0]
                            if extended_image.ndim >= 3
                            else extended_image
                        )
                        overview_path = _generate_patch_overview(
                            viz_dir,
                            overview_channel,
                            (image_height, image_width),
                            selected_patch_infos,
                        )
                    except Exception as exc:
                        logging.warning(f"Failed to generate patch overview image: {exc}")
                        overview_path = None

        if use_segmentation_patches and not selected_patch_infos:
            selected_patch_infos, target_patch_count = prepare_segmentation_patch_visuals()

        reason_labels = {
            "high_density": "High density region",
            "repair_hotspot": "Repair hotspot",
            "low_density": "Sparse region",
            "representative": "Representative area",
            "coverage": "Additional coverage",
            "all": "Full-sample patch",
        }

        for info in selected_patch_infos:
            display_idx = info.get("display_index", 0)
            info["display_index"] = display_idx
            info.setdefault("display_name", f"Patch {display_idx + 1}")
            info.setdefault(
                "selection_reason_label",
                reason_labels.get(info.get("selection_reason"), "Representative patch"),
            )

            display_name = info["display_name"]
            selection_label = info.get("selection_reason_label")
            files_record: Dict[str, str] = {}

            # Prepare per-patch data depending on visualization mode
            seg_idx = info.get("seg_idx")
            orig_seg_result = None
            image_patch = None
            matched_cell_mask = None
            matched_nucleus_mask = None
            matched_cell_boundary = None
            matched_nucleus_boundary = None
            original_cell_mask = None
            original_nucleus_mask = None
            original_cell_boundary = None
            original_nucleus_boundary = None
            seg_result_for_errors = None

            if use_segmentation_patches:
                if seg_idx is None:
                    continue
                seg_result = codex_patches.repaired_seg_res_batch[seg_idx]
                if seg_result is None:
                    continue

                if original_seg_results and seg_idx < len(original_seg_results):
                    orig_seg_result = original_seg_results[seg_idx]

                load_start = time.perf_counter()
                if codex_patches.is_using_disk_based_patches():
                    image_patch = codex_patches.load_patch_from_disk(info["metadata_index"], "extracted")
                else:
                    image_patch = codex_patches.valid_patches[seg_idx]
                timing_totals["load_patch"] += time.perf_counter() - load_start

                matched_cell_mask = seg_result.get("cell_matched_mask", seg_result.get("cell"))
                matched_nucleus_mask = seg_result.get("nucleus_matched_mask", seg_result.get("nucleus"))
                matched_cell_boundary = _get_or_compute_labeled_boundary(seg_result, "cell_matched_mask", "cell_matched_boundary")
                matched_nucleus_boundary = _get_or_compute_labeled_boundary(seg_result, "nucleus_matched_mask", "nucleus_matched_boundary")

                if orig_seg_result is not None:
                    original_cell_mask = orig_seg_result.get("cell")
                    original_nucleus_mask = orig_seg_result.get("nucleus")
                    original_cell_boundary = _get_or_compute_labeled_boundary(orig_seg_result, "cell", "cell_boundary")
                    original_nucleus_boundary = _get_or_compute_labeled_boundary(orig_seg_result, "nucleus", "nucleus_boundary")

                seg_result_for_errors = seg_result
            else:
                if extended_image is None:
                    logging.warning("Extended image unavailable for composite visualization; skipping patch %d", display_idx)
                    continue

                x_start = int(info.get("x_start", 0))
                y_start = int(info.get("y_start", 0))
                width = int(info.get("width", PATCH_VIS_SIZE))
                height = int(info.get("height", PATCH_VIS_SIZE))
                x_end = min(image_width, x_start + width)
                y_end = min(image_height, y_start + height)

                image_patch = extended_image[y_start:y_end, x_start:x_end]
                matched_cell_mask = aggregated_masks.get("cell_matched_mask")[y_start:y_end, x_start:x_end]
                matched_nucleus_mask = aggregated_masks.get("nucleus_matched_mask")[y_start:y_end, x_start:x_end]
                matched_cell_boundary = _compute_labeled_boundary(matched_cell_mask)
                matched_nucleus_boundary = _compute_labeled_boundary(matched_nucleus_mask)

                if aggregated_masks.get("cell") is not None:
                    original_cell_mask = aggregated_masks["cell"][y_start:y_end, x_start:x_end]
                    original_cell_boundary = _compute_labeled_boundary(original_cell_mask)
                if aggregated_masks.get("nucleus") is not None:
                    original_nucleus_mask = aggregated_masks["nucleus"][y_start:y_end, x_start:x_end]
                    original_nucleus_boundary = _compute_labeled_boundary(original_nucleus_mask)

                seg_result_for_errors = {
                    "cell_matched_mask": matched_cell_mask,
                    "nucleus_matched_mask": matched_nucleus_mask,
                    "cell": original_cell_mask,
                    "nucleus": original_nucleus_mask,
                }

            if matched_cell_mask is None or matched_nucleus_mask is None or image_patch is None:
                logging.warning("Missing data for visualization patch %d; skipping", display_idx)
                continue

            matched_cell_count = _count_labels(matched_cell_mask)
            matched_nucleus_count = _count_labels(matched_nucleus_mask)
            original_cell_count = _count_labels(original_cell_mask)
            original_nucleus_count = _count_labels(original_nucleus_mask)

            overlay_title = f"Segmentation Overlay — {display_name}"
            if is_large_sample and selection_label:
                overlay_title += f" ({selection_label})"

            base_channel = image_patch[:, :, 0] if image_patch.ndim == 3 else image_patch

            # 1. Repaired segmentation overlay
            try:
                t0 = time.perf_counter()
                fig = create_segmentation_overlay(
                    base_channel,
                    matched_nucleus_mask,
                    matched_cell_mask,
                    show_ids=False,
                    alpha=0.6,
                    reference_nucleus_mask=original_nucleus_mask,
                    reference_cell_mask=original_cell_mask,
                    show_reference_highlights=False,
                    fill_cell_mask=False,
                    fill_nucleus_mask=False,
                    cell_boundary_mask=matched_cell_boundary,
                    nucleus_boundary_mask=matched_nucleus_boundary,
                    custom_title=overlay_title,
                    cell_count=matched_cell_count,
                    nucleus_count=matched_nucleus_count,
                )
                filename = f"vis_segmentation_overlay_patch_{display_idx}.png"
                fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                plt.close(fig)
                files_record["overlay_repaired"] = filename
                timing_totals["overlay_repaired"] += time.perf_counter() - t0
            except Exception as e:
                logging.warning(
                    "Failed to create segmentation overlay for display index %d: %s",
                    display_idx,
                    e,
                )

            if original_nucleus_mask is not None or original_cell_mask is not None:
                try:
                    t0 = time.perf_counter()
                    fig = create_segmentation_overlay(
                        base_channel,
                        original_nucleus_mask,
                        original_cell_mask,
                        show_ids=False,
                        alpha=0.6,
                        fill_cell_mask=False,
                        fill_nucleus_mask=False,
                        cell_boundary_mask=original_cell_boundary,
                        nucleus_boundary_mask=original_nucleus_boundary,
                        custom_title=f"Segmentation Overlay (Pre-Repair) — {display_name}",
                        cell_count=original_cell_count,
                        nucleus_count=original_nucleus_count,
                    )
                    filename = f"vis_segmentation_overlay_pre_repair_patch_{display_idx}.png"
                    fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    files_record["overlay_pre_repair"] = filename
                    timing_totals["overlay_pre_repair"] += time.perf_counter() - t0
                except Exception as e:
                    logging.warning(
                        "Failed to create pre-repair overlay for display index %d: %s",
                        display_idx,
                        e,
                    )

            try:
                t0 = time.perf_counter()
                fig = create_nucleus_mask_visualization(
                    base_channel,
                    matched_nucleus_mask,
                    show_ids=False,
                )
                filename = f"vis_nucleus_mask_patch_{display_idx}.png"
                fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                plt.close(fig)
                files_record["nucleus_mask"] = filename
                timing_totals["nucleus_mask"] += time.perf_counter() - t0
            except Exception as e:
                logging.warning(
                    "Failed to create nucleus mask visualization for display index %d: %s",
                    display_idx,
                    e,
                )

            try:
                t0 = time.perf_counter()
                fig = create_wholecell_mask_visualization(
                    base_channel,
                    matched_cell_mask,
                    show_ids=False,
                )
                filename = f"vis_wholecell_mask_patch_{display_idx}.png"
                fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                plt.close(fig)
                files_record["wholecell_mask"] = filename
                timing_totals["cell_mask"] += time.perf_counter() - t0
            except Exception as e:
                logging.warning(
                    "Failed to create whole-cell mask visualization for display index %d: %s",
                    display_idx,
                    e,
                )

            if original_nucleus_mask is not None:
                try:
                    fig = create_nucleus_mask_visualization(
                        base_channel,
                        original_nucleus_mask,
                        show_ids=False,
                    )
                    filename = f"vis_nucleus_mask_pre_repair_patch_{display_idx}.png"
                    fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    files_record["nucleus_mask_pre_repair"] = filename
                except Exception as e:
                    logging.warning(
                        "Failed to create pre-repair nucleus visualization for display index %d: %s",
                        display_idx,
                        e,
                    )

            if original_cell_mask is not None:
                try:
                    fig = create_wholecell_mask_visualization(
                        base_channel,
                        original_cell_mask,
                        show_ids=False,
                    )
                    filename = f"vis_wholecell_mask_pre_repair_patch_{display_idx}.png"
                    fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    files_record["wholecell_mask_pre_repair"] = filename
                except Exception as e:
                    logging.warning(
                        "Failed to create pre-repair whole-cell visualization for display index %d: %s",
                        display_idx,
                        e,
                    )

            if original_nucleus_mask is not None:
                try:
                    t0 = time.perf_counter()
                    fig = create_segmentation_overlay(
                        base_channel,
                        original_nucleus_mask,
                        matched_cell_mask,
                        show_ids=False,
                        alpha=0.6,
                        reference_nucleus_mask=original_nucleus_mask,
                        reference_cell_mask=None,
                        show_cell_overlay=False,
                        show_nucleus_overlay=False,
                        custom_title=f"Unmatched Nuclei — {display_name}\n\n",
                        show_reference_highlights=True,
                        cell_boundary_mask=matched_cell_boundary,
                        nucleus_boundary_mask=matched_nucleus_boundary,
                    )
                    filename = f"vis_segmentation_overlay_unmatched_nucleus_patch_{display_idx}.png"
                    fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    files_record["overlay_unmatched_nucleus"] = filename
                    timing_totals["highlight_unmatched_nucleus"] += time.perf_counter() - t0
                except Exception as e:
                    logging.warning(
                        "Failed to create unmatched nucleus overlay for display index %d: %s",
                        display_idx,
                        e,
                    )

            if original_cell_mask is not None:
                try:
                    t0 = time.perf_counter()
                    fig = create_segmentation_overlay(
                        base_channel,
                        None,
                        matched_cell_mask,
                        show_ids=False,
                        alpha=0.6,
                        reference_nucleus_mask=None,
                        reference_cell_mask=original_cell_mask,
                        show_cell_overlay=False,
                        show_nucleus_overlay=False,
                        custom_title=f"Unmatched Cells — {display_name}\n\n",
                        show_reference_highlights=True,
                        cell_boundary_mask=matched_cell_boundary,
                        nucleus_boundary_mask=matched_nucleus_boundary,
                    )
                    filename = f"vis_segmentation_overlay_unmatched_cell_patch_{display_idx}.png"
                    fig.savefig(os.path.join(viz_dir, filename), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    files_record["overlay_unmatched_cell"] = filename
                    timing_totals["highlight_unmatched_cell"] += time.perf_counter() - t0
                except Exception as e:
                    logging.warning(
                        "Failed to create unmatched cell overlay for display index %d: %s",
                        display_idx,
                        e,
                    )

            patch_summary_entries.append(
                {
                    "display_index": display_idx,
                    "display_name": display_name,
                    "selection_reason": info.get("selection_reason"),
                    "selection_reason_label": selection_label,
                    "segmentation_index": int(info.get("source_patch_index", -1)) if info.get("source_patch_index") is not None else None,
                    "source_patch_index": info.get("source_patch_index"),
                    "metadata_index": int(info.get("metadata_index", -1)) if info.get("metadata_index") is not None else None,
                    "x_start": int(info.get("x_start", 0)),
                    "y_start": int(info.get("y_start", 0)),
                    "width": int(info.get("width", image_width)),
                    "height": int(info.get("height", image_height)),
                    "cell_count": int(info.get("cell_count", 0)),
                    "nucleus_count": int(info.get("nucleus_count", 0)),
                    "matched_fraction": float(info.get("matched_fraction")) if info.get("matched_fraction") is not None else None,
                    "unmatched_cells": float(info.get("unmatched_cells", 0.0)),
                    "unmatched_nuclei": float(info.get("unmatched_nuclei", 0.0)),
                    "cell_density": float(info.get("cell_density", 0.0)),
                    "files": files_record,
                }
            )

        overview_path = None
        if is_large_sample and extended_image is not None and selected_patch_infos:
            try:
                overview_channel = (
                    extended_image[:, :, 0]
                    if extended_image.ndim >= 3
                    else extended_image
                )
                overview_path = _generate_patch_overview(
                    viz_dir,
                    overview_channel,
                    (image_height, image_width),
                    selected_patch_infos,
                )
            except Exception as exc:
                logging.warning(f"Failed to generate patch overview image: {exc}")
                overview_path = None

        summary_payload = {
            "is_large_sample": bool(is_large_sample),
            "large_sample_threshold": LARGE_SAMPLE_PIXEL_THRESHOLD,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "total_pixels": int(total_pixels),
            "patch_visualization_size": PATCH_VIS_SIZE,
            "target_patch_count": int(target_patch_count),
            "selected_patch_count": len(selected_patch_infos),
            "overview_image": os.path.basename(overview_path) if overview_path else None,
            "patches": patch_summary_entries,
        }

        summary_path = os.path.join(viz_dir, "segmentation_patch_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
        except Exception as exc:
            logging.warning(f"Failed to write segmentation patch summary: {exc}")
        # 5. Create quality heatmaps across all patches
        if len(codex_patches.repaired_seg_res_batch) > 1:
            try:
                t0 = time.perf_counter()
                quality_figs = create_quality_heatmaps(
                    codex_patches.repaired_seg_res_batch,
                    patches_metadata_df[patches_metadata_df["is_informative"] == True],
                    viz_dir
                )
                for fig in quality_figs.values():
                    plt.close(fig)
                logging.info(
                    "Segmentation visualization heatmaps completed in %.2f seconds",
                    time.perf_counter() - t0,
                )
            except Exception as e:
                logging.warning(f"Failed to create quality heatmaps: {e}")
        
        # 6. Plot morphology statistics
        try:
            t0 = time.perf_counter()
            fig = plot_cell_morphology_stats(
                codex_patches.repaired_seg_res_batch,
                viz_dir,
                metadata_df=cell_metadata_df,
            )
            plt.close(fig)
            logging.info(
                "Segmentation morphology stats plotted in %.2f seconds",
                time.perf_counter() - t0,
            )
        except Exception as e:
            logging.warning(f"Failed to plot morphology statistics: {e}")
        
        logging.info(
            "Segmentation visualization timing summary (seconds): %s",
            {k: round(v, 2) for k, v in timing_totals.items()},
        )
        logging.info("Segmentation visualization completed.")
    # ---------------------------------
    # (E) Segmentation Analysis
    # ---------------------------------
    if config["segmentation"]["segmentation_analysis"]:      
        logging.info("Running segmentation analysis...")
        run_segmentation_analysis(codex_patches, config, args)
        logging.info("Segmentation analysis completed.")
    
    # ---------------------------------
    # (F) Generate Analysis Report
    # ---------------------------------
    if config.get("report", {}).get("generate_report", True):
        logging.info("Generating analysis report...")
        try:
            report_path = generate_pipeline_report(args.out_dir, config, codex_patches)
            logging.info(f"Analysis report saved to: {report_path}")
        except Exception as e:
            logging.warning(f"Failed to generate report: {e}")
            # Don't fail the pipeline if report generation fails
    
    codex_patches.finalize_all_channel_cache()
    logging.info("Pipeline run completed.")


# def run_segmentation_analysis(codex_patches: CodexPatches, config: dict, args=None) -> None:
#     """Run segmentation analysis including bias and density analysis.

#     Args:
#         codex_patches: CodexPatches object containing patches and segmentation data
#         config: Configuration Dictionary for evaluation options
#         args: Optional additional arguments
#     """
#     # Extract segmentation data and metadata
#     original_seg_res_batch = codex_patches.original_seg_res_batch
#     repaired_seg_res_batch = codex_patches.repaired_seg_res_batch
#     patches_metadata_df = codex_patches.get_patches_metadata()
#     antibody_df = codex_patches.antibody_df
#     logging.info(f"antibody_df:\n{antibody_df}")
#     antibody_list = antibody_df["antibody_name"].to_list()

#     # Filter for informative patches
#     informative_idx = patches_metadata_df["is_informative"] == True
#     logging.info(f"Number of informative patches: {informative_idx.sum()}")
#     image_ndarray = codex_patches.all_channel_patches[informative_idx]
#     logging.info(f"image_ndarray.shape: {image_ndarray.shape}")

#     output_dir = config.get("output_dir", "./output")

#     # --- Matched fraction ------------------------------------------
#     # This is precalculated after segmentation in segmentation.py
#     matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
#     patches_metadata_df.loc[informative_idx, "matched_fraction"] = matched_fraction_list

#     # Get microns per pixel from config
#     image_mpp = config.get("data", {}).get("image_mpp", 0.5)

#     # List to store all density metrics
#     all_density_metrics = []

#     # Process each patch
#     res_list = []
#     for idx, repaired_seg_res in enumerate(repaired_seg_res_batch):
#         logging.info(f"Processing patch {idx+1}/{len(repaired_seg_res_batch)}")
#         if repaired_seg_res is None:
#             logging.warning(f"Repaired segmentation result for patch {idx} is None.")
#             continue

#         # Visualize results if specified in config
#         patch_output_dir = f"{output_dir}/patch_{idx}"
        
#         # Extract masks from original and repaired segmentation results
#         original_seg_res = original_seg_res_batch[idx]
#         repaired_seg_res = repaired_seg_res_batch[idx]

#         # Get masks from original segmentation results
#         cell_mask = original_seg_res.get("cell")
#         nucleus_mask = original_seg_res.get("nucleus")
#         logging.info(f"cell_mask.shape: {cell_mask.shape}")
#         logging.info(f"nucleus_mask.shape: {nucleus_mask.shape}")

#         # Get masks from repaired segmentation results
#         cell_matched_mask = repaired_seg_res["cell_matched_mask"]
#         nucleus_matched_mask = repaired_seg_res["nucleus_matched_mask"]
#         logging.info(f"cell_matched_mask.shape: {cell_matched_mask.shape}")
#         logging.info(f"nucleus_matched_mask.shape: {nucleus_matched_mask.shape}")

#         # Compute density metrics
#         density_metrics = density_metrics.update_patch_metrics(
#             cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask, image_mpp
#         )
#         all_density_metrics.append(density_metrics)

#         # Analyze repair bias across channels
#         bias_results = bias_analysis.analyze_intensity_bias_across_channels(
#             image_ndarray[idx],
#             cell_mask,
#             nucleus_mask,
#             cell_matched_mask,
#             nucleus_matched_mask,
#         )

#         # Visualize bias analysis results
#         bias_analysis.visualize_channel_bias(
#             bias_results,
#             output_dir=os.path.join(patch_output_dir, "bias_analysis"),
#             channels_per_figure=config["channels_per_figure"],
#         )

#         # Create antibody intensity density plots for this patch
#         density_results = density_analysis.visualize_intensity_distributions(
#             image_ndarray[idx],
#             cell_mask,
#             nucleus_mask,
#             cell_matched_mask,
#             nucleus_matched_mask,
#             output_dir=os.path.join(patch_output_dir, "density_plots"),
#             use_log_scale=True,
#             channel_names=antibody_list,
#         )

#         # Store results
#         evaluation_result = {
#             "patch_idx": idx,
#             "bias_analysis": bias_results,
#             "density_analysis": density_results,
#             "density_metrics": density_metrics,
#         }
#         res_list.append(evaluation_result)

#     # Update metadata with density metrics
#     patches_metadata_df = density_metrics.update_metadata_with_density_metrics(
#         patches_metadata_df, informative_idx, all_density_metrics
#     )
#     codex_patches.seg_evaluation_metrics = res_list
#     codex_patches.set_metadata(patches_metadata_df)
def _get_or_compute_labeled_boundary(container: dict, mask_key: str, boundary_key: str):
    """Return a labeled boundary mask, computing and caching if needed."""
    boundary = container.get(boundary_key)
    if boundary is not None:
        return boundary

    mask = container.get(mask_key)
    if mask is None:
        return None

    bool_boundary = find_boundaries(mask, mode="inner")
    boundary = np.zeros_like(mask, dtype=np.uint32)
    boundary[bool_boundary] = mask[bool_boundary]
    container[boundary_key] = boundary
    return boundary


def _compute_labeled_boundary(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    bool_boundary = find_boundaries(mask, mode="inner")
    boundary = np.zeros_like(mask, dtype=np.uint32)
    boundary[bool_boundary] = mask[bool_boundary]
    return boundary


def _merge_masks_for_visual(
    metadata_df,
    informative_indices,
    repaired_results,
    original_results,
    result_key: str,
    image_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    if not informative_indices:
        return None

    use_original = result_key in {"cell", "nucleus", "cell_boundary", "nucleus_boundary"}
    source_results = original_results if use_original and original_results else repaired_results
    if not source_results:
        return None

    aggregated = np.zeros(image_shape, dtype=np.uint32)
    current_label = 1

    for idx_pos, patch_idx in enumerate(informative_indices):
        if idx_pos >= len(source_results):
            break
        seg_result = source_results[idx_pos]
        if seg_result is None:
            continue
        mask = seg_result.get(result_key)
        if mask is None:
            continue

        mask = np.asarray(mask, dtype=np.uint32)
        if mask.size == 0:
            continue

        # Squeeze out batch dimension if present (shape: (1, H, W) -> (H, W))
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)

        row = metadata_df.loc[patch_idx]
        y_start = int(round(float(row.get("y_start", 0))))
        x_start = int(round(float(row.get("x_start", 0))))
        height = mask.shape[0]
        width = mask.shape[1]
        y_end = min(image_shape[0], y_start + height)
        x_end = min(image_shape[1], x_start + width)
        if y_end <= y_start or x_end <= x_start:
            continue

        patch_mask = mask[: y_end - y_start, : x_end - x_start]
        target_slice = aggregated[y_start:y_end, x_start:x_end]
        non_zero = patch_mask > 0
        if not np.any(non_zero):
            continue

        offset_mask = patch_mask.copy()
        if current_label != 1:
            offset_mask[non_zero] += current_label - 1

        target_slice[non_zero] = offset_mask[non_zero]
        aggregated[y_start:y_end, x_start:x_end] = target_slice

        max_label = int(patch_mask.max())
        if max_label > 0:
            current_label += max_label

    return aggregated


def _generate_composite_patch_infos(
    image_width: int,
    image_height: int,
    aggregated_masks: Dict[str, np.ndarray],
    patch_size: int,
) -> List[Dict]:
    if image_width <= 0 or image_height <= 0:
        return []

    width = int(image_width)
    height = int(image_height)
    size = min(patch_size, max(width, height))
    stride = max(size // 2, 1)

    x_positions = list(range(0, max(width - size, 0) + 1, stride))
    y_positions = list(range(0, max(height - size, 0) + 1, stride))

    final_x = max(width - size, 0)
    if x_positions and x_positions[-1] != final_x:
        x_positions.append(final_x)

    final_y = max(height - size, 0)
    if y_positions and y_positions[-1] != final_y:
        y_positions.append(final_y)

    infos: List[Dict] = []
    idx = 0
    for y_start in y_positions:
        for x_start in x_positions:
            x_end = min(width, x_start + size)
            y_end = min(height, y_start + size)
            if x_end <= x_start or y_end <= y_start:
                continue

            roi_area = (x_end - x_start) * (y_end - y_start)
            if roi_area == 0:
                continue

            cell_matched = aggregated_masks.get("cell_matched_mask")
            nucleus_matched = aggregated_masks.get("nucleus_matched_mask")
            if cell_matched is None or nucleus_matched is None:
                continue

            roi_cell_matched = cell_matched[y_start:y_end, x_start:x_end]
            roi_nucleus_matched = nucleus_matched[y_start:y_end, x_start:x_end]
            cell_count = _count_labels(roi_cell_matched)
            nucleus_count = _count_labels(roi_nucleus_matched)

            roi_cell_orig = aggregated_masks.get("cell")
            roi_nucleus_orig = aggregated_masks.get("nucleus")
            orig_cell_count = _count_labels(roi_cell_orig[y_start:y_end, x_start:x_end]) if roi_cell_orig is not None else cell_count
            orig_nucleus_count = _count_labels(roi_nucleus_orig[y_start:y_end, x_start:x_end]) if roi_nucleus_orig is not None else nucleus_count

            unmatched_cells = max(0, orig_cell_count - cell_count)
            unmatched_nuclei = max(0, orig_nucleus_count - nucleus_count)
            matched_fraction = cell_count / orig_cell_count if orig_cell_count > 0 else None
            cell_density = cell_count / roi_area

            infos.append(
                {
                    "seg_idx": idx,
                    "metadata_index": None,
                    "x_start": x_start,
                    "y_start": y_start,
                    "width": x_end - x_start,
                    "height": y_end - y_start,
                    "cell_count": cell_count,
                    "nucleus_count": nucleus_count,
                    "matched_fraction": matched_fraction,
                    "unmatched_cells": unmatched_cells,
                    "unmatched_nuclei": unmatched_nuclei,
                    "cell_density": cell_density,
                    "selection_reason": "representative",
                }
            )
            idx += 1

    return infos
