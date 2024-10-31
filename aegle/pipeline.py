import os
import numpy as np
import logging

from aegle.codex_image import CodexImage
from aegle.codex_patches import CodexPatches

from aegle.visualization import save_patches_rgb, save_image_rgb

# from aegle.segment import run_cell_segmentation
# from aegle.seg_eval import run_segmentation_evaluation


def run_pipeline(config, args):
    """
    Run the CODEX image analysis pipeline.

    Args:
        config (dict): Configuration parameters loaded from YAML file.
        args (Namespace): Command-line arguments parsed by argparse.
    """
    logging.info("----- Running pipeline with provided configuration and arguments.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {args.output_dir}")

    # Initialize CodexImage object
    logging.info("----- Initializing CodexImage object.")
    codex_image = CodexImage(config, args)
    logging.info("CodexImage object initialized successfully.")

    # Extract target channels from the image
    logging.info("----- Extracting target channels from the image.")
    codex_image.extract_target_channels()
    logging.info("Target channels extracted successfully.")

    # Extend the image for full patch coverage
    logging.info("----- Extending image for full patch coverage.")
    codex_image.extend_image()
    logging.info("Image extension completed successfully.")

    # # Initialize CodexPatches object and generate patches
    logging.info("----- Initializing CodexPatches object and generating patches.")
    codex_patches = CodexPatches(codex_image, config, args)
    codex_patches.save_patches()
    codex_patches.save_metadata()
    logging.info("Patches generated and metadata saved successfully.")

    # Optional: Visualize patches
    if config.get("visualization", {}).get("visualize_patches", False):
        save_image_rgb(
            codex_image.extended_extracted_channel_image,
            "extended_extracted_channel_image.png",
            args,
        )
        save_image_rgb(
            codex_image.extracted_channel_image, "extracted_channel_image.png", args
        )

        # logging.info("Visualizing patches.")
        # save_patches_rgb(
        #     codex_patches.extracted_channel_patches,
        #     codex_patches.patches_metadata,
        #     config,
        #     args,
        # )
        # logging.info("Patch visualization completed.")

    # # Cell Segmentation and Post-Segmentation Repairs
    # logging.info("Running cell segmentation.")
    # matched_seg_res, fraction_matched, seg_res = run_cell_segmentation(
    #     all_patches_ndarray, patches_metadata_df, config, args
    # )
    # logging.info("Cell segmentation completed.")

    # # Post-Segmentation Filtering
    # logging.info("Running post-segmentation filtering.")
    # post_segmentation_filter(matched_seg_res)
    # logging.info("Post-segmentation filtering completed.")

    # # Segmentation Evaluation
    # logging.info("Running segmentation evaluation.")
    # run_segmentation_evaluation(all_patches_ndarray, matched_seg_res, config, args)
    # logging.info("Segmentation evaluation completed.")

    # # TODO: Stitch Image Generation (Future Steps)
    # # Cell Profiling
    # logging.info("Running cell profiling.")
    # run_cell_profiling(seg_res, config, args)
    # logging.info("Cell profiling completed.")

    # ---------------------------------
    # Future Steps (Placeholders)
    # ---------------------------------
    logging.info("Pipeline run completed.")
