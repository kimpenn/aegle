import os
import numpy as np
import logging
import pickle
import shutil

from aegle.codex_image import CodexImage
from aegle.codex_patches import CodexPatches

from aegle.visualization import save_patches_rgb, save_image_rgb

from aegle.segment import run_cell_segmentation, visualize_cell_segmentation
from aegle.evaluation import run_seg_evaluation
from aegle.cell_profiling import run_cell_profiling
from aegle.quick_evaluation import run_quick_evaluation


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

    # ---------------------------------
    # (A) Full Image Preprocessing
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

    # Optional: Visualize whole sample image
    if config.get("visualization", {}).get("visualize_whole_sample", False):
        save_image_rgb(
            codex_image.extended_extracted_channel_image,
            "extended_extracted_channel_image.png",
            args,
        )
        save_image_rgb(
            codex_image.extracted_channel_image, "extracted_channel_image.png", args
        )

    # Step 2: Initialize CodexPatches object and generate patches
    logging.info("----- Initializing CodexPatches object and generating patches.")
    codex_patches = CodexPatches(codex_image, config, args)
    codex_patches.save_patches(config["visualization"]["save_all_channel_patches"])
    codex_patches.save_metadata()
    logging.info("Patches generated and metadata saved successfully.")

    # Optional: Add disruptions to patches for testing
    # Extract distruption type and level from config
    disruption_config = config.get("testing", {}).get("data_disruption", {})
    logging.info(f"Disruption config: {disruption_config}")
    if disruption_config and disruption_config.get("type", None) is not None:
        disruption_type = disruption_config.get("type", None)
        disruption_level = disruption_config.get("level", 1)
        logging.info(
            f"Adding disruptions {disruption_type} at level {disruption_level} to patches for testing."
        )
        codex_patches.add_disruptions(disruption_type, disruption_level)
        logging.info("Disruptions added to patches.")
        if disruption_config.get("save_disrupted_patches", False):
            logging.info("Saving disrupted patches.")
            codex_patches.save_disrupted_patches()
            logging.info("Disrupted patches saved.")
        if config.get("visualization", {}).get("visualize_patches", False):
            # This will overwrite the visualizations of the original patches with the disrupted patches
            save_patches_rgb(
                codex_patches.disrupted_extracted_channel_patches,
                codex_patches.patches_metadata,
                config,
                args,
                max_workers=config.get("visualization", {}).get("workers", None),
            )

    # Optional: Visualize patches
    if config.get("visualization", {}).get("visualize_patches", False):
        logging.info("Visualizing patches.")
        save_patches_rgb(
            codex_patches.extracted_channel_patches,
            codex_patches.patches_metadata,
            config,
            args,
            max_workers=config.get("visualization", {}).get("workers", None),
        )
        logging.info("Patch visualization completed.")

    # ---------------------------------
    # (C) Cell Segmentation and auto evaluation
    # ---------------------------------
    logging.info("Running cell segmentation.")
    run_cell_segmentation(codex_patches, config, args)

    if config["evaluation"]["compute_metrics"]:
        # TODO: if the number of cells are too large we should skip the evaluation
        run_seg_evaluation(codex_patches, config, args)

    # if config["segmentation"].get("save_segmentation_pickle", False):
    #     file_name = "codex_patches.pkl"
    #     file_name = os.path.join(args.out_dir, file_name)
    #     logging.info(f"Saving CodexPatches object to {file_name}")
    #     with open(file_name, "wb") as f:
    #         pickle.dump(codex_patches, f)

    if config.get("visualization", {}).get("visualize_segmentation", False):

        visualize_cell_segmentation(
            codex_patches.valid_patches,
            codex_patches.repaired_seg_res_batch,
            config,
            args,
        )
        logging.info("Segmentation visualization completed.")

        visualize_cell_segmentation(
            codex_patches.valid_patches,
            codex_patches.original_seg_res_batch,
            config,
            args,
        )
        logging.info("Segmentation visualization completed.")

    # ---------------------------------
    # (D) Cell Profiling
    # ---------------------------------
    logging.info("Running cell profiling.")
    run_cell_profiling(codex_patches, config, args)
    logging.info("Cell profiling completed.")

    logging.info("Pipeline run completed.")

    # ---------------------------------
    # (E) Quick Evaluation of Segmentation and Repaired segmentation
    # ---------------------------------
    logging.info("Running quick evaluation of segmentation and repaired segmentation.")
    run_quick_evaluation(codex_patches, config, args)
    logging.info("Quick evaluation completed.")
