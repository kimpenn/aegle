# aegle/pipeline.py

import os
import logging
import numpy as np

from aegle.data_loading import load_data
from aegle.preprocessing import preprocess_image
from aegle.patching import generate_image_patches
from aegle.visualization import save_patches_rgb
from aegle.metadata import generate_and_save_patch_metadata
from aegle.segment import run_cell_segmentation


def run_pipeline(config, args):
    """
    Run the CODEX image analysis pipeline.

    Args:
        config (dict): Configuration parameters loaded from YAML file.
        args (Namespace): Command-line arguments parsed by argparse.
    """

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    image_ndarray, antibody_df, target_channels_dict = load_data(config, args)

    # Preprocess images (e.g., selecting channels)
    image_ndarray_target_channel = preprocess_image(
        image_ndarray, antibody_df, target_channels_dict
    )

    # Patch the image and save results
    all_patches_ndarray = generate_image_patches(
        image_ndarray_target_channel, config, args
    )

    # Generate and save metadata for the patches
    patches_metadata_df = generate_and_save_patch_metadata(
        all_patches_ndarray, config, args
    )

    # Optional: Visualize patches
    if config.get("visualization", {}).get("visualize_patches", False):
        save_patches_rgb(all_patches_ndarray, patches_metadata_df, config, args)

    # Cell Segmentation and Post-Segmentation Repairs
    run_cell_segmentation(all_patches_ndarray, patches_metadata_df, config, args)
    # ---------------------------------
    # Future Steps (Placeholders)
    # ---------------------------------
    # Segmentation Evaluation
    # Cell Profiling
    # TODO: Implement these steps as needed
