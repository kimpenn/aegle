# metadata.py

import logging
import pandas as pd
import numpy as np
import os


def generate_and_save_patch_metadata(all_patches_ndarray, config, args):
    """
    Generate and save metadata for the patches.

    Args:
        all_patches_ndarray (np.ndarray): Array of image patches.
        config (dict): Configuration parameters.
        args (Namespace): Command-line arguments.
    """
    patching_config = config.get("patching", {})
    patch_height = patching_config.get("patch_height", 1440)
    patch_width = patching_config.get("patch_width", 1920)

    # Generate metadata for the patches
    patch_metadata = generate_patch_metadata(
        all_patches_ndarray, patch_height, patch_width
    )

    patch_metadata = qc_patch_metadata(patch_metadata, config, args)
    # Save the metadata
    metadata_file_name = os.path.join(args.out_dir, "patches_metadata.csv")
    save_metadata(patch_metadata, metadata_file_name)
    return patch_metadata


def qc_patch_metadata(patch_metadata, config, args):
    """
    Perform quality control on the patch metadata by marking empty or noisy patches.

    Args:
        patch_metadata (pd.DataFrame): DataFrame containing patch metadata.
        config (dict): Configuration parameters.
        args (Namespace): Command-line arguments.

    Returns:
        pd.DataFrame: DataFrame containing QC'd patch metadata.
    """
    logging.info("Performing quality control on patch metadata...")

    # Get QC parameters from config
    qc_config = config.get("qc", {})
    non_zero_perc_threshold = qc_config.get("non_zero_perc_threshold", 0.05)
    mean_intensity_threshold = qc_config.get("mean_intensity_threshold", 1)
    std_intensity_threshold = qc_config.get("std_intensity_threshold", 1)

    # Identify patches that are empty or noisy
    patch_metadata["is_empty"] = (
        patch_metadata["nuclear_non_zero_perc"] < non_zero_perc_threshold
    )

    patch_metadata["is_noisy"] = (
        patch_metadata["nuclear_mean"] < mean_intensity_threshold
    ) & (patch_metadata["nuclear_std"] < std_intensity_threshold)

    # Mark patches as bad (either empty or noisy)
    patch_metadata["is_bad_patch"] = (
        patch_metadata["is_empty"] | patch_metadata["is_noisy"]
    )

    num_bad_patches = patch_metadata["is_bad_patch"].sum()
    total_patches = len(patch_metadata)

    logging.info(
        f"Identified {num_bad_patches} bad patches out of {total_patches} total patches."
    )

    return patch_metadata


def generate_patch_metadata(all_patches_ndarray, patch_height, patch_width):
    """
    Generate metadata for the patches.

    Args:
        all_patches_ndarray (np.ndarray): Array of image patches.
        patch_height (int): Height of patches.
        patch_width (int): Width of patches.

    Returns:
        pd.DataFrame: DataFrame containing metadata for each patch.
    """
    logging.info("Generating metadata for patches...")
    num_patches = all_patches_ndarray.shape[0]
    patch_metadata = pd.DataFrame(
        {
            "patch_id": range(num_patches),
            "height": patch_height,
            "width": patch_width,
            "nuclear_mean": [np.mean(patch[:, :, 0]) for patch in all_patches_ndarray],
            "nuclear_std": [np.std(patch[:, :, 0]) for patch in all_patches_ndarray],
            "nuclear_non_zero_perc": [
                np.count_nonzero(patch[:, :, 0]) / (patch_height * patch_width)
                for patch in all_patches_ndarray
            ],
            "wholecell_mean": [
                np.mean(patch[:, :, 1]) for patch in all_patches_ndarray
            ],
            "wholecell_std": [np.std(patch[:, :, 1]) for patch in all_patches_ndarray],
            "wholecell_non_zero_perc": [
                np.count_nonzero(patch[:, :, 1]) / (patch_height * patch_width)
                for patch in all_patches_ndarray
            ],
        }
    )
    return patch_metadata


def save_metadata(patch_metadata, metadata_file_name):
    """
    Save the patch metadata to a CSV file.

    Args:
        patch_metadata (pd.DataFrame): DataFrame containing patch metadata.
        metadata_file_name (str): Path to save the metadata CSV.
    """
    patch_metadata.to_csv(metadata_file_name, index=False)
    logging.info(f"Saved metadata to {metadata_file_name}")
