# patching.py

import logging
import numpy as np
import os


def generate_image_patches(image_ndarray_target_channel, config, args):
    """
    Patch the image and save as NumPy array.

    Args:
        image_ndarray_target_channel (np.ndarray): Image with selected channels.
        config (dict): Configuration parameters.
        args (Namespace): Command-line arguments.

    Returns:
        all_patches_ndarray (np.ndarray): Array of image patches.
    """
    patching_config = config.get("patching", {})
    patch_height = patching_config.get("patch_height", 1440)
    patch_width = patching_config.get("patch_width", 1920)
    overlap = patching_config.get("overlap", 0.1)

    # Define patch size and overlap
    overlap_height = int(patch_height * overlap)
    overlap_width = int(patch_width * overlap)

    # Calculate step size for cropping
    step_height = patch_height - overlap_height
    step_width = patch_width - overlap_width

    # Extract image dimensions
    _, img_height, img_width, _ = image_ndarray_target_channel.shape
    logging.info(f"Image dimensions: {img_height}x{img_width}")
    logging.info(f"Patch dimensions: {patch_height}x{patch_width}")
    logging.info(f"Overlap: {overlap_height}x{overlap_width}")
    logging.info(f"Step size: {step_height}x{step_width}")

    # Extend the image to ensure full coverage
    extended_image = extend_image(
        image_ndarray_target_channel, patch_height, patch_width, step_height, step_width
    )

    # Crop the extended image into patches
    all_patches_ndarray = crop_image_into_patches(
        extended_image, patch_height, patch_width, step_height, step_width
    )

    # Save patches as a NumPy file
    patches_file_name = os.path.join(
        args.output_dir,
        f"patches_ndarray.npy",
    )
    np.save(patches_file_name, all_patches_ndarray)
    logging.info(f"Saved patches to {patches_file_name}")

    return all_patches_ndarray


def extend_image(image, patch_height, patch_width, step_height, step_width):
    """
    Extend the image to ensure full coverage when cropping patches.

    Args:
        image (np.ndarray): The image to extend.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        step_height (int): Step size in height.
        step_width (int): Step size in width.

    Returns:
        np.ndarray: Extended image.
    """
    _, img_height, img_width, _ = image.shape
    pad_height = (
        patch_height - (img_height - patch_height) % step_height
    ) % patch_height
    pad_width = (patch_width - (img_width - patch_width) % step_width) % patch_width

    extended_image = np.pad(
        image,
        ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    logging.info(f"Extended image shape: {extended_image.shape}")
    return extended_image


def crop_image_into_patches(image, patch_height, patch_width, step_height, step_width):
    """
    Crop the extended image into patches.

    Args:
        image (np.ndarray): Extended image.
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        step_height (int): Step size in height.
        step_width (int): Step size in width.

    Returns:
        np.ndarray: Array of image patches.
    """
    patches = []
    _, img_height, img_width, _ = image.shape
    for y in range(0, img_height - patch_height + 1, step_height):
        for x in range(0, img_width - patch_width + 1, step_width):
            patch = image[:, y : y + patch_height, x : x + patch_width, :]
            patches.append(patch)
    all_patches_ndarray = np.concatenate(patches, axis=0)
    return all_patches_ndarray
