import os
import logging
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from skimage import exposure


def save_patches_rgb(
    all_patches_ndarray, patches_metadata_df, config, args, max_workers=10
):
    """
    Save patches as RGB images into 'good_patches' and 'bad_patches' directories.

    Args:
        all_patches_ndarray (np.ndarray): Array of image patches.
        patches_metadata_df (pd.DataFrame): DataFrame with patch metadata and QC info.
        config (dict): Configuration parameters.
        args (Namespace): Command-line arguments.
        max_workers (int): Maximum number of threads for parallel processing.
    """
    good_patches_dir = os.path.join(
        args.output_dir, "patches_visualizations", "good_patches"
    )
    bad_patches_dir = os.path.join(
        args.output_dir, "patches_visualizations", "bad_patches"
    )

    # Create directories if they don't exist
    os.makedirs(good_patches_dir, exist_ok=True)
    os.makedirs(bad_patches_dir, exist_ok=True)

    logging.info(
        f"Visualizing and saving patches to {good_patches_dir} and {bad_patches_dir}..."
    )
    logging.info(f"Total patches to save: {len(all_patches_ndarray)}")
    # Use ThreadPoolExecutor to save patches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for patch_id, patch in enumerate(all_patches_ndarray):
            # Check if the patch is bad or good
            is_bad_patch = patches_metadata_df.loc[patch_id, "is_bad_patch"]

            # Choose output directory based on QC result
            patch_vis_dir = bad_patches_dir if is_bad_patch else good_patches_dir

            # Submit the task to save the patch
            executor.submit(save_patch_rgb, patch, patch_id, patch_vis_dir)


def save_patch_rgb(patch, patch_id, patch_vis_dir):
    """
    Save a patch as an image using PIL.

    Args:
        patch (np.ndarray): Image patch.
        patch_id (int): ID of the patch.
        patch_vis_dir (str): Directory to save visualizations.
    """
    # Normalize the patch to be between 0 and 255
    logging.info(
        f"Saving patch {patch_id}: shape {patch.shape}; patch_vis_dir: {patch_vis_dir}"
    )
    if patch.shape[-1] == 1:
        # Single-channel grayscale
        img = Image.fromarray(
            (
                np.uint8(patch[:, :, 0] / np.max(patch) * 255)
                if np.max(patch) != 0
                else np.zeros_like(patch[:, :, 0])
            ),
            "L",
        )
    elif patch.shape[-1] == 2:
        # Two channels, create a composite RGB image
        composite = np.zeros((patch.shape[0], patch.shape[1], 3), dtype=np.uint8)
        # Map first channel to Green
        composite[:, :, 1] = (
            np.uint8(patch[:, :, 0] / np.max(patch[:, :, 0]) * 255)
            if np.max(patch[:, :, 0]) != 0
            else 0
        )
        # Map second channel to Blue
        composite[:, :, 2] = (
            np.uint8(patch[:, :, 1] / np.max(patch[:, :, 1]) * 255)
            if np.max(patch[:, :, 1]) != 0
            else 0
        )
        img = Image.fromarray(composite)
    else:
        # For RGB or multi-channel images
        img = Image.fromarray(
            np.uint8(patch / np.max(patch) * 255)
            if np.max(patch) != 0
            else np.zeros_like(patch, dtype=np.uint8)
        )

    # Save the image to disk
    image_file = os.path.join(patch_vis_dir, f"patch_{patch_id}.png")
    img.save(image_file)
    logging.info(f"Saved visualization for patch {patch_id} at {image_file}")
    # do I need to close the image?
    # No, PIL handles the closing of the image file automatically when saved.


def save_image_rgb(image, filename, args):
    """
    Save an RGB image to the specified output directory.

    Args:
        image (np.ndarray): The image to save.
        args (Namespace): Command-line arguments containing output directory.
    """
    image_file = os.path.join(args.output_dir, filename)
    logging.info(f"Saving image to {image_file}")
    # Two channels, create a composite RGB image
    composite = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # Map first channel to Green
    composite[:, :, 1] = (
        np.uint8(image[:, :, 0] / np.max(image[:, :, 0]) * 255)
        if np.max(image[:, :, 0]) != 0
        else 0
    )
    # Map second channel to Blue
    composite[:, :, 2] = (
        np.uint8(image[:, :, 1] / np.max(image[:, :, 1]) * 255)
        if np.max(image[:, :, 1]) != 0
        else 0
    )
    # Enhance contrast across the entire RGB image using adaptive histogram equalization
    composite = exposure.equalize_adapthist(composite, clip_limit=0.03)

    # Convert back to [0, 255] range and uint8 format
    composite = (composite * 255).astype(np.uint8)
    # print(composite)
    img = Image.fromarray(composite)
    img.save(image_file)
    logging.info(f"=== Image shape: {image.shape}")
    logging.info(f"Saved image visualization at {image_file}")
