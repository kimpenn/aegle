import os
import logging
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def save_patches_rgb(all_patches_ndarray, config, args, max_workers=10):
    output_dir = os.path.join(args.output_dir, "patches_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Visualizing and saving patches to {output_dir}...")

    # Use ThreadPoolExecutor to save patches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for patch_id, patch in enumerate(all_patches_ndarray):
            executor.submit(save_patch_rgb, patch, patch_id, output_dir)


def save_patch_rgb(patch, patch_id, patch_vis_dir):
    """
    Save a patch as an image using PIL.

    Args:
        patch (np.ndarray): Image patch.
        patch_id (int): ID of the patch.
        patch_vis_dir (str): Directory to save visualizations.
    """
    # Normalize the patch to be between 0 and 255
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
        composite[:, :, 0] = (
            np.uint8(patch[:, :, 0] / np.max(patch[:, :, 0]) * 255)
            if np.max(patch[:, :, 0]) != 0
            else 0
        )
        composite[:, :, 1] = (
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
