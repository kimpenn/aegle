import os
import logging
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from skimage import exposure
from skimage.segmentation import find_boundaries


def save_patches_rgb(
    all_patches_ndarray, patches_metadata_df, config, args, max_workers=1
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
        args.out_dir, "patches_visualizations", "good_patches"
    )
    bad_patches_dir = os.path.join(
        args.out_dir, "patches_visualizations", "bad_patches"
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


def create_rgb(image_data):
    """
    Create a PIL Image from a patch.

    Args:
        image_data (np.ndarray)

    Returns:
        Image: PIL Image object.
    """
    if image_data.shape[-1] == 1:
        # Single-channel grayscale
        img_array = (
            np.uint8(image_data[:, :, 0] / np.max(image_data) * 255)
            if np.max(image_data) != 0
            else np.zeros_like(image_data[:, :, 0], dtype=np.uint8)
        )
        img = Image.fromarray(img_array, mode="L")
    elif image_data.shape[-1] == 2:
        # Two channels, create a composite RGB image
        composite = np.zeros(
            (image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8
        )
        # Map first channel to Green
        composite[:, :, 1] = (
            np.uint8(image_data[:, :, 0] / np.max(image_data[:, :, 0]) * 255)
            if np.max(image_data[:, :, 0]) != 0
            else 0
        )
        # Map second channel to Blue
        composite[:, :, 2] = (
            np.uint8(image_data[:, :, 1] / np.max(image_data[:, :, 1]) * 255)
            if np.max(image_data[:, :, 1]) != 0
            else 0
        )
        img = Image.fromarray(composite, mode="RGB")
    else:
        # For RGB or multi-channel images
        img_array = (
            np.uint8(image_data / np.max(image_data) * 255)
            if np.max(image_data) != 0
            else np.zeros_like(image_data, dtype=np.uint8)
        )
        img = Image.fromarray(img_array)
    return img


def save_patch_rgb(patch, patch_id, patch_vis_dir):
    """
    Save a patch as an image using PIL.

    Args:
        patch (np.ndarray): Image patch.
        patch_id (int): ID of the patch.
        patch_vis_dir (str): Directory to save visualizations.
    """
    img = create_rgb(patch)

    # Save the image to disk
    image_file = os.path.join(patch_vis_dir, f"patch_{patch_id}.png")
    img.save(image_file)
    logging.info(f"Saved visualization for patch {patch_id} at {image_file}")


def save_image_rgb(image, filename, args):
    """
    Save an RGB image to the specified output directory.

    Args:
        image (np.ndarray): The image to save.
        filename (str): The filename for the saved image.
        args (Namespace): Command-line arguments containing output directory.
    """
    image_file = os.path.join(args.out_dir, filename)
    logging.info(f"Saving image to {image_file}")

    # Use create_rgb to get the initial image
    img = create_rgb(image)

    # Convert PIL Image to numpy array for processing
    img_array = np.array(img)

    # Enhance contrast using adaptive histogram equalization
    # Note: exposure.equalize_adapthist expects images in [0, 1] range
    img_array_float = img_array / 255.0  # Normalize to [0, 1]
    img_array_eq = exposure.equalize_adapthist(img_array_float, clip_limit=0.03)

    # Convert back to [0, 255] range and uint8 format
    img_array_eq = (img_array_eq * 255).astype(np.uint8)

    # Convert back to PIL Image
    img_eq = Image.fromarray(img_array_eq)

    # Save the processed image
    img_eq.save(image_file)
    logging.info(f"=== Image shape: {image.shape}")
    logging.info(f"Saved image visualization at {image_file}")


def make_outline_overlay(rgb_data, predictions):
    """Overlay a segmentation mask with image data for easy visualization

    Args:
        rgb_data: 3 channel array of images, output of ``create_rgb``
        predictions: segmentation predictions to be visualized

    Returns:
        numpy.array: overlay image of input data and predictions

    Raises:
        ValueError: If predictions are not 4D
        ValueError: If there is not matching RGB data for each prediction
    """
    if len(predictions.shape) != 4:
        raise ValueError(f"Predictions must be 4D, got {predictions.shape}")

    if predictions.shape[0] > rgb_data.shape[0]:
        raise ValueError("Must supply an rgb image for each prediction")

    boundaries = np.zeros_like(rgb_data)
    overlay_data = np.copy(rgb_data)

    for img in range(predictions.shape[0]):
        boundary = find_boundaries(
            predictions[img, ..., 0], connectivity=1, mode="inner"
        )
        boundaries[img, boundary > 0, :] = 1

    overlay_data[boundaries > 0] = 1

    return overlay_data
