import numpy as np
import matplotlib.pyplot as plt
from deepcell.utils.plot_utils import create_rgb_image

import os

import logging


def extend_image(image, patch_height, patch_width, step_height, step_width):
    # Get the current image dimensions
    img_height, img_width = image.shape[1], image.shape[2]
    logging.info(f"Original image dimensions: {img_height}x{img_width}")

    # Calculate the number of patches in both dimensions
    extra_height = (img_height - patch_height) % step_height
    extra_width = (img_width - patch_width) % step_width

    # Log extra dimensions
    logging.info(f"Extra height after patches: {extra_height}")
    logging.info(f"Extra width after patches: {extra_width}")

    # Extend the image if necessary by adding padding
    if extra_height != 0:
        pad_height = step_height - extra_height
        logging.info(f"Padding height required: {pad_height}")
    else:
        pad_height = 0
        logging.info("No padding required for height")

    if extra_width != 0:
        pad_width = step_width - extra_width
        logging.info(f"Padding width required: {pad_width}")
    else:
        pad_width = 0
        logging.info("No padding required for width")

    # Add zero-padding (assuming the image has 4 dimensions: [batch, height, width, channels])
    padded_image = np.pad(
        image,
        ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Log the new image dimensions after padding
    new_height, new_width = padded_image.shape[1], padded_image.shape[2]
    logging.info(f"New image dimensions after padding: {new_height}x{new_width}")

    return padded_image


def crop_image_into_patches(image, patch_height, patch_width, step_height, step_width):
    patches = []
    img_height, img_width = image.shape[1], image.shape[2]

    for y in range(0, img_height - patch_height + 1, step_height):
        for x in range(0, img_width - patch_width + 1, step_width):
            # Crop the patch
            patch = image[:, y : y + patch_height, x : x + patch_width, :]
            patches.append(patch)
    cat_patches = np.concatenate(patches, axis=0)
    return cat_patches


# Function to save the raw ndarray patches for downstream segmentation
def save_patches(patches, save_dir):
    # Concatenate patches along the first dimension
    all_patches = np.concatenate(patches, axis=0)
    # Save the concatenated patches as a .npy file
    patch_file_name = f"{save_dir}/all_patches.npy"
    np.save(patch_file_name, all_patches)
    print(f"Saved all patches concatenated as {patch_file_name}")


# Function to visualize a patch using RGB overlay
def visualize_patch(patch, idx):
    print(f"Patch shape: {patch.shape}")
    if patch.shape[0] != 1:
        raise ValueError("The input patch should have a shape of (1, H, W, C)")
    rgb_patch = create_rgb_image(patch, channel_colors=["green", "blue"])
    print(f"RGB Patch shape: {rgb_patch.shape}")
    # Visualize the RGB patch
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_patch[0])  # Show the first element as it's shape (1, H, W, 3)
    plt.axis("off")
    plt.title(f"Patch {idx} in RGB")
    plt.show()


# Function to visualize a patch using RGB overlay and save it to local disk
def visualize_and_save_patch(patch, idx, output_dir="patch_visualizations"):
    # print(f"Patch shape: {patch.shape}")
    if patch.shape[0] != 1:
        raise ValueError("The input patch should have a shape of (1, H, W, C)")

    # Create RGB image
    rgb_patch = create_rgb_image(patch, channel_colors=["green", "blue"])
    # print(f"RGB Patch shape: {rgb_patch.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the RGB patch as an image
    output_path = os.path.join(output_dir, f"patch_{idx}.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_patch[0])  # Show the first element as it's shape (1, H, W, 3)
    plt.axis("off")
    plt.title(f"Patch {idx} in RGB")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved patch visualization at: {output_path}")
