# codex_patches.py

import os
import logging
import numpy as np
import pandas as pd
import cv2
from skimage.io import imsave


class CodexPatches:
    def __init__(self, codex_image, config, args):
        """
        Initialize the CodexPatches object and generate patches from CodexImage.

        Args:
            codex_image (CodexImage): CodexImage object with preprocessed image data.
            config (dict): Configuration parameters.
            args (Namespace): Command-line arguments.
        """
        self.codex_image = codex_image
        self.config = config
        self.args = args
        self.all_channel_patches = None
        self.extracted_channel_patches = None
        self.noisy_extracted_channel_patches = None
        self.patches_metadata = None

        self.generate_patches()
        self.init_patch_metadata()
        self.qc_patch_metadata()

    def generate_patches(self):
        """
        Generate patches from the preprocessed image and store them in an ndarray.
        """
        patching_config = self.config.get("patching", {})
        patch_height = patching_config.get("patch_height", 1440)
        patch_width = patching_config.get("patch_width", 1920)
        overlap = patching_config.get("overlap", 0.1)

        # Calculate overlap and step sizes
        overlap_height = int(patch_height * overlap)
        overlap_width = int(patch_width * overlap)
        step_height = patch_height - overlap_height
        step_width = patch_width - overlap_width

        self.all_channel_patches = self.crop_image_into_patches(
            self.codex_image.extended_all_channel_image,
            patch_height,
            patch_width,
            step_height,
            step_width,
        )

        self.extracted_channel_patches = self.crop_image_into_patches(
            self.codex_image.extended_extracted_channel_image,
            patch_height,
            patch_width,
            step_height,
            step_width,
        )

    def crop_image_into_patches(
        self, image, patch_height, patch_width, step_height, step_width
    ):
        """
        Crop the extended image into patches.
        """
        patches = []
        img_height, img_width, _ = image.shape
        for y in range(0, img_height - patch_height + 1, step_height):
            for x in range(0, img_width - patch_width + 1, step_width):
                patch = image[y : y + patch_height, x : x + patch_width, :]
                patches.append(patch)
        return np.array(patches)

    def init_patch_metadata(self):
        """
        Generate metadata for the patches and store it in self.patches_metadata.
        """
        patching_config = self.config.get("patching", {})
        patch_height = patching_config.get("patch_height", 1440)
        patch_width = patching_config.get("patch_width", 1920)

        logging.info("Generating metadata for patches...")
        num_patches = self.extracted_channel_patches.shape[0]
        logging.info(
            f"=== size of extracted_channel_patches: {self.extracted_channel_patches.shape}"
        )
        self.patches_metadata = pd.DataFrame(
            {
                "patch_id": range(num_patches),
                "height": patch_height,
                "width": patch_width,
                "nuclear_mean": [
                    np.mean(patch[:, :, 0]) for patch in self.extracted_channel_patches
                ],
                "nuclear_std": [
                    np.std(patch[:, :, 0]) for patch in self.extracted_channel_patches
                ],
                "nuclear_non_zero_perc": [
                    np.count_nonzero(patch[:, :, 0]) / (patch_height * patch_width)
                    for patch in self.extracted_channel_patches
                ],
                "wholecell_mean": [
                    np.mean(patch[:, :, 1]) for patch in self.extracted_channel_patches
                ],
                "wholecell_std": [
                    np.std(patch[:, :, 1]) for patch in self.extracted_channel_patches
                ],
                "wholecell_non_zero_perc": [
                    np.count_nonzero(patch[:, :, 1]) / (patch_height * patch_width)
                    for patch in self.extracted_channel_patches
                ],
            }
        )

    def qc_patch_metadata(self):
        """
        Perform quality control on the patch metadata by marking empty or noisy patches.

        Returns:
            pd.DataFrame: DataFrame containing QC'd patch metadata.
        """
        logging.info("Performing quality control on patch metadata...")

        # Get QC parameters from config
        qc_config = self.config.get("qc", {})
        non_zero_perc_threshold = qc_config.get("non_zero_perc_threshold", 0.05)
        mean_intensity_threshold = qc_config.get("mean_intensity_threshold", 1)
        std_intensity_threshold = qc_config.get("std_intensity_threshold", 1)

        # Identify patches that are empty or noisy
        self.patches_metadata["is_empty"] = (
            self.patches_metadata["nuclear_non_zero_perc"] < non_zero_perc_threshold
        )

        self.patches_metadata["is_noisy"] = (
            self.patches_metadata["nuclear_mean"] < mean_intensity_threshold
        ) & (self.patches_metadata["nuclear_std"] < std_intensity_threshold)

        # Mark patches as bad (either empty or noisy)
        self.patches_metadata["is_bad_patch"] = (
            self.patches_metadata["is_empty"] | self.patches_metadata["is_noisy"]
        )
        self.patches_metadata["is_infomative"] = ~self.patches_metadata["is_bad_patch"]
        num_bad_patches = self.patches_metadata["is_bad_patch"].sum()
        total_patches = len(self.patches_metadata)

        logging.info(
            f"Identified {num_bad_patches} bad patches out of {total_patches} total patches."
        )

    def save_metadata(self):
        """
        Save the patch metadata to a CSV file.
        """
        metadata_file_name = os.path.join(self.args.output_dir, "patches_metadata.csv")
        self.patches_metadata.to_csv(metadata_file_name, index=False)
        logging.info(f"Saved metadata to {metadata_file_name}")

    def save_patches(self, save_all=True):
        """
        Save the patches as NumPy arrays.
        """
        patches_file_name = os.path.join(
            self.args.output_dir, "extracted_channel_patches.npy"
        )
        np.save(patches_file_name, self.extracted_channel_patches)
        logging.info(f"Saved extracted_channel_patches to {patches_file_name}")

        if save_all:
            patches_file_name = os.path.join(
                self.args.output_dir, "all_channel_patches.npy"
            )
            np.save(patches_file_name, self.all_channel_patches)
            logging.info(f"Saved all_channel_patches to {patches_file_name}")

    def add_noise(self, noise_type="gaussian", **kwargs):
        """
        Add noise to the patches for robustness testing.

        Args:
            noise_type (str): Type of noise to add ('gaussian', 'salt_and_pepper', 'poisson', 'speckle').
            **kwargs: Additional parameters for noise configuration (e.g., sigma for Gaussian).
        """
        noisy_patches = []

        for patch in self.all_patches_ndarray:
            if noise_type == "gaussian":
                sigma = kwargs.get("sigma", 500)
                noisy_patch = self._apply_gaussian_noise(patch, sigma)
            elif noise_type == "downsampling":
                scale = kwargs.get("scale", 0.5)
                noisy_patch = self._downsample_patch(patch, scale)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            noisy_patches.append(noisy_patch)

        self.noisy_patches_ndarray = np.array(noisy_patches)
        logging.info(f"Added {noise_type} noise to patches with params {kwargs}")

    def _apply_gaussian_noise(self, image, sigma):
        row, col, ch = image.shape
        gauss = np.random.normal(0, sigma, (row, col, ch))
        noisy = np.clip(image + gauss, 0, 65535).astype(np.uint16)
        return noisy

    def _downsample_patch(self, image, scale):
        # scale: [0.3, 0.5, 0.7]
        height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
        downsampled = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return downsampled

    def save_noisy_patches(self, noise_type="gaussian", scale=None):
        """
        Save patches with added noise.

        Args:
            noise_type (str): Type of noise ("gaussian" or "downsampling").
            scale (float, optional): Scale for downsampling (if applicable).
        """
        if noise_type == "gaussian":
            for i, noisy_patches in enumerate(self.noisy_patches_ndarray):
                filename = os.path.join(
                    self.args.output_dir, f"gaussian_noise_level_{i}.npy"
                )
                np.save(filename, noisy_patches)
                logging.info(f"Saved Gaussian noise level {i} patches to {filename}.")
        elif noise_type == "downsampling" and scale:
            downsampled_patches = self.noisy_patches_ndarray.get(scale)
            if downsampled_patches is not None:
                filename = os.path.join(
                    self.args.output_dir, f"downsampled_{int(scale * 100)}.npy"
                )
                np.save(filename, np.array(downsampled_patches))
                logging.info(
                    f"Saved downsampled patches at {int(scale * 100)}% to {filename}."
                )
