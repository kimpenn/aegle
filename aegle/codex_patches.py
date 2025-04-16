# codex_patches.py

import os
import sys
import logging
import numpy as np
import pandas as pd
import cv2
from skimage.io import imsave
from aegle.codex_image import CodexImage
import pickle

def save_large_array(array, file_path):
    """ Save a large NumPy array using different methods. """
    # Method 1: np.savez_compressed (Recommended for large arrays)
    np.savez_compressed(file_path.replace('.npy', '.npz'), array=array)
class CodexPatches:
    def __init__(self, codex_image: CodexImage, config, args):
        """
        Initialize the CodexPatches object and generate patches from CodexImage.

        Args:
            codex_image (CodexImage): CodexImage object with preprocessed image data.
            config (dict): Configuration parameters.
            args (Namespace): Command-line arguments.
        """
        self.config = config
        self.args = args
        self.codex_image = codex_image
        self.antibody_df = codex_image.antibody_df.copy()

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)
        self.all_channel_patches = None
        self.extracted_channel_patches = None
        self.noisy_extracted_channel_patches = None
        self.patches_metadata = None

        self.valid_patches = None
        self.repaired_seg_res_batch = None
        self.original_seg_res_batch = None

        self.tif_image_details = codex_image.tif_image_details

        self.generate_patches()
        self.init_patch_metadata()
        self.qc_patch_metadata()

    def get_patches(self):
        if self.noisy_extracted_channel_patches is not None:
            self.logger.info("Returning noisy patches.")
            return self.noisy_extracted_channel_patches
        self.logger.info("Returning extracted patches.")
        return self.extracted_channel_patches

    def get_patches_metadata(self):
        return self.patches_metadata

    def generate_patches(self):
        """
        Generate patches from the preprocessed image and store them in an ndarray.
        """
        patching_config = self.config.get("patching", {})
        patch_height = patching_config.get("patch_height", 1440)
        patch_width = patching_config.get("patch_width", 1920)
        overlap = patching_config.get("overlap", 0.1)
        self.logger.info(f"patching_config: {patching_config}")

        # If both patch dimensions are negative, treat the entire image as one patch
        if patch_height < 0 and patch_width < 0:
            self.logger.info(
                "patch_height and patch_width are both < 0; using the entire extended image as a single patch."
            )
            # The extended images
            extracted_img = self.codex_image.extended_extracted_channel_image
            all_img = self.codex_image.extended_all_channel_image

            # Store them as a single entry list
            self.extracted_channel_patches = np.array([extracted_img])
            logging.info(
                f"Shape of extracted_channel_patches: {self.extracted_channel_patches.shape}"
            )
            self.all_channel_patches = np.array([all_img])
            logging.info(f"Shape of all_channel_patches: {self.all_channel_patches.shape}")
            # Create metadata for the single patch
            h, w, c = extracted_img.shape
            self.patches_metadata = [
                {
                    "patch_index": 0,
                    "x_start": 0,
                    "y_start": 0,
                    "patch_width": w,
                    "patch_height": h,
                }
            ]

        else:
            # Calculate overlap and step sizes
            overlap_height = int(patch_height * overlap)
            overlap_width = int(patch_width * overlap)
            step_height = patch_height - overlap_height
            step_width = patch_width - overlap_width

            self.logger.info(
                f"before crop_image_into_patches: {self.codex_image.extended_extracted_channel_image.shape}"
            )
            self.extracted_channel_patches = self.crop_image_into_patches(
                self.codex_image.extended_extracted_channel_image,
                patch_height,
                patch_width,
                step_height,
                step_width,
            )

            self.all_channel_patches = self.crop_image_into_patches(
                self.codex_image.extended_all_channel_image,
                patch_height,
                patch_width,
                step_height,
                step_width,
            )
        self.logger.info(
            f"Number of patches generated: {len(self.extracted_channel_patches)}"
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

        self.logger.info("Generating metadata for patches...")
        num_patches = self.extracted_channel_patches.shape[0]
        self.logger.info(
            f"=== size of extracted_channel_patches: {self.extracted_channel_patches.shape}"
        )
        self.patches_metadata = pd.DataFrame(
            {
                "patch_id": range(num_patches),
                "height": patch_height,
                "width": patch_width,
                "nucleus_mean": [
                    np.mean(patch[:, :, 0]) for patch in self.extracted_channel_patches
                ],
                "nucleus_std": [
                    np.std(patch[:, :, 0]) for patch in self.extracted_channel_patches
                ],
                "nucleus_non_zero_perc": [
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
        self.logger.info("Performing quality control on patch metadata...")

        # Get QC parameters from config
        qc_config = self.config.get("patch_qc", {})
        non_zero_perc_threshold = qc_config.get("non_zero_perc_threshold", 0.05)
        mean_intensity_threshold = qc_config.get("mean_intensity_threshold", 1)

        # Identify patches that are empty or noisy
        self.patches_metadata["is_empty"] = (
            self.patches_metadata["nucleus_non_zero_perc"] < non_zero_perc_threshold
        )

        self.patches_metadata["is_noisy"] = (
            self.patches_metadata["nucleus_mean"] < mean_intensity_threshold
        )
        # Mark patches as bad (either empty or noisy)
        self.patches_metadata["is_bad_patch"] = (
            self.patches_metadata["is_empty"] | self.patches_metadata["is_noisy"]
        )
        self.patches_metadata["is_infomative"] = ~self.patches_metadata["is_bad_patch"]
        num_bad_patches = self.patches_metadata["is_bad_patch"].sum()
        total_patches = len(self.patches_metadata)

        self.logger.info(
            f"Identified {num_bad_patches} bad patches out of {total_patches} total patches."
        )

    def save_metadata(self):
        """
        Save the patch metadata to a CSV file.
        """
        metadata_file_name = os.path.join(self.args.out_dir, "patches_metadata.csv")
        self.patches_metadata.to_csv(metadata_file_name, index=False)
        self.logger.info(f"Saved metadata to {metadata_file_name}")

    def save_patches(self, save_all_channel_patches=True):
        """
        Save the patches as NumPy arrays.
        """
        patches_file_name = os.path.join(
            self.args.out_dir, "extracted_channel_patches.npy"
        )
        np.save(patches_file_name, self.extracted_channel_patches)
        self.logger.info(f"Shape of extracted_channel_patches: {self.extracted_channel_patches.shape}")
        self.logger.info(f"Saved extracted_channel_patches to {patches_file_name}")

        if save_all_channel_patches:
            patches_file_name = os.path.join(
                self.args.out_dir, "all_channel_patches.npy"
            )
            np.save(patches_file_name, self.all_channel_patches)
            self.logger.info(f"Saved all_channel_patches to {patches_file_name}")

    def add_disruptions(self, disruption_type, disruption_level):
        """
        Add noise to the informative patches for robustness testing.

        Args:
            disruption_type (str): Type of disruption to add ('gaussian', 'downsampling').
            disruption_level (int): Level of disruption to apply (e.g., 1, 2, 3).
        """

        disrupted_patches = []

        for i, patch in enumerate(self.extracted_channel_patches):
            if disruption_type == "gaussian":
                if self.patches_metadata.loc[i, "is_infomative"]:
                    sigma_dict = {
                        8: {1: 1.94, 2: 3.88, 3: 5.82},
                        16: {1: 500, 2: 1000, 3: 1500},
                    }
                    bit_per_sample = self.tif_image_details["BitsPerSample"]
                    sigma = sigma_dict[bit_per_sample][disruption_level]
                    noisy_patch = self._apply_gaussian_noise(patch, sigma)
                    self.logger.debug(
                        f"Added {disruption_type} noise to patch ({patch.shape}) with sigma {sigma}"
                    )
                else:
                    noisy_patch = patch
            elif disruption_type == "downsampling":
                # level 1 = 0.3, level 2 = 0.5, level 3 = 0.7
                scale = 0.3 + 0.2 * disruption_level
                noisy_patch = self._downsample_patch(patch, scale)
                self.logger.debug(
                    f"Added {disruption_type} noise to patch with sigma {scale}"
                )
            else:
                raise ValueError(f"Unsupported disruption_type: {disruption_type}")

            disrupted_patches.append(noisy_patch)

        self.disrupted_extracted_channel_patches = np.array(disrupted_patches)

    def save_disrupted_patches(self):
        """
        Save patches with added noise.
        """
        patches_file_name = os.path.join(
            self.args.out_dir, "disrupted_extracted_channel_patches.npy"
        )
        np.save(patches_file_name, self.disrupted_extracted_channel_patches)
        self.logger.info(
            f"Saved disrupted_extracted_channel_patches to {patches_file_name}"
        )

    def _apply_gaussian_noise(self, image, sigma):
        mean_original = np.mean(image)
        std_original = np.std(image)

        self.logger.info(f"Original Image Mean: {mean_original}, Std: {std_original}")
        row, col, ch = image.shape
        gauss = np.random.normal(0, sigma, (row, col, ch))
        noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy

    def _downsample_patch(self, image, scale):
        # scale: [0.3, 0.5, 0.7]
        height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
        downsampled = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return downsampled

    def set_seg_res(self, repaired_seg_res_batch, original_seg_res_batch=None):
        self.repaired_seg_res_batch = repaired_seg_res_batch
        self.original_seg_res_batch = original_seg_res_batch

    def set_metadata(self, patches_metadata):
        self.patches_metadata = patches_metadata

    def save_seg_res(self):
        def inspect_and_save(data, file_name):
            """Inspect the type, shape, and sample content of the data before saving."""
            if data is not None:
                self.logger.info(f"Saving {file_name}...")

                # Type and basic info
                self.logger.info(f"Type: {type(data)}")
                if hasattr(data, "shape"):
                    self.logger.info(f"Shape: {data.shape}")
                elif isinstance(data, list):
                    self.logger.info(f"Length: {len(data)}")
                
                # Check the first element's type and shape (if applicable)
                if isinstance(data, (list, tuple)) and len(data) > 0:
                    first_elem = data[0]
                    self.logger.info(f"First element type: {type(first_elem)}")
                    if hasattr(first_elem, "shape"):
                        self.logger.info(f"First element shape: {first_elem.shape}")

                # Sample content (Avoid printing too much data)
                sample_content = data[:5] if isinstance(data, (list, np.ndarray)) else str(data)[:500]
                self.logger.info(f"Sample content: {sample_content}")

                file_name = os.path.join(self.args.out_dir, file_name)
                self.logger.info(f"Saving to {file_name}...")
                # Save with Pickle (Protocol 4 for large files)
                with open(file_name, 'wb') as f:
                    pickle.dump(data, f, protocol=4)

                self.logger.info(f"Saved {file_name}")

        # Inspect and save `repaired_seg_res_batch`
        inspect_and_save(self.repaired_seg_res_batch, "matched_seg_res_batch.pickle")

        # Inspect and save `original_seg_res_batch`
        inspect_and_save(self.original_seg_res_batch, "original_seg_res_batch.pickle")

    # def save_seg_res(self):
    #     if self.repaired_seg_res_batch is not None:
    #         # seg_res_file_name = os.path.join(
    #         #     self.args.out_dir, "matched_seg_res_batch.npy"
    #         # )
    #         # np.save(seg_res_file_name, self.repaired_seg_res_batch)
    #         # np.save(seg_res_file_name, self.repaired_seg_res_batch, allow_pickle=True)            
    #         # seg_res_file_name = os.path.join(
    #         #     self.args.out_dir, "matched_seg_res_batch.npz"
    #         # )            
    #         # np.savez_compressed(seg_res_file_name, self.repaired_seg_res_batch)
    #         seg_res_file_name = "matched_seg_res_batch.pickle"
    #         with open(seg_res_file_name, 'wb') as f:
    #             pickle.dump(self.repaired_seg_res_batch, f, protocol=4)

    #         self.logger.info(f"Saved matched_seg_res_batch to {seg_res_file_name}")

    #     if self.original_seg_res_batch is not None:
    #         # seg_res_file_name = os.path.join(
    #         #     self.args.out_dir, "original_seg_res_batch.npy"
    #         # )
    #         # np.save(seg_res_file_name, self.original_seg_res_batch, allow_pickle=True) 
    #         # seg_res_file_name = os.path.join(
    #         #     self.args.out_dir, "original_seg_res_batch.npz"
    #         # )          
    #         # np.savez_compressed(seg_res_file_name, self.original_seg_res_batch)      
    #         seg_res_file_name = "original_seg_res_batch.pickle"
    #         with open(seg_res_file_name, 'wb') as f:
    #             pickle.dump(self.repaired_seg_res_batch, f, protocol=4)                    
    #         self.logger.info(f"Saved original_seg_res_batch to {seg_res_file_name}")
