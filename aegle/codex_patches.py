# codex_patches.py

import os
import sys
import gzip
import logging
import numpy as np
import pandas as pd
import cv2
from skimage.io import imsave
from skimage.segmentation import find_boundaries
from typing import Dict, List, Optional, Tuple
from aegle.codex_image import CodexImage
import pickle
import tifffile


def save_large_array(array, file_path):
    """Save a large NumPy array using different methods."""
    # Method 1: np.savez_compressed (Recommended for large arrays)
    np.savez_compressed(file_path.replace(".npy", ".npz"), array=array)


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
        self.update_patch_metadata()
        self.qc_patch_metadata()

    def get_patches(self):
        if self.noisy_extracted_channel_patches is not None:
            self.logger.info("Returning noisy patches.")
            return self.noisy_extracted_channel_patches
        
        # Check if we're using disk-based patches
        if hasattr(self, 'patch_files') and self.patch_files:
            self.logger.info("Using disk-based patches for large images.")
            return self._get_disk_based_patches_info()
        
        self.logger.info("Returning extracted patches.")
        return self.extracted_channel_patches
    
    def _get_disk_based_patches_info(self):
        """Return patch information for disk-based patches."""
        # Return a special object that indicates disk-based patches
        return {"disk_based": True, "patch_files": self.patch_files}
    
    def load_patch_from_disk(self, patch_index, patch_type="extracted"):
        """
        Load a specific patch from disk.
        
        Args:
            patch_index (int): Index of the patch to load
            patch_type (str): Type of patch to load ("extracted" or "all")
            
        Returns:
            np.ndarray: The loaded patch
        """
        if not hasattr(self, 'patch_files') or not self.patch_files:
            raise ValueError("No disk-based patches available")
            
        if patch_index >= len(self.patch_files):
            raise ValueError(f"Patch index {patch_index} out of range")
            
        patch_info = self.patch_files[patch_index]
        file_path = patch_info[patch_type]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Patch file not found: {file_path}")
            
        self.logger.info(f"Loading patch {patch_index} ({patch_type}) from disk: {os.path.basename(file_path)}")
        patch = np.load(file_path)
        self.logger.info(f"Loaded patch {patch_index} with shape: {patch.shape}")
        
        return patch
    
    def cleanup_intermediate_patches(self):
        """Remove intermediate patch files from disk."""
        if not hasattr(self, 'patch_files') or not self.patch_files:
            self.logger.info("No intermediate patch files to clean up")
            return
            
        patches_dir = os.path.join(self.args.out_dir, "intermediate_patches")
        total_files = len(self.patch_files) * 2  # extracted + all for each patch
        files_removed = 0
        
        self.logger.info(f"Cleaning up {total_files} intermediate patch files...")
        
        for i, patch_info in enumerate(self.patch_files):
            try:
                # Remove extracted patch file
                if os.path.exists(patch_info["extracted"]):
                    os.remove(patch_info["extracted"])
                    files_removed += 1
                    self.logger.info(f"Removed {os.path.basename(patch_info['extracted'])}")
                
                # Remove all-channel patch file  
                if os.path.exists(patch_info["all"]):
                    os.remove(patch_info["all"])
                    files_removed += 1
                    self.logger.info(f"Removed {os.path.basename(patch_info['all'])}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to remove patch {i} files: {e}")
        
        # Remove the intermediate patches directory if empty
        try:
            if os.path.exists(patches_dir) and not os.listdir(patches_dir):
                os.rmdir(patches_dir)
                self.logger.info(f"Removed empty directory: {patches_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to remove patches directory: {e}")
            
        self.logger.info(f"Cleanup completed. Removed {files_removed}/{total_files} files")
    
    def is_using_disk_based_patches(self):
        """Check if this instance is using disk-based patches."""
        return hasattr(self, 'patch_files') and bool(self.patch_files)

    def get_patches_metadata(self):
        return self.patches_metadata

    def generate_patches(self):
        """
        Generate patches from the preprocessed image and store them in an ndarray.
        Supports four modes:
        - "full_image": Use entire image as single patch
        - "halves": Split image into 2 pieces (vertical or horizontal)
        - "quarters": Split image into 2x2 grid (4 pieces)  
        - "patches": Small patches with overlap for segmentation
        """
        patching_config = self.config.get("patching", {})
        split_mode = patching_config.get("split_mode", "patches")
        self.logger.info(f"patching_config: {patching_config}")
        self.logger.info(f"split_mode: {split_mode}")

        # Get image dimensions
        extracted_img = self.codex_image.extended_extracted_channel_image
        all_img = self.codex_image.extended_all_channel_image
        img_height, img_width, img_channels = extracted_img.shape
        
        if split_mode == "full_image":
            self._generate_full_image_patch(extracted_img, all_img)
            
        elif split_mode == "halves":
            split_direction = patching_config.get("split_direction", "vertical")
            self._generate_halves_patches(extracted_img, all_img, split_direction)
            
        elif split_mode == "quarters":
            self._generate_quarters_patches(extracted_img, all_img)
            
        elif split_mode == "patches":
            patch_height = patching_config.get("patch_height", 1440)
            patch_width = patching_config.get("patch_width", 1920)
            overlap = patching_config.get("overlap", 0.1)
            self._generate_small_patches(extracted_img, all_img, patch_height, patch_width, overlap)
            
        else:
            raise ValueError(f"Unsupported split_mode: {split_mode}. Use 'full_image', 'halves', 'quarters', or 'patches'.")
            
        self.logger.info(
            f"Number of patches generated: {len(self.extracted_channel_patches)}"
        )

    def _generate_full_image_patch(self, extracted_img, all_img):
        """Generate a single patch using the entire image."""
        self.logger.info("Using the entire extended image as a single patch.")
        
        # Store them as a single entry list
        self.extracted_channel_patches = np.array([extracted_img])
        self.all_channel_patches = np.array([all_img])
        
        self.logger.info(
            f"Shape of extracted_channel_patches: {self.extracted_channel_patches.shape}"
        )
        self.logger.info(
            f"Shape of all_channel_patches: {self.all_channel_patches.shape}"
        )
        
        # Create metadata for the single patch
        h, w, c = extracted_img.shape
        self.patches_metadata = [
            {
                "patch_id": 0,  # Add patch_id for consistency
                "patch_index": 0,
                "x_start": 0,
                "y_start": 0,
                "patch_width": w,
                "patch_height": h,
            }
        ]

    def _generate_halves_patches(self, extracted_img, all_img, split_direction):
        """Generate 2 patches by splitting the image in half."""
        self.logger.info(f"Splitting image into halves ({split_direction}).")
        
        h, w, c = extracted_img.shape
        
        if split_direction == "vertical":
            # Split into left and right halves
            split_x = w // 2
            coordinates = [
                {"x_start": 0, "y_start": 0, "width": split_x, "height": h},
                {"x_start": split_x, "y_start": 0, "width": w - split_x, "height": h}
            ]
        elif split_direction == "horizontal":
            # Split into top and bottom halves
            split_y = h // 2
            coordinates = [
                {"x_start": 0, "y_start": 0, "width": w, "height": split_y},
                {"x_start": 0, "y_start": split_y, "width": w, "height": h - split_y}
            ]
        else:
            raise ValueError(f"Invalid split_direction: {split_direction}. Use 'vertical' or 'horizontal'.")
        
        self._extract_patches_from_coordinates(extracted_img, all_img, coordinates)

    def _generate_quarters_patches(self, extracted_img, all_img):
        """Generate 4 patches by splitting the image into quarters (2x2 grid)."""
        self.logger.info("Splitting image into quarters (2x2 grid).")
        
        h, w, c = extracted_img.shape
        split_x = w // 2
        split_y = h // 2
        
        # Create 2x2 grid coordinates
        coordinates = [
            {"x_start": 0, "y_start": 0, "width": split_x, "height": split_y},  # Top-left
            {"x_start": split_x, "y_start": 0, "width": w - split_x, "height": split_y},  # Top-right
            {"x_start": 0, "y_start": split_y, "width": split_x, "height": h - split_y},  # Bottom-left
            {"x_start": split_x, "y_start": split_y, "width": w - split_x, "height": h - split_y}  # Bottom-right
        ]
        
        self._extract_patches_from_coordinates(extracted_img, all_img, coordinates)

    def _generate_small_patches(self, extracted_img, all_img, patch_height, patch_width, overlap):
        """Generate small patches with overlap for segmentation."""
        self.logger.info(f"Generating small patches ({patch_height}x{patch_width}) with {overlap} overlap.")
        
        # Calculate overlap and step sizes
        overlap_height = int(patch_height * overlap)
        overlap_width = int(patch_width * overlap)
        step_height = patch_height - overlap_height
        step_width = patch_width - overlap_width

        self.logger.info(
            f"before crop_image_into_patches: {extracted_img.shape}"
        )
        
        self.extracted_channel_patches = self.crop_image_into_patches(
            extracted_img,
            patch_height,
            patch_width,
            step_height,
            step_width,
        )

        self.all_channel_patches = self.crop_image_into_patches(
            all_img,
            patch_height,
            patch_width,
            step_height,
            step_width,
        )

    def _extract_patches_from_coordinates(self, extracted_img, all_img, coordinates):
        """Extract patches from given coordinates and save them to disk."""
        patching_config = self.config.get("patching", {})
        split_mode = patching_config.get("split_mode", "patches")
        
        # Create patches directory for intermediate files
        patches_dir = os.path.join(self.args.out_dir, "intermediate_patches")
        os.makedirs(patches_dir, exist_ok=True)
        
        patch_files = []
        metadata = []
        total_patches = len(coordinates)
        
        self.logger.info(f"Saving {total_patches} patches to disk for memory management...")
        
        for i, coord in enumerate(coordinates):
            x_start = coord["x_start"]
            y_start = coord["y_start"]
            width = coord["width"]
            height = coord["height"]
            
            self.logger.info(f"Processing and saving patch {i+1}/{total_patches} to disk...")
            
            # Extract patches
            extracted_patch = extracted_img[y_start:y_start + height, x_start:x_start + width, :]
            all_patch = all_img[y_start:y_start + height, x_start:x_start + width, :]
            
            # Generate filenames with split mode and patch index
            extracted_filename = f"patch_{i}_{split_mode}_extracted.npy"
            all_filename = f"patch_{i}_{split_mode}_all.npy"
            
            extracted_path = os.path.join(patches_dir, extracted_filename)
            all_path = os.path.join(patches_dir, all_filename)
            
            # Save patches to disk with progress logging
            self.logger.info(f"Saving extracted patch {i+1} ({extracted_patch.shape}) to {extracted_filename}")
            np.save(extracted_path, extracted_patch)
            
            self.logger.info(f"Saving all-channel patch {i+1} ({all_patch.shape}) to {all_filename}")
            np.save(all_path, all_patch)
            
            # Store file paths and metadata
            patch_files.append({
                "extracted": extracted_path,
                "all": all_path,
                "shape_extracted": extracted_patch.shape,
                "shape_all": all_patch.shape
            })
            
            # Create metadata
            metadata.append({
                "patch_id": i,  # Add patch_id for consistency
                "patch_index": i,
                "x_start": x_start,
                "y_start": y_start,
                "patch_width": width,
                "patch_height": height,
            })
            
            self.logger.info(f"Patch {i+1}/{total_patches} saved successfully")
        
        # Store file paths instead of patch data
        self.patch_files = patch_files
        self.patches_metadata = metadata
        
        # Create dummy arrays for compatibility (will not be used for large patches)
        self.extracted_channel_patches = np.array([])
        self.all_channel_patches = np.array([])
        
        self.logger.info(f"All {total_patches} patches saved to disk successfully")
        self.logger.info(f"Intermediate patches directory: {patches_dir}")

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

    def update_patch_metadata(self):
        """
        Generate metadata for the patches and store it in self.patches_metadata.
        If metadata already exists from generate_patches, enhance it with statistics.
        """
        patching_config = self.config.get("patching", {})
        split_mode = patching_config.get("split_mode", "patches")
        
        self.logger.info("Generating metadata for patches...")
        
        # Handle disk-based patches
        if self.is_using_disk_based_patches():
            num_patches = len(self.patch_files)
            self.logger.info(f"=== Processing metadata for {num_patches} disk-based patches")
            
            # Convert to DataFrame if it's a list of dicts
            if not isinstance(self.patches_metadata, pd.DataFrame):
                self.patches_metadata = pd.DataFrame(self.patches_metadata)
            
            # Add statistics by loading patches one at a time
            for i in range(num_patches):
                self.logger.info(f"Loading patch {i+1}/{num_patches} for metadata calculation...")
                
                try:
                    # Load patch from disk
                    patch = self.load_patch_from_disk(i, "extracted")
                    patch_height, patch_width = patch.shape[:2]
                    
                    # Update dimensions
                    self.patches_metadata.loc[i, "height"] = patch_height
                    self.patches_metadata.loc[i, "width"] = patch_width
                    
                    # Calculate statistics
                    self.patches_metadata.loc[i, "nucleus_mean"] = np.mean(patch[:, :, 0])
                    self.patches_metadata.loc[i, "nucleus_std"] = np.std(patch[:, :, 0])
                    self.patches_metadata.loc[i, "nucleus_non_zero_perc"] = (
                        np.count_nonzero(patch[:, :, 0]) / (patch_height * patch_width)
                    )
                    self.patches_metadata.loc[i, "wholecell_mean"] = np.mean(patch[:, :, 1])
                    self.patches_metadata.loc[i, "wholecell_std"] = np.std(patch[:, :, 1])
                    self.patches_metadata.loc[i, "wholecell_non_zero_perc"] = (
                        np.count_nonzero(patch[:, :, 1]) / (patch_height * patch_width)
                    )
                    
                    # Clear patch from memory immediately
                    del patch
                    
                    self.logger.info(f"Metadata calculated for patch {i+1}/{num_patches}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to calculate metadata for patch {i}: {e}")
                    raise
                    
        else:
            # Handle in-memory patches (existing logic)
            num_patches = self.extracted_channel_patches.shape[0]
            self.logger.info(
                f"=== size of extracted_channel_patches: {self.extracted_channel_patches.shape}"
            )
            
            # Check if metadata already exists from generate_patches (for new split modes)
            if self.patches_metadata is not None and len(self.patches_metadata) == num_patches:
                # Convert to DataFrame if it's a list of dicts
                if not isinstance(self.patches_metadata, pd.DataFrame):
                    self.patches_metadata = pd.DataFrame(self.patches_metadata)
                
                # Add statistics to existing metadata
                for i, patch in enumerate(self.extracted_channel_patches):
                    # Get actual patch dimensions
                    patch_height, patch_width = patch.shape[:2]
                    
                    # Update height and width if they're -1 (from old config format)
                    if "patch_height" in self.patches_metadata.columns:
                        self.patches_metadata.loc[i, "patch_height"] = patch_height
                    if "patch_width" in self.patches_metadata.columns:
                        self.patches_metadata.loc[i, "patch_width"] = patch_width
                    
                    # Add missing columns if they don't exist
                    if "height" not in self.patches_metadata.columns:
                        self.patches_metadata["height"] = patch_height
                    if "width" not in self.patches_metadata.columns:
                        self.patches_metadata["width"] = patch_width
                    
            else:
                # Create new metadata for legacy patches mode
                patch_height = patching_config.get("patch_height", 1440)
                patch_width = patching_config.get("patch_width", 1920)
                
                self.patches_metadata = pd.DataFrame(
                    {
                        "patch_id": range(num_patches),
                        "height": patch_height,
                        "width": patch_width,
                    }
                )
            
            # Add statistical metadata for all patches
            for i, patch in enumerate(self.extracted_channel_patches):
                patch_height, patch_width = patch.shape[:2]
                
                self.patches_metadata.loc[i, "nucleus_mean"] = np.mean(patch[:, :, 0])
                self.patches_metadata.loc[i, "nucleus_std"] = np.std(patch[:, :, 0])
                self.patches_metadata.loc[i, "nucleus_non_zero_perc"] = (
                    np.count_nonzero(patch[:, :, 0]) / (patch_height * patch_width)
                )
                self.patches_metadata.loc[i, "wholecell_mean"] = np.mean(patch[:, :, 1])
                self.patches_metadata.loc[i, "wholecell_std"] = np.std(patch[:, :, 1])
                self.patches_metadata.loc[i, "wholecell_non_zero_perc"] = (
                    np.count_nonzero(patch[:, :, 1]) / (patch_height * patch_width)
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
        self.patches_metadata["is_informative"] = ~self.patches_metadata["is_bad_patch"]
        num_bad_patches = self.patches_metadata["is_bad_patch"].sum()
        total_patches = len(self.patches_metadata)

        self.logger.info(
            f"Identified {num_bad_patches} bad patches out of {total_patches} total patches."
        )

    def save_patches(self, save_all_channel_patches=True):
        """
        Save the patches as NumPy arrays.
        """
        patches_file_name = os.path.join(
            self.args.out_dir, "extracted_channel_patches.npy"
        )
        compressed_file_name = f"{patches_file_name}.gz"
        with gzip.open(compressed_file_name, "wb") as file_handle:
            np.save(file_handle, self.extracted_channel_patches)
        self.logger.info(
            f"Shape of extracted_channel_patches: {self.extracted_channel_patches.shape}"
        )
        self.logger.info(
            f"Saved extracted_channel_patches to {compressed_file_name} using gzip compression"
        )

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
                if self.patches_metadata.loc[i, "is_informative"]:
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

    def save_metadata(self, file_name: Optional[str] = None):
        """Persist patch metadata as CSV in the run output directory."""
        if file_name is None:
            target_path = os.path.join(self.args.out_dir, "patches_metadata.csv")
        else:
            target_path = (
                file_name
                if os.path.isabs(file_name)
                else os.path.join(self.args.out_dir, file_name)
            )

        base, ext = os.path.splitext(target_path)
        if ext.lower() not in {"", ".csv"}:
            self.logger.warning(
                "Replacing unsupported metadata extension '%s' with '.csv'", ext
            )
            target_path = f"{base}.csv"
        elif ext == "":
            target_path = f"{target_path}.csv"

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        self.patches_metadata.to_csv(target_path, index=False)
        self.logger.info(f"Saved patch metadata to {target_path}")

    def save_seg_res(self):
        compression_cfg = self.config.get("segmentation", {})
        compression = str(
            compression_cfg.get("segmentation_pickle_compression", "none")
        ).lower()
        compression_level = compression_cfg.get(
            "segmentation_pickle_compression_level", None
        )

        if isinstance(compression_level, str):
            try:
                compression_level = int(compression_level)
            except ValueError:
                self.logger.warning(
                    "Invalid segmentation_pickle_compression_level '%s'; ignoring",
                    compression_level,
                )
                compression_level = None

        def inspect_and_save(data, base_file_name):
            """Inspect the type, shape, and sample content of the data before saving."""
            if data is None:
                return

            compression_mode = compression
            compression_label = "none"
            self.logger.info(f"Saving {base_file_name}...")

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
            sample_content = (
                data[:5] if isinstance(data, (list, np.ndarray)) else str(data)[:500]
            )
            self.logger.debug(f"Sample content: {sample_content}")

            file_path = os.path.join(self.args.out_dir, base_file_name)
            opener = open
            opener_kwargs = {}
            suffix = ""

            if compression_mode in {"gzip", "gz"}:
                import gzip

                suffix = ".gz"
                opener = gzip.open
                compression_label = "gzip"
                if compression_level is not None:
                    opener_kwargs["compresslevel"] = compression_level
            elif compression_mode in {"bz2", "bzip2"}:
                import bz2

                suffix = ".bz2"
                opener = bz2.open
                compression_label = "bz2"
                if compression_level is not None:
                    opener_kwargs["compresslevel"] = compression_level
            elif compression_mode in {"lzma", "xz"}:
                import lzma

                suffix = ".xz"
                opener = lzma.open
                compression_label = "lzma"
                if compression_level is not None:
                    opener_kwargs["preset"] = compression_level
            elif compression_mode not in {"none", "", None}:
                self.logger.warning(
                    "Unsupported segmentation_pickle_compression '%s'; defaulting to no compression",
                    compression_mode,
                )
                compression_mode = "none"
                compression_label = "none"

            target_path = file_path + suffix
            self.logger.info(f"Saving to {target_path}...")

            with opener(target_path, "wb", **opener_kwargs) as f:
                pickle.dump(data, f, protocol=4)

            if suffix:
                self.logger.info(
                    "Saved %s using %s compression", target_path, compression_label
                )
            else:
                self.logger.info(f"Saved {target_path}")

        # Inspect and save `repaired_seg_res_batch`
        inspect_and_save(self.repaired_seg_res_batch, "matched_seg_res_batch.pickle")

        # Inspect and save `original_seg_res_batch`
        inspect_and_save(self.original_seg_res_batch, "original_seg_res_batch.pickle")

    def export_segmentation_masks(self, config, args):
        if self.repaired_seg_res_batch is None:
            self.logger.warning("No repaired segmentation results to export")
            return

        metadata_df = self.get_patches_metadata()
        if metadata_df is None or metadata_df.empty:
            self.logger.warning("No patch metadata available for segmentation export")
            return

        exp_id = config.get("exp_id") or config.get("name", "experiment")
        output_dir = os.path.join(args.out_dir, "segmentations")
        os.makedirs(output_dir, exist_ok=True)
        base_name = exp_id if exp_id else "segmentation"

        image_mpp = config.get("data", {}).get("image_mpp")
        split_mode = self.config.get("patching", {}).get("split_mode", "patches")

        informative_mask = metadata_df["is_informative"] == True
        informative_indices = metadata_df[informative_mask].index.tolist()
        original_results = self.original_seg_res_batch or []
        repaired_results = self.repaired_seg_res_batch or []

        exported = False

        if split_mode == "patches":
            exported = self._export_patch_masks(
                base_name,
                output_dir,
                informative_indices,
                metadata_df,
                original_results,
                repaired_results,
                image_mpp,
            )
        else:
            exported = self._export_aggregated_masks(
                base_name,
                output_dir,
                metadata_df,
                informative_indices,
                original_results,
                repaired_results,
                image_mpp,
            )

        if not exported:
            self.logger.warning("No segmentation masks exported")

    def _merge_segmentation_masks(
        self,
        metadata_df,
        informative_indices,
        seg_results,
        result_key,
        image_shape,
    ):
        if not seg_results or result_key is None:
            return None

        aggregated = np.zeros(image_shape, dtype=np.uint32)
        current_label = 1

        for idx_pos, patch_idx in enumerate(informative_indices):
            if idx_pos >= len(seg_results):
                break
            seg_result = seg_results[idx_pos]
            if seg_result is None:
                continue
            mask = seg_result.get(result_key)
            if mask is None:
                continue

            mask = np.asarray(mask, dtype=np.uint32)
            if mask.size == 0:
                continue

            if "y_start" in metadata_df.columns:
                y_val = metadata_df.loc[patch_idx, "y_start"]
                y_start = int(round(y_val)) if not pd.isna(y_val) else 0
            else:
                y_start = 0
            if "x_start" in metadata_df.columns:
                x_val = metadata_df.loc[patch_idx, "x_start"]
                x_start = int(round(x_val)) if not pd.isna(x_val) else 0
            else:
                x_start = 0

            h, w = mask.shape
            y_end = min(y_start + h, image_shape[0])
            x_end = min(x_start + w, image_shape[1])
            if y_end <= y_start or x_end <= x_start:
                continue

            target_slice = aggregated[y_start:y_end, x_start:x_end]
            patch_slice = mask[: y_end - y_start, : x_end - x_start]

            nonzero = patch_slice > 0
            if not np.any(nonzero):
                continue

            max_label = int(patch_slice.max())
            if max_label <= 0:
                continue

            reassigned = np.zeros_like(patch_slice, dtype=np.uint32)
            reassigned[nonzero] = patch_slice[nonzero] + current_label - 1

            overwrite_mask = nonzero & (target_slice == 0)
            target_slice[overwrite_mask] = reassigned[overwrite_mask]

            aggregated[y_start:y_end, x_start:x_end] = target_slice
            current_label += max_label

        return aggregated

    @staticmethod
    def _build_pyramid(base_level, min_size=256):
        pyramid = [base_level]
        current = base_level
        while (
            current.shape[-2] >= 2 * min_size
            and current.shape[-1] >= 2 * min_size
        ):
            down = current[:, ::2, ::2]
            if down.shape[-2] < min_size or down.shape[-1] < min_size:
                break
            pyramid.append(down)
            current = down
        return pyramid

    def _save_mask_stack(self, mask: Optional[np.ndarray], channel_name: str, output_path: str, image_mpp: Optional[float]) -> bool:
        if mask is None:
            return False
        mask_array = np.asarray(mask)
        if mask_array.size == 0:
            return False
        mask_array = np.squeeze(mask_array).astype(np.uint32, copy=False)

        stack = mask_array[None, ...]
        pyramid = self._build_pyramid(stack)

        metadata = {
            "axes": "CYX",
            "channel_names": [channel_name],
        }
        if image_mpp:
            try:
                metadata["PhysicalSizeX"] = float(image_mpp)
                metadata["PhysicalSizeY"] = float(image_mpp)
            except (TypeError, ValueError):
                pass

        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write(
                pyramid[0],
                subifds=len(pyramid) - 1,
                dtype=np.uint32,
                compression="zlib",
                photometric="minisblack",
                metadata=metadata,
            )
            for level in pyramid[1:]:
                tif.write(
                    level,
                    subfiletype=1,
                    dtype=np.uint32,
                    compression="zlib",
                    photometric="minisblack",
                )

        self.logger.info("Exported %s segmentation mask to %s", channel_name, output_path)
        return True

    def _export_patch_masks(
        self,
        base_name: str,
        output_dir: str,
        informative_indices: List[int],
        metadata_df: pd.DataFrame,
        original_results: List[Dict],
        repaired_results: List[Dict],
        image_mpp: Optional[float],
    ) -> bool:
        if len(repaired_results) != len(informative_indices):
            self.logger.warning(
                "Mismatch between repaired results (%d) and informative patches (%d)",
                len(repaired_results),
                len(informative_indices),
            )

        exported = False

        def compute_boundary(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if mask is None:
                return None
            mask_array = np.asarray(mask)
            if mask_array.size == 0:
                return None
            boundary_bool = find_boundaries(mask_array, mode="inner")
            boundary = np.zeros_like(mask_array, dtype=np.uint32)
            boundary[boundary_bool] = mask_array[boundary_bool]
            return boundary

        for idx_pos, patch_idx in enumerate(informative_indices):
            seg_result = repaired_results[idx_pos] if idx_pos < len(repaired_results) else None
            if seg_result is None:
                continue

            orig_seg_result = original_results[idx_pos] if idx_pos < len(original_results) else None
            patch_meta = metadata_df.loc[patch_idx]
            patch_id = patch_meta.get("patch_id", patch_meta.get("patch_index", idx_pos))
            try:
                patch_id_int = int(patch_id)
            except (TypeError, ValueError):
                patch_id_int = idx_pos

            patch_base = f"{base_name}.patch_{patch_id_int:04d}"

            cell_mask = None
            if orig_seg_result is not None:
                cell_mask = orig_seg_result.get("cell")
            if cell_mask is None:
                cell_mask = seg_result.get("cell")

            nucleus_mask = None
            if orig_seg_result is not None:
                nucleus_mask = orig_seg_result.get("nucleus")
            if nucleus_mask is None:
                nucleus_mask = seg_result.get("nucleus")

            cell_boundary = None
            if orig_seg_result is not None:
                cell_boundary = orig_seg_result.get("cell_boundary")
            if cell_boundary is None:
                cell_boundary = seg_result.get("cell_boundary")
            if cell_boundary is None:
                cell_boundary = compute_boundary(cell_mask)

            nucleus_boundary = None
            if orig_seg_result is not None:
                nucleus_boundary = orig_seg_result.get("nucleus_boundary")
            if nucleus_boundary is None:
                nucleus_boundary = seg_result.get("nucleus_boundary")
            if nucleus_boundary is None:
                nucleus_boundary = compute_boundary(nucleus_mask)

            cell_matched_mask = seg_result.get("cell_matched_mask", seg_result.get("cell"))
            nucleus_matched_mask = seg_result.get("nucleus_matched_mask", seg_result.get("nucleus"))

            cell_matched_boundary = seg_result.get("cell_matched_boundary")
            if cell_matched_boundary is None:
                cell_matched_boundary = compute_boundary(cell_matched_mask)

            nucleus_matched_boundary = seg_result.get("nucleus_matched_boundary")
            if nucleus_matched_boundary is None:
                nucleus_matched_boundary = compute_boundary(nucleus_matched_mask)

            cell_outside_nucleus_mask = seg_result.get("cell_outside_nucleus_mask")

            masks_to_save = {
                "cell": cell_mask,
                "cell_boundary": cell_boundary,
                "cell_matched_mask": cell_matched_mask,
                "cell_matched_boundary": cell_matched_boundary,
                "cell_outside_nucleus_mask": cell_outside_nucleus_mask,
                "nucleus": nucleus_mask,
                "nucleus_boundary": nucleus_boundary,
                "nucleus_matched_mask": nucleus_matched_mask,
                "nucleus_matched_boundary": nucleus_matched_boundary,
            }

            for name, mask in masks_to_save.items():
                if mask is None:
                    continue
                mask_path = os.path.join(
                    output_dir, f"{patch_base}.{name}.segmentations.ome.tiff"
                )
                if self._save_mask_stack(mask, name, mask_path, image_mpp):
                    exported = True

        return exported

    def _export_aggregated_masks(
        self,
        base_name: str,
        output_dir: str,
        metadata_df: pd.DataFrame,
        informative_indices: List[int],
        original_results: List[Dict],
        repaired_results: List[Dict],
        image_mpp: Optional[float],
    ) -> bool:
        channel_order = [
            ("cell", "cell"),
            ("nucleus", "nucleus"),
            ("cell_boundary", "cell_boundary"),
            ("nucleus_boundary", "nucleus_boundary"),
        ]
        matched_channel_order = [
            ("cell_matched_mask", "cell_matched_mask"),
            ("nucleus_matched_mask", "nucleus_matched_mask"),
            ("cell_outside_nucleus_mask", "cell_outside_nucleus_mask"),
            ("cell_matched_boundary", "cell_matched_boundary"),
            ("nucleus_matched_boundary", "nucleus_matched_boundary"),
        ]

        image_shape = self.codex_image.extended_extracted_channel_image.shape[:2]

        if len(repaired_results) != len(informative_indices):
            self.logger.warning(
                "Mismatch between repaired results (%d) and informative patches (%d)",
                len(repaired_results),
                len(informative_indices),
            )

        aggregated_masks: Dict[str, Optional[np.ndarray]] = {}

        for channel_key, result_key in channel_order:
            aggregated_masks[channel_key] = self._merge_segmentation_masks(
                metadata_df,
                informative_indices,
                original_results,
                result_key,
                image_shape,
            )

        for channel_key, result_key in matched_channel_order:
            aggregated_masks[channel_key] = self._merge_segmentation_masks(
                metadata_df,
                informative_indices,
                repaired_results,
                result_key,
                image_shape,
            )

        exported = False

        for name in [
            "cell",
            "cell_boundary",
            "cell_matched_mask",
            "cell_matched_boundary",
            "cell_outside_nucleus_mask",
            "nucleus",
            "nucleus_boundary",
            "nucleus_matched_mask",
            "nucleus_matched_boundary",
        ]:
            mask = aggregated_masks.get(name)
            if mask is None:
                continue
            mask_path = os.path.join(
                output_dir, f"{base_name}.{name}.segmentations.ome.tiff"
            )
            if self._save_mask_stack(mask, name, mask_path, image_mpp):
                exported = True

        return exported
