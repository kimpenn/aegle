# codex_image.py

import os
import sys
import logging
import gc
import numpy as np
import pandas as pd
import tifffile as tiff
import logging
import skimage.filters as filters


class CodexImage:
    def __init__(self, config, args):
        """
        Initialize the CodexImage object by loading image and antibody data.

        Args:
            config (dict): Configuration parameters.
            args (Namespace): Command-line arguments.
        """
        # Set up configuration and arguments
        self.config = config
        self.args = args

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

        # Initialize variables
        self.antibody_df = None
        self.target_channels_dict = {}

        self.all_channel_image = None
        self.extracted_channel_image = None
        self.extended_all_channel_image = None
        self.extended_extracted_channel_image = None

        self.img_height = None
        self.img_width = None
        self.img_height_extended = None
        self.img_width_extended = None
        self.n_channels = None

        self.logger.info(
            "Initializing CodexImage object with provided configuration and arguments."
        )
        self.load_data()
        self._configure_patching()

        self.logger.info(
            f"""
            Patching parameters set: 
                patch_height={self.patch_height}, patch_width={self.patch_width}, patch_overlap={self.patch_overlap}
                step_height={self.step_height}, step_width={self.step_width}
              """
        )
        self.logger.info(f"Image shape: {self.all_channel_image.shape}")
        self.logger.info(f"Loaded {self.antibody_df.shape[0]} antibodies.")
        self.logger.info(f"Target channels: {self.target_channels_dict}")
        # self.channel_stats_df = self.calculate_channel_statistics()
        # save channel stats to a file

    def _configure_patching(self):
        """Derive patching parameters from configuration, handling missing values."""
        patching_config = self.config.get("patching", {})
        self.split_mode = patching_config.get("split_mode", "patches")

        raw_patch_height = patching_config.get("patch_height")
        raw_patch_width = patching_config.get("patch_width")
        raw_overlap = patching_config.get("patch_overlap")
        if raw_overlap is None:
            raw_overlap = patching_config.get("overlap")

        patch_height = self._normalize_patch_dimension(raw_patch_height, "patch_height")
        patch_width = self._normalize_patch_dimension(raw_patch_width, "patch_width")
        patch_overlap = self._normalize_overlap(raw_overlap)

        # For modes that operate on the whole image, fall back to full dimensions.
        if self.split_mode in {"full_image", "halves", "quarters"}:
            if patch_height is None:
                self.logger.debug(
                    "patch_height not provided for %s mode; defaulting to image height %s",
                    self.split_mode,
                    self.img_height,
                )
            if patch_width is None:
                self.logger.debug(
                    "patch_width not provided for %s mode; defaulting to image width %s",
                    self.split_mode,
                    self.img_width,
                )
            patch_height = self.img_height
            patch_width = self.img_width
            patch_overlap = 0.0

        if patch_height is None or patch_width is None:
            error_msg = (
                "patch_height and patch_width must be positive integers when split_mode is 'patches'. "
                f"Received patch_height={raw_patch_height!r}, patch_width={raw_patch_width!r}, split_mode={self.split_mode!r}."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if patch_overlap is None:
            patch_overlap = 0.0

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_overlap = patch_overlap

        overlap_height = int(self.patch_height * self.patch_overlap)
        overlap_width = int(self.patch_width * self.patch_overlap)
        self.step_height = max(1, self.patch_height - overlap_height)
        self.step_width = max(1, self.patch_width - overlap_width)

    def _normalize_patch_dimension(self, value, name):
        """Return a positive integer for patch dimensions or None if unset."""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid {name} value: {value!r}") from exc
        if isinstance(value, (int, float)):
            if value <= 0:
                return None
            return int(value)
        raise TypeError(f"{name} must be numeric when provided; got {type(value).__name__}")

    def _normalize_overlap(self, value):
        """Validate overlap value and return a float in [0, 1)."""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid overlap value: {value!r}") from exc
        if not isinstance(value, (int, float)):
            raise TypeError(f"overlap must be numeric when provided; got {type(value).__name__}")
        if value < 0 or value >= 1:
            raise ValueError(f"overlap must be in the range [0, 1); received {value}")
        return float(value)

    def calculate_channel_statistics(self):
        """
        Calculate statistics (Min, Median, Max, 95%, Mean, Std Dev, Threshold) for each channel in the image.

        Returns:
            pd.DataFrame: DataFrame containing the calculated statistics for each channel.
        """
        self.logger.info("Calculating channel statistics...")
        stats = []

        for channel_idx in range(self.n_channels):
            self.logger.info(f"Calculating statistics for channel {channel_idx}...")
            channel_data = self.all_channel_image[:, :, channel_idx]
            self.logger.info("Calculating min...")
            min_val = np.min(channel_data)
            self.logger.info("Calculating median...")
            median_val = np.median(channel_data)
            self.logger.info("Calculating max...")
            max_val = np.max(channel_data)
            self.logger.info("Calculating 95th percentile...")
            percentile_95 = np.percentile(channel_data, 95)
            self.logger.info("Calculating mean...")
            mean_val = np.mean(channel_data)
            self.logger.info("Calculating standard deviation...")
            std_dev = np.std(channel_data)

            stats.append(
                {
                    "Channel": self.antibody_df["antibody_name"].iloc[channel_idx],
                    "Min": min_val,
                    "Median": median_val,
                    "Max": max_val,
                    "95%": percentile_95,
                    "Mean": mean_val,
                    "Std Dev": std_dev,
                }
            )

        stats_df = pd.DataFrame(stats)
        self.logger.info("Channel statistics calculated successfully.")
        file_name = os.path.join(self.args.out_dir, "channel_stats.csv")
        stats_df.to_csv(file_name, index=False)
        return stats_df

    def load_data(self):
        """
        Load image and antibodies data based on the provided configuration.
        """
        self.logger.info("Loading data...")
        data_config = self.config.get("data", {})
        file_name = data_config.get("file_name", "")
        antibodies_file = data_config.get("antibodies_file", "")

        # Construct full paths
        image_path = os.path.join(self.args.data_dir, file_name)
        antibodies_path = os.path.join(self.args.data_dir, antibodies_file)

        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found at {image_path}")
            sys.exit(1)
        if not os.path.exists(antibodies_path):
            self.logger.error(f"Antibodies file not found at {antibodies_path}")
            sys.exit(1)
        self.logger.info(f"Image path: {image_path}")
        self.logger.info(f"Antibodies path: {antibodies_path}")

        # Load image and antibodies data
        self.all_channel_image = self.load_image(image_path)
        self.antibody_df = self.load_antibodies(antibodies_path)

        # Set target channels
        self.set_target_channels()

    def load_image(self, image_path):
        """
        Load the image from the given path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array with channels last.
        """
        self.logger.info(f"Reading image from {image_path}")

        # Extract the BitsPerSample information from the first page
        with tiff.TiffFile(image_path) as tif:
            if 256 in tif.pages[0].tags:
                img_width = tif.pages[0].tags[256].value
                self.logger.info(f"ImageWidth: {img_width}")
            else:
                img_width = None
                self.logger.warning("ImageWidth tag not found.")
            if 257 in tif.pages[0].tags:
                img_height = tif.pages[0].tags[257].value
                self.logger.info(f"ImageLength: {img_height}")
            else:
                img_height = None
                self.logger.warning("ImageLength tag not found.")
            if 258 in tif.pages[0].tags:
                bits_per_sample = tif.pages[0].tags[258].value
                self.logger.info(f"BitsPerSample: {bits_per_sample}")
            else:
                bits_per_sample = None
                self.logger.warning("BitsPerSample tag not found.")
            if 277 in tif.pages[0].tags:
                samples_per_pixel = tif.pages[0].tags[277].value
                self.logger.info(f"SamplesPerPixel: {samples_per_pixel}")
            else:
                samples_per_pixel = None
                self.logger.warning("SamplesPerPixel tag not found.")

        image_ndarray = tiff.imread(image_path)
        # log the image data type
        self.logger.info(f"Image data type: {image_ndarray.dtype}")

        # image_ndarray = np.expand_dims(image_ndarray, axis=0)
        # Move the feature channels to the last axis
        image_ndarray = np.transpose(image_ndarray, (1, 2, 0))
        self.img_height, self.img_width, self.n_channels = (
            image_ndarray.shape[0],
            image_ndarray.shape[1],
            image_ndarray.shape[2],
        )
        self.tif_image_details = {
            "DataType": image_ndarray.dtype,
            "Shape": image_ndarray.shape,
            "ImageWidth": img_width,
            "ImageLength": img_height,
            "BitsPerSample": bits_per_sample,
            "SamplesPerPixel": samples_per_pixel,
        }
        self.logger.info(
            f"Image loaded successfully. Details: {self.tif_image_details}"
        )
        return image_ndarray

    def load_antibodies(self, antibodies_path):
        """
        Load the antibodies data from a TSV file.

        Args:
            antibodies_path (str): Path to the antibodies TSV file.

        Returns:
            pd.DataFrame: DataFrame containing antibodies information.
        """
        self.logger.info(f"Reading antibodies from {antibodies_path}")
        try:
            antibody_df = pd.read_csv(antibodies_path, sep="\t")
            self.logger.info(
                f"Antibodies loaded successfully. Total antibodies: {len(antibody_df)}"
            )
        except FileNotFoundError:
            self.logger.error(f"Antibodies file not found at {antibodies_path}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading antibodies file: {e}")
            sys.exit(1)
        return antibody_df

    def set_target_channels(self):
        """
        Set the target channels for nucleus and whole-cell staining based on the configuration.
        """
        self.logger.info("Setting target channels...")
        channels_config = self.config.get("channels", {})
        nucleus_channel = channels_config.get("nucleus_channel", "DAPI")
        wholecell_channel = channels_config.get("wholecell_channel", ["CD4"])
        if not isinstance(wholecell_channel, list):
            wholecell_channel = [wholecell_channel]

        target_channels = [nucleus_channel] + wholecell_channel
        for channel in target_channels:
            if channel not in self.antibody_df["antibody_name"].values:
                self.logger.error(
                    f"Channel {channel} not found in the antibodies file."
                )
                raise ValueError(f"Channel {channel} not found in the antibodies file.")

        self.target_channels_dict = {
            "nucleus": nucleus_channel,
            "wholecell": wholecell_channel,
        }
        self.logger.info(
            f"Target channels set successfully: {self.target_channels_dict}"
        )

    def extract_target_channels(self):
        """
        Preprocess the image by selecting the target channels.

        Returns:
            np.ndarray: Image data with only the target channels.
        """
        self.logger.info("Extracting target channels from image...")
        # Identify nucleus and wholecell channel names
        nucleus_channel = self.target_channels_dict["nucleus"]
        wholecell_channels = self.target_channels_dict["wholecell"]
        logging.info(f"Extracting nucleus channel: {nucleus_channel}")
        logging.info(
            f"Extracting wholecell channels with {len(wholecell_channels)} antibodies: {wholecell_channels}"
        )
        # Get the index for the nucleus channel
        nucleus_idx = self.antibody_df[
            self.antibody_df["antibody_name"] == nucleus_channel
        ].index.tolist()
        # Get the indices for the wholecell channels
        wcell_idx = self.antibody_df[
            self.antibody_df["antibody_name"].isin(wholecell_channels)
        ].index.tolist()

        # Ensure we actually found the nucleus channel and at least one wholecell channel
        if len(nucleus_idx) != 1 or len(wcell_idx) < 1:
            self.logger.error(
                f"Could not find all target channels. "
                f"nucleus: {nucleus_channel} (found {len(nucleus_idx)}) | "
                f"Wholecell: {wholecell_channels} (found {len(wcell_idx)})"
            )
            sys.exit(1)

        # Extract the nucleus image
        nucleus_image = self.all_channel_image[:, :, nucleus_idx[0]]

        # Extract the wholecell image(s)
        if len(wcell_idx) == 1:
            # If only one wholecell channel is provided, extract it directly
            self.logger.info(f"Extracting wholecell channel: {wholecell_channels[0]}")
            wcell_image = self.all_channel_image[:, :, wcell_idx[0]]
        else:
            # If multiple wholecell channels are specified, combine them via maximum projection
            self.logger.info(f"Extracting wholecell channels: {wholecell_channels}")
            wcell_image = np.zeros(
                (nucleus_image.shape[0], nucleus_image.shape[1]),
                dtype=nucleus_image.dtype,
            )
            for idx in wcell_idx:
                wcell_image = np.maximum(wcell_image, self.all_channel_image[:, :, idx])

        # Stack nucleus and wholecell channels along the third dimension
        # Should be the order of (nucleus, wholecell)
        # ref: https://github.com/vanvalenlab/deepcell-tf/blob/f1839a4eac03df1ceb17f28390bb7a1aa3f8dac8/deepcell/applications/mesmer.py#L176
        extracted_channel_image = np.stack((nucleus_image, wcell_image), axis=-1)

        self.logger.info(
            f"Target channels extracted successfully. Shape: {extracted_channel_image.shape}. Data type: {extracted_channel_image.dtype}"
        )
        self.extracted_channel_image = extracted_channel_image
        return extracted_channel_image

    # def extract_target_channels(self):
    #         """
    #         Preprocess the image by selecting the target channels.

    #         Returns:
    #             np.ndarray: Image data with only the target channels.
    #         """
    #         self.logger.info("Extracting target channels from image...")

    #         # Baseline method: Select the first wholecell channel and the nucleus channel
    #         # For deepcell, the first channel is the nucleus channel and the second is the wholecell channel
    #         target_channels = [self.target_channels_dict["nucleus"]] + [
    #             self.target_channels_dict["wholecell"][0]
    #         ]

    #         # rows in antibody_df must be in the same order as the channels in the image
    #         target_channel_idx = self.antibody_df[
    #             self.antibody_df["antibody_name"].isin(target_channels)
    #         ].index.tolist()

    #         if len(target_channel_idx) != len(target_channels):
    #             self.logger.error(f"Could not find all target channels: {target_channels}")
    #             sys.exit(1)

    #         # Extract target channels
    #         extracted_channel_image = self.all_channel_image[:, :, target_channel_idx]
    #         self.logger.info(
    #             f"Target channels extracted successfully. Shape: {extracted_channel_image.shape}"
    #         )
    #         self.extracted_channel_image = extracted_channel_image
    def extend_image(self):
        """
        Extend the image to ensure full coverage when cropping patches.
        """
        self.logger.info(
            "Extending image to ensure full coverage for cropping patches..."
        )
        pad_height = (
            self.patch_height - (self.img_height - self.patch_height) % self.step_height
        ) % self.patch_height
        pad_width = (
            self.patch_width - (self.img_width - self.patch_width) % self.step_width
        ) % self.patch_width

        self.logger.info(
            f"Padding dimensions calculated: pad_height={pad_height}, pad_width={pad_width}"
        )

        # self.extended_all_channel_image = np.pad(
        #     self.all_channel_image,
        #     ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
        #     mode="constant",
        #     constant_values=0,
        # )
        self.extended_all_channel_image = np.pad(
            self.all_channel_image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        self.logger.info(
            f"Extended all channel image shape: {self.extended_all_channel_image.shape}"
        )
        # self.extended_extracted_channel_image = np.pad(
        #     self.extracted_channel_image,
        #     ((0, 0), (0, pad_height), (0, pad_width), (0, 0)),
        #     mode="constant",
        #     constant_values=0,
        # )
        self.extended_extracted_channel_image = np.pad(
            self.extracted_channel_image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        self.logger.info(
            f"Extended extracted channel image shape: {self.extended_extracted_channel_image.shape}"
        )

        # Release references to the original arrays now that the extended
        # versions are materialised. This keeps only one copy of each large
        # tensor in memory before segmentation begins.
        self.all_channel_image = None
        self.extracted_channel_image = None
        gc.collect()
