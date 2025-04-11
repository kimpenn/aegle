# data_loading.py

import os
import logging
import sys
import numpy as np
import pandas as pd
import tifffile as tiff


def load_data(config, args):
    """
    Load image and antibodies data.

    Args:
        config (dict): Configuration parameters.
        args (Namespace): Command-line arguments.

    Returns:
        image_ndarray (np.ndarray): Loaded image data.
        antibody_df (pd.DataFrame): DataFrame containing antibody information.
        target_channels_dict (dict): Dictionary containing nucleus and wholecell channels.
    """
    data_config = config.get("data", {})
    file_name = data_config.get("file_name", "NW_Ovary_16/Scan1/NW_1_Scan1_dev.qptiff")
    antibodies_file = data_config.get(
        "antibodies_file", "NW_Ovary_16/Scan1/extras/antibodies.tsv"
    )

    # Construct full paths
    image_path = os.path.join(args.data_dir, file_name)
    antibodies_path = os.path.join(args.data_dir, antibodies_file)

    # Load image and antibodies data
    image_ndarray = load_image(image_path)
    antibody_df = load_antibodies(antibodies_path)

    channels_config = config.get("channels", {})
    nucleus_channel = channels_config.get("nucleus_channel", "DAPI")
    wholecell_channel = channels_config.get("wholecell_channel", ["CD4"])
    if not isinstance(wholecell_channel, list):
        wholecell_channel = [wholecell_channel]

    target_channels = [nucleus_channel] + wholecell_channel
    for channel in target_channels:
        if channel not in antibody_df["antibody_name"].values:
            raise ValueError(f"Channel {channel} not found in the antibodies file.")

    target_channels_dict = {
        "nucleus": nucleus_channel,
        "wholecell": wholecell_channel,
    }

    return image_ndarray, antibody_df, target_channels_dict


def load_image(image_path):
    """
    Load the image from the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a NumPy array with channels last.
    """
    logging.info(f"Reading image from {image_path}")
    image_ndarray = tiff.imread(image_path)
    image_ndarray = np.expand_dims(image_ndarray, axis=0)
    # Move the feature channels to the last axis
    image_ndarray = np.transpose(image_ndarray, (0, 2, 3, 1))
    tif_image_details = {"Data Type": image_ndarray.dtype, "Shape": image_ndarray.shape}
    logging.info(f"Image details: {tif_image_details}")
    return image_ndarray


def load_antibodies(antibodies_path):
    """
    Load the antibodies data from a TSV file.

    Args:
        antibodies_path (str): Path to the antibodies TSV file.

    Returns:
        pd.DataFrame: DataFrame containing antibodies information.
    """
    logging.info(f"Reading antibodies from {antibodies_path}")
    antibody_df = pd.read_csv(antibodies_path, sep="\t")
    return antibody_df
