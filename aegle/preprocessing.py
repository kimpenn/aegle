# preprocessing.py

import logging
import numpy as np
import pandas as pd
import sys


def preprocess_image(image_ndarray, antibody_df, target_channels_dict):
    """
    Preprocess the image by selecting target channels.

    Args:
        image_ndarray (np.ndarray): Loaded image data.
        antibody_df (pd.DataFrame): DataFrame containing antibody information.
        target_channels_dict (dict): Dictionary containing nuclear and wholecell channels.

    Returns:
        image_ndarray_target_channel (np.ndarray): Image with selected channels.
    """

    # TODO: Handle multiple antibodies for wholecell_channel if needed
    image_ndarray_target_channel, target_channel_idx = select_channels(
        image_ndarray, antibody_df, target_channels_dict
    )
    return image_ndarray_target_channel


def select_channels(image_ndarray, antibody_df, target_channels_dict):
    """
    Select the target channels from the image ndarray.

    Args:
        image_ndarray (np.ndarray): Image data.
        antibody_df (pd.DataFrame): Antibody DataFrame.
        target_channels_dict (dict): Dictionary containing nuclear and wholecell channels.

    Returns:
        np.ndarray: Image data with only the target channels.
        list: Indices of the selected channels.
    """
    # Baseline method: Select the first wholecell channel and the nuclear channel
    target_channels = [target_channels_dict["nuclear"]] + [
        target_channels_dict["wholecell"][0]
    ]

    # rows in antibody_df must be in the same order as the channels in the image
    target_channel_idx = antibody_df[
        antibody_df["antibody_name"].isin(target_channels)
    ].index.tolist()

    if len(target_channel_idx) != len(target_channels):
        logging.error(f"Could not find all target channels: {target_channels}")
        sys.exit(1)

    # Extract target channels
    image_ndarray_target_channel = image_ndarray[:, :, :, target_channel_idx]
    logging.info(f"Selected channels shape: {image_ndarray_target_channel.shape}")
    return image_ndarray_target_channel, target_channel_idx
