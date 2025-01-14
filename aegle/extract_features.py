import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm


def extract_features(
    image_dict, segmentation_masks, channels_to_quantify, output_file, size_cutoff=0
):
    """
    Extract features from the given image dictionary and segmentation masks.

    Parameters
    ----------
    image_dict : dict
        Dictionary containing image data. Keys are channel names, values are 2D numpy arrays.
    segmentation_masks : ndarray
        2D numpy array containing segmentation masks.
    channels_to_quantify : list
        List of channel names to quantify.
    output_file : str
        Path to the output CSV file.
    size_cutoff : int, optional
        Minimum size of nucleus to consider. Nuclei smaller than this are ignored. Default is 0.

    Returns
    -------
    None
        The function doesn't return anything but writes the extracted features to a CSV file.

    """
    segmentation_masks = segmentation_masks.squeeze()

    # Count pixels for each nucleus
    _, counts = np.unique(segmentation_masks, return_counts=True)

    # Identify nucleus IDs above the size cutoff, excluding background (ID 0)
    nucleus_ids = np.where(counts > size_cutoff)[0][1:]

    # Filter out small objects from segmentation masks
    filterimg = np.where(
        np.isin(segmentation_masks, nucleus_ids), segmentation_masks, 0
    ).astype(np.int32)

    # Extract morphological features
    props = regionprops_table(
        filterimg,
        properties=(
            "centroid",
            "eccentricity",
            "perimeter",
            "convex_area",
            "area",
            "axis_major_length",
            "axis_minor_length",
            "label",
        ),
    )
    props_df = pd.DataFrame(props)
    props_df.set_index(props_df["label"], inplace=True)

    # Pre-allocate array for mean intensities
    mean_intensities = np.empty((len(nucleus_ids), len(channels_to_quantify)))

    # For each channel, compute mean intensities for all labels using vectorized operations
    for idx, chan in enumerate(tqdm(channels_to_quantify, desc="Processing channels")):
        chan_data = image_dict[chan]
        labels_matrix = (
            np.isin(segmentation_masks, nucleus_ids).astype(int) * segmentation_masks
        ).astype(np.int64)
        sum_per_label = np.bincount(labels_matrix.ravel(), weights=chan_data.ravel())[
            nucleus_ids
        ]
        count_per_label = np.bincount(labels_matrix.ravel())[nucleus_ids]
        mean_intensities[:, idx] = sum_per_label / count_per_label

    # Convert the array to a DataFrame
    mean_df = pd.DataFrame(
        mean_intensities, index=nucleus_ids, columns=channels_to_quantify
    )

    # Join with morphological features
    markers = mean_df.join(props_df)

    # rename column
    markers.rename(columns={"centroid-0": "y"}, inplace=True)
    markers.rename(columns={"centroid-1": "x"}, inplace=True)

    # Export to CSV
    # markers.to_csv(output_file)

    return markers, props_df
