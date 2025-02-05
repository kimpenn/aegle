import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm
from skimage.filters import laplace
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from scipy.ndimage import labeled_comprehension
import logging


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


def extract_features_v2(
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
    markers : pd.DataFrame
        Expression matrix (mean intensity of each antibody per cell).
    props_df : pd.DataFrame
        Cell metadata (morphological features, CoV, Laplacian variance, spatial entropy).
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

    # Pre-allocate arrays for computed features
    mean_intensities = np.empty((len(nucleus_ids), len(channels_to_quantify)))
    cov_values = np.empty_like(mean_intensities)  # Coefficient of Variation
    laplacian_variances = np.empty_like(mean_intensities)  # Laplacian variance

    # For each channel, compute statistics for all labels using vectorized operations
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

        # Compute Coefficient of Variation (CoV)
        sum_sq_per_label = np.bincount(
            labels_matrix.ravel(), weights=(chan_data**2).ravel()
        )[nucleus_ids]
        std_dev_per_label = np.sqrt(
            sum_sq_per_label / count_per_label - (mean_intensities[:, idx] ** 2)
        )
        cov_values[:, idx] = std_dev_per_label / (
            mean_intensities[:, idx] + 1e-8
        )  # Avoid division by zero

        # Compute Laplacian-based spatial variance
        laplacian_img = np.abs(laplace(chan_data))
        laplacian_sum_per_label = np.bincount(
            labels_matrix.ravel(), weights=laplacian_img.ravel()
        )[nucleus_ids]
        laplacian_variances[:, idx] = laplacian_sum_per_label / count_per_label

    # Convert mean intensities into the expression matrix (markers)
    markers = pd.DataFrame(
        mean_intensities, index=nucleus_ids, columns=channels_to_quantify
    )

    # Convert computed spatial features into metadata
    cov_df = pd.DataFrame(
        cov_values,
        index=nucleus_ids,
        columns=[f"{c}_cov" for c in channels_to_quantify],
    )
    lap_var_df = pd.DataFrame(
        laplacian_variances,
        index=nucleus_ids,
        columns=[f"{c}_laplacian_var" for c in channels_to_quantify],
    )

    pixel_entropy_df = compute_pixel_entropy_vectorized(
        image_dict, segmentation_masks, channels_to_quantify, nucleus_ids
    )

    # Compute antibody entropy per cell (distribution across channels)
    def compute_cell_entropy(row):
        if np.sum(row) == 0:
            return 0  # Avoid log(0) errors
        probs = row / np.sum(row)  # Normalize to probability distribution
        return entropy(probs, base=2)  # Shannon entropy with base 2

    cell_entropy_df = pd.DataFrame(
        markers.apply(compute_cell_entropy, axis=1), columns=["cell_entropy"]
    )

    # Merge all metadata into props_df
    props_df = props_df.join([cov_df, lap_var_df, pixel_entropy_df, cell_entropy_df])

    # Rename columns
    props_df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    # Export to CSV (optional)
    # markers.to_csv(output_file.replace('.csv', '_expression.csv'))
    # props_df.to_csv(output_file.replace('.csv', '_metadata.csv'))

    return markers, props_df


def compute_pixel_entropy_vectorized(
    image_dict, segmentation_masks, channels_to_quantify, nucleus_ids
):
    """
    Compute spatial entropy of pixel intensities per cell using a vectorized approach.

    Parameters
    ----------
    image_dict : dict
        Dictionary containing image data. Keys are channel names, values are 2D numpy arrays.
    segmentation_masks : ndarray
        2D numpy array containing segmentation masks.
    channels_to_quantify : list
        List of channel names to quantify.
    nucleus_ids : ndarray
        List of unique cell IDs (excluding background).

    Returns
    -------
    pixel_entropy_df : pd.DataFrame
        DataFrame with spatial entropy values per cell.
    """
    num_channels = len(channels_to_quantify)
    mask_flat = segmentation_masks.ravel()

    # Stack all channels into a single 3D array: (height, width, num_channels)
    stacked_image = np.dstack([image_dict[chan] for chan in channels_to_quantify])

    # Flatten into (num_pixels, num_channels)
    stacked_image_flat = stacked_image.reshape(-1, num_channels)

    # Filter out background pixels (segmentation_masks == 0)
    valid_pixels = mask_flat > 0
    mask_flat = mask_flat[valid_pixels]
    stacked_image_flat = stacked_image_flat[valid_pixels]

    # Compute per-pixel probability distributions per cell
    unique_cells = np.unique(mask_flat)
    pixel_distributions = np.zeros((unique_cells.shape[0], 1))
    print(f"unique_cells: {unique_cells.shape}")
    print(f"pixel_distributions: {pixel_distributions.shape}")
    for i, cell_id in enumerate(
        tqdm(unique_cells, desc="Computing pixel distributions")
    ):
        pixel_values = stacked_image_flat[mask_flat == cell_id]
        pixel_sum = (
            np.sum(pixel_values, axis=1, keepdims=True) + 1e-8
        )  # Avoid division by zero
        pixel_probs = pixel_values / pixel_sum  # Normalize per pixel
        pixel_distributions[i] = np.mean(entropy(pixel_probs.T, base=2, axis=0))

    print(f"pixel_distributions: {pixel_distributions.shape}")
    # Convert to DataFrame
    pixel_entropy_df = pd.DataFrame(
        pixel_distributions, index=unique_cells, columns=["pixel_entropy"]
    )
    return pixel_entropy_df
