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


def extract_features_v2_optimized(
    image_dict,
    nucleus_masks,
    channels_to_quantify,
    *,
    cell_masks=None,
):
    """Compute marker intensities and morphology statistics for matched cells."""
    logger = logging.getLogger(__name__)

    nucleus_masks = np.asarray(nucleus_masks).squeeze()
    if cell_masks is None:
        cell_masks = nucleus_masks
    else:
        cell_masks = np.asarray(cell_masks).squeeze()

    # Identify all nuclei (labels > 0) and exclude background (label=0)
    nucleus_ids = np.unique(nucleus_masks)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    nucleus_ids = nucleus_ids.astype(np.int64)

    logger.info("Number of labeled nuclei found: %d", len(nucleus_ids))

    if len(nucleus_ids) == 0:
        return (
            pd.DataFrame(columns=channels_to_quantify),
            pd.DataFrame(columns=["label"]),
        )

    # Create unique channel names by adding suffixes for duplicates
    unique_channels = []
    channel_counts = {}
    has_duplicates = False
    
    for channel in channels_to_quantify:
        if channel in channel_counts:
            logger.warning(f"Duplicate channel name found: {channel}")
            channel_counts[channel] += 1
            unique_name = f"{channel}_{channel_counts[channel]}"
            has_duplicates = True
        else:
            channel_counts[channel] = 0
            unique_name = channel
        unique_channels.append(unique_name)
    
    # Log channels - show original vs unique only if duplicates exist
    if has_duplicates:
        logger.info("Original channels: %s", channels_to_quantify)
        logger.info("Unique channels: %s", unique_channels)
    else:
        logger.info("Channels: %s", channels_to_quantify)

    nucleus_filterimg = np.where(
        np.isin(nucleus_masks, nucleus_ids), nucleus_masks, 0
    ).astype(np.int32)

    # Extract nucleus morphology metrics (kept for backward compatibility)
    nucleus_props = regionprops_table(
        nucleus_filterimg,
        properties=(
            "centroid",
            "eccentricity",
            "perimeter",
            "convex_area",
            "area",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "label",
        ),
    )
    props_df = pd.DataFrame(nucleus_props)
    props_df.set_index(props_df["label"], inplace=True)

    nucleus_aliases = {
        "eccentricity": "nucleus_eccentricity",
        "perimeter": "nucleus_perimeter",
        "convex_area": "nucleus_convex_area",
        "area": "nucleus_area",
        "axis_major_length": "nucleus_axis_major_length",
        "axis_minor_length": "nucleus_axis_minor_length",
        "solidity": "nucleus_solidity",
        "centroid-0": "nucleus_centroid_y",
        "centroid-1": "nucleus_centroid_x",
    }
    for legacy_name, alias in nucleus_aliases.items():
        if legacy_name in props_df.columns:
            props_df[alias] = props_df[legacy_name]

    cell_filterimg = np.where(
        np.isin(cell_masks, nucleus_ids), cell_masks, 0
    ).astype(np.int32)

    cell_props = regionprops_table(
        cell_filterimg,
        properties=(
            "centroid",
            "eccentricity",
            "perimeter",
            "convex_area",
            "area",
            "axis_major_length",
            "axis_minor_length",
            "solidity",
            "label",
        ),
    )
    cell_props_df = pd.DataFrame(cell_props)
    if not cell_props_df.empty:
        cell_props_df.set_index(cell_props_df["label"], inplace=True)
        cell_props_df = cell_props_df.reindex(props_df.index, fill_value=float('nan'))
        cell_aliases = {
            "eccentricity": "cell_eccentricity",
            "perimeter": "cell_perimeter",
            "convex_area": "cell_convex_area",
            "area": "cell_area",
            "axis_major_length": "cell_axis_major_length",
            "axis_minor_length": "cell_axis_minor_length",
            "solidity": "cell_solidity",
            "centroid-0": "cell_centroid_y",
            "centroid-1": "cell_centroid_x",
        }
        for column, alias in cell_aliases.items():
            if column in cell_props_df.columns:
                props_df[alias] = cell_props_df[column]

    max_label = int(max(cell_masks.max(), nucleus_masks.max()))

    nucleus_labels_matrix = (
        np.isin(nucleus_masks, nucleus_ids).astype(int) * nucleus_masks
    ).astype(np.int64)
    cell_labels_matrix = (
        np.isin(cell_masks, nucleus_ids).astype(int) * cell_masks
    ).astype(np.int64)

    # Pre-compute counts per label (used for all channels)
    nucleus_count_per_label = np.bincount(
        nucleus_labels_matrix.ravel(), minlength=max_label + 1
    )[nucleus_ids]
    cell_count_per_label = np.bincount(
        cell_labels_matrix.ravel(), minlength=max_label + 1
    )[nucleus_ids]

    # Pre-allocate arrays for computed features
    n_nuclei = len(nucleus_ids)
    n_channels = len(channels_to_quantify)
    wholecell_means = np.empty((n_nuclei, n_channels))
    nucleus_means = np.empty_like(wholecell_means)
    cytoplasm_means = np.full_like(wholecell_means, np.nan)
    cov_values = np.empty_like(wholecell_means)
    laplacian_variances = np.empty_like(wholecell_means)

    # *** KEY OPTIMIZATION 2: Process in smaller batches to reduce memory pressure ***
    # Determine batch size based on image dimensions - adjust as needed
    # For very large images, processing a subset of channels at once can help
    batch_size = max(1, n_channels // 4)  # Process 1/4 of channels at a time

    for batch_idx in range(0, n_channels, batch_size):
        batch_end = min(batch_idx + batch_size, n_channels)
        batch_channels = channels_to_quantify[batch_idx:batch_end]

        logger.debug(
            "Processing channel batch %d/%d (channels %d-%d)",
            batch_idx // batch_size + 1,
            (n_channels + batch_size - 1) // batch_size,
            batch_idx + 1,
            batch_end,
        )

        for rel_idx, chan in enumerate(batch_channels):
            idx = batch_idx + rel_idx
            logger.debug("Processing channel %s (%d/%d)", chan, idx + 1, n_channels)

            chan_data = image_dict[chan]

            cell_sum_per_label = np.bincount(
                cell_labels_matrix.ravel(),
                weights=chan_data.ravel(),
                minlength=max_label + 1,
            )[nucleus_ids]
            wholecell_means[:, idx] = np.divide(
                cell_sum_per_label,
                cell_count_per_label,
                out=np.zeros_like(cell_sum_per_label),
                where=cell_count_per_label > 0,
            )

            nucleus_sum_per_label = np.bincount(
                nucleus_labels_matrix.ravel(),
                weights=chan_data.ravel(),
                minlength=max_label + 1,
            )[nucleus_ids]
            nucleus_means[:, idx] = np.divide(
                nucleus_sum_per_label,
                nucleus_count_per_label,
                out=np.zeros_like(nucleus_sum_per_label),
                where=nucleus_count_per_label > 0,
            )

            cytoplasm_counts = cell_count_per_label - nucleus_count_per_label
            valid_cytoplasm = cytoplasm_counts > 0
            cytoplasm_sums = cell_sum_per_label - nucleus_sum_per_label
            cytoplasm_channel_means = np.full_like(cytoplasm_sums, np.nan)
            cytoplasm_channel_means[valid_cytoplasm] = (
                cytoplasm_sums[valid_cytoplasm]
                / cytoplasm_counts[valid_cytoplasm]
            )
            cytoplasm_means[:, idx] = cytoplasm_channel_means

            # Compute Coefficient of Variation (CoV)
            sum_sq_per_label = np.bincount(
                cell_labels_matrix.ravel(),
                weights=(chan_data**2).ravel(),
                minlength=max_label + 1,
            )[nucleus_ids]

            cell_mean_square = np.divide(
                sum_sq_per_label,
                cell_count_per_label,
                out=np.zeros_like(sum_sq_per_label),
                where=cell_count_per_label > 0,
            )
            std_dev_per_label = np.sqrt(
                cell_mean_square - (wholecell_means[:, idx] ** 2)
            )
            cov_values[:, idx] = std_dev_per_label / (wholecell_means[:, idx] + 1e-8)

            # *** OPTIMIZATION 4: Optimize Laplacian calculation ***
            laplacian_img = np.abs(laplace(chan_data))
            laplacian_sum_per_label = np.bincount(
                cell_labels_matrix.ravel(),
                weights=laplacian_img.ravel(),
                minlength=max_label + 1,
            )[nucleus_ids]
            laplacian_variances[:, idx] = np.divide(
                laplacian_sum_per_label,
                cell_count_per_label,
                out=np.zeros_like(laplacian_sum_per_label),
                where=cell_count_per_label > 0,
            )

            del laplacian_img

    # Convert to dataframes using unique channel names
    markers = pd.DataFrame(
        wholecell_means, index=nucleus_ids, columns=unique_channels
    )

    cov_df = pd.DataFrame(
        cov_values,
        index=nucleus_ids,
        columns=[f"{c}_cov" for c in unique_channels],
    )

    lap_var_df = pd.DataFrame(
        laplacian_variances,
        index=nucleus_ids,
        columns=[f"{c}_laplacian_var" for c in unique_channels],
    )

    nucleus_intensity_df = pd.DataFrame(
        nucleus_means,
        index=nucleus_ids,
        columns=[f"{c}_nucleus_mean" for c in unique_channels],
    )

    cytoplasm_intensity_df = pd.DataFrame(
        cytoplasm_means,
        index=nucleus_ids,
        columns=[f"{c}_cytoplasm_mean" for c in unique_channels],
    )

    # Compute antibody entropy per cell
    def compute_cell_entropy(row):
        if np.sum(row) == 0:
            return 0
        probs = row / np.sum(row)
        return entropy(probs, base=2)

    cell_entropy_df = pd.DataFrame(
        markers.apply(compute_cell_entropy, axis=1), columns=["cell_entropy"]
    )

    # Merge all metadata
    props_df = props_df.join(
        [
            cov_df,
            lap_var_df,
            cell_entropy_df,
            nucleus_intensity_df,
            cytoplasm_intensity_df,
        ]
    )
    props_df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    return markers, props_df


def extract_features_v2(
    image_dict,
    nucleus_masks,
    channels_to_quantify,
    *,
    cell_masks=None,
):
    """Backward-compatible wrapper around the optimized feature extractor."""
    return extract_features_v2_optimized(
        image_dict, nucleus_masks, channels_to_quantify, cell_masks=cell_masks
    )

