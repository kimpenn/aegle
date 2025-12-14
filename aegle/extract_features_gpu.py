"""GPU-accelerated feature extraction using CuPy.

This module provides GPU-accelerated bincount operations for cell profiling,
offering 10-20x speedup over CPU for large samples (>1M cells).

GPU Image Processing Modes:
    - "auto": Automatically select based on available VRAM (recommended)
    - "full_gpu": Transfer entire (H,W,C) image to GPU for fast slicing
    - "chw_cpu": Pre-convert to channel-first format in RAM
    - "legacy": Use per-channel transfer with image_dict (backward compatible)
"""

import gc
import hashlib
import logging
import os
import time
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)

# Properties computed by regionprops_table
REGIONPROPS_PROPERTIES = (
    "centroid",
    "eccentricity",
    "perimeter",
    "convex_area",
    "area",
    "axis_major_length",
    "axis_minor_length",
    "solidity",
    "label",
)


def _compute_regionprops_cache_key(mask, properties, mask_type=""):
    """Generate a deterministic cache key for regionprops results.

    Args:
        mask: Segmentation mask array
        properties: Tuple of property names
        mask_type: Optional prefix (e.g., "nucleus" or "cell")

    Returns:
        str: Cache key based on mask shape, sample hash, and properties
    """
    # Sample first 1M elements for speed (deterministic)
    mask_flat = mask.ravel()
    sample_size = min(1_000_000, len(mask_flat))
    mask_sample = mask_flat[:sample_size]

    # Hash the sample
    mask_hash = hashlib.md5(mask_sample.tobytes()).hexdigest()[:12]

    # Hash the properties
    props_str = "_".join(sorted(properties))
    props_hash = hashlib.md5(props_str.encode()).hexdigest()[:8]

    prefix = f"{mask_type}_" if mask_type else ""
    return f"{prefix}regionprops_{mask.shape[0]}x{mask.shape[1]}_{mask_hash}_{props_hash}"


def _get_or_compute_regionprops(mask, properties, cache_dir=None, mask_type=""):
    """Load regionprops from cache or compute and cache.

    Args:
        mask: Segmentation mask array
        properties: Tuple of property names to compute
        cache_dir: Directory for cache files (None = no caching)
        mask_type: Type of mask for logging ("nucleus" or "cell")

    Returns:
        pd.DataFrame: Regionprops results
    """
    if cache_dir:
        cache_key = _compute_regionprops_cache_key(mask, properties, mask_type)
        cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached {mask_type} regionprops from {cache_path}")
            return pd.read_parquet(cache_path)

    # Compute regionprops (expensive)
    t0 = time.time()
    props = regionprops_table(mask, properties=properties)
    props_df = pd.DataFrame(props)
    elapsed = time.time() - t0
    logger.info(f"Computed {mask_type} regionprops in {elapsed:.1f}s")

    # Cache if directory provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        props_df.to_parquet(cache_path)
        logger.info(f"Cached {mask_type} regionprops to {cache_path}")

    return props_df


def extract_features_v2_gpu(
    image_input,
    nucleus_masks,
    channels_to_quantify,
    *,
    cell_masks=None,
    compute_laplacian=False,
    compute_cov=False,
    channel_dtype=np.float32,
    gpu_batch_size=0,
    gpu_image_mode="auto",
    cache_dir=None,
):
    """
    GPU-accelerated version of extract_features_v2_optimized using CuPy.

    Falls back to CPU version if GPU unavailable or encounters errors.

    Args:
        image_input: Either:
            - Dict mapping channel names to 2D image arrays (legacy mode)
            - 3D array (H, W, C) for optimized full_gpu or chw_cpu mode
        nucleus_masks: 2D array of nucleus segmentation labels
        channels_to_quantify: List of channel names to process
        cell_masks: 2D array of cell segmentation labels (optional)
        compute_laplacian: Whether to compute Laplacian variance (expensive)
        compute_cov: Whether to compute coefficient of variation
        channel_dtype: Data type for channel processing
        gpu_batch_size: Number of channels to process per GPU batch (0=auto)
        gpu_image_mode: Processing mode:
            - "auto": Select based on available VRAM (recommended)
            - "full_gpu": Transfer entire image to GPU (requires ~40GB VRAM)
            - "chw_cpu": Pre-convert to channel-first in RAM
            - "legacy": Use per-channel transfer with image_dict
        cache_dir: Directory for caching regionprops results (None = no caching)

    Returns:
        markers: DataFrame of cell-by-channel intensity values
        props_df: DataFrame of cell metadata (morphology, coordinates, etc.)

    Coordinate System:
        All centroid coordinates follow the standard image coordinate convention:
        - Origin (0, 0) at the top-left corner
        - centroid_x: column index (increases rightward)
        - centroid_y: row index (increases downward)

        This mapping is derived from scikit-image regionprops which returns
        centroids as (row, col), aliased here as (y, x).
    """
    # Check GPU availability
    from aegle.gpu_utils import (
        is_cupy_available,
        get_gpu_memory_info,
        estimate_gpu_batch_size,
        clear_gpu_memory,
        log_gpu_memory,
        can_fit_full_image_on_gpu,
    )

    # Detect input type: raw patch array or image_dict
    if isinstance(image_input, np.ndarray) and image_input.ndim == 3:
        patch_img = image_input
        image_dict = None
        is_raw_patch = True
        logger.info(f"Received raw patch with shape {patch_img.shape}")
    else:
        patch_img = None
        image_dict = image_input
        is_raw_patch = False
        logger.debug("Received image_dict (legacy mode)")

    if not is_cupy_available():
        logger.warning("GPU not available, falling back to CPU version")
        from aegle.extract_features import extract_features_v2_optimized

        # Build image_dict if we have raw patch
        if is_raw_patch:
            image_dict = {
                ch: patch_img[:, :, idx]
                for idx, ch in enumerate(channels_to_quantify)
            }

        return extract_features_v2_optimized(
            image_dict,
            nucleus_masks,
            channels_to_quantify,
            cell_masks=cell_masks,
            compute_laplacian=compute_laplacian,
            compute_cov=compute_cov,
            channel_dtype=channel_dtype,
        )

    import cupy as cp

    logger.info("Using GPU-accelerated feature extraction")
    gpu_start_time = time.time()

    # Log GPU memory before starting
    log_gpu_memory("GPU memory before extraction")

    channel_dtype = np.dtype(channel_dtype or np.float32)

    # Prepare masks (same as CPU version)
    nucleus_masks = np.asarray(nucleus_masks).squeeze()
    if cell_masks is None:
        cell_masks = nucleus_masks
    else:
        cell_masks = np.asarray(cell_masks).squeeze()

    nucleus_ids = np.unique(nucleus_masks)
    nucleus_ids = nucleus_ids[nucleus_ids != 0]
    nucleus_ids = nucleus_ids.astype(np.int64)

    logger.info(f"Number of labeled nuclei found: {len(nucleus_ids)}")

    if len(nucleus_ids) == 0:
        return (
            pd.DataFrame(columns=channels_to_quantify),
            pd.DataFrame(columns=["label"]),
        )

    # Handle duplicate channels (same as CPU version)
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

    if has_duplicates:
        logger.info("Original channels: %s", channels_to_quantify)
        logger.info("Unique channels: %s", unique_channels)
    else:
        logger.info("Channels: %s", channels_to_quantify)

    # Compute morphology (CPU - fast enough, complex to port to GPU)
    # With optional caching for faster testing/development
    logger.debug("Computing morphology features on CPU...")
    logger.info("[1]: np where to get nucleus_filterimg")
    nucleus_filterimg = np.where(
        np.isin(nucleus_masks, nucleus_ids), nucleus_masks, 0
    ).astype(np.int32)
    logger.info("[2]: regionprops_table to get nucleus_props")
    props_df = _get_or_compute_regionprops(
        nucleus_filterimg, REGIONPROPS_PROPERTIES, cache_dir, "nucleus"
    )
    props_df.set_index(props_df["label"], inplace=True)

    # Apply nucleus aliases
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

    # Cell morphology
    logger.info("[3]: np where to get cell_filterimg")
    cell_filterimg = np.where(
        np.isin(cell_masks, nucleus_ids), cell_masks, 0
    ).astype(np.int32)
    logger.info("[4]: regionprops_table to get cell_props")
    cell_props_df = _get_or_compute_regionprops(
        cell_filterimg, REGIONPROPS_PROPERTIES, cache_dir, "cell"
    )
    if not cell_props_df.empty:
        cell_props_df.set_index(cell_props_df["label"], inplace=True)
        cell_props_df = cell_props_df.reindex(props_df.index, fill_value=float("nan"))

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

    # Prepare label matrices
    nucleus_labels_matrix = (
        np.isin(nucleus_masks, nucleus_ids).astype(int) * nucleus_masks
    ).astype(np.int64)
    cell_labels_matrix = (
        np.isin(cell_masks, nucleus_ids).astype(int) * cell_masks
    ).astype(np.int64)

    # Pre-compute counts per label (on CPU - used for all channels)
    nucleus_count_per_label = np.bincount(
        nucleus_labels_matrix.ravel(), minlength=max_label + 1
    )[nucleus_ids]
    cell_count_per_label = np.bincount(
        cell_labels_matrix.ravel(), minlength=max_label + 1
    )[nucleus_ids]

    # === DETERMINE GPU IMAGE MODE ===
    n_channels = len(channels_to_quantify)
    n_nuclei = len(nucleus_ids)

    # Determine processing mode
    if gpu_image_mode == "auto" and is_raw_patch:
        can_fit, required_gb, available_gb = can_fit_full_image_on_gpu(
            patch_img.shape, n_nuclei, channel_dtype
        )
        if can_fit:
            gpu_image_mode = "full_gpu"
            logger.info(
                f"AUTO: Using full_gpu mode "
                f"({required_gb:.1f}GB required, {available_gb:.1f}GB available)"
            )
        else:
            gpu_image_mode = "chw_cpu"
            logger.info(
                f"AUTO: Using chw_cpu fallback "
                f"({required_gb:.1f}GB required > {available_gb:.1f}GB available)"
            )
    elif gpu_image_mode == "auto":
        gpu_image_mode = "legacy"
        logger.debug("Using legacy mode (image_dict provided)")
    elif gpu_image_mode in ("full_gpu", "chw_cpu") and not is_raw_patch:
        logger.warning(
            f"gpu_image_mode={gpu_image_mode} requires raw patch, "
            "but image_dict was provided. Falling back to legacy mode."
        )
        gpu_image_mode = "legacy"

    logger.info(f"GPU image mode: {gpu_image_mode}")

    # === GPU PROCESSING STARTS HERE ===
    logger.info("Transferring masks to GPU...")

    # Transfer masks to GPU (these stay on GPU for all batches)
    try:
        cell_labels_gpu = cp.asarray(cell_labels_matrix, dtype=cp.int64)
        nucleus_labels_gpu = cp.asarray(nucleus_labels_matrix, dtype=cp.int64)
        cell_labels_flat = cell_labels_gpu.ravel()
        nucleus_labels_flat = nucleus_labels_gpu.ravel()

        # Transfer nucleus_ids to GPU
        nucleus_ids_gpu = cp.asarray(nucleus_ids, dtype=cp.int64)

        logger.info("Masks transferred to GPU successfully")
        log_gpu_memory("GPU memory after transferring masks")

    except cp.cuda.memory.OutOfMemoryError as e:
        logger.error(f"GPU OOM when transferring masks: {e}")
        logger.warning("Falling back to CPU version")
        clear_gpu_memory()
        from aegle.extract_features import extract_features_v2_optimized
        return extract_features_v2_optimized(
            image_dict,
            nucleus_masks,
            channels_to_quantify,
            cell_masks=cell_masks,
            compute_laplacian=compute_laplacian,
            compute_cov=compute_cov,
            channel_dtype=channel_dtype,
        )

    # Pre-allocate output arrays (on CPU)
    wholecell_means = np.empty((n_nuclei, n_channels), dtype=channel_dtype)
    nucleus_means = np.empty_like(wholecell_means)
    cytoplasm_means = np.full_like(wholecell_means, np.nan)
    cov_values = None if not compute_cov else np.empty_like(wholecell_means)
    laplacian_variances = (
        None if not compute_laplacian else np.empty_like(wholecell_means)
    )

    # === PREPARE IMAGE DATA BASED ON MODE ===
    patch_img_gpu = None
    patch_chw = None

    if gpu_image_mode == "full_gpu":
        # Transfer entire image to GPU once
        logger.info("FULL_GPU: Transferring entire image to GPU...")
        t0 = time.time()
        patch_img_gpu = cp.asarray(patch_img, dtype=channel_dtype)
        transfer_time = time.time() - t0
        logger.info(f"FULL_GPU: Transferred image to GPU in {transfer_time:.1f}s")
        log_gpu_memory("After full image transfer")

    elif gpu_image_mode == "chw_cpu":
        # Pre-convert to channel-first format for contiguous slicing
        # Check if we have enough RAM for the conversion (~2x image size peak)
        image_size_gb = patch_img.nbytes / 1e9
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / 1e9
            if available_ram_gb < image_size_gb * 2.2:  # Need ~2.2x for safe transpose
                logger.warning(
                    f"CHW_CPU: Insufficient RAM for conversion "
                    f"(need ~{image_size_gb * 2.2:.1f}GB, have {available_ram_gb:.1f}GB). "
                    f"Falling back to legacy mode."
                )
                gpu_image_mode = "legacy"
                # Build image_dict for legacy mode
                image_dict = {
                    ch: patch_img[:, :, idx]
                    for idx, ch in enumerate(channels_to_quantify)
                }
            else:
                logger.info("CHW_CPU: Converting to channel-first format...")
                t0 = time.time()
                patch_chw = np.ascontiguousarray(
                    patch_img.transpose(2, 0, 1).astype(channel_dtype)
                )
                convert_time = time.time() - t0
                logger.info(f"CHW_CPU: Converted to CHW format in {convert_time:.1f}s")
                # Free original patch to save RAM
                del patch_img
                gc.collect()
        except ImportError:
            # psutil not available, try the conversion anyway
            logger.info("CHW_CPU: Converting to channel-first format...")
            t0 = time.time()
            patch_chw = np.ascontiguousarray(
                patch_img.transpose(2, 0, 1).astype(channel_dtype)
            )
            convert_time = time.time() - t0
            logger.info(f"CHW_CPU: Converted to CHW format in {convert_time:.1f}s")
            # Free original patch to save RAM
            del patch_img
            gc.collect()

    # Determine batch size for legacy mode
    if gpu_image_mode == "legacy":
        if gpu_batch_size == 0:
            gpu_batch_size = estimate_gpu_batch_size(
                n_channels=n_channels,
                image_shape=cell_masks.shape,
                n_cells=n_nuclei,
                dtype=channel_dtype,
            )
        n_batches = (n_channels + gpu_batch_size - 1) // gpu_batch_size
        logger.info(
            f"LEGACY: Processing {n_channels} channels in {n_batches} batches "
            f"of {gpu_batch_size}"
        )
    else:
        # For full_gpu and chw_cpu, process all channels in one pass
        gpu_batch_size = n_channels
        n_batches = 1
        logger.info(f"Processing {n_channels} channels in optimized mode")

    # === PROCESS CHANNELS ===
    for batch_idx in range(0, n_channels, gpu_batch_size):
        batch_end = min(batch_idx + gpu_batch_size, n_channels)
        batch_num = batch_idx // gpu_batch_size + 1

        if gpu_image_mode == "legacy":
            batch_channels = channels_to_quantify[batch_idx:batch_end]
            logger.info(
                f"Processing GPU batch {batch_num}/{n_batches} "
                f"(channels {batch_idx+1}-{batch_end})"
            )

        try:
            # Process each channel in the batch
            for rel_idx in range(batch_end - batch_idx):
                idx = batch_idx + rel_idx
                chan = channels_to_quantify[idx]

                # Get channel data based on mode
                if gpu_image_mode == "full_gpu":
                    # Slice on GPU - fast! (~3ms vs 180s CPU)
                    chan_data_gpu = patch_img_gpu[:, :, idx]
                    chan_data_flat = chan_data_gpu.ravel()
                    chan_data = None  # Not needed for GPU-only path

                elif gpu_image_mode == "chw_cpu":
                    # Slice from CHW array - already contiguous
                    chan_data = patch_chw[idx]
                    chan_data_gpu = cp.asarray(chan_data)
                    chan_data_flat = chan_data_gpu.ravel()

                else:  # legacy
                    # Load from image_dict and make contiguous (slow)
                    chan_data = np.asarray(
                        image_dict[chan], dtype=channel_dtype, order="C"
                    )
                    chan_data_gpu = cp.asarray(chan_data)
                    chan_data_flat = chan_data_gpu.ravel()

                # === Wholecell mean ===
                cell_sum_per_label_gpu = cp.bincount(
                    cell_labels_flat,
                    weights=chan_data_flat,
                    minlength=max_label + 1,
                )[nucleus_ids_gpu]

                # Transfer to CPU for division (avoid GPU overhead for small ops)
                cell_sum_per_label = cp.asnumpy(cell_sum_per_label_gpu)
                wholecell_means[:, idx] = np.divide(
                    cell_sum_per_label,
                    cell_count_per_label,
                    out=np.zeros_like(cell_sum_per_label, dtype=channel_dtype),
                    where=cell_count_per_label > 0,
                )

                # === Nucleus mean ===
                nucleus_sum_per_label_gpu = cp.bincount(
                    nucleus_labels_flat,
                    weights=chan_data_flat,
                    minlength=max_label + 1,
                )[nucleus_ids_gpu]

                nucleus_sum_per_label = cp.asnumpy(nucleus_sum_per_label_gpu)
                nucleus_means[:, idx] = np.divide(
                    nucleus_sum_per_label,
                    nucleus_count_per_label,
                    out=np.zeros_like(nucleus_sum_per_label, dtype=channel_dtype),
                    where=nucleus_count_per_label > 0,
                )

                # === Cytoplasm mean ===
                cytoplasm_counts = cell_count_per_label - nucleus_count_per_label
                valid_cytoplasm = cytoplasm_counts > 0
                cytoplasm_sums = cell_sum_per_label - nucleus_sum_per_label
                cytoplasm_channel_means = np.full_like(cytoplasm_sums, np.nan)
                cytoplasm_channel_means[valid_cytoplasm] = (
                    cytoplasm_sums[valid_cytoplasm] / cytoplasm_counts[valid_cytoplasm]
                )
                cytoplasm_means[:, idx] = cytoplasm_channel_means

                # === Coefficient of Variation (optional) ===
                if compute_cov:
                    chan_data_sq_gpu = chan_data_gpu**2
                    sum_sq_per_label_gpu = cp.bincount(
                        cell_labels_flat,
                        weights=chan_data_sq_gpu.ravel(),
                        minlength=max_label + 1,
                    )[nucleus_ids_gpu]

                    sum_sq_per_label = cp.asnumpy(sum_sq_per_label_gpu)
                    cell_mean_square = np.divide(
                        sum_sq_per_label,
                        cell_count_per_label,
                        out=np.zeros_like(sum_sq_per_label),
                        where=cell_count_per_label > 0,
                    )
                    std_dev_per_label = np.sqrt(
                        cell_mean_square - (wholecell_means[:, idx] ** 2)
                    )
                    cov_values[:, idx] = std_dev_per_label / (
                        wholecell_means[:, idx] + 1e-8
                    )

                    del chan_data_sq_gpu, sum_sq_per_label_gpu

                # === Laplacian variance (optional) ===
                if compute_laplacian:
                    # Fall back to CPU for Laplacian (cupyx.scipy.ndimage.laplace not widely available)
                    from skimage.filters import laplace

                    laplacian_img = np.abs(laplace(chan_data))
                    laplacian_sum_per_label = np.bincount(
                        cell_labels_matrix.ravel(),
                        weights=laplacian_img.ravel(),
                        minlength=max_label + 1,
                    )[nucleus_ids]

                    laplacian_variances[:, idx] = np.divide(
                        laplacian_sum_per_label,
                        cell_count_per_label,
                        out=np.zeros_like(
                            laplacian_sum_per_label, dtype=channel_dtype
                        ),
                        where=cell_count_per_label > 0,
                    )

                    del laplacian_img

                # Clean up GPU memory for this channel
                del chan_data_gpu, chan_data_flat
                del cell_sum_per_label_gpu, nucleus_sum_per_label_gpu

            # Note: Don't clear GPU memory pool here - masks stay on GPU for all batches
            # Memory cleanup happens once after all batches complete (line ~435)

        except cp.cuda.memory.OutOfMemoryError as e:
            logger.error(f"GPU OOM in batch {batch_num}: {e}")
            logger.warning("Falling back to CPU for remaining channels")

            # Clear GPU memory
            clear_gpu_memory()

            # Process remaining channels on CPU
            from aegle.extract_features import extract_features_v2_optimized

            # Build subset image dict for remaining channels
            remaining_channels = channels_to_quantify[batch_idx:]

            # Build image_dict for CPU fallback if we were in raw patch mode
            if is_raw_patch:
                if patch_chw is not None:
                    # CHW mode: convert back to dict
                    remaining_image_dict = {
                        ch: patch_chw[channels_to_quantify.index(ch) if hasattr(channels_to_quantify, 'index') else list(channels_to_quantify).index(ch)]
                        for ch in remaining_channels
                    }
                else:
                    # full_gpu mode: we need original patch_img but it might be deleted
                    logger.error("Cannot fall back to CPU in full_gpu mode after OOM")
                    raise RuntimeError("GPU OOM with no CPU fallback available")
            else:
                remaining_image_dict = {ch: image_dict[ch] for ch in remaining_channels}

            # Call CPU version for remaining channels
            remaining_markers, _ = extract_features_v2_optimized(
                remaining_image_dict,
                nucleus_masks,
                remaining_channels,
                cell_masks=cell_masks,
                compute_laplacian=compute_laplacian,
                compute_cov=compute_cov,
                channel_dtype=channel_dtype,
            )

            # Copy results to our arrays
            for rel_idx, chan in enumerate(remaining_channels):
                idx = batch_idx + rel_idx
                wholecell_means[:, idx] = remaining_markers[chan].values

            break  # Exit batch loop, CPU fallback handled the rest

    # Clean up GPU arrays and mode-specific data
    del cell_labels_gpu, nucleus_labels_gpu, cell_labels_flat, nucleus_labels_flat
    del nucleus_ids_gpu

    if patch_img_gpu is not None:
        del patch_img_gpu
    if patch_chw is not None:
        del patch_chw
        gc.collect()

    clear_gpu_memory()

    # === POST-PROCESSING (same as CPU version) ===

    # Convert to dataframes
    markers = pd.DataFrame(wholecell_means, index=nucleus_ids, columns=unique_channels)

    cov_df = (
        None
        if cov_values is None
        else pd.DataFrame(
            cov_values,
            index=nucleus_ids,
            columns=[f"{c}_cov" for c in unique_channels],
        )
    )

    lap_var_df = (
        None
        if laplacian_variances is None
        else pd.DataFrame(
            laplacian_variances,
            index=nucleus_ids,
            columns=[f"{c}_laplacian_var" for c in unique_channels],
        )
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

    # Compute entropy (vectorized - already on CPU)
    logger.debug("Computing cell entropy (vectorized)...")
    marker_array = markers.values
    row_sums = marker_array.sum(axis=1, keepdims=True)
    valid_mask = (row_sums > 0).ravel()

    cell_entropy = np.zeros(len(marker_array))
    if valid_mask.any():
        probs = np.zeros_like(marker_array)
        probs[valid_mask] = marker_array[valid_mask] / row_sums[valid_mask]
        log_probs = np.where(probs > 0, np.log2(probs), 0)
        cell_entropy[valid_mask] = -(probs[valid_mask] * log_probs[valid_mask]).sum(
            axis=1
        )

    cell_entropy_df = pd.DataFrame(
        cell_entropy, index=markers.index, columns=["cell_entropy"]
    )

    # Merge all metadata
    dfs_to_join = [
        cov_df,
        lap_var_df,
        cell_entropy_df,
        nucleus_intensity_df,
        cytoplasm_intensity_df,
    ]
    dfs_to_join = [df for df in dfs_to_join if df is not None]
    if dfs_to_join:
        props_df = props_df.join(dfs_to_join)
    props_df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    gpu_elapsed = time.time() - gpu_start_time
    logger.info(f"GPU feature extraction completed in {gpu_elapsed:.2f}s")

    # Log final GPU memory
    log_gpu_memory("GPU memory after extraction")

    return markers, props_df
