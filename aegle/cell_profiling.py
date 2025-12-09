import os
import logging
import pandas as pd
import numpy as np

from aegle.extract_features import extract_features_v2_optimized
from aegle.extract_features_gpu import extract_features_v2_gpu
from aegle.gpu_utils import is_cupy_available

# Create a logger specific to this module
logger = logging.getLogger(__name__)


def run_cell_profiling(codex_patches, config, args):
    """
    Runs cell-level profiling on each patch, generating a cell-by-marker matrix
    and a cell metadata table. For split modes 'full_image', 'halves', or 'quarters',
    results are merged into single files. For 'patches' mode, separate files per patch are saved.

    Args:
        codex_patches (CodexPatches): The patched CODEX data, with segmentation results.
        config (dict): The pipeline configuration.
        args (Namespace): Command-line arguments with paths and other settings.
    """
    logger.info("----- Running cell profiling.")

    # Retrieve the antibody_df directly from codex_patches
    antibody_df = codex_patches.antibody_df
    # Convert to a simple list of antibody names
    antibodies = antibody_df["antibody_name"].values

    # Create a directory for cell profiling outputs
    profiling_out_dir = os.path.join(args.out_dir, "cell_profiling")
    os.makedirs(profiling_out_dir, exist_ok=True)

    # Get split mode from config to determine output strategy
    patching_config = config.get("patching", {})
    split_mode = patching_config.get("split_mode", "patches")
    logger.info(f"Split mode: {split_mode}")

    profiling_config = config.get("profiling", {})
    features_config = profiling_config.get("features", {})
    compute_laplacian = features_config.get("compute_laplacian", False)
    compute_cov = features_config.get("compute_cov", False)
    channel_dtype = features_config.get("channel_dtype", np.float32)

    # GPU configuration
    use_gpu = features_config.get("use_gpu", False)
    gpu_batch_size = features_config.get("gpu_batch_size", 0)
    gpu_available = is_cupy_available()

    if use_gpu and not gpu_available:
        logger.warning("GPU requested but not available, using CPU version")
        use_gpu = False

    logger.info(
        "Profiling features: compute_laplacian=%s, compute_cov=%s, channel_dtype=%s, use_gpu=%s",
        compute_laplacian,
        compute_cov,
        channel_dtype,
        use_gpu,
    )

    # Get patches metadata and select only informative patches
    patches_metadata_df = codex_patches.get_patches_metadata()
    logger.info(f"Patches metadata: {patches_metadata_df}")
    # save patches metadata to csv
    patches_metadata_df.to_csv(os.path.join(profiling_out_dir, "patches_metadata.csv"), index=False)
    informative_idx = patches_metadata_df["is_informative"] == True
    
    # Get indices of informative patches
    informative_patches = [i for i, is_informative in enumerate(informative_idx) if is_informative]
    
    total_patches = len(informative_patches)
    logger.info(f"Profiling {total_patches} informative patches out of {len(patches_metadata_df)} total patches.")
    
    # Determine logging frequency based on total patches
    if total_patches <= 10:
        log_frequency = 1  # Log every patch for small datasets
    elif total_patches <= 100:
        log_frequency = 10  # Log every 10th patch
    else:
        log_frequency = max(1, total_patches // 20)  # Log ~20 times during the process
    
    # Initialize counters for summary statistics
    total_cells = 0
    processed = 0
    global_cell_offset = 0  # Tracks cumulative global cell IDs when masks are merged

    # Pre-compute expected cells for progress logging
    total_expected_cells = 0
    for seg_result in codex_patches.repaired_seg_res_batch:
        cell_mask = seg_result.get("cell_matched_mask")
        if cell_mask is None:
            continue
        labels = np.unique(cell_mask)
        total_expected_cells += (labels != 0).sum()

    next_progress_fraction = 0.05

    # Initialize lists for merging results (used for full_image, halves, quarters modes)
    all_exp_dfs = []
    all_metadata_dfs = []
    all_overview_dfs = []
    all_nucleus_overview_dfs = []
    should_merge = split_mode in ["full_image", "halves", "quarters"]

    # Process only informative patches
    for i, patch_idx in enumerate(informative_patches):
        # Get segmentation masks from repaired segmentation
        seg_batch_idx = i  # Index in the repaired_seg_res_batch corresponds to position in informative patches list
        seg_result = codex_patches.repaired_seg_res_batch[seg_batch_idx]

        nucleus_mask = seg_result.get("nucleus_matched_mask")
        cell_mask = seg_result.get("cell_matched_mask")
        if nucleus_mask is None or cell_mask is None:
            logger.warning("Skipping patch %s due to missing matched masks", patch_idx)
            continue

        patch_max_label = int(cell_mask.max()) if cell_mask is not None else 0

        # Get the full multi-channel patch from all channels (or extracted channels)
        patch_img = codex_patches.get_all_channel_patch(patch_idx)
        if codex_patches.is_using_disk_based_patches():
            logger.info(
                f"Loaded patch {patch_idx} from disk for profiling with shape: {patch_img.shape}"
            )

        # Build the image_dict for extract_features_v2
        # Each entry is {antibody_name: 2D image}
        image_dict = {}
        for channel_idx, ab in enumerate(antibodies):
            image_dict[ab] = patch_img[:, :, channel_idx]

        # Extract features (exp_df: cell-by-marker, metadata_df: cell metadata)
        if use_gpu:
            exp_df, metadata_df = extract_features_v2_gpu(
                image_dict,
                nucleus_mask,
                antibodies,
                cell_masks=cell_mask,
                compute_laplacian=compute_laplacian,
                compute_cov=compute_cov,
                channel_dtype=channel_dtype,
                gpu_batch_size=gpu_batch_size,
            )
        else:
            exp_df, metadata_df = extract_features_v2_optimized(
                image_dict,
                nucleus_mask,
                antibodies,
                cell_masks=cell_mask,
                compute_laplacian=compute_laplacian,
                compute_cov=compute_cov,
                channel_dtype=channel_dtype,
            )
        logger.info(f"Extracted features for patch {patch_idx} with shape: {exp_df.shape}")
        logger.info(f"Metadata for patch {patch_idx} with shape: {metadata_df.shape}")
        logger.info(f"Exp DataFrame: {exp_df.head()}")
        logger.info(f"Metadata DataFrame: {metadata_df.head()}")
        
        # Add patch information to track origin of cells
        if len(exp_df) > 0:  # Only add columns if there are cells
            if patch_max_label > 0:
                if "label" in metadata_df.columns:
                    local_ids = pd.to_numeric(metadata_df["label"], errors="coerce")
                    local_ids.index = metadata_df.index
                else:
                    local_ids = pd.Series(
                        metadata_df.index.to_numpy(),
                        index=metadata_df.index,
                        dtype=np.int64,
                    )

                local_ids = local_ids.fillna(0).astype(np.int64)
                mask_ids = (local_ids + global_cell_offset).astype(np.int64)

                metadata_df.insert(0, "cell_mask_id", mask_ids.to_numpy())
                exp_df.insert(0, "cell_mask_id", mask_ids.reindex(exp_df.index).to_numpy())

            exp_df["patch_id"] = patch_idx
            metadata_df["patch_id"] = patch_idx

            # Create unique cell IDs across patches
            # Check if cell_id column exists, if not create it from index
            if "cell_id" in exp_df.columns:
                exp_df["global_cell_id"] = exp_df["cell_id"].astype(str) + f"_patch_{patch_idx}"
            else:
                # Use the DataFrame index as cell_id if cell_id column doesn't exist
                exp_df["global_cell_id"] = exp_df.index.astype(str) + f"_patch_{patch_idx}"
                
            if "cell_id" in metadata_df.columns:
                metadata_df["global_cell_id"] = metadata_df["cell_id"].astype(str) + f"_patch_{patch_idx}"
            else:
                # Use the DataFrame index as cell_id if cell_id column doesn't exist
                metadata_df["global_cell_id"] = metadata_df.index.astype(str) + f"_patch_{patch_idx}"
            
            # For merge modes, transform coordinates from patch-local to global coordinates
            # Coordinate System: Standard image convention (TIFF/scikit-image)
            # - Origin (0,0) at top-left corner
            # - x (column) increases rightward, y (row) increases downward
            if should_merge and len(metadata_df) > 0:
                # Get patch coordinate information
                if isinstance(patches_metadata_df, pd.DataFrame):
                    patch_info = patches_metadata_df.iloc[patch_idx]
                else:
                    patch_info = patches_metadata_df[patch_idx]
                
                patch_x_start = patch_info.get("x_start", 0)
                patch_y_start = patch_info.get("y_start", 0)
                
                # Save original patch-level coordinates
                if "x" in metadata_df.columns:
                    metadata_df["patch_x"] = metadata_df["x"].copy()
                    metadata_df["x"] = metadata_df["x"] + patch_x_start
                    
                if "y" in metadata_df.columns:
                    metadata_df["patch_y"] = metadata_df["y"].copy()
                    metadata_df["y"] = metadata_df["y"] + patch_y_start
                
                # Also handle centroid columns if they exist
                if "centroid_x" in metadata_df.columns:
                    metadata_df["patch_centroid_x"] = metadata_df["centroid_x"].copy()
                    metadata_df["centroid_x"] = metadata_df["centroid_x"] + patch_x_start
                    
                if "centroid_y" in metadata_df.columns:
                    metadata_df["patch_centroid_y"] = metadata_df["centroid_y"].copy()
                    metadata_df["centroid_y"] = metadata_df["centroid_y"] + patch_y_start

            overview_df = exp_df.copy()
            overview_df = overview_df.drop(columns=["patch_id", "global_cell_id"], errors="ignore")
            for col in ["y", "x"]:
                if col in metadata_df.columns:
                    overview_df[col] = metadata_df[col]

            if "cell_area" in metadata_df.columns:
                overview_df["area"] = metadata_df["cell_area"]
            elif "area" in metadata_df.columns:
                logger.warning("Using area column from metadata_df instead of cell_area")
                overview_df["area"] = metadata_df["area"]

            nucleus_overview_df = exp_df.copy()
            nucleus_overview_df = nucleus_overview_df.drop(
                columns=["patch_id", "global_cell_id"],
                errors="ignore",
            )
            for col in ["y", "x"]:
                if col in metadata_df.columns:
                    nucleus_overview_df[col] = metadata_df[col]

            if "nucleus_area" in metadata_df.columns:
                nucleus_overview_df["area"] = metadata_df["nucleus_area"]
            elif "area" in metadata_df.columns:
                logger.warning("Using area column from metadata_df instead of nucleus_area")
                nucleus_overview_df["area"] = metadata_df["area"]

            marker_columns = [
                col
                for col in nucleus_overview_df.columns
                if col not in {"cell_mask_id", "y", "x", "area"}
            ]

            for marker in marker_columns:
                nucleus_col = f"{marker}_nucleus_mean"
                if nucleus_col in metadata_df.columns:
                    nucleus_overview_df[marker] = metadata_df[nucleus_col]
                else:
                    logger.debug(
                        "Missing nucleus intensity column %s for marker %s",
                        nucleus_col,
                        marker,
                    )

            preferred_order = ["cell_mask_id", "y", "x", "area"]
            ordered_columns = [
                col
                for col in preferred_order
                if col in overview_df.columns
            ] + [
                col
                for col in overview_df.columns
                if col not in preferred_order
            ]
            overview_df = overview_df[ordered_columns]

            nucleus_overview_df = nucleus_overview_df[ordered_columns]

        metadata_export = metadata_df.drop(columns=["area"], errors="ignore")

        # Update counter for detected cells
        total_cells += exp_df.shape[0]
        processed += 1

        # Progress logging by cells profiled (every 5% of expected total)
        if total_expected_cells > 0:
            while total_cells >= next_progress_fraction * total_expected_cells and next_progress_fraction <= 1.0:
                logger.info(
                    "Cell profiling progress: %.0f%% (%d/%d cells)",
                    next_progress_fraction * 100,
                    total_cells,
                    total_expected_cells,
                )
                next_progress_fraction += 0.05
        
        # Memory cleanup for disk-based patches
        if codex_patches.is_using_disk_based_patches():
            # Clear patch from memory
            del patch_img
            import gc
            gc.collect()

        if patch_max_label > 0:
            global_cell_offset += patch_max_label

        # Log progress periodically
        percent_complete = (processed / total_patches) * 100
        if i % log_frequency == 0 or i == total_patches - 1:
            logger.info(
                f"Progress: {percent_complete:.1f}% ({processed}/{total_patches}) - "
                f"Patch {patch_idx}: {exp_df.shape[0]} cells profiled"
            )

            if should_merge:
                # Collect results for later merging
                if len(exp_df) > 0:  # Only add non-empty DataFrames
                    all_exp_dfs.append(exp_df)
                    all_metadata_dfs.append(metadata_export)
                    all_overview_dfs.append(overview_df)
                    all_nucleus_overview_dfs.append(nucleus_overview_df)
            else:
                # Save individual patch results (patches mode)
                exp_file = os.path.join(
                    profiling_out_dir, f"patch-{patch_idx}-cell_by_marker.csv"
                )
                meta_file = os.path.join(
                    profiling_out_dir, f"patch-{patch_idx}-cell_metadata.csv"
                )
                overview_file = os.path.join(
                    profiling_out_dir, f"patch-{patch_idx}-cell_overview.csv"
                )
                nucleus_overview_file = os.path.join(
                    profiling_out_dir, f"patch-{patch_idx}-nucleus_overview.csv"
                )

                exp_df.to_csv(exp_file, index=False)
                metadata_export.to_csv(meta_file, index=False)
                overview_df.to_csv(overview_file, index=False)
                nucleus_overview_df.to_csv(nucleus_overview_file, index=False)

    # Handle merging for full_image, halves, quarters modes
    if should_merge and all_exp_dfs:
        logger.info(f"Merging results from {len(all_exp_dfs)} patches into single files...")
        
        # Merge all DataFrames
        merged_exp_df = pd.concat(all_exp_dfs, ignore_index=True)
        merged_metadata_df = pd.concat(all_metadata_dfs, ignore_index=True)
        merged_overview_df = pd.concat(all_overview_dfs, ignore_index=True)
        merged_nucleus_overview_df = pd.concat(
            all_nucleus_overview_dfs, ignore_index=True
        )
        
        # Save merged results
        merged_exp_file = os.path.join(profiling_out_dir, "cell_by_marker.csv")
        merged_meta_file = os.path.join(profiling_out_dir, "cell_metadata.csv")
        
        merged_exp_df.to_csv(merged_exp_file, index=False)
        merged_metadata_df.to_csv(merged_meta_file, index=False)
        merged_overview_df.to_csv(
            os.path.join(profiling_out_dir, "cell_overview.csv"), index=False
        )
        merged_nucleus_overview_df.to_csv(
            os.path.join(profiling_out_dir, "nucleus_overview.csv"), index=False
        )
        
        logger.info(f"Saved merged results: {merged_exp_df.shape[0]} cells total")
        logger.info(f"  - {merged_exp_file}")
        logger.info(f"  - {merged_meta_file}")
        logger.info(
            "  - %s",
            os.path.join(profiling_out_dir, "nucleus_overview.csv"),
        )
    elif should_merge and not all_exp_dfs:
        logger.warning("No cells found in any patches - no output files generated.")

    # Log final summary
    logger.info(f"----- Cell profiling completed. Total cells profiled: {total_cells} across {total_patches} patches.")
