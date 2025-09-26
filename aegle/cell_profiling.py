import os
import logging
import pandas as pd
import numpy as np

from aegle.extract_features import extract_features_v2_optimized

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

    # Initialize lists for merging results (used for full_image, halves, quarters modes)
    all_exp_dfs = []
    all_metadata_dfs = []
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

        # Get the full multi-channel patch from all channels (or extracted channels)
        if codex_patches.is_using_disk_based_patches():
            # Load patch from disk
            patch_img = codex_patches.load_patch_from_disk(patch_idx, "all")
            logger.info(f"Loaded patch {patch_idx} from disk for profiling with shape: {patch_img.shape}")
        else:
            # Load from memory
            patch_img = codex_patches.all_channel_patches[patch_idx]

        # Build the image_dict for extract_features_v2
        # Each entry is {antibody_name: 2D image}
        image_dict = {}
        for channel_idx, ab in enumerate(antibodies):
            image_dict[ab] = patch_img[:, :, channel_idx]

        # Extract features (exp_df: cell-by-marker, metadata_df: cell metadata)
        exp_df, metadata_df = extract_features_v2_optimized(
            image_dict, nucleus_mask, antibodies, cell_masks=cell_mask
        )
        
        # Add patch information to track origin of cells
        if len(exp_df) > 0:  # Only add columns if there are cells
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
        
        # Update counter for detected cells
        total_cells += exp_df.shape[0]
        processed += 1
        
        # Memory cleanup for disk-based patches
        if codex_patches.is_using_disk_based_patches():
            # Clear patch from memory
            del patch_img
            import gc
            gc.collect()
        
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
                all_metadata_dfs.append(metadata_df)
        else:
            # Save individual patch results (patches mode)
            exp_file = os.path.join(
                profiling_out_dir, f"patch-{patch_idx}-cell_by_marker.csv"
            )
            meta_file = os.path.join(
                profiling_out_dir, f"patch-{patch_idx}-cell_metadata.csv"
            )

            exp_df.to_csv(exp_file, index=False)
            metadata_df.to_csv(meta_file, index=False)

    # Handle merging for full_image, halves, quarters modes
    if should_merge and all_exp_dfs:
        logger.info(f"Merging results from {len(all_exp_dfs)} patches into single files...")
        
        # Merge all DataFrames
        merged_exp_df = pd.concat(all_exp_dfs, ignore_index=True)
        merged_metadata_df = pd.concat(all_metadata_dfs, ignore_index=True)
        
        # Save merged results
        merged_exp_file = os.path.join(profiling_out_dir, "cell_by_marker.csv")
        merged_meta_file = os.path.join(profiling_out_dir, "cell_metadata.csv")
        
        merged_exp_df.to_csv(merged_exp_file, index=False)
        merged_metadata_df.to_csv(merged_meta_file, index=False)
        
        logger.info(f"Saved merged results: {merged_exp_df.shape[0]} cells total")
        logger.info(f"  - {merged_exp_file}")
        logger.info(f"  - {merged_meta_file}")
    elif should_merge and not all_exp_dfs:
        logger.warning("No cells found in any patches - no output files generated.")

    # Log final summary
    logger.info(f"----- Cell profiling completed. Total cells profiled: {total_cells} across {total_patches} patches.")
