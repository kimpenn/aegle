import os
import logging
import pandas as pd
import numpy as np

from aegle.extract_features import extract_features_v2, extract_features_v2_optimized

# Create a logger specific to this module
logger = logging.getLogger(__name__)


def run_cell_profiling(codex_patches, config, args):
    """
    Runs cell-level profiling on each patch, generating a cell-by-marker matrix
    and a cell metadata table. Results are saved as CSV files, one per patch.

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

    # Get patches metadata and select only informative patches
    patches_metadata_df = codex_patches.get_patches_metadata()
    informative_idx = patches_metadata_df["is_infomative"] == True
    
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

    # Process only informative patches
    for i, patch_idx in enumerate(informative_patches):
        # Get segmentation mask from repaired segmentation
        seg_batch_idx = i  # Index in the repaired_seg_res_batch corresponds to position in informative patches list
        seg_result = codex_patches.repaired_seg_res_batch[seg_batch_idx]
        
        # TODO: consider profiling all four masks: nuc, nuc_matched, cell, cell_matched
        segmentation_masks = seg_result["nucleus_matched_mask"]

        # Get the full multi-channel patch from all channels (or extracted channels)
        patch_img = codex_patches.all_channel_patches[patch_idx]

        # Build the image_dict for extract_features_v2
        # Each entry is {antibody_name: 2D image}
        image_dict = {}
        for channel_idx, ab in enumerate(antibodies):
            image_dict[ab] = patch_img[:, :, channel_idx]

        # Extract features (exp_df: cell-by-marker, metadata_df: cell metadata)
        exp_df, metadata_df = extract_features_v2_optimized(
            image_dict, segmentation_masks, antibodies
        )
        
        # Update counter for detected cells
        total_cells += exp_df.shape[0]
        processed += 1
        
        # Log progress periodically
        percent_complete = (processed / total_patches) * 100
        if i % log_frequency == 0 or i == total_patches - 1:
            logger.info(
                f"Progress: {percent_complete:.1f}% ({processed}/{total_patches}) - "
                f"Patch {patch_idx}: {exp_df.shape[0]} cells profiled"
            )

        # Save results to CSV
        exp_file = os.path.join(
            profiling_out_dir, f"patch-{patch_idx}-cell_by_marker.csv"
        )
        meta_file = os.path.join(
            profiling_out_dir, f"patch-{patch_idx}-cell_metadata.csv"
        )

        exp_df.to_csv(exp_file, index=False)
        metadata_df.to_csv(meta_file, index=False)

    # Log final summary
    logger.info(f"----- Cell profiling completed. Total cells profiled: {total_cells} across {total_patches} patches.")
