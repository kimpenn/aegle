import os
import logging
import pandas as pd
import numpy as np

from aegle.extract_features import extract_features_v2, extract_features_v2_optimized


def run_cell_profiling(codex_patches, config, args):
    """
    Runs cell-level profiling on each patch, generating a cell-by-marker matrix
    and a cell metadata table. Results are saved as CSV files, one per patch.

    Args:
        codex_patches (CodexPatches): The patched CODEX data, with segmentation results.
        config (dict): The pipeline configuration.
        args (Namespace): Command-line arguments with paths and other settings.
    """
    logging.info("----- Running cell profiling.")

    # Retrieve the antibody_df directly from codex_patches
    antibody_df = codex_patches.antibody_df
    # Convert to a simple list of antibody names
    antibodies = antibody_df["antibody_name"].values

    # Create a directory for cell profiling outputs
    profiling_out_dir = os.path.join(args.out_dir, "cell_profiling")
    os.makedirs(profiling_out_dir, exist_ok=True)

    # Iterate through patches
    num_patches = len(codex_patches.patches_metadata)
    logging.info(f"Profiling {num_patches} patches.")

    for patch_idx in range(num_patches):
        # Get segmentation mask from repaired segmentation
        seg_result = codex_patches.repaired_seg_res_batch[patch_idx]
        segmentation_masks = seg_result["nuclear_matched_mask"]

        # Get the full multi-channel patch from all channels (or extracted channels)
        patch_img = codex_patches.all_channel_patches[patch_idx]

        # Build the image_dict for extract_features_v2
        # Each entry is {antibody_name: 2D image}
        image_dict = {}
        for channel_idx, ab in enumerate(antibodies):
            image_dict[ab] = patch_img[:, :, channel_idx]

        # Extract features (exp_df: cell-by-marker, metadata_df: cell metadata)
        # exp_df, metadata_df = extract_features_v2(
        #     image_dict, segmentation_masks, antibodies
        # )
        
        exp_df, metadata_df = extract_features_v2_optimized(
            image_dict, segmentation_masks, antibodies
        )        

        logging.info(
            f"Patch {patch_idx} profiling complete: "
            f"exp_df shape {exp_df.shape}, metadata_df shape {metadata_df.shape}"
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

    logging.info("----- Cell profiling completed.")
