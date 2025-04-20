import os
import logging
import pickle
from typing import Dict, Optional, Any, Tuple
import time
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import label
from scipy.spatial import cKDTree

from aegle.codex_patches import CodexPatches
from aegle.segmentation_analysis.intensity_analysis import intensity_extraction, bias_visualize
from aegle.segmentation_analysis.intensity_analysis import distribution_visualization
from aegle.segmentation_analysis.spatial_analysis import density_metrics, density_visualization

def run_segmentation_analysis(codex_patches: CodexPatches, config: dict, args=None) -> None:
    """Run segmentation analysis including bias and density analysis.

    Args:
        codex_patches: CodexPatches object containing patches and segmentation data
        config: Configuration Dictionary for evaluation options
        args: Optional additional arguments
    """
    output_dir = os.path.join(args.out_dir, "segmentation_analysis")

    # Extract segmentation data and metadata
    original_seg_res_batch = codex_patches.original_seg_res_batch
    repaired_seg_res_batch = codex_patches.repaired_seg_res_batch
    patches_metadata_df = codex_patches.get_patches_metadata()
    antibody_df = codex_patches.antibody_df
    antibody_list = antibody_df["antibody_name"].to_list()

    # Filter for informative patches
    informative_idx = patches_metadata_df["is_infomative"] == True
    logging.info(f"Number of informative patches: {informative_idx.sum()}")
    image_ndarray = codex_patches.all_channel_patches[informative_idx]
    logging.info(f"image_ndarray.shape: {image_ndarray.shape}")

    # --- Matched fraction ------------------------------------------
    # This is precalculated after segmentation in segmentation.py
    matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
    patches_metadata_df.loc[informative_idx, "matched_fraction"] = matched_fraction_list

    # Get microns per pixel from config
    image_mpp = config.get("data", {}).get("image_mpp", 0.5)

    # List to store all density metrics
    all_count_density_metrics = []

    # Process each patch
    res_list = []
    for idx, repaired_seg_res in enumerate(repaired_seg_res_batch):
        logging.info(f"Processing patch {idx+1}/{len(repaired_seg_res_batch)}")
        if repaired_seg_res is None:
            logging.warning(f"Repaired segmentation result for patch {idx} is None.")
            continue

        # Visualize results if specified in config
        patch_output_dir = f"{output_dir}/patch_{idx}"
        logging.info(f"patch_output_dir: {patch_output_dir}")
        os.makedirs(patch_output_dir, exist_ok=True)
        
        # Extract masks from original and repaired segmentation results
        original_seg_res = original_seg_res_batch[idx]
        repaired_seg_res = repaired_seg_res_batch[idx]

        # Get masks from original segmentation results
        cell_mask = original_seg_res.get("cell")
        nucleus_mask = original_seg_res.get("nucleus")
        logging.info(f"cell_mask.shape: {cell_mask.shape}")
        logging.info(f"nucleus_mask.shape: {nucleus_mask.shape}")
        logging.info(f"# objs in cell_mask: {len(np.unique(cell_mask))}")
        logging.info(f"# objs in nucleus_mask: {len(np.unique(nucleus_mask))}")

        # Get masks from repaired segmentation results
        cell_matched_mask = repaired_seg_res["cell_matched_mask"]
        nucleus_matched_mask = repaired_seg_res["nucleus_matched_mask"]
        logging.info(f"cell_matched_mask.shape: {cell_matched_mask.shape}")
        logging.info(f"nucleus_matched_mask.shape: {nucleus_matched_mask.shape}")
        logging.info(f"# objs in cell_matched_mask: {len(np.unique(cell_matched_mask))}")
        logging.info(f"# objs in nucleus_matched_mask: {len(np.unique(nucleus_matched_mask))}")

        # count the time it takes to run the following code to get the unmatched nuclei and label them
        start_time = time.time()
        # Create a mask for nuclei that appear in original segmentation but not in matched segmentation
        nucleus_unmatched_mask = np.logical_and(nucleus_mask > 0, nucleus_matched_mask == 0)
        # relabel the unmatched nuclei
        # label returns a tuple: (labeled_array, num_features)
        nucleus_unmatched_mask, num_objects = label(nucleus_unmatched_mask) # Unpack the tuple
        logging.info(f"Created nucleus_unmatched_mask with {num_objects} unmatched nuclei") # Use num_objects from label
        logging.info(f"nucleus_unmatched_mask.shape: {nucleus_unmatched_mask.shape}")
        end_time = time.time()
        logging.info(f"Time taken to run the following code to get the unmatched nuclei and label them: {end_time - start_time} seconds")
        # (1) Compute count density metrics
        count_density_results = density_metrics.calculate_count_density_metrics(
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
            nucleus_unmatched_mask,
            image_mpp
        )
        all_count_density_metrics.append(count_density_results)

        # Visualize density metrics
        density_visualization.visualize_density_distributions(
            count_density_results,
            output_dir=os.path.join(patch_output_dir, "cell_density_visualization"),
            save_plots=True
        )
        
        # (2) Extract intensity data across channels
        intensity_data = intensity_extraction.extract_intensity_data_across_channels(
            image_ndarray[idx],
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
            nucleus_unmatched_mask,
            channel_names=antibody_list,
        )

        # (3) Visualize bias analysis results
        bias_visualize.visualize_channel_intensity_bias(
            intensity_data,
            output_dir=os.path.join(patch_output_dir, "channel_intensity_repair_bias"),
            channels_per_figure=config.get("channels_per_figure", 10),
            channel_names=antibody_list,
        )

        # (4) Create density plots for this patch
        distribution_visualization.visualize_intensity_distributions(
            intensity_data,
            output_dir=os.path.join(patch_output_dir, "channel_intensity_distributions_plots"),
            use_log_scale=True,
            channel_names=antibody_list,
        )

        # Store results
        evaluation_result = {
            "patch_idx": idx,
            "intensity_analysis": intensity_data,
            "density_metrics": count_density_results,
        }
        res_list.append(evaluation_result)

    # Update metadata with density metrics
    patches_metadata_df = density_metrics.update_metadata_with_density_metrics(
        patches_metadata_df, informative_idx, all_count_density_metrics
    )
    codex_patches.seg_evaluation_metrics = res_list
    codex_patches.set_metadata(patches_metadata_df)
    codex_patches.save_metadata()
    
    file_name = os.path.join(output_dir, "codex_patches_segmentation_analysis.pickle")
    logging.info(f"Saving segmentation analysis results to {file_name}")
    with open(file_name, "wb") as f:
        pickle.dump(codex_patches, f)
