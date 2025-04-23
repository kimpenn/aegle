import os
import logging
import pickle
from typing import Dict, Optional, Any, Tuple, List
import time
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import label
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from aegle.codex_patches import CodexPatches
from aegle.segmentation_analysis.intensity_analysis import intensity_extraction, bias_visualize
from aegle.segmentation_analysis.intensity_analysis import distribution_visualization
from aegle.segmentation_analysis.spatial_analysis import density_metrics, density_visualization

# Create a module-specific logger
logger = logging.getLogger(__name__)

def process_single_patch(
    patch_id: str, 
    repaired_seg_res: Dict, 
    original_seg_res: Dict, 
    patch_image: np.ndarray, 
    antibody_list: List[str], 
    image_mpp: float,
    output_dir: str,
    config: Dict
) -> Dict:
    """Process a single patch for segmentation analysis.
    
    Args:
        patch_id: Unique identifier of the patch
        repaired_seg_res: Repaired segmentation results for this patch
        original_seg_res: Original segmentation results for this patch
        patch_image: Image data for this patch
        antibody_list: List of antibody names
        image_mpp: Microns per pixel
        output_dir: Base output directory
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results for this patch
    """
    patch_start_time = time.time()
    logger.info(f"Processing patch {patch_id}")
    
    if repaired_seg_res is None:
        logger.warning(f"Repaired segmentation result for patch {patch_id} is None.")
        return None

    # Visualize results if specified in config
    patch_output_dir = f"{output_dir}/patch_{patch_id}"
    os.makedirs(patch_output_dir, exist_ok=True)
    
    # Get masks from original segmentation results
    cell_mask = original_seg_res.get("cell")
    nucleus_mask = original_seg_res.get("nucleus")
    
    # Get masks from repaired segmentation results
    cell_matched_mask = repaired_seg_res["cell_matched_mask"]
    nucleus_matched_mask = repaired_seg_res["nucleus_matched_mask"]
    
    # Create a mask for nuclei that appear in original segmentation but not in matched segmentation
    nucleus_unmatched_mask = np.logical_and(nucleus_mask > 0, nucleus_matched_mask == 0)
    # relabel the unmatched nuclei
    nucleus_unmatched_mask, num_objects = label(nucleus_unmatched_mask)
    logger.info(f"Patch {patch_id}: Created nucleus_unmatched_mask with {num_objects} unmatched nuclei")
    
    # Get density calculation configs
    density_config = config.get("density_analysis", {})
    calculate_global_density = density_config.get("calculate_global_density", True)
    calculate_local_density = density_config.get("calculate_local_density", True)
    
    count_density_results = {}
    
    # (1) Compute count density metrics if enabled
    if calculate_global_density or calculate_local_density:
        count_density_results = density_metrics.calculate_count_density_metrics(
            cell_mask,
            nucleus_mask,
            cell_matched_mask,
            nucleus_matched_mask,
            nucleus_unmatched_mask,
            image_mpp,
            calculate_global_density=calculate_global_density,
            calculate_local_density=calculate_local_density
        )
        
        # Visualize density metrics if calculated
        if count_density_results:
            density_visualization.visualize_density_distributions(
                count_density_results,
                output_dir=os.path.join(patch_output_dir, "cell_density_visualization"),
                save_plots=True
            )
    else:
        logger.info(f"Skipping density calculations for patch {patch_id} as per configuration")
    
    # (2) Extract intensity data across channels
    intensity_data = intensity_extraction.extract_intensity_data_across_channels(
        patch_image,
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
    
    # Store and return results
    evaluation_result = {
        "patch_id": patch_id,
        "intensity_analysis": intensity_data,
        "density_metrics": count_density_results,
    }
    
    patch_processing_time = time.time() - patch_start_time
    logger.info(f"Completed processing patch {patch_id} in {patch_processing_time:.2f} seconds")
    
    return evaluation_result

def run_segmentation_analysis(codex_patches: CodexPatches, config: dict, args=None) -> None:
    """Run segmentation analysis including bias and density analysis.

    Args:
        codex_patches: CodexPatches object containing patches and segmentation data
        config: Configuration Dictionary for evaluation options
        args: Optional additional arguments
    """
    start_time = time.time()
    output_dir = os.path.join(args.out_dir, "segmentation_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Extract segmentation data and metadata
    original_seg_res_batch = codex_patches.original_seg_res_batch
    repaired_seg_res_batch = codex_patches.repaired_seg_res_batch
    patches_metadata_df = codex_patches.get_patches_metadata()
    antibody_df = codex_patches.antibody_df
    antibody_list = antibody_df["antibody_name"].to_list()

    # Filter for informative patches
    informative_idx = patches_metadata_df["is_infomative"] == True
    logger.info(f"Number of informative patches: {informative_idx.sum()}")
    image_ndarray = codex_patches.all_channel_patches[informative_idx]
    logger.info(f"image_ndarray.shape: {image_ndarray.shape}")

    # --- Matched fraction ------------------------------------------
    # This is precalculated after segmentation in segmentation.py
    matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
    patches_metadata_df.loc[informative_idx, "matched_fraction"] = matched_fraction_list

    # Get microns per pixel from config
    image_mpp = config.get("data", {}).get("image_mpp", 0.5)
    
    # Get number of parallel workers from config, default to 4
    num_workers = config.get("parallel_processing", {}).get("workers", 4)
    logger.info(f"Using {num_workers} workers for parallel processing")

    # Get density calculation settings from config
    density_config = config.get("density_analysis", {})
    calculate_global_density = density_config.get("calculate_global_density", True)
    calculate_local_density = density_config.get("calculate_local_density", True)
    logger.info(f"Density analysis settings - Global: {calculate_global_density}, Local: {calculate_local_density}")

    # Process patches in parallel
    logger.info(f"Starting parallel processing of {len(repaired_seg_res_batch)} patches")
    
    # Create a partial function with fixed arguments
    process_patch = partial(
        process_single_patch,
        antibody_list=antibody_list,
        image_mpp=image_mpp,
        output_dir=output_dir,
        config=config
    )
    
    # List to store all density metrics and results
    all_count_density_metrics = []
    res_list = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all patch processing tasks
        futures = []
        # Get the informative patches metadata
        informative_patches = patches_metadata_df[informative_idx].reset_index(drop=True)
        
        for idx, repaired_seg_res in enumerate(repaired_seg_res_batch):
            # Get the actual patch_id from the metadata
            patch_id = str(informative_patches.loc[idx, "patch_id"])
            futures.append(
                executor.submit(
                    process_patch,
                    patch_id=patch_id,
                    repaired_seg_res=repaired_seg_res,
                    original_seg_res=original_seg_res_batch[idx],
                    patch_image=image_ndarray[idx],
                )
            )
        
        # Collect results as they complete
        for future in futures:
            result = future.result()
            if result is not None:
                res_list.append(result)
                if "density_metrics" in result and result["density_metrics"]:
                    # Add patch_id to density metrics for proper matching
                    result["density_metrics"]["patch_id"] = result["patch_id"]
                    all_count_density_metrics.append(result["density_metrics"])
    
    # Update metadata with density metrics only if density calculations were enabled
    if calculate_global_density or calculate_local_density:
        if all_count_density_metrics:
            patches_metadata_df = density_metrics.update_metadata_with_density_metrics(
                patches_metadata_df, informative_idx, all_count_density_metrics
            )
        else:
            logger.warning("No density metrics calculated to update metadata with")
    
    # Save results
    codex_patches.seg_evaluation_metrics = res_list
    codex_patches.set_metadata(patches_metadata_df)
    codex_patches.save_metadata()
    
    file_name = os.path.join(output_dir, "codex_patches_segmentation_analysis.pickle")
    logger.info(f"Saving segmentation analysis results to {file_name}")
    with open(file_name, "wb") as f:
        pickle.dump(codex_patches, f)
    
    total_time = time.time() - start_time
    logger.info(f"Segmentation analysis completed in {total_time:.2f} seconds")
