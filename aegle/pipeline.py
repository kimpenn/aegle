import os
import numpy as np
import logging
import pickle
import shutil
import time

from aegle.codex_image import CodexImage
from aegle.codex_patches import CodexPatches

from aegle.visualization import save_patches_rgb, save_image_rgb

from aegle.segment import run_cell_segmentation, visualize_cell_segmentation
from aegle.evaluation import run_seg_evaluation
from aegle.cell_profiling import run_cell_profiling
from aegle.segmentation_analysis.segmentation_analysis import run_segmentation_analysis
# from aegle.segmentation_analysis.intensity_analysis import bias_analysis, distribution_analysis
# from aegle.segmentation_analysis.spatial_analysis import density_metrics
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

def run_pipeline(config, args):
    """
    Run the CODEX image analysis pipeline.

    Args:
        config (dict): Configuration parameters loaded from YAML file.
        args (Namespace): Command-line arguments parsed by argparse.
    """
    logging.info("----- Running pipeline with provided configuration and arguments.")
    os.makedirs(args.out_dir, exist_ok=True)
    logging.info(f"Output directory set to: {args.out_dir}")
    copied_config_path = os.path.join(args.out_dir, "copied_config.yaml")
    shutil.copy(args.config_file, copied_config_path)

    # ---------------------------------
    # (A) Load Image and Antibodies Data
    # ---------------------------------
    # Step 1: Initialize CodexImage object
    # - Read and Codex image as well the dataframe about antibodies
    logging.info("----- Initializing CodexImage object.")
    codex_image = CodexImage(config, args)
    logging.info("CodexImage object initialized successfully.")
    if config["data"]["generate_channel_stats"]:
        logging.info("----- Generating channel statistics.")
        codex_image.calculate_channel_statistics()
        logging.info("Channel statistics generated successfully.")

    # Step 2: Extract target channels from the image based on configuration
    logging.info("----- Extracting target channels from the image.")
    codex_image.extract_target_channels()
    logging.info("Target channels extracted successfully.")

    # ---------------------------------
    # (B) Patched Image Preprocessing
    # ---------------------------------
    # Step 1: Extend the image for full patch coverage
    logging.info("----- Extending image for full patch coverage.")
    codex_image.extend_image()
    logging.info("Image extension completed successfully.")

    # Optional: Visualize whole sample image
    if config.get("visualization", {}).get("visualize_whole_sample", False):
        logging.info("----- Visualizing whole sample image...")
        start_time = time.time()
        
        # Get visualization settings from config or use defaults
        downsample_factor = config.get("visualization", {}).get("downsample_factor", None)
        enhance_contrast = config.get("visualization", {}).get("enhance_contrast", True)
        
        # Check if we should auto-downsample
        image_shape = codex_image.extended_extracted_channel_image.shape
        total_pixels = image_shape[0] * image_shape[1]
        
        # Auto-downsample if explicitly requested (-1) or implicitly needed (large image)
        if total_pixels > 100000000 and downsample_factor == -1:
            # Calculate a reasonable downsample factor based on image size
            # Aim for ~25-50 million pixels in the final image
            auto_downsample_factor = max(2, int(np.sqrt(total_pixels / 25000000)))
            logging.info(f"Auto-downsampling image (original size: {image_shape})")
            logging.info(f"Calculated downsample factor: {auto_downsample_factor}")
            downsample_factor = auto_downsample_factor
        
        save_image_rgb(
            codex_image.extended_extracted_channel_image,
            "extended_extracted_channel_image.png",
            args,
            downsample_factor=downsample_factor,
            enhance_contrast=enhance_contrast
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Whole sample visualization completed in {elapsed_time:.2f} seconds.")

    # Step 2: Initialize CodexPatches object and generate patches
    logging.info("----- Initializing CodexPatches object and generating patches.")
    codex_patches = CodexPatches(codex_image, config, args)
    codex_patches.save_patches(config["visualization"]["save_all_channel_patches"])
    codex_patches.save_metadata()
    logging.info("Patches generated and metadata saved successfully.")

    # Optional: Add disruptions to patches for testing
    # Extract distruption type and level from config
    disruption_config = config.get("testing", {}).get("data_disruption", {})
    logging.info(f"Disruption config: {disruption_config}")
    has_disruptions = False
    if disruption_config and disruption_config.get("type", None) is not None:
        disruption_type = disruption_config.get("type", None)
        disruption_level = disruption_config.get("level", 1)
        logging.info(
            f"Adding disruptions {disruption_type} at level {disruption_level} to patches for testing."
        )
        codex_patches.add_disruptions(disruption_type, disruption_level)
        logging.info("Disruptions added to patches.")
        has_disruptions = True
        if disruption_config.get("save_disrupted_patches", False):
            logging.info("Saving disrupted patches.")
            codex_patches.save_disrupted_patches()
            logging.info("Disrupted patches saved.")

    # Optional: Visualize patches
    # Priority: if disruptions exist and visualize_patches is True, visualize disrupted patches
    # Otherwise, visualize original patches
    if config.get("visualization", {}).get("visualize_patches", False):
        if has_disruptions and disruption_config.get("visualize_disrupted", True):
            logging.info("Visualizing disrupted patches.")
            save_patches_rgb(
                codex_patches.disrupted_extracted_channel_patches,
                codex_patches.patches_metadata,
                config,
                args,
                max_workers=config.get("visualization", {}).get("workers", None),
            )
            logging.info("Disrupted patch visualization completed.")
        else:
            logging.info("Visualizing original patches.")
            save_patches_rgb(
                codex_patches.extracted_channel_patches,
                codex_patches.patches_metadata,
                config,
                args,
                max_workers=config.get("visualization", {}).get("workers", None),
            )
            logging.info("Original patch visualization completed.")

    # ---------------------------------
    # (C) Cell Segmentation and auto evaluation
    # ---------------------------------
    logging.info("Running cell segmentation.")
    run_cell_segmentation(codex_patches, config, args)

    if config["evaluation"]["compute_metrics"]:
        # TODO: if the number of cells are too large we should skip the evaluation
        run_seg_evaluation(codex_patches, config, args)
        # save the seg_evaluation_metrics to a pickle file
        with open(os.path.join(args.out_dir, "seg_evaluation_metrics.pkl"), "wb") as f:
            pickle.dump(codex_patches.seg_evaluation_metrics, f)

    # if config.get("visualization", {}).get("visualize_segmentation", False):
    # TODO: Fix this visualization function
    #     visualize_cell_segmentation(
    #         codex_patches.valid_patches,
    #         codex_patches.repaired_seg_res_batch,
    #         config,
    #         args,
    #     )
    #     logging.info("Segmentation visualization completed.")

    #     visualize_cell_segmentation(
    #         codex_patches.valid_patches,
    #         codex_patches.original_seg_res_batch,
    #         config,
    #         args,
    #     )
    #     logging.info("Segmentation visualization completed.")

    # ---------------------------------
    # (D) Cell Profiling
    # ---------------------------------
    logging.info("Running cell profiling.")
    run_cell_profiling(codex_patches, config, args)
    logging.info("Cell profiling completed.")
    
    # # Clean up intermediate patch files if using disk-based patches
    # # This is done after cell profiling since profiling needs access to the "all" channel patches
    # if codex_patches.is_using_disk_based_patches():
    #     try:
    #         codex_patches.cleanup_intermediate_patches()
    #         logging.info("Intermediate patch files cleaned up successfully")
    #     except Exception as e:
    #         logging.warning(f"Failed to clean up intermediate patch files: {e}")

    # ---------------------------------
    # (E) Segmentation Analysis
    # ---------------------------------
    if config["segmentation"]["segmentation_analysis"]:      
        logging.info("Running segmentation analysis...")
        run_segmentation_analysis(codex_patches, config, args)
        logging.info("Segmentation analysis completed.")

        logging.info("Pipeline run completed.")


# def run_segmentation_analysis(codex_patches: CodexPatches, config: dict, args=None) -> None:
#     """Run segmentation analysis including bias and density analysis.

#     Args:
#         codex_patches: CodexPatches object containing patches and segmentation data
#         config: Configuration Dictionary for evaluation options
#         args: Optional additional arguments
#     """
#     # Extract segmentation data and metadata
#     original_seg_res_batch = codex_patches.original_seg_res_batch
#     repaired_seg_res_batch = codex_patches.repaired_seg_res_batch
#     patches_metadata_df = codex_patches.get_patches_metadata()
#     antibody_df = codex_patches.antibody_df
#     logging.info(f"antibody_df:\n{antibody_df}")
#     antibody_list = antibody_df["antibody_name"].to_list()

#     # Filter for informative patches
#     informative_idx = patches_metadata_df["is_informative"] == True
#     logging.info(f"Number of informative patches: {informative_idx.sum()}")
#     image_ndarray = codex_patches.all_channel_patches[informative_idx]
#     logging.info(f"image_ndarray.shape: {image_ndarray.shape}")

#     output_dir = config.get("output_dir", "./output")

#     # --- Matched fraction ------------------------------------------
#     # This is precalculated after segmentation in segmentation.py
#     matched_fraction_list = [res["matched_fraction"] for res in repaired_seg_res_batch]
#     patches_metadata_df.loc[informative_idx, "matched_fraction"] = matched_fraction_list

#     # Get microns per pixel from config
#     image_mpp = config.get("data", {}).get("image_mpp", 0.5)

#     # List to store all density metrics
#     all_density_metrics = []

#     # Process each patch
#     res_list = []
#     for idx, repaired_seg_res in enumerate(repaired_seg_res_batch):
#         logging.info(f"Processing patch {idx+1}/{len(repaired_seg_res_batch)}")
#         if repaired_seg_res is None:
#             logging.warning(f"Repaired segmentation result for patch {idx} is None.")
#             continue

#         # Visualize results if specified in config
#         patch_output_dir = f"{output_dir}/patch_{idx}"
        
#         # Extract masks from original and repaired segmentation results
#         original_seg_res = original_seg_res_batch[idx]
#         repaired_seg_res = repaired_seg_res_batch[idx]

#         # Get masks from original segmentation results
#         cell_mask = original_seg_res.get("cell")
#         nucleus_mask = original_seg_res.get("nucleus")
#         logging.info(f"cell_mask.shape: {cell_mask.shape}")
#         logging.info(f"nucleus_mask.shape: {nucleus_mask.shape}")

#         # Get masks from repaired segmentation results
#         cell_matched_mask = repaired_seg_res["cell_matched_mask"]
#         nucleus_matched_mask = repaired_seg_res["nucleus_matched_mask"]
#         logging.info(f"cell_matched_mask.shape: {cell_matched_mask.shape}")
#         logging.info(f"nucleus_matched_mask.shape: {nucleus_matched_mask.shape}")

#         # Compute density metrics
#         density_metrics = density_metrics.update_patch_metrics(
#             cell_mask, nucleus_mask, cell_matched_mask, nucleus_matched_mask, image_mpp
#         )
#         all_density_metrics.append(density_metrics)

#         # Analyze repair bias across channels
#         bias_results = bias_analysis.analyze_intensity_bias_across_channels(
#             image_ndarray[idx],
#             cell_mask,
#             nucleus_mask,
#             cell_matched_mask,
#             nucleus_matched_mask,
#         )

#         # Visualize bias analysis results
#         bias_analysis.visualize_channel_bias(
#             bias_results,
#             output_dir=os.path.join(patch_output_dir, "bias_analysis"),
#             channels_per_figure=config["channels_per_figure"],
#         )

#         # Create antibody intensity density plots for this patch
#         density_results = density_analysis.visualize_intensity_distributions(
#             image_ndarray[idx],
#             cell_mask,
#             nucleus_mask,
#             cell_matched_mask,
#             nucleus_matched_mask,
#             output_dir=os.path.join(patch_output_dir, "density_plots"),
#             use_log_scale=True,
#             channel_names=antibody_list,
#         )

#         # Store results
#         evaluation_result = {
#             "patch_idx": idx,
#             "bias_analysis": bias_results,
#             "density_analysis": density_results,
#             "density_metrics": density_metrics,
#         }
#         res_list.append(evaluation_result)

#     # Update metadata with density metrics
#     patches_metadata_df = density_metrics.update_metadata_with_density_metrics(
#         patches_metadata_df, informative_idx, all_density_metrics
#     )
#     codex_patches.seg_evaluation_metrics = res_list
#     codex_patches.set_metadata(patches_metadata_df)
