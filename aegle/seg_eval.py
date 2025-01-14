import os
import logging
from typing import List, Optional, Dict
from pathlib import Path
import pickle
from single_method_eval import single_method_eval


def run_segmentation_evaluation(
    all_patches_ndarray: np.ndarray,
    matched_seg_res: List[dict],
    config: dict,
    args: Optional[dict],
):
    """
    Evaluate the segmentation results for each patch.

    Args:
        all_patches_ndarray (np.ndarray): Array of image patches.
        matched_seg_res (List[dict]): Segmentation results for each patch.
        config (dict): Configuration parameters.
        args: Command-line arguments.

    Returns:
        None
    """

    # Ensure the output directory exists
    output_dir = Path(args.out_dir) / "segmentation_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all patches and run evaluation
    evaluation_results = []
    for i, patch_seg_res in enumerate(matched_seg_res):
        try:
            logging.info(f"Evaluating patch {i + 1}/{len(matched_seg_res)}...")

            # Run single method evaluation on the current patch
            metrics, fraction_background, background_cv, background_pca = (
                single_method_eval(all_patches_ndarray[i], patch_seg_res, output_dir)
            )

            # Log the results and collect them for final output
            logging.info(f"Patch {i + 1} evaluation completed.")
            evaluation_results.append(
                {
                    "patch_index": i,
                    "metrics": metrics,
                    "fraction_background": fraction_background,
                    "background_cv": background_cv,
                    "background_pca": background_pca,
                }
            )

        except Exception as e:
            logging.error(f"Error evaluating patch {i}: {e}", exc_info=True)

    # Save the evaluation results as a file
    results_path = output_dir / "segmentation_evaluation_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(evaluation_results, f)

    logging.info(f"Segmentation evaluation completed. Results saved to {results_path}.")
