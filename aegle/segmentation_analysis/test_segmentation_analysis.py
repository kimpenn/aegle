import os
import sys
import logging
import pickle
import argparse
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

import numpy as np
from skimage.measure import regionprops
from scipy.spatial import cKDTree

src_path = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline"
sys.path.append(src_path)
from aegle.codex_patches import CodexPatches
from aegle.segmentation_analysis.segmentation_analysis import run_segmentation_analysis

def check_cell_id_matching(
    cell_mask: np.ndarray,
    cell_matched_mask: np.ndarray,
    tolerance: float = 1.0
) -> Tuple[int, int, Dict]:
    """
    Check if original and matched masks share the same IDs for the same cells.
    Uses spatial indexing (KD-tree) for efficient nearest neighbor search.
    
    Args:
        cell_mask: Original cell mask (2D array)
        cell_matched_mask: Repaired cell mask (2D array)
        tolerance: Maximum distance in pixels to consider centroids as matching
        
    Returns:
        Tuple containing:
        - Number of matching cells
        - Total number of cells
        - Dictionary of mismatches with format {original_id: matched_id}
    """
    # Get centroids for original and matched cells
    orig_cell_props = regionprops(cell_mask)
    matched_cell_props = regionprops(cell_matched_mask)
    
    # Extract centroids and IDs
    orig_centroids = np.array([prop.centroid for prop in orig_cell_props])
    orig_ids = np.array([prop.label for prop in orig_cell_props])
    
    matched_centroids = np.array([prop.centroid for prop in matched_cell_props])
    matched_ids = np.array([prop.label for prop in matched_cell_props])
    
    # Build KD-tree for matched centroids
    tree = cKDTree(matched_centroids)
    
    # Find nearest neighbors for original centroids
    distances, indices = tree.query(orig_centroids, k=1)
    
    # Count matches and identify mismatches
    matching_cells = 0
    total_cells = len(orig_centroids)
    mismatches = {}
    
    for i, (dist, matched_idx) in enumerate(zip(distances, indices)):
        if dist <= tolerance:
            orig_id = orig_ids[i]
            matched_id = matched_ids[matched_idx]
            if orig_id == matched_id:
                matching_cells += 1
            else:
                mismatches[orig_id] = matched_id
    
    return matching_cells, total_cells, mismatches

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

def load_codex_patches(pickle_path: str) -> CodexPatches:
    """Load CodexPatches object from pickle file with error handling."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"CodexPatches pickle file not found at: {pickle_path}")

    try:
        logging.info(f"Loading CodexPatches from: {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading CodexPatches: {str(e)}")

def create_test_config(output_dir: str, image_mpp: float = 0.5) -> Dict:
    """Create a test configuration dictionary."""
    return {
        "data": {
            "image_mpp": image_mpp
        },
        "channels_per_figure": 10,
        "output_dir": output_dir
    }

def test_cell_id_matching(codex_patches: CodexPatches, patch_idx: int = 0) -> Dict:
    """
    Test cell ID matching for a specific patch.
    
    Args:
        codex_patches: CodexPatches object
        patch_idx: Index of the patch to test
        
    Returns:
        Dictionary containing matching results
    """
    original_seg_res = codex_patches.original_seg_res_batch[patch_idx]
    repaired_seg_res = codex_patches.repaired_seg_res_batch[patch_idx]

    if original_seg_res is None or repaired_seg_res is None:
        raise ValueError(f"Missing segmentation results for patch {patch_idx}")

    cell_mask = original_seg_res.get("cell")
    cell_matched_mask = repaired_seg_res["cell_matched_mask"]

    if cell_mask is None or cell_matched_mask is None:
        raise ValueError(f"Missing cell masks for patch {patch_idx}")

    matching_cells, total_cells, mismatches = check_cell_id_matching(
        cell_mask, cell_matched_mask
    )

    return {
        "matching_cells": matching_cells,
        "total_cells": total_cells,
        "matching_percentage": matching_cells/total_cells*100 if total_cells > 0 else 0,
        "mismatches": mismatches
    }

def run_analysis(
    codex_patches: CodexPatches,
    config: Dict,
    output_dir: str,
    test_cell_matching: bool = True
) -> None:
    """
    Run the full segmentation analysis pipeline.
    
    Args:
        codex_patches: CodexPatches object
        config: Configuration dictionary
        output_dir: Output directory for results
        test_cell_matching: Whether to run cell ID matching tests
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Test cell ID matching if requested
    if test_cell_matching:
        logging.info("Testing cell ID matching...")
        try:
            matching_results = test_cell_id_matching(codex_patches)
            logging.info(
                f"Cell ID matching analysis: {matching_results['matching_cells']}/"
                f"{matching_results['total_cells']} cells have matching IDs "
                f"({matching_results['matching_percentage']:.2f}%)"
            )
            if matching_results['mismatches']:
                logging.warning(f"Found {len(matching_results['mismatches'])} cell ID mismatches")
        except Exception as e:
            logging.error(f"Error in cell ID matching test: {str(e)}")

    # Create mock args object with required out_dir attribute
    class MockArgs:
        def __init__(self, out_dir):
            self.out_dir = out_dir

    mock_args = MockArgs(output_dir)

    # Run the full segmentation analysis
    logging.info("Running full segmentation analysis...")
    try:
        run_segmentation_analysis(codex_patches, config, mock_args)
        logging.info("Segmentation analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in segmentation analysis: {str(e)}")
        raise

def main():
    """Main function to run segmentation analysis tests."""
    parser = argparse.ArgumentParser(description="Run segmentation analysis tests")
    parser.add_argument(
        "--input", 
        type=str,
        # default="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/out/main/test0206_main/D18_Scan1_1_markerset_2/codex_patches_segmentation_analysis.pickle",
        default="/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/aegle/segmentation_analysis/test_output/mock_codex_patches_segmentation_analysis.pickle",
        help="Path to input CodexPatches pickle file"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="output_segmentation_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--image-mpp", 
        type=float,
        default=0.5,
        help="Microns per pixel for the image"
    )
    parser.add_argument(
        "--log-level", 
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--skip-cell-matching", 
        action="store_true",
        help="Skip cell ID matching tests"
    )
    
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logging.info(f"Running segmentation analysis with args: {args}")

    try:
        # Load CodexPatches
        codex_patches = load_codex_patches(args.input)

        # Create test configuration
        config = create_test_config(args.output, args.image_mpp)

        # Run analysis
        run_analysis(
            codex_patches,
            config,
            args.output,
            not args.skip_cell_matching
        )

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 