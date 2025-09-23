#!/usr/bin/env python3
"""
Test script for segmentation visualization features.
This demonstrates how to use the new visualization functions.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add the pipeline directory to path
sys.path.insert(0, '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline')

from aegle.visualization_segmentation import (
    create_segmentation_overlay,
    create_quality_heatmaps,
    plot_cell_morphology_stats,
    visualize_segmentation_errors,
    create_segmentation_comparison
)


def test_visualization_on_existing_results(output_dir):
    """
    Test visualization functions on existing segmentation results.
    
    Args:
        output_dir: Directory containing pipeline outputs
    """
    # Check if segmentation pickle exists
    seg_pickle = os.path.join(output_dir, "segmentation", "codex_patches_with_segmentation.pkl")
    if not os.path.exists(seg_pickle):
        print(f"Segmentation pickle not found: {seg_pickle}")
        return
    
    # Load segmentation results
    print("Loading segmentation results...")
    with open(seg_pickle, 'rb') as f:
        codex_patches = pickle.load(f)
    
    # Create test visualization directory
    test_viz_dir = os.path.join(output_dir, "test_segmentation_viz")
    os.makedirs(test_viz_dir, exist_ok=True)
    
    # Test 1: Single patch overlay
    print("\n1. Testing segmentation overlay...")
    if hasattr(codex_patches, 'repaired_seg_res_batch') and len(codex_patches.repaired_seg_res_batch) > 0:
        seg_result = codex_patches.repaired_seg_res_batch[0]
        orig_seg_result = None
        if hasattr(codex_patches, 'original_seg_res_batch') and len(codex_patches.original_seg_res_batch) > 0:
            orig_seg_result = codex_patches.original_seg_res_batch[0]
        
        # Get image patch
        if hasattr(codex_patches, 'valid_patches') and len(codex_patches.valid_patches) > 0:
            image_patch = codex_patches.valid_patches[0]
            
            # Create overlay with different options
            fig = create_segmentation_overlay(
                image_patch[:, :, 0],  # Nuclear channel
                seg_result.get("nucleus_matched_mask", seg_result.get("nucleus")),
                seg_result.get("cell_matched_mask", seg_result.get("cell")),
                show_ids=True,
                alpha=0.6,
                reference_nucleus_mask=orig_seg_result.get("nucleus") if orig_seg_result else None,
                reference_cell_mask=orig_seg_result.get("cell") if orig_seg_result else None,
            )
            fig.savefig(os.path.join(test_viz_dir, "test_overlay_with_ids.png"), dpi=150)
            plt.close(fig)
            print("  ✓ Saved overlay with cell IDs")
    
    # Test 2: Error visualization
    print("\n2. Testing error visualization...")
    if hasattr(codex_patches, 'repaired_seg_res_batch') and len(codex_patches.repaired_seg_res_batch) > 0:
        seg_result = codex_patches.repaired_seg_res_batch[0]
        
        if hasattr(codex_patches, 'valid_patches') and len(codex_patches.valid_patches) > 0:
            image_patch = codex_patches.valid_patches[0]
            
            fig = visualize_segmentation_errors(
                image_patch[:, :, 0],
                seg_result,
                error_types=['oversized', 'undersized', 'high_eccentricity', 'unmatched'],
                thresholds={'oversized': 1500, 'undersized': 100, 'high_eccentricity': 0.9}
            )
            fig.savefig(os.path.join(test_viz_dir, "test_error_detection.png"), dpi=150)
            plt.close(fig)
            print("  ✓ Saved error detection visualization")
    
    # Test 3: Morphology statistics
    print("\n3. Testing morphology statistics...")
    if hasattr(codex_patches, 'repaired_seg_res_batch'):
        fig = plot_cell_morphology_stats(
            codex_patches.repaired_seg_res_batch,
            test_viz_dir
        )
        plt.close(fig)
        print("  ✓ Saved morphology statistics")
    
    # Test 4: Comparison visualization (if we have both original and repaired)
    print("\n4. Testing segmentation comparison...")
    if (hasattr(codex_patches, 'original_seg_res_batch') and 
        hasattr(codex_patches, 'repaired_seg_res_batch') and
        len(codex_patches.original_seg_res_batch) > 0):
        
        if hasattr(codex_patches, 'valid_patches') and len(codex_patches.valid_patches) > 0:
            image_patch = codex_patches.valid_patches[0]
            
            comparison_dict = {
                'Original': codex_patches.original_seg_res_batch[0],
                'Repaired': codex_patches.repaired_seg_res_batch[0]
            }
            
            fig = create_segmentation_comparison(
                image_patch[:, :, 0],
                comparison_dict
            )
            fig.savefig(os.path.join(test_viz_dir, "test_segmentation_comparison.png"), dpi=150)
            plt.close(fig)
            print("  ✓ Saved segmentation comparison")
    
    print(f"\nAll test visualizations saved to: {test_viz_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test segmentation visualization features')
    parser.add_argument('output_dir', help='Pipeline output directory containing segmentation results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    test_visualization_on_existing_results(args.output_dir)


if __name__ == '__main__':
    main()
