#!/usr/bin/env python3
"""
Analyze cell morphology statistics to determine appropriate thresholds.
This helps establish data-driven thresholds for abnormal cells.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops

sys.path.insert(0, '/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline')


def analyze_cell_statistics(segmentation_pickle_path):
    """
    Analyze cell morphology statistics from segmentation results.
    
    Args:
        segmentation_pickle_path: Path to codex_patches_with_segmentation.pkl
    """
    # Load segmentation results
    with open(segmentation_pickle_path, 'rb') as f:
        codex_patches = pickle.load(f)
    
    # Collect all cell properties
    all_areas = []
    all_eccentricities = []
    all_solidities = []
    all_perimeters = []
    all_aspect_ratios = []
    
    for seg_result in codex_patches.repaired_seg_res_batch:
        if seg_result is None:
            continue
            
        cell_mask = seg_result.get('cell_matched_mask', seg_result.get('cell'))
        
        for region in regionprops(cell_mask):
            if region.label == 0:
                continue
                
            all_areas.append(region.area)
            all_eccentricities.append(region.eccentricity)
            all_solidities.append(region.solidity)
            all_perimeters.append(region.perimeter)
            
            # Calculate aspect ratio
            if region.minor_axis_length > 0:
                aspect_ratio = region.major_axis_length / region.minor_axis_length
                all_aspect_ratios.append(aspect_ratio)
    
    # Calculate statistics and percentiles
    print("\n=== Cell Morphology Statistics ===")
    print(f"Total cells analyzed: {len(all_areas)}")
    
    print("\n1. Cell Area (pixels²):")
    print(f"   Mean: {np.mean(all_areas):.1f} ± {np.std(all_areas):.1f}")
    print(f"   Median: {np.median(all_areas):.1f}")
    print(f"   Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"     {p}%: {np.percentile(all_areas, p):.1f}")
    
    print("\n2. Eccentricity (0=circle, 1=line):")
    print(f"   Mean: {np.mean(all_eccentricities):.3f} ± {np.std(all_eccentricities):.3f}")
    print(f"   Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"     {p}%: {np.percentile(all_eccentricities, p):.3f}")
    
    print("\n3. Solidity (0=irregular, 1=convex):")
    print(f"   Mean: {np.mean(all_solidities):.3f} ± {np.std(all_solidities):.3f}")
    print(f"   Percentiles:")
    for p in [1, 5, 10, 25, 50]:
        print(f"     {p}%: {np.percentile(all_solidities, p):.3f}")
    
    # Suggest thresholds based on percentiles
    print("\n=== Suggested Thresholds ===")
    print("Based on statistical analysis:")
    
    # Area thresholds (using 1st and 99th percentiles)
    area_low = np.percentile(all_areas, 1)
    area_high = np.percentile(all_areas, 99)
    print(f"\n1. Cell Area:")
    print(f"   Undersized: < {area_low:.0f} pixels² (1st percentile)")
    print(f"   Oversized: > {area_high:.0f} pixels² (99th percentile)")
    
    # Eccentricity threshold (using 95th percentile)
    ecc_threshold = np.percentile(all_eccentricities, 95)
    print(f"\n2. Eccentricity:")
    print(f"   High eccentricity: > {ecc_threshold:.3f} (95th percentile)")
    
    # Solidity threshold (using 5th percentile)
    sol_threshold = np.percentile(all_solidities, 5)
    print(f"\n3. Solidity:")
    print(f"   Low solidity: < {sol_threshold:.3f} (5th percentile)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Area histogram with thresholds
    axes[0, 0].hist(all_areas, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(area_low, color='red', linestyle='--', label=f'1%: {area_low:.0f}')
    axes[0, 0].axvline(area_high, color='red', linestyle='--', label=f'99%: {area_high:.0f}')
    axes[0, 0].set_xlabel('Cell Area (pixels²)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Cell Area Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Eccentricity histogram
    axes[0, 1].hist(all_eccentricities, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(ecc_threshold, color='red', linestyle='--', 
                       label=f'95%: {ecc_threshold:.3f}')
    axes[0, 1].set_xlabel('Eccentricity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Eccentricity Distribution')
    axes[0, 1].legend()
    
    # Solidity histogram
    axes[1, 0].hist(all_solidities, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(sol_threshold, color='red', linestyle='--',
                       label=f'5%: {sol_threshold:.3f}')
    axes[1, 0].set_xlabel('Solidity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Solidity Distribution')
    axes[1, 0].legend()
    
    # Scatter plot: Area vs Eccentricity
    axes[1, 1].scatter(all_areas, all_eccentricities, alpha=0.5, s=1)
    axes[1, 1].axhline(ecc_threshold, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(area_low, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(area_high, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Cell Area (pixels²)')
    axes[1, 1].set_ylabel('Eccentricity')
    axes[1, 1].set_title('Area vs Eccentricity')
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(segmentation_pickle_path)
    fig.savefig(os.path.join(output_dir, 'cell_threshold_analysis.png'), dpi=150)
    plt.close()
    
    # Save thresholds to file
    thresholds = {
        'area_undersized': float(area_low),
        'area_oversized': float(area_high),
        'eccentricity_high': float(ecc_threshold),
        'solidity_low': float(sol_threshold),
        'n_cells_analyzed': len(all_areas)
    }
    
    import json
    with open(os.path.join(output_dir, 'suggested_thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - cell_threshold_analysis.png")
    print(f"  - suggested_thresholds.json")
    
    return thresholds


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze cell statistics to determine thresholds')
    parser.add_argument('segmentation_pickle', 
                       help='Path to codex_patches_with_segmentation.pkl')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.segmentation_pickle):
        print(f"Error: File not found: {args.segmentation_pickle}")
        sys.exit(1)
    
    analyze_cell_statistics(args.segmentation_pickle)


if __name__ == '__main__':
    main()
