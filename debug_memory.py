#!/usr/bin/env python3
"""
Memory Analysis Script for CODEX Pipeline OOM Debugging

This script helps identify memory bottlenecks by:
1. Analyzing image dimensions and estimating memory requirements
2. Providing optimization strategies based on the specific case
3. Testing different patching configurations
"""

import sys
import os
import yaml
import numpy as np
import logging
from pathlib import Path

# Add the source directory to the Python path
sys.path.append("/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline")

from aegle.memory_monitor import MemoryMonitor


def estimate_memory_requirements(config_file):
    """Analyze memory requirements for a given configuration"""
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("MEMORY ANALYSIS FOR CODEX PIPELINE")
    print("=" * 80)
    
    # From the log, we know the image dimensions
    height, width, channels = 28196, 24648, 39
    
    print(f"\n📊 IMAGE SPECIFICATIONS:")
    print(f"   Dimensions: {height} x {width} x {channels}")
    print(f"   Data type: uint16 (2 bytes per pixel)")
    
    # Calculate memory for different components
    total_pixels = height * width * channels
    bytes_per_pixel = 2  # uint16
    
    # Original image memory
    original_memory_gb = (total_pixels * bytes_per_pixel) / (1024**3)
    print(f"   Original image memory: {original_memory_gb:.2f} GB")
    
    # Extended image memory (usually same size unless padding needed)
    extended_memory_gb = original_memory_gb
    print(f"   Extended image memory: {extended_memory_gb:.2f} GB")
    
    # Target channels (2 channels: DAPI + merged whole-cell)
    target_channels = 2
    target_memory_gb = (height * width * target_channels * bytes_per_pixel) / (1024**3)
    print(f"   Target channels memory: {target_memory_gb:.2f} GB")
    
    # Patching configuration
    patch_height = config.get('patching', {}).get('patch_height', -1)
    patch_width = config.get('patching', {}).get('patch_width', -1)
    
    print(f"\n🔧 PATCHING CONFIGURATION:")
    print(f"   patch_height: {patch_height}")
    print(f"   patch_width: {patch_width}")
    
    if patch_height < 0 and patch_width < 0:
        print("   ⚠️  SINGLE PATCH MODE - Using entire image as one patch!")
        num_patches = 1
        patch_memory_gb = target_memory_gb
    else:
        # Calculate number of patches
        overlap = config.get('patching', {}).get('overlap', 0.1)
        step_height = int(patch_height * (1 - overlap))
        step_width = int(patch_width * (1 - overlap))
        
        patches_y = (height - patch_height) // step_height + 1
        patches_x = (width - patch_width) // step_width + 1
        num_patches = patches_y * patches_x
        
        patch_memory_gb = (patch_height * patch_width * target_channels * bytes_per_pixel * num_patches) / (1024**3)
    
    print(f"   Number of patches: {num_patches}")
    print(f"   Patches memory: {patch_memory_gb:.2f} GB")
    
    # Segmentation memory estimates
    print(f"\n🧠 SEGMENTATION MEMORY ESTIMATES:")
    
    # Model prediction output (typically 2 channels: cell + nucleus masks)
    seg_output_memory_gb = (height * width * 2 * 4) / (1024**3)  # float32 initially
    print(f"   Model output (float32): {seg_output_memory_gb:.2f} GB")
    
    # After uint32 conversion
    seg_uint32_memory_gb = (height * width * 2 * 4) / (1024**3)  # uint32
    print(f"   After uint32 conversion: {seg_uint32_memory_gb:.2f} GB")
    
    # Boundary masks
    boundary_memory_gb = seg_uint32_memory_gb  # Same size as masks
    print(f"   Boundary masks: {boundary_memory_gb:.2f} GB")
    
    # Total peak memory estimate
    print(f"\n💾 PEAK MEMORY ESTIMATES:")
    
    # During segmentation
    peak_during_seg = (
        original_memory_gb +          # Original image
        target_memory_gb +            # Target channels
        patch_memory_gb +             # Patches
        seg_output_memory_gb +        # Segmentation output
        seg_uint32_memory_gb +        # Converted masks
        boundary_memory_gb            # Boundary masks
    )
    print(f"   Peak during segmentation: {peak_during_seg:.2f} GB")
    
    # Add TensorFlow/GPU memory overhead (estimated)
    tf_overhead_gb = 2.0  # Conservative estimate
    total_peak_gb = peak_during_seg + tf_overhead_gb
    print(f"   Total peak (with TF overhead): {total_peak_gb:.2f} GB")
    
    # System analysis
    monitor = MemoryMonitor()
    mem_info = monitor.get_memory_info()
    
    print(f"\n🖥️  SYSTEM RESOURCES:")
    print(f"   Total system memory: {mem_info['system']['total']:.2f} GB")
    print(f"   Available memory: {mem_info['system']['available']:.2f} GB")
    print(f"   Memory usage: {mem_info['system']['percent']:.1f}%")
    
    # Risk assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    available_gb = mem_info['system']['available']
    
    if total_peak_gb > available_gb:
        print(f"   🔴 HIGH RISK: Peak memory ({total_peak_gb:.2f} GB) > Available ({available_gb:.2f} GB)")
        print(f"   💡 Deficit: {total_peak_gb - available_gb:.2f} GB")
    elif total_peak_gb > available_gb * 0.8:
        print(f"   🟡 MEDIUM RISK: Peak memory ({total_peak_gb:.2f} GB) > 80% of available ({available_gb:.2f} GB)")
    else:
        print(f"   🟢 LOW RISK: Peak memory ({total_peak_gb:.2f} GB) within safe limits")
    
    return {
        'peak_memory_gb': total_peak_gb,
        'available_gb': available_gb,
        'num_patches': num_patches,
        'patch_memory_gb': patch_memory_gb,
        'original_memory_gb': original_memory_gb
    }


def suggest_optimizations(analysis_result):
    """Suggest optimization strategies based on memory analysis"""
    
    print(f"\n🔧 OPTIMIZATION STRATEGIES:")
    print("=" * 50)
    
    peak_gb = analysis_result['peak_memory_gb']
    available_gb = analysis_result['available_gb']
    num_patches = analysis_result['num_patches']
    
    if peak_gb > available_gb:
        print("🚨 IMMEDIATE ACTIONS NEEDED:")
        
        if num_patches == 1:
            print("1. 📐 ENABLE PATCHING:")
            print("   - Change patch_height and patch_width to positive values")
            print("   - Recommended: 2048x2048 or 1024x1024 patches")
            print("   - Example: patch_height: 2048, patch_width: 2048")
            
        print("\n2. 🧹 MEMORY MANAGEMENT:")
        print("   - Add del statements after processing each component")
        print("   - Call gc.collect() more frequently")
        print("   - Process patches in smaller batches")
        
        print("\n3. 💾 DISK OPTIMIZATION:")
        print("   - Save intermediate results to disk")
        print("   - Use memory-mapped arrays (np.memmap)")
        print("   - Set save_all_channel_patches: false")
        
        print("\n4. 🔧 PROCESSING OPTIMIZATIONS:")
        print("   - Downsample image before processing")
        print("   - Use lower precision where possible (uint8 vs uint16)")
        print("   - Process channels sequentially instead of all at once")
    
    print(f"\n📋 RECOMMENDED CONFIGURATION CHANGES:")
    print("patching:")
    print("  patch_height: 2048")
    print("  patch_width: 2048") 
    print("  overlap: 0.1")
    print("")
    print("visualization:")
    print("  save_all_channel_patches: false")
    print("  visualize_patches: false")
    print("")
    print("segmentation:")
    print("  save_segmentation_images: false  # Until memory is optimized")
    

def test_patch_configurations():
    """Test different patching configurations and their memory impact"""
    
    print(f"\n🧪 PATCH CONFIGURATION TESTING:")
    print("=" * 50)
    
    height, width, channels = 28196, 24648, 2  # Target channels
    bytes_per_pixel = 2
    
    configs = [
        (4096, 4096, "Large patches"),
        (2048, 2048, "Medium patches"), 
        (1024, 1024, "Small patches"),
        (512, 512, "Very small patches")
    ]
    
    for patch_h, patch_w, desc in configs:
        overlap = 0.1
        step_h = int(patch_h * (1 - overlap))
        step_w = int(patch_w * (1 - overlap))
        
        patches_y = (height - patch_h) // step_h + 1
        patches_x = (width - patch_w) // step_w + 1
        num_patches = patches_y * patches_x
        
        patch_memory_gb = (patch_h * patch_w * channels * bytes_per_pixel * num_patches) / (1024**3)
        single_patch_gb = (patch_h * patch_w * channels * bytes_per_pixel) / (1024**3)
        
        print(f"{desc:20s}: {num_patches:5d} patches, {patch_memory_gb:6.2f} GB total, {single_patch_gb:6.3f} GB each")


if __name__ == "__main__":
    config_file = "/workspaces/codex-analysis/0-phenocycler-penntmc-pipeline/exps/configs/main/test0206_main/D16_Scan1_2/config.yaml"
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    
    analysis = estimate_memory_requirements(config_file)
    suggest_optimizations(analysis)
    test_patch_configurations()
    
    print(f"\n" + "=" * 80)
    print("💡 NEXT STEPS:")
    print("1. Modify your config.yaml with the recommended settings")
    print("2. Run with memory monitoring: python src/main.py --log_level DEBUG ...")
    print("3. Monitor logs for memory usage patterns")
    print("4. Adjust patch sizes based on actual memory consumption")
    print("=" * 80) 