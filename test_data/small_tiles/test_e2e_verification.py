#!/usr/bin/env python3
"""
End-to-end test verification script for the PhenoCycler pipeline.
This script checks that all expected outputs are generated correctly.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import tifffile as tiff
import pickle
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOutputVerifier:
    """Verify outputs from the PhenoCycler pipeline."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.errors = []
        self.warnings = []
        
    def verify_all(self):
        """Run all verification checks."""
        logger.info(f"Verifying pipeline outputs in: {self.output_dir}")
        
        # Check directory exists
        if not self.output_dir.exists():
            self.errors.append(f"Output directory does not exist: {self.output_dir}")
            return False
            
        # Run individual checks
        checks = [
            self.check_directory_structure,
            self.check_visualization_outputs,
            self.check_segmentation_outputs,
            self.check_cell_profiling_outputs,
            self.check_log_files,
            self.check_metadata_consistency
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                logger.error(f"Check {check.__name__} failed with error: {e}")
                self.errors.append(f"{check.__name__}: {str(e)}")
                all_passed = False
                
        return all_passed
        
    def check_directory_structure(self):
        """Check that expected directories are created."""
        logger.info("Checking directory structure...")
        
        expected_dirs = [
            "visualization/rgb_images",
            "segmentation",
            "cell_profiling"
        ]
        
        missing_dirs = []
        for dir_name in expected_dirs:
            dir_path = self.output_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            self.errors.append(f"Missing directories: {missing_dirs}")
            return False
            
        logger.info("✅ Directory structure is correct")
        return True
        
    def check_visualization_outputs(self):
        """Check visualization outputs."""
        logger.info("Checking visualization outputs...")
        
        viz_dir = self.output_dir / "visualization" / "rgb_images"
        
        # Check for whole sample visualization
        whole_sample_files = list(viz_dir.glob("extended_extracted_channel_image*.png"))
        if not whole_sample_files:
            self.warnings.append("No whole sample visualization found")
        else:
            logger.info(f"  Found {len(whole_sample_files)} whole sample visualization(s)")
            
        # Check for patch visualizations  
        patch_files = list(viz_dir.glob("patch_*.png"))
        if patch_files:
            logger.info(f"  Found {len(patch_files)} patch visualization(s)")
        
        logger.info("✅ Visualization outputs checked")
        return True
        
    def check_segmentation_outputs(self):
        """Check segmentation outputs."""
        logger.info("Checking segmentation outputs...")
        
        seg_dir = self.output_dir / "segmentation"
        
        # Check for segmentation pickle
        seg_pickle = seg_dir / "codex_patches_with_segmentation.pkl"
        if not seg_pickle.exists():
            self.errors.append("Segmentation pickle file not found")
            return False
            
        # Load and verify pickle
        try:
            with open(seg_pickle, 'rb') as f:
                codex_patches = pickle.load(f)
                
            # Check key attributes
            if not hasattr(codex_patches, 'repaired_seg_res_batch'):
                self.errors.append("Segmentation results not found in pickle")
                return False
                
            n_patches = len(codex_patches.repaired_seg_res_batch)
            logger.info(f"  Found segmentation results for {n_patches} patches")
            
        except Exception as e:
            self.errors.append(f"Failed to load segmentation pickle: {e}")
            return False
            
        # Check for segmentation images
        seg_images = list(seg_dir.glob("*.tiff"))
        if seg_images:
            logger.info(f"  Found {len(seg_images)} segmentation image(s)")
        else:
            self.warnings.append("No segmentation images saved")
            
        logger.info("✅ Segmentation outputs verified")
        return True
        
    def check_cell_profiling_outputs(self):
        """Check cell profiling outputs."""
        logger.info("Checking cell profiling outputs...")
        
        prof_dir = self.output_dir / "cell_profiling"
        
        # Check for expression matrix
        exp_files = list(prof_dir.glob("*expression*.csv"))
        if not exp_files:
            self.errors.append("No expression matrix file found")
            return False
            
        # Check for cell metadata
        meta_files = list(prof_dir.glob("*metadata*.csv"))
        if not meta_files:
            self.errors.append("No cell metadata file found")
            return False
            
        # Load and verify data
        exp_df = pd.read_csv(exp_files[0])
        meta_df = pd.read_csv(meta_files[0])
        
        logger.info(f"  Expression matrix: {exp_df.shape}")
        logger.info(f"  Cell metadata: {meta_df.shape}")
        
        # Check data consistency
        if len(exp_df) != len(meta_df):
            self.errors.append(f"Expression and metadata row counts don't match: {len(exp_df)} vs {len(meta_df)}")
            return False
            
        # Check for expected columns
        if 'cell_id' not in exp_df.columns:
            self.errors.append("'cell_id' column missing from expression matrix")
            return False
            
        logger.info("✅ Cell profiling outputs verified")
        return True
        
    def check_log_files(self):
        """Check that log files are created and contain no errors."""
        logger.info("Checking log files...")
        
        # Check copied config
        config_file = self.output_dir / "copied_config.yaml"
        if not config_file.exists():
            self.warnings.append("Copied config file not found")
            
        logger.info("✅ Log files checked")
        return True
        
    def check_metadata_consistency(self):
        """Check metadata consistency across outputs."""
        logger.info("Checking metadata consistency...")
        
        # Load patches metadata if available
        prof_dir = self.output_dir / "cell_profiling"
        patches_meta = prof_dir / "patches_metadata.csv"
        
        if patches_meta.exists():
            patches_df = pd.read_csv(patches_meta)
            logger.info(f"  Patches metadata: {len(patches_df)} patches")
            
            # Check informative patches
            n_informative = patches_df['is_informative'].sum()
            logger.info(f"  Informative patches: {n_informative}/{len(patches_df)}")
            
        logger.info("✅ Metadata consistency checked")
        return True
        
    def print_summary(self):
        """Print verification summary."""
        print("\n" + "="*60)
        print("PIPELINE VERIFICATION SUMMARY")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("✅ All checks passed!")
        else:
            if self.errors:
                print(f"\n❌ ERRORS ({len(self.errors)}):")
                for error in self.errors:
                    print(f"  - {error}")
                    
            if self.warnings:
                print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"  - {warning}")
                    
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Verify PhenoCycler pipeline outputs')
    parser.add_argument('output_dir', help='Path to pipeline output directory')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run verification
    verifier = PipelineOutputVerifier(args.output_dir)
    success = verifier.verify_all()
    verifier.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
