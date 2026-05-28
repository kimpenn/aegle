import os
import argparse
import logging
import pandas as pd
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

def apply_morphology_filters(profiling_out_dir: str, qc_config: Dict[str, Any]):
    """
    Applies quality control filters to cell metadata and intensities.
    Expects profiling_out_dir to contain 'cell_metadata.csv' and 'cell_by_marker.csv'.
    """
    metadata_path = os.path.join(profiling_out_dir, "cell_metadata.csv")
    expression_path = os.path.join(profiling_out_dir, "cell_by_marker.csv")
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Could not find {metadata_path}. Skipping QC filtering.")
        return
        
    logger.info(f"Loading metadata from {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    
    original_count = len(metadata_df)
    valid_mask = pd.Series(True, index=metadata_df.index)
    
    rules = qc_config.get("rules", {})
    if not rules:
        logger.info("No QC rules defined in config. Returning original data.")
        return

    logger.info("Applying morphology QC rules...")
    for column, bounds in rules.items():
        if column not in metadata_df.columns:
            logger.warning(f"Column '{column}' not found in metadata. Skipping rule.")
            continue
            
        if "min" in bounds:
            min_val = bounds["min"]
            kept = metadata_df[column] >= min_val
            valid_mask &= kept
            logger.info(f"Rule: {column} >= {min_val} (Kept {kept.sum()})")
            
        if "max" in bounds:
            max_val = bounds["max"]
            kept = metadata_df[column] <= max_val
            valid_mask &= kept
            logger.info(f"Rule: {column} <= {max_val} (Kept {kept.sum()})")

    passed_count = valid_mask.sum()
    logger.info(f"QC Filtering Complete: {passed_count}/{original_count} cells passed ({(passed_count/original_count)*100:.2f}%).")
    
    filtered_metadata = metadata_df[valid_mask]
    filtered_meta_path = os.path.join(profiling_out_dir, "filtered_cell_metadata.csv")
    filtered_metadata.to_csv(filtered_meta_path, index=False)
    logger.info(f"Saved filtered metadata to {filtered_meta_path}")
    
    # Filter and Save Expression Data if it exists
    if os.path.exists(expression_path):
        exp_df = pd.read_csv(expression_path)
        if len(exp_df) == original_count:
            filtered_exp = exp_df[valid_mask]
            filtered_exp_path = os.path.join(profiling_out_dir, "filtered_cell_by_marker.csv")
            filtered_exp.to_csv(filtered_exp_path, index=False)
            logger.info(f"Saved filtered cell-by-marker to {filtered_exp_path}")
        else:
            logger.warning("Row count mismatch between metadata and expression data. Cannot safely filter expression data.")
            
    # Also attempt to filter overview csvs
    for overview_name in ["cell_overview.csv", "nucleus_overview.csv"]:
        overview_path = os.path.join(profiling_out_dir, overview_name)
        if os.path.exists(overview_path):
            overview_df = pd.read_csv(overview_path)
            if len(overview_df) == original_count:
                filtered_overview = overview_df[valid_mask]
                filtered_overview_path = os.path.join(profiling_out_dir, f"filtered_{overview_name}")
                filtered_overview.to_csv(filtered_overview_path, index=False)
                logger.info(f"Saved filtered overview to {filtered_overview_path}")


if __name__ == "__main__":
    # Ensure standard logging when run as standalone script
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Filter cell metadata and marker profiles based on QC metrics (e.g. morphology, nucleus_retained_fraction).")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing cell_metadata.csv and cell_by_marker.csv (Usually out_dir/cell_profiling)")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to YAML configuration file with filtering rules")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # The config will have the rules nested under 'qc_filtering'
    qc_config = config.get("qc_filtering", config)
    
    apply_morphology_filters(args.input_dir, qc_config)
