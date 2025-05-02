#!/usr/bin/env python
"""
Script to merge external RMSF predictions with the main protein dataset.

This script reads two CSV files:
1. A file containing external RMSF predictions (with columns: domain_id, resid, resname, voxel_rmsf, esm_rmsf)
2. The main dataset file with protein features

It merges these files based on matching domain_id, resid, and resname keys,
adding the voxel_rmsf and esm_rmsf columns to the main dataset.
"""

import os
import argparse
import logging
import pandas as pd
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, required_cols: list, name: str) -> bool:
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        name: Name of the dataset for logging
        
    Returns:
        True if valid, False otherwise
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns in {name}: {', '.join(missing_cols)}")
        return False
    
    return True

def merge_rmsf_data(
    main_file: str, 
    rmsf_file: str, 
    output_file: Optional[str] = None
) -> Tuple[pd.DataFrame, bool]:
    """
    Merge the main dataset with external RMSF predictions.
    
    Args:
        main_file: Path to the main dataset CSV
        rmsf_file: Path to the external RMSF predictions CSV
        output_file: Optional path to save merged dataset
        
    Returns:
        Tuple of (merged DataFrame, success flag)
    """
    # Check if files exist
    if not os.path.exists(main_file):
        logger.error(f"Main dataset file not found: {main_file}")
        return None, False
    
    if not os.path.exists(rmsf_file):
        logger.error(f"RMSF predictions file not found: {rmsf_file}")
        return None, False
    
    # Load datasets
    try:
        logger.info(f"Loading main dataset: {main_file}")
        main_df = pd.read_csv(main_file)
        
        logger.info(f"Loading RMSF predictions: {rmsf_file}")
        rmsf_df = pd.read_csv(rmsf_file)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, False
    
    # Validate columns
    main_required = ["domain_id", "resid", "resname"]
    
    # Check if at least one RMSF predictor is present
    rmsf_predictors = []
    for col in ["voxel_rmsf", "esm_rmsf"]:
        if col in rmsf_df.columns:
            rmsf_predictors.append(col)
    
    if not rmsf_predictors:
        logger.error("No RMSF predictor columns (voxel_rmsf, esm_rmsf) found in the predictions file")
        return None, False
    
    # Required columns including the merge keys and at least one RMSF predictor
    rmsf_required = ["domain_id", "resid", "resname"] + [rmsf_predictors[0]]
    
    if not validate_data(main_df, main_required, "main dataset"):
        return None, False
    
    if not validate_data(rmsf_df, rmsf_required, "RMSF predictions dataset"):
        return None, False
    
    # Log dataset info
    logger.info(f"Main dataset: {len(main_df)} rows, {len(main_df.columns)} columns")
    logger.info(f"RMSF predictions dataset: {len(rmsf_df)} rows, {len(rmsf_df.columns)} columns")
    logger.info(f"RMSF predictors found: {', '.join(rmsf_predictors)}")
    
    # Merge datasets
    try:
        # Select only the necessary columns from the RMSF dataset
        rmsf_cols = ["domain_id", "resid", "resname"] + rmsf_predictors
        
        logger.info("Merging datasets on domain_id, resid, and resname")
        merged_df = pd.merge(
            main_df, 
            rmsf_df[rmsf_cols], 
            on=["domain_id", "resid", "resname"],
            how="left"
        )
        
        logger.info(f"Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        
        # Check for missing values in RMSF predictors
        for predictor in rmsf_predictors:
            missing_count = merged_df[predictor].isna().sum()
            if missing_count > 0:
                logger.warning(f"Missing {predictor} values for {missing_count} rows ({missing_count/len(merged_df)*100:.2f}%)")
        
        # Save merged dataset if output_file is provided
        if output_file:
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            merged_df.to_csv(output_file, index=False)
            logger.info(f"Merged dataset saved to {output_file}")
        
        return merged_df, True
    
    except Exception as e:
        logger.error(f"Error merging datasets: {e}")
        return None, False

def main():
    """Main function to parse arguments and call merge_rmsf_data."""
    parser = argparse.ArgumentParser(description="Merge external RMSF predictions with the main protein dataset.")
    parser.add_argument("main_file", help="Path to the main dataset CSV file")
    parser.add_argument("rmsf_file", help="Path to the RMSF predictions CSV file")
    parser.add_argument("-o", "--output", help="Path to save the merged dataset CSV file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # If output is not provided, generate default output path
    output_file = args.output
    if not output_file:
        main_name = os.path.splitext(os.path.basename(args.main_file))[0]
        output_file = f"{main_name}_with_rmsf_predictions.csv"
    
    # Call merge function
    merged_df, success = merge_rmsf_data(args.main_file, args.rmsf_file, output_file)
    
    if not success:
        logger.error("Merge operation failed")
        return 1
    
    logger.info("Merge operation completed successfully")
    return 0

if __name__ == "__main__":
    main()