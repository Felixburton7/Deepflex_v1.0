"""
Data validation module for VoxelFlex (Temperature-Aware).

Provides validation functions for input data formats, focusing on RMSF data.
Voxel validation happens during loading rather than separately.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Use the centralized logger
logger = logging.getLogger("voxelflex.data")

def validate_aggregated_rmsf_data(rmsf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate aggregated RMSF DataFrame for required columns, types, and potential issues.
    
    Args:
        rmsf_df: DataFrame loaded from the aggregated RMSF CSV.
        
    Returns:
        Validated and potentially filtered DataFrame.
        
    Raises:
        ValueError: If input is empty or essential columns are missing/invalid.
    """
    logger.info("Validating aggregated RMSF data...")
    
    if not isinstance(rmsf_df, pd.DataFrame) or rmsf_df.empty:
        raise ValueError("Input RMSF data is not a non-empty DataFrame.")

    df_validated = rmsf_df.copy()

    # Define required columns and their expected types
    required_cols = {
        'domain_id': str,
        'resid': int,
        'resname': str,
        'temperature_feature': float,
        'target_rmsf': float
    }
    
    # Check for required columns
    missing_req = [col for col in required_cols if col not in df_validated.columns]
    if missing_req:
        raise ValueError(f"Aggregated RMSF data missing required columns: {missing_req}. Found: {list(df_validated.columns)}")

    # Optional columns for stratification and analysis
    optional_cols = ['relative_accessibility', 'dssp', 'secondary_structure_encoded']
    ss_col_found = 'dssp' in df_validated.columns or 'secondary_structure_encoded' in df_validated.columns
    available_optional = [col for col in optional_cols if col in df_validated.columns]
    
    # Log initial state
    initial_rows = len(df_validated)
    logger.info(f"Initial RMSF rows: {initial_rows}")

    # --- Optimize Type Conversion ---
    # Use predefined types for numeric columns - more efficient than multiple conversions
    type_mapping = {
        'resid': 'Int64',  # Use nullable Int type to handle potential NaNs
        'temperature_feature': float,
        'target_rmsf': float
    }
    
    # Convert only the columns needed (dtype dictionary for efficiency)
    for col, dtype in type_mapping.items():
        if col in df_validated.columns:
            try:
                df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
                if dtype == 'Int64':
                    df_validated[col] = df_validated[col].astype('Int64')
            except Exception as e:
                logger.warning(f"Error converting column '{col}': {e}")

    # Drop rows with NaNs in required columns
    orig_len = len(df_validated)
    df_validated.dropna(subset=list(required_cols.keys()), inplace=True)
    rows_dropped_nan = orig_len - len(df_validated)
    
    if rows_dropped_nan > 0:
        logger.info(f"Dropped {rows_dropped_nan} rows due to NaN/invalid values in required columns.")

    # Check and fix negative RMSF values
    if 'target_rmsf' in df_validated.columns:
        neg_rmsf_mask = df_validated['target_rmsf'] < 0
        neg_count = neg_rmsf_mask.sum()
        
        if neg_count > 0:
            logger.warning(f"Found {neg_count} negative 'target_rmsf' values. Setting them to 0.")
            df_validated.loc[neg_rmsf_mask, 'target_rmsf'] = 0.0

    # Check for duplicate entries (domain_id, resid, temperature_feature)
    key_cols = ['domain_id', 'resid', 'temperature_feature']
    if all(c in df_validated.columns for c in key_cols):
        # Handle the resid/Int64 conversion properly
        try:
            duplicates_mask = df_validated.duplicated(subset=key_cols, keep='first')
            dup_count = duplicates_mask.sum()
            
            if dup_count > 0:
                logger.warning(f"Found {dup_count} duplicate entries based on {key_cols}. Keeping first occurrence.")
                df_validated = df_validated[~duplicates_mask]
        except Exception as e:
            logger.warning(f"Could not check for duplicates: {e}")

    # --- Summarize Results ---
    final_rows = len(df_validated)
    
    if initial_rows > 0:
        percentage_str = f"({final_rows / initial_rows:.1%})"
    else:
        percentage_str = "(0.0%)"
        
    logger.info(f"RMSF validation finished. Valid rows: {final_rows} / {initial_rows} {percentage_str}.")

    if final_rows == 0:
        raise ValueError("No valid RMSF data remaining after validation.")

    # --- Log Summary Statistics ---
    logger.info("Summary statistics of validated RMSF data:")
    try:
        logger.info(f"  Unique Domains: {df_validated['domain_id'].nunique()}")
        logger.info(f"  Unique Temperatures: {sorted(df_validated['temperature_feature'].unique())}")
        logger.info(f"  Target RMSF Range: [{df_validated['target_rmsf'].min():.4f}, {df_validated['target_rmsf'].max():.4f}]")
        logger.info(f"  Target RMSF Mean: {df_validated['target_rmsf'].mean():.4f}")
    except Exception as e:
        logger.warning(f"Could not generate full summary statistics: {e}")

    return df_validated