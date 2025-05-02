#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import logging
from typing import Optional

# --- Configuration (Define Paths and Columns Here) ---

# Input Files
# ORIGINAL_CSV_PATH = "/home/s_felix/ESM-Flex-2/data/raw/aggregated_holdout_dataset.csv"
ORIGINAL_CSV_PATH = "/home/s_felix/drDataScience/data/analysis_complete_holdout_dataset.csv"
PREDICTIONS_CSV_PATH = "/home/s_felix/ESM-Flex-2/predictions/holdout_set_results_latest/prediction_from_npy_train_temperatures_mc10/predictions_prediction_from_npy_train_temperatures_mc10.csv"

# Output File
OUTPUT_CSV_PATH = "/home/s_felix/drDataScience/data/final_analysis_dataset.csv"

# Column Names
# In Original CSV (input)
ORIG_DOMAIN_ID_COL = 'domain_id'
ORIG_TEMP_COL = 'temperature' # Adjust if it's 'temperature_feature' in your raw file
ORIG_RESID_COL = 'resid'
# In Prediction CSV (input)
PRED_INSTANCE_KEY_COL = 'instance_key'
PRED_RESID_COL = 'resid' # This is the 1-based relative index
PRED_RMSF_COL = 'rmsf_pred'
PRED_UNCERTAINTY_COL = 'uncertainty'
# In Merged CSV (output)
NEW_RMSF_COL = 'Attention_ESM_rmsf'
NEW_UNCERTAINTY_COL = 'Attention_ESM_uncertainty'

# Internal temporary column name for relative index
RELATIVE_RESID_IDX_COL = '_relative_resid_idx'
TEMP_INSTANCE_KEY_COL = '_instance_key' # Temporary key column in original df

# Separator (Should match data_processor.py)
INSTANCE_KEY_SEPARATOR = "@"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function ---
def create_instance_key(domain_id: str, temperature: float) -> Optional[str]:
    """Creates a unique key combining domain ID and temperature."""
    try:
        # Ensure temperature is float before formatting
        temp_float = float(temperature)
        # Use consistent formatting (e.g., one decimal place)
        return f"{str(domain_id)}{INSTANCE_KEY_SEPARATOR}{temp_float:.1f}"
    except (ValueError, TypeError):
        logger.error(f"Could not format instance key for domain '{domain_id}', temp '{temperature}'")
        return None # Return None on error

# --- Main Merge Logic ---
def merge_data():
    """
    Loads original data and predictions, merges them based on instance key
    and relative residue index, and saves the combined dataset.
    """
    # --- Load Original Data ---
    logger.info(f"Loading original data from: {ORIGINAL_CSV_PATH}")
    if not os.path.exists(ORIGINAL_CSV_PATH):
        logger.error(f"Original data file not found: {ORIGINAL_CSV_PATH}")
        return
    try:
        df_orig = pd.read_csv(ORIGINAL_CSV_PATH)
        logger.info(f"Loaded {len(df_orig)} rows from original data.")
    except Exception as e:
        logger.error(f"Error loading original data: {e}", exc_info=True)
        return

    # --- Load Prediction Data ---
    logger.info(f"Loading prediction data from: {PREDICTIONS_CSV_PATH}")
    if not os.path.exists(PREDICTIONS_CSV_PATH):
        logger.error(f"Predictions data file not found: {PREDICTIONS_CSV_PATH}")
        return
    try:
        df_pred = pd.read_csv(PREDICTIONS_CSV_PATH)
        logger.info(f"Loaded {len(df_pred)} rows from predictions.")
    except Exception as e:
        logger.error(f"Error loading predictions data: {e}", exc_info=True)
        return

    # --- Prepare Original DataFrame ---
    logger.info("Preparing original DataFrame...")
    try:
        # 1. Check and rename temperature column
        if ORIG_TEMP_COL not in df_orig.columns:
            logger.error(f"Original CSV missing specified temperature column: '{ORIG_TEMP_COL}'")
            return
        # No need to rename internally if ORIG_TEMP_COL is correct

        # 2. Ensure essential columns exist and clean types/NaNs
        essential_orig_cols = [ORIG_DOMAIN_ID_COL, ORIG_TEMP_COL, ORIG_RESID_COL]
        if not all(col in df_orig.columns for col in essential_orig_cols):
            missing = [c for c in essential_orig_cols if c not in df_orig.columns]
            logger.error(f"Original CSV missing essential columns for merging: {missing}")
            return

        df_orig[ORIG_TEMP_COL] = pd.to_numeric(df_orig[ORIG_TEMP_COL], errors='coerce')
        df_orig[ORIG_RESID_COL] = pd.to_numeric(df_orig[ORIG_RESID_COL], errors='coerce')
        df_orig.dropna(subset=[ORIG_DOMAIN_ID_COL, ORIG_TEMP_COL, ORIG_RESID_COL], inplace=True)
        df_orig[ORIG_RESID_COL] = df_orig[ORIG_RESID_COL].astype(int)
        df_orig[ORIG_DOMAIN_ID_COL] = df_orig[ORIG_DOMAIN_ID_COL].astype(str)
        logger.info(f"{len(df_orig)} original rows remaining after cleaning essential columns.")
        if df_orig.empty:
             logger.error("No valid rows left in original data after cleaning.")
             return

        # 3. Create temporary 'instance_key' column
        df_orig[TEMP_INSTANCE_KEY_COL] = df_orig.apply(
            lambda row: create_instance_key(row[ORIG_DOMAIN_ID_COL], row[ORIG_TEMP_COL]),
            axis=1
        )
        orig_len = len(df_orig)
        df_orig.dropna(subset=[TEMP_INSTANCE_KEY_COL], inplace=True) # Drop rows where key creation failed
        if len(df_orig) < orig_len:
            logger.warning(f"Dropped {orig_len - len(df_orig)} original rows due to invalid instance key creation (check domain IDs/temps).")

        # 4. Create relative 1-based residue index within each instance group
        logger.info("Calculating relative residue index...")
        # Sort first to ensure correct ordering for rank/cumcount
        df_orig = df_orig.sort_values(by=[TEMP_INSTANCE_KEY_COL, ORIG_RESID_COL])
        # Calculate 0-based count within group, add 1 for 1-based index
        df_orig[RELATIVE_RESID_IDX_COL] = df_orig.groupby(TEMP_INSTANCE_KEY_COL).cumcount() + 1
        logger.info("Relative residue index calculated.")

    except Exception as e:
        logger.error(f"Error preparing original DataFrame: {e}", exc_info=True)
        return

    # --- Prepare Prediction DataFrame ---
    logger.info("Preparing prediction DataFrame...")
    try:
        # 1. Check required columns
        required_pred_cols = [PRED_INSTANCE_KEY_COL, PRED_RESID_COL, PRED_RMSF_COL, PRED_UNCERTAINTY_COL]
        if not all(col in df_pred.columns for col in required_pred_cols):
            missing = [c for c in required_pred_cols if c not in df_pred.columns]
            logger.error(f"Prediction CSV missing required columns: {missing}")
            return

        # 2. Rename columns for merging and final output
        df_pred = df_pred.rename(columns={
            # Keep PRED_INSTANCE_KEY_COL for merging
            PRED_RESID_COL: RELATIVE_RESID_IDX_COL, # Match the relative index column
            PRED_RMSF_COL: NEW_RMSF_COL,
            PRED_UNCERTAINTY_COL: NEW_UNCERTAINTY_COL
        })

        # 3. Ensure correct types for merge keys and data
        df_pred[PRED_INSTANCE_KEY_COL] = df_pred[PRED_INSTANCE_KEY_COL].astype(str)
        df_pred[RELATIVE_RESID_IDX_COL] = pd.to_numeric(df_pred[RELATIVE_RESID_IDX_COL], errors='coerce')
        df_pred.dropna(subset=[RELATIVE_RESID_IDX_COL], inplace=True) # Drop if resid conversion failed
        df_pred[RELATIVE_RESID_IDX_COL] = df_pred[RELATIVE_RESID_IDX_COL].astype(int)
        df_pred[NEW_RMSF_COL] = pd.to_numeric(df_pred[NEW_RMSF_COL], errors='coerce')
        df_pred[NEW_UNCERTAINTY_COL] = pd.to_numeric(df_pred[NEW_UNCERTAINTY_COL], errors='coerce')

        # Select only necessary columns for merge to avoid duplicate columns like 'temperature'
        df_pred_to_merge = df_pred[[PRED_INSTANCE_KEY_COL, RELATIVE_RESID_IDX_COL, NEW_RMSF_COL, NEW_UNCERTAINTY_COL]]

    except Exception as e:
        logger.error(f"Error preparing prediction DataFrame: {e}", exc_info=True)
        return

    # --- Perform Merge ---
    logger.info("Merging original data with predictions...")
    try:
        # Use left merge to keep all rows from the original dataframe
        df_merged = pd.merge(
            df_orig,
            df_pred_to_merge,
            left_on=[TEMP_INSTANCE_KEY_COL, RELATIVE_RESID_IDX_COL], # Keys from original df
            right_on=[PRED_INSTANCE_KEY_COL, RELATIVE_RESID_IDX_COL], # Keys from prediction df
            how='left',
            suffixes=('', '_pred') # Add suffix to prediction key if it overlaps, though we selected columns
        )
        logger.info(f"Merge completed. Resulting DataFrame has {len(df_merged)} rows.")

        # Verify merge quality (optional but recommended)
        num_matched = df_merged[NEW_RMSF_COL].notna().sum()
        num_unmatched_orig = len(df_orig) - num_matched # Rows in orig without a direct pred match
        num_unmatched_pred = len(df_pred) - num_matched # Rows in pred without a direct orig match (should be less common with left merge)
        logger.info(f"Successfully merged predictions for {num_matched} rows.")
        if num_unmatched_orig > 0:
            logger.warning(f"{num_unmatched_orig} rows from the original dataset did not have a corresponding prediction entry (NaNs added).")
        if num_unmatched_pred > 0:
            logger.warning(f"{num_unmatched_pred} rows from the prediction dataset did not match any entry in the original dataset (check keys/indices).")


        # Clean up temporary columns used for merging
        df_merged = df_merged.drop(columns=[TEMP_INSTANCE_KEY_COL, RELATIVE_RESID_IDX_COL])
        # Drop potentially duplicated instance key from predictions if suffix was added (unlikely with column selection)
        if f"{PRED_INSTANCE_KEY_COL}_pred" in df_merged.columns:
             df_merged = df_merged.drop(columns=[f"{PRED_INSTANCE_KEY_COL}_pred"])


    except Exception as e:
        logger.error(f"Error during merge operation: {e}", exc_info=True)
        return

    # --- Save Merged Data ---
    logger.info(f"Saving merged data to: {OUTPUT_CSV_PATH}")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        # Save with specific float formatting
        df_merged.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
        logger.info("Merged data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving merged data: {e}", exc_info=True)
        return

# --- Run the Merge ---
if __name__ == "__main__":
    logger.info("--- Starting Prediction Merge Script ---")
    merge_data()
    logger.info("--- Prediction Merge Script Finished ---")