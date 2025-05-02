import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

# --- Configuration ---
# Assumes script is run from /home/s_felix/FINAL_PROJECT/packages/DeepFlex/predictions
BASE_PREDICTIONS_DIR = Path.cwd()
TARGET_FILE = Path("/home/s_felix/FINAL_PROJECT/Data_Analysis/data/01_final_analysis_dataset.csv")
OUTPUT_FILE = Path("/home/s_felix/FINAL_PROJECT/Data_Analysis/data/02_final_analysis_with_preds.csv")

# Specific temperatures and their directory names
TEMPERATURE_DIRS = {
    '320K': 'Holdout_single_temp_model_320K',
    '348K': 'Holdout_single_temp_model_348K',
    '379K': 'Holdout_single_temp_model_379K',
    '413K': 'Holdout_single_temp_model_413K',
    '450K': 'Holdout_single_temp_model_450K'
}

# Exact relative path within each temperature directory
SOURCE_RELATIVE_PATH = "prediction_from_npy_train_temperatures_mc5/predictions_prediction_from_npy_train_temperatures_mc5.csv"

# Names for the new columns in the output file
NEW_PRED_COL = "single_temp_DeepFlex_pred"
NEW_UNCERT_COL = "single_temp_DeepFlex_pred_uncertainty"

# --- Main Logic ---
all_source_data = []

print(f"Starting script.")
print(f"Base predictions directory: {BASE_PREDICTIONS_DIR}")
print(f"Target file: {TARGET_FILE}")
print(f"Output file will be: {OUTPUT_FILE}")

# --- Load and Prepare Target Data ---
if not TARGET_FILE.is_file():
    print(f"\nERROR: Target file not found at {TARGET_FILE}")
    exit()

print("\nLoading target data...")
try:
    target_df = pd.read_csv(TARGET_FILE)
    # Ensure correct types for merge keys in target
    target_df['domain_id'] = target_df['domain_id'].astype(str)
    # Convert temperature to float for reliable comparison later
    target_df['temperature'] = pd.to_numeric(target_df['temperature'], errors='coerce')
    target_df['resid'] = target_df['resid'].astype(int)

    # Handle potential errors during temperature conversion
    if target_df['temperature'].isnull().any():
        warnings.warn(f"WARNING: Some 'temperature' values in {TARGET_FILE} could not be converted to numbers and were set to NaN.")
        target_df.dropna(subset=['temperature'], inplace=True) # Remove rows with invalid temperatures

    print(f"Target data loaded successfully: {target_df.shape[0]} rows.")
    print("Target data columns and types (first 5):")
    print(target_df.info(max_cols=5))

except Exception as e:
    print(f"\nERROR: Failed to load or process target file {TARGET_FILE}: {e}")
    exit()


# --- Process Each Source Temperature File ---
print("\nProcessing source prediction files...")
for temp_key, temp_dir_name in TEMPERATURE_DIRS.items():
    print(f"--- Processing Temperature: {temp_key} ---")
    source_file_path = BASE_PREDICTIONS_DIR / temp_dir_name / SOURCE_RELATIVE_PATH

    if not source_file_path.is_file():
        warnings.warn(f"  WARNING: Source file NOT FOUND at expected path, skipping: {source_file_path}")
        continue

    print(f"  Loading source file: {source_file_path}")
    try:
        source_df_temp = pd.read_csv(source_file_path)

        # --- Prepare source data for merging ---
        # 1. Check required columns exist
        required_cols = ['instance_key', 'resid', 'rmsf_pred', 'uncertainty']
        if not all(col in source_df_temp.columns for col in required_cols):
             missing = [col for col in required_cols if col not in source_df_temp.columns]
             warnings.warn(f"  WARNING: Missing required columns {missing} in {source_file_path}. Skipping this file.")
             continue

        # 2. Parse instance_key to get domain_id and temperature
        try:
            # Handle potential errors if split doesn't yield 2 parts
            split_key = source_df_temp['instance_key'].str.split('@', n=1, expand=True)
            if split_key.shape[1] < 2:
                raise ValueError("Split on '@' did not yield at least two parts.")

            source_df_temp['parsed_domain_id'] = split_key[0].astype(str)
             # Convert temperature part to numeric, coercing errors to NaN
            source_df_temp['parsed_temperature'] = pd.to_numeric(split_key[1], errors='coerce')

            # Check for and remove rows where temperature parsing failed
            invalid_temp_rows = source_df_temp['parsed_temperature'].isnull()
            if invalid_temp_rows.any():
                warnings.warn(f"  WARNING: Found {invalid_temp_rows.sum()} rows with unparseable temperature in 'instance_key' in {source_file_path}. These rows will be dropped.")
                source_df_temp = source_df_temp.dropna(subset=['parsed_temperature'])

            if source_df_temp.empty:
                 warnings.warn(f"  WARNING: No valid rows remaining after parsing 'instance_key' in {source_file_path}. Skipping.")
                 continue

            # Convert temperature to float after handling NaNs
            source_df_temp['parsed_temperature'] = source_df_temp['parsed_temperature'].astype(float)

        except Exception as parse_e:
            warnings.warn(f"  WARNING: Error parsing 'instance_key' in {source_file_path}: {parse_e}. Skipping this file.")
            continue

        # 3. Ensure resid is integer type
        source_df_temp['resid'] = source_df_temp['resid'].astype(int)

        # 4. Select and rename columns for merging
        source_df_to_merge = source_df_temp[[
            'parsed_domain_id',
            'parsed_temperature',
            'resid',
            'rmsf_pred',  # Original name
            'uncertainty' # Original name
        ]].rename(columns={
            'parsed_domain_id': 'domain_id',      # Rename to match target
            'parsed_temperature': 'temperature',  # Rename to match target
            'rmsf_pred': NEW_PRED_COL,            # Rename to final desired name
            'uncertainty': NEW_UNCERT_COL       # Rename to final desired name
        })

        print(f"  Successfully processed source file: {source_df_to_merge.shape[0]} rows prepared for merge.")
        all_source_data.append(source_df_to_merge)

    except Exception as e:
        warnings.warn(f"  WARNING: Failed to load or process file {source_file_path}: {e}")
        continue

# --- Combine all loaded source data ---
if not all_source_data:
    print("\nERROR: No source prediction data could be loaded or processed. Cannot perform merge. Exiting.")
    exit()

print("\nCombining all processed source dataframes...")
combined_source_df = pd.concat(all_source_data, ignore_index=True)
print(f"Combined source data shape for merge: {combined_source_df.shape}")
if combined_source_df.empty:
     print("\nERROR: Combined source dataframe is empty. Cannot perform merge. Exiting.")
     exit()


# --- Merge Target with Combined Source Data ---
print("\nMerging target data with combined source predictions...")
print(f"Merging on columns: ['domain_id', 'temperature', 'resid']")

# Crucial step: Handle potential float precision issues for temperature matching.
# Round both temperature columns to a reasonable number of decimal places (e.g., 1, matching @XXX.X) before merging.
target_df['temp_for_merge'] = target_df['temperature'].round(1)
combined_source_df['temp_for_merge'] = combined_source_df['temperature'].round(1)

# Perform the left merge
merged_df = pd.merge(
    target_df,
    combined_source_df[['domain_id', 'temp_for_merge', 'resid', NEW_PRED_COL, NEW_UNCERT_COL]],
    # Use the rounded temperature for matching
    on=['domain_id', 'temp_for_merge', 'resid'],
    how='left' # Keep all rows from the target dataframe
)

# Clean up: remove the temporary merge column
merged_df.drop(columns=['temp_for_merge'], inplace=True)
# We don't strictly need to drop from combined_source_df as it's not used further

print(f"Merge complete. Final DataFrame shape: {merged_df.shape}")

# --- Validation Checks ---
# 1. Check if new columns exist
if NEW_PRED_COL not in merged_df.columns or NEW_UNCERT_COL not in merged_df.columns:
     print(f"\nERROR: Merge likely failed! New columns ('{NEW_PRED_COL}', '{NEW_UNCERT_COL}') not found in the result.")
     # Optional: print head of target and source for debugging
     # print("\nTarget head (for merge check):")
     # print(target_df[['domain_id', 'temperature', 'resid', 'temp_for_merge']].head())
     # print("\nSource head (for merge check):")
     # print(combined_source_df[['domain_id', 'temperature', 'resid', 'temp_for_merge']].head())
     exit()
else:
     print("New columns successfully added.")

# 2. Check how many rows received non-null values from the merge
matched_preds_count = merged_df[NEW_PRED_COL].notna().sum()
matched_uncert_count = merged_df[NEW_UNCERT_COL].notna().sum()
print(f"Number of rows with matched '{NEW_PRED_COL}': {matched_preds_count} / {len(merged_df)}")
print(f"Number of rows with matched '{NEW_UNCERT_COL}': {matched_uncert_count} / {len(merged_df)}")

if matched_preds_count == 0:
    warnings.warn("\nWARNING: Zero matches found between the target data and the loaded prediction data. "
                  "Please double-check:\n"
                  "  1. The source file paths and contents.\n"
                  "  2. The matching keys ('domain_id', 'temperature', 'resid') in both target and source files.\n"
                  "  3. Potential data type mismatches or formatting issues (especially domain IDs and residue numbers).")
elif matched_preds_count < len(merged_df):
     print(f"Note: {len(merged_df) - matched_preds_count} rows in the target file did not have a matching prediction.")


# --- Save Output ---
print(f"\nSaving merged data to: {OUTPUT_FILE}...")
try:
    # Ensure the output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Save with sufficient float precision
    merged_df.to_csv(OUTPUT_FILE, index=False, float_format='%.8f')
    print(f"Script finished successfully. Output saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"\nERROR: Failed to save output file {OUTPUT_FILE}: {e}")