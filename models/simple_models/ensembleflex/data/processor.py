# /home/s_felix/ensembleflex/ensembleflex/data/processor.py

"""
Data processing for the EnsembleFlex ML pipeline.

This module provides functions for preprocessing aggregated protein data,
including cleaning, feature engineering, window-based feature generation,
data splitting, and preparation for machine learning models.
Temperature-specific logic has been removed for the unified model approach.
"""

import os
import logging
# from functools import lru_cache # Not used directly here
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # For split_data

# Updated imports for ensembleflex structure
from ensembleflex.data.loader import load_file, validate_data_columns # Removed load_temperature_data, lru_cache
from ensembleflex.utils.helpers import progress_bar # Assuming progress_bar helper exists

logger = logging.getLogger(__name__)

# --- Amino Acid and Secondary Structure Mappings ---
# (Keep these mappings as they are based on fundamental properties)
AA_MAP = {
    'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6,
    'GLU': 7, 'GLY': 8, 'HIS': 12, 'HSD': 9, 'HSE': 10, 'HSP': 11, # Group Histidines
    'ILE': 13, 'LEU': 14, 'LYS': 15, 'MET': 16, 'PHE': 17, 'PRO': 23,
    'SER': 18, 'THR': 19, 'TRP': 20, 'TYR': 21, 'VAL': 22,
    'UNK': 0, None: 0 # Handle unknown and potential None values
}

SS_MAP = {
    'H': 0, 'G': 0, 'I': 0, # Helices
    'E': 1, 'B': 1,         # Strands
    'T': 2, 'S': 2, 'C': 2, # Coil/Turn/Bend
    '-': 2, None: 2         # Unknown/None default to Coil
}

# --- Data Cleaning and Basic Feature Engineering ---

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans input DataFrame by handling missing values and ensuring types.
    (No fundamental changes needed for aggregated data structure).

    Args:
        df: Input DataFrame with raw protein data.

    Returns:
        Cleaned DataFrame.
    """
    logger.debug(f"Starting data cleaning on DataFrame with shape {df.shape}.")
    cleaned_df = df.copy()

    # Define fill values for different column types/meanings
    fill_values = {
        'dssp': 'C',                         # Default SS to Coil
        'relative_accessibility': 0.5,       # Moderate accessibility
        'core_exterior': 'core',             # Assume core if unknown
        'phi': 0.0,                          # Neutral angle
        'psi': 0.0,                          # Neutral angle
        'esm_rmsf': lambda d: d.mean(),      # Fill with mean of existing values
        'voxel_rmsf': lambda d: d.mean(),    # Fill with mean of existing values
        'temperature': lambda d: d.median()  # Fill missing temp with median? Or 0? Or raise error? Median seems safer.
        # Add other columns needing specific fill logic here
    }

    for col, fill_val in fill_values.items():
        if col in cleaned_df.columns:
            nan_count = cleaned_df[col].isna().sum()
            if nan_count == 0:
                continue # Skip if no NaNs

            if callable(fill_val): # Handle functions like mean(), median()
                try:
                    calculated_val = fill_val(cleaned_df[col].dropna())
                    if pd.isna(calculated_val): # Handle case where all values were NaN
                         calculated_val = 0.0 if col in ['esm_rmsf', 'voxel_rmsf', 'temperature', 'relative_accessibility', 'phi', 'psi'] else 'UNK' # Fallback
                         logger.warning(f"All values in '{col}' were NaN. Filling with fallback: {calculated_val}")

                    if col in ['esm_rmsf', 'voxel_rmsf', 'temperature', 'relative_accessibility', 'phi', 'psi']: # Ensure numeric type before filling
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

                    cleaned_df[col].fillna(calculated_val, inplace=True)
                    logger.debug(f"Filled {nan_count} NaNs in '{col}' with calculated value: {calculated_val:.4f}")

                except Exception as e:
                    logger.error(f"Failed to calculate fill value for '{col}': {e}. Skipping fill.")

            else: # Handle fixed fill values
                if col in ['esm_rmsf', 'voxel_rmsf', 'temperature', 'relative_accessibility', 'phi', 'psi']: # Ensure numeric type
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                cleaned_df[col].fillna(fill_val, inplace=True)
                logger.debug(f"Filled {nan_count} NaNs in '{col}' with default value: {fill_val}")

    # Ensure temperature is float after potential filling
    if 'temperature' in cleaned_df.columns:
        cleaned_df['temperature'] = cleaned_df['temperature'].astype(float)

    logger.debug(f"Data cleaning finished. DataFrame shape: {cleaned_df.shape}")
    return cleaned_df

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features like normalized position and encoded categories.
       (No changes needed here, operates row-wise or on existing columns).
    """
    logger.debug("Adding derived features...")
    derived_df = df.copy()

    # 1. Normalized Residue Position
    if 'normalized_resid' not in derived_df.columns and 'resid' in derived_df.columns and 'domain_id' in derived_df.columns:
        # Calculate max - min per group, handle single-residue domains (max-min=0)
        grp = derived_df.groupby('domain_id')['resid']
        min_res = grp.transform('min')
        max_minus_min = (grp.transform('max') - min_res).replace(0, 1) # Avoid division by zero
        derived_df['normalized_resid'] = (derived_df['resid'] - min_res) / max_minus_min
        logger.debug("Added 'normalized_resid'.")

    # 2. Encoded Residue Name
    if 'resname' in derived_df.columns and 'resname_encoded' not in derived_df.columns:
        derived_df['resname_encoded'] = derived_df['resname'].map(AA_MAP).fillna(0).astype(int)
        logger.debug("Added 'resname_encoded'.")

    # 3. Encoded Core/Exterior
    if 'core_exterior' in derived_df.columns and 'core_exterior_encoded' not in derived_df.columns:
        # Ensure 'core' maps to 0, others (e.g., 'surface', 'exterior') to 1
        derived_df['core_exterior_encoded'] = derived_df['core_exterior'].apply(
            lambda x: 0 if isinstance(x, str) and x.lower() == 'core' else 1
        ).astype(int)
        logger.debug("Added 'core_exterior_encoded'.")

    # 4. Encoded Secondary Structure
    if 'dssp' in derived_df.columns and 'secondary_structure_encoded' not in derived_df.columns:
        # Apply mapping, fill potential NaNs introduced by unknown DSSP codes
        derived_df['secondary_structure_encoded'] = derived_df['dssp'].map(SS_MAP).fillna(2).astype(int)
        logger.debug("Added 'secondary_structure_encoded'.")

    # 5. Normalized Phi/Psi Angles
    if 'phi' in derived_df.columns and 'phi_norm' not in derived_df.columns:
        # Normalize angle to [-1, 1] using sine for cyclical nature
        derived_df['phi_norm'] = np.sin(np.radians(derived_df['phi'].fillna(0)))
        logger.debug("Added 'phi_norm'.")
    if 'psi' in derived_df.columns and 'psi_norm' not in derived_df.columns:
        derived_df['psi_norm'] = np.sin(np.radians(derived_df['psi'].fillna(0)))
        logger.debug("Added 'psi_norm'.")

    return derived_df


# --- Domain Filtering ---

def filter_domains(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Filters DataFrame based on domain inclusion/exclusion and size rules.
    (No changes needed, operates on 'domain_id' and calculates size dynamically).

    Args:
        df: Input DataFrame with protein data.
        config: Configuration dictionary with domain filtering rules.

    Returns:
        Filtered DataFrame.
    """
    domain_config = config.get("dataset", {}).get("domains", {})
    if not domain_config: # No filtering rules specified
        logger.info("No domain filtering rules specified in config.")
        return df.copy()

    filtered_df = df.copy()
    initial_rows = len(filtered_df)
    logger.debug(f"Starting domain filtering from {initial_rows} rows.")

    # Apply inclusion/exclusion lists
    include_domains = set(domain_config.get("include", []))
    exclude_domains = set(domain_config.get("exclude", []))

    if include_domains:
        filtered_df = filtered_df[filtered_df['domain_id'].isin(include_domains)]
        logger.debug(f"Applied inclusion filter: {len(filtered_df)} rows remain.")
    if exclude_domains:
        filtered_df = filtered_df[~filtered_df['domain_id'].isin(exclude_domains)]
        logger.debug(f"Applied exclusion filter: {len(filtered_df)} rows remain.")

    # Apply size filters (calculate sizes only if needed)
    # NOTE: 'protein_size' in aggregated data refers to the size of the original protein,
    # not the number of rows per domain in the aggregated file. We need to calculate
    # the actual number of unique residues per domain if size filtering is based on that.
    min_size = domain_config.get("min_protein_size", 0)
    max_size = domain_config.get("max_protein_size") # None means no upper limit

    if min_size > 0 or max_size is not None:
        if filtered_df.empty:
             logger.warning("DataFrame is empty before size filtering.")
        else:
            # Calculate unique residues per domain for size filtering
            domain_residue_counts = filtered_df.groupby('domain_id')['resid'].transform('nunique')
            if min_size > 0:
                filtered_df = filtered_df[domain_residue_counts >= min_size]
                logger.debug(f"Applied min residue count filter ({min_size}): {len(filtered_df)} rows remain.")
            if max_size is not None:
                filtered_df = filtered_df[domain_residue_counts <= max_size]
                logger.debug(f"Applied max residue count filter ({max_size}): {len(filtered_df)} rows remain.")

    final_rows = len(filtered_df)
    if final_rows < initial_rows:
         logger.info(f"Domain filtering removed {initial_rows - final_rows} rows. {final_rows} rows remaining.")
    else:
         logger.debug("No rows removed by domain filtering.")

    return filtered_df

# --- Window Feature Generation (Optimized) ---

def create_window_features_optimized(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Creates window-based features using vectorized operations and pd.concat.
    (No changes needed here, groups by domain_id and sorts by resid,
     ignoring other columns like temperature).

    This version is optimized to reduce DataFrame fragmentation warnings and
    improve performance compared to iterative column addition.

    Args:
        df: Input DataFrame with protein data, indexed uniquely.
        window_size: Number of residues on each side (k). Window is size 2k+1.
        feature_cols: List of *existing* feature column names in df to use.
                      **IMPORTANT:** Should *not* include 'temperature'.

    Returns:
        DataFrame with added window features.
    """
    if 'domain_id' not in df.columns or 'resid' not in df.columns:
        raise ValueError("DataFrame must contain 'domain_id' and 'resid' columns for windowing.")
    if window_size <= 0:
        logger.warning("Window size must be positive. Skipping window feature creation.")
        return df.copy()

    original_cols = set(df.columns)
    # Only use features that actually exist in the input df
    valid_feature_cols = [col for col in feature_cols if col in original_cols]

    if not valid_feature_cols:
        logger.warning("No valid feature columns found for windowing.")
        return df.copy()

    all_domain_windows_dfs = [] # Collect DataFrames of window features per domain
    logger.info(f"Creating window features (k={window_size}) for {len(valid_feature_cols)} features using optimized method...")

    # Ensure the DataFrame index is unique if it's not already the default RangeIndex
    if not isinstance(df.index, pd.RangeIndex) or not df.index.is_unique:
        logger.debug("Resetting index for reliable window feature alignment.")
        df_indexed = df.reset_index(drop=True) # Work with a re-indexed copy
    else:
        df_indexed = df # Use original if index is okay

    # Process each domain
    # Grouping by domain_id handles the structure correctly even with multiple temperatures per residue
    grouped_domains = df_indexed.groupby('domain_id', observed=False, sort=False) # sort=False might be faster
    for domain_id, domain_indices in progress_bar(grouped_domains.groups.items(), desc="Processing Domains"):
        # Sorting by resid is crucial for correct offsets within the sequence dimension
        domain_df = df_indexed.loc[domain_indices].sort_values('resid')
        domain_window_series_dict = {} # Collect {col_name: pd.Series} for this domain

        for feature in valid_feature_cols:
            feature_values = domain_df[feature].values # Efficient NumPy array access
            dtype = feature_values.dtype
            # Use a sensible default fill value based on dtype (0 for int, NaN for float)
            default_fill = 0 if np.issubdtype(dtype, np.integer) else np.nan

            for offset in range(-window_size, window_size + 1):
                if offset == 0: continue # Skip self

                col_name = f"{feature}_offset_{offset}"
                # Pre-allocate array with default fill value
                offset_values = np.full_like(feature_values, fill_value=default_fill, dtype=dtype)

                # Apply shifts using slicing (vectorized)
                if offset < 0: # Look backwards (use values from start)
                    offset_values[-offset:] = feature_values[:offset]
                else: # Look forwards (use values towards end)
                    offset_values[:-offset] = feature_values[offset:]

                # Store as Series with the correct index from domain_df
                domain_window_series_dict[col_name] = pd.Series(offset_values, index=domain_df.index, name=col_name)

        # Combine all window series for this domain into one DataFrame
        if domain_window_series_dict:
            all_domain_windows_dfs.append(pd.concat(domain_window_series_dict.values(), axis=1))

    # Concatenate window features from all domains (aligns by original index)
    if not all_domain_windows_dfs:
         logger.warning("No window features were generated.")
         return df.copy() # Return a copy of the original df

    logger.info("Concatenating window features for all domains...")
    window_features_combined = pd.concat(all_domain_windows_dfs)

    # Merge window features back to the *original* df using the index
    result_df = pd.concat([df, window_features_combined], axis=1)

    # --- NaN Filling (After all columns are combined) ---
    all_new_col_names = list(window_features_combined.columns)
    logger.info(f"Added {len(all_new_col_names)} window feature columns. Filling NaNs...")

    # Pre-calculate fill values for base features to avoid repeated computation
    base_feature_fill_values = {}
    for col in all_new_col_names:
        base_feature = col.split('_offset_')[0]
        if base_feature not in base_feature_fill_values and base_feature in result_df.columns:
            # Use median for floats, mode for integers/objects
            if np.issubdtype(result_df[base_feature].dtype, np.floating):
                fill_val = result_df[base_feature].median()
                fill_type = "median"
                if pd.isna(fill_val): fill_val = 0.0; fill_type = "fallback (0.0)"
            elif np.issubdtype(result_df[base_feature].dtype, np.integer):
                try: fill_val = result_df[base_feature].mode()[0]; fill_type = "mode"
                except IndexError: fill_val = 0; fill_type = "fallback (0)"
            else: # Object or other types
                 try: fill_val = result_df[base_feature].mode()[0]; fill_type = "mode"
                 except IndexError: fill_val = ''; fill_type = "fallback ('')" # Use empty string for objects
            logger.debug(f"Using {fill_type} ({fill_val}) for base feature {base_feature}")
            base_feature_fill_values[base_feature] = fill_val

    # Fill NaNs using pre-calculated values
    for col in all_new_col_names:
        if col not in result_df.columns: continue # Should not happen

        base_feature = col.split('_offset_')[0]
        fill_val = base_feature_fill_values.get(base_feature, 0) # Default to 0 if base feature somehow missed

        nan_count = result_df[col].isna().sum()
        if nan_count > 0:
             result_df[col].fillna(fill_val, inplace=True)
             # logger.debug(f"Filled {nan_count} NaNs in '{col}' with {fill_val}.")


    logger.info("Window feature creation and NaN filling complete (Optimized).")
    return result_df

# --- Main Processing Function ---

def process_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Processes features: cleaning, deriving features, adding window features.
    Operates on the aggregated DataFrame.

    Args:
        df: Input **aggregated** DataFrame with raw protein data and temperature column.
        config: Configuration dictionary.

    Returns:
        DataFrame with processed features ready for filtering/splitting.
    """
    try:
        logger.info(f"Starting feature processing for DataFrame with shape {df.shape}.")
        # 1. Basic Cleaning (includes handling temperature column NaNs)
        cleaned_df = clean_data(df)

        # 2. Add Derived Features (Encodings, Normalizations)
        derived_df = _add_derived_features(cleaned_df)

        # 3. Add Protein Size (if necessary and possible)
        if "protein_size" not in derived_df.columns and 'domain_id' in derived_df.columns and 'resid' in derived_df.columns:
            logger.debug("Calculating 'protein_size'...")
            # Calculate size based on unique residues per domain
            derived_df["protein_size"] = derived_df.groupby("domain_id")["resid"].transform("nunique")
        elif "protein_size" not in derived_df.columns:
             logger.warning("Cannot calculate 'protein_size': missing 'domain_id' or 'resid'.")

        # 4. Ensure Target Column is Numeric and filled
        target_col = config["dataset"]["target"] # Should be "rmsf"
        if target_col in derived_df.columns:
            initial_nan_count = derived_df[target_col].isna().sum()
            derived_df[target_col] = pd.to_numeric(derived_df[target_col], errors='coerce')
            # Fill NaNs in target with 0.0 - Check if this is appropriate for the task!
            # Alternatively, drop rows: derived_df.dropna(subset=[target_col], inplace=True)
            fill_target_val = 0.0
            final_nan_count = derived_df[target_col].isna().sum()
            if final_nan_count > 0:
                derived_df[target_col].fillna(fill_target_val, inplace=True)
                logger.warning(f"Filled {final_nan_count} NaN values in target column '{target_col}' with {fill_target_val}.")
            elif initial_nan_count > 0 and final_nan_count == 0: # Only log if NaNs were actually present and filled
                 logger.debug(f"NaNs in target column '{target_col}' handled (filled/coerced).")

        else:
            logger.error(f"Target column '{target_col}' not found in data after cleaning/deriving features!")
            raise ValueError(f"Target column '{target_col}' missing.")

        # 5. Identify Active Features for Windowing (based on config *after* derivation)
        feature_config = config["dataset"]["features"]
        use_features_config = feature_config.get("use_features", {})
        active_features_for_windowing = []
        for feature, enabled in use_features_config.items():
            # Check if feature exists *now*, is enabled, AND is NOT the temperature column
            if enabled and feature in derived_df.columns and feature != 'temperature':
                active_features_for_windowing.append(feature)
            elif enabled and feature == 'temperature':
                logger.debug(f"Feature '{feature}' is enabled but explicitly excluded from windowing.")
            elif enabled and feature not in derived_df.columns:
                logger.warning(f"Feature '{feature}' is enabled but not found after deriving features.")
        logger.info(f"Identified {len(active_features_for_windowing)} base features for windowing.")

        # 6. Add Window Features (using optimized method)
        window_config = feature_config.get("window", {})
        processed_df = derived_df # Start with the derived features df
        if window_config.get("enabled", False):
            window_size = window_config.get("size", 3)
            if window_size > 0 and active_features_for_windowing:
                 processed_df = create_window_features_optimized(
                     processed_df, window_size, active_features_for_windowing
                 )
            else:
                 logger.warning("Windowing enabled but size <= 0 or no active base features found. Skipping.")
                 processed_df = processed_df.copy() # Ensure consistency by returning a copy
        else:
             logger.info("Window feature creation skipped (disabled in config).")
             processed_df = processed_df.copy() # Return a copy even if windowing disabled

        logger.info(f"Feature processing complete. Final shape: {processed_df.shape}")
        return processed_df

    except Exception as e:
        logger.exception(f"Feature processing failed catastrophically: {e}")
        # Return a copy of the original DataFrame on error to prevent side effects
        return df.copy()


# --- Data Preparation for Models ---

def prepare_data_for_model(
    df: pd.DataFrame,
    config: Dict[str, Any],
    include_target: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Prepares final data matrices (X, y) and feature names for model input.

    Selects features based on config, including generated window features and
    the 'temperature' feature if enabled.

    Args:
        df: Processed DataFrame containing all potential features.
        config: Configuration dictionary.
        include_target: Whether to include the target variable (y) in the output.

    Returns:
        Tuple containing:
        - X (np.ndarray): Feature matrix.
        - y (Optional[np.ndarray]): Target vector (or None if include_target=False).
        - feature_names (List[str]): List of names for columns in X.

    Raises:
        ValueError: If the target column is missing when include_target is True,
                    or if no valid features are selected.
    """
    logger.debug("Preparing data for model...")
    feature_config = config["dataset"]["features"]
    use_features_config = feature_config.get("use_features", {})
    window_config = feature_config.get("window", {})
    window_enabled = window_config.get("enabled", False)
    window_size = window_config.get("size", 0) if window_enabled else 0

    # 1. Determine base features enabled in config (excluding temperature initially)
    base_features_enabled = {
        feature for feature, enabled in use_features_config.items()
        if enabled and feature in df.columns and feature != 'temperature'
    }
    logger.debug(f"Base features (excluding temp) enabled & present: {sorted(list(base_features_enabled))}")

    # 2. Determine window features to include (if enabled) based on base features
    window_feature_names = []
    if window_enabled and window_size > 0:
        for base_feature in base_features_enabled:
            for offset in range(-window_size, window_size + 1):
                if offset == 0: continue
                col_name = f"{base_feature}_offset_{offset}"
                if col_name in df.columns:
                    window_feature_names.append(col_name)
        logger.debug(f"Found {len(window_feature_names)} potential window features.")

    # 3. Combine base and window features
    final_feature_names = sorted(list(base_features_enabled)) + sorted(window_feature_names)

    # 4. Explicitly check and add 'temperature' feature if enabled
    if use_features_config.get('temperature', False):
        if 'temperature' in df.columns:
            final_feature_names.append('temperature')
            logger.debug("Including 'temperature' as an input feature.")
        else:
            logger.warning("'temperature' feature enabled in config but column not found in data!")
    else:
        logger.debug("'temperature' feature is not enabled in config.")


    # 5. Sanity check and prepare X matrix
    missing_final_features = [f for f in final_feature_names if f not in df.columns]
    if missing_final_features:
        logger.warning(f"Some selected features are missing from the final DataFrame: {missing_final_features}. They will be excluded.")
        final_feature_names = [f for f in final_feature_names if f in df.columns]

    if not final_feature_names:
        raise ValueError("No valid features selected or available for the model.")

    logger.info(f"Preparing feature matrix X with {len(final_feature_names)} columns: {final_feature_names}")
    # Select columns and convert to NumPy, ensuring numeric types where possible
    X_df = df[final_feature_names].copy() # Use copy to avoid SettingWithCopyWarning
    # Attempt conversion, coercing errors - check dtypes afterwards
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
             try:
                 X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0) # Fill coerced NaNs with 0
             except ValueError: logger.warning(f"Could not convert object column '{col}' to numeric. It might cause issues.")
        elif pd.api.types.is_bool_dtype(X_df[col].dtype):
             X_df[col] = X_df[col].astype(int) # Convert bools to int
        # Ensure temperature is float
        elif col == 'temperature':
             X_df[col] = X_df[col].astype(float)

    # Check final dtypes
    non_numeric_cols = X_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        logger.warning(f"Non-numeric columns remain in feature matrix: {list(non_numeric_cols)}. Models might fail.")

    X = X_df.values

    # 6. Prepare target vector y (if requested)
    y = None
    if include_target:
        target_col = config["dataset"]["target"] # Assumes target name is "rmsf"
        if target_col in df.columns:
            # Ensure target is numeric and handle potential NaNs (e.g., fill with 0 or mean)
            y_series = pd.to_numeric(df[target_col], errors='coerce')
            nan_count = y_series.isna().sum()
            if nan_count > 0:
                # Consider if 0.0 is appropriate, or maybe mean/median of the target column
                fill_val = y_series.median() if not y_series.dropna().empty else 0.0
                y_series.fillna(fill_val, inplace=True)
                logger.warning(f"Filled {nan_count} NaNs in target column '{target_col}' with median value ({fill_val:.4f}) before creating y vector.")
            y = y_series.values
        else:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame for y preparation.")

    logger.debug(f"Data preparation complete. X shape: {X.shape}, y shape: {y.shape if y is not None else 'None'}")
    return X, y, final_feature_names


# --- Main Loading and Processing Function ---


def load_and_process_data(
    data_input: Union[str, pd.DataFrame, None] = None, # Accept path OR DataFrame
    config: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Loads data (from path or DataFrame), validates required columns,
    processes features, and filters domains based on the provided config.

    This is the main entry point for getting processed data ready for splitting or prediction.

    Args:
        data_input: Path to the aggregated data file (str) OR a pandas DataFrame.
                    If None, it attempts to load from the path specified in the config.
        config: Configuration dictionary (required).

    Returns:
        Processed and filtered DataFrame.

    Raises:
        FileNotFoundError: If a specified data file path does not exist.
        ValueError: If config is missing, required columns are absent in loaded data,
                    or data_input type is invalid.
        TypeError: If data_input is not a str, DataFrame, or None.
    """
    if not config:
         raise ValueError("Configuration dictionary is required for load_and_process_data.")

    logger.info("Loading and processing data...")
    df = None # Initialize df

    # --- Load initial data ---
    if isinstance(data_input, str): # Input is a file path
        data_path = data_input
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Specified data file not found: {data_path}")
        logger.info(f"Loading data from specified path: {data_path}")
        try:
            # Use load_file directly (caching handled within load_file)
            df = load_file(data_path)
        except Exception as e:
            raise ValueError(f"Error loading data from {data_path}: {e}") from e
    elif isinstance(data_input, pd.DataFrame): # Input is already a DataFrame
         logger.info("Processing provided DataFrame.")
         df = data_input.copy() # Work on a copy
    elif data_input is None: # Load from config path
        data_dir = config["paths"]["data_dir"]
        file_pattern = config["dataset"]["file_pattern"] # e.g., "aggregated_rmsf_data.csv"
        config_file_path = os.path.join(data_dir, file_pattern)

        if not os.path.exists(config_file_path):
             raise FileNotFoundError(f"Aggregated data file specified in config not found: {config_file_path}")

        logger.info(f"Loading data from config path: {config_file_path}")
        try:
            df = load_file(config_file_path)
        except Exception as e:
            raise ValueError(f"Error loading data from {config_file_path}: {e}") from e
    else: # Invalid input type
        raise TypeError(f"Invalid input type for data_input: {type(data_input)}. Expected str, pd.DataFrame, or None.")

    # --- Validation & Processing ---
    if df is None or df.empty:
         raise ValueError("Loaded or provided DataFrame is empty.")

    # Validate Required Columns (Static list from config)
    required_cols = config["dataset"]["features"].get("required", [])
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Loaded data is missing required columns: {missing_cols}")
        logger.debug("All required columns found in loaded data.")
    else:
         logger.warning("No required columns specified in config.")


    # Process Features (Cleaning, Deriving, Windowing)
    processed_df = process_features(df, config) # Pass the loaded/copied df

    # Filter Domains
    filtered_df = filter_domains(processed_df, config)

    logger.info(f"Data loading and processing complete. Final shape: {filtered_df.shape}")
    return filtered_df

# --- Data Splitting ---

def split_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, validation, and test sets.

    Supports stratified splitting by domain ID to prevent data leakage.
    Operates on the aggregated data, ensuring all temperature points for a
    domain stay together in the same split.

    Args:
        df: Processed **aggregated** DataFrame with features and target.
        config: Configuration dictionary containing split parameters.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    split_config = config["dataset"]["split"]
    test_size = split_config.get("test_size", 0.2)
    val_size = split_config.get("validation_size", 0.15) # Proportion of *original* data for validation
    random_state = config["system"].get("random_state", 42) # Use global random state
    stratify_by_domain = split_config.get("stratify_by_domain", True)

    logger.info(f"Splitting aggregated data: Test={test_size*100:.1f}%, Val={val_size*100:.1f}%, Stratify by domain={stratify_by_domain}")

    if stratify_by_domain:
        if 'domain_id' not in df.columns:
            raise ValueError("Cannot stratify by domain: 'domain_id' column missing.")

        unique_domains = df['domain_id'].unique()
        n_domains = len(unique_domains)
        logger.debug(f"Found {n_domains} unique domains for stratified splitting.")

        if n_domains < 3: # Need at least one domain in each split
             logger.warning("Too few domains for reliable stratified splitting. Falling back to random split.")
             stratify_by_domain = False # Override stratification

    if stratify_by_domain:
        # Calculate actual validation proportion relative to the training set size
        if (1 - test_size) <= 0: raise ValueError("test_size must be less than 1")
        val_ratio_of_train = val_size / (1 - test_size)
        if not (0 < val_ratio_of_train < 1):
             raise ValueError(f"Invalid split sizes: val_size ({val_size}) must be smaller than remaining train size ({1-test_size})")

        # Split unique domains into train+val / test
        domains_train_val, domains_test = train_test_split(
            unique_domains,
            test_size=test_size,
            random_state=random_state
        )
        # Split train+val domains into train / val
        domains_train, domains_val = train_test_split(
            domains_train_val,
            test_size=val_ratio_of_train,
            random_state=random_state # Use same seed for reproducibility at this stage too
        )

        logger.debug(f"Domain split: Train={len(domains_train)}, Val={len(domains_val)}, Test={len(domains_test)}")

        # Create DataFrames based on domain splits. All rows (all temps) for a domain go together.
        train_df = df[df['domain_id'].isin(domains_train)].copy()
        val_df = df[df['domain_id'].isin(domains_val)].copy()
        test_df = df[df['domain_id'].isin(domains_test)].copy()

    else: # Regular sample-based splitting (if stratification disabled or not possible)
        # NOTE: This will randomly split rows, potentially separating different temperature
        # points for the same residue into different sets. Stratification is highly recommended.
        logger.warning("Performing random sample split (not stratified by domain). This may split temperature points for the same residue.")
         # Calculate actual validation proportion relative to the training set size
        if (1 - test_size) <= 0: raise ValueError("test_size must be less than 1")
        val_ratio_of_train = val_size / (1 - test_size)
        if not (0 < val_ratio_of_train < 1):
             raise ValueError(f"Invalid split sizes: val_size ({val_size}) must be smaller than remaining train size ({1-test_size})")

        # First split into (train + val) and test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
        # Then split (train + val) into train and val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_of_train,
            random_state=random_state # Use same seed
        )

    logger.info(f"Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} rows.")

    # Final check for empty splits
    if train_df.empty or val_df.empty or test_df.empty:
        logger.error("One or more data splits are empty! Check split sizes and data.")

    return train_df, val_df, test_df