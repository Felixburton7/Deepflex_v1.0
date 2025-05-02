# """
# Module for building the final ML-ready feature set.
# Combines RMSF, structural properties, and calculates derived features.
# Performs critical alignment validation.
# """
# import logging
# import pandas as pd
# import numpy as np
# from typing import Dict, Optional, Any, Tuple

# class FeatureBuilderError(Exception):
#     """Custom exception for FeatureBuilder errors."""
#     pass

# class FeatureBuilder:
#     """
#     Builds ML feature DataFrames by combining processed data sources.

#     Args:
#         config (Dict[str, Any]): The 'feature_building' section of the main config.
#         replica_avg_rmsf (Dict[str, pd.DataFrame]): Dict mapping temp_str to DataFrame
#                                                     containing domain_id, resid, resname, rmsf_{temp}.
#         overall_avg_rmsf (Optional[pd.DataFrame]): DataFrame with domain_id, resid,
#                                                     resname, rmsf_average.
#         structure_properties (Dict[str, pd.DataFrame]): Dict mapping domain_id to DataFrame
#                                                         from PropertiesCalculator.
#     """
#     def __init__(self,
#                  config: Dict[str, Any],
#                  replica_avg_rmsf: Dict[str, pd.DataFrame],
#                  overall_avg_rmsf: Optional[pd.DataFrame],
#                  structure_properties: Dict[str, pd.DataFrame]):

#         self.config = config
#         self.replica_avg_rmsf = replica_avg_rmsf
#         self.overall_avg_rmsf = overall_avg_rmsf
#         self.structure_properties = structure_properties
#         self.add_rmsf_log = self.config.get("add_rmsf_log", True)

#         if not replica_avg_rmsf and overall_avg_rmsf is None:
#              raise FeatureBuilderError("Cannot build features: No RMSF data provided.")
#         if not structure_properties:
#              raise FeatureBuilderError("Cannot build features: No structure properties data provided.")

#         # Pre-process structure properties for faster lookup (optional)
#         # Ensure 'resid' is numeric in properties dict values
#         for df in self.structure_properties.values():
#             if 'resid' in df.columns:
#                 df['resid'] = pd.to_numeric(df['resid'], errors='coerce')


#     def build_features(self, temperature: Optional[str] = None) -> Optional[pd.DataFrame]:
#         """
#         Builds the feature DataFrame for a specific temperature or the overall average.

#         Args:
#             temperature (Optional[str]): Temperature string (e.g., "320") or None to build
#                                          the overall average feature set.

#         Returns:
#             Optional[pd.DataFrame]: The final feature DataFrame, or None if building fails.
#         """
#         if temperature:
#             if temperature not in self.replica_avg_rmsf:
#                 logging.error(f"Cannot build features for T={temperature}: Missing average RMSF data.")
#                 return None
#             base_rmsf_df = self.replica_avg_rmsf[temperature].copy()
#             rmsf_col = f"rmsf_{temperature}"
#             log_rmsf_col = f"rmsf_log_{temperature}"
#             logging.info(f"Building features for temperature: {temperature}")
#         else:
#             if self.overall_avg_rmsf is None or self.overall_avg_rmsf.empty:
#                 logging.error("Cannot build average features: Missing overall average RMSF data.")
#                 return None
#             base_rmsf_df = self.overall_avg_rmsf.copy()
#             # Ensure the average column name exists and is correct
#             if 'rmsf_average' not in base_rmsf_df.columns:
#                  # Attempt to find/calculate if missing (should ideally be present from rmsf processing)
#                  avg_cols = [c for c in base_rmsf_df if c.startswith("rmsf_avg_at_")]
#                  if avg_cols: base_rmsf_df['rmsf_average'] = base_rmsf_df[avg_cols].mean(axis=1)
#                  else:
#                       logging.error("Cannot build average features: 'rmsf_average' column missing and cannot be calculated.")
#                       return None
#             rmsf_col = "rmsf_average"
#             log_rmsf_col = "rmsf_log" # Generic name for log of average
#             logging.info("Building features for overall average.")


#         # --- Combine with Structure Properties ---
#         all_features_list = []
#         processed_domains = base_rmsf_df['domain_id'].unique()
#         validation_failed_domains = set()

#         for domain_id in processed_domains:
#             domain_rmsf_df = base_rmsf_df[base_rmsf_df['domain_id'] == domain_id].copy()
#             if domain_id not in self.structure_properties:
#                 logging.warning(f"Skipping domain {domain_id} for T={temperature or 'average'}: Missing structure properties.")
#                 validation_failed_domains.add(domain_id)
#                 continue

#             domain_props_df = self.structure_properties[domain_id]
#             if domain_props_df is None or domain_props_df.empty:
#                  logging.warning(f"Skipping domain {domain_id} for T={temperature or 'average'}: Empty structure properties.")
#                  validation_failed_domains.add(domain_id)
#                  continue

#             # Ensure 'resid' is numeric in both for merging
#             try:
#                 domain_rmsf_df['resid'] = pd.to_numeric(domain_rmsf_df['resid'])
#                 # Properties 'resid' conversion done in __init__
#             except Exception as e:
#                  logging.error(f"Resid type conversion failed for domain {domain_id}: {e}. Skipping.")
#                  validation_failed_domains.add(domain_id)
#                  continue

#             # --- CRITICAL VALIDATION: RESIDUE ALIGNMENT ---
#             rmsf_resids = set(domain_rmsf_df['resid'])
#             props_resids = set(domain_props_df['resid'])

#             if len(domain_rmsf_df) != len(domain_props_df) or rmsf_resids != props_resids:
#                 logging.error(f"Residue mismatch for domain {domain_id}! "
#                               f"RMSF count: {len(domain_rmsf_df)}, Properties count: {len(domain_props_df)}. "
#                               f"RMSF resids unique: {len(rmsf_resids)}, Props resids unique: {len(props_resids)}. "
#                               f" Skipping this domain for T={temperature or 'average'}.")
#                 # Log details of mismatch if needed (e.g., symmetric difference)
#                 # diff = rmsf_resids.symmetric_difference(props_resids)
#                 # logging.debug(f"Residue difference: {diff}")
#                 validation_failed_domains.add(domain_id)
#                 continue

#             # --- Merge Data ---
#             try:
#                 # Merge on domain_id (implicitly done by processing per domain) and resid
#                 # Select necessary columns from properties DF to avoid duplication
#                 props_cols_to_merge = ['resid', 'chain', 'dssp', 'relative_accessibility', 'core_exterior', 'phi', 'psi']
#                 merged_df = pd.merge(domain_rmsf_df, domain_props_df[props_cols_to_merge], on='resid', how='inner')

#                 # Double check length after merge (should be identical if validation passed)
#                 if len(merged_df) != len(domain_rmsf_df):
#                     logging.error(f"Length mismatch after merging for domain {domain_id}! "
#                                   f"RMSF: {len(domain_rmsf_df)}, Merged: {len(merged_df)}. Skipping.")
#                     validation_failed_domains.add(domain_id)
#                     continue

#                 all_features_list.append(merged_df)

#             except Exception as e:
#                  logging.error(f"Merging failed for domain {domain_id}: {e}. Skipping.", exc_info=True)
#                  validation_failed_domains.add(domain_id)
#                  continue

#         if not all_features_list:
#             logging.error(f"No domain data could be merged successfully for T={temperature or 'average'}. Cannot build feature set.")
#             return None

#         # Combine features for all successfully processed domains
#         final_df = pd.concat(all_features_list, ignore_index=True)
#         logging.info(f"Successfully merged data for {len(final_df['domain_id'].unique())} domains for T={temperature or 'average'}.")
#         if validation_failed_domains:
#              logging.warning(f"Failed validation/merge for {len(validation_failed_domains)} domains for T={temperature or 'average'}.")


#         # --- Calculate Derived Features ---
#         logging.debug("Calculating derived features...")
#         # 1. Protein Size (per domain)
#         final_df['protein_size'] = final_df.groupby('domain_id')['resid'].transform('nunique')

#         # 2. Normalized Residue ID (per domain)
#         def normalize_resid(x):
#             min_val = x.min()
#             max_val = x.max()
#             range_val = max_val - min_val
#             return (x - min_val) / range_val if range_val > 0 else 0.0 # Avoid division by zero for single-residue domains
#         final_df['normalized_resid'] = final_df.groupby('domain_id')['resid'].transform(normalize_resid)

#         # 3. Log RMSF
#         if self.add_rmsf_log:
#              # Add small epsilon to avoid log(0) issues if RMSF can be exactly zero
#              epsilon = 1e-9
#              final_df[log_rmsf_col] = np.log(final_df[rmsf_col] + epsilon)

#         # --- Encode Categorical Features ---
#         logging.debug("Encoding categorical features...")
#         # 4. Core/Exterior Encoding
#         core_ext_mapping = {"core": 0, "exterior": 1, "unknown": 2} # 'unknown' if SASA failed maybe? Defaulted to core earlier.
#         final_df['core_exterior_encoded'] = final_df['core_exterior'].map(core_ext_mapping).fillna(2).astype(int)

#         # 5. Secondary Structure (3-state: Helix, Sheet, Coil) Encoding
#         def encode_ss3(ss):
#             ss = str(ss).upper() # Ensure string and uppercase
#             if ss in ['H', 'G', 'I']: return 0 # Helix
#             elif ss in ['E', 'B']: return 1 # Sheet
#             else: return 2 # Coil/Loop/Other
#         final_df['secondary_structure_encoded'] = final_df['dssp'].apply(encode_ss3)

#         # 6. Resname Encoding (simple integer mapping)
#         unique_resnames = sorted([r for r in final_df['resname'].unique() if r and isinstance(r, str) and len(r) == 3])
#         resname_map = {name: i for i, name in enumerate(unique_resnames)}
#         final_df['resname_encoded'] = final_df['resname'].map(resname_map).fillna(-1).astype(int) # Use -1 for unknown/non-standard

#         # 7. Normalized Phi/Psi Angles (map -180 to 180 -> -1 to 1)
#         final_df['phi_norm'] = final_df['phi'] / 180.0
#         final_df['psi_norm'] = final_df['psi'] / 180.0
#         # Clip values just in case they exceed +/- 180
#         final_df[['phi_norm', 'psi_norm']] = final_df[['phi_norm', 'psi_norm']].clip(-1.0, 1.0)

#         # --- Final Touches ---
#         # Define final column order (adjust as needed)
#         core_cols = ['domain_id', 'resid', 'resname', rmsf_col]
#         if self.add_rmsf_log: core_cols.append(log_rmsf_col)

#         info_cols = ['protein_size', 'normalized_resid', 'chain']
#         structure_cols = ['core_exterior', 'relative_accessibility', 'dssp', 'phi', 'psi']
#         encoded_cols = ['core_exterior_encoded', 'secondary_structure_encoded', 'resname_encoded', 'phi_norm', 'psi_norm']

#         # Get all columns present and order them
#         present_cols = final_df.columns.tolist()
#         final_order = ([col for col in core_cols if col in present_cols] +
#                        [col for col in info_cols if col in present_cols] +
#                        [col for col in structure_cols if col in present_cols] +
#                        [col for col in encoded_cols if col in present_cols])
#         # Add any remaining columns not explicitly listed (shouldn't be many)
#         remaining_cols = [col for col in present_cols if col not in final_order]
#         final_order.extend(remaining_cols)

#         final_df = final_df[final_order]

#         # Final check for NaNs in key numerical columns (RMSF, coords should ideally not be NaN)
#         nan_check_cols = [rmsf_col, 'protein_size', 'normalized_resid', 'relative_accessibility', 'phi_norm', 'psi_norm']
#         if self.add_rmsf_log: nan_check_cols.append(log_rmsf_col)
#         for col in nan_check_cols:
#              if col in final_df.columns and final_df[col].isnull().any():
#                   logging.warning(f"NaN values detected in final feature column '{col}' for T={temperature or 'average'}. Consider filling/investigation.")
#                   # Fill with 0 for now? Or mean? Filling with 0 might be safer.
#                   final_df[col].fillna(0.0, inplace=True)


#         logging.info(f"Finished building features for T={temperature or 'average'}. Final shape: {final_df.shape}")
#         return final_df



"""
Module for building the final ML-ready feature set.
Combines RMSF, structural properties (loaded from disk), and calculates derived features.
Performs critical alignment validation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import os # Import os
from tqdm import tqdm

from ..exceptions import FeatureBuilderError # Use defined exception

class FeatureBuilder:
    """
    Builds ML feature DataFrames by combining processed data sources, loading
    properties from intermediate files.

    Args:
        config (Dict[str, Any]): The 'feature_building' section of the main config.
        replica_avg_rmsf (Dict[str, pd.DataFrame]): Dict mapping temp_str to DataFrame
                                                    containing domain_id, resid, resname,
                                                    rmsf_avg_{temp}. (Result from aggregation)
        overall_avg_rmsf (Optional[pd.DataFrame]): DataFrame with domain_id, resid,
                                                    resname, rmsf_average. (Result from aggregation)
        properties_file_paths (Dict[str, Optional[str]]): Dict mapping domain_id to the
                                                           path of its intermediate properties
                                                           file (.parquet). Path is None if failed.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 replica_avg_rmsf: Dict[str, pd.DataFrame],
                 overall_avg_rmsf: Optional[pd.DataFrame],
                 properties_file_paths: Dict[str, Optional[str]]): # Changed argument

        self.config = config
        # These RMSF averages should fit in memory as they are aggregated results
        self.replica_avg_rmsf = replica_avg_rmsf
        self.overall_avg_rmsf = overall_avg_rmsf
        # Store paths to properties, not the dataframes
        self.properties_file_paths = properties_file_paths
        self.add_rmsf_log = self.config.get("add_rmsf_log", True)

        if not replica_avg_rmsf and overall_avg_rmsf is None:
             # This check remains valid for the aggregated RMSF data
             raise FeatureBuilderError("Cannot build features: No aggregated RMSF data provided.")
        if not properties_file_paths:
             raise FeatureBuilderError("Cannot build features: No properties file paths provided.")


    def build_features(self, temperature: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Builds the feature DataFrame for a specific temperature or the overall average,
        loading properties data from disk per domain.

        Args:
            temperature (Optional[str]): Temperature string (e.g., "320") or None to build
                                         the overall average feature set.

        Returns:
            Optional[pd.DataFrame]: The final feature DataFrame, or None if building fails.
        """
        base_rmsf_df = None
        rmsf_col = None
        log_rmsf_col = None

        if temperature:
            if temperature not in self.replica_avg_rmsf:
                logging.error(f"Cannot build features for T={temperature}: Missing aggregated average RMSF data.")
                return None
            base_rmsf_df = self.replica_avg_rmsf[temperature].copy()
            # Aggregation step now produces 'rmsf_avg_{temp}'
            rmsf_col = f"rmsf_avg_{temperature}"
            if rmsf_col not in base_rmsf_df.columns:
                 logging.error(f"Expected column '{rmsf_col}' not found in replica average data for T={temperature}. Cannot build features.")
                 return None
            log_rmsf_col = f"rmsf_log_{temperature}"
            logging.info(f"Building features for temperature: {temperature}")
        else:
            if self.overall_avg_rmsf is None or self.overall_avg_rmsf.empty:
                logging.error("Cannot build average features: Missing overall average RMSF data.")
                return None
            base_rmsf_df = self.overall_avg_rmsf.copy()
            # Aggregation step produces 'rmsf_average'
            rmsf_col = "rmsf_average"
            if rmsf_col not in base_rmsf_df.columns:
                 logging.error(f"Expected column '{rmsf_col}' not found in overall average RMSF data. Cannot build features.")
                 return None
            log_rmsf_col = "rmsf_log" # Generic name for log of average
            logging.info("Building features for overall average.")

        if base_rmsf_df is None: # Should not happen if logic above is correct, but belts and braces
            logging.error("Base RMSF DataFrame could not be determined.")
            return None

        # --- Combine with Structure Properties (Loading from disk) ---
        all_features_list = []
        processed_domains = base_rmsf_df['domain_id'].unique()
        validation_failed_domains = set()
        load_failed_domains = set()

        log_cfg = self.config.get('logging', {}) if isinstance(self.config, dict) else {}
        show_progress = log_cfg.get('show_progress_bars', True)
        desc = f"Merging Features (T={temperature or 'Avg'})"

        for domain_id in tqdm(processed_domains, desc=desc, disable=not show_progress):
            domain_rmsf_df = base_rmsf_df[base_rmsf_df['domain_id'] == domain_id].copy()

            # Get path and load properties for THIS domain
            properties_path = self.properties_file_paths.get(domain_id)
            if not properties_path or not os.path.exists(properties_path):
                logging.warning(f"Skipping domain {domain_id}: Missing or invalid intermediate properties file path '{properties_path}'.")
                load_failed_domains.add(domain_id)
                continue

            try:
                domain_props_df = pd.read_parquet(properties_path)
                if domain_props_df is None or domain_props_df.empty:
                    logging.warning(f"Skipping domain {domain_id}: Loaded empty intermediate properties data.")
                    load_failed_domains.add(domain_id)
                    continue
                # Ensure 'resid' is numeric after loading
                domain_props_df['resid'] = pd.to_numeric(domain_props_df['resid'], errors='coerce')
                domain_props_df.dropna(subset=['resid'], inplace=True) # Drop if conversion failed

            except Exception as e:
                 logging.error(f"Failed to load properties from {properties_path} for domain {domain_id}: {e}. Skipping.")
                 load_failed_domains.add(domain_id)
                 continue

            # Ensure 'resid' is numeric in RMSF df for merging
            try:
                domain_rmsf_df['resid'] = pd.to_numeric(domain_rmsf_df['resid'], errors='coerce')
                domain_rmsf_df.dropna(subset=['resid'], inplace=True)
            except Exception as e:
                 logging.error(f"Resid type conversion failed for RMSF data of domain {domain_id}: {e}. Skipping.")
                 validation_failed_domains.add(domain_id)
                 # Clean up loaded properties df before skipping
                 del domain_props_df
                 continue

            # --- CRITICAL VALIDATION: RESIDUE ALIGNMENT ---
            rmsf_resids = set(domain_rmsf_df['resid'])
            props_resids = set(domain_props_df['resid'])

            if len(domain_rmsf_df) != len(domain_props_df) or rmsf_resids != props_resids:
                logging.error(f"Residue mismatch for domain {domain_id}! "
                              f"RMSF count: {len(domain_rmsf_df)}, Properties count: {len(domain_props_df)}. "
                              f"RMSF resids unique: {len(rmsf_resids)}, Props resids unique: {len(props_resids)}. "
                              f" Skipping this domain.")
                              #f" RMSF path: {rmsf_col}, Props path: {properties_path}") # Help debugging
                validation_failed_domains.add(domain_id)
                # Clean up loaded properties df before skipping
                del domain_props_df
                continue

            # --- Merge Data ---
            try:
                # Merge on domain_id (implicitly done by processing per domain) and resid
                props_cols_to_merge = ['resid', 'chain', 'dssp', 'relative_accessibility', 'core_exterior', 'phi', 'psi']
                # Ensure only columns that exist in the loaded df are selected
                props_cols_to_merge = [col for col in props_cols_to_merge if col in domain_props_df.columns]
                if 'resid' not in props_cols_to_merge: props_cols_to_merge.append('resid') # Ensure resid is present

                merged_df = pd.merge(domain_rmsf_df, domain_props_df[props_cols_to_merge], on='resid', how='inner')

                # Double check length after merge (should be identical if validation passed)
                if len(merged_df) != len(domain_rmsf_df):
                    logging.error(f"Length mismatch after merging for domain {domain_id}! "
                                  f"RMSF: {len(domain_rmsf_df)}, Merged: {len(merged_df)}. Skipping.")
                    validation_failed_domains.add(domain_id)
                    # Clean up loaded properties df before skipping
                    del domain_props_df
                    continue

                all_features_list.append(merged_df)

            except Exception as e:
                 logging.error(f"Merging failed for domain {domain_id}: {e}. Skipping.", exc_info=True)
                 validation_failed_domains.add(domain_id)
                 # Clean up loaded properties df before skipping
                 del domain_props_df
                 continue

            # Data for this domain (`domain_props_df`) goes out of scope here
            # Explicitly delete if garbage collection is slow (usually not needed)
            del domain_props_df


        # --- Post-Loop Processing ---
        if not all_features_list:
            logging.error(f"No domain data could be loaded and merged successfully for T={temperature or 'average'}. Cannot build feature set.")
            return None

        # Combine features for all successfully processed domains
        final_df = pd.concat(all_features_list, ignore_index=True)
        successful_merge_domains = final_df['domain_id'].nunique()
        logging.info(f"Successfully merged data for {successful_merge_domains} domains for T={temperature or 'average'}.")
        if validation_failed_domains:
             logging.warning(f"Failed validation/merge for {len(validation_failed_domains)} domains.")
        if load_failed_domains:
             logging.warning(f"Failed to load properties for {len(load_failed_domains)} domains.")


        # --- Calculate Derived Features (No changes needed here) ---
        logging.debug("Calculating derived features...")
        # 1. Protein Size
        final_df['protein_size'] = final_df.groupby('domain_id')['resid'].transform('nunique')

        # 2. Normalized Residue ID
        def normalize_resid(x):
            min_val = x.min()
            max_val = x.max()
            range_val = max_val - min_val
            return (x - min_val) / range_val if range_val > 0 else 0.0
        final_df['normalized_resid'] = final_df.groupby('domain_id')['resid'].transform(normalize_resid)

        # 3. Log RMSF
        if self.add_rmsf_log:
             epsilon = 1e-9
             if rmsf_col in final_df.columns:
                  final_df[log_rmsf_col] = np.log(final_df[rmsf_col] + epsilon)
             else:
                  logging.error(f"RMSF column '{rmsf_col}' not found for log calculation.")


        # --- Encode Categorical Features (No changes needed here) ---
        logging.debug("Encoding categorical features...")
        # 4. Core/Exterior Encoding
        core_ext_mapping = {"core": 0, "exterior": 1, "unknown": 2}
        final_df['core_exterior_encoded'] = final_df['core_exterior'].map(core_ext_mapping).fillna(2).astype(int)

        # 5. Secondary Structure (3-state) Encoding
        def encode_ss3(ss):
            ss = str(ss).upper()
            if ss in ['H', 'G', 'I']: return 0
            elif ss in ['E', 'B']: return 1
            else: return 2
        final_df['secondary_structure_encoded'] = final_df['dssp'].apply(encode_ss3)

        # 6. Resname Encoding
        unique_resnames = sorted([r for r in final_df['resname'].unique() if r and isinstance(r, str) and len(r) == 3])
        resname_map = {name: i for i, name in enumerate(unique_resnames)}
        final_df['resname_encoded'] = final_df['resname'].map(resname_map).fillna(-1).astype(int)

        # 7. Normalized Phi/Psi Angles
        final_df['phi_norm'] = final_df['phi'] / 180.0
        final_df['psi_norm'] = final_df['psi'] / 180.0
        final_df[['phi_norm', 'psi_norm']] = final_df[['phi_norm', 'psi_norm']].clip(-1.0, 1.0)

        # --- Final Touches (No changes needed here) ---
        # Define final column order
        core_cols = ['domain_id', 'resid', 'resname', rmsf_col]
        if self.add_rmsf_log and log_rmsf_col in final_df.columns: core_cols.append(log_rmsf_col)

        info_cols = ['protein_size', 'normalized_resid', 'chain']
        structure_cols = ['core_exterior', 'relative_accessibility', 'dssp', 'phi', 'psi']
        encoded_cols = ['core_exterior_encoded', 'secondary_structure_encoded', 'resname_encoded', 'phi_norm', 'psi_norm']

        present_cols = final_df.columns.tolist()
        final_order = ([col for col in core_cols if col in present_cols] +
                       [col for col in info_cols if col in present_cols] +
                       [col for col in structure_cols if col in present_cols] +
                       [col for col in encoded_cols if col in present_cols])
        remaining_cols = [col for col in present_cols if col not in final_order]
        final_order.extend(remaining_cols)
        final_df = final_df[final_order]

        # Final check for NaNs
        nan_check_cols = [rmsf_col, 'protein_size', 'normalized_resid', 'relative_accessibility', 'phi_norm', 'psi_norm']
        if self.add_rmsf_log and log_rmsf_col in final_df.columns: nan_check_cols.append(log_rmsf_col)
        for col in nan_check_cols:
             if col in final_df.columns and final_df[col].isnull().any():
                  nan_count = final_df[col].isnull().sum()
                  logging.warning(f"NaN values ({nan_count}) detected in final feature column '{col}'. Filling with 0.0.")
                  final_df[col].fillna(0.0, inplace=True)


        logging.info(f"Finished building features for T={temperature or 'average'}. Final shape: {final_df.shape}")
        return final_df