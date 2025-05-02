import pandas as pd
import numpy as np
from collections import defaultdict
import os
import random
from typing import Dict, List, Tuple, Set, Optional, Any
import logging
import json
import re # Import regex for potentially complex ID parsing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard 1-letter amino acid codes
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Include common Histidine variants if they weren't fixed by fix_data_.py
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H'
}

VALID_AA_1LETTER = set(AA_MAP.values())


# Map 3-letter code to 3-letter code (for consistency checks)
AA_3LETTER_MAP = {
    'ALA': 'ALA', 'CYS': 'CYS', 'ASP': 'ASP', 'GLU': 'GLU', 'PHE': 'PHE',
    'GLY': 'GLY', 'HIS': 'HIS', 'ILE': 'ILE', 'LYS': 'LYS', 'LEU': 'LEU',
    'MET': 'MET', 'ASN': 'ASN', 'PRO': 'PRO', 'GLN': 'GLN', 'ARG': 'ARG',
    'SER': 'SER', 'THR': 'THR', 'VAL': 'VAL', 'TRP': 'TRP', 'TYR': 'TYR',
    # Histidine variants
    'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS'
}

# List of structural features to extract from the enriched dataset
STRUCTURAL_FEATURES = [
    'normalized_resid', 'core_exterior_encoded', 'secondary_structure_encoded',
    'relative_accessibility', 'phi_norm', 'psi_norm', 'protein_size',
    'voxel_rmsf', 'bfactor_norm'
]

# Define the separator for domain_id and temperature
INSTANCE_KEY_SEPARATOR = "@"

def create_instance_key(domain_id: str, temperature: float) -> str:
    """Creates a unique key combining domain ID and temperature."""
    # Format temperature to avoid floating point inconsistencies in keys
    return f"{str(domain_id)}{INSTANCE_KEY_SEPARATOR}{temperature:.1f}"

def get_domain_id_from_instance_key(instance_key: str) -> Optional[str]:
    """Extracts the original domain_id from the combined instance key."""
    if INSTANCE_KEY_SEPARATOR in instance_key:
        return instance_key.split(INSTANCE_KEY_SEPARATOR, 1)[0]
    else:
        logger.warning(f"Instance key '{instance_key}' did not contain the expected separator '{INSTANCE_KEY_SEPARATOR}'. Returning the full key as domain_id.")
        return instance_key


def load_data(csv_path: str, config: Dict = None) -> Optional[pd.DataFrame]:
    """Load data from the enriched CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found: {csv_path}")
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    try:
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Required columns
        required_cols = ['domain_id', 'resid', 'resname', 'temperature', 'rmsf']

        # Determine which optional features are requested and available
        required_features = []
        if config and config.get('data', {}).get('features', {}):
            feature_config = config['data']['features']
            enabled_features = []
            if feature_config.get('use_position_info', True): enabled_features.append('normalized_resid')
            if feature_config.get('use_structure_info', True): enabled_features.extend(['core_exterior_encoded', 'secondary_structure_encoded'])
            if feature_config.get('use_accessibility', True): enabled_features.append('relative_accessibility')
            if feature_config.get('use_backbone_angles', True): enabled_features.extend(['phi_norm', 'psi_norm'])
            if feature_config.get('use_protein_size', True): enabled_features.append('protein_size')
            if feature_config.get('use_voxel_rmsf', True): enabled_features.append('voxel_rmsf')
            if feature_config.get('use_bfactor', True): enabled_features.append('bfactor_norm')

            for feature in enabled_features:
                if feature in df.columns:
                    required_features.append(feature)
                else:
                    logger.warning(f"Feature '{feature}' specified in config but not found in dataset")
        else:
            logger.info("No feature configuration provided or features section missing. Checking for all standard features.")
            for feature in STRUCTURAL_FEATURES:
                 if feature in df.columns:
                      required_features.append(feature)
                      logger.info(f"Found structural feature: {feature}")

        required_cols.extend(required_features)

        # Check for missing required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            core_missing = [c for c in ['domain_id', 'resid', 'resname', 'temperature', 'rmsf'] if c not in df.columns]
            if core_missing:
                logger.error(f"CSV missing ESSENTIAL columns: {core_missing}. Cannot proceed.")
                return None
            else:
                logger.warning(f"CSV missing requested FEATURE columns: {[c for c in missing_cols if c in required_features]}. Proceeding without them.")
                required_cols = [c for c in required_cols if c in df.columns]


        # Rename temperature column for consistency
        if 'temperature' in df.columns:
            df.rename(columns={'temperature': 'temperature_feature'}, inplace=True)
        elif 'temperature_feature' not in df.columns:
            logger.error("Neither 'temperature' nor 'temperature_feature' columns found in the dataset")
            return None

        # Rename RMSF column for consistency
        if 'rmsf' in df.columns:
            df.rename(columns={'rmsf': 'target_rmsf'}, inplace=True)
        elif 'target_rmsf' not in df.columns:
            logger.error("Neither 'rmsf' nor 'target_rmsf' columns found in the dataset")
            return None

        # Check for NaN in essential columns (now including target_rmsf)
        nan_check_cols = ['domain_id', 'resid', 'resname', 'temperature_feature', 'target_rmsf']
        nan_counts = df[nan_check_cols].isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values in essential columns:\n{nan_counts[nan_counts > 0]}")
            logger.warning("Attempting to drop rows with NaNs in these essential columns...")
            df.dropna(subset=nan_check_cols, inplace=True)
            logger.info(f"{len(df)} rows remaining after dropping NaNs.")
            if len(df) == 0:
                logger.error("No valid rows remaining after dropping NaNs. Cannot proceed.")
                return None

        # Convert numerical columns to numeric, coercing errors
        numeric_cols_present = [col for col in ['temperature_feature', 'target_rmsf', 'resid'] if col in df.columns]
        numeric_cols_present.extend([col for col in required_features if col in df.columns])

        for col in numeric_cols_present:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype != original_type and not pd.api.types.is_numeric_dtype(original_type):
                     logger.debug(f"Column '{col}' coerced from {original_type} to {df[col].dtype}.")

        # Drop rows where essential numeric conversions failed (resulted in NaN)
        essential_numeric_cols = ['temperature_feature', 'target_rmsf', 'resid']
        nan_after_coerce = df[essential_numeric_cols].isnull().sum()
        if nan_after_coerce.sum() > 0:
             logger.warning(f"Found NaNs after numeric conversion in essential columns:\n{nan_after_coerce[nan_after_coerce > 0]}")
             logger.warning("Dropping rows with NaNs in these essential numeric columns...")
             df.dropna(subset=essential_numeric_cols, inplace=True)
             logger.info(f"{len(df)} rows remaining after dropping conversion NaNs.")

        # Convert resid to integer type after ensuring no NaNs
        df['resid'] = df['resid'].astype(int)

        # Map resnames to standard 3-letter codes for consistency
        if 'resname' in df.columns:
            df['resname'] = df['resname'].apply(
                lambda x: AA_3LETTER_MAP.get(str(x).upper().strip(), str(x).upper().strip()) if pd.notna(x) else x
            )
            unknown_res = df[~df['resname'].isin(AA_MAP.keys())]['resname'].unique()
            if len(unknown_res) > 0:
                 logger.warning(f"Found potentially unknown residue names after mapping: {unknown_res}")


        if len(df) == 0:
            logger.error("No valid data rows remaining after initial processing and cleaning. Cannot proceed.")
            return None

        logger.info(f"Initial data loading and cleaning complete. {len(df)} rows remaining.")
        return df
    except Exception as e:
        logger.error(f"Error loading or performing initial validation on CSV file {csv_path}: {e}", exc_info=True)
        raise


def group_by_domain_and_temp(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group data by the unique combination of domain_id and temperature_feature.
    Generates instance keys like 'domain_id@temperature'.
    """
    instance_groups = {}
    required_cols = ['domain_id', 'resid', 'temperature_feature']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"DataFrame missing one or more required columns for grouping: {required_cols}.")
        return instance_groups

    grouped = df.groupby(['domain_id', 'temperature_feature'], observed=True)
    logger.info(f"Grouping by ('domain_id', 'temperature_feature')... Found {len(grouped)} potential groups.")

    processed_groups = 0
    for (domain_id, temp), group_df in grouped:
        domain_id_str = str(domain_id)
        try:
            temp_float = float(temp)
        except (ValueError, TypeError):
            logger.warning(f"Skipping group due to invalid temperature value: domain='{domain_id_str}', temp='{temp}'.")
            continue

        instance_key = create_instance_key(domain_id_str, temp_float)
        instance_groups[instance_key] = group_df.sort_values('resid')
        processed_groups += 1

    logger.info(f"Grouped data into {len(instance_groups)} unique (domain_id, temperature) instances.")
    if len(grouped) != processed_groups:
         logger.warning(f"Processed {processed_groups} groups, but initially found {len(grouped)}. Some might have been skipped due to invalid temperatures.")

    return instance_groups

def extract_sequence_rmsf_temp_features(instance_groups: Dict[str, pd.DataFrame], config: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    Extract sequence, RMSF, temp, and features for each unique (domain, temp) instance.
    Imputes NaN feature values using instance median. Skips instances with residue gaps.
    """
    processed_data = {}
    processed_count = 0
    skipped_residues = defaultdict(int)
    skipped_instances_missing_cols = set()
    skipped_instances_length_mismatch = set()
    skipped_instances_no_sequence = set()
    skipped_instances_non_sequential = set()
    nan_imputation_counts = defaultdict(int)

    features_to_extract = []
    if config and config.get('data', {}).get('features', {}):
        feature_config = config['data']['features']
        if feature_config.get('use_position_info', True): features_to_extract.append('normalized_resid')
        if feature_config.get('use_structure_info', True): features_to_extract.extend(['core_exterior_encoded', 'secondary_structure_encoded'])
        if feature_config.get('use_accessibility', True): features_to_extract.append('relative_accessibility')
        if feature_config.get('use_backbone_angles', True): features_to_extract.extend(['phi_norm', 'psi_norm'])
        if feature_config.get('use_protein_size', True): features_to_extract.append('protein_size')
        if feature_config.get('use_voxel_rmsf', True): features_to_extract.append('voxel_rmsf')
        if feature_config.get('use_bfactor', True): features_to_extract.append('bfactor_norm')
    else:
        logger.info("No feature config section found, attempting to extract all standard features.")
        features_to_extract = STRUCTURAL_FEATURES

    logger.info(f"Attempting to extract structural features: {features_to_extract}")

    # Process each instance (domain_id@temp)
    for instance_key, instance_df in instance_groups.items():
        required_core_cols = ['resname', 'target_rmsf', 'temperature_feature', 'resid']
        available_features_in_df = []

        for feature in features_to_extract:
            if feature in instance_df.columns:
                available_features_in_df.append(feature)

        required_cols_for_instance = required_core_cols + available_features_in_df

        if not all(col in instance_df.columns for col in required_cols_for_instance):
            missing = [c for c in required_cols_for_instance if c not in instance_df.columns]
            logger.warning(f"Skipping instance {instance_key} due to missing columns: {missing}. Available: {instance_df.columns.tolist()}")
            skipped_instances_missing_cols.add(instance_key)
            continue

        sequence = ''
        rmsf_values = []
        temperature = float(instance_df['temperature_feature'].iloc[0])
        protein_size = None
        if 'protein_size' in available_features_in_df:
            protein_size = float(instance_df['protein_size'].iloc[0])

        feature_arrays = {feature: [] for feature in available_features_in_df if feature != 'protein_size'}
        residue_numbers = []
        valid_instance = True
        last_resid = -1

        # --- Pre-calculate medians for imputation ---
        feature_medians = {}
        for feature in feature_arrays.keys():
            median_val = instance_df[feature].median()
            if pd.isna(median_val):
                 logger.warning(f"Instance {instance_key}: Could not calculate median for feature '{feature}' (all values might be NaN?). Imputation will use 0.")
                 feature_medians[feature] = 0.0
            else:
                 feature_medians[feature] = median_val

        # Iterate through sorted residues for this instance
        for _, row in instance_df.iterrows():
            current_resid = row['resid']

            if last_resid != -1 and current_resid != last_resid + 1:
                logger.warning(f"Instance {instance_key}: Non-sequential residue number detected (expected {last_resid+1}, got {current_resid}). Skipping instance.")
                valid_instance = False
                break
            last_resid = current_resid

            # --- *** START: Corrected Residue Check Logic *** ---
            residue_name = str(row['resname']).upper().strip() # Get name processed by load_data
            one_letter_code = None

            if residue_name in AA_MAP: # Check if it's a standard 3-letter code
                one_letter_code = AA_MAP[residue_name]
            elif residue_name in VALID_AA_1LETTER: # Check if it's already a standard 1-letter code
                one_letter_code = residue_name
            # Add elif for non-standard but mappable residues like MSE if needed
            # elif residue_name == 'MSE': one_letter_code = 'M'

            if one_letter_code is not None: # Successfully identified a standard AA code
                sequence += one_letter_code
                rmsf_values.append(row['target_rmsf'])
                residue_numbers.append(current_resid)

                # Extract available structural features for this residue
                for feature in feature_arrays.keys(): # Use keys from the dict which excludes protein_size
                    feature_val = row[feature]
                    # Impute NaN using pre-calculated median
                    if pd.isna(feature_val):
                         impute_value = feature_medians[feature]
                         if nan_imputation_counts[(instance_key, feature)] == 0:
                              logger.debug(f"Instance {instance_key}, Feature '{feature}': Found NaN(s). Imputing with instance median ({impute_value:.4f}).")
                         feature_arrays[feature].append(impute_value)
                         nan_imputation_counts[(instance_key, feature)] += 1
                    else:
                         feature_arrays[feature].append(feature_val)
            else:
                # This 'else' block now correctly handles TRULY unknown/unmapped residue names
                skipped_residues[residue_name] += 1
                logger.warning(f"Instance {instance_key}, Residue {current_resid}: Encountered truly unknown residue name '{residue_name}'. Skipping residue.")
            # --- *** END: Corrected Residue Check Logic *** ---

        if not valid_instance:
            skipped_instances_non_sequential.add(instance_key)
            continue

        if sequence:
            per_residue_feature_lengths = {f: len(arr) for f, arr in feature_arrays.items()}
            all_lengths_match = (len(sequence) == len(rmsf_values) and
                                 all(len(sequence) == length for length in per_residue_feature_lengths.values()))

            if all_lengths_match:
                processed_data[instance_key] = {
                    'sequence': sequence,
                    'rmsf': np.array(rmsf_values, dtype=np.float32),
                    'temperature': float(temperature)
                }
                for feature, arr in feature_arrays.items():
                    processed_data[instance_key][feature] = np.array(arr, dtype=np.float32)
                if protein_size is not None:
                    processed_data[instance_key]['protein_size'] = protein_size
                processed_count += 1
            else:
                failed_features = {f: length for f, length in per_residue_feature_lengths.items() if len(sequence) != length}
                logger.warning(f"Length mismatch for instance {instance_key} after processing: "
                               f"Sequence={len(sequence)}, RMSF={len(rmsf_values)}, "
                               f"Features={per_residue_feature_lengths}. "
                               f"Mismatch in features: {failed_features}. Skipping.")
                skipped_instances_length_mismatch.add(instance_key)
        else:
            logger.warning(f"Instance {instance_key} resulted in an empty sequence after processing. Skipping.")
            skipped_instances_no_sequence.add(instance_key)

    imputed_instances = len(set(key[0] for key in nan_imputation_counts.keys()))
    total_imputations = sum(nan_imputation_counts.values())
    if total_imputations > 0:
         logger.info(f"Imputed {total_imputations} NaN feature values across {imputed_instances} instances using instance medians.")

    if skipped_residues:
        logger.warning(f"Encountered unknown residues (total counts across all instances): {dict(skipped_residues)}")
    if skipped_instances_missing_cols:
        logger.warning(f"Skipped {len(skipped_instances_missing_cols)} instances due to missing columns.")
    if skipped_instances_non_sequential:
        logger.warning(f"Skipped {len(skipped_instances_non_sequential)} instances due to non-sequential residue numbers (gaps).")
    if skipped_instances_length_mismatch:
        logger.warning(f"Skipped {len(skipped_instances_length_mismatch)} instances due to final length mismatch (likely due to skipped unknown residues).")
    if skipped_instances_no_sequence:
        logger.warning(f"Skipped {len(skipped_instances_no_sequence)} instances due to empty sequence after processing.")

    total_skipped = (len(skipped_instances_missing_cols) +
                     len(skipped_instances_non_sequential) +
                     len(skipped_instances_length_mismatch) +
                     len(skipped_instances_no_sequence))
    logger.info(f"Successfully extracted and validated data for {processed_count} instances.")
    logger.info(f"Total instances skipped due to errors: {total_skipped}")

    return processed_data

def extract_topology(domain_id: str) -> str:
    """
    Extract topology identifier (e.g., PDB ID) from the domain_id part.
    Handles cases like '1xyz', '1xyz_A', '1xyz.A', '1xyz-A', etc.
    """
    if not isinstance(domain_id, str) or len(domain_id) < 4:
        logger.warning(f"Cannot reliably extract topology from short/invalid domain_id: '{domain_id}'. Using fallback.")
        return f"unknown_{hash(domain_id)}"

    match = re.match(r"^(\d[a-zA-Z0-9]{3})", domain_id)
    if match:
        pdb_id = match.group(1).upper()
        return pdb_id
    else:
        base_id_match = re.match(r"^([a-zA-Z0-9]+)", domain_id)
        if base_id_match:
             base_id = base_id_match.group(1)
             topo_candidate = base_id[:4].upper()
             logger.debug(f"Domain ID '{domain_id}' doesn't start with PDB pattern. Using fallback topology candidate: '{topo_candidate}'")
             return topo_candidate
        else:
             logger.warning(f"Could not extract meaningful topology from domain_id: '{domain_id}'. Using hash.")
             return f"unknown_{hash(domain_id)}"


def split_by_topology(data: Dict[str, Dict], train_ratio=0.85, val_ratio=0.075, seed=42) -> Tuple[Dict, Dict, Dict]: # <-- Adjusted defaults
    """
    Split data by topology based on the domain_id part of the instance_key.
    Ensures all temperature instances for a domain go to the same split.
    """
    if not data:
        logger.warning("No data provided to split_by_topology. Returning empty splits.")
        return {}, {}, {}

    random.seed(seed)
    # Use updated ratios in log message
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    logger.info(f"Splitting {len(data)} instances by topology using seed {seed}. Ratios: Train={train_ratio:.3f}, Val={val_ratio:.3f}, Test={test_ratio:.3f}")

    topology_groups = defaultdict(list)
    instance_keys = list(data.keys())

    for instance_key in instance_keys:
        domain_id = get_domain_id_from_instance_key(instance_key)
        if domain_id:
            topology = extract_topology(domain_id)
            topology_groups[topology].append(instance_key)
        else:
            logger.warning(f"Could not extract domain ID from instance key '{instance_key}'. Skipping for split assignment.")

    unique_topologies_found = list(topology_groups.keys())
    logger.info(f"Found {len(unique_topologies_found)} unique topologies from {len(instance_keys)} instances.")
    if not unique_topologies_found:
         logger.error("No topologies could be extracted. Cannot perform split.")
         return {}, {}, {}

    random.shuffle(unique_topologies_found)

    n_topologies = len(unique_topologies_found)
    if n_topologies < 3:
        logger.warning(f"Very few topologies ({n_topologies}). Splits might be skewed or empty.")

    # Ensure ratios sum to <= 1.0
    if train_ratio + val_ratio > 1.0:
        logger.warning(f"Train ({train_ratio}) + Val ({val_ratio}) ratios exceed 1.0. Adjusting validation ratio.")
        val_ratio = max(0.0, 1.0 - train_ratio)
        test_ratio = 0.0
        logger.warning(f"Adjusted ratios: Train={train_ratio:.3f}, Val={val_ratio:.3f}, Test={test_ratio:.3f}")

    train_idx = int(round(n_topologies * train_ratio)) # Use round for possibly better distribution
    val_idx = train_idx + int(round(n_topologies * val_ratio))

    # Adjust indices to prevent empty splits if possible and respect boundaries
    if n_topologies >= 1:
        if train_idx == 0: train_idx = 1 # Ensure train gets at least one
    if n_topologies >= 2:
        if val_idx == train_idx and val_ratio > 0: val_idx = min(train_idx + 1, n_topologies) # Ensure val gets one if needed
    if n_topologies >= 3:
         if val_idx == n_topologies and test_ratio > 0: # Check if test should get something
              val_idx = max(train_idx, n_topologies - 1) # Give last one to test
              if val_idx == train_idx: # Only train and test possible
                   train_idx = max(0, train_idx -1) # Adjust train boundary if needed


    # Ensure indices are within bounds
    train_idx = max(0, min(train_idx, n_topologies))
    val_idx = max(train_idx, min(val_idx, n_topologies))


    train_topologies_set = set(unique_topologies_found[:train_idx])
    val_topologies_set = set(unique_topologies_found[train_idx:val_idx])
    test_topologies_set = set(unique_topologies_found[val_idx:])

    logger.info(f"Topology split indices: Train end={train_idx}, Val end={val_idx}, Total={n_topologies}")
    logger.info(f"Split topology counts: Train={len(train_topologies_set)}, Val={len(val_topologies_set)}, Test={len(test_topologies_set)}")
    logger.debug(f"Train/Val overlap: {len(train_topologies_set.intersection(val_topologies_set))}")
    logger.debug(f"Train/Test overlap: {len(train_topologies_set.intersection(test_topologies_set))}")
    logger.debug(f"Val/Test overlap: {len(val_topologies_set.intersection(test_topologies_set))}")

    train_data, val_data, test_data = {}, {}, {}
    assigned_instances = 0
    unassigned_instances = []

    for topology, instance_key_list in topology_groups.items():
        assigned_split = False
        # Check in order: Train, Val, Test
        if topology in train_topologies_set:
            target_dict = train_data
        elif topology in val_topologies_set:
            target_dict = val_data
        elif topology in test_topologies_set:
            target_dict = test_data
        else:
            target_dict = None # Should not happen if logic above is correct

        if target_dict is not None:
            for instance_key in instance_key_list:
                if instance_key in data:
                     target_dict[instance_key] = data[instance_key]
                     assigned_instances += 1
                     assigned_split = True # Mark that this topology was assigned
            if not assigned_split and instance_key_list: # Should not happen if target_dict is not None
                 logger.error(f"Internal error: Topology '{topology}' matched a split set but no instances were assigned.")
        else:
             # This case means the topology wasn't in any set, indicates boundary issue or overlap
             logger.warning(f"Topology '{topology}' with {len(instance_key_list)} instances was not assigned to any split! Instances: {instance_key_list[:5]}...")
             unassigned_instances.extend(instance_key_list)


    logger.info(f"Split instances: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    if assigned_instances != len(data):
        logger.warning(f"Mismatch in assigned instances ({assigned_instances}) vs total instances ({len(data)}). Unassigned count: {len(unassigned_instances)}")
        logger.debug(f"Unassigned keys sample: {unassigned_instances[:10]}")

    if not train_data: logger.warning("Training set is empty after split!")
    if not val_data and val_ratio > 0: logger.warning("Validation set is empty after split, although val_ratio > 0!")
    if not test_data and test_ratio > 0: logger.warning("Test set is empty after split, although test_ratio > 0!")

    return train_data, val_data, test_data

def calculate_feature_normalization_params(train_data: Dict[str, Dict], features: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate normalization parameters (min, max, mean, std) for each feature based on training data.
    Handles both per-residue (arrays) and global (scalar) features.
    """
    if not train_data or not features:
        logger.warning("No training data or features provided for normalization parameter calculation")
        return {}

    feature_values = {feature: [] for feature in features}
    feature_is_global = {feature: False for feature in features}

    logger.info("Gathering feature values from training data for normalization...")
    sample_keys = list(train_data.keys())[:min(10, len(train_data))]
    global_candidates = set(features)
    for key in sample_keys:
        domain_info = train_data[key]
        for feature in list(global_candidates):
             if feature in domain_info:
                  if isinstance(domain_info[feature], np.ndarray) and domain_info[feature].size > 1:
                       global_candidates.remove(feature)
             else:
                  if feature in global_candidates: global_candidates.remove(feature)

    for key in sample_keys:
         domain_info = train_data[key]
         for feature in list(global_candidates):
              if feature in domain_info:
                   val = domain_info[feature]
                   is_scalar = np.isscalar(val)
                   is_size_one_array = isinstance(val, np.ndarray) and val.size == 1
                   if not (is_scalar or is_size_one_array):
                        global_candidates.remove(feature)

    feature_is_global = {f: (f in global_candidates) for f in features}
    if any(feature_is_global.values()):
         logger.info(f"Identified potential global features: {[f for f, is_g in feature_is_global.items() if is_g]}")

    num_instances_processed = 0
    for instance_key, domain_info in train_data.items():
        num_instances_processed += 1
        for feature in features:
            if feature in domain_info:
                value = domain_info[feature]
                if feature_is_global[feature]:
                     scalar_val = value if np.isscalar(value) else value.item(0)
                     if pd.notna(scalar_val): feature_values[feature].append(scalar_val)
                elif isinstance(value, np.ndarray):
                     valid_values = value[~np.isnan(value)] # Filter NaNs during collection
                     if valid_values.size > 0: feature_values[feature].extend(valid_values.tolist())
                elif np.isscalar(value) and pd.notna(value):
                     feature_values[feature].append(value)

        if num_instances_processed % 1000 == 0:
             logger.debug(f"Processed {num_instances_processed} instances for feature normalization...")


    logger.info("Calculating normalization statistics...")
    norm_params = {}
    for feature in features:
        values_list = feature_values[feature]
        if values_list:
            values_np = np.array(values_list, dtype=np.float64)
            if np.isnan(values_np).any(): # Should be less likely now
                 logger.warning(f"NaNs found in feature '{feature}' during final calculation. Filtering them out.")
                 values_np = values_np[~np.isnan(values_np)]
                 if values_np.size == 0:
                      logger.warning(f"No valid numeric values left for feature '{feature}' after NaN filter.")
                      continue

            if values_np.size == 0:
                 logger.warning(f"No values collected for feature '{feature}'. Skipping normalization.")
                 continue

            feature_min = float(np.min(values_np))
            feature_max = float(np.max(values_np))
            feature_mean = float(np.mean(values_np))
            feature_std = float(np.std(values_np))

            if feature_std < 1e-9:
                logger.warning(f"Feature '{feature}' has near-zero standard deviation ({feature_std:.2e}). Normalization might be unstable. Min={feature_min}, Max={feature_max}")
                feature_std = 0.0

            norm_params[feature] = {
                'min': feature_min,
                'max': feature_max,
                'mean': feature_mean,
                'std': feature_std,
                'count': values_np.size,
                'is_global': feature_is_global[feature]
            }
            log_level = logging.DEBUG if values_np.size > 1000 else logging.INFO
            logger.log(log_level, f"Feature '{feature}' (Global={feature_is_global[feature]}): "
                       f"Count={values_np.size}, Min={feature_min:.4f}, Max={feature_max:.4f}, "
                       f"Mean={feature_mean:.4f}, Std={feature_std:.4f}")
        else:
            logger.warning(f"No values found for feature '{feature}'. Skipping normalization.")

    return norm_params


def save_split_data(data: Dict[str, Dict], output_dir: str, split_name: str, feature_list: Optional[List[str]] = None):
    """
    Save split data (instance list, FASTA, RMSF, Temp, Features) using instance_keys.
    """
    if not data:
        logger.warning(f"No data to save for split '{split_name}'. Skipping save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    instance_keys = sorted(list(data.keys()))

    instance_list_path = os.path.join(output_dir, f"{split_name}_instances.txt")
    try:
        with open(instance_list_path, 'w') as f:
            for key in instance_keys:
                f.write(f"{key}\n")
        logger.info(f"Saved {len(instance_keys)} instance keys to {instance_list_path}")
    except IOError as e:
        logger.error(f"Error writing instance list {instance_list_path}: {e}")

    fasta_path = os.path.join(output_dir, f"{split_name}_sequences.fasta")
    sequences_saved = 0
    try:
        with open(fasta_path, 'w') as f:
            for key in instance_keys:
                instance_info = data.get(key, {})
                if 'sequence' in instance_info and instance_info['sequence']:
                    f.write(f">{key}\n{instance_info['sequence']}\n")
                    sequences_saved += 1
                else:
                    logger.warning(f"Missing or empty 'sequence' key for instance {key} when saving FASTA for split {split_name}.")
        logger.info(f"Saved {sequences_saved} sequences to {fasta_path}")
    except IOError as e:
        logger.error(f"Error writing FASTA file {fasta_path}: {e}")

    rmsf_path = os.path.join(output_dir, f"{split_name}_rmsf.npy")
    rmsf_data_to_save = {}
    for key in instance_keys:
        instance_info = data.get(key, {})
        if 'rmsf' in instance_info:
            rmsf_array = instance_info['rmsf']
            if isinstance(rmsf_array, np.ndarray) and rmsf_array.dtype == np.float32:
                rmsf_data_to_save[key] = rmsf_array
            else:
                try:
                    rmsf_data_to_save[key] = np.array(rmsf_array, dtype=np.float32)
                except Exception as conv_err:
                    logger.error(f"Could not convert RMSF data for instance {key} to numpy array: {conv_err}. Skipping RMSF for this instance.")
                    continue
        else:
            logger.warning(f"Missing 'rmsf' key for instance {key} when saving RMSF data for split {split_name}.")

    if rmsf_data_to_save:
        try:
            np.save(rmsf_path, rmsf_data_to_save, allow_pickle=True)
            logger.info(f"Saved RMSF data for {len(rmsf_data_to_save)} instances to {rmsf_path}")
        except Exception as e:
            logger.error(f"Error saving RMSF numpy file {rmsf_path}: {e}", exc_info=True)
    else:
        logger.warning(f"No valid RMSF data found to save for split {split_name}.")

    temp_path = os.path.join(output_dir, f"{split_name}_temperatures.npy")
    temp_data_to_save = {}
    for key in instance_keys:
        instance_info = data.get(key, {})
        if 'temperature' in instance_info:
            temp_val = instance_info['temperature']
            try:
                temp_data_to_save[key] = float(temp_val)
            except (ValueError, TypeError) as temp_err:
                logger.error(f"Could not convert temperature for instance {key} to float: Value='{temp_val}'. Error: {temp_err}. Skipping temperature for this instance.")
                continue
        else:
            logger.warning(f"Missing 'temperature' key for instance {key} when saving temperature data for split {split_name}.")

    if temp_data_to_save:
        try:
            np.save(temp_path, temp_data_to_save, allow_pickle=True)
            logger.info(f"Saved Temperature data for {len(temp_data_to_save)} instances to {temp_path}")
        except Exception as e:
            logger.error(f"Error saving Temperature numpy file {temp_path}: {e}", exc_info=True)
    else:
        logger.warning(f"No valid Temperature data found to save for split {split_name}.")

    if feature_list:
         logger.info(f"Saving structural features for split {split_name}: {feature_list}")
         features_found_in_data = set()
         if data:
              first_key = next(iter(data))
              features_found_in_data = set(data[first_key].keys())

         for feature in feature_list:
            if feature not in features_found_in_data:
                 logger.debug(f"Feature '{feature}' not found in processed data keys. Skipping save.")
                 continue

            feature_data_to_save = {}
            feature_is_global = None

            for key in instance_keys:
                instance_info = data.get(key, {})
                if feature in instance_info:
                    feature_val = instance_info[feature]

                    if feature_is_global is None:
                        feature_is_global = np.isscalar(feature_val) or (isinstance(feature_val, np.ndarray) and feature_val.size == 1)

                    if feature_is_global:
                         try:
                             scalar_val = float(feature_val.item(0)) if isinstance(feature_val, np.ndarray) else float(feature_val)
                             feature_data_to_save[key] = scalar_val
                         except (ValueError, TypeError) as feat_err:
                             logger.error(f"Could not process global feature '{feature}' for instance {key}. Value='{feature_val}'. Error: {feat_err}. Skipping.")
                             continue
                    else:
                         if isinstance(feature_val, np.ndarray) and feature_val.dtype == np.float32:
                              feature_data_to_save[key] = feature_val
                         else:
                              try:
                                   feature_data_to_save[key] = np.array(feature_val, dtype=np.float32)
                              except Exception as conv_err:
                                   logger.error(f"Could not convert feature '{feature}' for instance {key} to numpy array: {conv_err}. Skipping feature for this instance.")
                                   continue

            if feature_data_to_save:
                feature_path = os.path.join(output_dir, f"{split_name}_{feature}.npy")
                try:
                    np.save(feature_path, feature_data_to_save, allow_pickle=True)
                    logger.info(f"Saved '{feature}' data for {len(feature_data_to_save)} instances to {feature_path}")
                except Exception as e:
                    logger.error(f"Error saving '{feature}' numpy file {feature_path}: {e}", exc_info=True)
            else:
                logger.warning(f"No valid data found for feature '{feature}' in split {split_name}.")
    else:
         logger.info(f"No feature list provided. Skipping saving of individual feature files for split {split_name}.")


def calculate_and_save_temp_scaling(train_data: Dict[str, Dict], output_dir: str, filename: str):
    """
    Calculates min/max temperature from the training data instances and saves them.
    """
    if not train_data:
        logger.error("No training data provided. Cannot calculate temperature scaling parameters.")
        return

    temps = []
    for instance_key, instance_info in train_data.items():
        if 'temperature' in instance_info:
            try:
                temps.append(float(instance_info['temperature']))
            except (ValueError, TypeError):
                 logger.warning(f"Invalid temperature value '{instance_info['temperature']}' for instance {instance_key}. Skipping for scaling calculation.")

    if not temps:
        logger.error("No valid temperature values found in training data. Cannot calculate scaling parameters.")
        return

    temp_min = float(np.min(temps))
    temp_max = float(np.max(temps))
    if temp_min > temp_max: temp_min, temp_max = temp_max, temp_min

    scaling_params = {'temp_min': temp_min, 'temp_max': temp_max}
    save_path = os.path.join(output_dir, filename)

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(scaling_params, f, indent=4)
        logger.info(f"Calculated temperature scaling params (Min={temp_min:.2f}, Max={temp_max:.2f}) using {len(temps)} training instance temperatures.")
        logger.info(f"Saved temperature scaling parameters to {save_path}")
    except IOError as e:
        logger.error(f"Error saving temperature scaling parameters to {save_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving temperature scaling parameters: {e}", exc_info=True)

def save_feature_normalization_params(norm_params: Dict[str, Dict[str, float]], output_dir: str, filename: str):
    """
    Save feature normalization parameters to a JSON file.
    """
    if not norm_params:
        logger.warning("No normalization parameters to save.")
        return

    save_path = os.path.join(output_dir, filename)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            serializable_params = {}
            for feature, params in norm_params.items():
                 # Convert potentially numpy types (like count) to standard python types
                 serializable_params[feature] = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
                                                   if isinstance(v, (int, float, np.number)) else v
                                                   for k, v in params.items()}

            json.dump(serializable_params, f, indent=4)
        logger.info(f"Saved feature normalization parameters for {len(norm_params)} features to {save_path}")
    except IOError as e:
        logger.error(f"Error saving feature normalization parameters to {save_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving feature normalization parameters: {e}", exc_info=True)

# --- Main Processing Function ---
def process_data(csv_path: str, output_dir: str, temp_scaling_filename: str, config: Dict = None, train_ratio=0.85, val_ratio=0.075, seed=42): # <-- Adjusted defaults
    """Main function to process RMSF data, extract features, create splits, and save normalization params."""
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    logger.info(f"--- Starting Data Processing Pipeline (Handles Multiple Temps, Imputes NaN, Skips Gaps) ---")
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output Directory: {output_dir}")
    # Use updated ratios in log message
    logger.info(f"Split Ratios (Topology-based): Train={train_ratio:.3f}, Val={val_ratio:.3f}, Test={test_ratio:.3f}")
    logger.info(f"Random Seed: {seed}")
    logger.info(f"Temp Scaling Filename: {temp_scaling_filename}")

    try:
        df = load_data(csv_path, config)
        if df is None: raise ValueError("Failed to load data.")

        instance_groups = group_by_domain_and_temp(df)
        if not instance_groups: raise ValueError("Failed to group data by domain and temperature.")

        # Includes NaN imputation and gap skipping
        data = extract_sequence_rmsf_temp_features(instance_groups, config)
        if not data: raise ValueError("No valid instance data extracted after cleaning.")

        # Pass potentially adjusted ratios
        train_data, val_data, test_data = split_by_topology(data, train_ratio, val_ratio, seed)

        if not train_data:
             logger.error("Training data split is empty. Cannot proceed with normalization or saving.")
             return None, None, None

        # Determine features available for normalization from the actual training data
        first_train_key = next(iter(train_data))
        available_features = [k for k in train_data[first_train_key].keys() if k not in ['sequence', 'rmsf', 'temperature']]
        logger.info(f"Features available for normalization (based on first train instance): {available_features}")

        features_to_normalize = []
        if config and config.get('data', {}).get('features', {}):
            feature_config = config['data']['features']
            # Check config flags against available features
            if feature_config.get('use_position_info', True) and 'normalized_resid' in available_features: features_to_normalize.append('normalized_resid')
            if feature_config.get('use_structure_info', True):
                if 'core_exterior_encoded' in available_features: features_to_normalize.append('core_exterior_encoded')
                if 'secondary_structure_encoded' in available_features: features_to_normalize.append('secondary_structure_encoded')
            if feature_config.get('use_accessibility', True) and 'relative_accessibility' in available_features: features_to_normalize.append('relative_accessibility')
            if feature_config.get('use_backbone_angles', True):
                if 'phi_norm' in available_features: features_to_normalize.append('phi_norm')
                if 'psi_norm' in available_features: features_to_normalize.append('psi_norm')
            if feature_config.get('use_protein_size', True) and 'protein_size' in available_features: features_to_normalize.append('protein_size')
            if feature_config.get('use_voxel_rmsf', True) and 'voxel_rmsf' in available_features: features_to_normalize.append('voxel_rmsf')
            if feature_config.get('use_bfactor', True) and 'bfactor_norm' in available_features: features_to_normalize.append('bfactor_norm')
        else:
            features_to_normalize = available_features

        logger.info(f"Final list of features selected for normalization: {features_to_normalize}")

        norm_params = calculate_feature_normalization_params(train_data, features_to_normalize)

        if norm_params and config and 'data' in config and 'features' in config['data']:
            normalization_params_file = config['data']['features'].get('normalization_params_file', 'feature_normalization.json')
            save_feature_normalization_params(norm_params, output_dir, normalization_params_file)

        # Save Splits
        save_split_data(train_data, output_dir, 'train', features_to_normalize)
        save_split_data(val_data, output_dir, 'val', features_to_normalize)
        save_split_data(test_data, output_dir, 'test', features_to_normalize)

        calculate_and_save_temp_scaling(train_data, output_dir, temp_scaling_filename)

        logger.info("--- Data Processing Completed Successfully ---")
        return train_data, val_data, test_data

    except FileNotFoundError as e:
        logger.error(f"Processing failed: {e}")
        return None, None, None
    except ValueError as e:
        logger.error(f"Processing failed: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)
        return None, None, None


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Process protein RMSF data (multi-temp aware), extract features, split by topology, impute NaNs, skip gaps.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the input enriched RMSF CSV file.')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save the processed data splits and scaling info.')
    parser.add_argument('--scaling_file', type=str, default='temp_scaling_params.json', help='Filename for saving temperature scaling parameters (min/max).')
    # --- Adjusted default split ratios ---
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Fraction of topologies for the training set.')
    parser.add_argument('--val_ratio', type=float, default=0.075, help='Fraction of topologies for the validation set.')
    # Test ratio is inferred: 1.0 - train_ratio - val_ratio
    # --- End adjustment ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling topologies.')
    args = parser.parse_args()

    config_data = None
    if args.config:
        if not os.path.exists(args.config):
             logger.error(f"Configuration file not found: {args.config}")
             sys.exit(1)
        try:
            import yaml
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {args.config}")
        except ImportError:
             logger.error("PyYAML is not installed. Cannot load config file. Please install with 'pip install pyyaml'.")
             sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {args.config}: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load config file {args.config}: {e}", exc_info=True)
            logger.warning("Proceeding without loaded configuration (using defaults and CLI args).")

    if not args.csv or not os.path.exists(args.csv):
         logger.error(f"Input CSV file not found or not specified: {args.csv}")
         sys.exit(1)

    # Use CLI args for ratios, falling back to new defaults if not provided
    train_r = args.train_ratio
    val_r = args.val_ratio

    process_data(
        csv_path=args.csv,
        output_dir=args.output,
        temp_scaling_filename=args.scaling_file,
        config=config_data,
        train_ratio=train_r, # Pass value from args (or its default)
        val_ratio=val_r,   # Pass value from args (or its default)
        seed=args.seed
    )

