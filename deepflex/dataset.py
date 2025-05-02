import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import logging
import json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedRMSFDataset(Dataset):
    """
    PyTorch Dataset for Enhanced Temperature-Aware RMSF prediction.

    Handles loading sequences, target RMSF values, temperatures, and structural features.
    """
    def __init__(self,
                 domain_ids: List[str],
                 sequences: Dict[str, str],
                 rmsf_values: Dict[str, np.ndarray],
                 temperatures: Dict[str, float],
                 feature_data: Dict[str, Dict[str, np.ndarray]],
                 feature_norm_params: Optional[Dict[str, Dict[str, float]]] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the enhanced dataset.

        Args:
            domain_ids: Ordered list of domain IDs for this dataset split.
            sequences: Dictionary mapping domain IDs to amino acid sequences.
            rmsf_values: Dictionary mapping domain IDs to target RMSF values (NumPy arrays).
            temperatures: Dictionary mapping domain IDs to temperature values (float).
            feature_data: Dictionary mapping domain IDs to feature data (dictionaries of feature name to NumPy arrays).
            feature_norm_params: Optional dictionary of normalization parameters for each feature.
            config: Optional configuration dictionary.
        """
        self.domain_ids = domain_ids
        self.sequences = sequences
        self.rmsf_values = rmsf_values  # Target RMSF
        self.temperatures = temperatures
        self.feature_data = feature_data
        self.feature_norm_params = feature_norm_params
        self.config = config or {}
        
        # Get list of enabled features from configuration
        self.enabled_features = []
        feature_config = self.config.get('data', {}).get('features', {})
        
        # Position information
        if feature_config.get('use_position_info', True) and 'normalized_resid' in self.get_available_features():
            self.enabled_features.append('normalized_resid')
        
        # Structure information
        if feature_config.get('use_structure_info', True):
            if 'core_exterior_encoded' in self.get_available_features():
                self.enabled_features.append('core_exterior_encoded')
            if 'secondary_structure_encoded' in self.get_available_features():
                self.enabled_features.append('secondary_structure_encoded')
        
        # Accessibility
        if feature_config.get('use_accessibility', True) and 'relative_accessibility' in self.get_available_features():
            self.enabled_features.append('relative_accessibility')
        
        # Backbone angles
        if feature_config.get('use_backbone_angles', True):
            if 'phi_norm' in self.get_available_features():
                self.enabled_features.append('phi_norm')
            if 'psi_norm' in self.get_available_features():
                self.enabled_features.append('psi_norm')
        
        # Protein size (global feature)
        if feature_config.get('use_protein_size', True) and 'protein_size' in self.get_available_features():
            self.enabled_features.append('protein_size')
        
        # Additional predictive features
        if feature_config.get('use_voxel_rmsf', True) and 'voxel_rmsf' in self.get_available_features():
            self.enabled_features.append('voxel_rmsf')
        
        if feature_config.get('use_bfactor', True) and 'bfactor_norm' in self.get_available_features():
            self.enabled_features.append('bfactor_norm')

        logger.info(f"Enabled features: {self.enabled_features}")
        
        # Data Consistency Check
        valid_domain_ids = []
        removed_count = 0
        for did in list(self.domain_ids):  # Iterate over a copy
            # Check for sequence, RMSF, AND temperature
            if did in self.sequences and did in self.rmsf_values and did in self.temperatures:
                # Basic length check remains useful
                if len(self.sequences[did]) != len(self.rmsf_values[did]):
                    logger.warning(f"Length mismatch for {did}: Seq={len(self.sequences[did])}, RMSF={len(self.rmsf_values[did])}. Removing.")
                    removed_count += 1
                # Check if temperature is valid
                elif self.temperatures[did] is None or np.isnan(self.temperatures[did]):
                    logger.warning(f"Invalid temperature for {did}: {self.temperatures[did]}. Removing.")
                    removed_count += 1
                # Check if all enabled features are available for this domain
                elif not self._check_features_available(did):
                    logger.warning(f"Missing features for {did}. Removing.")
                    removed_count += 1
                else:
                    valid_domain_ids.append(did)  # Keep if all checks pass
            else:
                logger.warning(f"Domain ID {did} missing sequence, RMSF, or temperature. Removing.")
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} domain IDs from dataset due to missing/inconsistent data.")
            self.domain_ids = valid_domain_ids

        # Calculate and log dataset statistics
        self._log_stats()

    def get_available_features(self) -> List[str]:
        """Get list of available features in the dataset."""
        available_features = set()
        # Check the first domain for available features
        for did in self.domain_ids:
            if did in self.feature_data:
                for feature in self.feature_data[did]:
                    available_features.add(feature)
            break
        return sorted(list(available_features))

    def _check_features_available(self, domain_id: str) -> bool:
        """Check if all enabled features are available for a domain."""
        if domain_id not in self.feature_data:
            return False
        
        for feature in self.enabled_features:
            if feature not in self.feature_data[domain_id]:
                return False
        
        return True

    def _normalize_feature(self, feature_name: str, feature_value: np.ndarray) -> np.ndarray:
        """
        Normalize a feature using stored normalization parameters.
        
        Args:
            feature_name: Name of the feature to normalize
            feature_value: Feature value array or scalar
            
        Returns:
            Normalized feature value
        """
        if not self.feature_norm_params or feature_name not in self.feature_norm_params:
            return feature_value
        
        params = self.feature_norm_params[feature_name]
        
        # Min-max normalization
        if 'min' in params and 'max' in params:
            feature_min = params['min']
            feature_max = params['max']
            
            # Handle case where min == max (avoid division by zero)
            if feature_max - feature_min < 1e-8:
                return np.zeros_like(feature_value, dtype=np.float32)
            
            return (feature_value - feature_min) / (feature_max - feature_min)
        
        # Z-score normalization
        elif 'mean' in params and 'std' in params:
            feature_mean = params['mean']
            feature_std = params['std']
            
            # Handle case where std is near zero
            if feature_std < 1e-8:
                return np.zeros_like(feature_value, dtype=np.float32)
            
            return (feature_value - feature_mean) / feature_std
        
        # No normalization if parameters are incomplete
        return feature_value

    def _log_stats(self):
        """Log statistics about the loaded dataset."""
        if not self.domain_ids:
            logger.warning("Dataset created with 0 proteins.")
            return

        num_proteins = len(self.domain_ids)
        logger.info(f"Dataset created with {num_proteins} proteins.")
        try:
            seq_lengths = [len(self.sequences[did]) for did in self.domain_ids]
            rmsf_lengths = [len(self.rmsf_values[did]) for did in self.domain_ids]
            temp_values = [self.temperatures[did] for did in self.domain_ids]

            logger.info(f"  Sequence length stats: Min={min(seq_lengths)}, Max={max(seq_lengths)}, " +
                        f"Mean={np.mean(seq_lengths):.1f}, Median={np.median(seq_lengths):.1f}")
            logger.info(f"  RMSF length stats:     Min={min(rmsf_lengths)}, Max={max(rmsf_lengths)}, " +
                        f"Mean={np.mean(rmsf_lengths):.1f}, Median={np.median(rmsf_lengths):.1f}")
            logger.info(f"  Temperature stats:     Min={min(temp_values):.1f}, Max={max(temp_values):.1f}, " +
                        f"Mean={np.mean(temp_values):.1f}, Median={np.median(temp_values):.1f}")

            if np.mean(seq_lengths) != np.mean(rmsf_lengths):
                logger.warning("Mean sequence length differs from mean RMSF length. Verify processing.")
                
            # Log feature statistics
            for feature in self.enabled_features:
                feature_stats = []
                for did in self.domain_ids:
                    if did in self.feature_data and feature in self.feature_data[did]:
                        feature_val = self.feature_data[did][feature]
                        if isinstance(feature_val, np.ndarray) and feature_val.size > 0:
                            feature_stats.append(np.mean(feature_val))
                        elif np.isscalar(feature_val):
                            feature_stats.append(feature_val)
                
                if feature_stats:
                    logger.info(f"  {feature} stats:     Min={min(feature_stats):.4f}, Max={max(feature_stats):.4f}, " +
                                f"Mean={np.mean(feature_stats):.4f}, Median={np.median(feature_stats):.4f}")
                
        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {e}", exc_info=True)

    def __len__(self) -> int:
        return len(self.domain_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get data for a single protein by index.

        Args:
            idx: Index of the protein.

        Returns:
            Dictionary containing:
              - 'domain_id': The domain identifier (string).
              - 'sequence': The amino acid sequence (string).
              - 'rmsf': The target RMSF values (NumPy array of float32).
              - 'temperature': The temperature value (float).
              - 'features': Dictionary of feature arrays (if enabled).
        """
        if idx < 0 or idx >= len(self.domain_ids):
            raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.domain_ids)}")

        domain_id = self.domain_ids[idx]

        # Retrieve data, handling potential KeyError if consistency check failed unexpectedly
        try:
            sequence = self.sequences[domain_id]
            rmsf = self.rmsf_values[domain_id]
            temperature = self.temperatures[domain_id]
        except KeyError as e:
            logger.error(f"Data inconsistency: Cannot find '{e}' for domain ID {domain_id} at index {idx}. Was it filtered out?")
            raise RuntimeError(f"Inconsistent dataset state: Missing data for {domain_id}") from e

        # Ensure RMSF is float32
        if rmsf.dtype != np.float32:
            rmsf = rmsf.astype(np.float32)
            
        # Collect and normalize feature data
        features = {}
        for feature in self.enabled_features:
            if domain_id in self.feature_data and feature in self.feature_data[domain_id]:
                feature_val = self.feature_data[domain_id][feature]
                
                # Ensure feature value is a numpy array of float32
                if np.isscalar(feature_val):
                    # Handle global features (like protein_size)
                    feature_val = np.array([feature_val], dtype=np.float32)
                elif not isinstance(feature_val, np.ndarray):
                    feature_val = np.array(feature_val, dtype=np.float32)
                elif feature_val.dtype != np.float32:
                    feature_val = feature_val.astype(np.float32)
                
                # Normalize the feature if normalization parameters are available
                normalized_val = self._normalize_feature(feature, feature_val)
                features[feature] = normalized_val

        return {
            'domain_id': domain_id,
            'sequence': sequence,
            'rmsf': rmsf,  # This is the TARGET RMSF
            'temperature': float(temperature),  # Ensure float type
            'features': features  # Dictionary of feature arrays
        }

def load_sequences_from_fasta(fasta_path: str) -> Dict[str, str]:
    """Loads sequences from a FASTA file."""
    sequences = {}
    current_id = None
    current_seq = ""
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = current_seq
                    current_id = line[1:].split()[0]  # Use ID before first space
                    current_seq = ""
                else:
                    current_seq += line.upper()
            if current_id is not None:  # Add last sequence
                sequences[current_id] = current_seq
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except FileNotFoundError:
        logger.error(f"FASTA file not found: {fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading FASTA file {fasta_path}: {e}")
        raise
    return sequences

def load_numpy_dict(npy_path: str) -> Dict[str, Any]:
    """Loads a dictionary saved as a NumPy file."""
    if not os.path.exists(npy_path):
        logger.error(f"NumPy file not found: {npy_path}")
        raise FileNotFoundError(f"NumPy file not found: {npy_path}")
    try:
        # allow_pickle=True is required for loading dictionaries
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        # Ensure keys are strings for consistency
        string_key_data = {str(k): v for k, v in loaded_data.items()}
        logger.info(f"Loaded {len(string_key_data)} entries from {npy_path}")
        return string_key_data
    except Exception as e:
        logger.error(f"Error loading or processing NumPy dictionary from {npy_path}: {e}")
        raise

def load_feature_norm_params(json_path: str) -> Dict[str, Dict[str, float]]:
    """Load feature normalization parameters from a JSON file."""
    if not os.path.exists(json_path):
        logger.warning(f"Feature normalization parameters file not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            norm_params = json.load(f)
        logger.info(f"Loaded normalization parameters for {len(norm_params)} features from {json_path}")
        return norm_params
    except Exception as e:
        logger.error(f"Error loading feature normalization parameters from {json_path}: {e}")
        return {}

def load_split_data(data_dir: str, split: str, config: Optional[Dict] = None) -> Tuple[List[str], Dict[str, str], Dict[str, np.ndarray], Dict[str, float], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    """
    Load data (domain IDs, sequences, RMSF values, temperatures, feature data) for a specific split.

    Args:
        data_dir: Directory containing the processed data files.
        split: Split name ('train', 'val', or 'test').
        config: Optional configuration dictionary.

    Returns:
        Tuple of (domain_ids, sequences, rmsf_values, temperatures, feature_data, feature_norm_params).
        Returns ([], {}, {}, {}, {}, {}) if data loading fails for essential components.
    """
    logger.info(f"--- Loading {split} data from directory: {data_dir} ---")
    sequences, rmsf_values, temperatures = {}, {}, {}
    feature_data = {}
    domain_ids = []
    feature_norm_params = {}

    try:
        # --- Load normalization parameters if available ---
        norm_params_file = None
        if config and 'data' in config and 'features' in config['data']:
            norm_params_file = config['data']['features'].get('normalization_params_file', 'feature_normalization.json')
        
        if norm_params_file:
            norm_params_path = os.path.join(data_dir, norm_params_file)
            feature_norm_params = load_feature_norm_params(norm_params_path)
        
        # --- Load domain IDs (essential) ---
                # --- Load instance keys (essential) --- # Changed comment for clarity
        instance_keys_path = os.path.join(data_dir, f"{split}_instances.txt") # <-- CHANGED FILENAME
        if not os.path.exists(instance_keys_path):
            logger.error(f"Instance key file not found: {instance_keys_path}") # <-- CHANGED ERROR MSG
            return [], {}, {}, {}, {}, {}
        with open(instance_keys_path, 'r') as f:
            # Still store in variable called domain_ids internally, or rename if preferred (e.g., instance_keys_list)
            # Sticking with domain_ids is fine as long as we remember it holds 'domain_id@temp' keys
            domain_ids = [line.strip() for line in f if line.strip()]
        if not domain_ids:
            logger.warning(f"Instance key file is empty: {instance_keys_path}") # <-- CHANGED WARNING MSG
        logger.info(f"Loaded {len(domain_ids)} instance keys from {instance_keys_path}") # <-- CHANGED INFO MSG

        # --- Load sequences (essential) ---
        sequences_path = os.path.join(data_dir, f"{split}_sequences.fasta")
        sequences = load_sequences_from_fasta(sequences_path)
        if not sequences:
            logger.error(f"No sequences loaded from required file: {sequences_path}")
            return [], {}, {}, {}, {}, {}  # Treat as fatal if no sequences

        # --- Load RMSF values (essential) ---
        rmsf_path = os.path.join(data_dir, f"{split}_rmsf.npy")
        rmsf_dict = load_numpy_dict(rmsf_path)
        # Ensure values are float32 numpy arrays
        rmsf_values = {k: np.array(v, dtype=np.float32) for k, v in rmsf_dict.items()}
        if not rmsf_values:
            logger.error(f"No RMSF data loaded from required file: {rmsf_path}")
            return [], {}, {}, {}, {}, {}  # Treat as fatal

        # --- Load Temperatures (essential) ---
        temperatures_path = os.path.join(data_dir, f"{split}_temperatures.npy")
        temp_dict = load_numpy_dict(temperatures_path)
        # Ensure values are floats
        temperatures = {k: float(v) for k, v in temp_dict.items()}
        if not temperatures:
            logger.error(f"No Temperature data loaded from required file: {temperatures_path}")
            return [], {}, {}, {}, {}, {}  # Treat as fatal
        
        # --- Load Structural Features (if available) ---
        # Initialize feature_data dictionary for each domain
        for domain_id in domain_ids:
            feature_data[domain_id] = {}
        
        # Determine which features to load based on config
        features_to_load = []
        if config and config.get('data', {}).get('features', {}):
            feature_config = config['data']['features']
            
            if feature_config.get('use_position_info', True):
                features_to_load.append('normalized_resid')
            
            if feature_config.get('use_structure_info', True):
                features_to_load.extend(['core_exterior_encoded', 'secondary_structure_encoded'])
            
            if feature_config.get('use_accessibility', True):
                features_to_load.append('relative_accessibility')
            
            if feature_config.get('use_backbone_angles', True):
                features_to_load.extend(['phi_norm', 'psi_norm'])
            
            if feature_config.get('use_protein_size', True):
                features_to_load.append('protein_size')
            
            if feature_config.get('use_voxel_rmsf', True):
                features_to_load.append('voxel_rmsf')
            
            if feature_config.get('use_bfactor', True):
                features_to_load.append('bfactor_norm')
        else:
            # If no config, try to load all common structural features
            features_to_load = [
                'normalized_resid', 'core_exterior_encoded', 'secondary_structure_encoded',
                'relative_accessibility', 'phi_norm', 'psi_norm', 'protein_size',
                'voxel_rmsf', 'bfactor_norm'
            ]
        
        # Load each feature file if it exists
        for feature in features_to_load:
            feature_path = os.path.join(data_dir, f"{split}_{feature}.npy")
            if os.path.exists(feature_path):
                try:
                    feature_dict = load_numpy_dict(feature_path)
                    for domain_id, feature_val in feature_dict.items():
                        if domain_id in feature_data:
                            feature_data[domain_id][feature] = feature_val
                    logger.info(f"Loaded {feature} data for {len(feature_dict)} domains")
                except Exception as e:
                    logger.warning(f"Error loading {feature} data: {e}")
            else:
                logger.warning(f"{feature} data file not found: {feature_path}")

    except FileNotFoundError as e:
        logger.error(f"Failed to load essential data file: {e}")
        return [], {}, {}, {}, {}, {}
    except Exception as e:
        logger.error(f"An error occurred during data loading for split '{split}': {e}", exc_info=True)
        return [], {}, {}, {}, {}, {}

    # --- Verify data consistency across all loaded components ---
    logger.info("Verifying data consistency for split '{}'...".format(split))
    original_domain_count = len(domain_ids)
    valid_domain_ids = []
    missing_data_counts = defaultdict(int)
    length_mismatches = 0

    for did in domain_ids:
        has_seq = did in sequences
        has_rmsf = did in rmsf_values
        has_temp = did in temperatures
        has_features = did in feature_data and len(feature_data[did]) > 0

        if has_seq and has_rmsf and has_temp:
            # Check sequence-RMSF length consistency
            seq_len = len(sequences[did])
            rmsf_len = len(rmsf_values[did])
            if seq_len == rmsf_len:
                # Check temperature validity
                if temperatures[did] is not None and not np.isnan(temperatures[did]):
                    # If we require features, check they are available
                    if config and config.get('model', {}).get('architecture', {}).get('use_enhanced_features', True):
                        if has_features:
                            valid_domain_ids.append(did)
                        else:
                            missing_data_counts['missing_features'] += 1
                            logger.debug(f"Missing features for {did}. Removing.")
                    else:
                        # Features not required
                        valid_domain_ids.append(did)
                else:
                    missing_data_counts['invalid_temp'] += 1
                    logger.debug(f"Invalid temperature for {did}. Removing.")
            else:
                length_mismatches += 1
                logger.debug(f"Length mismatch for {did}: seq={seq_len}, RMSF={rmsf_len}. Removing.")
        else:
            if not has_seq: missing_data_counts['sequence'] += 1; logger.debug(f"Missing sequence for {did}")
            if not has_rmsf: missing_data_counts['rmsf'] += 1; logger.debug(f"Missing RMSF for {did}")
            if not has_temp: missing_data_counts['temperature'] += 1; logger.debug(f"Missing temperature for {did}")

    logger.info(f"Initial domain IDs in list: {original_domain_count}")
    if sum(missing_data_counts.values()) > 0:
        logger.warning(f"Missing data counts: {dict(missing_data_counts)}")
    if length_mismatches > 0:
        logger.warning(f"Found {length_mismatches} domains with sequence-RMSF length mismatches.")

    final_domain_count = len(valid_domain_ids)
    if final_domain_count != original_domain_count:
        removed_count = original_domain_count - final_domain_count
        logger.info(f"Removed {removed_count} domains due to inconsistencies.")
        logger.info(f"Final number of valid, consistent domains for split '{split}': {final_domain_count}")

    # Filter all dictionaries to only include valid domains
    final_sequences = {did: sequences[did] for did in valid_domain_ids if did in sequences}
    final_rmsf_values = {did: rmsf_values[did] for did in valid_domain_ids if did in rmsf_values}
    final_temperatures = {did: temperatures[did] for did in valid_domain_ids if did in temperatures}
    final_feature_data = {did: feature_data.get(did, {}) for did in valid_domain_ids}

    if final_domain_count == 0:
        logger.error(f"No valid, consistent data found for split '{split}' after filtering. Please check the processed data files in {data_dir}.")
        # Return empty structures to avoid downstream errors
        return [], {}, {}, {}, {}, {}

    logger.info(f"--- Successfully loaded and verified {final_domain_count} samples for split '{split}' ---")
    return valid_domain_ids, final_sequences, final_rmsf_values, final_temperatures, final_feature_data, feature_norm_params

def enhanced_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for the Enhanced DataLoader.

    Batches domain IDs, sequences, RMSF values (as Tensors), temperatures (as Tensors),
    and feature data. Padding is NOT done here.

    Args:
        batch: List of items from the EnhancedRMSFDataset

    Returns:
        Dictionary of batched data
    """
    domain_ids = [item['domain_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    # Convert RMSF numpy arrays to tensors (target values)
    rmsf_values = [torch.tensor(item['rmsf'], dtype=torch.float32) for item in batch]
    # Extract and convert temperatures to a tensor
    temperatures = torch.tensor([item['temperature'] for item in batch], dtype=torch.float32)
    
    # Process feature data
    # First, determine which features are available in all batch items
    feature_names = set()
    for item in batch:
        feature_names.update(item['features'].keys())
    feature_names = sorted(list(feature_names))
    
    # Initialize feature tensors
    feature_tensors = {}
    for feature in feature_names:
        # Check if this is a per-residue feature or a global feature
        is_global_feature = all(
            feature in item['features'] and len(item['features'][feature].shape) == 1 and item['features'][feature].shape[0] == 1
            for item in batch if feature in item['features']
        )
        
        if is_global_feature:
            # Global feature - create a batch tensor
            global_values = []
            for item in batch:
                if feature in item['features']:
                    # Extract scalar value from the single-element array
                    global_values.append(float(item['features'][feature][0]))
                else:
                    # Use 0 as a placeholder if feature is missing
                    global_values.append(0.0)
            feature_tensors[feature] = torch.tensor(global_values, dtype=torch.float32)
        else:
            # Per-residue feature - store as list of tensors
            per_residue_values = []
            for item in batch:
                if feature in item['features']:
                    per_residue_values.append(torch.tensor(item['features'][feature], dtype=torch.float32))
                else:
                    # Use empty tensor if feature is missing for this item
                    # Length will be aligned with sequence in the model
                    per_residue_values.append(torch.tensor([], dtype=torch.float32))
            feature_tensors[feature] = per_residue_values

    return {
        'domain_ids': domain_ids,
        'sequences': sequences,
        'rmsf_values': rmsf_values,  # List of Tensors (targets)
        'temperatures': temperatures,  # Tensor of shape [batch_size] (input features)
        'features': feature_tensors  # Dictionary of feature tensors
    }

def create_enhanced_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    max_length: Optional[int] = None,
    length_bucket_size: int = 50,
    num_workers: int = 0,
    config: Optional[Dict] = None
) -> Optional[DataLoader]:
    """
    Creates a PyTorch DataLoader for the Enhanced RMSF Dataset with length-based batching.

    Args:
        data_dir: Directory containing the processed data splits.
        split: Split name ('train', 'val', or 'test').
        batch_size: Target number of sequences per batch.
        shuffle: Whether to shuffle data.
        max_length: Optional maximum sequence length for filtering.
        length_bucket_size: Size of length ranges for grouping.
        num_workers: Number of worker processes.
        config: Optional configuration dictionary.

    Returns:
        A PyTorch DataLoader instance, or None if data loading fails.
    """
    # 1. Load data (including temperatures and features)
    domain_ids, sequences, rmsf_values, temperatures, feature_data, feature_norm_params = load_split_data(data_dir, split, config)

    if not domain_ids:
        logger.error(f"Failed to load any valid data for split '{split}'. Cannot create DataLoader.")
        return None

    # 2. Filter by max length if specified
    if max_length is not None and max_length > 0:
        original_count = len(domain_ids)
        # Keep only IDs whose sequences are <= max_length
        filtered_domain_ids = [
            did for did in domain_ids if len(sequences.get(did, '')) <= max_length
        ]
        filtered_count = len(filtered_domain_ids)
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} sequences " +
                       f"longer than {max_length} residues for split '{split}'.")
            domain_ids = filtered_domain_ids
            # Filter all dictionaries based on the remaining domain_ids
            sequences = {did: sequences[did] for did in domain_ids if did in sequences}
            rmsf_values = {did: rmsf_values[did] for did in domain_ids if did in rmsf_values}
            temperatures = {did: temperatures[did] for did in domain_ids if did in temperatures}
            feature_data = {did: feature_data[did] for did in domain_ids if did in feature_data}

        if not domain_ids:
            logger.warning(f"No sequences remaining after filtering by max_length={max_length} for split '{split}'. Cannot create DataLoader.")
            return None

    # 3. Group domain IDs by length buckets
    length_buckets = defaultdict(list)
    for did in domain_ids:
        # Use sequence length for bucketing
        seq_len = len(sequences.get(did, ''))
        if seq_len > 0:  # Avoid bucketing empty sequences if any slipped through
            bucket_idx = seq_len // length_bucket_size
            length_buckets[bucket_idx].append(did)
        else:
            logger.warning(f"Domain ID {did} has zero length sequence during bucketing. Skipping.")

    if not length_buckets:
        logger.error(f"No non-empty sequences found to create length buckets for split '{split}'. Cannot create DataLoader.")
        return None

    logger.info(f"Grouped {len(domain_ids)} sequences into {len(length_buckets)} length buckets.")

    # 4. Create batches within buckets
    all_batches = []
    sorted_bucket_indices = sorted(length_buckets.keys())

    for bucket_idx in sorted_bucket_indices:
        bucket_domain_ids = length_buckets[bucket_idx]
        if shuffle:
            random.shuffle(bucket_domain_ids)

        for i in range(0, len(bucket_domain_ids), batch_size):
            batch_domain_ids = bucket_domain_ids[i: i + batch_size]
            all_batches.append(batch_domain_ids)

    # 5. Shuffle the order of batches for training
    if shuffle:
        random.shuffle(all_batches)

    # 6. Flatten the batches to get the final ordered list of domain IDs for the epoch
    ordered_domain_ids = [did for batch in all_batches for did in batch]

    # 7. Create the Dataset with the final order and all required data dicts
    dataset = EnhancedRMSFDataset(
        ordered_domain_ids, 
        sequences, 
        rmsf_values, 
        temperatures, 
        feature_data, 
        feature_norm_params,
        config
    )

    if len(dataset) == 0:
        logger.error(f"Final dataset for split '{split}' is empty after processing. Cannot create DataLoader.")
        return None

    # 8. Create the DataLoader
    logger.info(f"Creating DataLoader for {len(dataset)} samples for split '{split}' with batch size {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is handled by length batching strategy
        collate_fn=enhanced_collate_fn,  # Use enhanced collate function
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False  # Keep all data
    )

# For backward compatibility
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Legacy collate function for compatibility.
    
    This redirects to enhanced_collate_fn for consistent behavior.
    """
    return enhanced_collate_fn(batch)

# For backward compatibility
def create_length_batched_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    max_length: Optional[int] = None,
    length_bucket_size: int = 50,
    num_workers: int = 0
) -> Optional[DataLoader]:
    """
    Legacy function for backward compatibility.
    
    This redirects to create_enhanced_dataloader with default config.
    """
    return create_enhanced_dataloader(
        data_dir, split, batch_size, shuffle, max_length, length_bucket_size, num_workers, None
    )

# Example Usage (if script is run directly)
if __name__ == "__main__":
    logger.info("Testing Enhanced DataLoader creation...")
    
    # Create dummy data for testing
    dummy_data_dir = "data/processed_dummy_enhanced"
    os.makedirs(dummy_data_dir, exist_ok=True)

    # Sample configuration
    config = {
        'data': {
            'features': {
                'use_position_info': True,
                'use_structure_info': True,
                'use_accessibility': True,
                'use_backbone_angles': True,
                'use_protein_size': True,
                'use_voxel_rmsf': True,
                'use_bfactor': True,
                'normalization_params_file': 'feature_normalization.json'
            }
        },
        'model': {
            'architecture': {
                'use_enhanced_features': True
            }
        }
    }
    
    # Create dummy data
    dummy_domains = [f"D{i:03d}" for i in range(10)]
    dummy_sequences = {}
    dummy_rmsf = {}
    dummy_temps = {}
    dummy_features = {
        'normalized_resid': {},
        'core_exterior_encoded': {},
        'secondary_structure_encoded': {},
        'relative_accessibility': {},
        'phi_norm': {},
        'psi_norm': {},
        'protein_size': {},
        'voxel_rmsf': {}
    }
    
    for i, did in enumerate(dummy_domains):
        length = random.randint(50, 250)
        dummy_sequences[did] = "A" * length
        dummy_rmsf[did] = np.random.rand(length).astype(np.float32) * 2.0
        dummy_temps[did] = random.choice([298.0, 310.0, 320.0, 330.0])
        
        # Per-residue features
        dummy_features['normalized_resid'][did] = np.linspace(0, 1, length).astype(np.float32)
        dummy_features['core_exterior_encoded'][did] = np.random.randint(0, 2, length).astype(np.float32)
        dummy_features['secondary_structure_encoded'][did] = np.random.randint(0, 3, length).astype(np.float32)
        dummy_features['relative_accessibility'][did] = np.random.rand(length).astype(np.float32)
        dummy_features['phi_norm'][did] = np.random.rand(length).astype(np.float32)
        dummy_features['psi_norm'][did] = np.random.rand(length).astype(np.float32)
        dummy_features['voxel_rmsf'][did] = np.random.rand(length).astype(np.float32)
        
        # Global features
        dummy_features['protein_size'][did] = float(length)
    
    # Save dummy data
    with open(os.path.join(dummy_data_dir, "train_domains.txt"), "w") as f: 
        f.write("\n".join(dummy_domains))
    
    with open(os.path.join(dummy_data_dir, "train_sequences.fasta"), "w") as f:
        for did, seq in dummy_sequences.items(): 
            f.write(f">{did}\n{seq}\n")
    
    np.save(os.path.join(dummy_data_dir, "train_rmsf.npy"), dummy_rmsf)
    np.save(os.path.join(dummy_data_dir, "train_temperatures.npy"), dummy_temps)
    
    # Save feature files
    for feature, feature_dict in dummy_features.items():
        np.save(os.path.join(dummy_data_dir, f"train_{feature}.npy"), feature_dict)
    
    # Save feature normalization parameters
    norm_params = {
        'normalized_resid': {'min': 0.0, 'max': 1.0},
        'core_exterior_encoded': {'min': 0.0, 'max': 1.0},
        'secondary_structure_encoded': {'min': 0.0, 'max': 2.0},
        'relative_accessibility': {'min': 0.0, 'max': 1.0},
        'phi_norm': {'min': -1.0, 'max': 1.0},
        'psi_norm': {'min': -1.0, 'max': 1.0},
        'protein_size': {'min': 50.0, 'max': 250.0},
        'voxel_rmsf': {'min': 0.0, 'max': 2.0}
    }
    
    with open(os.path.join(dummy_data_dir, "feature_normalization.json"), "w") as f:
        json.dump(norm_params, f, indent=2)
    
    # Test dataloader creation
    train_loader = create_enhanced_dataloader(
        data_dir=dummy_data_dir,
        split='train',
        batch_size=2,
        shuffle=True,
        max_length=200,
        length_bucket_size=25,
        config=config
    )

    if train_loader:
        logger.info("DataLoader created successfully. Iterating through a few batches...")
        batch_count = 0
        max_batches_to_show = 2
        for i, batch in enumerate(train_loader):
            if i >= max_batches_to_show: break
            logger.info(f"Batch {i+1}:")
            logger.info(f"  Domain IDs: {batch['domain_ids']}")
            logger.info(f"  Num sequences: {len(batch['sequences'])}")
            logger.info(f"  Seq lengths: {[len(s) for s in batch['sequences']]}")
            logger.info(f"  RMSF Tensors: {[t.shape for t in batch['rmsf_values']]}")
            logger.info(f"  Temperatures Tensor: {batch['temperatures']}")
            logger.info(f"  Temperatures Tensor Shape: {batch['temperatures'].shape}")
            
            logger.info("  Features:")
            for feature, data in batch['features'].items():
                if isinstance(data, list):
                    logger.info(f"    {feature}: List of {len(data)} tensors with shapes {[t.shape for t in data]}")
                else:
                    logger.info(f"    {feature}: Tensor with shape {data.shape}")
            
            batch_count += 1
        logger.info(f"Iterated through {batch_count} batches.")
    else:
        logger.error("Failed to create DataLoader.")

    # Clean up dummy data if desired
    # import shutil
    # shutil.rmtree(dummy_data_dir)
    # logger.info(f"Cleaned up dummy data directory: {dummy_data_dir}")