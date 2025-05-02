import os
import torch
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import logging
from pathlib import Path
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import torch.nn as nn
from dataset import load_numpy_dict
import re # ensure re is imported


# Import our new model and dataset functions
from model import EnhancedTemperatureAwareESMModel, create_model_from_config
from dataset import load_sequences_from_fasta, load_numpy_dict, load_feature_norm_params
from train import log_gpu_memory, get_temperature_scaler
from data_processor import create_instance_key, get_domain_id_from_instance_key, INSTANCE_KEY_SEPARATOR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def load_model_for_prediction(checkpoint_path: str, device: torch.device) -> Tuple[Optional[EnhancedTemperatureAwareESMModel], Optional[Dict]]:
    """Load a trained enhanced model from checkpoint."""
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
         logger.error(f"Failed to load checkpoint file {checkpoint_path}: {e}", exc_info=True)
         return None, None

    required_keys = ['config', 'model_state_dict', 'epoch']
    if not all(key in checkpoint for key in required_keys):
         missing = [k for k in required_keys if k not in checkpoint]
         logger.error(f"Checkpoint {checkpoint_path} is missing required keys: {missing}. Found: {list(checkpoint.keys())}")
         return None, None

    config_from_ckpt = checkpoint['config']
    logger.info("Config loaded from checkpoint.")
    logger.debug(f"Checkpoint Config: {json.dumps(config_from_ckpt, indent=2)}")

    try:
        logger.info(f"Creating model from checkpoint config")
        model = create_model_from_config(config_from_ckpt)
        logger.info("Model instance created.")
    except Exception as e:
         logger.error(f"Error creating model from config: {e}", exc_info=True)
         return None, None

    # Load model state dictionary
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
             logger.warning(f"State dict missing keys: {missing_keys}")
        if unexpected_keys:
             logger.warning(f"State dict has unexpected keys: {unexpected_keys}")

        logger.info(f"Model weights loaded successfully.")
    except Exception as e:
         logger.error(f"Error loading state_dict into model: {e}", exc_info=True)
         return None, None

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded to {device} and set to eval mode.")
    logger.info(f"  Trained for {checkpoint['epoch']+1} epochs.")
    if 'val_corr' in checkpoint:
        logger.info(f"  Best Validation Corr at save time: {checkpoint.get('val_corr', 'N/A'):.6f}")

    return model, config_from_ckpt

def group_sequences_by_length(sequences: Dict[str, str], batch_size: int, bucket_size: int = 50) -> List[List[Tuple[str, str]]]:
    """Groups sequences by length into batches for efficient prediction."""
    if not sequences: return []
    length_buckets = defaultdict(list)
    for seq_id, seq in sequences.items():
        bucket_idx = len(seq) // bucket_size
        length_buckets[bucket_idx].append((seq_id, seq))

    all_batches = []
    # Process shortest first generally helps memory management
    for bucket_idx in sorted(length_buckets.keys()):
        bucket_items = length_buckets[bucket_idx]
        for i in range(0, len(bucket_items), batch_size):
            batch = bucket_items[i : i + batch_size]
            all_batches.append(batch)
    logger.info(f"Grouped {len(sequences)} sequences into {len(all_batches)} batches for prediction.")
    return all_batches

def load_feature_data(
    features_to_load: List[str],
    data_dir: str,
    instance_keys_to_load: List[str], # Renamed for clarity
    file_prefix: str = "predict" # Added prefix argument
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load structural feature data for specific instance keys.

    Args:
        features_to_load: List of feature names to load (e.g., 'normalized_resid').
        data_dir: Directory containing feature files.
        instance_keys_to_load: List of instance keys to load features for.
        file_prefix: Prefix of the feature files (e.g., 'train', 'predict', 'holdout').

    Returns:
        Dictionary mapping instance keys to their feature dictionaries.
    """
    # Initialize dict for all requested keys, even if features aren't found for some
    feature_data = {key: {} for key in instance_keys_to_load}
    loaded_some_feature = False

    for feature in features_to_load:
        # Construct filename using the prefix
        feature_filename = f"{file_prefix}_{feature}.npy"
        feature_path = os.path.join(data_dir, feature_filename)

        if os.path.exists(feature_path):
            try:
                # Load the entire dictionary for this feature
                feature_dict = load_numpy_dict(feature_path)
                found_count = 0
                # Populate the main feature_data dict for the requested keys
                for key in instance_keys_to_load:
                    if key in feature_dict:
                        feature_data[key][feature] = feature_dict[key]
                        found_count += 1
                        loaded_some_feature = True
                if found_count > 0:
                     logger.debug(f"Loaded '{feature}' data for {found_count} requested instances from {feature_filename}")
                # else: logger.debug(f"Feature file {feature_filename} loaded, but contained no data for the requested instance keys.")

            except FileNotFoundError:
                logger.warning(f"Feature file not found during load attempt (should not happen after exists check): {feature_path}")
            except Exception as e:
                logger.warning(f"Error loading or processing feature file {feature_path}: {e}")
        else:
            logger.warning(f"Required feature file not found: {feature_path}")

    # Clean up keys that have no features loaded at all
    if loaded_some_feature:
        # Keep keys only if they have *some* feature data associated
        # This check might be too strict if *some* features are expected missing for certain keys
        # Let's keep all original keys requested, even if empty feature dicts
        # feature_data = {k: v for k, v in feature_data.items() if v}
        pass # Keep all keys in the dict for simplicity downstream
    else:
         logger.warning(f"No features loaded (either none expected or none found matching prefix '{file_prefix}' in {data_dir}).")


    return feature_data
# def load_feature_data(features_to_load: List[str], data_dir: str, domain_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
#     """
#     Load structural feature data for sequences.
    
#     Args:
#         features_to_load: List of feature names to load
#         data_dir: Directory containing feature files
#         domain_ids: List of domain IDs to load features for
        
#     Returns:
#         Dictionary mapping domain IDs to feature dictionaries
#     """
#     feature_data = {domain_id: {} for domain_id in domain_ids}
    
#     for feature in features_to_load:
#         feature_path = os.path.join(data_dir, f"predict_{feature}.npy")
#         if os.path.exists(feature_path):
#             try:
#                 feature_dict = load_numpy_dict(feature_path)
#                 for domain_id in domain_ids:
#                     if domain_id in feature_dict:
#                         feature_data[domain_id][feature] = feature_dict[domain_id]
#                 logger.info(f"Loaded {feature} data for {len(feature_dict)} domains")
#             except Exception as e:
#                 logger.warning(f"Error loading {feature} data: {e}")
#         else:
#             logger.warning(f"Feature file not found: {feature_path}")
    
#     return feature_data


def activate_mc_dropout(model: nn.Module):
    """Activates dropout layers for Monte Carlo Dropout inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            logger.debug(f"Activated MC Dropout for: {module}")

def prepare_prediction_features(
    feature_lookup_keys: List[str], # Renamed from domain_ids for clarity
    model_config: Dict[str, Any],
    scaling_params: Dict[str, Dict[str, float]],
    feature_data: Dict[str, Dict[str, np.ndarray]],
    target_sequence_lengths: Optional[List[int]] = None # Optional: To ensure feature length matches sequence
) -> Optional[Dict[str, Any]]:
    """
    Prepare and normalize features for prediction using specific lookup keys.

    Args:
        feature_lookup_keys: List of keys (expected to be instance_keys) to use for lookup in feature_data.
        model_config: Model configuration dictionary.
        scaling_params: Feature normalization parameters (e.g., from feature_normalization.json).
        feature_data: Dictionary mapping instance_keys to raw feature dictionaries.
        target_sequence_lengths: Optional list of sequence lengths corresponding to feature_lookup_keys,
                                 used to ensure feature arrays match sequence length.

    Returns:
        Dictionary of processed feature tensors ready for the model, or None if features are not used/available.
    """
    # Check if features are enabled in the model config
    if not model_config or not model_config.get('model', {}).get('architecture', {}).get('use_enhanced_features', False):
        logger.debug("Enhanced features not enabled in model config. Skipping feature preparation.")
        return None
    # Check if necessary data structures are provided
    if not feature_data:
        logger.warning("Feature data dictionary not provided, but enhanced features are enabled. Cannot prepare features.")
        return None
    if not scaling_params:
         logger.warning("Feature normalization parameters not provided. Features will be used without normalization.")
         # Continue without normalization if scaling_params is missing

    processed_batch_features = {} # Dictionary to hold prepared feature tensors/lists

    # Determine which features the model expects based on config
    feature_config = model_config.get('data', {}).get('features', {})
    expected_features = []
    if feature_config.get('use_position_info', True): expected_features.append('normalized_resid')
    if feature_config.get('use_structure_info', True): expected_features.extend(['core_exterior_encoded', 'secondary_structure_encoded'])
    if feature_config.get('use_accessibility', True): expected_features.append('relative_accessibility')
    if feature_config.get('use_backbone_angles', True): expected_features.extend(['phi_norm', 'psi_norm'])
    if feature_config.get('use_protein_size', True): expected_features.append('protein_size')
    if feature_config.get('use_voxel_rmsf', True): expected_features.append('voxel_rmsf')
    if feature_config.get('use_bfactor', True): expected_features.append('bfactor_norm')

    # Check which expected features are actually available in feature_data and have normalization params (if provided)
    # Use the first key to check general availability, specific lookups happen in the loop
    available_feature_keys = set()
    if feature_data:
         first_key_data = next(iter(feature_data.values()), {})
         available_feature_keys = set(first_key_data.keys())

    # Process each expected feature
    for feature_name in expected_features:
        if feature_name not in available_feature_keys:
             logger.debug(f"Expected feature '{feature_name}' not found in available feature data keys. Skipping.")
             continue

        is_global = scaling_params.get(feature_name, {}).get('is_global', False) if scaling_params else False
        feature_batch_values = [] # Will hold tensors for this feature across the batch

        for i, lookup_key in enumerate(feature_lookup_keys):
            seq_len = target_sequence_lengths[i] if target_sequence_lengths and i < len(target_sequence_lengths) else None

            # Default value if feature is missing for this specific key or is NaN
            # For per-residue: zero array of appropriate length
            # For global: zero scalar
            default_value = 0.0 if is_global else np.zeros(seq_len if seq_len is not None else 1, dtype=np.float32) # Default length 1 if seq_len unknown

            feature_val = feature_data.get(lookup_key, {}).get(feature_name, default_value)

            # Handle potential NaNs in the loaded feature value
            is_nan = pd.isna(feature_val)
            # Check if *any* value is NaN (for arrays) OR if the scalar itself is NaN
            if (isinstance(is_nan, np.ndarray) and is_nan.any()) or \
            (not isinstance(is_nan, np.ndarray) and is_nan): # Check if scalar is True (meaning NaN)
                logger.warning(f"NaN found for feature '{feature_name}' in instance '{lookup_key}'. Replacing with default ({default_value}).")
                feature_val = default_value # Replace the original value

            # --- Normalization ---
            normalized_val = feature_val # Default to original value
            if scaling_params and feature_name in scaling_params:
                params = scaling_params[feature_name]
                norm_min = params.get('min')
                norm_max = params.get('max')
                if norm_min is not None and norm_max is not None:
                    range_val = norm_max - norm_min
                    if abs(range_val) < 1e-8: # Avoid division by zero if min == max
                         normalized_val = np.zeros_like(feature_val, dtype=np.float32) if isinstance(feature_val, np.ndarray) else 0.0
                    else:
                         # Apply min-max scaling
                         # Ensure feature_val is float for the calculation if it's scalar
                         current_val_float = np.array(feature_val, dtype=np.float32)
                         normalized_val = (current_val_float - norm_min) / range_val
                # else: logger.debug(f"Min/max not found for '{feature_name}' in scaling params. Using raw value.") # Optional debug
            # else: logger.debug(f"No scaling params found for '{feature_name}'. Using raw value.") # Optional debug


            # --- Type Conversion and Length Adjustment ---
            if is_global:
                # Ensure global features are single-element tensors
                tensor_val = torch.tensor([normalized_val], dtype=torch.float32)
            else:
                # Ensure per-residue features are numpy arrays first
                if not isinstance(normalized_val, np.ndarray):
                    normalized_val = np.array([normalized_val], dtype=np.float32) # Make it an array if scalar slipped through

                # Ensure length matches sequence length
                if seq_len is not None and normalized_val.shape[0] != seq_len:
                     logger.warning(f"Adjusting length of feature '{feature_name}' for instance '{lookup_key}'. "
                                    f"Expected {seq_len}, got {normalized_val.shape[0]}.")
                     if normalized_val.shape[0] > seq_len:
                          normalized_val = normalized_val[:seq_len] # Truncate
                     else: # Pad
                          padding = np.zeros(seq_len - normalized_val.shape[0], dtype=np.float32)
                          normalized_val = np.concatenate((normalized_val, padding))

                tensor_val = torch.tensor(normalized_val, dtype=torch.float32)

            feature_batch_values.append(tensor_val)

        # Store the list of tensors (for per-residue) or create a stacked tensor (for global)
        # The model's _process_protein_features expects this structure
        if is_global:
             # Stack global features into a single tensor for the batch [batch_size] (or [batch_size, 1])
             try:
                  processed_batch_features[feature_name] = torch.stack(feature_batch_values).squeeze(-1) # Shape [batch_size]
             except Exception as e:
                  logger.error(f"Error stacking global feature '{feature_name}': {e}. Values: {feature_batch_values}")
                  # Fallback: store as list
                  processed_batch_features[feature_name] = feature_batch_values
        else:
             # Keep per-residue features as a list of tensors
             processed_batch_features[feature_name] = feature_batch_values


    if not processed_batch_features:
         logger.warning("No features were prepared, although enhanced features seem enabled.")
         return None

    return processed_batch_features

# def predict_rmsf_at_temperature(
#     model: EnhancedTemperatureAwareESMModel,
#     sequences: Dict[str, str],
#     target_temperature: float,
#     temp_scaler: Callable[[float], float],
#     batch_size: int,
#     device: torch.device,
#     feature_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
#     feature_norm_params: Optional[Dict[str, Dict[str, float]]] = None,
#     model_config: Optional[Dict[str, Any]] = None,
#     use_amp: bool = True
# ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]: # Return results and id_map
#     """
#     Predict RMSF values for sequences at a specific target temperature. Handles different input ID formats.

#     Args:
#         model: The trained model
#         sequences: Dictionary mapping sequence IDs (from FASTA header) to sequences
#         target_temperature: The single temperature (raw, unscaled) to predict at
#         temp_scaler: The function to scale the raw target temperature
#         batch_size: Batch size for inference
#         device: Device ('cuda' or 'cpu')
#         feature_data: Optional dictionary of structural feature data (keyed by instance_key)
#         feature_norm_params: Optional dictionary of feature normalization parameters
#         model_config: Optional model configuration dictionary from checkpoint
#         use_amp: Whether to use Automatic Mixed Precision (GPU only)

#     Returns:
#         Tuple containing:
#           - results: Dictionary mapping original sequence IDs (from FASTA) to predicted RMSF values (NumPy array)
#           - id_to_instance_key_map: Dictionary mapping original sequence IDs to the corresponding instance_key used for prediction/feature lookup.
#     """
#     model.eval()
#     if not sequences: return {}, {}

#     # Scale the target temperature ONCE
#     scaled_target_temp = temp_scaler(target_temperature)
#     logger.info(f"Predicting for raw temperature {target_temperature:.1f}K (scaled: {scaled_target_temp:.4f})")

#     # Prepare batches based on length
#     batches = group_sequences_by_length(sequences, batch_size)
#     results = {} # Keyed by original fasta ID
#     id_to_instance_key_map = {} # Map original fasta ID to instance_key used
#     prediction_start_time = time.time()
#     autocast_device_type = device.type
#     amp_enabled = (device.type == 'cuda' and use_amp)
#     # Import helper function defined in data_processor
#     from data_processor import create_instance_key, INSTANCE_KEY_SEPARATOR

#     # Check if model uses enhanced features
#     uses_features = model_config and model_config.get('model', {}).get('architecture', {}).get('use_enhanced_features', False)

#     with torch.no_grad():
#         for batch_data in tqdm(batches, desc=f"Predicting @ {target_temperature:.0f}K", leave=False):
#             # batch_ids are the original IDs from the FASTA headers
#             batch_ids = [item[0] for item in batch_data]
#             batch_seqs = [item[1] for item in batch_data]
#             batch_seq_lengths = [len(seq) for seq in batch_seqs] # Get sequence lengths
#             current_batch_size = len(batch_ids)

#             # Create tensor of the same scaled temperature for the whole batch
#             scaled_temps_batch = torch.tensor([scaled_target_temp] * current_batch_size,
#                                               device=device, dtype=torch.float32)

#             # --- Feature Preparation ---
#             batch_features_for_model = None
#             feature_lookup_keys_batch = [] # Keys used to look up features (guaranteed instance_key format)

#             if uses_features:
#                 if not feature_data:
#                      logger.warning("Model expects features, but no feature_data was provided. Predictions will be sequence-only.")
#                 else:
#                     # Determine the correct keys to use for feature lookup
#                     for fasta_id in batch_ids:
#                         # Check if the fasta_id already looks like an instance_key
#                         if INSTANCE_KEY_SEPARATOR in fasta_id:
#                             # Assume it's correct, maybe warn if temp doesn't match target?
#                             try:
#                                 _, temp_part = fasta_id.rsplit(INSTANCE_KEY_SEPARATOR, 1)
#                                 temp_in_key = float(temp_part)
#                                 if abs(temp_in_key - target_temperature) > 1: # Allow some tolerance (e.g., 450.0 vs 450)
#                                      logger.warning(f"FASTA ID '{fasta_id}' looks like an instance key, but its temperature ({temp_in_key:.1f}K) "
#                                                     f"differs significantly from the target prediction temperature ({target_temperature:.1f}K). "
#                                                     f"Using features associated with '{fasta_id}'.")
#                             except ValueError:
#                                  logger.warning(f"Could not parse temperature from FASTA ID '{fasta_id}'. Constructing key using target temperature.")
#                                  fasta_id_base = fasta_id # Use the whole thing if parsing fails
#                                  constructed_key = create_instance_key(fasta_id_base, target_temperature)
#                                  feature_lookup_keys_batch.append(constructed_key)
#                                  id_to_instance_key_map[fasta_id] = constructed_key
#                                  continue # Skip to next id

#                             # Use the original fasta_id as the lookup key
#                             feature_lookup_keys_batch.append(fasta_id)
#                             id_to_instance_key_map[fasta_id] = fasta_id # Map to itself

#                         else:
#                             # Construct the instance key using the target temperature
#                             constructed_key = create_instance_key(fasta_id, target_temperature)
#                             feature_lookup_keys_batch.append(constructed_key)
#                             id_to_instance_key_map[fasta_id] = constructed_key # Map original ID to constructed key

#                     # Prepare features using the determined lookup keys
#                     batch_features_for_model = prepare_prediction_features(
#                         feature_lookup_keys=feature_lookup_keys_batch,
#                         model_config=model_config,
#                         scaling_params=feature_norm_params,
#                         feature_data=feature_data,
#                         target_sequence_lengths=batch_seq_lengths # Pass sequence lengths
#                     )
#             else:
#                  # If not using features, still populate the map (mapping ID to itself, no temp needed)
#                  for fasta_id in batch_ids:
#                       id_to_instance_key_map[fasta_id] = fasta_id # Or maybe construct key anyway for consistency? Let's construct it.
#                       id_to_instance_key_map[fasta_id] = create_instance_key(fasta_id, target_temperature)


#             # --- Model Prediction ---
#             try:
#                 # Forward pass with optional AMP
#                 with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
#                     # Pass sequences, scaled temperatures, and prepared features
#                     batch_predictions_np = model.predict(
#                         sequences=batch_seqs,
#                         scaled_temperatures=scaled_temps_batch,
#                         features=batch_features_for_model # Pass the prepared features
#                     )

#                 # Store results using the *original* FASTA IDs as keys
#                 if len(batch_predictions_np) == len(batch_ids):
#                     for seq_id, pred_np in zip(batch_ids, batch_predictions_np):
#                         results[seq_id] = pred_np
#                 else:
#                     logger.error(f"Prediction output length mismatch: {len(batch_predictions_np)} preds vs {len(batch_ids)} IDs.")
#                     # Handle partial assignment or error as needed
#                     for seq_id in batch_ids:
#                          if seq_id not in results: results[seq_id] = np.array([]) # Ensure all original IDs have an entry

#             except Exception as e:
#                  logger.error(f"Error predicting batch starting with {batch_ids[0]}: {e}", exc_info=True)
#                  # Add placeholder or skip IDs in this batch
#                  for seq_id in batch_ids: results[seq_id] = np.array([]) # Example: empty array on error

#             # Optional: Periodic GPU cache clearing
#             if device.type == 'cuda' and len(results) % (10 * batch_size) == 0:
#                  torch.cuda.empty_cache()

#     prediction_duration = time.time() - prediction_start_time
#     num_predicted = sum(1 for r in results.values() if r.size > 0) # Count successful predictions
#     logger.info(f"Prediction completed for {num_predicted}/{len(sequences)} sequences in {prediction_duration:.2f}s.")
#     if num_predicted > 0: logger.info(f"Avg time per sequence: {prediction_duration / num_predicted:.4f}s")

#     return results, id_to_instance_key_map # Return both results and the map




# Replace the existing function with this one:
def predict_rmsf_at_temperature(
    model: EnhancedTemperatureAwareESMModel,
    sequences: Dict[str, str],
    target_temperature_override: Optional[float], # Changed: Can be None if using npy
    temperatures_from_npy: Optional[Dict[str, float]], # Added: Dict from _temperatures.npy
    temp_scaler: Callable[[float], float],
    batch_size: int,
    device: torch.device,
    feature_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    feature_norm_params: Optional[Dict[str, Dict[str, float]]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    use_amp: bool = True,
    mc_dropout_samples: int = 0 # Added: Number of MC Dropout samples
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]: # Return results, uncertainties, id_map
    """
    Predict RMSF values and uncertainty for sequences.
    Uses temperatures from .npy file if provided, otherwise uses the single override.
    Handles different input ID formats. Estimates uncertainty via MC Dropout if mc_dropout_samples > 1.

    Args:
        model: The trained model.
        sequences: Dictionary mapping sequence IDs (from FASTA header) to sequences.
        target_temperature_override: Single temperature (raw) OR None if using npy file.
        temperatures_from_npy: Optional dict mapping instance_keys to raw temperatures.
        temp_scaler: The function to scale the raw temperature(s).
        batch_size: Batch size for inference.
        device: Device ('cuda' or 'cpu').
        feature_data: Optional dictionary of structural feature data (keyed by instance_key).
        feature_norm_params: Optional dictionary of feature normalization parameters.
        model_config: Optional model configuration dictionary from checkpoint.
        use_amp: Whether to use Automatic Mixed Precision (GPU only).
        mc_dropout_samples: Number of forward passes for MC Dropout uncertainty estimation. If <= 1, performs standard prediction.

    Returns:
        Tuple containing:
          - results: Dictionary mapping original sequence IDs (from FASTA) to predicted RMSF mean values (NumPy array).
          - uncertainties: Dictionary mapping original sequence IDs to predicted RMSF uncertainty (std dev) (NumPy array). Returns zero array if mc_dropout_samples <= 1.
          - id_to_instance_key_map: Dictionary mapping original sequence IDs to the corresponding instance_key used for prediction/feature lookup.
    """
    if not sequences: return {}, {}, {}

    if temperatures_from_npy is None and target_temperature_override is None:
        raise ValueError("Must provide either target_temperature_override or temperatures_from_npy.")
    if temperatures_from_npy is not None and target_temperature_override is not None:
        logger.warning("Both target_temperature_override and temperatures_from_npy provided. Using temperatures_from_npy.")
        target_temperature_override = None # Prioritize npy file

    # Determine if we are doing MC Dropout
    do_mc_dropout = mc_dropout_samples > 1
    if do_mc_dropout:
        logger.info(f"Performing prediction with MC Dropout using {mc_dropout_samples} samples.")
        # Activate dropout layers specifically
        model.eval() # Start in eval mode
        activate_mc_dropout(model) # Turn *only* dropout layers back to train mode
    else:
        logger.info("Performing standard prediction (MC Dropout disabled).")
        model.eval() # Ensure model is in standard eval mode

    # Scale the single override temperature ONCE if needed
    scaled_target_temp_override = None
    if target_temperature_override is not None:
        scaled_target_temp_override = temp_scaler(target_temperature_override)
        logger.info(f"Using OVERRIDE raw temperature {target_temperature_override:.1f}K (scaled: {scaled_target_temp_override:.4f}) for all sequences.")
    else:
        logger.info("Using instance-specific temperatures loaded from .npy file.")

    # Prepare batches based on length
    batches = group_sequences_by_length(sequences, batch_size)
    results_mean = {} # Keyed by original fasta ID -> Mean prediction
    results_uncertainty = {} # Keyed by original fasta ID -> Std Dev prediction
    id_to_instance_key_map = {} # Map original fasta ID to instance_key used
    prediction_start_time = time.time()
    autocast_device_type = device.type
    amp_enabled = (device.type == 'cuda' and use_amp)

    # Check if model uses enhanced features
    uses_features = model_config and model_config.get('model', {}).get('architecture', {}).get('use_enhanced_features', False)

    with torch.no_grad(): # Still disable gradient calculation overall
        for batch_data in tqdm(batches, desc=f"Predicting", leave=False):
            batch_ids = [item[0] for item in batch_data]
            batch_seqs = [item[1] for item in batch_data]
            batch_seq_lengths = [len(seq) for seq in batch_seqs] # Get sequence lengths
            current_batch_size = len(batch_ids)

            # --- Determine temperatures and instance keys for the batch ---
            # We do this once per batch, regardless of MC samples
            batch_raw_temps_list = []
            batch_scaled_temps_list = []
            batch_feature_lookup_keys = []
            valid_indices_in_batch = [] # Indices within the current batch that are valid

            for i, fasta_id in enumerate(batch_ids):
                raw_temp_for_instance = None
                instance_key_for_lookup = fasta_id # Default assumption

                if temperatures_from_npy is not None:
                    if fasta_id not in temperatures_from_npy:
                         logger.error(f"FASTA ID / Instance Key '{fasta_id}' not found in provided temperatures .npy file. Skipping sequence.")
                         results_mean[fasta_id] = np.array([], dtype=np.float32) # Add placeholder
                         results_uncertainty[fasta_id] = np.array([], dtype=np.float32)
                         id_to_instance_key_map[fasta_id] = fasta_id # Map anyway
                         continue # Skip processing this sequence in the batch
                    raw_temp_for_instance = temperatures_from_npy[fasta_id]
                    instance_key_for_lookup = fasta_id # Key is already correct

                elif target_temperature_override is not None:
                    raw_temp_for_instance = target_temperature_override
                    if INSTANCE_KEY_SEPARATOR not in fasta_id:
                        instance_key_for_lookup = create_instance_key(fasta_id, raw_temp_for_instance)
                    else: # Already looks like a key, check temp consistency
                         try:
                             _, temp_part = fasta_id.rsplit(INSTANCE_KEY_SEPARATOR, 1)
                             temp_in_key = float(temp_part)
                             if abs(temp_in_key - raw_temp_for_instance) > 1:
                                 logger.warning(f"FASTA ID '{fasta_id}' temp ({temp_in_key:.1f}K) differs from override ({raw_temp_for_instance:.1f}K). Using override.")
                         except ValueError: logger.warning(f"Could not parse temp from FASTA ID '{fasta_id}'.")
                         instance_key_for_lookup = fasta_id # Use original key anyway

                else: # Should not happen
                     logger.error(f"Internal logic error: No temperature source for {fasta_id}.")
                     results_mean[fasta_id] = np.array([], dtype=np.float32); results_uncertainty[fasta_id] = np.array([], dtype=np.float32); id_to_instance_key_map[fasta_id] = fasta_id
                     continue

                # Store valid data for this item
                batch_raw_temps_list.append(raw_temp_for_instance)
                batch_scaled_temps_list.append(temp_scaler(raw_temp_for_instance))
                batch_feature_lookup_keys.append(instance_key_for_lookup)
                id_to_instance_key_map[fasta_id] = instance_key_for_lookup # Store mapping
                valid_indices_in_batch.append(i) # Store the original index within this batch

            # Skip batch if no valid sequences remained
            if not valid_indices_in_batch: continue

            # Prepare tensors and sequences only for the valid items
            valid_batch_seqs = [batch_seqs[i] for i in valid_indices_in_batch]
            valid_batch_ids = [batch_ids[i] for i in valid_indices_in_batch]
            valid_batch_seq_lengths = [batch_seq_lengths[i] for i in valid_indices_in_batch]
            scaled_temps_tensor = torch.tensor(batch_scaled_temps_list, device=device, dtype=torch.float32)

            # --- Feature Preparation (for valid items) ---
            batch_features_for_model = None
            if uses_features:
                if not feature_data: logger.warning("Model expects features, but no feature_data was provided.")
                else:
                    batch_features_for_model = prepare_prediction_features(
                        feature_lookup_keys=batch_feature_lookup_keys, # Keys for valid items
                        model_config=model_config,
                        scaling_params=feature_norm_params,
                        feature_data=feature_data,
                        target_sequence_lengths=valid_batch_seq_lengths # Lengths for valid items
                    )

            # --- Model Prediction (potentially multiple passes for MC Dropout) ---
            all_mc_preds_np = [] # List to store predictions from each MC sample run

            num_passes = mc_dropout_samples if do_mc_dropout else 1
            for mc_run in range(num_passes):
                try:
                    with torch.amp.autocast(device_type=autocast_device_type, enabled=amp_enabled):
                        # We use model.forward directly here to get raw tensor outputs easily
                        # Ensure inputs match model.forward signature
                        outputs_dict = model.forward(
                            sequences=valid_batch_seqs,
                            temperatures=scaled_temps_tensor,
                            features=batch_features_for_model,
                            target_rmsf_values=None # No targets needed for prediction
                        )
                        # Extract list of prediction tensors
                        batch_predictions_tensors = outputs_dict['predictions']

                    # Convert current pass predictions to numpy and store
                    current_pass_preds_np = []
                    for pred_tensor in batch_predictions_tensors:
                         if pred_tensor is not None and pred_tensor.numel() > 0:
                              current_pass_preds_np.append(pred_tensor.cpu().numpy())
                         else:
                              current_pass_preds_np.append(np.array([], dtype=np.float32)) # Handle errors
                    all_mc_preds_np.append(current_pass_preds_np)

                except Exception as e:
                    mc_status = f"MC run {mc_run+1}/{num_passes}" if do_mc_dropout else "Standard prediction"
                    logger.error(f"Error during {mc_status} for batch starting with {valid_batch_ids[0]}: {e}", exc_info=True)
                    # If one MC pass fails, we likely can't continue for this batch
                    all_mc_preds_np = [] # Clear any partial results for this batch
                    for seq_id in valid_batch_ids: # Mark predictions as failed
                        results_mean[seq_id] = np.array([])
                        results_uncertainty[seq_id] = np.array([])
                    break # Break MC loop for this batch

            # --- Aggregate MC Dropout results ---
            if all_mc_preds_np: # Check if prediction runs were successful
                # Stack predictions along a new dimension (samples, batch_item, seq_len)
                # Need to handle potentially different lengths if sequences weren't bucketed perfectly
                # For simplicity, assume sequences in a batch have same length (due to bucketing) or handle padding/unpadding
                # Let's assume length consistency within MC samples for a given sequence

                # Iterate through each item in the batch
                for i, original_fasta_id in enumerate(valid_batch_ids):
                     preds_for_item = [mc_pass_results[i] for mc_pass_results in all_mc_preds_np if i < len(mc_pass_results) and mc_pass_results[i].size > 0]

                     if not preds_for_item: # Handle case where all MC passes failed for this item
                          results_mean[original_fasta_id] = np.array([])
                          results_uncertainty[original_fasta_id] = np.array([])
                          continue

                     # Stack the predictions for this item
                     try:
                          stacked_preds = np.stack(preds_for_item, axis=0) # Shape (num_samples, seq_len)
                          mean_pred = np.mean(stacked_preds, axis=0)
                          std_dev_pred = np.std(stacked_preds, axis=0) if do_mc_dropout else np.zeros_like(mean_pred) # Uncertainty is 0 if not MC dropout

                          results_mean[original_fasta_id] = mean_pred.astype(np.float32)
                          results_uncertainty[original_fasta_id] = std_dev_pred.astype(np.float32)
                     except ValueError as e: # Handle potential stacking errors if lengths mismatch unexpectedly
                          logger.error(f"Error stacking predictions for {original_fasta_id} (lengths={[p.shape for p in preds_for_item]}): {e}")
                          results_mean[original_fasta_id] = np.array([])
                          results_uncertainty[original_fasta_id] = np.array([])


            # Optional: Periodic GPU cache clearing
            if device.type == 'cuda' and len(results_mean) % (10 * batch_size) == 0:
                 torch.cuda.empty_cache()

    # Ensure model is back in standard eval mode if MC Dropout was used
    if do_mc_dropout: model.eval()

    prediction_duration = time.time() - prediction_start_time
    num_predicted = sum(1 for r in results_mean.values() if r.size > 0) # Count successful predictions
    logger.info(f"Prediction completed for {num_predicted}/{len(sequences)} sequences in {prediction_duration:.2f}s.")
    if num_predicted > 0: logger.info(f"Avg time per sequence: {prediction_duration / num_predicted:.4f}s")

    return results_mean, results_uncertainty, id_to_instance_key_map # Return all dicts

def plot_rmsf(
    sequence: str,
    predictions: np.ndarray,
    title: str,
    output_path: str,
    window_size: int = 1,
    figsize: Tuple[int, int] = (15, 6)
):
    """Plot predicted RMSF values against residue position."""
    
    return 
    # if predictions is None or len(predictions) == 0:
    #     logger.warning(f"No prediction data to plot for '{title}'. Skipping plot.")
    #     return

    # plt.style.use('seaborn-v0_8-whitegrid')
    # fig, ax = plt.subplots(figsize=figsize)

    # pred_len = len(predictions)
    # residue_indices = np.arange(1, pred_len + 1)

    # if window_size > 1:
    #     s = pd.Series(predictions)
    #     plot_data = s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    #     plot_label = f'RMSF Prediction (Smoothed, win={window_size})'
    # else:
    #     plot_data = predictions
    #     plot_label = 'RMSF Prediction'

    # ax.plot(residue_indices, plot_data, '-', color='dodgerblue', linewidth=1.5, label=plot_label)

    # ax.set_xlabel('Residue Position')
    # ax.set_ylabel('Predicted RMSF')
    # ax.set_title(f'Predicted RMSF for {title} (Length: {pred_len})') # Title now includes Temp
    # ax.set_xlim(0, pred_len + 1)
    # ax.grid(True, linestyle=':', alpha=0.7)

    # # Add stats text box
    # mean_rmsf = np.mean(predictions)
    # median_rmsf = np.median(predictions)
    # stats_text = (f'Mean: {mean_rmsf:.3f}\n'
    #               f'Median: {median_rmsf:.3f}')
    # ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
    #         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.5))

    # ax.legend(loc='upper right')
    # plt.tight_layout()

    # try:
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     plt.savefig(output_path, dpi=100, bbox_inches='tight') # Lower DPI for potentially many plots
    # except Exception as e:
    #     logger.error(f"Failed to save plot to {output_path}: {e}")
    # finally:
    #     plt.close(fig)


def save_predictions(
    predictions: Dict[str, np.ndarray],
    uncertainties: Dict[str, np.ndarray], # Added uncertainties
    id_to_instance_key_map: Dict[str, str],
    output_path: str,
    # target_temperature: float # Removed - temp info is in instance_key now
    ):
    """Save predictions and uncertainties to a CSV file, using instance_key."""
    if not predictions:
        logger.warning("No predictions provided to save.")
        return

    data_to_save = []
    saved_count = 0
    missing_key_map_count = 0
    missing_uncertainty_count = 0

    # Iterate through predictions dictionary (keyed by original fasta ID)
    for fasta_id, rmsf_values in predictions.items():
        if rmsf_values is None or len(rmsf_values) == 0:
             logger.debug(f"Skipping save for '{fasta_id}' due to empty prediction.")
             continue

        # Get the corresponding instance_key from the map
        instance_key = id_to_instance_key_map.get(fasta_id)
        if instance_key is None:
             logger.warning(f"Could not find instance_key mapping for FASTA ID '{fasta_id}'. Skipping save for this sequence.")
             missing_key_map_count += 1
             continue

        # Get the corresponding uncertainty array
        uncertainty_values = uncertainties.get(fasta_id)
        if uncertainty_values is None or len(uncertainty_values) != len(rmsf_values):
            logger.warning(f"Uncertainty data missing or length mismatch for '{fasta_id}'. Saving uncertainty as NaN.")
            missing_uncertainty_count += 1
            uncertainty_values = np.full_like(rmsf_values, np.nan) # Fill with NaN

        # Get raw temperature from instance key for reporting (optional, but good)
        try:
            raw_temp = float(instance_key.split(INSTANCE_KEY_SEPARATOR)[-1])
        except:
            raw_temp = np.nan # Fallback if key format is weird

        # Append data for each residue, using the instance_key
        for i, rmsf in enumerate(rmsf_values):
            uncertainty = uncertainty_values[i] if i < len(uncertainty_values) else np.nan
            data_to_save.append({
                # Use the instance_key for the 'domain_id' column for compatibility downstream
                'instance_key': instance_key, # Changed column name for clarity
                'resid': i + 1,
                'rmsf_pred': rmsf,
                'uncertainty': uncertainty, # Added uncertainty column
                'temperature': raw_temp # Added original temperature column
            })
        saved_count += 1

    if missing_key_map_count > 0:
         logger.error(f"Failed to find instance_key mapping for {missing_key_map_count} FASTA IDs during saving.")
    if missing_uncertainty_count > 0:
         logger.warning(f"Uncertainty missing or mismatched for {missing_uncertainty_count} sequences.")

    if not data_to_save:
        logger.warning("No valid prediction data points found to save in CSV.")
        return

    try:
        df = pd.DataFrame(data_to_save)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Format floats nicely
        df.to_csv(output_path, index=False, float_format='%.6f')
        logger.info(f"Predictions for {saved_count} sequences saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions DataFrame to {output_path}: {e}")
        
# def save_predictions(
#     predictions: Dict[str, np.ndarray],
#     id_to_instance_key_map: Dict[str, str], # Added map parameter
#     output_path: str,
#     target_temperature: float):
#     """Save predictions to a CSV file, ensuring instance_key is used."""
#     if not predictions:
#         logger.warning("No predictions provided to save.")
#         return

#     data_to_save = []
#     saved_count = 0
#     missing_key_map_count = 0

#     # Iterate through predictions dictionary (keyed by original fasta ID)
#     for fasta_id, rmsf_values in predictions.items():
#         if rmsf_values is None or len(rmsf_values) == 0:
#              logger.debug(f"Skipping save for '{fasta_id}' due to empty prediction.")
#              continue

#         # Get the corresponding instance_key from the map
#         instance_key = id_to_instance_key_map.get(fasta_id)
#         if instance_key is None:
#              logger.warning(f"Could not find instance_key mapping for FASTA ID '{fasta_id}'. Skipping save for this sequence.")
#              missing_key_map_count += 1
#              continue

#         # Append data for each residue, using the instance_key
#         for i, rmsf in enumerate(rmsf_values):
#             data_to_save.append({
#                 # Use the instance_key for the 'domain_id' column for compatibility downstream
#                 'domain_id': instance_key,
#                 'resid': i + 1,
#                 'rmsf_pred': rmsf,
#                 'predicted_at_temp': target_temperature
#             })
#         saved_count += 1

#     if missing_key_map_count > 0:
#          logger.error(f"Failed to find instance_key mapping for {missing_key_map_count} FASTA IDs during saving.")

#     if not data_to_save:
#         logger.warning("No valid prediction data points found to save in CSV.")
#         return

#     try:
#         df = pd.DataFrame(data_to_save)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         df.to_csv(output_path, index=False, float_format='%.6f')
#         logger.info(f"Predictions for {saved_count} sequences (T={target_temperature:.0f}K) saved to {output_path}")
#     except Exception as e:
#         logger.error(f"Failed to save predictions DataFrame to {output_path}: {e}")
        
# def predict(config: Dict[str, Any]):
#     """Main prediction function."""
#     predict_start_time = time.time()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")
#     if device.type == 'cuda': log_gpu_memory()

#     # --- Get Required Config ---
#     model_checkpoint = config.get('model_checkpoint')
#     fasta_path = config.get('fasta_path')
#     output_dir = config.get('output_dir', 'predictions')
#     target_temperature = config.get('temperature') # Raw temperature

#     if not model_checkpoint or not fasta_path or target_temperature is None:
#         logger.critical("Missing required config: 'model_checkpoint', 'fasta_path', or 'temperature'.")
#         return

#     # --- Output Dir & Logging ---
#     # Include temperature in output subdir for organization
#     try:
#          temp_str = f"{target_temperature:.0f}K"
#     except TypeError: # Handle if temperature is somehow not a number
#          logger.error(f"Invalid target temperature provided: {target_temperature}. Must be numeric.")
#          return
#     output_dir_temp = os.path.join(output_dir, temp_str)
#     os.makedirs(output_dir_temp, exist_ok=True)

#     log_path = os.path.join(output_dir_temp, 'prediction.log')
#     # Remove existing handlers to avoid duplicate logging if run multiple times
#     for handler in logger.handlers[:]:
#         if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
#             logger.removeHandler(handler)
#             handler.close()
#     # Add new file handler for this run
#     file_handler = logging.FileHandler(log_path, mode='w')
#     file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'))
#     logger.addHandler(file_handler)
#     logger.info(f"--- Starting Prediction Run for T={target_temperature:.1f}K ---")
#     logger.info(f"Prediction Config: {json.dumps(config, indent=2)}")
#     logger.info(f"Saving results to: {output_dir_temp}")

#     # --- Load Model ---
#     model, model_config_from_ckpt = load_model_for_prediction(model_checkpoint, device)
#     if model is None:
#         logger.error("Failed to load model. Aborting prediction.")
#         logger.removeHandler(file_handler); file_handler.close()
#         return
#     if device.type == 'cuda': log_gpu_memory()

#     # --- Load Temperature Scaler ---
#     checkpoint_dir = os.path.dirname(model_checkpoint)
#     # Try finding scaling/norm params relative to checkpoint first, then from config model_dir
#     potential_param_dirs = [checkpoint_dir]
#     config_model_dir = model_config_from_ckpt.get('output',{}).get('model_dir')
#     if config_model_dir and os.path.abspath(config_model_dir) != os.path.abspath(checkpoint_dir):
#          potential_param_dirs.append(config_model_dir)

#     temp_scaler = None
#     scaling_filename = model_config_from_ckpt.get('data', {}).get('temp_scaling_filename', 'temp_scaling_params.json')
#     for p_dir in potential_param_dirs:
#         temp_scaling_path = os.path.join(p_dir, scaling_filename)
#         if os.path.exists(temp_scaling_path):
#             logger.info(f"Found temperature scaling file: {temp_scaling_path}")
#             try:
#                 temp_scaler = get_temperature_scaler(temp_scaling_path)
#                 break # Found and loaded successfully
#             except Exception as e:
#                  logger.error(f"Failed to load temperature scaler from {temp_scaling_path}: {e}. Aborting.")
#                  logger.removeHandler(file_handler); file_handler.close()
#                  return
#     if temp_scaler is None:
#          logger.error(f"Temperature scaling file '{scaling_filename}' not found in potential directories: {potential_param_dirs}. Aborting.")
#          logger.removeHandler(file_handler); file_handler.close()
#          return

#     # --- Load Feature Normalization Parameters ---
#     feature_norm_params = None
#     if model_config_from_ckpt.get('model', {}).get('architecture', {}).get('use_enhanced_features', True):
#         norm_params_filename = model_config_from_ckpt.get('data', {}).get('features', {}).get('normalization_params_file', 'feature_normalization.json')
#         found_norm_params = False
#         for p_dir in potential_param_dirs:
#             norm_params_path = os.path.join(p_dir, norm_params_filename)
#             if os.path.exists(norm_params_path):
#                 logger.info(f"Found feature normalization file: {norm_params_path}")
#                 try:
#                     feature_norm_params = load_feature_norm_params(norm_params_path)
#                     if feature_norm_params:
#                          logger.info(f"Loaded normalization parameters for {len(feature_norm_params)} features")
#                          found_norm_params = True
#                          break # Found and loaded
#                     else:
#                          logger.warning(f"Loaded feature normalization file {norm_params_path}, but it was empty.")
#                 except Exception as e:
#                     logger.warning(f"Error loading feature normalization parameters from {norm_params_path}: {e}. Continuing without normalization.")
#         if not found_norm_params:
#              logger.warning(f"Feature normalization file '{norm_params_filename}' not found or empty in potential directories: {potential_param_dirs}. Feature normalization will not be applied.")

#     # --- Load Sequences ---
#     try:
#         sequences = load_sequences_from_fasta(fasta_path)
#         if not sequences: raise ValueError("No sequences found in FASTA file.")
#         logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
#     except Exception as e:
#          logger.critical(f"Error loading sequences from {fasta_path}: {e}", exc_info=True)
#          logger.removeHandler(file_handler); file_handler.close()
#          return

#     # --- Filter Sequences by Max Length (Optional) ---
#     max_length = config.get('max_length')
#     if max_length is not None and max_length > 0:
#         original_count = len(sequences)
#         sequences = {sid: seq for sid, seq in sequences.items() if len(seq) <= max_length}
#         filtered_count = len(sequences)
#         if filtered_count < original_count:
#             logger.info(f"Filtered out {original_count - filtered_count} sequences longer than {max_length}.")
#         if not sequences:
#             logger.critical(f"No sequences remaining after filtering by max_length={max_length}. Aborting.")
#             logger.removeHandler(file_handler); file_handler.close()
#             return

#     # --- Load Structural Features if model uses them ---
#     feature_data = None
#     if model_config_from_ckpt.get('model', {}).get('architecture', {}).get('use_enhanced_features', True):
#         logger.info("Model uses enhanced features. Attempting to load feature data...")

#         # Features should have been saved during processing with 'predict_' prefix
#         # We need the same list of features the model expects
#         expected_features = []
#         feature_config = model_config_from_ckpt.get('data', {}).get('features', {})
#         if feature_config.get('use_position_info', True): expected_features.append('normalized_resid')
#         if feature_config.get('use_structure_info', True): expected_features.extend(['core_exterior_encoded', 'secondary_structure_encoded'])
#         if feature_config.get('use_accessibility', True): expected_features.append('relative_accessibility')
#         if feature_config.get('use_backbone_angles', True): expected_features.extend(['phi_norm', 'psi_norm'])
#         if feature_config.get('use_protein_size', True): expected_features.append('protein_size')
#         if feature_config.get('use_voxel_rmsf', True): expected_features.append('voxel_rmsf')
#         if feature_config.get('use_bfactor', True): expected_features.append('bfactor_norm')

#         # Load features if expected. Assume they are in the same dir as the FASTA by default.
#         # Or should we load from data/processed? Assume same dir as fasta for now.
#         # --> Using data_dir from config might be more robust if 'concatenate_fastas' was used.
#         # Let's use the directory containing the FASTA file.
#         fasta_dir = os.path.dirname(fasta_path)
#         logger.info(f"Looking for feature .npy files (e.g., predict_*.npy) in: {fasta_dir}")

#         # We need all instance keys to potentially load features for them
#         # Construct potential instance keys from the input fasta sequences
#         from data_processor import create_instance_key, INSTANCE_KEY_SEPARATOR
#         potential_instance_keys = []
#         for fasta_id in sequences.keys():
#              if INSTANCE_KEY_SEPARATOR in fasta_id:
#                   potential_instance_keys.append(fasta_id)
#              else:
#                   potential_instance_keys.append(create_instance_key(fasta_id, target_temperature))


#         if expected_features:
#             # This function needs modification if predict_*.npy files don't exist.
#             # Let's simplify: Assume predict_*.npy files were created by concatenate_fastas
#             # and are in data_dir (data/processed by default)
#             processed_data_dir = model_config_from_ckpt.get('data', {}).get('data_dir', 'data/processed')
#             logger.info(f"Loading pre-concatenated features (predict_*.npy) from: {processed_data_dir}")
#             feature_data = load_feature_data(expected_features, processed_data_dir, potential_instance_keys)

#             if feature_data:
#                  loaded_feature_count = sum(len(f_dict) > 0 for f_dict in feature_data.values())
#                  logger.info(f"Loaded feature data for {loaded_feature_count} potential instance keys.")
#                  if loaded_feature_count == 0:
#                       logger.warning("No feature data loaded, although features are expected. Check predict_*.npy files.")
#                       feature_data = None # Treat as if no data was loaded
#             else:
#                 logger.warning(f"Failed to load any feature data from {processed_data_dir}. Check predict_*.npy files.")
#         else:
#              logger.info("No specific features expected by model config. Skipping feature loading.")

#     # --- Predict RMSF ---
#     # Returns dict keyed by fasta_id, and a map from fasta_id to instance_key
#     predictions, id_to_instance_key_map = predict_rmsf_at_temperature(
#         model=model,
#         sequences=sequences,
#         target_temperature=target_temperature,
#         temp_scaler=temp_scaler,
#         batch_size=config.get('batch_size', 8),
#         device=device,
#         feature_data=feature_data, # Pass loaded features
#         feature_norm_params=feature_norm_params, # Pass norm params
#         model_config=model_config_from_ckpt, # Pass model config
#         use_amp=(device.type == 'cuda')
#     )

#     # --- Save & Plot Results ---
#     if predictions:
#          output_csv_path = os.path.join(output_dir_temp, f'predictions_{temp_str}.csv')
#          # Pass the map to save_predictions
#          save_predictions(
#              predictions=predictions,
#              id_to_instance_key_map=id_to_instance_key_map,
#              output_path=output_csv_path,
#              target_temperature=target_temperature
#          )

#          if config.get('plot_predictions', True):
#              plots_dir = os.path.join(output_dir_temp, 'plots')
#              os.makedirs(plots_dir, exist_ok=True)
#              smoothing = config.get('smoothing_window', 1)
#              logger.info(f"Generating plots (smoothing={smoothing})...")
#              plot_count = 0
#              # Iterate using original fasta IDs
#              for fasta_id, pred_array in tqdm(predictions.items(), desc="Plotting", leave=False):
#                   if fasta_id in sequences and pred_array.size > 0 :
#                       try:
#                           # Add temperature to plot title
#                           plot_title = f"{fasta_id} @ {target_temperature:.0f}K"
#                           # Sanitize fasta_id for filename
#                           safe_fasta_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in fasta_id)
#                           plot_filename = f'{safe_fasta_id}_{temp_str}.png'

#                           plot_rmsf(
#                               sequence=sequences[fasta_id],
#                               predictions=pred_array,
#                               title=plot_title,
#                               output_path=os.path.join(plots_dir, plot_filename),
#                               window_size=smoothing
#                           )
#                           plot_count += 1
#                       except Exception as e:
#                           logger.error(f"Failed to generate plot for {fasta_id}: {e}")
#                   elif pred_array.size == 0:
#                        logger.debug(f"Skipping plot for {fasta_id} - no prediction data.")
#                   # else: logger.warning(f"Cannot plot for {fasta_id}: Original sequence not found.") # Should not happen if keys match
#              logger.info(f"Generated {plot_count} plots.")
#     else:
#          logger.warning("Prediction resulted in no output.")

#     # --- Finalize ---
#     predict_end_time = time.time()
#     logger.info(f"--- Prediction Run Finished (T={target_temperature:.1f}K) ---")
#     logger.info(f"Total prediction time: {predict_end_time - predict_start_time:.2f} seconds.")
#     logger.info(f"Results saved in: {output_dir_temp}")
#     # Remove the file handler specific to this run
#     logger.removeHandler(file_handler)
#     file_handler.close()




# Replace the existing function with this one:
def predict(config: Dict[str, Any]):
    """Main prediction function."""
    predict_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda': log_gpu_memory()

    # --- Get Required Config ---
    model_checkpoint = config.get('model_checkpoint')
    fasta_path = config.get('fasta_path')
    output_dir = config.get('output_dir', 'predictions')
    target_temperature_override = config.get('temperature') # May be None if using npy
    temperature_npy_path = config.get('temperature_npy') # New optional arg
    mc_samples = config.get('mc_samples', 0) # New uncertainty arg

    # --- Input Validation ---
    if not model_checkpoint or not os.path.exists(model_checkpoint):
        logger.critical(f"Model checkpoint not found or not specified: {model_checkpoint}")
        return
    if not fasta_path or not os.path.exists(fasta_path):
        logger.critical(f"Input FASTA file not found or not specified: {fasta_path}")
        return
    if target_temperature_override is None and (temperature_npy_path is None or not os.path.exists(temperature_npy_path)):
         logger.critical("Must provide either --temperature OR a valid path via --temperature_npy.")
         return
    if target_temperature_override is not None and temperature_npy_path is not None:
         logger.warning("Both --temperature and --temperature_npy provided. Prioritizing --temperature_npy.")
         target_temperature_override = None # Disable override

    # --- Output Dir & Logging ---
    run_name = f"prediction_run_{time.strftime('%Y%m%d_%H%M%S')}"
    if target_temperature_override is not None:
         try: run_name = f"prediction_{target_temperature_override:.0f}K"
         except TypeError: pass
    elif temperature_npy_path is not None:
         run_name = f"prediction_from_npy_{os.path.splitext(os.path.basename(temperature_npy_path))[0]}"
    if mc_samples > 1: run_name += f"_mc{mc_samples}" # Add MC suffix

    output_dir_run = os.path.join(output_dir, run_name)
    os.makedirs(output_dir_run, exist_ok=True)

    log_path = os.path.join(output_dir_run, 'prediction.log')
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
            logger.removeHandler(handler); handler.close()
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"--- Starting Prediction Run ---")
    logger.info(f"Using FASTA: {fasta_path}")
    if temperature_npy_path: logger.info(f"Using Temperatures from: {temperature_npy_path}")
    if target_temperature_override: logger.info(f"Using Override Temperature: {target_temperature_override:.1f}K")
    if mc_samples > 1: logger.info(f"Using MC Dropout Samples: {mc_samples}")
    logger.info(f"Prediction Config (Args): {json.dumps(config, indent=2)}")
    logger.info(f"Saving results to: {output_dir_run}")

    # --- Load Model ---
    model, model_config_from_ckpt = load_model_for_prediction(model_checkpoint, device)
    if model is None: logger.error("Failed to load model."); logger.removeHandler(file_handler); file_handler.close(); return
    if device.type == 'cuda': log_gpu_memory()

    # --- Load Temperature Scaler (MUST use scaler from original training) ---
    checkpoint_dir = os.path.dirname(model_checkpoint)
    potential_param_dirs = [checkpoint_dir]
    config_model_dir = model_config_from_ckpt.get('output',{}).get('model_dir')
    if config_model_dir and os.path.abspath(config_model_dir) != os.path.abspath(checkpoint_dir):
         potential_param_dirs.append(config_model_dir)
    temp_scaler = None
    scaling_filename = model_config_from_ckpt.get('data', {}).get('temp_scaling_filename', 'temp_scaling_params.json')
    logger.info(f"Attempting to load temperature scaler params '{scaling_filename}' used during training...")
    for p_dir in potential_param_dirs:
        temp_scaling_path = os.path.join(p_dir, scaling_filename)
        if os.path.exists(temp_scaling_path):
            logger.info(f"Found temperature scaling file: {temp_scaling_path}")
            try: temp_scaler = get_temperature_scaler(temp_scaling_path); break
            except Exception as e: logger.error(f"Failed to load temp scaler from {temp_scaling_path}: {e}. Aborting."); logger.removeHandler(file_handler); file_handler.close(); return
    if temp_scaler is None: logger.error(f"Temp scaling file '{scaling_filename}' not found in {potential_param_dirs}. Aborting."); logger.removeHandler(file_handler); file_handler.close(); return

    # --- Load Feature Normalization Parameters (MUST use params from original training) ---
    feature_norm_params = None
    if model_config_from_ckpt.get('model', {}).get('architecture', {}).get('use_enhanced_features', True):
        norm_params_filename = model_config_from_ckpt.get('data', {}).get('features', {}).get('normalization_params_file', 'feature_normalization.json')
        logger.info(f"Attempting to load feature norm params '{norm_params_filename}' used during training...")
        found_norm_params = False
        for p_dir in potential_param_dirs:
            norm_params_path = os.path.join(p_dir, norm_params_filename)
            if os.path.exists(norm_params_path):
                logger.info(f"Found feature normalization file: {norm_params_path}")
                try:
                    feature_norm_params = load_feature_norm_params(norm_params_path)
                    if feature_norm_params: logger.info(f"Loaded normalization params for {len(feature_norm_params)} features"); found_norm_params = True; break
                    else: logger.warning(f"Loaded feature normalization file {norm_params_path}, but it was empty.")
                except Exception as e: logger.warning(f"Error loading feature normalization parameters from {norm_params_path}: {e}. Continuing without.")
        if not found_norm_params: logger.warning(f"Feature norm file '{norm_params_filename}' not found or empty in {potential_param_dirs}. Features will not be normalized.")

    # --- Load Sequences ---
    try:
        sequences = load_sequences_from_fasta(fasta_path)
        if not sequences: raise ValueError("No sequences found in FASTA file.")
        logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except Exception as e: logger.critical(f"Error loading sequences from {fasta_path}: {e}", exc_info=True); logger.removeHandler(file_handler); file_handler.close(); return

    # --- Load Temperatures from NPY if path provided ---
    temperatures_from_npy_dict = None
    if temperature_npy_path:
        try:
            temperatures_from_npy_dict = load_numpy_dict(temperature_npy_path)
            logger.info(f"Loaded {len(temperatures_from_npy_dict)} temperatures from {temperature_npy_path}")
        except Exception as e: logger.error(f"Failed to load temperatures from {temperature_npy_path}: {e}. Aborting."); logger.removeHandler(file_handler); file_handler.close(); return

    # --- Filter Sequences by Max Length (Optional) ---
    max_length = config.get('max_length')
    if max_length is not None and max_length > 0:
        original_count = len(sequences)
        sequences = {sid: seq for sid, seq in sequences.items() if len(seq) <= max_length}
        filtered_count = len(sequences)
        if filtered_count < original_count: logger.info(f"Filtered out {original_count - filtered_count} sequences longer than {max_length}.")
        if not sequences: logger.critical(f"No sequences remaining after filtering by max_length={max_length}. Aborting."); logger.removeHandler(file_handler); file_handler.close(); return

    # --- Load Structural Features if model uses them ---
    feature_data = None
    if model_config_from_ckpt.get('model', {}).get('architecture', {}).get('use_enhanced_features', True):
        logger.info("Model uses enhanced features. Attempting to load feature data...")
        expected_features = []
        feature_config = model_config_from_ckpt.get('data', {}).get('features', {})
        if feature_config.get('use_position_info', True): expected_features.append('normalized_resid')
        if feature_config.get('use_structure_info', True): expected_features.extend(['core_exterior_encoded', 'secondary_structure_encoded'])
        if feature_config.get('use_accessibility', True): expected_features.append('relative_accessibility')
        if feature_config.get('use_backbone_angles', True): expected_features.extend(['phi_norm', 'psi_norm'])
        if feature_config.get('use_protein_size', True): expected_features.append('protein_size')
        if feature_config.get('use_voxel_rmsf', True): expected_features.append('voxel_rmsf')
        if feature_config.get('use_bfactor', True): expected_features.append('bfactor_norm')

        # Determine feature file prefix based on fasta filename
        fasta_filename = os.path.basename(fasta_path)
        match = re.match(r"^(train|val|test|predict|holdout)_sequences\.fasta$", fasta_filename)
        feature_file_prefix = match.group(1) if match else "predict"
        feature_data_dir = os.path.dirname(fasta_path)
        logger.info(f"Inferred feature prefix '{feature_file_prefix}' from FASTA.")
        logger.info(f"Looking for feature files ({feature_file_prefix}_*.npy) in: {feature_data_dir}")

        # Construct potential lookup keys
        potential_lookup_keys = []
        for fasta_id in sequences.keys():
             if temperatures_from_npy_dict is not None:
                  potential_lookup_keys.append(fasta_id) # FASTA ID must be instance key
             elif target_temperature_override is not None:
                   potential_lookup_keys.append(create_instance_key(fasta_id, target_temperature_override) if INSTANCE_KEY_SEPARATOR not in fasta_id else fasta_id)

        if expected_features:
            feature_data = load_feature_data(expected_features, feature_data_dir, potential_lookup_keys, file_prefix=feature_file_prefix)
            if feature_data:
                 loaded_feature_count = sum(len(f_dict) > 0 for f_dict in feature_data.values())
                 logger.info(f"Loaded feature data for {loaded_feature_count} potential instance keys.")
                 if loaded_feature_count == 0: logger.warning(f"No feature data loaded using prefix '{feature_file_prefix}'."); feature_data = None
            else: logger.warning(f"Failed to load any feature data from {feature_data_dir} with prefix '{feature_file_prefix}'.")
        else: logger.info("No specific features expected. Skipping feature loading.")

    # --- Predict RMSF ---
    # Returns dicts keyed by fasta_id: mean_preds, uncertainties, and map from fasta_id to instance_key
    predictions, uncertainties, id_to_instance_key_map = predict_rmsf_at_temperature(
        model=model,
        sequences=sequences,
        target_temperature_override=target_temperature_override,
        temperatures_from_npy=temperatures_from_npy_dict,
        temp_scaler=temp_scaler,
        batch_size=config.get('batch_size', 8),
        device=device,
        feature_data=feature_data,
        feature_norm_params=feature_norm_params,
        model_config=model_config_from_ckpt,
        use_amp=(device.type == 'cuda'),
        mc_dropout_samples=mc_samples # Pass MC samples arg
    )

    # --- Save & Plot Results ---
    if predictions:
         output_csv_path = os.path.join(output_dir_run, f'predictions_{run_name}.csv')
         # Pass uncertainties to save_predictions
         save_predictions(
             predictions=predictions,
             uncertainties=uncertainties, # Pass uncertainty dict
             id_to_instance_key_map=id_to_instance_key_map,
             output_path=output_csv_path
             # No need to pass single temp anymore
         )

         if config.get('plot_predictions', True):
             plots_dir = os.path.join(output_dir_run, 'plots')
             os.makedirs(plots_dir, exist_ok=True)
             smoothing = config.get('smoothing_window', 1)
             logger.info(f"Generating plots (smoothing={smoothing})...")
             plot_count = 0
             for fasta_id, pred_array in tqdm(predictions.items(), desc="Plotting", leave=False):
                  if fasta_id in sequences and pred_array.size > 0 :
                      try:
                          instance_key = id_to_instance_key_map.get(fasta_id)
                          if instance_key and temperatures_from_npy_dict:
                               plot_temp = temperatures_from_npy_dict.get(instance_key, target_temperature_override if target_temperature_override else 0)
                          elif target_temperature_override is not None: plot_temp = target_temperature_override
                          else: plot_temp = 0

                          plot_title = f"{fasta_id} @ {plot_temp:.0f}K"
                          safe_fasta_id = "".join(c if c.isalnum() or c in ('-', '_', '@', '.') else '_' for c in fasta_id)
                          plot_filename = f'{safe_fasta_id}_T{plot_temp:.0f}.png'

                          plot_rmsf(
                              sequence=sequences[fasta_id],
                              predictions=pred_array,
                              title=plot_title,
                              output_path=os.path.join(plots_dir, plot_filename),
                              window_size=smoothing
                              # TODO: Optionally modify plot_rmsf to show uncertainty bands?
                          )
                          plot_count += 1
                      except Exception as e: logger.error(f"Failed to generate plot for {fasta_id}: {e}")
                  elif pred_array.size == 0: logger.debug(f"Skipping plot for {fasta_id} - no prediction data.")
             logger.info(f"Generated {plot_count} plots.")
    else: logger.warning("Prediction resulted in no output.")

    # --- Finalize ---
    predict_end_time = time.time()
    logger.info(f"--- Prediction Run Finished ---")
    logger.info(f"Total prediction time: {predict_end_time - predict_start_time:.2f} seconds.")
    logger.info(f"Results saved in: {output_dir_run}")
    logger.removeHandler(file_handler)
    file_handler.close()
    
    

# --- Also need to update the command-line parser in predict.py ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RMSF using a trained enhanced model, optionally using per-instance temperatures and MC Dropout uncertainty.')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--fasta_path', type=str, required=True, help='Path to the input FASTA file (headers must be instance_keys if using --temperature_npy)')
    # Make temperature optional, add npy path
    parser.add_argument('--temperature', type=float, default=None, help='(Optional) Target temperature (Kelvin) to use for ALL sequences if --temperature_npy is not provided.')
    parser.add_argument('--temperature_npy', type=str, default=None, help='(Optional) Path to .npy file mapping instance_keys (from FASTA) to RAW temperatures. Overrides --temperature.')
    parser.add_argument('--mc_samples', type=int, default=0, help='Number of Monte Carlo Dropout samples for uncertainty estimation (e.g., 10-50). Default 0 disables MC Dropout.')
    # --- End changes ---
    parser.add_argument('--output_dir', type=str, default='predictions', help='Base directory to save prediction results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for prediction')
    parser.add_argument('--max_length', type=int, default=None, help='Optional: Max sequence length filter')
    parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate plots')
    parser.add_argument('--smoothing_window', type=int, default=1, help='Smoothing window for plots (1=none)')

    # --- Placeholder for direct script execution (for testing) ---
    # Remove or adapt this if predict() is only called via main.py
    args = parser.parse_args()
    config_dict = vars(args)
    predict(config_dict)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Predict RMSF using a trained enhanced model')
#     parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file)')
#     parser.add_argument('--fasta_path', type=str, required=True, help='Path to the input FASTA file')
#     parser.add_argument('--temperature', type=float, required=True, help='Target temperature (in Kelvin) for prediction')
#     parser.add_argument('--output_dir', type=str, default='predictions', help='Base directory to save prediction results')
#     parser.add_argument('--batch_size', type=int, default=8, help='Batch size for prediction')
#     parser.add_argument('--max_length', type=int, default=None, help='Optional: Max sequence length filter')
#     parser.add_argument('--plot_predictions', action=argparse.BooleanOptionalAction, default=True, help='Generate plots')
#     parser.add_argument('--smoothing_window', type=int, default=1, help='Smoothing window for plots (1=none)')

#     args = parser.parse_args()
#     config_dict = vars(args)
#     predict(config_dict)