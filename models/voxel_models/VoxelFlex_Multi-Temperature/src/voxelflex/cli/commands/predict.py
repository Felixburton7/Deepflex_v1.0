"""
Prediction command for VoxelFlex (Temperature-Aware).

Loads a trained model and predicts RMSF for specified domains at a given temperature.
Uses on-demand loading of raw voxel data.
"""
import os
import time
import json
import logging
import gc
import h5py  # Added missing import
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.predict")

# Project imports
from voxelflex.data.data_loader import (
    PredictionDataset,
    load_list_from_file
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import EnhancedProgressBar, log_memory_usage, log_stage, log_section_header
from voxelflex.utils.file_utils import ensure_dir, resolve_path, save_json, load_json 
from voxelflex.utils.system_utils import get_device, clear_memory, check_memory_usage
from voxelflex.utils.temp_scaling import get_temperature_scaler

def predict_rmsf(
    config: Dict[str, Any],
    model_path: str,
    target_temperature: float,
    domain_ids_to_predict: Optional[List[str]] = None,
    output_csv_filename: Optional[str] = None # Allow specifying output filename
 ) -> Optional[str]:
    """
    Runs prediction for specified domains at a target temperature.

    Args:
        config: Configuration dictionary.
        model_path: Path to the trained model checkpoint (.pt).
        target_temperature: The temperature (in K) for which to predict RMSF.
        domain_ids_to_predict: Optional list of specific HDF5 domain keys to predict.
                                If None, uses domains from the test split file defined in config.
        output_csv_filename: Optional base filename for the output CSV.

    Returns:
        Path to the saved predictions CSV file, or None on failure.
    """
    run_output_dir = config["output"]["run_dir"]
    metrics_dir = config["output"]["metrics_dir"]
    ensure_dir(metrics_dir)

    # Determine output filename
    if output_csv_filename is None:
        # Generate default name if not provided
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_str = f"{target_temperature:.0f}K"
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_csv_filename = f"predictions_{model_name}_{temp_str}_{timestamp}.csv"

    # Construct full path
    predictions_path = os.path.join(metrics_dir, output_csv_filename)

    log_section_header(logger, f"STARTING PREDICTION at {target_temperature:.1f}K")
    logger.info(f"Model: {model_path}")
    logger.info(f"Output CSV: {predictions_path}")
    log_memory_usage(logger)
    device = get_device(config["system_utilization"].get("adjust_for_gpu", True))
    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))

    model = None # Ensure model is defined for finally block
    try:
        # --- Load Model & Temperature Scaler ---
        with log_stage("SETUP", "Loading model and temperature scaler"):
            try:
                logger.info(f"Loading checkpoint from: {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu') # Load to CPU first

                # Extract necessary info from checkpoint
                model_cfg_from_ckpt = checkpoint.get('config', {}).get('model', {})
                input_shape = checkpoint.get('input_shape')
                scaling_params_from_ckpt = checkpoint.get('temp_scaling_params')
                train_cfg_from_ckpt = checkpoint.get('config', {}).get('training', {}) # For mixed precision setting

                if not model_cfg_from_ckpt: 
                    raise ValueError("Model config section missing from checkpoint.")
                if not input_shape: 
                    raise ValueError("Model input_shape missing from checkpoint.")
                if not scaling_params_from_ckpt:
                     logger.warning("Temperature scaling parameters missing from checkpoint. Attempting to load from config path...")
                     # Try loading from the path specified in the *current* config, assuming it matches the run
                     scaler_path_from_config = config["data"]["temp_scaling_params_file"]
                     if os.path.exists(scaler_path_from_config):
                         scaling_params_from_ckpt = load_json(scaler_path_from_config)
                         logger.info(f"Loaded scaling params from config path: {scaler_path_from_config}")
                     else:
                          raise ValueError("Temperature scaling parameters missing from checkpoint and config path.")

                # Create model instance
                model = get_model(model_cfg_from_ckpt, input_shape=input_shape)
                # Load weights
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if missing_keys: 
                    logger.warning(f"Prediction Load Warning: Missing keys in model state_dict: {missing_keys}")
                if unexpected_keys: 
                    logger.warning(f"Prediction Load Warning: Unexpected keys in model state_dict: {unexpected_keys}")

                model.to(device)
                model.eval() # Set to evaluation mode
                logger.info(f"Loaded model '{model_cfg_from_ckpt.get('architecture')}' to {device}.")

                # Get temperature scaler function
                temp_scaler_func = get_temperature_scaler(params=scaling_params_from_ckpt)
                scaled_target_temp_value = temp_scaler_func(target_temperature)
                logger.info(f"Target temp {target_temperature}K scaled to: {scaled_target_temp_value:.4f}")
                # Create tensor for prediction - can be expanded per batch later
                scaled_target_temp_tensor_base = torch.tensor([[scaled_target_temp_value]], dtype=torch.float32).to(device)

                del checkpoint
                gc.collect() # Free checkpoint memory

            except Exception as e:
                logger.exception(f"Error loading model or temperature scaler: {e}")
                return None

        # --- Prepare Domain List ---
        with log_stage("SETUP", "Preparing domain list for prediction"):
            if domain_ids_to_predict is None:
                logger.info("No specific domains provided. Using domains from test split file.")
                test_split_file = config['input'].get('test_split_file')
                if not test_split_file or not os.path.exists(test_split_file):
                    logger.error(f"Test split file not found or specified ({test_split_file}). Cannot determine domains to predict.")
                    return None
                domain_ids_to_predict = load_list_from_file(test_split_file)
                if not domain_ids_to_predict:
                     logger.error(f"Test split file ({test_split_file}) is empty. No domains to predict.")
                     return None
                logger.info(f"Loaded {len(domain_ids_to_predict)} domains from test split file.")

            # Ensure the list contains unique domains
            domain_ids_to_predict = sorted(list(set(domain_ids_to_predict)))
            logger.info(f"Predicting for {len(domain_ids_to_predict)} unique domains.")

        # --- Identify Samples to Predict (Domain + Residue from HDF5) ---
        with log_stage("SETUP", "Identifying target residues from HDF5"):
            samples_to_predict: List[Tuple[str, str]] = []
            try:
                 with h5py.File(config["input"]["voxel_file"], 'r') as f_h5:
                      for domain_id in domain_ids_to_predict:
                           if domain_id not in f_h5:
                                logger.warning(f"Requested prediction domain '{domain_id}' not found in HDF5. Skipping.")
                                continue
                           # Find residue group (same logic as preprocess/data_loader)
                           domain_group = f_h5[domain_id]
                           residue_group = None
                           potential_chain_keys = sorted([k for k in domain_group.keys() if isinstance(domain_group[k], h5py.Group)])
                           for chain_key in potential_chain_keys:
                                try:
                                     potential_residue_group = domain_group[chain_key]
                                     if any(key.isdigit() for key in potential_residue_group.keys()):
                                          residue_group = potential_residue_group
                                          break
                                except Exception:
                                    continue

                           if residue_group:
                                for resid_str in residue_group.keys():
                                     if resid_str.isdigit():
                                          samples_to_predict.append((domain_id, resid_str))
                           else:
                                logger.warning(f"No residue group found for domain '{domain_id}'. Skipping.")
            except Exception as e:
                 logger.exception(f"Error reading HDF5 to identify target residues: {e}")
                 return None

            if not samples_to_predict:
                 logger.error("No valid residues found in HDF5 for the specified domains. Cannot run prediction.")
                 return None
            logger.info(f"Identified {len(samples_to_predict)} target residues from HDF5.")

        # --- Create Dataset & DataLoader ---
        # PredictionDataset now handles pre-loading/filtering based on voxel availability
        with log_stage("SETUP", "Creating Prediction Dataset and DataLoader"):
            try:
                pred_dataset = PredictionDataset(
                    samples_to_load=samples_to_predict,
                    voxel_hdf5_path=config["input"]["voxel_file"],
                    expected_channels=model_cfg_from_ckpt['input_channels'], # Use channels from loaded model config
                    target_shape_chw=input_shape # Validate against loaded model shape
                )

                if len(pred_dataset) == 0:
                    logger.error("Prediction dataset is empty after attempting to load voxels. Cannot proceed.")
                    return None

                # Use configured prediction batch size
                pred_batch_size = config.get('prediction', {}).get('batch_size', 128)
                pred_loader = DataLoader(
                    pred_dataset,
                    batch_size=pred_batch_size,
                    shuffle=False, # Important for consistent output order if needed
                    num_workers=0, # Keep prediction simple, load in main thread
                    pin_memory=False # Not typically needed for CPU->GPU transfer here
                )
                logger.info(f"Prediction DataLoader created: {len(pred_loader)} batches, BatchSize={pred_batch_size}.")

            except Exception as e:
                logger.exception(f"Error creating prediction dataset/loader: {e}")
                return None

        # --- Load Optional Resname Information (for output CSV) ---
        resname_lookup: Dict[Tuple[str, int], str] = {}
        try:
             with log_stage("SETUP", "Loading residue name information (optional)"):
                  from voxelflex.data.data_loader import load_aggregated_rmsf_data, create_domain_mapping
                  rmsf_df_for_names = load_aggregated_rmsf_data(config['input']['aggregated_rmsf_file'])
                  # We only need domain_id, resid, resname
                  resname_df = rmsf_df_for_names[['domain_id', 'resid', 'resname']].copy()
                  resname_df['resid'] = pd.to_numeric(resname_df['resid'], errors='coerce').astype('Int64')
                  resname_df.dropna(inplace=True)
                  resname_df.drop_duplicates(subset=['domain_id', 'resid'], keep='first', inplace=True)
                  resname_map = resname_df.set_index(['domain_id', 'resid'])['resname'].to_dict()

                  # Create lookup using HDF5 key -> RMSF mapping if needed (or assume direct match?)
                  # Let's assume prediction uses HDF5 keys, so we need mapping
                  hdf5_keys_in_dataset = list(set(s[0] for s in pred_dataset.samples))
                  domain_mapping_pred = create_domain_mapping(hdf5_keys_in_dataset, resname_df['domain_id'].unique().tolist())

                  # Populate the lookup for samples that are actually in the dataset
                  for domain_id, resid_str in pred_dataset.samples:
                       try:
                            resid_int = int(resid_str)
                            mapped_rmsf_id = domain_mapping_pred.get(domain_id, domain_id) # Fallback to original ID
                            resname = resname_map.get((mapped_rmsf_id, resid_int))
                            if resname is None: # Try base name if mapped failed
                                 base_rmsf_id = mapped_rmsf_id.split('_')[0]
                                 if base_rmsf_id != mapped_rmsf_id:
                                      resname = resname_map.get((base_rmsf_id, resid_int))

                            resname_lookup[(domain_id, resid_int)] = resname if resname else "UNK"
                       except ValueError: 
                           continue # Skip if resid_str is not int

                  logger.info(f"Residue name lookup created with {len(resname_lookup)} entries.")
                  del rmsf_df_for_names, resname_df, resname_map, domain_mapping_pred
                  gc.collect()
        except Exception as e:
             logger.warning(f"Could not load residue names from RMSF file: {e}. Resnames will be 'UNK'.")

        # --- Run Prediction Loop ---
        with log_stage("PREDICTION", "Running inference loop"):
            logger.info("Starting prediction loop...")
            results_list = []
            progress = EnhancedProgressBar(len(pred_loader), desc=f"Predict {target_temperature:.0f}K")
            # Use mixed precision if enabled during training
            autocast_enabled = train_cfg_from_ckpt.get('mixed_precision', {}).get('enabled', False) and device.type == 'cuda'

            with torch.no_grad(): # Ensure no gradients are calculated
                for i, batch_data in enumerate(pred_loader):
                    # batch_data is (list_of_domain_ids, list_of_resid_strs, voxel_tensor)
                    if not batch_data or len(batch_data) != 3:
                        logger.warning(f"Skipping invalid batch data at index {i}.")
                        progress.update(1)
                        continue

                    batch_domains, batch_resids_str, batch_voxels = batch_data
                    # Check if batch is effectively empty
                    if not batch_domains or batch_voxels.numel() == 0:
                        logger.warning(f"Skipping empty batch {i+1}.")
                        progress.update(1)
                        continue

                    try:
                        batch_voxels = batch_voxels.to(device, non_blocking=True)
                        current_batch_size = batch_voxels.size(0)
                        # Expand the base scaled temperature tensor for the current batch size
                        batch_scaled_temps = scaled_target_temp_tensor_base.expand(current_batch_size, 1)

                        # Run model inference
                        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                             batch_outputs = model(voxel_input=batch_voxels, scaled_temp=batch_scaled_temps)

                        # Process outputs
                        preds_np = batch_outputs.detach().cpu().numpy().flatten()

                        # Collect results for this batch
                        for j in range(current_batch_size):
                             domain_id = batch_domains[j]
                             resid_str = batch_resids_str[j]
                             try:
                                  resid_int = int(resid_str)
                                  resname = resname_lookup.get((domain_id, resid_int), "UNK") # Get resname
                                  results_list.append({
                                       'domain_id': domain_id, # Use HDF5 key as identifier
                                       'resid': resid_int,
                                       'resname': resname,
                                       'predicted_rmsf': preds_np[j],
                                       'prediction_temperature': target_temperature
                                  })
                             except ValueError:
                                  logger.warning(f"Invalid residue ID format encountered in prediction output: {domain_id}:{resid_str}")
                             except IndexError:
                                  logger.warning(f"Index error accessing prediction output for batch {i+1}, item {j}")

                    except Exception as e:
                         logger.exception(f"Error predicting batch {i+1}: {e}")
                         # Continue to next batch if one fails

                    progress.update(1)
                    del batch_voxels, batch_scaled_temps, batch_outputs, preds_np # Clean up batch tensors
                    if i % 50 == 0: 
                        clear_memory(force_gc=False, clear_cuda=(device.type=='cuda'))

            progress.finish()
            logger.info(f"Prediction loop finished. Collected {len(results_list)} results.")

        # --- Save Results ---
        with log_stage("OUTPUT", "Saving prediction results"):
            if results_list:
                try:
                    results_df = pd.DataFrame(results_list)
                    # Ensure correct order of columns
                    output_cols = ['domain_id', 'resid', 'resname', 'prediction_temperature', 'predicted_rmsf']
                    results_df = results_df[output_cols]
                    results_df.to_csv(predictions_path, index=False, float_format='%.6f')
                    logger.info(f"Predictions saved successfully to: {predictions_path}")
                except Exception as e:
                    logger.exception(f"Failed to save predictions CSV to {predictions_path}: {e}")
                    return None # Indicate failure
            else:
                logger.warning("No results generated during prediction loop. Output file will not be created.")
                return None # Indicate no results

        logger.info("Prediction command finished successfully.")
        return predictions_path # Return path on success

    finally:
        # Cleanup
        del model # Ensure model is deleted
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
        log_memory_usage(logger)