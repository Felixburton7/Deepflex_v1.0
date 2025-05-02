# # FILE: src/voxelflex/cli/commands/predict.py


# FILE: src/voxelflex/cli/commands/predict.py
"""
Prediction command for Voxelflex (Optimized for Speed).

This version uses a single Dataset/DataLoader for all predictions,
loading voxel data on demand within the DataLoader workers.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
import psutil
from torch.utils.data import DataLoader

# Ensure these imports point to the correct modules
from voxelflex.data.data_loader import (
    load_rmsf_data,
    create_domain_mapping,
    create_optimized_rmsf_lookup,
    check_memory_usage,
    clear_memory,
    PredictionRMSFDataset  # Import the new Dataset
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import (
    get_logger,
    EnhancedProgressBar,
    log_memory_usage,
)
from voxelflex.utils.file_utils import ensure_dir, resolve_path # Added resolve_path
from voxelflex.utils.system_utils import (
    get_device,
    set_num_threads,
    is_memory_critical,
    adjust_workers_for_memory,
    emergency_memory_reduction,
    MEMORY_WARNING_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_EMERGENCY_THRESHOLD
)


logger = get_logger(__name__)


def generate_prediction_samples(
    voxel_file_path: str,
    domains_to_predict: List[str],
    rmsf_lookup: Dict[Tuple[str, int], float],
    domain_mapping: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str]], Optional[Tuple[int, ...]]]:
    """
    Scans HDF5 and RMSF data to generate the list of (domain, resid) samples
    for prediction without loading voxel data. Also determines input shape.
    """
    logger.info("Generating list of samples for prediction...")
    samples = []
    input_shape = None
    skipped_residues = 0
    h5_child_key_cache = {} # Cache child key per domain

    # Use a progress bar for potentially long scanning process
    progress = EnhancedProgressBar(len(domains_to_predict), prefix="Scanning Domains", suffix="Complete")

    try:
        with h5py.File(voxel_file_path, 'r') as f:
            for i, domain_id in enumerate(domains_to_predict):
                if domain_id not in f:
                    logger.warning(f"Domain {domain_id} listed for prediction not found in HDF5 file.")
                    progress.update(i + 1)
                    continue

                domain_group = f[domain_id]
                if not domain_group: continue

                # Find child key (e.g., 'PDB') - Reuse logic from dataset
                child_key = None
                if domain_id in h5_child_key_cache:
                     child_key = h5_child_key_cache[domain_id]
                else:
                    for key in domain_group.keys():
                        if isinstance(domain_group[key], h5py.Group):
                             sub_keys = list(domain_group[key].keys())
                             if any(k.isdigit() for k in sub_keys):
                                  h5_child_key_cache[domain_id] = key
                                  child_key = key
                                  break
                    if child_key is None:
                         h5_child_key_cache[domain_id] = None # Cache failure

                if child_key is None:
                    logger.warning(f"Could not find residue group key for domain {domain_id}.")
                    progress.update(i + 1)
                    continue

                residue_group = domain_group[child_key]
                mapped_domain = domain_mapping.get(domain_id)
                if mapped_domain is None:
                     # Should not happen if mapping is complete, but check anyway
                     logger.warning(f"No RMSF mapping found for voxel domain {domain_id}, skipping its residues.")
                     progress.update(i + 1)
                     continue

                base_domain = mapped_domain.split('_')[0] # For fallback lookup

                for resid_str in residue_group.keys():
                    if not resid_str.isdigit(): continue # Ensure it's a residue key

                    try:
                        resid_int = int(resid_str)
                        # Check if RMSF data exists for this residue
                        lookup_key = (mapped_domain, resid_int)
                        base_lookup_key = (base_domain, resid_int)

                        if lookup_key in rmsf_lookup or (base_domain != mapped_domain and base_lookup_key in rmsf_lookup):
                            samples.append((domain_id, resid_str))

                            # Determine input shape from the first valid sample
                            if input_shape is None:
                                try:
                                    residue_dataset = residue_group[resid_str]
                                    if isinstance(residue_dataset, h5py.Dataset):
                                        shape_raw = residue_dataset.shape
                                        if len(shape_raw) == 4 and shape_raw[3] in [4, 5]:
                                            input_shape = (shape_raw[3], shape_raw[0], shape_raw[1], shape_raw[2])
                                        else:
                                             input_shape = shape_raw
                                        logger.info(f"Determined input shape from {domain_id}/{resid_str}: {input_shape}")
                                except Exception as shape_e:
                                     logger.warning(f"Could not get shape for {domain_id}/{resid_str}: {shape_e}")
                        else:
                            skipped_residues += 1
                    except ValueError:
                         logger.warning(f"Invalid residue ID format {resid_str} in {domain_id}")
                         skipped_residues += 1
                    except Exception as e:
                         logger.error(f"Error processing key {resid_str} in {domain_id}: {e}")
                         skipped_residues += 1

                progress.update(i + 1) # Update progress after each domain
        progress.finish()

    except Exception as e:
        logger.error(f"Failed during sample generation: {e}")
        raise

    logger.info(f"Generated {len(samples)} samples for prediction.")
    if skipped_residues > 0:
        logger.warning(f"Skipped {skipped_residues} residues during sample generation (likely no matching RMSF).")
    if input_shape is None:
         logger.warning("Could not determine input shape during sample scanning. Using default.")
         input_shape = (5, 24, 24, 24) # Fallback default shape

    return samples, input_shape


# FILE: src/voxelflex/cli/commands/predict.py
"""
Prediction command for Voxelflex (Optimized for Speed).

This version uses a single Dataset/DataLoader for all predictions,
loading voxel data on demand within the DataLoader workers.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
import psutil
from torch.utils.data import DataLoader
import gc # Import gc

# Ensure these imports point to the correct modules
from voxelflex.data.data_loader import (
    load_rmsf_data,
    create_domain_mapping,
    create_optimized_rmsf_lookup,
    check_memory_usage,
    clear_memory,
    PredictionRMSFDataset  # Import the new Dataset
)
from voxelflex.models.cnn_models import get_model
from voxelflex.utils.logging_utils import (
    get_logger,
    EnhancedProgressBar,
    log_memory_usage,
)
from voxelflex.utils.file_utils import ensure_dir, resolve_path # Added resolve_path
from voxelflex.utils.system_utils import (
    get_device,
    set_num_threads,
    is_memory_critical,
    adjust_workers_for_memory,
    emergency_memory_reduction,
    MEMORY_WARNING_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_EMERGENCY_THRESHOLD
)


logger = get_logger(__name__)


def generate_prediction_samples(
    voxel_file_path: str,
    domains_to_predict: List[str],
    rmsf_lookup: Dict[Tuple[str, int], float],
    domain_mapping: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str]], Optional[Tuple[int, ...]]]:
    """
    Scans HDF5 and RMSF data to generate the list of (domain, resid) samples
    for prediction without loading voxel data. Also determines input shape.
    """
    logger.info("Generating list of samples for prediction...")
    samples = []
    input_shape = None
    skipped_residues = 0
    h5_child_key_cache = {} # Cache child key per domain

    # Use a progress bar for potentially long scanning process
    progress = EnhancedProgressBar(len(domains_to_predict), prefix="Scanning Domains", suffix="Complete")

    try:
        # Increase HDF5 buffer size
        with h5py.File(voxel_file_path, 'r', rdcc_nbytes=1024*1024*16) as f:
            for i, domain_id in enumerate(domains_to_predict):
                if domain_id not in f:
                    logger.warning(f"Domain {domain_id} listed for prediction not found in HDF5 file.")
                    progress.update(i + 1)
                    continue

                domain_group = f[domain_id]
                if not domain_group or not isinstance(domain_group, h5py.Group): # Check if it's a group
                    logger.warning(f"HDF5 entry for {domain_id} is not a Group.")
                    progress.update(i + 1)
                    continue

                # Find child key (e.g., 'PDB') - Reuse logic from dataset
                child_key = None
                if domain_id in h5_child_key_cache:
                     child_key = h5_child_key_cache[domain_id]
                else:
                    for key in domain_group.keys():
                        item = domain_group.get(key) # Use .get for safety
                        if item and isinstance(item, h5py.Group):
                             sub_keys = list(item.keys())
                             if any(k.isdigit() for k in sub_keys):
                                  h5_child_key_cache[domain_id] = key
                                  child_key = key
                                  break
                    if child_key is None:
                         h5_child_key_cache[domain_id] = None # Cache failure

                if child_key is None or child_key not in domain_group: # Check if key exists
                    logger.warning(f"Could not find valid residue group key for domain {domain_id}.")
                    progress.update(i + 1)
                    continue

                residue_group = domain_group[child_key]
                if not isinstance(residue_group, h5py.Group): # Check if residue group is valid
                     logger.warning(f"Entry {child_key} within {domain_id} is not a Group.")
                     progress.update(i+1)
                     continue

                mapped_domain = domain_mapping.get(domain_id)
                if mapped_domain is None:
                     logger.warning(f"No RMSF mapping found for voxel domain {domain_id}, skipping its residues.")
                     progress.update(i + 1)
                     continue

                base_domain = mapped_domain.split('_')[0] # For fallback lookup

                for resid_str in residue_group.keys():
                    if not resid_str.isdigit(): continue # Ensure it's a residue key

                    try:
                        resid_int = int(resid_str)
                        # Check if RMSF data exists for this residue
                        lookup_key = (mapped_domain, resid_int)
                        base_lookup_key = (base_domain, resid_int)

                        if lookup_key in rmsf_lookup or (base_domain != mapped_domain and base_lookup_key in rmsf_lookup):
                            samples.append((domain_id, resid_str))

                            # Determine input shape from the first valid sample's dataset
                            if input_shape is None:
                                try:
                                    residue_dataset = residue_group[resid_str]
                                    if isinstance(residue_dataset, h5py.Dataset):
                                        shape_raw = residue_dataset.shape
                                        if len(shape_raw) == 4 and shape_raw[3] in [4, 5]:
                                            input_shape = (shape_raw[3], shape_raw[0], shape_raw[1], shape_raw[2])
                                        else:
                                             input_shape = shape_raw
                                        logger.info(f"Determined input shape from {domain_id}/{resid_str}: {input_shape}")
                                except Exception as shape_e:
                                     logger.warning(f"Could not get shape for {domain_id}/{resid_str}: {shape_e}")
                        else:
                            skipped_residues += 1
                    except ValueError:
                         logger.warning(f"Invalid residue ID format {resid_str} in {domain_id}")
                         skipped_residues += 1
                    except Exception as e:
                         logger.error(f"Error processing key {resid_str} in {domain_id}: {e}")
                         skipped_residues += 1

                progress.update(i + 1) # Update progress after each domain
        progress.finish()

    except Exception as e:
        logger.error(f"Failed during sample generation: {e}", exc_info=True) # Log traceback
        raise

    logger.info(f"Generated {len(samples)} samples for prediction.")
    if skipped_residues > 0:
        logger.warning(f"Skipped {skipped_residues} residues during sample generation (likely no matching RMSF).")
    if input_shape is None:
         logger.error("Could not determine input shape during sample scanning. Prediction might fail.")
         # If you know the shape, hardcode it as a fallback
         # input_shape = (5, 24, 24, 24) # Example fallback
         raise ValueError("Failed to determine input shape for the model.")

    return samples, input_shape


def predict_rmsf(
    config: Dict[str, Any],
    model_path: str,
    domain_ids: Optional[List[str]] = None # Allow specifying domains via arg
) -> str:
    """
    Make predictions using a single Dataset/DataLoader approach.
    """
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZED PREDICTION PROCESS (Single Dataset)")
    logger.info("=" * 60)
    logger.info(f"Model checkpoint: {model_path}")

    # --- 1. Initial Setup & Memory Checks ---
    memory_stats = check_memory_usage()
    logger.info(f"Initial memory usage: System: {memory_stats['system_percent']:.1f}%, "
                f"Process: {memory_stats['process_rss_gb']:.2f} GB")
    if memory_stats['system_percent'] > MEMORY_WARNING_THRESHOLD * 100:
        logger.warning("High initial memory usage. Attempting reduction.")
        emergency_memory_reduction()
        memory_stats = check_memory_usage()
        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
            raise MemoryError(f"Cannot start prediction with critical memory usage: {memory_stats['system_percent']:.1f}%")

    device = get_device(config["system_utilization"]["adjust_for_gpu"])
    logger.info(f"Using device: {device}")
    clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))

    # --- 2. Load Model ---
    logger.info(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('config', {}).get('model', config['model'])
        ckpt_input_shape = checkpoint.get('input_shape') # Get shape from checkpoint if available

        model = get_model(
            architecture=model_config['architecture'],
            input_channels=model_config['input_channels'],
            channel_growth_rate=model_config['channel_growth_rate'],
            num_residual_blocks=model_config['num_residual_blocks'],
            dropout_rate=model_config['dropout_rate'],
            base_filters=model_config['base_filters']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        processed_domains_during_training = checkpoint.get('processed_domains', [])
        logger.info(f"Model originally trained on {len(processed_domains_during_training)} domains.")
        del checkpoint
        clear_memory(force_gc=True, clear_cuda=(device.type == 'cuda'))
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

    # --- 3. Load RMSF Data (Once) ---
    logger.info("Loading RMSF data...")
    try:
        rmsf_data = load_rmsf_data(
            rmsf_dir=config["input"]["rmsf_dir"],
            replica=config["input"].get("replica", "replica_average"),
            temperature=config["input"]["temperature"]
        )
        global_rmsf_lookup = create_optimized_rmsf_lookup(rmsf_data)
        # Ensure 'average_rmsf' column exists after load_rmsf_data
        if 'average_rmsf' not in rmsf_data.columns:
            raise KeyError("'average_rmsf' column not found in RMSF data after loading/renaming.")
        fallback_rmsf_value = rmsf_data['average_rmsf'].median()

        memory_stats = check_memory_usage()
        logger.info(f"Memory after loading RMSF data: {memory_stats['system_percent']:.1f}%")
        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
             logger.warning("Memory critical after loading RMSF data. Attempting reduction.")
             emergency_memory_reduction()
    except Exception as e:
        logger.error(f"Error loading RMSF data: {str(e)}", exc_info=True)
        raise

    # --- 4. Determine Domains and Samples for Prediction (Once) ---
    voxel_file_path = resolve_path(config["input"]["voxel_file"])
    if not os.path.exists(voxel_file_path):
        raise FileNotFoundError(f"Voxel file not found: {voxel_file_path}")

    try:
        with h5py.File(voxel_file_path, 'r') as f:
            all_hdf5_domains = list(f.keys())
        logger.info(f"Found {len(all_hdf5_domains)} total domains in HDF5 file.")

        use_training_domains = config.get("prediction", {}).get("use_training_domains", False)
        config_domain_ids = config["input"].get("domain_ids", [])
        target_domain_ids = domain_ids if domain_ids else config_domain_ids # Use CLI arg if provided

        domains_to_predict_keys = []
        if target_domain_ids:
             logger.info(f"Filtering domains based on provided list (n={len(target_domain_ids)}).")
             available_set = set(all_hdf5_domains)
             base_to_full_map = {d.split('_')[0]: d for d in all_hdf5_domains}

             for d_id in target_domain_ids:
                 if d_id in available_set:
                      if d_id not in domains_to_predict_keys: domains_to_predict_keys.append(d_id)
                 else:
                      base_d_id = d_id.split('_')[0]
                      found_match = base_to_full_map.get(base_d_id)
                      if found_match:
                            if found_match not in domains_to_predict_keys: domains_to_predict_keys.append(found_match)
                      else:
                           logger.warning(f"Specified domain_id '{d_id}' not found by full or base name.")

             if not domains_to_predict_keys:
                  raise ValueError("None of the specified domain_ids matched domains in the HDF5 file.")
        elif use_training_domains and processed_domains_during_training:
             logger.info("Using domains encountered during training for prediction.")
             available_set = set(all_hdf5_domains)
             domains_to_predict_keys = [d for d in processed_domains_during_training if d in available_set]
             if len(domains_to_predict_keys) < len(processed_domains_during_training):
                 logger.warning(f"{len(processed_domains_during_training) - len(domains_to_predict_keys)} training domains not found in HDF5.")
        else:
            logger.info("Using all available domains from HDF5 file for prediction.")
            domains_to_predict_keys = all_hdf5_domains

        logger.info(f"Final list contains {len(domains_to_predict_keys)} domains for sample generation.")
        if not domains_to_predict_keys:
            raise ValueError("No domains selected for prediction.")

        all_rmsf_domains = rmsf_data['domain_id'].unique().tolist()
        global_domain_mapping = create_domain_mapping(domains_to_predict_keys, all_rmsf_domains)

        all_samples, determined_input_shape = generate_prediction_samples(
            voxel_file_path,
            domains_to_predict_keys,
            global_rmsf_lookup,
            global_domain_mapping
        )
        final_input_shape = determined_input_shape or ckpt_input_shape
        if final_input_shape is None:
             raise ValueError("Could not determine model input shape from HDF5 or checkpoint.")

        if not all_samples:
            raise ValueError("No valid samples generated for prediction. Check data alignment.")

    except Exception as e:
        logger.error(f"Error during sample generation phase: {str(e)}", exc_info=True)
        raise

    # --- 5. Create Dataset & DataLoader (Once) ---
    logger.info("Creating single prediction dataset and dataloader...")
    try:
        prediction_dataset = PredictionRMSFDataset(
            sample_list=all_samples,
            voxel_file_path=voxel_file_path,
            domain_mapping=global_domain_mapping,
            rmsf_lookup=global_rmsf_lookup,
            fallback_rmsf=fallback_rmsf_value,
            input_shape=final_input_shape # Pass the determined shape
        )

        memory_stats = check_memory_usage()
        config_dataloader_batch_size = config["prediction"].get("batch_size", 256)
        if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
            predicted_batch_size = max(8, config_dataloader_batch_size // 8)
            logger.warning(f"High memory ({memory_stats['system_percent']:.1f}%). Using DataLoader batch size: {predicted_batch_size}.")
        elif memory_stats['system_percent'] > MEMORY_WARNING_THRESHOLD * 100:
            predicted_batch_size = max(16, config_dataloader_batch_size // 4)
            logger.warning(f"Moderate memory ({memory_stats['system_percent']:.1f}%). Using DataLoader batch size: {predicted_batch_size}.")
        else:
            predicted_batch_size = config_dataloader_batch_size
            logger.info(f"Normal memory ({memory_stats['system_percent']:.1f}%). Using DataLoader batch size: {predicted_batch_size} (from config)")

        num_workers = adjust_workers_for_memory(config['system_utilization'].get('num_workers', 4))
        pin_memory_setting = (device.type == 'cuda') and (memory_stats['system_percent'] < MEMORY_CRITICAL_THRESHOLD * 100)
        persistent_workers_setting = (num_workers > 0)

        logger.info(f"DataLoader: batch={predicted_batch_size}, workers={num_workers}, pin_mem={pin_memory_setting}, persist={persistent_workers_setting}")

        prediction_loader = DataLoader(
            prediction_dataset,
            batch_size=predicted_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory_setting,
            persistent_workers=persistent_workers_setting,
            prefetch_factor=4 if num_workers > 0 else None
        )

    except Exception as e:
        logger.error(f"Error creating prediction dataset/dataloader: {str(e)}", exc_info=True)
        raise

    # --- 6. Prepare Output File ---
    predictions_dir = os.path.join(config["output"]["base_dir"], "metrics")
    ensure_dir(predictions_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    predictions_path = os.path.join(predictions_dir, f"predictions_{timestamp}.csv")
    logger.info(f"Output predictions will be saved to: {predictions_path}")
    try:
        with open(predictions_path, 'w') as f:
            f.write("domain_id,resid,resname,predicted_rmsf,actual_rmsf\n")
    except IOError as e:
        logger.error(f"Cannot write to output predictions file {predictions_path}: {e}")
        raise

    # --- 7. Run Prediction Loop ---
    logger.info(f"Starting prediction for {len(prediction_dataset)} total samples...")
    show_progress = config["logging"]["show_progress_bars"]
    progress_bar = None
    if show_progress:
        progress_bar = EnhancedProgressBar(
            total=len(prediction_loader),
            prefix="Predicting Residues",
            suffix="Complete",
            stage_info="PREDICT_LOOP"
        )

    total_predictions_written = 0
    chunk_size = 2000
    predictions_chunk = []
    batch_times = []

    try:
        with torch.no_grad():
             prediction_start_time = time.time()
             for i, batch_data in enumerate(prediction_loader):
                 iter_start_time = time.time()
                 if i > 0 and i % 100 == 0:
                     memory_stats = check_memory_usage()
                     if memory_stats['system_percent'] > MEMORY_CRITICAL_THRESHOLD * 100:
                         logger.warning(f"Critical memory ({memory_stats['system_percent']:.1f}%) during loop. Emergency reduction...")
                         emergency_memory_reduction()

                 try:
                     inputs, targets = batch_data
                 except ValueError as ve:
                     logger.error(f"Error unpacking batch {i}. Expected 2 items, got {len(batch_data)}. Error: {ve}. Skipping.")
                     if progress_bar: progress_bar.update(i + 1)
                     continue

                 inputs = inputs.to(device, non_blocking=pin_memory_setting)

                 with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                      outputs = model(inputs)

                 outputs_cpu = outputs.float().cpu().numpy()
                 targets_cpu = targets.float().cpu().numpy()

                 current_batch_size = len(inputs)
                 start_idx_in_full_list = min(i * predicted_batch_size, len(prediction_dataset.samples))
                 end_idx_in_full_list = min(start_idx_in_full_list + current_batch_size, len(prediction_dataset.samples))

                 if start_idx_in_full_list >= end_idx_in_full_list: continue

                 original_samples_for_batch = prediction_dataset.samples[start_idx_in_full_list:end_idx_in_full_list]

                 if len(original_samples_for_batch) != len(outputs_cpu):
                      logger.warning(f"Mismatch between result size ({len(outputs_cpu)}) and sample slice size ({len(original_samples_for_batch)}) for batch {i}. Results may be misaligned.")
                      # Attempt to process based on the smaller size to avoid index errors
                      process_len = min(len(original_samples_for_batch), len(outputs_cpu))
                 else:
                      process_len = len(original_samples_for_batch)


                 # --- CORRECTED RESNAME LOOKUP ---
                                  # --- CORRECTED RESNAME LOOKUP ---
                 for idx_in_batch in range(process_len):
                     try:
                         domain_id, resid_str = original_samples_for_batch[idx_in_batch]
                         pred_val = float(outputs_cpu[idx_in_batch])
                         true_val = float(targets_cpu[idx_in_batch]) # Actual RMSF still comes from DataLoader target

                         # --- MODIFICATION ---
                         # Skip the lookup in the original rmsf_data DataFrame - always use UNK
                         resname = "UNK"
                         # --- END MODIFICATION ---

                         predictions_chunk.append(f"{domain_id},{resid_str},{resname},{pred_val:.6f},{true_val:.6f}")

                     except Exception as proc_e: # <<< Corrected Indentation Here
                         logger.warning(f"Error processing result for sample index {idx_in_batch} in batch {i}: {proc_e}")
                 # --- END RESNAME LOOKUP CORRECTION ---

                 # --- [Original Code Commented Out for Reference] ---
                 # for idx_in_batch in range(process_len):
                 #     try:
                 #         domain_id, resid_str = original_samples_for_batch[idx_in_batch]
                 #         pred_val = float(outputs_cpu[idx_in_batch])
                 #         true_val = float(targets_cpu[idx_in_batch])
                 #
                 #         resname = "UNK" # Default
                 #         try:
                 #             rmsf_domain_for_lookup = global_domain_mapping.get(domain_id, domain_id)
                 #             resid_int_for_lookup = int(resid_str)
                 #
                 #             # Perform lookup in the original rmsf_data DataFrame
                 #             matching_rows = rmsf_data.loc[
                 #                 (rmsf_data['domain_id'] == rmsf_domain_for_lookup) &
                 #                 (rmsf_data['resid'] == resid_int_for_lookup)
                 #             ]
                 #
                 #             if not matching_rows.empty:
                 #                 resname = matching_rows.iloc[0]['resname']
                 #
                 #         except ValueError:
                 #             logger.warning(f"Cannot convert resid '{resid_str}' to int for resname lookup.")
                 #         except Exception as res_e:
                 #             logger.debug(f"Minor error during resname lookup for {domain_id}:{resid_str}: {res_e}")
                 #
                 #         predictions_chunk.append(f"{domain_id},{resid_str},{resname},{pred_val:.6f},{true_val:.6f}")
                 #
                 #     except Exception as proc_e:
                 #          logger.warning(f"Error processing result for sample index {idx_in_batch} in batch {i}: {proc_e}")
                 # --- [End Original Code Comment] ---


                 if len(predictions_chunk) >= chunk_size:
                     try:
                         with open(predictions_path, 'a') as f:
                             f.write('\n'.join(predictions_chunk) + '\n')
                         total_predictions_written += len(predictions_chunk)
                         predictions_chunk = []
                     except IOError as write_e:
                          logger.error(f"Error writing predictions chunk to {predictions_path}: {write_e}")

                 del inputs, targets, outputs, outputs_cpu, targets_cpu
                 gc.collect() # Add a garbage collect within the loop periodically
                 if progress_bar: progress_bar.update(i + 1)
                 batch_times.append(time.time() - iter_start_time)

                 if (i + 1) % 500 == 0 and device.type == 'cuda':
                     torch.cuda.empty_cache()

    except Exception as loop_e:
         logger.error(f"Error during prediction loop: {loop_e}", exc_info=True)
         if predictions_chunk: # Try saving remaining chunk on error
             try:
                 with open(predictions_path, 'a') as f: f.write('\n'.join(predictions_chunk) + '\n')
                 total_predictions_written += len(predictions_chunk)
             except IOError: pass
         raise

    finally:
        if progress_bar: progress_bar.finish()
        # Ensure final chunk is written even if loop finishes normally
        if predictions_chunk:
             try:
                 with open(predictions_path, 'a') as f: f.write('\n'.join(predictions_chunk) + '\n')
                 total_predictions_written += len(predictions_chunk)
             except IOError as write_e:
                  logger.error(f"Error writing final predictions chunk: {write_e}")


    # --- 8. Final Logging & Cleanup ---
    prediction_duration = time.time() - prediction_start_time
    logger.info(f"\nPrediction loop completed in {prediction_duration:.2f} seconds.")
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        samples_per_sec = predicted_batch_size / avg_batch_time if avg_batch_time > 0 else 0
        logger.info(f"Average DataLoader batch processing time: {avg_batch_time:.4f}s")
        logger.info(f"Estimated samples/sec: {samples_per_sec:.1f}")

    logger.info(f"Wrote {total_predictions_written} total predictions.")
    logger.info(f"Predictions saved to {predictions_path}")
    logger.info("OPTIMIZED PREDICTION PROCESS COMPLETED SUCCESSFULLY")

    clear_memory(force_gc=True, clear_cuda=True, aggressive=True)

    return predictions_path
