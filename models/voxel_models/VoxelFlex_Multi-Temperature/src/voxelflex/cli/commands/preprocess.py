"""
Preprocessing command for VoxelFlex (Temperature-Aware).

Performs metadata-only preprocessing - creating the master samples list
and temperature scaling parameters without processing voxel data.
"""

import os
import time
import json
import logging
import gc
import math
import h5py
from typing import Dict, Any, Tuple, List, Optional, Callable, Set, DefaultDict
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger("voxelflex.cli.preprocess")

from voxelflex.data.data_loader import (
    load_aggregated_rmsf_data, create_master_rmsf_lookup, create_domain_mapping
)
from voxelflex.utils.logging_utils import (
    log_stage, EnhancedProgressBar, log_memory_usage, log_section_header, get_logger
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, load_list_from_file, save_list_to_file, resolve_path
from voxelflex.utils.system_utils import clear_memory
from voxelflex.utils.temp_scaling import calculate_and_save_temp_scaling

MASTER_SAMPLES_FILENAME = "master_samples.parquet"  # Default to Parquet

def run_preprocessing(config: Dict[str, Any]) -> bool:
    """
    Main function for metadata-only preprocessing.
    Generates and saves master sample list and temperature scaler.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if preprocessing succeeded, False otherwise
        
    Raises:
        Various exceptions may be raised for data or file issues
    """
    if MASTER_SAMPLES_FILENAME.endswith(".parquet"):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            PYARROW_AVAILABLE = True
        except ImportError:
            logger.error("PyArrow needed for Parquet output. Please install with `pip install pyarrow`")
            raise ImportError("PyArrow needed for Parquet output. `pip install pyarrow`")

    log_section_header(logger, "STARTING PREPROCESSING (Metadata Only)")
    start_time = time.time()

    # Extract paths from config
    input_cfg = config['input']
    data_cfg = config['data']
    output_cfg = config['output']
    voxel_file_path = input_cfg['voxel_file']
    rmsf_file_path = input_cfg['aggregated_rmsf_file']
    processed_dir = data_cfg['processed_dir']
    run_output_dir = output_cfg['run_dir']
    master_samples_output_path = os.path.join(processed_dir, MASTER_SAMPLES_FILENAME)

    # Ensure output directories exist
    ensure_dir(processed_dir)
    ensure_dir(run_output_dir)

    failed_domains: Set[str] = set()
    master_samples_list: List[Dict[str, Any]] = []

    try:
        # --- 1. Load RMSF Data and Create Lookups ---
        with log_stage("PREPROCESS", "Loading RMSF Data & Mappings"):
            # Load with optimized dtypes
            rmsf_df = load_aggregated_rmsf_data(rmsf_file_path)
            
            # Create and serialize the lookup
            rmsf_lookup = create_master_rmsf_lookup(rmsf_df)
            if not rmsf_lookup:
                raise ValueError("RMSF lookup empty.")
                
            # Read HDF5 keys efficiently
            try:
                # Get domain keys from HDF5 - optimized to just read keys not data
                with h5py.File(voxel_file_path, 'r') as f_h5:
                    hdf5_domain_keys = list(f_h5.keys())
                    # Get temperature range while we have the data
                    temp_values = rmsf_df['temperature_feature'].dropna().unique()
                    temp_min = float(np.min(temp_values))
                    temp_max = float(np.max(temp_values))
                    logger.info(f"Detected temperature range: [{temp_min:.1f}, {temp_max:.1f}]")
                    
            except Exception as e:
                raise RuntimeError(f"Failed read HDF5 keys: {e}")
                
            if not hdf5_domain_keys:
                raise ValueError("No keys in HDF5.")
                
            # Create and serialize domain mapping
            rmsf_domain_ids = rmsf_df['domain_id'].unique().tolist()
            domain_mapping = create_domain_mapping(hdf5_domain_keys, rmsf_domain_ids)
            available_hdf5_keys = set(domain_mapping.keys())
            
            if not available_hdf5_keys:
                raise ValueError("No HDF5 keys mappable.")
                
            logger.info(f"Found {len(available_hdf5_keys)} mappable HDF5 domain keys.")
            
            # Calculate temperature scaler parameters from all the data
            temp_scaling_params_path = data_cfg["temp_scaling_params_file"]
            
            # Directly save temperature scaling parameters
            temp_scaling_params = {'temp_min': temp_min, 'temp_max': temp_max}
            try:
                ensure_dir(os.path.dirname(temp_scaling_params_path))
                save_json(temp_scaling_params, temp_scaling_params_path)
                logger.info(f"Saved temperature scaling parameters to {temp_scaling_params_path}")
            except Exception as e:
                logger.error(f"Failed to save temperature scaling parameters: {e}")
            
            # We can release the full dataframe to save memory
            del rmsf_df
            gc.collect()

        # --- 2. Load Domain Splits & Filter ---
        with log_stage("PREPROCESS", "Loading and Filtering Domain Splits"):
            split_domains_map: Dict[str, str] = {}
            all_split_hdf5_keys_in_use: Set[str] = set()
            
            for split in ["train", "val", "test"]:
                split_file = input_cfg.get(f"{split}_split_file")
                
                if not split_file or not os.path.exists(split_file):
                    logger.warning(f"Split file '{split}' not found.")
                    continue
                    
                domains_in_file = load_list_from_file(split_file)
                
                if not domains_in_file:
                    logger.warning(f"Split file '{split}' empty.")
                    continue
                    
                valid_split_domains = []
                for d in domains_in_file:
                    if d in available_hdf5_keys:
                        valid_split_domains.append(d)
                    elif d in hdf5_domain_keys:
                        logger.warning(f"Split '{split}': Domain '{d}' unmappable.")
                    else:
                        logger.warning(f"Split '{split}': Domain '{d}' not in HDF5.")
                        
                max_doms = input_cfg.get('max_domains')
                if max_doms is not None and max_doms > 0 and len(valid_split_domains) > max_doms:
                    valid_split_domains = valid_split_domains[:max_doms]
                    
                for d in valid_split_domains:
                    split_domains_map[d] = split
                    
                all_split_hdf5_keys_in_use.update(valid_split_domains)
                logger.info(f"Split '{split}': Using {len(valid_split_domains)} domains.")
                
            if not any(s == 'train' for s in split_domains_map.values()):
                raise ValueError("Train split empty/invalid.")
                
            if not any(s == 'val' for s in split_domains_map.values()):
                raise ValueError("Validation split empty/invalid.")
                
            if not any(s == 'test' for s in split_domains_map.values()):
                logger.warning("Test split empty/not specified.")
                
            logger.info(f"Total unique domains across valid splits: {len(all_split_hdf5_keys_in_use)}")

        # --- 3. Generate Master Sample List (with Split Info) ---
        with log_stage("PREPROCESS", "Generating Master Sample List"):
            residues_without_rmsf = 0
            domains_with_residues_checked: Set[str] = set()
            logger.info(f"Checking HDF5 residues for {len(all_split_hdf5_keys_in_use)} domains...")
            progress_domains = EnhancedProgressBar(len(all_split_hdf5_keys_in_use), desc="Checking HDF5 Residues")

            with h5py.File(voxel_file_path, 'r') as f_h5:
                for i, hdf5_domain_id in enumerate(all_split_hdf5_keys_in_use):
                    residue_group = None
                    domain_had_residues = False
                    samples_before_domain = len(master_samples_list)

                    try:
                        if hdf5_domain_id not in f_h5:
                            continue

                        potential_chain_keys = [k for k in f_h5[hdf5_domain_id].keys() 
                                               if isinstance(f_h5[hdf5_domain_id][k], h5py.Group)]
                        
                        # Find residue group efficiently
                        for chain_key in potential_chain_keys:
                            try:
                                chain_group = f_h5[hdf5_domain_id][chain_key]
                                # Check if any residue IDs exist (must be digit strings)
                                if any(k.isdigit() for k in chain_group.keys()):
                                    residue_group = chain_group
                                    break
                            except Exception:
                                continue

                        if residue_group is None:
                            logger.debug(f"No valid residue group found for {hdf5_domain_id}")
                            failed_domains.add(hdf5_domain_id)
                            progress_domains.update(1)
                            continue

                        # --- Process the found residue_group ---
                        domains_with_residues_checked.add(hdf5_domain_id)
                        rmsf_domain_id = domain_mapping.get(hdf5_domain_id)
                        
                        if rmsf_domain_id is None:
                            logger.debug(f"Domain {hdf5_domain_id} lost mapping.")
                            failed_domains.add(hdf5_domain_id)
                            progress_domains.update(1)
                            continue

                        split_assignment = split_domains_map.get(hdf5_domain_id, "unknown")
                        
                        # Get residue keys efficiently
                        residue_keys = [k for k in residue_group.keys() if k.isdigit()]
                        
                        for resid_str in residue_keys:
                            domain_had_residues = True
                            
                            try:
                                resid_int = int(resid_str)
                                lookup_key = (rmsf_domain_id, resid_int)
                                temp_rmsf_pairs = rmsf_lookup.get(lookup_key)
                                
                                if temp_rmsf_pairs is None:
                                    base_rmsf_id = rmsf_domain_id.split('_')[0]
                                    if base_rmsf_id != rmsf_domain_id:
                                        temp_rmsf_pairs = rmsf_lookup.get((base_rmsf_id, resid_int))
                                        
                                if temp_rmsf_pairs:
                                    for raw_temp, target_rmsf in temp_rmsf_pairs:
                                        if (raw_temp is not None and not np.isnan(raw_temp) and 
                                            target_rmsf is not None and not np.isnan(target_rmsf) and 
                                            target_rmsf >= 0):
                                            
                                            # Create a sample with all required fields
                                            # Ensure domain_id and resid_str are strings
                                            master_samples_list.append({
                                                'hdf5_domain_id': str(hdf5_domain_id),
                                                'resid_str': str(resid_str),
                                                'resid_int': resid_int,
                                                'raw_temp': float(raw_temp),
                                                'target_rmsf': float(target_rmsf),
                                                'split': split_assignment
                                            })
                                else:
                                    residues_without_rmsf += 1
                                    
                            except ValueError:
                                logger.debug(f"Invalid resid '{resid_str}' in {hdf5_domain_id}.")
                            except Exception as e:
                                logger.debug(f"Error processing {hdf5_domain_id}:{resid_str}: {e}")
                        
                        # Verify we got samples for this domain
                        samples_after_domain = len(master_samples_list)
                        if domain_had_residues and samples_after_domain == samples_before_domain:
                            logger.warning(f"Domain '{hdf5_domain_id}' had residues but no valid samples (check RMSF lookup).")
                            failed_domains.add(hdf5_domain_id)

                    except Exception as e:
                        logger.warning(f"Error processing domain {hdf5_domain_id}: {e}")
                        failed_domains.add(hdf5_domain_id)

                    progress_domains.update(1)  # Update progress
                
            progress_domains.finish()

            if not master_samples_list:
                raise ValueError("Master sample list is empty after processing all domains.")
                
            logger.info(f"Generated {len(master_samples_list)} total sample entries.")
            
            if residues_without_rmsf > 0:
                logger.info(f"  {residues_without_rmsf} HDF5 residues lacked corresponding RMSF data.")
                
            for domain_id in all_split_hdf5_keys_in_use:
                if domain_id not in domains_with_residues_checked and domain_id not in failed_domains:
                    logger.warning(f"Domain '{domain_id}' was in splits but never processed?")
                    failed_domains.add(domain_id)
                    
            # Free memory
            del rmsf_lookup
            gc.collect()

        # --- 4. Save Master Sample List ---
        with log_stage("PREPROCESS", "Saving Master Sample List"):
            if not master_samples_list:
                raise ValueError("Cannot save empty master sample list.")
                
            master_samples_df = pd.DataFrame(master_samples_list)
            logger.info(f"Saving {len(master_samples_df)} samples to {master_samples_output_path}...")
            
            try:
                if MASTER_SAMPLES_FILENAME.endswith(".parquet"):
                    # Use pyarrow with specified schema for more efficient storage and proper types
                    schema = pa.schema([
                        ('hdf5_domain_id', pa.string()),
                        ('resid_str', pa.string()),
                        ('resid_int', pa.int64()),
                        ('raw_temp', pa.float32()),
                        ('target_rmsf', pa.float32()),
                        ('split', pa.string())
                    ])
                    
                    table = pa.Table.from_pandas(master_samples_df, schema=schema, preserve_index=False)
                    # Use compression for smaller file size
                    pq.write_table(table, master_samples_output_path, compression='snappy')
                else:
                    # If using CSV, ensure proper types and format
                    master_samples_df.to_csv(master_samples_output_path, index=False, float_format='%.6f')
                    
                logger.info(f"Master sample list saved successfully to {master_samples_output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save master sample list: {e}")
                raise
                
            if not os.path.exists(master_samples_output_path) or os.path.getsize(master_samples_output_path) == 0:
                raise RuntimeError(f"Master sample file {master_samples_output_path} not created/empty.")

            # Calculate temperature scaling using training samples
            train_samples = master_samples_df[master_samples_df['split'] == 'train']
            train_temps = train_samples['raw_temp'].tolist()
            temp_min, temp_max = calculate_and_save_temp_scaling(train_temps, temp_scaling_params_path)

        # --- 5. Final Summary ---
        log_section_header(logger, "PREPROCESSING FINISHED (Metadata Only)")
        total_duration = time.time() - start_time
        logger.info(f"Total Preprocessing Time: {total_duration:.2f}s.")
        logger.info(f"Saved master sample list ({len(master_samples_list)} samples) to {master_samples_output_path}")
        logger.info("Voxel data was NOT processed or saved in this step.")
        
        num_files_in_processed = len([f for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))])
        logger.info(f"Found {num_files_in_processed} file(s) in {processed_dir}.")

        if failed_domains:
            failed_list_path = os.path.join(run_output_dir, "failed_preprocess_domains.txt")
            logger.warning(f"Found {len(failed_domains)} domains with check/mapping/residue failure.")
            try:
                save_list_to_file(sorted(list(failed_domains)), failed_list_path)
                logger.info(f"Failed domains list saved: {failed_list_path}")
            except Exception as save_err:
                logger.error(f"Could not save failed domains list: {save_err}")
        else:
            logger.info("No domains completely failed during initial checks/mapping.")

        return True

    except Exception as e:
        logger.exception(f"Metadata preprocessing pipeline failed: {e}")
        if failed_domains:
            try:
                failed_list_path = os.path.join(run_output_dir, "failed_preprocess_domains_partial.txt")
                save_list_to_file(sorted(list(failed_domains)), failed_list_path)
                logger.info(f"Saved partial failed domains list: {failed_list_path}")
            except:
                pass
        return False
    finally:
        if 'master_samples_list' in locals():
            del master_samples_list
        gc.collect()
        logger.info("End of metadata preprocessing run.")
        log_memory_usage(logger, level=logging.INFO)