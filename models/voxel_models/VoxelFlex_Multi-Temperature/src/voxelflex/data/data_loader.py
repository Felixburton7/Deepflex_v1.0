# src/voxelflex/data/data_loader.py (Chunked IterableDataset Implementation for All Splits)
import os
from pathlib import Path
import logging
import time
import gc
import h5py
import json
import psutil
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set, Iterator
from collections import defaultdict, OrderedDict, namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger("voxelflex.data")

# Utils and Validators (ensure they are imported)
from voxelflex.data.validators import validate_aggregated_rmsf_data
from voxelflex.utils.file_utils import resolve_path, ensure_dir, load_json, save_json
from voxelflex.utils.logging_utils import EnhancedProgressBar

# --- Constants and Helpers ---
MASTER_SAMPLES_FILENAME = "master_samples.parquet"
TARGET_SHAPE: Tuple[int, int, int, int] = (5, 21, 21, 21)
TARGET_DTYPE: np.dtype = np.dtype(np.float32)
INPUT_CHANNELS: int = TARGET_SHAPE[0]

def process_voxel(voxel_raw: np.ndarray) -> Optional[np.ndarray]:
    """Processes a raw voxel array from HDF5 to the target format."""
    try:
        if not isinstance(voxel_raw, np.ndarray): return None
        if voxel_raw.dtype == bool: processed_array = voxel_raw.astype(TARGET_DTYPE)
        elif np.issubdtype(voxel_raw.dtype, np.floating): processed_array = voxel_raw.astype(TARGET_DTYPE, copy=False)
        elif np.issubdtype(voxel_raw.dtype, np.integer): processed_array = voxel_raw.astype(TARGET_DTYPE)
        else: return None
        if processed_array.ndim == 4 and processed_array.shape[-1] == INPUT_CHANNELS: processed_array = np.transpose(processed_array, (3, 0, 1, 2))
        elif processed_array.ndim != len(TARGET_SHAPE): return None
        if processed_array.shape != TARGET_SHAPE: return None
        if not np.isfinite(processed_array).all(): return None
        return processed_array
    except Exception: return None
def load_process_domain_from_handle(
    h5_handle,
    domain_id: str,
    expected_channels: int = INPUT_CHANNELS,
    target_shape_chw: Optional[Tuple[int, ...]] = TARGET_SHAPE,
    log_per_residue: bool = False
) -> Dict[str, np.ndarray]:
    """
    Loads and processes all valid residues for a given domain_id from an open HDF5 handle
    with enhanced error handling for corrupted data.
    
    Args:
        h5_handle: Open HDF5 file handle
        domain_id: Domain ID to process
        expected_channels: Expected number of channels in voxel data
        target_shape_chw: Expected shape of processed voxels
        log_per_residue: Whether to log details for each residue
        
    Returns:
        Dictionary mapping residue IDs to processed voxel arrays
    """
    domain_data_dict = {}
    processed_count = 0
    failed_count = 0
    t_start = time.perf_counter()
    
    try:
        # First check if domain exists in HDF5
        if domain_id not in h5_handle:
            logger.warning(f"Domain '{domain_id}' not found...")
            return {}
            
        # Get domain group and find residue group
        domain_group = h5_handle[domain_id]
        residue_group = None
        
        # Find the chain containing residues (look for group with numeric keys)
        potential_chain_keys = [k for k in domain_group.keys() if isinstance(domain_group[k], h5py.Group)]
        
        for chain_key in potential_chain_keys:
            try:
                potential_res_group = domain_group[chain_key]
                # Check if this group contains residue IDs (numeric keys)
                if any(key.isdigit() for key in potential_res_group.keys()): 
                    residue_group = potential_res_group
                    break
            except Exception as chain_error:
                # Skip chains that can't be accessed
                logger.debug(f"Skipping chain {chain_key} in {domain_id}: {chain_error}")
                continue
                
        # If no valid residue group found, return empty
        if residue_group is None:
            logger.warning(f"No valid residue group for {domain_id}.")
            return {}
            
        # Get all residue keys and sort them numerically
        try:
            residue_keys = sorted([k for k in residue_group.keys() if k.isdigit()], key=int)
        except Exception as key_error:
            logger.warning(f"Error sorting residue keys for {domain_id}: {key_error}")
            # Fallback: use keys as-is if sorting fails
            residue_keys = [k for k in residue_group.keys() if k.isdigit()]
            
        # Process each residue
        for resid_str in residue_keys:
            voxel_raw = None
            
            try:
                # Use try-except blocks around all HDF5 operations for robust error handling
                try:
                    voxel_dataset = residue_group[resid_str]
                except Exception as access_err:
                    logger.debug(f"Cannot access dataset for {domain_id}:{resid_str}: {access_err}")
                    failed_count += 1
                    continue
                    
                if not isinstance(voxel_dataset, h5py.Dataset):
                    continue
                    
                # Read the dataset with error handling
                try:
                    voxel_raw = voxel_dataset[:]
                except Exception as read_err:
                    # More specific error reporting
                    error_msg = str(read_err)
                    if "inflate() failed" in error_msg or "bad object header" in error_msg:
                        logger.warning(f"Error reading/processing {domain_id}:{resid_str}: '{error_msg}'")
                    else:
                        logger.warning(f"Error reading/processing {domain_id}:{resid_str}: {read_err}")
                    failed_count += 1
                    continue
                    
                # Process voxel data
                processed_array = process_voxel(voxel_raw)
                
                if processed_array is not None:
                    domain_data_dict[resid_str] = processed_array
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error processing {domain_id}:{resid_str}: {e}", exc_info=False)
            finally:
                # Always clean up raw data
                del voxel_raw
                
        # Log summary
        t_end = time.perf_counter()
        if processed_count > 0 or failed_count > 0:
            logger.debug(f"Domain {domain_id}: Processed={processed_count}, Failed={failed_count}. Time: {(t_end-t_start)*1000:.1f}ms")
            
        return domain_data_dict
        
    except Exception as e:
        logger.error(f"Critical error loading domain {domain_id}: {e}", exc_info=True)
        return {}


try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

class ChunkedVoxelDataset(IterableDataset):
    """
    IterableDataset that loads and processes protein voxel data in chunks of domains.
    
    This dataset is designed for memory-efficient loading of large HDF5 files,
    processing domains in manageable chunks rather than all at once.
    """
    def __init__(self,
                 master_samples_path: str,
                 split: str,
                 domain_list: List[str],
                 voxel_hdf5_path: str,
                 temp_scaling_params: Dict[str, float],
                 chunk_size: int = 100,
                 shuffle_domain_list: bool = False):
        """
        Initialize the chunked dataset for a specific split.
        
        Args:
            master_samples_path: Path to the master_samples.parquet file
            split: Dataset split ('train', 'val', 'test')
            domain_list: Complete list of domain IDs for this split
            voxel_hdf5_path: Path to the HDF5 voxel file
            temp_scaling_params: Dict with 'temp_min' and 'temp_max'
            chunk_size: Number of domains to load per chunk
            shuffle_domain_list: Whether to shuffle domains before chunking
        """
        super().__init__()
        self.split = split
        self.domain_list = domain_list  # Keep original order initially
        self.voxel_hdf5_path = Path(voxel_hdf5_path).resolve()
        self.temp_min = temp_scaling_params.get('temp_min', 320.0)
        self.temp_max = temp_scaling_params.get('temp_max', 450.0)
        
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            logger.warning(f"Invalid chunk_size ({chunk_size}), defaulting to 100.")
            self.chunk_size = 100
        else:
            self.chunk_size = chunk_size
            
        self.shuffle_domain_list = shuffle_domain_list

        # Store number of domains for length estimation
        self.num_domains = len(domain_list)

        # Load metadata from master samples file
        self.metadata_lookup = {}
        try:
            logger.info(f"IterableDataset [{split}]: Loading metadata from {master_samples_path}")
            t0 = time.time()
            
            # Select only necessary columns
            cols = ['hdf5_domain_id', 'resid_str', 'raw_temp', 'target_rmsf', 'split']
            
            # Efficient reading with filtering if pyarrow is available
            if PYARROW_AVAILABLE:
                filters = [('split', '==', split)]
                df = pd.read_parquet(master_samples_path, columns=cols, filters=filters)
            else:
                df_full = pd.read_csv(master_samples_path, usecols=cols)
                df = df_full[df_full['split'] == split].copy()
                del df_full  # Free memory

            # Create lookup with string keys for consistency - this is the critical fix
            for _, row in df.iterrows():
                key = (str(row['hdf5_domain_id']), str(row['resid_str']))
                if key not in self.metadata_lookup:
                    self.metadata_lookup[key] = []
                self.metadata_lookup[key].append((float(row['raw_temp']), float(row['target_rmsf'])))

            # Store count of actual metadata entries for length estimation
            self.metadata_count = len(df)

            logger.info(f"IterableDataset [{split}]: Metadata lookup created ({len(self.metadata_lookup)} entries) in {time.time()-t0:.2f}s")
            del df
            gc.collect()
        except Exception as e:
            logger.exception(f"IterableDataset [{split}]: Failed to load metadata lookup: {e}")
            self.metadata_lookup = {}
            self.domain_list = []
            self.metadata_count = 0

    def _load_process_chunk_voxels(self, domain_chunk: List[str], worker_id: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Loads and processes voxels for a list of domains with enhanced error handling.
        
        Args:
            domain_chunk: List of domain IDs to process
            worker_id: ID of the worker processing this chunk
            
        Returns:
            Dictionary mapping domain IDs to dictionaries of residue voxels
        """
        t_start = time.perf_counter()
        logger.debug(f"Worker {worker_id}: Loading chunk ({len(domain_chunk)} domains): {domain_chunk[:3]}...")
        chunk_voxel_data: Dict[str, Dict[str, np.ndarray]] = {}
        
        try:
            # Open HDF5 file locally within this method for the chunk
            with h5py.File(self.voxel_hdf5_path, 'r') as h5_file:
                for domain_id in domain_chunk:
                    # Check domain existence with safer error handling
                    if domain_id not in h5_file:
                        logger.warning(f"Domain '{domain_id}' not found in HDF5 file.")
                        continue

                    try:
                        domain_data = load_process_domain_from_handle(
                            h5_file, domain_id,
                            expected_channels=INPUT_CHANNELS,
                            target_shape_chw=TARGET_SHAPE,
                            log_per_residue=False
                        )
                        if domain_data:
                            chunk_voxel_data[domain_id] = domain_data
                    except Exception as domain_error:
                        # Add more specific error context
                        logger.warning(f"Error processing domain {domain_id}: {domain_error}")
                        # Continue with next domain rather than failing the whole chunk
                        continue
                        
            t_end = time.perf_counter()
            logger.info(f"Worker {worker_id}: Loaded/processed chunk ({len(chunk_voxel_data)}/{len(domain_chunk)} domains ok) in {t_end - t_start:.2f}s.")
            return chunk_voxel_data
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to open/read HDF5 for chunk: {e}", exc_info=True)
            return {}

    def _prepare_samples_for_chunk(self, chunk_voxel_data: Dict[str, Dict[str, np.ndarray]], worker_id: int) -> List[Dict[str, Any]]:
        """
        Combines loaded voxels with metadata for the chunk.
        
        Args:
            chunk_voxel_data: Dictionary of domain voxel data
            worker_id: ID of the worker processing this chunk
            
        Returns:
            List of sample dictionaries with voxels, temperatures, and targets
        """
        t_start = time.perf_counter()
        chunk_samples = []
        skipped_meta = 0
        temp_range = self.temp_max - self.temp_min
        use_midpoint = abs(temp_range) < 1e-6

        # Process domains/residues in a deterministic order for consistency
        for domain_id in sorted(chunk_voxel_data.keys()):
            residues = chunk_voxel_data[domain_id]
            for resid_str in sorted(residues.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                voxel_np = residues[resid_str]
                try:
                    # Use string keys for lookup - matching the format used during initialization
                    key = (str(domain_id), str(resid_str))
                    metadata_entries = self.metadata_lookup.get(key, [])
                    
                    if not metadata_entries:
                        # Debug logging to help diagnose lookup issues
                        logger.debug(f"Worker {worker_id}: No metadata for {domain_id}:{resid_str}")
                        skipped_meta += 1
                        continue
                        
                    for raw_temp, target_rmsf in metadata_entries:
                        # Scale temperature between 0-1
                        if use_midpoint:
                            scaled_temp = 0.5
                        else:
                            scaled_temp = (raw_temp - self.temp_min) / temp_range
                        scaled_temp = min(max(scaled_temp, 0.0), 1.0)

                        # Ensure voxel array is C-contiguous for efficient tensor conversion
                        if not voxel_np.flags['C_CONTIGUOUS']:
                            voxel_np = np.ascontiguousarray(voxel_np)
                            
                        # Convert to PyTorch tensors
                        voxel_tensor = torch.from_numpy(voxel_np)
                        scaled_temp_tensor = torch.tensor([scaled_temp], dtype=torch.float32)
                        target_tensor = torch.tensor(target_rmsf, dtype=torch.float32)

                        # Add sample to batch
                        chunk_samples.append({
                            'voxels': voxel_tensor, 
                            'scaled_temps': scaled_temp_tensor, 
                            'targets': target_tensor
                        })
                except Exception as e:
                    logger.warning(f"Worker {worker_id}: Error preparing sample {domain_id}:{resid_str}: {e}")
                    skipped_meta += 1

        t_end = time.perf_counter()
        if chunk_samples:
            logger.info(f"Worker {worker_id} [{self.split}]: Yielding {len(chunk_samples)} samples for chunk.")
        else:
            logger.warning(f"Worker {worker_id} [{self.split}]: No samples prepared for chunk. Skipped {skipped_meta} residues.")
        logger.debug(f"Worker {worker_id}: Prepared {len(chunk_samples)} samples, skipped {skipped_meta} (meta). Time: {t_end - t_start:.2f}s.")
        return chunk_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator logic loading, processing, and yielding samples chunk by chunk.
        
        Yields:
            Dictionary with 'voxels', 'scaled_temps', and 'targets' tensors
        """
        # Get worker info for parallel processing
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-worker case (e.g., debugging)
            worker_id = 0
            num_workers = 1
            domains_for_worker = self.domain_list
        else:
            # Multi-worker case
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Partition domains equally among workers
            per_worker = int(math.ceil(len(self.domain_list) / float(num_workers)))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.domain_list))
            domains_for_worker = self.domain_list[start_idx:end_idx]

        if not domains_for_worker:
            logger.warning(f"Worker {worker_id}: No domains assigned for split '{self.split}'.")
            return

        # Make a copy to avoid modifying original list
        effective_domain_list = list(domains_for_worker)
        
        # Shuffle if requested (typically for training)
        if self.shuffle_domain_list:
            np.random.shuffle(effective_domain_list)
            logger.info(f"Worker {worker_id}: Shuffled domain list for split '{self.split}'.")
        else:
            logger.info(f"Worker {worker_id}: Processing domains in original order for split '{self.split}'.")

        # Process domains in chunks
        num_chunks = math.ceil(len(effective_domain_list) / self.chunk_size)
        logger.info(f"Worker {worker_id}: Starting iteration for '{self.split}' split over {num_chunks} chunks (size ~{self.chunk_size}).")

        for i in range(num_chunks):
            chunk_start_idx = i * self.chunk_size
            chunk_end_idx = min(chunk_start_idx + self.chunk_size, len(effective_domain_list))
            domain_chunk_ids = effective_domain_list[chunk_start_idx:chunk_end_idx]
            if not domain_chunk_ids:
                continue

            logger.info(f"Worker {worker_id} [{self.split}]: Processing chunk {i+1}/{num_chunks} ({len(domain_chunk_ids)} domains)")
            
            # Load and process voxels for this chunk
            chunk_voxels = self._load_process_chunk_voxels(domain_chunk_ids, worker_id)
            if not chunk_voxels:
                logger.warning(f"Worker {worker_id} [{self.split}]: No voxels loaded for chunk {i+1}.")
                continue

            # Prepare samples from the loaded chunk
            chunk_samples = self._prepare_samples_for_chunk(chunk_voxels, worker_id)
            
            # Free voxel memory after preparing samples
            del chunk_voxels
            gc.collect()

            if not chunk_samples:
                logger.warning(f"Worker {worker_id} [{self.split}]: No samples prepared for chunk {i+1}.")
                continue

            # Yield samples one by one
            for sample in chunk_samples:
                yield sample

            # Clean up after yielding all samples from this chunk
            logger.debug(f"Worker {worker_id}: Finished yielding chunk {i+1}. Clearing samples.")
            del chunk_samples
            gc.collect()

        logger.info(f"Worker {worker_id}: Finished iteration for split '{self.split}'.")

    def __len__(self):
        """
        Provides an estimated length for the dataset based on metadata.
        This is an approximation since IterableDatasets don't technically have fixed lengths.
        
        Returns:
            Estimated number of samples in the dataset
        """
        # Return the count of metadata entries as the best estimate of dataset length
        return self.metadata_count


# No longer needs to manage persistent HDF5 handles for ChunkedVoxelDataset
def worker_init_fn(worker_id):
    """Worker initialization function (minimal version)."""
    worker_info = get_worker_info()
    if worker_info:
        seed = worker_info.seed % 2**32 # Ensure seed is in valid range
        np.random.seed(seed)
        # torch.manual_seed(seed) # Can cause issues if main process also seeds
        logger.debug(f"Worker {worker_id}: Initialized with NumPy seed {seed}")
    else:
         logger.warning(f"Worker {worker_id}: Could not get worker_info for seeding.")

# --- simple_collate_fn (Keep as is) ---
def simple_collate_fn(batch):
    # ... (implementation as before) ...
    batch = [item for item in batch if item is not None]
    if not batch: logger.warning("Collate fn received empty batch."); return None
    try:
        return {
            'voxels': torch.stack([item['voxels'] for item in batch]),
            'scaled_temps': torch.stack([item['scaled_temps'] for item in batch]),
            'targets': torch.stack([item['targets'] for item in batch])
        }
    except RuntimeError as e: logger.error(f"Collation error (shape mismatch?): {e}"); return None
    except Exception as e: logger.error(f"Unexpected collation error: {e}", exc_info=True); return None



# --- Preprocessing Helpers (Keep as before) ---
def load_aggregated_rmsf_data(aggregated_rmsf_file: str) -> pd.DataFrame:
    rmsf_file = resolve_path(aggregated_rmsf_file)
    logger.info(f"Loading RMSF data from: {rmsf_file}")
    if not os.path.exists(rmsf_file): raise FileNotFoundError(f"RMSF file not found: {rmsf_file}")
    try:
        dtype_spec = {'domain_id': str, 'resid': 'Int64', 'temperature_feature': float, 'target_rmsf': float}
        rmsf_df = pd.read_csv(rmsf_file, dtype=dtype_spec, low_memory=False)
        logger.info(f"Loaded {len(rmsf_df)} rows with optimized types")
        return validate_aggregated_rmsf_data(rmsf_df)
    except Exception as e: logger.exception(f"Failed to read RMSF CSV: {e}"); raise

def create_master_rmsf_lookup(rmsf_df: pd.DataFrame) -> Dict[Tuple[str, int], List[Tuple[float, float]]]:
    logger.info("Creating RMSF lookup..."); t0 = time.time(); required = ['domain_id', 'resid', 'temperature_feature', 'target_rmsf']
    if not all(c in rmsf_df.columns for c in required): raise ValueError(f"Missing: {required}")
    lookup = defaultdict(list); grouped = rmsf_df.groupby(['domain_id', 'resid'])
    for name, group in grouped:
        try: lookup[(str(name[0]), int(name[1]))] = list(zip(group['temperature_feature'].astype(float), group['target_rmsf'].astype(float)))
        except (ValueError, TypeError): logger.warning(f"Skipping RMSF lookup entry due to conversion error for {name}")
    base_lookup={}; added_count=0
    for k, v in lookup.items():
        base = k[0].split('_')[0]
        if base != k[0]: base_key=(base, k[1]);
        if base_key not in lookup and base_key not in base_lookup: base_lookup[base_key]=v; added_count+=1
    lookup.update(base_lookup)
    logger.info(f"RMSF lookup created: {len(lookup)} keys ({added_count} base names) in {time.time()-t0:.2f}s")
    return dict(lookup)

def create_domain_mapping(voxel_domain_keys: List[str], rmsf_domain_ids: List[str]) -> Dict[str, str]:
    logger.info("Creating domain mapping...");
    if not voxel_domain_keys: logger.warning("Voxel keys empty."); return {}
    if not rmsf_domain_ids: logger.warning("RMSF IDs empty."); return {}
    v_set = set(voxel_domain_keys); r_set = set(rmsf_domain_ids); mapping={}; matches={'exact':0, 'base':0}
    r_base_map={r.split('_')[0]: r for r in rmsf_domain_ids};
    for h_key in v_set:
        if h_key in r_set: mapping[h_key]=h_key; matches['exact']+=1; continue
        base_h = h_key.split('_')[0]
        if base_h in r_set: mapping[h_key]=base_h; matches['base']+=1;
        elif base_h in r_base_map: mapping[h_key]=r_base_map[base_h]; matches['base']+=1
    total = len(mapping); cov = (total/len(voxel_domain_keys))*100 if voxel_domain_keys else 0
    logger.info(f"Domain mapping: {total}/{len(voxel_domain_keys)} keys ({cov:.1f}%)"); logger.info(f"  Matches: Exact={matches['exact']}, Base={matches['base']}")
    return mapping


# --- PredictionDataset for evaluation and inference ---
class PredictionDataset(Dataset):
    """
    Map-style dataset for prediction/evaluation that loads voxels
    for specific domain-residue pairs.
    """
    def __init__(self, 
                 samples_to_load: List[Tuple[str, str]],
                 voxel_hdf5_path: str,
                 expected_channels: int = INPUT_CHANNELS,
                 target_shape_chw: Tuple[int, ...] = TARGET_SHAPE):
        """
        Initialize prediction dataset.
        
        Args:
            samples_to_load: List of (domain_id, resid_str) tuples to load
            voxel_hdf5_path: Path to HDF5 file with voxel data
            expected_channels: Number of channels in voxel data
            target_shape_chw: Expected shape of processed voxels
        """
        super().__init__()
        self.voxel_hdf5_path = Path(voxel_hdf5_path).resolve()
        self.expected_channels = expected_channels
        self.target_shape = target_shape_chw
        
        # Load and filter samples
        self.samples = []
        self.voxel_data = {}
        
        logger.info(f"PredictionDataset: Loading {len(samples_to_load)} samples from HDF5...")
        
        # Pre-load and validate all samples
        self._preload_samples(samples_to_load)
        
        logger.info(f"PredictionDataset: Successfully loaded {len(self.samples)} valid samples.")

    def _preload_samples(self, samples_to_load: List[Tuple[str, str]]):
        """
        Pre-load and validate voxel data for specified samples.
        
        Args:
            samples_to_load: List of (domain_id, resid_str) tuples to load
        """
        # Group by domain for efficient loading
        domain_to_residues = defaultdict(list)
        for domain_id, resid_str in samples_to_load:
            domain_to_residues[domain_id].append(resid_str)
            
        # Load domains
        valid_count = 0
        with h5py.File(self.voxel_hdf5_path, 'r') as h5_file:
            for domain_id, residues in domain_to_residues.items():
                domain_data = load_process_domain_from_handle(
                    h5_file, domain_id,
                    expected_channels=self.expected_channels,
                    target_shape_chw=self.target_shape
                )
                
                # Add valid samples to the dataset
                for resid_str in residues:
                    if resid_str in domain_data:
                        self.samples.append((domain_id, resid_str))
                        self.voxel_data[(domain_id, resid_str)] = domain_data[resid_str]
                        valid_count += 1
                        
        logger.info(f"PredictionDataset: Loaded {valid_count}/{len(samples_to_load)} valid voxel arrays.")

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (domain_id, resid_str, voxel_tensor)
        """
        domain_id, resid_str = self.samples[idx]
        voxel_array = self.voxel_data[(domain_id, resid_str)]
        voxel_tensor = torch.from_numpy(voxel_array)
        return domain_id, resid_str, voxel_tensor

# --- DataLoader helper functions ---
def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader.
    
    Args:
        worker_id: ID of the worker being initialized
    """
    worker_info = get_worker_info()
    if worker_info:
        # Set worker-specific random seed for reproducibility
        seed = worker_info.seed % 2**32  # Ensure seed is in valid range
        np.random.seed(seed)
        logger.debug(f"Worker {worker_id}: Initialized with NumPy seed {seed}")
    else:
        logger.warning(f"Worker {worker_id}: Could not get worker_info for seeding.")

def simple_collate_fn(batch):
    """
    Collate function for DataLoader that handles None values and shape mismatches.
    
    Args:
        batch: Batch of samples to collate
        
    Returns:
        Dictionary of batched tensors or None if batch is empty
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        logger.warning("Collate fn received empty batch.")
        return None
        
    try:
        return {
            'voxels': torch.stack([item['voxels'] for item in batch]),
            'scaled_temps': torch.stack([item['scaled_temps'] for item in batch]),
            'targets': torch.stack([item['targets'] for item in batch])
        }
    except RuntimeError as e:
        logger.error(f"Collation error (shape mismatch?): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected collation error: {e}", exc_info=True)
        return None

# --- Preprocessing Helpers ---
def load_aggregated_rmsf_data(aggregated_rmsf_file: str) -> pd.DataFrame:
    """
    Load and validate aggregated RMSF data from CSV.
    
    Args:
        aggregated_rmsf_file: Path to CSV file with RMSF data
        
    Returns:
        DataFrame with validated RMSF data
        
    Raises:
        FileNotFoundError: If RMSF file not found
        Exception: For other loading errors
    """
    rmsf_file = Path(aggregated_rmsf_file).resolve()
    logger.info(f"Loading RMSF data from: {rmsf_file}")
    
    if not rmsf_file.exists():
        raise FileNotFoundError(f"RMSF file not found: {rmsf_file}")
        
    try:
        # Define expected dtypes for optimized loading
        dtype_spec = {
            'domain_id': str, 
            'resid': 'Int64', 
            'temperature_feature': float, 
            'target_rmsf': float
        }
        
        # Load CSV with optimized settings
        rmsf_df = pd.read_csv(rmsf_file, dtype=dtype_spec, low_memory=False)
        logger.info(f"Loaded {len(rmsf_df)} rows with optimized types")
        
        # Validate and clean data
        from voxelflex.data.validators import validate_aggregated_rmsf_data
        return validate_aggregated_rmsf_data(rmsf_df)
    except Exception as e:
        logger.exception(f"Failed to read RMSF CSV: {e}")
        raise

def create_master_rmsf_lookup(rmsf_df: pd.DataFrame) -> Dict[Tuple[str, int], List[Tuple[float, float]]]:
    """
    Create lookup mapping (domain_id, resid) to list of (temperature, rmsf) pairs.
    
    Args:
        rmsf_df: DataFrame with RMSF data
        
    Returns:
        Dictionary mapping (domain_id, resid) to [(temp1, rmsf1), (temp2, rmsf2), ...]
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Creating RMSF lookup...")
    t0 = time.time()
    
    # Check required columns
    required = ['domain_id', 'resid', 'temperature_feature', 'target_rmsf']
    if not all(c in rmsf_df.columns for c in required):
        raise ValueError(f"Missing required columns: {required}")
    
    # Group by domain and residue for efficient lookup
    lookup = defaultdict(list)
    grouped = rmsf_df.groupby(['domain_id', 'resid'])
    
    for name, group in grouped:
        try:
            # Create list of (temp, rmsf) tuples
            lookup[(str(name[0]), int(name[1]))] = list(zip(
                group['temperature_feature'].astype(float),
                group['target_rmsf'].astype(float)
            ))
        except (ValueError, TypeError):
            logger.warning(f"Skipping RMSF lookup entry due to conversion error for {name}")
    
    # Handle base domain IDs (without chain/model suffix)
    base_lookup = {}
    added_count = 0
    
    for k, v in lookup.items():
        base = k[0].split('_')[0]
        if base != k[0]:
            base_key = (base, k[1])
            if base_key not in lookup and base_key not in base_lookup:
                base_lookup[base_key] = v
                added_count += 1
    
    # Merge base lookups into main lookup
    lookup.update(base_lookup)
    
    logger.info(f"RMSF lookup created: {len(lookup)} keys ({added_count} base names) in {time.time()-t0:.2f}s")
    return dict(lookup)

def create_domain_mapping(voxel_domain_keys: List[str], rmsf_domain_ids: List[str]) -> Dict[str, str]:
    """
    Create mapping from HDF5 domain keys to RMSF domain IDs.
    
    Args:
        voxel_domain_keys: List of domain keys from HDF5 file
        rmsf_domain_ids: List of domain IDs from RMSF data
        
    Returns:
        Dictionary mapping HDF5 keys to RMSF IDs
    """
    logger.info("Creating domain mapping...")
    
    if not voxel_domain_keys:
        logger.warning("Voxel keys empty.")
        return {}
        
    if not rmsf_domain_ids:
        logger.warning("RMSF IDs empty.")
        return {}
    
    # Create sets for efficient lookups
    v_set = set(voxel_domain_keys)
    r_set = set(rmsf_domain_ids)
    
    mapping = {}
    matches = {'exact': 0, 'base': 0}
    
    # Create mapping from base domain IDs to full IDs
    r_base_map = {r.split('_')[0]: r for r in rmsf_domain_ids}
    
    # Match each HDF5 key to RMSF ID
    for h_key in v_set:
        # Try exact match first
        if h_key in r_set:
            mapping[h_key] = h_key
            matches['exact'] += 1
            continue
            
        # Try base domain match
        base_h = h_key.split('_')[0]
        if base_h in r_set:
            mapping[h_key] = base_h
            matches['base'] += 1
        elif base_h in r_base_map:
            mapping[h_key] = r_base_map[base_h]
            matches['base'] += 1
    
    # Log mapping statistics
    total = len(mapping)
    cov = (total/len(voxel_domain_keys))*100 if voxel_domain_keys else 0
    logger.info(f"Domain mapping: {total}/{len(voxel_domain_keys)} keys ({cov:.1f}%)")
    logger.info(f"  Matches: Exact={matches['exact']}, Base={matches['base']}")
    
    return mapping

def load_list_from_file(file_path: str) -> List[str]:
    """
    Load a list of strings from a text file, one item per line.
    
    Args:
        file_path: Path to text file
        
    Returns:
        List of strings from file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}. Returning empty list.")
        return []
        
    try:
        with open(file_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            items = [line.strip() for line in f if line.strip()]
        
        logger.debug(f"Loaded {len(items)} items from: {file_path}")
        return items
    except Exception as e:
        logger.error(f"Failed to load list from {file_path}: {e}")
        return []  # Return empty list on error