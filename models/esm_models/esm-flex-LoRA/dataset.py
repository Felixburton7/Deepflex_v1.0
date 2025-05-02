import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import random
import logging
from typing import List, Dict, Tuple, Optional, Any, Iterator
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

class RMSFDataset(Dataset):
    """PyTorch Dataset for RMSF prediction from pre-processed files."""
    def __init__(self, domain_ids: List[str], sequences: Dict[str, str], rmsf_values: Dict[str, np.ndarray]):
        self.domain_ids = domain_ids
        self.sequences = sequences
        self.rmsf_values = rmsf_values
        self._validate_data()
        logger.info(f"RMSFDataset initialized with {len(self.domain_ids)} valid domain IDs.")

    def _validate_data(self):
        """Ensures all domain_ids have corresponding, non-empty, length-matched sequence and rmsf."""
        valid_ids = []
        original_count = len(self.domain_ids)
        for did in self.domain_ids:
            seq = self.sequences.get(did)
            rmsf = self.rmsf_values.get(did)
            if seq and isinstance(rmsf, np.ndarray) and rmsf.size > 0:
                if len(seq) == len(rmsf):
                    valid_ids.append(did)
                else:
                    logger.warning(f"Dataset validation: Length mismatch for {did} (Seq={len(seq)}, RMSF={len(rmsf)}). Excluding.")
            else:
                 logger.warning(f"Dataset validation: Missing/empty seq or RMSF for {did}. Excluding.")
        removed_count = original_count - len(valid_ids)
        if removed_count > 0:
             logger.warning(f"Removed {removed_count} invalid entries during dataset validation.")
        self.domain_ids = valid_ids
        if not self.domain_ids:
             raise ValueError("No valid data remains after dataset validation.")

    def __len__(self) -> int:
        return len(self.domain_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < len(self.domain_ids)):
             raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.domain_ids)}")
        domain_id = self.domain_ids[idx]
        sequence = self.sequences[domain_id]
        rmsf = self.rmsf_values[domain_id] # Already float32 numpy array
        return {'domain_id': domain_id, 'sequence': sequence, 'rmsf': rmsf, 'length': len(sequence)}

def load_sequences_from_fasta(fasta_path: str) -> Dict[str, str]:
    """Loads sequences from a FASTA file."""
    sequences = {}
    try:
        with open(fasta_path, 'r') as f:
            current_id, current_seq = None, ""
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id: sequences[current_id] = current_seq
                    current_id = line[1:].split()[0]
                    current_seq = ""
                elif current_id: # Ensure we have an ID before adding sequence parts
                    current_seq += line.upper().replace("-","").replace(".","") # Basic cleaning
            if current_id: sequences[current_id] = current_seq # Add last sequence
    except FileNotFoundError: logger.error(f"FASTA not found: {fasta_path}"); raise
    except Exception as e: logger.error(f"Error reading FASTA {fasta_path}: {e}"); raise
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return sequences

def load_split_data(data_dir: str, split: str) -> Optional[Tuple[List[str], Dict[str, str], Dict[str, np.ndarray]]]:
    """Load domain IDs, sequences, and RMSF values for a specific split."""
    logger.info(f"Loading {split} data from {data_dir}")
    domain_ids_path = os.path.join(data_dir, f"{split}_domains.txt")
    sequences_path = os.path.join(data_dir, f"{split}_sequences.fasta")
    rmsf_path = os.path.join(data_dir, f"{split}_rmsf.npy")

    if not all(os.path.exists(p) for p in [domain_ids_path, sequences_path, rmsf_path]):
        logger.error(f"One or more required files missing for split '{split}' in {data_dir}.")
        # Check which specifically are missing
        if not os.path.exists(domain_ids_path): logger.error(f"Missing: {domain_ids_path}")
        if not os.path.exists(sequences_path): logger.error(f"Missing: {sequences_path}")
        if not os.path.exists(rmsf_path): logger.error(f"Missing: {rmsf_path}")
        return None

    try:
        with open(domain_ids_path, 'r') as f: domain_ids = [line.strip() for line in f if line.strip()]
        if not domain_ids: logger.warning(f"Domain ID file empty: {domain_ids_path}")
        sequences = load_sequences_from_fasta(sequences_path)
        rmsf_data_loaded = np.load(rmsf_path, allow_pickle=True).item()
        # Ensure RMSF values are float32 numpy arrays
        rmsf_data = {k: np.array(v, dtype=np.float32) for k, v in rmsf_data_loaded.items()}

        # Final consistency filter: Keep only domain IDs present in all three loaded structures
        final_domain_ids = [did for did in domain_ids if did in sequences and did in rmsf_data]
        removed_count = len(domain_ids) - len(final_domain_ids)
        if removed_count > 0:
             logger.warning(f"Removed {removed_count} domain IDs during final load consistency check for split '{split}'.")

        if not final_domain_ids:
             logger.error(f"No consistent domain IDs found for split '{split}' across all files.")
             return None

        # Filter the dictionaries
        final_sequences = {did: sequences[did] for did in final_domain_ids}
        final_rmsf_values = {did: rmsf_data[did] for did in final_domain_ids}

        logger.info(f"Successfully loaded and verified {len(final_domain_ids)} entries for split '{split}'.")
        return final_domain_ids, final_sequences, final_rmsf_values

    except Exception as e:
        logger.error(f"Failed to load data for split '{split}': {e}", exc_info=True)
        return None

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate: batches sequences as strings, RMSF as tensors, adds lengths."""
    domain_ids = [item['domain_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    lengths = [item['length'] for item in batch]
    rmsf_values = [torch.tensor(item['rmsf'], dtype=torch.float32) for item in batch]
    return {'domain_ids': domain_ids, 'sequences': sequences, 'rmsf_values': rmsf_values, 'lengths': lengths}

class LengthBasedBatchSampler(Sampler[List[int]]):
    """Sampler yielding batches of indices grouped by sequence length."""
    def __init__(self, dataset: RMSFDataset, batch_size: int, length_bucket_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_bucket_size = length_bucket_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        self.batches = self._create_batches()
        self.num_batches = len(self.batches)
        logger.info(f"LengthBasedBatchSampler: Grouped {self.num_samples} samples into {self.num_batches} batches.")

    def _create_batches(self) -> List[List[int]]:
        buckets = defaultdict(list)
        for i in range(self.num_samples):
            length = self.dataset[i]['length'] # Get length directly from dataset item
            bucket_idx = length // self.length_bucket_size
            buckets[bucket_idx].append(i)

        all_batches = []
        bucket_indices = sorted(buckets.keys()) if not self.shuffle else list(buckets.keys())
        if self.shuffle: random.shuffle(bucket_indices)

        for bucket_idx in bucket_indices:
            indices = buckets[bucket_idx]
            if self.shuffle: random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) == self.batch_size or (not self.drop_last and len(batch_indices) > 0):
                    all_batches.append(batch_indices)
        if self.shuffle: random.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle: self.batches = self._create_batches() # Re-shuffle every epoch
        yield from self.batches

    def __len__(self) -> int: return self.num_batches

def create_dataloader(data_dir: str, split: str, batch_size: int, shuffle: bool, max_length: Optional[int], length_bucket_size: int, num_workers: int, pin_memory: bool, drop_last: bool) -> Optional[DataLoader]:
    """Creates a PyTorch DataLoader, potentially using length-based batching."""
    loaded_data = load_split_data(data_dir, split)
    if loaded_data is None: return None
    domain_ids, sequences, rmsf_values = loaded_data
    if not domain_ids: return None

    dataset = RMSFDataset(domain_ids, sequences, rmsf_values)

    # Filter Dataset by max_length *before* creating sampler
    indices_to_keep = list(range(len(dataset)))
    if max_length is not None:
         original_count = len(dataset)
         indices_to_keep = [i for i in indices_to_keep if dataset[i]['length'] <= max_length]
         if len(indices_to_keep) < original_count:
             logger.info(f"Filtering '{split}' dataset by max_length={max_length}. Keeping {len(indices_to_keep)}/{original_count}.")
             dataset = torch.utils.data.Subset(dataset, indices_to_keep) # Use Subset for filtering
             if len(dataset) == 0: logger.warning(f"No samples left in '{split}' after max_length filter."); return None

    # Create Sampler and DataLoader
    use_length_sampler = (length_bucket_size > 0)
    if use_length_sampler:
        logger.info(f"Using LengthBasedBatchSampler for '{split}' (Bucket size: {length_bucket_size}).")
        # Pass the original dataset (if Subset) or the Subset object to the sampler
        data_source_for_sampler = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        # We need the *indices* relative to the original dataset for the sampler
        # This is complex with Subset. Let's simplify: apply filtering *after* getting lengths for bucketing.
        # Revert: Filter *before* creating dataset seems cleaner if done carefully. Let's assume RMSFDataset handles indices.
        # Re-revert: Subset is standard. Sampler needs access to lengths of the original data via the subset indices.

        # Let's try simpler approach first: Create sampler on the *potentially subsetted* dataset
        batch_sampler = LengthBasedBatchSampler(dataset, batch_size, length_bucket_size, shuffle, drop_last)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory and torch.cuda.is_available())
    else:
        logger.info(f"Using standard DataLoader for '{split}'.")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory and torch.cuda.is_available(), drop_last=drop_last)

    logger.info(f"DataLoader created for '{split}' with {len(dataset)} samples.")
    return dataloader

