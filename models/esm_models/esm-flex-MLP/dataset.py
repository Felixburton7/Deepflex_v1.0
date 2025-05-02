import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMSFDataset(Dataset):
    """
    PyTorch Dataset for RMSF prediction from protein sequences.

    Handles loading and providing access to protein sequences and their
    corresponding RMSF values for training or evaluation.
    """

    def __init__(self,
                 domain_ids: List[str],
                 sequences: Dict[str, str],
                 rmsf_values: Dict[str, np.ndarray]):
        """
        Initialize the RMSF dataset.

        Args:
            domain_ids: Ordered list of domain IDs for this dataset split.
            sequences: Dictionary mapping domain IDs to amino acid sequences (strings).
            rmsf_values: Dictionary mapping domain IDs to RMSF values (NumPy arrays).
        """
        self.domain_ids = domain_ids
        self.sequences = sequences
        self.rmsf_values = rmsf_values # Stored as NumPy arrays

        # Data Consistency Check (Optional but recommended)
        valid_domain_ids = []
        removed_count = 0
        for did in self.domain_ids:
            if did in self.sequences and did in self.rmsf_values:
                # Check for basic length consistency if possible (can be complex with processing steps)
                # if len(self.sequences[did]) != len(self.rmsf_values[did]):
                #     logger.warning(f"Initial length mismatch in dataset for {did}: Seq={len(self.sequences[did])}, RMSF={len(self.rmsf_values[did])}. Keeping for now.")
                     # Decide whether to remove here or let downstream handle it. Usually downstream is better.
                valid_domain_ids.append(did)
            else:
                logger.warning(f"Domain ID {did} missing sequence or RMSF value. Removing from dataset.")
                removed_count += 1

        if removed_count > 0:
             logger.info(f"Removed {removed_count} domain IDs due to missing data.")
             self.domain_ids = valid_domain_ids


        # Calculate and log dataset statistics
        if self.domain_ids:
            seq_lengths = [len(sequences[did]) for did in self.domain_ids]
            rmsf_lengths = [len(rmsf_values[did]) for did in self.domain_ids] # Get RMSF lengths too
            logger.info(f"Dataset created with {len(self.domain_ids)} proteins")
            logger.info(f"  Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, " +
                        f"mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.1f}")
            logger.info(f"  RMSF length stats:     min={min(rmsf_lengths)}, max={max(rmsf_lengths)}, " +
                        f"mean={np.mean(rmsf_lengths):.1f}, median={np.median(rmsf_lengths):.1f}")
            # Check for major discrepancies between seq and rmsf lengths stats
            if np.mean(seq_lengths) != np.mean(rmsf_lengths):
                 logger.warning("Mean sequence length differs from mean RMSF length. Check data processing.")
        else:
            logger.warning("Dataset created with 0 proteins.")

    def __len__(self) -> int:
        """Return the number of proteins in the dataset."""
        return len(self.domain_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a protein sequence and its RMSF values by index.

        Args:
            idx: Index of the protein in the `self.domain_ids` list.

        Returns:
            Dictionary containing:
              - 'domain_id': The domain identifier (string).
              - 'sequence': The amino acid sequence (string).
              - 'rmsf': The RMSF values (NumPy array of float32).
        """
        if idx < 0 or idx >= len(self.domain_ids):
             raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.domain_ids)}")

        domain_id = self.domain_ids[idx]
        sequence = self.sequences[domain_id]
        rmsf = self.rmsf_values[domain_id] # Already a numpy array

        # Ensure RMSF is float32 for consistency with model expectations
        if rmsf.dtype != np.float32:
             rmsf = rmsf.astype(np.float32)

        return {
            'domain_id': domain_id,
            'sequence': sequence,
            'rmsf': rmsf
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
                if not line: continue # Skip empty lines
                if line.startswith('>'):
                    if current_id is not None:
                        sequences[current_id] = current_seq
                    current_id = line[1:].split()[0] # Use ID before first space
                    current_seq = ""
                else:
                    # Validate sequence characters (optional but good)
                    # if not all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in line.upper()):
                    #     logger.warning(f"Non-standard characters found in sequence for {current_id} in {fasta_path}")
                    current_seq += line.upper() # Store sequences as uppercase
            # Add the last sequence
            if current_id is not None:
                sequences[current_id] = current_seq
    except FileNotFoundError:
        logger.error(f"FASTA file not found: {fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading FASTA file {fasta_path}: {e}")
        raise
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return sequences


def load_split_data(data_dir: str, split: str) -> Tuple[List[str], Dict[str, str], Dict[str, np.ndarray]]:
    """
    Load data (domain IDs, sequences, RMSF values) for a specific split.

    Args:
        data_dir: Directory containing the processed data files
                  (e.g., 'data/processed').
        split: Split name ('train', 'val', or 'test').

    Returns:
        Tuple of (domain_ids, sequences, rmsf_values).
        Returns ([], {}, {}) if data loading fails.
    """
    logger.info(f"Loading {split} data from directory: {data_dir}")

    # --- Load domain IDs ---
    domain_ids_path = os.path.join(data_dir, f"{split}_domains.txt")
    domain_ids = []
    try:
        with open(domain_ids_path, 'r') as f:
            domain_ids = [line.strip() for line in f if line.strip()]
        if not domain_ids:
             logger.warning(f"Domain ID file is empty or not found: {domain_ids_path}")
             # return [], {}, {} # Decide if this is a fatal error
        logger.info(f"Loaded {len(domain_ids)} domain IDs from {domain_ids_path}")
    except FileNotFoundError:
        logger.error(f"Domain ID file not found: {domain_ids_path}")
        return [], {}, {} # Cannot proceed without domain IDs
    except Exception as e:
        logger.error(f"Error reading domain ID file {domain_ids_path}: {e}")
        return [], {}, {}


    # --- Load sequences ---
    sequences_path = os.path.join(data_dir, f"{split}_sequences.fasta")
    sequences = {}
    try:
        sequences = load_sequences_from_fasta(sequences_path)
        if not sequences:
             logger.warning(f"No sequences loaded from {sequences_path}")
             # Decide if this is fatal or if we can proceed with missing sequences
    except FileNotFoundError:
         logger.warning(f"Sequence file not found: {sequences_path}. Proceeding without sequences for checks.")
    except Exception as e:
         logger.error(f"Failed to load sequences from {sequences_path}: {e}")
         # Decide if this is fatal


    # --- Load RMSF values ---
    rmsf_path = os.path.join(data_dir, f"{split}_rmsf.npy")
    rmsf_data = {}
    try:
        # Need allow_pickle=True because we saved a dictionary
        loaded_rmsf = np.load(rmsf_path, allow_pickle=True).item()
        # Ensure keys are strings and values are numpy arrays
        rmsf_data = {str(k): np.array(v, dtype=np.float32) for k, v in loaded_rmsf.items()}
        logger.info(f"Loaded RMSF data for {len(rmsf_data)} domains from {rmsf_path}")
    except FileNotFoundError:
        logger.error(f"RMSF data file not found: {rmsf_path}")
        # Decide if this is fatal. Often it is for training/validation.
        return [], {}, {}
    except Exception as e:
        logger.error(f"Error loading or processing RMSF data from {rmsf_path}: {e}")
        return [], {}, {}

    # --- Verify data consistency ---
    logger.info("Verifying data consistency...")
    original_domain_count = len(domain_ids)
    valid_domain_ids = []
    missing_seq_count = 0
    missing_rmsf_count = 0
    length_mismatches = 0

    for did in domain_ids:
        has_seq = did in sequences
        has_rmsf = did in rmsf_data

        if has_seq and has_rmsf:
            # Check sequence-RMSF length consistency
            seq_len = len(sequences[did])
            rmsf_len = len(rmsf_data[did])
            if seq_len == rmsf_len:
                valid_domain_ids.append(did)
            else:
                length_mismatches += 1
                logger.debug(f"Length mismatch for {did}: sequence={seq_len}, RMSF={rmsf_len}. Removing.")
        else:
            if not has_seq:
                missing_seq_count += 1
                logger.debug(f"Missing sequence for domain ID: {did}")
            if not has_rmsf:
                missing_rmsf_count += 1
                logger.debug(f"Missing RMSF data for domain ID: {did}")

    logger.info(f"Initial domain IDs: {original_domain_count}")
    if missing_seq_count > 0:
         logger.warning(f"Missing sequences for {missing_seq_count} domain IDs.")
    if missing_rmsf_count > 0:
         logger.warning(f"Missing RMSF data for {missing_rmsf_count} domain IDs.")
    if length_mismatches > 0:
        logger.warning(f"Found {length_mismatches} domains with sequence-RMSF length mismatches.")

    final_domain_count = len(valid_domain_ids)
    if final_domain_count != original_domain_count:
        logger.info(f"Removed {original_domain_count - final_domain_count} domains due to inconsistencies.")
        logger.info(f"Final number of valid domains for split '{split}': {final_domain_count}")

    # Filter sequences and RMSF data to only include valid domains
    final_sequences = {did: sequences[did] for did in valid_domain_ids if did in sequences}
    final_rmsf_values = {did: rmsf_data[did] for did in valid_domain_ids if did in rmsf_data}


    if final_domain_count == 0:
         logger.error(f"No valid, consistent data found for split '{split}'. Please check the processed data files in {data_dir}.")


    return valid_domain_ids, final_sequences, final_rmsf_values


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for the DataLoader.

    Batches together domain IDs, sequences, and converts RMSF NumPy arrays
    into PyTorch tensors. Padding is NOT done here; it should be handled
    by the model's tokenizer or forward method.

    Args:
        batch: A list of dictionaries, where each dictionary is an output
               from RMSFDataset.__getitem__.

    Returns:
        A dictionary containing batched data:
          - 'domain_ids': List of domain ID strings.
          - 'sequences': List of amino acid sequence strings.
          - 'rmsf_values': List of RMSF value tensors (torch.float32).
    """
    domain_ids = [item['domain_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    # Convert RMSF numpy arrays to tensors
    rmsf_values = [torch.tensor(item['rmsf'], dtype=torch.float32) for item in batch]

    return {
        'domain_ids': domain_ids,
        'sequences': sequences,
        'rmsf_values': rmsf_values # List of Tensors
    }


def create_length_batched_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    max_length: Optional[int] = None,
    length_bucket_size: int = 50,
    num_workers: int = 0 # Default to 0 for simplicity, increase if I/O bound
) -> Optional[DataLoader]:
    """
    Creates a PyTorch DataLoader with length-based batching strategy.

    Groups sequences of similar lengths together into batches to minimize
    padding overhead during model processing (especially with transformers).

    Args:
        data_dir: Directory containing the processed data splits.
        split: Split name ('train', 'val', or 'test').
        batch_size: The target number of sequences per batch.
        shuffle: Whether to shuffle the data (primarily the order of length buckets
                 and samples within buckets). Recommended for training.
        max_length: Optional maximum sequence length. Sequences longer than this
                    will be filtered out.
        length_bucket_size: The size of length ranges used for grouping sequences.
                           Smaller values mean tighter length grouping but potentially
                           more uneven batch sizes.
        num_workers: Number of worker processes for data loading.

    Returns:
        A PyTorch DataLoader instance, or None if data loading fails.
    """
    # 1. Load data for the specified split
    domain_ids, sequences, rmsf_values = load_split_data(data_dir, split)

    if not domain_ids:
        logger.error(f"Failed to load data for split '{split}'. Cannot create DataLoader.")
        return None

    # 2. Filter by max length if specified
    if max_length is not None:
        original_count = len(domain_ids)
        filtered_domain_ids = [
            did for did in domain_ids if len(sequences[did]) <= max_length
        ]
        filtered_count = len(filtered_domain_ids)
        if filtered_count < original_count:
            logger.info(f"Filtered out {original_count - filtered_count} sequences " +
                        f"longer than {max_length} residues for split '{split}'.")
            domain_ids = filtered_domain_ids
            # Update sequences and rmsf_values dictionaries if filtering occurred
            sequences = {did: sequences[did] for did in domain_ids}
            rmsf_values = {did: rmsf_values[did] for did in domain_ids}

        if not domain_ids:
             logger.warning(f"No sequences remaining after filtering by max_length={max_length} for split '{split}'.")
             return None # Cannot create dataloader if no sequences left


    # 3. Group domain IDs by length buckets
    length_buckets = defaultdict(list)
    for did in domain_ids:
        seq_len = len(sequences[did])
        bucket_idx = seq_len // length_bucket_size
        length_buckets[bucket_idx].append(did)

    logger.info(f"Grouped {len(domain_ids)} sequences into {len(length_buckets)} length buckets.")

    # 4. Create batches within buckets
    all_batches = []
    # Sort buckets by index (approx length) to process shorter sequences first (can help memory)
    sorted_bucket_indices = sorted(length_buckets.keys())

    for bucket_idx in sorted_bucket_indices:
        bucket_domain_ids = length_buckets[bucket_idx]
        if shuffle:
            random.shuffle(bucket_domain_ids) # Shuffle within the bucket

        # Create mini-batches from this bucket
        for i in range(0, len(bucket_domain_ids), batch_size):
            batch_domain_ids = bucket_domain_ids[i : i + batch_size]
            all_batches.append(batch_domain_ids) # Add the list of domain IDs for this batch

    # 5. Shuffle the order of batches (optional but recommended for training)
    if shuffle:
        random.shuffle(all_batches)

    # 6. Flatten the batches to get the final ordered list of domain IDs for the epoch
    ordered_domain_ids = [did for batch in all_batches for did in batch]

    # 7. Create the Dataset with the final order
    # We need to pass the original full sequence/rmsf dicts, but use the ordered IDs
    dataset = RMSFDataset(ordered_domain_ids, sequences, rmsf_values)

    # 8. Create the DataLoader
    logger.info(f"Creating DataLoader for {len(dataset)} samples for split '{split}' with batch size {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT: We already handled shuffling by length batching
        collate_fn=collate_fn, # Use our custom collate function
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), # Helps speed up CPU to GPU transfer
        drop_last=False # Keep all sequences, even if last batch is smaller
    )

# Example Usage (if script is run directly)
if __name__ == "__main__":
    logger.info("Testing DataLoader creation...")
    # Create dummy data for testing
    dummy_data_dir = "data/processed_dummy"
    os.makedirs(dummy_data_dir, exist_ok=True)

    dummy_domains = [f"D{i:03d}" for i in range(100)]
    dummy_sequences = {}
    dummy_rmsf = {}
    for i, did in enumerate(dummy_domains):
        length = random.randint(50, 250)
        dummy_sequences[did] = "A" * length
        dummy_rmsf[did] = np.random.rand(length).astype(np.float32) * 2.0

    # Save dummy data
    with open(os.path.join(dummy_data_dir, "train_domains.txt"), "w") as f:
        f.write("\n".join(dummy_domains))
    with open(os.path.join(dummy_data_dir, "train_sequences.fasta"), "w") as f:
        for did, seq in dummy_sequences.items():
            f.write(f">{did}\n{seq}\n")
    np.save(os.path.join(dummy_data_dir, "train_rmsf.npy"), dummy_rmsf)

    # Test dataloader creation
    train_loader = create_length_batched_dataloader(
        data_dir=dummy_data_dir,
        split='train',
        batch_size=16,
        shuffle=True,
        max_length=200,
        length_bucket_size=25
    )

    if train_loader:
        logger.info("DataLoader created successfully. Iterating through a few batches...")
        batch_count = 0
        max_batches_to_show = 3
        for i, batch in enumerate(train_loader):
            if i >= max_batches_to_show: break
            logger.info(f"Batch {i+1}:")
            logger.info(f"  Domain IDs: {batch['domain_ids']}")
            logger.info(f"  Number of sequences: {len(batch['sequences'])}")
            logger.info(f"  Sequence lengths: {[len(s) for s in batch['sequences']]}")
            logger.info(f"  RMSF tensor shapes: {[t.shape for t in batch['rmsf_values']]}")
            batch_count += 1
        logger.info(f"Iterated through {batch_count} batches.")
    else:
        logger.error("Failed to create DataLoader.")

    # Clean up dummy data
    # import shutil
    # shutil.rmtree(dummy_data_dir)
    # logger.info(f"Cleaned up dummy data directory: {dummy_data_dir}")

