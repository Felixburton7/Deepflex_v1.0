# === FILE: data_processor.py ===
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import random
from typing import Dict, List, Tuple, Set, Optional, Any
import logging
from tqdm import tqdm # <--- FIXED: ADDED IMPORT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# Standard 1-letter amino acid codes from 3-letter codes
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Common Histidine variants
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'HID': 'H', 'HIE': 'H',
}
# Example extended map if non_standard_handling == 'map'
# EXTENDED_AA_MAP = {**AA_MAP, 'MSE': 'M', 'SEP': 'S', 'TPO': 'T'}

def load_rmsf_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Load RMSF data from CSV file, selecting only necessary columns."""
    required_cols = ['domain_id', 'resid', 'resname', 'rmsf_320']
    if not os.path.exists(csv_path):
        logger.error(f"RMSF data file not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, usecols=required_cols)
        logger.info(f"Loaded {len(df)} rows from {csv_path} with columns: {df.columns.tolist()}")

        if not all(col in df.columns for col in required_cols):
             logger.error(f"CSV missing one or more required columns ({required_cols}). Found: {df.columns.tolist()}")
             return None
        if df[required_cols].isnull().values.any():
            logger.warning(f"NaN values found in required columns. Dropping rows with NaNs in {required_cols}.")
            nan_counts = df[required_cols].isnull().sum()
            logger.warning(f"NaN counts per column:\n{nan_counts[nan_counts > 0]}")
            df.dropna(subset=required_cols, inplace=True)
            logger.info(f"Remaining rows after dropping NaNs: {len(df)}")
            if df.empty:
                logger.error("DataFrame is empty after dropping NaN values.")
                return None

        df['domain_id'] = df['domain_id'].astype(str)
        # Ensure 'resid' is numeric and handle potential errors, then convert to nullable Int64
        df['resid'] = pd.to_numeric(df['resid'], errors='coerce')
        df.dropna(subset=['resid'], inplace=True) # Drop rows where 'resid' couldn't be converted
        df['resid'] = df['resid'].astype('Int64')

        df['resname'] = df['resname'].astype(str).str.upper().str.strip()
        # Ensure 'rmsf_320' is numeric and handle potential errors
        df['rmsf_320'] = pd.to_numeric(df['rmsf_320'], errors='coerce')
        df.dropna(subset=['rmsf_320'], inplace=True) # Drop rows where 'rmsf_320' couldn't be converted

        logger.info(f"Remaining rows after type conversion checks: {len(df)}")
        if df.empty:
             logger.error("DataFrame empty after type conversion checks.")
             return None

        return df
    except ValueError as e:
        logger.error(f"ValueError during CSV loading or type conversion ({csv_path}): {e}. Check columns/types.", exc_info=True)
        return None
    except KeyError as e:
        logger.error(f"KeyError likely due to missing column during CSV loading ({csv_path}): {e}. Required: {required_cols}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error loading or cleaning CSV file {csv_path}: {e}", exc_info=True)
        return None


def extract_sequences_and_rmsf(df: pd.DataFrame, non_standard_handling: str = 'ignore') -> Dict[str, Dict]:
    """
    Group data by domain_id and extract amino acid sequence and RMSF values.
    Ensures 1:1 correspondence between sequence characters and RMSF values based on sorted 'resid'.
    """
    processed_data = {}
    processed_count = 0
    skipped_domains = set()
    non_standard_residues_found = defaultdict(int)
    alignment_issues = 0

    if 'domain_id' not in df.columns:
        logger.error("DataFrame missing 'domain_id' column for grouping.")
        return {}

    # Use progress bar if DataFrame is large
    unique_domains = df['domain_id'].unique()
    # Use tqdm for the iterator if there are many domains
    domain_iterator = tqdm(unique_domains, desc="Processing Domains", leave=False, ncols=100) if len(unique_domains) > 500 else unique_domains

    for domain_id in domain_iterator:
        # Filter and sort the group for the current domain_id
        group_df = df[df['domain_id'] == domain_id].sort_values('resid')

        if group_df.empty:
             logger.warning(f"No data found for domain_id '{domain_id}' after initial loading/filtering. Skipping.")
             skipped_domains.add(domain_id)
             continue

        sequence = ''
        rmsf_values = []
        valid_domain = True
        last_resid = 0 # Track last residue ID processed

        for _, row in group_df.iterrows():
            residue_name = row['resname']
            # Check for NA/None after potential coercions, though dropna should handle this
            current_resid = row['resid']
            if pd.isna(current_resid):
                 logger.warning(f"Domain {domain_id}: Encountered NA residue ID. Skipping row.")
                 continue

            # Basic check for sequentiality (optional, can be noisy if gaps are expected)
            # if current_resid != last_resid + 1 and last_resid != 0:
            #     logger.debug(f"Domain {domain_id}: Potential gap or non-sequential resid {last_resid} -> {current_resid}.")

            if residue_name in AA_MAP:
                sequence += AA_MAP[residue_name]
                rmsf_values.append(row['rmsf_320'])
                last_resid = current_resid
            else:
                # Handle non-standard residues
                non_standard_residues_found[residue_name] += 1
                if non_standard_handling == 'discard':
                    # Log only once per domain for discard
                    if domain_id not in skipped_domains:
                         logger.warning(f"Domain {domain_id}: Non-standard residue '{residue_name}'. Discarding domain.")
                         skipped_domains.add(domain_id)
                    valid_domain = False
                    break # Stop processing this domain
                elif non_standard_handling == 'ignore':
                    # logger.debug(f"Domain {domain_id}: Ignoring non-standard residue '{residue_name}' at resid {current_resid}.")
                    pass # Skip row, don't add to sequence or RMSF
                # Add 'map' logic here if needed based on config
                # elif non_standard_handling == 'map' and residue_name in EXTENDED_AA_MAP: ...
                else:
                    logger.warning(f"Domain {domain_id}: Unknown non-standard residue '{residue_name}' with handling='{non_standard_handling}'. Ignoring residue.")
                    pass # Treat as ignore

        if not valid_domain:
            continue # Go to next domain_id

        if sequence: # Only proceed if a sequence was built
            if len(sequence) == len(rmsf_values):
                processed_data[domain_id] = {
                    'sequence': sequence,
                    'rmsf': np.array(rmsf_values, dtype=np.float32)
                }
                processed_count += 1
            else:
                # This indicates a logic error in the loop if it occurs
                logger.error(f"CRITICAL ALIGNMENT ISSUE for domain {domain_id}: Seq len {len(sequence)} != RMSF count {len(rmsf_values)}. Skipping.")
                skipped_domains.add(domain_id)
                alignment_issues += 1
        # Log only if not already skipped and resulted in empty sequence (e.g., only non-standard residues)
        elif domain_id not in skipped_domains:
             logger.warning(f"Domain {domain_id} resulted in empty sequence after processing. Skipping.")
             skipped_domains.add(domain_id)

    logger.info(f"Finished processing. Successfully extracted data for {processed_count} domains.")
    if skipped_domains: logger.warning(f"Skipped {len(skipped_domains)} domains due to non-standard residues, empty data, or errors.")
    if non_standard_residues_found: logger.warning(f"Encountered non-standard residues (counts): {dict(non_standard_residues_found)}")
    if alignment_issues > 0: logger.error(f"Encountered {alignment_issues} critical sequence-RMSF alignment issues.")

    return processed_data

def extract_topology(domain_id: str) -> str:
    """Extract topology identifier (first 4 chars of domain_id)."""
    if isinstance(domain_id, str) and len(domain_id) >= 4:
        return domain_id[:4].upper()
    else:
        logger.warning(f"Could not extract topology from short/invalid domain_id: {domain_id}. Using full ID.")
        return domain_id

def split_by_topology(data: Dict[str, Dict], train_ratio=0.7, val_ratio=0.15, seed=42) -> Tuple[Dict, Dict, Dict]:
    """Split data by topology to ensure no topology overlap between splits."""
    if not data: return {}, {}, {}
    random.seed(seed)
    logger.info(f"Splitting {len(data)} domains by topology using seed {seed}")

    topology_groups = defaultdict(list)
    for domain_id in data.keys():
        topology = extract_topology(domain_id)
        topology_groups[topology].append(domain_id)

    num_topologies = len(topology_groups)
    logger.info(f"Found {num_topologies} unique topologies.")
    if num_topologies < 3: logger.warning(f"Very few topologies ({num_topologies}). Splits might be uneven or empty.")

    topologies = list(topology_groups.keys()); random.shuffle(topologies)
    n = num_topologies
    train_idx = int(n * train_ratio); val_idx = train_idx + int(n * val_ratio)

    # Adjust indices to prevent empty splits if possible
    if n >= 3: # Need at least 3 topologies for meaningful split
        if train_idx == 0: train_idx = 1 # Give at least one to train
        if val_idx == train_idx: val_idx = train_idx + 1 # Give at least one to val
        if val_idx >= n: # Prevent val taking everything intended for test
            val_idx = n - 1 # Leave at least one for test
            if train_idx >= val_idx: # Ensure train is smaller than val start
                train_idx = max(0, val_idx - 1)

    # Handle edge cases with N < 3
    elif n == 2:
        train_idx = 1
        val_idx = 2 # Train gets 1, Test gets 1, Val is empty
        logger.warning("Only 2 topologies found. Assigning 1 to Train, 1 to Test, 0 to Val.")
    elif n == 1:
        train_idx = 1 # All go to train
        val_idx = 1
        logger.warning("Only 1 topology found. Assigning all to Train split.")
    # If n=0, indices remain 0

    train_topologies, val_topologies, test_topologies = set(topologies[:train_idx]), set(topologies[train_idx:val_idx]), set(topologies[val_idx:])
    logger.info(f"Split topology counts: Train={len(train_topologies)}, Val={len(val_topologies)}, Test={len(test_topologies)}")

    train_data, val_data, test_data = {}, {}, {}
    assigned_count = 0
    unassigned_topos = set()
    for topology, domain_ids in topology_groups.items():
        data_subset = {did: data[did] for did in domain_ids}
        assigned = False
        if topology in train_topologies: train_data.update(data_subset); assigned = True
        elif topology in val_topologies: val_data.update(data_subset); assigned = True
        elif topology in test_topologies: test_data.update(data_subset); assigned = True

        if not assigned:
             logger.error(f"Topology {topology} was not assigned to any split! Indices: train={train_idx}, val={val_idx}")
             unassigned_topos.add(topology)
        else:
             assigned_count += len(data_subset)

    logger.info(f"Split domain counts: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    if assigned_count != len(data): logger.warning(f"Domain count mismatch after split ({assigned_count} assigned vs {len(data)} input). Unassigned topos: {unassigned_topos}")
    return train_data, val_data, test_data

def save_split_data(data: Dict, output_dir: str, split_name: str):
    """Save split data (domain list, FASTA, RMSF numpy) to disk."""
    if not data: logger.warning(f"No data for split '{split_name}'. Skipping save."); return
    os.makedirs(output_dir, exist_ok=True)
    domain_ids = sorted(list(data.keys()))

    # Domain list
    domain_list_path = os.path.join(output_dir, f"{split_name}_domains.txt")
    try:
        with open(domain_list_path, 'w') as f: f.write("\n".join(domain_ids))
        logger.info(f"Saved {len(domain_ids)} domain IDs to {domain_list_path}")
    except IOError as e: logger.error(f"Error writing domain list {domain_list_path}: {e}")

    # FASTA
    fasta_path = os.path.join(output_dir, f"{split_name}_sequences.fasta")
    fasta_count = 0
    try:
        with open(fasta_path, 'w') as f:
            for domain_id in domain_ids:
                if 'sequence' in data[domain_id] and data[domain_id]['sequence']:
                    f.write(f">{domain_id}\n{data[domain_id]['sequence']}\n"); fasta_count += 1
        logger.info(f"Saved {fasta_count} sequences to {fasta_path}")
    except IOError as e: logger.error(f"Error writing FASTA file {fasta_path}: {e}")

    # RMSF NPY
    rmsf_path = os.path.join(output_dir, f"{split_name}_rmsf.npy")
    rmsf_data_to_save = {did: data[did]['rmsf'].astype(np.float32) for did in domain_ids if 'rmsf' in data[did] and isinstance(data[did]['rmsf'], np.ndarray) and data[did]['rmsf'].size > 0}
    if rmsf_data_to_save:
        try:
            np.save(rmsf_path, rmsf_data_to_save, allow_pickle=True)
            logger.info(f"Saved RMSF data for {len(rmsf_data_to_save)} domains to {rmsf_path}")
        except Exception as e: logger.error(f"Error saving RMSF numpy file {rmsf_path}: {e}")
    else: logger.warning(f"No valid RMSF data to save for split {split_name}.")


def process_data(config: Dict[str, Any]) -> bool:
    """Main function to process RMSF data based on config."""
    data_config = config['data']
    csv_path = data_config['raw_csv_path']
    output_dir = data_config['data_dir']
    # Ratios and seed usually related to training setup
    train_config = config.get('training', {})
    # Use get with defaults for ratios if not present
    train_ratio = float(train_config.get('train_ratio', 0.7))
    val_ratio = float(train_config.get('val_ratio', 0.15))
    seed = int(train_config.get('seed', 42))
    non_standard_handling = data_config.get('non_standard_residue_handling', 'ignore')

    logger.info("Starting data processing pipeline...")
    logger.info(f"Input CSV: {csv_path}, Output Dir: {output_dir}")
    logger.info(f"Splits: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={max(0, 1 - train_ratio - val_ratio):.2f}") # Ensure test isn't negative
    logger.info(f"Non-standard handling: {non_standard_handling}, Seed: {seed}")

    try:
        df = load_rmsf_data(csv_path)
        if df is None or df.empty:
             logger.error("Failed to load or clean raw CSV data.")
             return False
        data = extract_sequences_and_rmsf(df, non_standard_handling)
        if not data:
             logger.error("No valid sequence/RMSF data extracted.")
             return False
        train_data, val_data, test_data = split_by_topology(data, train_ratio, val_ratio, seed)
        # Check if splits have data before saving
        if not train_data: logger.warning("Training split is empty after processing!")
        if not val_data: logger.warning("Validation split is empty after processing!")
        if not test_data: logger.warning("Test split is empty after processing!")
        save_split_data(train_data, output_dir, 'train')
        save_split_data(val_data, output_dir, 'val')
        save_split_data(test_data, output_dir, 'test')
        logger.info("Data processing completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        return False

# Command-line interface
if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Process RMSF data based on config.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file.')
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: logger.error(f"Error loading config {args.config}: {e}"); exit(1)
    if not process_data(config): exit(1)