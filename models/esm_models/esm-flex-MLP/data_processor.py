import pandas as pd
import numpy as np
from collections import defaultdict
import os
import random
from typing import Dict, List, Tuple, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_rmsf_data(csv_path: str) -> pd.DataFrame:
    """Load RMSF data from CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"RMSF data file not found: {csv_path}")
        raise FileNotFoundError(f"RMSF data file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path} with columns: {df.columns.tolist()}")
        # Basic validation
        required_cols = ['domain_id', 'resid', 'resname', 'rmsf_320']
        if not all(col in df.columns for col in required_cols):
             logger.warning(f"CSV missing one or more required columns ({required_cols}). Found: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {e}")
        raise

def group_by_domain(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group data by domain_id."""
    domains = {}
    if 'domain_id' not in df.columns or 'resid' not in df.columns:
        logger.error("DataFrame missing 'domain_id' or 'resid' column for grouping.")
        return domains

    for domain_id, group in df.groupby('domain_id'):
        domains[str(domain_id)] = group.sort_values('resid')  # Ensure domain_id is string, sort by residue ID
    logger.info(f"Grouped data into {len(domains)} unique domains")
    return domains

def extract_sequences_and_rmsf(domains: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Extract amino acid sequence and RMSF values for each domain."""
    # Standard 1-letter amino acid codes
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        # Include common Histidine variants if they weren't fixed by fix_data_.py
        'HSD': 'H', 'HSE': 'H', 'HSP': 'H'
    }

    processed_data = {}
    processed_count = 0
    skipped_residues = defaultdict(int)
    skipped_domains_non_aa = set()

    for domain_id, domain_df in domains.items():
        if 'resname' not in domain_df.columns or 'rmsf_320' not in domain_df.columns:
             logger.warning(f"Skipping domain {domain_id} due to missing 'resname' or 'rmsf_320' columns.")
             continue

        sequence = ''
        rmsf_values = []
        valid_domain = True
        for _, row in domain_df.iterrows():
            residue = str(row['resname']).upper().strip() # Normalize residue name
            if residue in aa_map:
                sequence += aa_map[residue]
                rmsf_values.append(row['rmsf_320'])
            else:
                # Log unknown residues but continue processing the domain for now
                skipped_residues[residue] += 1
                skipped_domains_non_aa.add(domain_id)
                # Optionally, uncomment below to skip domains with *any* non-standard residue
                # logger.warning(f"Unknown residue '{residue}' found in domain {domain_id}. Skipping this domain.")
                # valid_domain = False
                # break # Stop processing this domain

        if valid_domain and sequence:  # Only add if sequence is not empty and domain is valid
            # Final check: ensure sequence length matches RMSF list length
            if len(sequence) == len(rmsf_values):
                processed_data[domain_id] = {
                    'sequence': sequence,
                    'rmsf': np.array(rmsf_values, dtype=np.float32) # Ensure float32
                }
                processed_count += 1
            else:
                logger.warning(f"Length mismatch for domain {domain_id}: "
                               f"Sequence length={len(sequence)}, RMSF values={len(rmsf_values)}. Skipping domain.")
        elif not sequence:
             logger.warning(f"Domain {domain_id} resulted in an empty sequence. Skipping.")


    if skipped_residues:
        logger.warning(f"Encountered unknown residues (counts): {dict(skipped_residues)}")
        logger.warning(f"These residues occurred in {len(skipped_domains_non_aa)} domains.") # Domains might have been skipped or processed depending on policy above.
    logger.info(f"Successfully extracted sequence and RMSF for {processed_count} domains")
    return processed_data

def extract_topology(domain_id: str) -> str:
    """
    Extract topology identifier from domain_id.
    Assumes PDB ID (first 4 chars) represents topology. Adjust if structure differs.
    """
    if isinstance(domain_id, str) and len(domain_id) >= 4:
        return domain_id[:4].upper() # Use first 4 chars, uppercase
    else:
        logger.warning(f"Could not extract topology from domain_id: {domain_id}. Using fallback ID.")
        return f"unknown_{hash(domain_id)}" # Fallback for unexpected formats

def split_by_topology(data: Dict[str, Dict], train_ratio=0.7, val_ratio=0.15, seed=42) -> Tuple[Dict, Dict, Dict]:
    """Split data by topology to ensure no topology overlap between splits."""
    if not data:
        logger.warning("No data provided to split_by_topology. Returning empty splits.")
        return {}, {}, {}

    random.seed(seed)
    logger.info(f"Splitting {len(data)} domains by topology using seed {seed}")

    # Group domain IDs by topology
    topology_groups = defaultdict(list)
    for domain_id in data.keys():
        topology = extract_topology(domain_id)
        topology_groups[topology].append(domain_id)

    logger.info(f"Found {len(topology_groups)} unique topologies.")

    # Shuffle the list of unique topologies
    topologies = list(topology_groups.keys())
    random.shuffle(topologies)

    # Calculate split indices based on the number of topologies
    n_topologies = len(topologies)
    if n_topologies < 3: # Need at least one topology per split ideally
        logger.warning(f"Very few topologies ({n_topologies}). Split ratios might not be accurate.")

    train_idx = int(n_topologies * train_ratio)
    val_idx = train_idx + int(n_topologies * val_ratio)
    # Ensure validation set has at least one topology if possible
    if train_idx == val_idx and n_topologies > train_idx:
        val_idx += 1
    # Ensure test set has at least one topology if possible
    if val_idx == n_topologies and n_topologies > 0:
        if train_idx < val_idx -1: # Steal one from val if val has > 1
             val_idx -= 1
        elif train_idx > 0 : # Steal one from train if train has > 0
             train_idx -= 1
             val_idx -=1
        # If only 1 or 2 topologies, splits will be uneven.


    # Split topologies into sets
    train_topologies = set(topologies[:train_idx])
    val_topologies = set(topologies[train_idx:val_idx])
    test_topologies = set(topologies[val_idx:])

    logger.info(f"Split topologies: Train={len(train_topologies)}, Val={len(val_topologies)}, Test={len(test_topologies)}")

    # Create data splits based on topology sets
    train_data, val_data, test_data = {}, {}, {}
    assigned_domains = 0
    for domain_id, domain_data in data.items():
        topology = extract_topology(domain_id)
        if topology in train_topologies:
            train_data[domain_id] = domain_data
            assigned_domains +=1
        elif topology in val_topologies:
            val_data[domain_id] = domain_data
            assigned_domains += 1
        elif topology in test_topologies:
            test_data[domain_id] = domain_data
            assigned_domains += 1
        else:
             # Should not happen if all topologies are assigned
             logger.warning(f"Domain {domain_id} with topology {topology} was not assigned to any split!")

    logger.info(f"Split domains: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    if assigned_domains != len(data):
         logger.warning(f"Mismatch in assigned domains ({assigned_domains}) vs total domains ({len(data)}).")

    return train_data, val_data, test_data

def save_split_data(data: Dict, output_dir: str, split_name: str):
    """Save split data (domain list, FASTA, RMSF numpy) to disk."""
    if not data:
        logger.warning(f"No data to save for split '{split_name}'. Skipping save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    domain_ids = list(data.keys())

    # Save domain list
    domain_list_path = os.path.join(output_dir, f"{split_name}_domains.txt")
    try:
        with open(domain_list_path, 'w') as f:
            for domain_id in domain_ids:
                f.write(f"{domain_id}\n")
        logger.info(f"Saved {len(domain_ids)} domain IDs to {domain_list_path}")
    except IOError as e:
        logger.error(f"Error writing domain list {domain_list_path}: {e}")

    # Save sequences in FASTA format
    fasta_path = os.path.join(output_dir, f"{split_name}_sequences.fasta")
    try:
        with open(fasta_path, 'w') as f:
            for domain_id in domain_ids:
                if 'sequence' in data[domain_id]:
                    f.write(f">{domain_id}\n{data[domain_id]['sequence']}\n")
                else:
                    logger.warning(f"Missing 'sequence' key for domain {domain_id} when saving FASTA for split {split_name}.")
        logger.info(f"Saved sequences to {fasta_path}")
    except IOError as e:
        logger.error(f"Error writing FASTA file {fasta_path}: {e}")

    # Save RMSF values as a NumPy dictionary
    rmsf_path = os.path.join(output_dir, f"{split_name}_rmsf.npy")
    rmsf_data = {}
    for domain_id in domain_ids:
        if 'rmsf' in data[domain_id]:
             # Ensure it's a numpy array before saving
             rmsf_array = data[domain_id]['rmsf']
             if not isinstance(rmsf_array, np.ndarray):
                  logger.warning(f"RMSF data for {domain_id} is not a numpy array (type: {type(rmsf_array)}). Attempting conversion.")
                  try:
                      rmsf_array = np.array(rmsf_array, dtype=np.float32)
                  except Exception as conv_err:
                       logger.error(f"Could not convert RMSF data for {domain_id} to numpy array: {conv_err}. Skipping.")
                       continue
             rmsf_data[domain_id] = rmsf_array

        else:
             logger.warning(f"Missing 'rmsf' key for domain {domain_id} when saving RMSF data for split {split_name}.")

    if rmsf_data: # Only save if there is data
        try:
            np.save(rmsf_path, rmsf_data, allow_pickle=True) # Allow pickle needed for dict saving
            logger.info(f"Saved RMSF data for {len(rmsf_data)} domains to {rmsf_path}")
        except IOError as e:
            logger.error(f"Error saving RMSF numpy file {rmsf_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving RMSF numpy file {rmsf_path}: {e}")
    else:
        logger.warning(f"No valid RMSF data found to save for split {split_name}.")


def process_data(csv_path: str, output_dir: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Main function to process RMSF data and create splits."""
    logger.info(f"Starting data processing pipeline...")
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Split Ratios: Train={train_ratio}, Val={val_ratio}, Test={1 - train_ratio - val_ratio:.2f}")
    logger.info(f"Random Seed: {seed}")

    try:
        # 1. Load Data
        df = load_rmsf_data(csv_path)

        # 2. Group by Domain
        domains = group_by_domain(df)

        # 3. Extract Sequences and RMSF
        data = extract_sequences_and_rmsf(domains)
        if not data:
             logger.error("No valid domain data extracted. Aborting processing.")
             return None, None, None # Indicate failure

        # 4. Split by Topology
        train_data, val_data, test_data = split_by_topology(data, train_ratio, val_ratio, seed)

        # 5. Save Splits
        save_split_data(train_data, output_dir, 'train')
        save_split_data(val_data, output_dir, 'val')
        save_split_data(test_data, output_dir, 'test')

        logger.info("Data processing completed successfully.")
        return train_data, val_data, test_data

    except FileNotFoundError as e:
         logger.error(f"Processing failed: {e}")
         return None, None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)
        return None, None, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process protein RMSF data from CSV, extract sequences, and split by topology.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the input raw RMSF CSV file.')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save the processed data splits (FASTA, NPY, TXT files).')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Fraction of topologies for the training set (default: 0.7).')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of topologies for the validation set (default: 0.15). Test set gets the remainder.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling topologies before splitting (default: 42).')
    args = parser.parse_args()

    process_data(args.csv, args.output, args.train_ratio, args.val_ratio, args.seed)
