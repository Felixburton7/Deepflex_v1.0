#!/usr/bin/env python3
"""
Split Aggregated Dataset into Train and Holdout Sets

This script reads a list of holdout domain IDs and splits the aggregated dataset
into separate train and holdout CSV files.
"""

import os
import pandas as pd
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
HOLDOUT_DOMAINS_PATH = "/home/s_felix/mdcath_sampling/mdcath_holdout_domains.txt"
AGGREGATED_DATASET_PATH = "/home/s_felix/packages/ensembleflex/data/aggregated_dataset.csv"
OUTPUT_DIR = "/home/s_felix/packages/ensembleflex/data"

# Output file paths
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "aggregated_train_dataset.csv")
HOLDOUT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "aggregated_holdout_dataset.csv")
MISSING_DOMAINS_PATH = os.path.join(OUTPUT_DIR, "missing_holdout_domains.txt")

def read_holdout_domains(filepath):
    """Read the holdout domains from a text file."""
    try:
        with open(filepath, 'r') as f:
            # Read and split the content, then strip whitespace from each domain ID
            domains = [domain.strip() for domain in f.read().split()]
        logger.info(f"Successfully read {len(domains):,} holdout domains")
        
        # Print the first few domains for verification
        if domains:
            sample = domains[:5]
            logger.info(f"Sample domains: {', '.join(sample)}")
            
        return domains
    except Exception as e:
        logger.error(f"Error reading holdout domains file: {e}")
        raise

def split_dataset(dataset_path, holdout_domains, train_output, holdout_output, missing_output):
    """Split the aggregated dataset into train and holdout sets."""
    try:
        # Read the aggregated dataset
        logger.info(f"Reading aggregated dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Check if 'domain_id' column exists
        if 'domain_id' not in df.columns:
            logger.error("Error: 'domain_id' column not found in the dataset")
            raise ValueError("'domain_id' column not found in the dataset")
        
        logger.info("Processing dataset...")
        # Create a progress bar for the operations
        with tqdm(total=4, desc="Splitting Dataset") as pbar:
            # Create holdout and train masks
            holdout_mask = df['domain_id'].isin(holdout_domains)
            train_mask = ~holdout_mask
            pbar.update(1)
            
            # Split the dataframe
            holdout_df = df[holdout_mask]
            train_df = df[train_mask]
            pbar.update(1)
            
            # Validate split
            unique_holdout_domains = holdout_df['domain_id'].unique()
            unique_train_domains = train_df['domain_id'].unique()
            
            logger.info(f"Split dataset: {len(df):,} total rows")
            logger.info(f"Holdout set: {len(holdout_df):,} rows, {len(unique_holdout_domains):,} unique domains")
            logger.info(f"Training set: {len(train_df):,} rows, {len(unique_train_domains):,} unique domains")
            
            # Check for any missing holdout domains
            missing_domains = set(holdout_domains) - set(unique_holdout_domains)
            if missing_domains:
                logger.warning(f"Warning: {len(missing_domains):,} holdout domains were not found in the dataset")
                missing_domains_list = list(missing_domains)
                logger.warning(f"First 10 missing domains: {', '.join(missing_domains_list[:10])}...")
                
                # Write missing domains to a text file
                with open(missing_output, 'w') as f:
                    for domain in missing_domains_list:
                        f.write(f"{domain}\n")
                logger.info(f"Saved {len(missing_domains):,} missing domains to {missing_output}")
            pbar.update(1)
            
            # Save the split datasets
            logger.info("Saving datasets...")
            holdout_df.to_csv(holdout_output, index=False)
            train_df.to_csv(train_output, index=False)
            pbar.update(1)
            
            logger.info(f"Successfully saved holdout dataset to {holdout_output}")
            logger.info(f"Successfully saved training dataset to {train_output}")
        
        return len(unique_holdout_domains), len(unique_train_domains)
    
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise

def main():
    """Main function to execute the script."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Display script information
        logger.info("=" * 70)
        logger.info("DATASET SPLITTER - Separating Train and Holdout Domains")
        logger.info("=" * 70)
        logger.info(f"Holdout domains file: {HOLDOUT_DOMAINS_PATH}")
        logger.info(f"Aggregated dataset: {AGGREGATED_DATASET_PATH}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("-" * 70)
        
        # Read holdout domains
        holdout_domains = read_holdout_domains(HOLDOUT_DOMAINS_PATH)
        
        # Split the dataset
        num_holdout, num_train = split_dataset(
            AGGREGATED_DATASET_PATH, 
            holdout_domains,
            TRAIN_OUTPUT_PATH,
            HOLDOUT_OUTPUT_PATH,
            MISSING_DOMAINS_PATH
        )
        
        logger.info("-" * 70)
        logger.info("SUMMARY:")
        logger.info(f"Dataset splitting completed successfully")
        logger.info(f"Training set: {num_train:,} domains")
        logger.info(f"Holdout set: {num_holdout:,} domains")
        logger.info(f"Missing domains list: {MISSING_DOMAINS_PATH}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())