import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
from typing import List, Optional, Tuple

# --- Configuration ---
DEFAULT_INPUT_BASE_DIR = "/home/s_felix/mdcath-processor/outputs/RMSF/replicas"
DEFAULT_OUTPUT_FILE = "/home/s_felix/mdcath-processor/noise_ceiling_analysis/all_replica_rmsf.parquet"
# Adjust based on memory; lower if memory issues occur during processing many files in parallel
DEFAULT_PREPROCESS_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() and os.cpu_count() > 4 else 2)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s [%(processName)s:%(funcName)s] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)],
                    force=True)
logger = logging.getLogger(__name__)

# --- Worker Function for Preprocessing ---
def process_single_rmsf_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Reads a single RMSF CSV, adds identifiers, cleans, and returns a DataFrame."""
    try:
        # Extract info from filename, e.g., rmsf_replica0_temperature320.csv
        parts = file_path.stem.split('_')
        replica_id = int(parts[1].replace('replica', ''))
        temp_str = parts[2].replace('temperature', '')
        if not temp_str.isdigit(): return None # Skip if temp not parsed correctly
        temp = int(temp_str)
        rmsf_col = f"rmsf_{temp_str}" # Column name used in the specific file

        # Define expected columns and types
        expected_cols = ['domain_id', 'resid', rmsf_col]
        # Read 'resid' as string initially to handle potential non-numeric values gracefully
        dtypes = {'domain_id': str, 'resid': str, rmsf_col: np.float32}

        df = pd.read_csv(file_path, usecols=expected_cols, dtype=dtypes, low_memory=False)

        if df.empty: return None

        # Add identifiers
        df['replica'] = replica_id
        df['temp'] = temp

        # --- Residue ID Cleaning ---
        # Rename specific RMSF column to a generic name
        df.rename(columns={rmsf_col: 'rmsf'}, inplace=True)
        # Attempt conversion to numeric, coercing errors to NaN
        df['resid_num'] = pd.to_numeric(df['resid'], errors='coerce')
        # Keep only rows where conversion was successful and value is not NaN
        df = df.dropna(subset=['resid_num', 'rmsf']) # Ensure RMSF is also not NaN
        if df.empty: return None
        df['resid'] = df['resid_num'].astype(int) # Convert valid ones to int
        # Keep only essential columns
        df = df[['domain_id', 'temp', 'replica', 'resid', 'rmsf']]
        # --- End Residue ID Cleaning ---

        # Optional: Drop duplicates within the file itself? Usually not needed if input is clean.
        # df.drop_duplicates(subset=['domain_id', 'temp', 'replica', 'resid'], keep='first', inplace=True)

        return df

    except FileNotFoundError:
        logger.warning(f"File not found during processing: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty CSV file: {file_path}")
        return None
    except ValueError as ve:
        logger.error(f"ValueError processing {file_path} (check structure/types): {ve}")
        return None
    except KeyError as ke:
        logger.error(f"Missing expected column {ke} in {file_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=False)
        return None

# --- Main Preprocessing Function ---
def create_parquet_from_replicas(input_base_dir: str, output_parquet_file: str, num_workers: int):
    """Scans replica directories, processes CSVs in parallel, concatenates, and saves to Parquet."""
    input_dir = Path(input_base_dir).resolve()
    output_file = Path(output_parquet_file).resolve()

    logger.info(f"Starting RMSF preprocessing.")
    logger.info(f"Input replica directory: {input_dir}")
    logger.info(f"Output Parquet file: {output_file}")
    logger.info(f"Using {num_workers} workers for reading CSVs.")

    output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    start_time = time.time()

    # 1. Find all relevant CSV files
    all_csv_files = list(input_dir.rglob('rmsf_replica*_temperature*.csv'))
    if not all_csv_files:
        logger.error(f"No RMSF CSV files found in subdirectories of {input_dir}")
        return
    logger.info(f"Found {len(all_csv_files)} potential RMSF CSV files to process.")

    # 2. Process files in parallel
    logger.info("Reading and processing CSV files in parallel...")
    all_dfs = []
    skipped_files = 0
    # Use context manager for the pool
    with mp.Pool(processes=num_workers) as pool:
        try:
            # Use imap_unordered for potential memory efficiency if DataFrames are large
            # and order doesn't matter before concatenation
            results_iterator = pool.imap_unordered(process_single_rmsf_file, all_csv_files)
            for df_or_none in tqdm(results_iterator, total=len(all_csv_files), desc="Processing CSVs", unit="file"):
                 if df_or_none is not None and not df_or_none.empty:
                     all_dfs.append(df_or_none)
                 else:
                      skipped_files += 1

        except Exception as pool_e:
            logger.error(f"Error during parallel CSV processing: {pool_e}", exc_info=True)
            return # Stop if the pool fails

    if not all_dfs:
        logger.error("No valid dataframes were created from the CSV files. Check logs.")
        return

    if skipped_files > 0:
        logger.warning(f"Skipped or failed to process {skipped_files} files.")

    # 3. Concatenate into a single DataFrame
    logger.info(f"Concatenating {len(all_dfs)} processed dataframes...")
    try:
        # ignore_index=True is crucial for a clean index after concatenation
        combined_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs # Free up memory
    except Exception as concat_e:
        logger.error(f"Error during dataframe concatenation: {concat_e}", exc_info=True)
        return

    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    logger.info(f"Memory usage: {combined_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # 4. Final Checks and Sorting (Optional but recommended)
    logger.info("Performing final checks and sorting...")
    # Ensure correct types (belt and suspenders)
    combined_df['temp'] = combined_df['temp'].astype(np.int16)
    combined_df['replica'] = combined_df['replica'].astype(np.int8)
    combined_df['resid'] = combined_df['resid'].astype(np.int32)
    combined_df['rmsf'] = combined_df['rmsf'].astype(np.float32)

    # Check for final duplicates across files (should be rare if file processing was correct)
    initial_rows = len(combined_df)
    combined_df.drop_duplicates(subset=['domain_id', 'temp', 'replica', 'resid'], keep='first', inplace=True)
    if len(combined_df) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(combined_df)} duplicate rows found across files.")

    # Sort for potentially better query performance later (optional)
    combined_df.sort_values(by=['domain_id', 'temp', 'replica', 'resid'], inplace=True)

    logger.info(f"Final DataFrame shape after checks: {combined_df.shape}")
    logger.info(f"Final Memory usage: {combined_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    logger.info("Columns and Dtypes:\n{}".format(combined_df.dtypes))


    # 5. Save to Parquet
    logger.info(f"Saving combined data to Parquet file: {output_file} ...")
    try:
        # Use 'pyarrow' engine (usually default and fast), compression helps reduce file size
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        logger.info(f"Successfully saved Parquet file.")
        # Verify file size
        file_size_mb = output_file.stat().st_size / (1024**2)
        logger.info(f"Parquet file size: {file_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save Parquet file: {e}", exc_info=True)
        return

    end_time = time.time()
    logger.info(f"Preprocessing completed in {end_time - start_time:.2f} seconds.")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess RMSF replica CSV files into a single Parquet file.')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_BASE_DIR,
                        help=f'Base directory containing replica subdirectories (default: {DEFAULT_INPUT_BASE_DIR}).')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Path to save the output Parquet file (default: {DEFAULT_OUTPUT_FILE}).')
    parser.add_argument('--workers', type=int, default=DEFAULT_PREPROCESS_WORKERS,
                        help=f'Number of worker processes for reading CSVs (default: {DEFAULT_PREPROCESS_WORKERS}).')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    create_parquet_from_replicas(args.input_dir, args.output_file, args.workers)