import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import json
import logging
import argparse
from collections import defaultdict
import time
from typing import Optional, Dict, List, Tuple, Any
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import gc # Garbage collector

# --- Configuration ---
DEFAULT_PARQUET_FILE = "/home/s_felix/mdcath-processor/noise_ceiling_analysis/all_replica_rmsf.parquet"
DEFAULT_OUTPUT_DIR = "/home/s_felix/mdcath-processor/noise_ceiling_analysis"
DEFAULT_CALC_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() and os.cpu_count() > 2 else 1)
MIN_VALID_RESIDUES = 2 # Minimum overlapping residues needed for correlation

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s [%(processName)s:%(funcName)s] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)],
                    force=True)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Pearson correlation safely, returning NaN on errors or invalid input."""
    # Re-using the robust version from previous script
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < MIN_VALID_RESIDUES or y.size < MIN_VALID_RESIDUES or x.size != y.size:
        return np.nan

    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if x_clean.size < MIN_VALID_RESIDUES: return np.nan
    if np.std(x_clean) < 1e-9 or np.std(y_clean) < 1e-9:
         if np.all(x_clean == x_clean[0]) and np.all(y_clean == y_clean[0]) and x_clean[0] == y_clean[0]:
             return 1.0
         return np.nan

    try:
        corr, _ = pearsonr(x_clean, y_clean)
        return np.nan if np.isnan(corr) else float(corr)
    except (ValueError, FloatingPointError) as e:
        logger.warning(f"Pearsonr calculation error: {e}")
        return np.nan


# --- Worker Function for Processing Groups ---
def process_instance_group(group: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Processes a DataFrame group (single domain_id, temp).
    Pivots data, calculates correlations, and returns results.
    """
    if group.empty: return None

    # Extract identifiers from the first row (they are constant within the group)
    domain_id = group['domain_id'].iloc[0]
    temp = group['temp'].iloc[0]
    instance_key = (domain_id, temp) # For logging

    # Check minimum number of replicas
    replicas_present = group['replica'].unique()
    num_replicas = len(replicas_present)
    if num_replicas < 2:
        # logger.debug(f"Skipping {instance_key}: Only {num_replicas} replica(s) found.")
        return None

    try:
        # Pivot to get replicas as columns, indexed by residue
        # This is the crucial step for alignment
        pivot_df = group.pivot(index='resid', columns='replica', values='rmsf')

        # Drop residues (rows) that have missing values in ANY replica column
        # This ensures we only correlate over residues present in ALL replicas for this group
        pivot_df.dropna(axis=0, how='any', inplace=True)

        num_aligned_residues = len(pivot_df)

        # Check if enough aligned residues remain
        if num_aligned_residues < MIN_VALID_RESIDUES:
             # logger.debug(f"Skipping {instance_key}: Only {num_aligned_residues} fully aligned residues found (min required: {MIN_VALID_RESIDUES}).")
             return None

        # --- Correlation Calculation ---
        pairwise_correlations = []
        # Columns are now replica IDs (as integers from pivot)
        replica_cols = pivot_df.columns.tolist()

        # Method 1: Iterating through combinations (Simple, clear)
        for rep1_idx, rep2_idx in combinations(range(len(replica_cols)), 2):
             rep1_id = replica_cols[rep1_idx]
             rep2_id = replica_cols[rep2_idx]
             corr = safe_pearsonr(pivot_df[rep1_id].values, pivot_df[rep2_id].values)
             if not np.isnan(corr):
                 pairwise_correlations.append(corr)

        # Method 2: Using numpy.corrcoef (Potentially faster for many replicas)
        # if len(replica_cols) >= 2:
        #      # Ensure data is float64 for np.corrcoef
        #      corr_matrix = np.corrcoef(pivot_df.astype(np.float64).values, rowvar=False) # Columns are variables (replicas)
        #      # Extract upper triangle (excluding diagonal)
        #      indices = np.triu_indices_from(corr_matrix, k=1)
        #      pairwise_correlations_np = corr_matrix[indices]
        #      # Filter out potential NaNs from np.corrcoef if columns were constant
        #      pairwise_correlations = [c for c in pairwise_correlations_np if not np.isnan(c)]
        # else: pairwise_correlations = [] # Should not happen due to earlier check


        if not pairwise_correlations:
            # logger.debug(f"Skipping {instance_key}: No valid pairwise correlations calculated.")
            return None

        # Calculate average correlation
        avg_corr = float(np.nanmean(pairwise_correlations))

        result = {
            'domain_id': domain_id,
            'temp': int(temp),
            'avg_pairwise_correlation': avg_corr,
            'num_replicas_in_group': num_replicas,
            'num_aligned_residues': num_aligned_residues,
            # 'pairwise_correlations': pairwise_correlations # Optional: exclude for smaller output
        }
        return result

    except MemoryError:
         logger.error(f"MemoryError processing group for {instance_key}. Group size: {len(group)}")
         # Try cleaning up memory (might help in some scenarios)
         del group, pivot_df
         gc.collect()
         return None
    except Exception as e:
        logger.error(f"Error processing group for {instance_key}: {e}", exc_info=False)
        return None


# --- Main Calculation Function ---
def calculate_noise_ceiling_parallel(parquet_file: str, output_dir: str, num_workers: int):
    """Loads Parquet data, processes domain/temp groups in parallel to calculate noise ceiling."""
    parquet_path = Path(parquet_file).resolve()
    output_dir_path = Path(output_dir).resolve()

    logger.info("Starting noise ceiling calculation from Parquet.")
    logger.info(f"Input Parquet file: {parquet_path}")
    logger.info(f"Output directory: {output_dir_path}")
    logger.info(f"Using {num_workers} workers for group processing.")

    if not parquet_path.is_file():
        logger.error(f"Input Parquet file not found: {parquet_path}")
        logger.error("Please run the preprocessing script first.")
        return

    output_dir_path.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # 1. Load the Parquet file
    logger.info("Loading Parquet file into DataFrame...")
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded DataFrame shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        logger.info("Columns and Dtypes:\n{}".format(df.dtypes))
    except Exception as e:
        logger.error(f"Failed to load Parquet file {parquet_path}: {e}", exc_info=True)
        return

    # Get unique temperatures present in the data
    sorted_temperatures = sorted(df['temp'].unique().tolist())
    logger.info(f"Temperatures found in data: {sorted_temperatures}")


    # 2. Group data and prepare tasks for multiprocessing
    logger.info("Grouping data by domain and temperature...")
    grouped = df.groupby(['domain_id', 'temp'], sort=False) # sort=False might be slightly faster
    num_groups = len(grouped)
    logger.info(f"Created {num_groups:,} groups (potential instances) to process.")

    # Create an iterator of groups for multiprocessing's map functions
    # Important: Passing large groups can consume memory. imap might be better.
    group_iterator = (group for _, group in grouped)

    # Free up the original large DataFrame memory *before* starting the pool
    del df
    gc.collect()
    logger.info("Original DataFrame released from memory.")

    # 3. Process groups in parallel
    logger.info(f"Processing {num_groups:,} groups using {num_workers} workers...")
    all_results_list = []
    processed_groups = 0
    errors_in_processing = 0

    # Use context manager for the pool
    with mp.Pool(processes=num_workers) as pool:
        try:
            # Use imap_unordered: efficient, handles memory better for large iterables
            results_iterator = pool.imap_unordered(process_instance_group, group_iterator)

            for result_data in tqdm(results_iterator, total=num_groups, desc="Processing Instances", unit="instance"):
                processed_groups += 1
                if result_data is not None:
                    all_results_list.append(result_data)
                else:
                    # Optional: Increment a counter for skipped/failed groups
                    errors_in_processing += 1

        except Exception as pool_e:
             logger.error(f"Fatal error during parallel group processing: {pool_e}", exc_info=True)
             # Decide how to handle partial results if needed
             # return

    logger.info(f"Finished parallel processing. Processed {processed_groups}/{num_groups} groups.")
    if errors_in_processing > 0:
         logger.warning(f"{errors_in_processing} groups resulted in errors or were skipped during processing.")


    # 4. Aggregate Results
    logger.info("Aggregating final results...")
    if not all_results_list:
        logger.error("No results were successfully generated by the workers. Check logs.")
        return

    # Convert list of dicts to DataFrame for easier aggregation
    results_df = pd.DataFrame(all_results_list)
    del all_results_list # Free memory
    logger.info(f"Aggregated results into DataFrame with shape: {results_df.shape}")

    aggregated_summary: Dict[str, Any] = {'per_temperature': {}}
    valid_correlations = results_df['avg_pairwise_correlation'].dropna()
    instances_successfully_analyzed = len(valid_correlations)
    domains_with_results = results_df['domain_id'].nunique() # Count unique domains in results


    # Overall stats
    overall_stats = {}
    if instances_successfully_analyzed > 0:
        overall_stats['num_instances_analyzed'] = instances_successfully_analyzed
        overall_stats['num_domains_analyzed'] = int(domains_with_results) # Ensure JSON serializable type
        overall_stats['mean_correlation'] = float(valid_correlations.mean())
        overall_stats['median_correlation'] = float(valid_correlations.median())
        overall_stats['std_dev_correlation'] = float(valid_correlations.std())
        overall_stats['min_correlation'] = float(valid_correlations.min())
        overall_stats['max_correlation'] = float(valid_correlations.max())
    aggregated_summary['overall'] = overall_stats

    # Per-temperature stats
    for temp in sorted_temperatures:
        temp_results = results_df[results_df['temp'] == temp]['avg_pairwise_correlation'].dropna()
        temp_stats = {}
        temp_stats['num_instances_analyzed'] = len(temp_results)
        if len(temp_results) > 0:
            temp_stats['mean_correlation'] = float(temp_results.mean())
            temp_stats['median_correlation'] = float(temp_results.median())
            temp_stats['std_dev_correlation'] = float(temp_results.std())
            temp_stats['min_correlation'] = float(temp_results.min())
            temp_stats['max_correlation'] = float(temp_results.max())
        aggregated_summary['per_temperature'][str(temp)] = temp_stats # Use string key for JSON

    # Optional: Save detailed per-instance results if needed (can be large)
    # detailed_results_path = output_dir_path / f"detailed_consistency_results_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    # try:
    #     results_df.to_parquet(detailed_results_path, index=False)
    #     logger.info(f"Detailed results saved to: {detailed_results_path}")
    # except Exception as e:
    #     logger.error(f"Failed to save detailed results parquet: {e}")


    # 5. Save Summary Results
    aggregated_summary['analysis_info'] = {
        'input_parquet_file': str(parquet_path),
        'output_directory': str(output_dir_path),
        'temperatures_found': sorted_temperatures,
        'total_instances_analyzed': instances_successfully_analyzed,
        'total_groups_processed': processed_groups,
        'num_workers': num_workers,
        'run_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    results_filename = f"noise_ceiling_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    results_path = output_dir_path / results_filename

    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, Path): return str(obj)
                return super(NpEncoder, self).default(obj)

        with open(results_path, 'w') as f:
            json.dump(aggregated_summary, f, indent=4, cls=NpEncoder)
        logger.info(f"Summary results saved to: {results_path}")
    except Exception as e:
        logger.error(f"Failed to save summary JSON: {e}", exc_info=True)


    # 6. Print Summary
    logger.info("\n--- Noise Ceiling Summary (from Parquet) ---")
    if overall_stats:
        logger.info(f"Overall Mean PCC (Noise Ceiling): {overall_stats['mean_correlation']:.4f}")
        logger.info(f"Overall Median PCC:              {overall_stats['median_correlation']:.4f}")
        logger.info(f"Overall Std Dev PCC:             {overall_stats['std_dev_correlation']:.4f}")
        logger.info(f"Overall Min/Max PCC:             {overall_stats['min_correlation']:.4f} / {overall_stats['max_correlation']:.4f}")
        logger.info(f"Based on {overall_stats.get('num_instances_analyzed', 0):,} successfully analyzed (Domain, Temp) instances")
        logger.info(f"Representing {overall_stats.get('num_domains_analyzed', 0):,} unique domains.")
    else:
        logger.info("No valid overall results calculated.")

    logger.info("\nPer Temperature Mean PCC:")
    for temp in sorted_temperatures:
        temp_key = str(temp)
        temp_stats_dict = aggregated_summary['per_temperature'].get(temp_key, {})
        if 'mean_correlation' in temp_stats_dict:
            logger.info(f"  Temp {temp:>3d} K: Mean={temp_stats_dict['mean_correlation']:.4f}, "
                        f"Median={temp_stats_dict['median_correlation']:.4f}, "
                        f"StdDev={temp_stats_dict['std_dev_correlation']:.4f}, "
                        f"N_Inst={temp_stats_dict['num_instances_analyzed']:,}")
        else:
            logger.info(f"  Temp {temp:>3d} K: No data successfully analyzed.")
    logger.info("----------------------------------------------")
    end_time = time.time()
    logger.info(f"Total calculation duration: {end_time - start_time:.2f} seconds.")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate RMSF noise ceiling from a preprocessed Parquet file using parallel processing.')
    parser.add_argument('--parquet_file', type=str, default=DEFAULT_PARQUET_FILE,
                        help=f'Path to the input Parquet file containing combined RMSF data (default: {DEFAULT_PARQUET_FILE}).')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save the results JSON file (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--workers', type=int, default=DEFAULT_CALC_WORKERS,
                        help=f'Number of worker processes for calculations (default: {DEFAULT_CALC_WORKERS}).')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    calculate_noise_ceiling_parallel(args.parquet_file, args.output_dir, args.workers)