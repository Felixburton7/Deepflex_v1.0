# /home/s_felix/ensembleflex/ensembleflex/data/loader.py

"""
Data loading utilities for the EnsembleFlex ML pipeline.

This module provides functions for loading protein data from various formats.
Temperature-specific loading logic has been removed for the unified model approach.
"""

import os
import logging
# import re # No longer needed
from typing import List, Dict, Any, Optional # Union, Tuple no longer needed here
from functools import lru_cache

import pandas as pd
import numpy as np
# import glob # No longer needed

logger = logging.getLogger(__name__)

# Removed: list_data_files
# Removed: get_temperature_files

def detect_file_format(file_path: str) -> str:
    """
    Detect file format based on extension and content.

    Args:
        file_path: Path to data file

    Returns:
        Format string ('csv', 'tsv', 'pickle', etc.)
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return 'csv'
    elif ext == '.tsv':
        return 'tsv'
    elif ext in ['.pkl', '.pickle']:
        return 'pickle'
    elif ext == '.json':
        return 'json'
    elif ext == '.parquet':
        return 'parquet'
    elif ext == '.h5':
        return 'hdf5'
    else:
        # Try to detect CSV/TSV by reading first line
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    return 'tsv'
                elif ',' in first_line:
                    return 'csv'
        except:
            pass

        # Default to CSV if can't determine
        logger.warning(f"Could not determine format for {file_path}, defaulting to CSV")
        return 'csv'

@lru_cache(maxsize=16)
def load_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a file with format auto-detection. Uses LRU cache.

    Args:
        file_path: Path to data file
        **kwargs: Additional arguments to pass to pandas

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or loading fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Detect format
    file_format = detect_file_format(file_path)
    logger.debug(f"Detected format '{file_format}' for file: {file_path}")

    try:
        # Load based on format
        if file_format == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_format == 'tsv':
            df = pd.read_csv(file_path, sep='\t', **kwargs)
        elif file_format == 'pickle':
            df = pd.read_pickle(file_path, **kwargs)
        elif file_format == 'json':
            df = pd.read_json(file_path, **kwargs)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif file_format == 'hdf5':
            df = pd.read_hdf(file_path, **kwargs)
        else:
            # Should not happen if detect_file_format is robust
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Successfully loaded file: {file_path} (shape: {df.shape})")
        return df

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to load file {file_path}: {e}")

def merge_data_files(file_paths: List[str], **kwargs) -> pd.DataFrame:
    """
    Merge multiple data files into a single DataFrame.
    Note: Primarily used by the aggregation script, less relevant for main pipeline now.

    Args:
        file_paths: List of paths to data files
        **kwargs: Additional arguments to pass to pandas

    Returns:
        Merged DataFrame
    """
    if not file_paths:
        raise ValueError("No files provided for merging")

    # Load and concatenate files
    dfs = []
    logger.info(f"Attempting to merge {len(file_paths)} files.")
    for file_path in file_paths:
        try:
            # Use the cached loader
            df = load_file(file_path, **kwargs)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping file {file_path} during merge due to error: {e}")

    if not dfs:
        raise ValueError("No data files could be loaded for merging")

    logger.info(f"Successfully loaded {len(dfs)} files for merging. Concatenating...")
    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merging complete. Final shape: {merged_df.shape}")
    return merged_df

def validate_data_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if all required columns are present, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.warning(f"Missing required columns in DataFrame: {missing_columns}")
        return False
    else:
        logger.debug("All required columns found in DataFrame.")
        return True

# Removed: load_temperature_data
# Removed: load_all_temperature_data

def summarize_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for a dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary of summary statistics
    """
    if df is None or df.empty:
        logger.warning("Cannot summarize empty or None DataFrame.")
        return {}

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "memory_usage": None,
        "domains": None,
        "residues_per_domain_rows": None, # Renamed for clarity
        "temperatures": None, # Added for aggregated data
        "column_types": {},
        "missing_values": {},
    }

    # Memory usage
    try:
        memory_bytes = df.memory_usage(deep=True).sum()

        if memory_bytes < 1024:
            summary["memory_usage"] = f"{memory_bytes} bytes"
        elif memory_bytes < 1024**2:
            summary["memory_usage"] = f"{memory_bytes / 1024:.2f} KB"
        elif memory_bytes < 1024**3:
            summary["memory_usage"] = f"{memory_bytes / (1024**2):.2f} MB"
        else:
            summary["memory_usage"] = f"{memory_bytes / (1024**3):.2f} GB"
    except Exception as e:
        logger.warning(f"Could not estimate memory usage: {e}")


    # Domain statistics if domain_id is present
    if "domain_id" in df.columns:
        try:
            domains = df["domain_id"].unique()
            summary["domains"] = {
                "count": len(domains),
                "examples": list(domains[:5])
            }
            # Note: In aggregated data, this counts rows per domain (residues * temperatures)
            # A more accurate 'residues per domain' requires grouping by ('domain_id', 'temperature') first
            residue_counts = df.groupby("domain_id").size()
            summary["residues_per_domain_rows"] = { # Renamed key
                "min": residue_counts.min(),
                "max": residue_counts.max(),
                "mean": residue_counts.mean(),
                "median": residue_counts.median()
            }
        except Exception as e:
             logger.warning(f"Could not calculate domain statistics: {e}")


    # Temperature statistics
    if "temperature" in df.columns:
         try:
             temps = df["temperature"].unique()
             summary["temperatures"] = {
                 "count": len(temps),
                 "values": sorted([t for t in temps if pd.notna(t)]),
                 "has_nan": df["temperature"].isnull().any()
             }
         except Exception as e:
             logger.warning(f"Could not calculate temperature statistics: {e}")

    # Column types and missing values
    for col in df.columns:
        try:
            summary["column_types"][col] = str(df[col].dtype)
            missing = df[col].isna().sum()
            if missing > 0:
                summary["missing_values"][col] = {
                    "count": missing,
                    "percentage": (missing / len(df)) * 100
                }
        except Exception as e:
            logger.warning(f"Could not process column '{col}' for summary: {e}")


    # Check for specific columns (no longer temperature-specific RMSF)
    if "rmsf" in df.columns:
        summary["target_column"] = "rmsf"

    omniflex_columns = ["esm_rmsf", "voxel_rmsf"]
    found_omniflex_columns = [col for col in omniflex_columns if col in df.columns]
    if found_omniflex_columns:
        summary["omniflex_columns"] = found_omniflex_columns

    return summary

def log_data_summary(summary: Dict[str, Any]) -> None:
    """
    Log a summary of dataset statistics.

    Args:
        summary: Dictionary of summary statistics generated by summarize_data
    """
    if not summary:
        logger.info("No data summary to log.")
        return

    logger.info("=== Dataset Summary ===")
    logger.info(f"Rows: {summary.get('num_rows', 'N/A')}, Columns: {summary.get('num_columns', 'N/A')}")

    if summary.get("memory_usage"):
        logger.info(f"Memory usage: {summary['memory_usage']}")

    if summary.get("domains"):
        logger.info(f"Domains: {summary['domains']['count']} unique domains")
        # Ensure examples are strings for joining
        logger.info(f"Examples: {', '.join(map(str, summary['domains']['examples']))}")

    if summary.get("residues_per_domain_rows"): # Using the renamed key
        stats = summary["residues_per_domain_rows"]
        logger.info(f"Rows per domain (Residues * Temps): min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}, median={stats['median']:.1f}")

    if summary.get("temperatures"):
        stats = summary["temperatures"]
        logger.info(f"Temperatures: {stats['count']} unique values detected.")
        logger.info(f"Numeric Temps Found: {stats['values']}")
        if stats['has_nan']:
            logger.warning("NaN temperature values detected (potentially from 'average' files).")

    missing_stats = summary.get("missing_values", {})
    if missing_stats:
        logger.info("Columns with missing values:")
        for col, stats in missing_stats.items():
            logger.info(f"  {col}: {stats['count']} missing ({stats['percentage']:.1f}%)")
    else:
        logger.info("No missing values detected.")

    if summary.get("target_column"):
        logger.info(f"Target column: {summary['target_column']}")

    if summary.get("omniflex_columns"):
        logger.info(f"OmniFlex columns found: {', '.join(summary['omniflex_columns'])}")

    logger.info("========================")