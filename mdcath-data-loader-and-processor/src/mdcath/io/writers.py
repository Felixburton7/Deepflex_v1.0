"""
Helper functions for writing processed data to files (CSV, PDB, etc.).
"""
import os
import pandas as pd
import logging

def save_dataframe_csv(df: pd.DataFrame, path: str, **kwargs):
    """
    Saves a Pandas DataFrame to a CSV file. Creates directory if needed.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Full path to the output CSV file.
        **kwargs: Additional keyword arguments passed to df.to_csv().
    """
    if df is None or df.empty:
        logging.warning(f"Attempted to save an empty DataFrame to {path}. Skipping.")
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if 'index' not in kwargs:
            kwargs['index'] = False
        df.to_csv(path, **kwargs)
        logging.debug(f"Successfully saved DataFrame ({len(df)} rows) to {path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {path}: {e}", exc_info=True)
        # Do not re-raise here, allow pipeline to potentially continue

def save_string(text: str, path: str):
    """
    Saves a string to a text file. Creates directory if needed.

    Args:
        text (str): String content to save.
        path (str): Full path to the output text file.
    """
    if not text:
         logging.warning(f"Attempted to save empty string to {path}. Skipping.")
         return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)
        logging.debug(f"Successfully saved string ({len(text)} chars) to {path}")
    except Exception as e:
        logging.error(f"Failed to save string to {path}: {e}", exc_info=True)
        # Do not re-raise here


# Updated path construction based on prompt refinement
def get_rmsf_output_path(output_base: str, type: str, flatten: bool, **kwargs) -> str:
    """
    Constructs the output path for RMSF files based on flattening config.

    Args:
        output_base (str): Base output directory (e.g., './outputs').
        type (str): 'replica' or 'average'.
        flatten (bool): Whether to flatten the directory structure within type.
        **kwargs: Must contain 'temperature', 'replica' for type='replica'.
                  Must contain 'temperature' for type='average' temp-specific file.
                  Can be empty for type='average' overall file.

    Returns:
        str: The constructed file path.
    """
    rmsf_root = os.path.join(output_base, "RMSF")

    if type == 'replica':
        if 'replica' not in kwargs or 'temperature' not in kwargs:
             raise ValueError("Missing 'replica' or 'temperature' for RMSF type 'replica'")
        replica = kwargs['replica']
        temp = kwargs['temperature']
        filename = f"rmsf_replica{replica}_temperature{temp}.csv"
        # Prompt specified flattening *within* replica folder
        # outputs/RMSF/replicas/replica_0/rmsf_replica0_temperature320.csv
        # outputs/RMSF/replicas/replica_0/rmsf_replica0_temperature348.csv
        replica_dir = os.path.join(rmsf_root, "replicas", f"replica_{replica}")
        path = os.path.join(replica_dir, filename)
        return path

    elif type == 'average':
        temp = kwargs.get('temperature')
        avg_dir = os.path.join(rmsf_root, "replica_average")
        if temp: # Temperature-specific average
            filename = f"rmsf_replica_average_temperature{temp}.csv"
             # Prompt specified flattening directly into replica_average/
             # outputs/RMSF/replica_average/rmsf_replica_average_temperature320.csv
            path = os.path.join(avg_dir, filename)
        else: # Overall average
            filename = "rmsf_all_temperatures_all_replicas.csv"
            # outputs/RMSF/replica_average/rmsf_all_temperatures_all_replicas.csv
            path = os.path.join(avg_dir, filename)
        return path
    else:
        raise ValueError(f"Invalid RMSF type specified: {type}")

